'''Training'''
from scipy.io import loadmat
import numpy as np
import argparse
import configparser
import torch
from torch import nn
from skimage.segmentation import slic
from torch_geometric.data import Data, Batch
from sklearn.preprocessing import scale, minmax_scale
import os
from PIL import Image
from utils import get_graph_list, split, get_edge_index,average_superpixel,compute_transformation_matrix
import math
from Model.module import GraphNet3, MLPNet,MLPNet_1,Conv1x1,Backbone,GATGraphNet,KNNGraphNet,KD1,PiexlsConvBlock,full_graphConv,FeatureFusionModel
from Trainer import JointTrainer
from Monitor import GradMonitor
# from visdom import Visdom
from tqdm import tqdm
import random
import time
import torchprofile


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='TRAIN SUBGRAPH')
    parser.add_argument('--name', type=str, default='Salinas',
                        help='DATASET NAME')
    parser.add_argument('--block', type=int, default=650,
                        help='BLOCK SIZE')
    parser.add_argument('--epoch', type=int, default=200,
                        help='ITERATION')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID')
    parser.add_argument('--comp', type=int, default=10,
                        help='COMPACTNESS')
    parser.add_argument('--batchsz', type=int, default=64,
                        help='BATCH SIZE')
    parser.add_argument('--run', type=int, default=10,
                        help='EXPERIMENT AMOUNT')
    parser.add_argument('--spc', type=int, default=5,
                        help='SAMPLE per CLASS')
    parser.add_argument('--hsz', type=int, default=256,
                        help='HIDDEN SIZE')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='LEARNING RATE')
    parser.add_argument('--wd', type=float, default=0.,
                        help='WEIGHT DECAY')
    arg = parser.parse_args()
    config = configparser.ConfigParser()
    config.read('dataInfo.ini')
    # viz = Visdom(port=8097)

    # Set random seed for reproducibility
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # 所有可用的 GPU 设置种子
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    set_seed(216)

    # Data processing
    # Reading hyperspectral image
    data_path = 'data/{0}/{0}.mat'.format(arg.name)
    m = loadmat(data_path)
    data = m[config.get(arg.name, 'data_key')]
    gt_path = 'data/{0}/{0}_gt.mat'.format(arg.name)
    m = loadmat(gt_path)
    gt = m[config.get(arg.name, 'gt_key')]

    # Normalizing data
    h, w, c = data.shape
    data = data.reshape((h * w, c))
    data = data.astype(float)
    if arg.name == 'Xiongan':
        minmax_scale(data, copy=False)
    data_normalization = scale(data).reshape((h, w, c))

    # Superpixel segmentation
    seg_root = 'data/rgb'
    seg_path = os.path.join(seg_root, '{}_seg_{}.npy'.format(arg.name, arg.block))
    if os.path.exists(seg_path):
        seg = np.load(seg_path)
    else:
        rgb_path = os.path.join(seg_root, '{}_rgb.jpg'.format(arg.name))
        img = Image.open(rgb_path)
        img_array = np.array(img)

        # The number of superpixel
        n_superpixel = int(math.ceil((h * w) / arg.block))
        seg = slic(img_array, n_superpixel, arg.comp)
        seg = seg - 1
        print(np.max(seg))

        # Saving
        np.save(seg_path, seg)

    Q = compute_transformation_matrix(seg, np.max(seg) + 1)
    print("Transformation Matrix Q.shape", Q.shape)

    # Load or generate edge index for the superpixel graph
    full_edge_index_path = 'data/{}/{}_edge_index.npy'.format(arg.name, arg.block)
    if os.path.exists(full_edge_index_path):
        edge_index = np.load(full_edge_index_path)
    else:
        edge_index, _ = get_edge_index(seg)
        np.save(full_edge_index_path, edge_index if isinstance(edge_index, np.ndarray) else edge_index.cpu().numpy())

    # Construct a full graph object for torch_geometric
    fullGraph = Data(None,
                     edge_index=torch.from_numpy(edge_index) if isinstance(edge_index, np.ndarray) else edge_index,
                     seg=torch.from_numpy(seg) if isinstance(seg, np.ndarray) else seg)

    print("edge_index：",edge_index.shape,edge_index)

    alltime = 0
    result_file_path = "train_memory_time.txt"

with open(result_file_path, "a") as result_file:
    for r in range(arg.run):

        print('*'*5 + 'Run {}'.format(r) + '*'*5)

        # Reading the training data set and testing data set
        m = loadmat('trainTestSplit/{}/sample{}_run{}.mat'.format(arg.name, arg.spc, r))
        tr_gt, te_gt = m['train_gt'], m['test_gt']
        tr_gt_torch, te_gt_torch = torch.from_numpy(tr_gt).long(), torch.from_numpy(te_gt).long()

        # Store label maps into fullGraph
        fullGraph.tr_gt, fullGraph.te_gt = tr_gt_torch, te_gt_torch

        # ------------------- Format Input Features -------------------
        data1 = data_normalization.astype(np.float32)
        data1 = torch.from_numpy(data1)
        data1 = data1.permute(2, 0, 1).unsqueeze(0)

        # ------------------- Build Model Components -------------------
        backbone = Backbone(c)
        student1_net = GraphNet3(c, 256,c)    # Student Network 1
        student2_net = KNNGraphNet(c, 256, c)  # Student Network 2
        kd = KD1(c, c, config.getint(arg.name, 'nc'))   # Graph Knowledge Distillation
        piexlsconvBlock = PiexlsConvBlock(c, c)    # Multi-scale Fusion Convolution
        mlp = MLPNet(c, c, config.getint(arg.name, 'nc'))
        mlp1 = MLPNet_1(c, c, config.getint(arg.name, 'nc'))
        featureFusion = FeatureFusionModel(c, c, config.getint(arg.name, 'nc'))   # Adaptive Weight Fusion

        # ------------------- Optimizer Configuration -------------------
        optimizer_all1 = torch.optim.Adam([
                                        {'params': backbone.parameters()},
                                        {'params': student1_net.parameters()},
                                        {'params': kd.parameters()},
                                        {'params': piexlsconvBlock.parameters(),'lr': 0.01},
                                        {'params': featureFusion.parameters()},
        ],
            weight_decay=arg.wd)

        # ------------------- Training Preparation ------------------
        criterion = nn.CrossEntropyLoss()    # Loss function
        trainer = JointTrainer([backbone,student1_net,student2_net,kd,piexlsconvBlock,featureFusion,mlp1])
        monitor = GradMonitor()

        # Select device
        device = torch.device('cuda:{}'.format(arg.gpu)) if arg.gpu != -1 else torch.device('cpu')
        max_acc = 0

        save_root = 'models/{}/{}/{}_overall_skip_2_SGConv_l1_clip'.format(arg.name, arg.spc, arg.block)
        pbar = tqdm(range(arg.epoch))

        for epoch in pbar:

            start = time.time()

            pbar.set_description_str('Epoch: {}'.format(epoch))

            # ----------- Training -----------
            train_loss = trainer.train_ce(Q,c,seg,data1, fullGraph, optimizer_all1, criterion, device, monitor.clear(), is_l1=True, is_clip=True)

            # ----------- Evaluation -----------
            te_loss,acc = trainer.evaluate(Q,c,seg,data1, fullGraph, criterion, device,r,epoch)

            # Update progress bar with metrics
            pbar.set_postfix_str('train_loss: {} te_loss:{}  acc1:{} acc2:{} acc3:{} acc4:{}'.format(train_loss,te_loss,acc1,acc2,acc3,acc4))

            temp = time.time() - start
            alltime = alltime + temp

            # ----------- Save Best Model -----------
            if acc > max_acc:
                max_acc = acc
                if not os.path.exists(save_root):
                    os.makedirs(save_root)
                trainer.save([
                              os.path.join(save_root, 'backboneNet_best_{}_{}.pkl'.format(arg.spc, r)),
                              os.path.join(save_root, 'student1_net_best_{}_{}.pkl'.format(arg.spc, r)),
                              os.path.join(save_root, 'student2_net_best_{}_{}.pkl'.format(arg.spc, r)),
                              os.path.join(save_root, 'kd_best_{}_{}.pkl'.format(arg.spc, r)),
                              os.path.join(save_root, 'piexlsconvBlock_best_{}_{}.pkl'.format(arg.spc, r)),
                              os.path.join(save_root, 'featureFusion_best_{}_{}.pkl'.format(arg.spc, r))
                ])

        # ----------- Log Peak Memory -----------
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"[Run {r}] Peak Memory Usage: {peak_memory:.2f} MB")
        result_file.write(f"Dataset: {arg.name}, Split Scale: {arg.spc}, [Run {r}] Peak Memory Usage: {peak_memory:.2f} MB\n")

    # ----------- Final Logging ----------
    result_file.write(f"Dataset: {arg.name}, Split Scale: {arg.spc}, Run: {r}, Total Time: {alltime:.2f} seconds\n")
    print("total time:",alltime)
    print('*'*5 + 'FINISH' + '*'*5)

