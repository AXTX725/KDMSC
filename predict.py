'''Predicting'''
from scipy.io import loadmat, savemat
import numpy as np
import argparse
import configparser
import torch
from torch import nn
from torch_geometric.data import Data, Batch
from skimage.segmentation import slic, mark_boundaries
from sklearn.preprocessing import scale
import os
from PIL import Image
from utils import get_graph_list, get_edge_index,compute_transformation_matrix
import math
from Model.module import GraphNet3,MLPNet,MLPNet_1,Conv1x1,Backbone,KNNGraphNet,KD1,PiexlsConvBlock,FeatureFusionModel
from Trainer import JointTrainer
import time


if __name__ == '__main__':
    # Argument parser for command-line configuration
    parser = argparse.ArgumentParser(description='TRAIN THE OVERALL')
    parser.add_argument('--name', type=str, default='PaviaU',
                        help='DATASET NAME')
    parser.add_argument('--block', type=int, default=400,
                        help='BLOCK SIZE')
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

    # Load dataset configuration from ini file
    config = configparser.ConfigParser()
    config.read('dataInfo.ini')

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
        # Saving
        np.save(seg_path, seg)
    # Compute transformation matrix Q from superpixels
    Q = compute_transformation_matrix(seg, np.max(seg) + 1)
    print("transformation matrix Q.shape", Q.shape)

    # Constructing full graphs
    full_edge_index_path = 'data/{}/{}_edge_index.npy'.format(arg.name, arg.block)
    if os.path.exists(full_edge_index_path):
        edge_index = np.load(full_edge_index_path)
    else:
        edge_index, _ = get_edge_index(seg)
        np.save(full_edge_index_path,
                edge_index if isinstance(edge_index, np.ndarray) else edge_index.cpu().numpy())
    fullGraph = Data(None,
                    edge_index=torch.from_numpy(edge_index) if isinstance(edge_index, np.ndarray) else edge_index,
                    seg=torch.from_numpy(seg) if isinstance(seg, np.ndarray) else seg)

    backbone = Backbone(c)
    student1_net = GraphNet3(c, 256, c)
    student2_net = KNNGraphNet(c, 256, c)
    kd = KD1(c, c, config.getint(arg.name, 'nc'))
    piexlsconvBlock = PiexlsConvBlock(c, c)
    mlp = MLPNet(c, c, config.getint(arg.name, 'nc'))
    featureFusion = FeatureFusionModel(c, c, config.getint(arg.name, 'nc'))

    device = torch.device('cuda:{}'.format(arg.gpu)) if arg.gpu != -1 else torch.device('cpu')

    alltime = 0
    result_file_path = "test_results.txt"

with open(result_file_path, "a") as result_file:
    for r in range(arg.run):

        start = time.time()
        data1 = data_normalization.astype(np.float32)
        data1 = torch.from_numpy(data1)
        data1 = data1.permute(2, 0, 1).unsqueeze(0)

        # ====== Load Pretrained Model Weights ======
        backbone.load_state_dict(
            torch.load(f"models/{arg.name}/{arg.spc}/{arg.block}_overall_skip_2_SGConv_l1_clip/backboneNet_best_{arg.spc}_{r}.pkl"))
        student1_net.load_state_dict(
                torch.load(f"models/{arg.name}/{arg.spc}/{arg.block}_overall_skip_2_SGConv_l1_clip/student1_net_best_{arg.spc}_{r}.pkl"))
        student2_net.load_state_dict(
            torch.load(
                f"models/{arg.name}/{arg.spc}/{arg.block}_overall_skip_2_SGConv_l1_clip/student2_net_best_{arg.spc}_{r}.pkl"))
        kd.load_state_dict(
            torch.load(f"models/{arg.name}/{arg.spc}/{arg.block}_overall_skip_2_SGConv_l1_clip/kd_best_{arg.spc}_{r}.pkl"))
        piexlsconvBlock.load_state_dict(
            torch.load(
                f"models/{arg.name}/{arg.spc}/{arg.block}_overall_skip_2_SGConv_l1_clip/piexlsconvBlock_best_{arg.spc}_{r}.pkl"))
        featureFusion.load_state_dict(
            torch.load(
                f"models/{arg.name}/{arg.spc}/{arg.block}_overall_skip_2_SGConv_l1_clip/featureFusion_best_{arg.spc}_{r}.pkl"))

        trainer = JointTrainer([backbone,student1_net,student2_net,kd,piexlsconvBlock,featureFusion])

        # predicting
        preds = trainer.predict(Q,c,seg,data1, fullGraph, device)
        seg_torch = torch.from_numpy(seg)
        map = preds[seg_torch]

        temp = time.time() - start
        alltime = alltime + temp

        # ====== Save Prediction Result ======
        save_root = 'prediction/{}/{}/{}_overall_skip_2_SGConv_l1_clip'.format(arg.name, arg.spc, arg.block)
        if not os.path.exists(save_root):
            os.makedirs(save_root)
        save_path = os.path.join(save_root, '{}.mat'.format(r))
        savemat(save_path, {'pred': map.cpu().numpy()})

    # ====== Log Runtime ======
    result_file.write(f"Dataset: {arg.name}, Split Scale: {arg.spc}, Run: {r}, Total Time: {alltime:.2f} seconds\n")
    print("total time:",alltime)
    print('*'*5 + 'FINISH' + '*'*5)

