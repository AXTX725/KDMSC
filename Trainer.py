import torch
from torch_geometric.data import Data, Batch
from torch.optim import optimizer as optimizer_
from torch_geometric.utils import accuracy
from torch_geometric.nn import DataParallel
from torch.nn.utils import clip_grad_norm_
import time
from utils import construt_nosimilarity_graph,perturb_graph,update_teacher_params,average_superpixel,KNN_graph,construct_full_connected_graph,average_superpixel_and_centroids,KNN_graph_with_similarity
import torch.nn.functional as F
import scipy.io as sio
import numpy as np
# import torchprofile
from thop import profile


class JointTrainer(object):
    r'''Joint trainer'''
    def __init__(self, models: list):
        super().__init__()
        self.models = models

    def train_ce(self,Q,c,seg,data, fullGraph: Data, optimizer, criterion, device, monitor = None, is_l1=False, is_clip=False):
            '''
            Train the model for one epoch using cross-entropy loss.

            Parameters:
            Q              -- Transformation matrix from pixel space to superpixel space (shape: [num_pixels, num_superpixels])
            c              -- Number of channels (input feature dimension)
            seg            -- Superpixel segmentation map (2D array of shape [H, W])
            data           -- Normalized hyperspectral image tensor (shape: [1, C, H, W])
            fullGraph      -- PyTorch Geometric Data object containing edge_index, segmentation, and labels
            optimizer      -- Optimizer for all networks
            criterion      -- Loss function (e.g., nn.CrossEntropyLoss)
            device         -- Device to run model on ('cuda' or 'cpu')
            monitor        -- Gradient monitor (optional, default: None)
            is_l1          -- Whether to use L1 loss regularization (unused here)
            is_clip        -- Whether to apply gradient clipping (unused here)

            Returns:
            loss           -- Total loss value (sum of all loss terms)
            '''
            backbone = self.models[0]               # CNN backbone for pixel-level feature extraction
            Net_student1 = self.models[1]           # Student network 1
            Net_student2 = self.models[2]           # Student network 1
            kd = self.models[3]                     # Graph knowledge distillation module
            piexlsConv = self.models[4]             # Multi-scale fusion convolution
            featurefuision = self.models[5]         # Adaptive weight fusion module
            mlpNet1 = self.models[6]

            '''Set all models to training mode'''
            backbone.train()
            Net_student1.train()
            Net_student2.train()
            kd.train()
            piexlsConv.train()
            featurefuision.train()
            mlpNet1.train()

            '''Move all models and loss to target device'''
            backbone.to(device)
            Net_student1.to(device)
            Net_student2.to(device)
            kd.to(device)
            piexlsConv.to(device)
            featurefuision.to(device)
            mlpNet1.to(device)
            criterion.to(device)

            '''Move input data to device'''
            data = data.to(device)

            '''Extract initial pixel-level features using CNN backbone'''
            F1 = backbone(data)
            F1.to(device)

            '''Compute average features within each superpixel'''
            superpixel_features = average_superpixel(c, seg, F1)  # 超像素求平均
            superpixel_features_tensor = torch.tensor(superpixel_features, dtype=torch.float32).to('cuda')

            # Assign superpixel features to graph nodes
            fullGraph.x = superpixel_features_tensor
            fullGraph = fullGraph.to(device)

            '''Apply graph perturbation'''
            x1, edge_index1 = perturb_graph(fullGraph.x, fullGraph.edge_index, 0.10, 0.10)
            x2, edge_index2 = perturb_graph(fullGraph.x, fullGraph.edge_index, 0.10, 0.10)

            '''Pass perturbed features through student graph encoders'''
            h1 = Net_student1(x1, edge_index1)
            h2 = Net_student2(x2, edge_index2)

            '''Perform dual-student graph knowledge distillation'''
            logit_h1, logit_h2, emseble_logit = kd(fullGraph, h1, h2)

            F = piexlsConv(data)
            F.to(device)

            tran_superpixel_features = torch.matmul(torch.tensor(Q, dtype=F.dtype, device=F.device).T, F.view(-1, F.shape[-1]))

            # Perform feature fusion
            logits = featurefuision(fullGraph, h1, h2, tran_superpixel_features)

            pp_logits = mlpNet1(fullGraph,tran_superpixel_features)

            indices = torch.nonzero(fullGraph.tr_gt, as_tuple=True)

            y = fullGraph.tr_gt[indices].to(device) - 1

            node_number = fullGraph.seg[indices]

            pixel_logits_h1 = logit_h1[node_number]
            loss1 = criterion(pixel_logits_h1, y)

            pixel_logits_h2 = logit_h2[node_number]
            loss2 = criterion(pixel_logits_h2, y)

            loss_kd = 5.00E-06 * (
                    torch.sum(torch.abs(emseble_logit - logit_h1)) + torch.sum(torch.abs(emseble_logit - logit_h2)))

            pp_pixel_logits = pp_logits[node_number]
            loss4 = criterion(pp_pixel_logits, y)

            pixel_logits = logits[node_number]
            loss3 = criterion(pixel_logits, y)

            loss = loss1 + loss2 + loss_kd + loss3 + loss4

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            update_teacher_params(Net_student2, Net_student1, 0.90)

            return loss

    def evaluate(self, Q,c,seg,data, fullGraph, criterion, device,r,epoch):
        '''
             Evaluate the model on the given input data.

             This function performs a forward pass through all model components,
             records the FLOPs (floating point operations) of each module using thop's profile function,
             and computes the total loss and accuracy based on predictions from multiple sub-networks.

             Args:
                 Q: Superpixel-to-pixel transformation matrix.
                 c: Number of superpixels.
                 seg: Superpixel segmentation map.
                 data: Pixel-level input features.
                 fullGraph: Superpixel-level graph data.
                 criterion: Loss function.
                 device: Computation device.
                 r: Unused here (reserved for potential future use).
                 epoch: Current epoch (unused here but can be used for logging or analysis).

             Returns:
                 A tuple containing:
                     - loss (float): Total evaluation loss.
                     - accuracy (float): Classification accuracy on test nodes.
             '''
        with open("flops_per_iteration.txt", "a") as f:
            # Load model components
            backbone = self.models[0]
            Net_student1 = self.models[1]
            Net_student2 = self.models[2]
            kd = self.models[3]
            piexlsConv = self.models[4]
            featurefuision = self.models[5]
            mlpNet1 = self.models[6]

            # Set models to evaluation mode
            backbone.eval()
            Net_student1.eval()
            Net_student2.eval()
            kd.eval()
            piexlsConv.eval()
            featurefuision.eval()
            mlpNet1.eval()

            # Move models and data to device
            backbone.to(device)
            Net_student1.to(device)
            Net_student2.to(device)
            kd.to(device)
            piexlsConv.to(device)
            featurefuision.to(device)
            mlpNet1.to(device)
            criterion.to(device)

            data = data.to(device)

            with torch.no_grad():
                # Extract pixel-level features
                F1 = backbone(data)
                F1.to(device)

                # Compute FLOPs for the backbone
                macs, _ = profile(backbone, inputs=(data,))
                print(f"Backbone FLOPs: {macs}")
                f.write(f"Backbone FLOPs: {macs}\n")

                # Compute superpixel-level features by averaging pixel features
                superpixel_features = average_superpixel(c, seg, F1)
                superpixel_features_tensor = torch.tensor(superpixel_features, dtype=torch.float32).to('cuda')

                # Assign features to graph nodes and move to device
                fullGraph.x = superpixel_features_tensor
                fullGraph = fullGraph.to(device)

                # Generate two augmented graph of the graph
                x1, edge_index1 = perturb_graph(fullGraph.x, fullGraph.edge_index, 0.10, 0.10)
                x2, edge_index2 = perturb_graph(fullGraph.x, fullGraph.edge_index, 0.10, 0.10)

                # Forward pass through student networks
                h1 = Net_student1(x1, edge_index1)
                extNet1_macs, _ = profile(Net_student1, inputs=(x1, edge_index1))
                print(f"Net_student1 FLOPs: {extNet1_macs}")
                f.write(f"Net_student1 FLOPs: {extNet1_macs}\n")

                h2 = Net_student2(x2, edge_index2)
                extNet2_macs, _ = profile(Net_student2, inputs=(x2, edge_index2))
                print(f"Net_student2 FLOPs: {extNet2_macs}")
                f.write(f"Net_student2 FLOPs: {extNet2_macs}\n")

                # Knowledge distillation output
                logit_h1, logit_h2, emseble_logit = kd(fullGraph, h1, h2)
                kd_macs, _ = profile(kd, inputs=(fullGraph, h1, h2))
                print(f"kd FLOPs: {kd_macs}")
                f.write(f"kd FLOPs: {kd_macs}\n")

                # Pixel-level feature extraction
                F = piexlsConv(data)
                F.to(device)
                piexlsConv_macs, _ = profile(piexlsConv, inputs=(data,))
                print(f"piexlsConv FLOPs: {piexlsConv_macs}")
                f.write(f"piexlsConv FLOPs: {piexlsConv_macs}\n")

                # Transform pixel features into superpixel features using Q^T * F
                tran_superpixel_features = torch.matmul(torch.tensor(Q, dtype=F.dtype, device=F.device).T,
                                                    F.view(-1, F.shape[-1]))

                # Feature fusion to produce final logits
                logits = featurefuision(fullGraph, h1, h2, tran_superpixel_features)
                print("logits.shape:",logits.shape)

                featurefuision_macs, _ = profile(featurefuision, inputs=(fullGraph, h1, h2, tran_superpixel_features))
                print(f"featurefuision FLOPs: {featurefuision_macs}")
                f.write(f"featurefuision FLOPs: {featurefuision_macs}\n")

                # Pixel-level logits prediction from transformed features
                pp_logits = mlpNet1(fullGraph,tran_superpixel_features)

                mlpNet1_macs, _ = profile(mlpNet1, inputs=(fullGraph, tran_superpixel_features))
                print(f"mlpNet1 FLOPs: {mlpNet1_macs}")
                f.write(f"mlpNet1 FLOPs: {mlpNet1_macs}\n")

                total_flops = (macs + extNet1_macs + extNet2_macs +
                               kd_macs + piexlsConv_macs + featurefuision_macs +
                               mlpNet1_macs)
                f.write(f"Total FLOPs for this iteration: {total_flops}\n")

                pred1 = torch.argmax(logit_h1, dim=-1)
                pred2 = torch.argmax(logit_h2, dim=-1)
                pred3 = torch.argmax(logits, dim=-1)
                pred4 = torch.argmax(pp_logits, dim=-1)

                indices = torch.nonzero(fullGraph.te_gt, as_tuple=True)

                y = fullGraph.te_gt[indices].to(device) - 1
                node_number = fullGraph.seg[indices]

                # Get predictions from each module
                pixel_pred1 = pred1[node_number]
                pixel_pred2 = pred2[node_number]
                pixel_pred3 = pred3[node_number]
                pixel_pred4 = pred4[node_number]

                # Collect the logits for loss computation
                pixel_logits1 = logit_h1[node_number]
                pixel_logits2 = logit_h2[node_number]
                pixel_logits3 = logits[node_number]
                pixel_logits4 = pp_logits[node_number]

                loss1 = criterion(pixel_logits1, y)
                loss2 = criterion(pixel_logits2, y)

                # Compute knowledge distillation loss
                loss_kd = 5.00E-06 * (
                        torch.sum(torch.abs(emseble_logit - logit_h1)) + torch.sum(torch.abs(emseble_logit - logit_h2)))

                loss3 = criterion(pixel_logits3, y)
                loss4 = criterion(pixel_logits4, y)

                loss = loss1 + loss2 + loss_kd + loss3 + loss4
        return loss.item(), accuracy(pixel_pred3, y)

    # Getting prediction results
    def predict(self,Q,c,seg, data, fullGraph, device: torch.device):
        '''
                    Perform prediction on the input data using the trained model.

                    This function executes a forward pass through the model components
                    to generate the final class predictions without computing loss or gradients.

                    Args:
                       Q: Superpixel-to-pixel transformation matrix.
                       c: Number of superpixels.
                       seg: Superpixel segmentation map.
                       data: Pixel-level input features.
                       fullGraph: Superpixel-level graph data.
                       device: Computation device (CPU or CUDA).

                    Returns:
                        pred (Tensor): Predicted class indices for each superpixel.
                 '''
        backbone = self.models[0]
        Net_student1 = self.models[1]
        Net_student2 = self.models[2]
        kd = self.models[3]
        piexlsConv = self.models[4]
        featurefuision = self.models[5]

        backbone.eval()
        Net_student1.eval()
        Net_student2.eval()
        kd.eval()
        piexlsConv.eval()
        featurefuision.eval()


        backbone.to(device)
        Net_student1.to(device)
        Net_student2.to(device)
        kd.to(device)
        piexlsConv.to(device)
        featurefuision.to(device)

        data = data.to(device)

        with torch.no_grad():

            F1 = backbone(data)
            F1.to(device)

            superpixel_features = average_superpixel(c, seg, F1)
            superpixel_features_tensor = torch.tensor(superpixel_features, dtype=torch.float32).to('cuda')

            fullGraph.x = superpixel_features_tensor
            fullGraph = fullGraph.to(device)

            x1, edge_index1 = perturb_graph(fullGraph.x, fullGraph.edge_index, 0.10, 0.10)
            x2, edge_index2 = perturb_graph(fullGraph.x, fullGraph.edge_index, 0.10, 0.10)

            h1 = Net_student1(x1, edge_index1)
            h2 = Net_student2(x2, edge_index2)

            logit_h1, logit_h2, emseble_logit = kd(fullGraph, h1, h2)

            F = piexlsConv(data)
            F.to(device)

            tran_superpixel_features = torch.matmul(torch.tensor(Q, dtype=F.dtype, device=F.device).T,
                                                    F.view(-1, F.shape[-1]))

            logits = featurefuision(fullGraph, h1, h2, tran_superpixel_features)


        pred = torch.argmax(logits, dim=-1)

        return pred

    # Getting hidden features
    def getHiddenFeature(self, subGraph, fullGraph, device, gt = None, seg = None):
        intNet = DataParallel(self.models[0])
        extNet = self.models[1]
        intNet.eval()
        extNet.eval()
        intNet.to(device)
        extNet.to(device)
        with torch.no_grad():
            fe = intNet(subGraph.to_data_list())
            fullGraph.x = fe
            fullGraph = fullGraph.to(device)
            fe = extNet(fullGraph)
        if gt is not None and seg is not None:
            indices = torch.nonzero(gt, as_tuple=True)
            gt = gt[indices] - 1
            node_number = seg[indices].to(device)
            fe = fe[node_number]
            return fe.cpu(), gt
        else:
            return fe.cpu()

    def get_parameters(self):
        return self.models[0].parameters(), self.models[1].parameters()

    # Save the parameters of the trained model
    def save(self, paths):
        torch.save(self.models[0].cpu().state_dict(), paths[0])
        torch.save(self.models[1].cpu().state_dict(), paths[1])
        torch.save(self.models[2].cpu().state_dict(), paths[2])
        torch.save(self.models[3].cpu().state_dict(), paths[3])
        torch.save(self.models[4].cpu().state_dict(), paths[4])
        torch.save(self.models[5].cpu().state_dict(), paths[5])