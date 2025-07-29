from torch_geometric import nn as gnn
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GATConv, BatchNorm


class GraphNet3(nn.Module):
    '''
    A three-layer Graph Neural Network (GNN) model for node feature extraction.

    This network consists of:
    - One initial BatchNorm layer for input normalization.
    - Three graph convolution layers (GCNConv and GraphConv).
    - Batch normalization and ReLU activation after each convolution.

    Parameters:
    - c_in: Dimension of input features
    - hidden_size1: Hidden size for the first two layers
    - hidden_size2: Hidden size for the third layer (output dimension)

    Forward Inputs:
    - x: Input node features (shape: [num_nodes, c_in])
    - edge_index: Graph connectivity (shape: [2, num_edges])

    Returns:
    - h: Output node features after 3-layer GNN processing
    '''
    def __init__(self, c_in, hidden_size1,hidden_size2):
        super().__init__()
        self.bn_0 = gnn.BatchNorm(c_in)
        self.gcn_1 = gnn.GCNConv(c_in, hidden_size1)
        self.bn_1 = gnn.BatchNorm(hidden_size1)
        self.gcn_2 = gnn.GraphConv(hidden_size1, hidden_size1)
        self.bn_2 = gnn.BatchNorm(hidden_size1)
        self.gcn_3 = gnn.GraphConv(hidden_size1, hidden_size2)
        self.bn_3 = gnn.BatchNorm(hidden_size2)

    def forward(self, x,edge_index):

        x_normalization = self.bn_0(x)
        h = self.bn_1(F.relu(self.gcn_1(x_normalization, edge_index)))
        h = self.bn_2(F.relu(self.gcn_2(h, edge_index)))
        h = self.bn_3(F.relu(self.gcn_3(h, edge_index)))

        return h


# External graph convolution feature module
class GraphNetFeature(nn.Module):
    '''
    Graph neural network model for feature extraction from graph-structured data.
    This model consists of two GCN layers, each followed by BatchNorm and ReLU activation.
    A residual connection is added between the normalized input and the final output.

    Args:
        c_in: Dimension of input features
        hidden_size: Dimension of hidden layers
    '''
    def __init__(self, c_in, hidden_size):
        super().__init__()
        self.bn_0 = gnn.BatchNorm(c_in)
        self.gcn_1 = gnn.GCNConv(c_in, hidden_size)
        self.bn_1 = gnn.BatchNorm(hidden_size)
        self.gcn_2 = gnn.GCNConv(hidden_size, hidden_size)
        self.bn_2 = gnn.BatchNorm(hidden_size)

    def forward(self, graph):
        x_normalization = self.bn_0(graph.x)
        # x_normalization = graph.x
        h = self.bn_1(F.relu(self.gcn_1(x_normalization, graph.edge_index)))
        h = self.bn_2(F.relu(self.gcn_2(h, graph.edge_index)))
        return x_normalization + h



class MLPNet(nn.Module):
    '''
    MLP-based classification head for graph features.

    This module takes in the output from a graph feature extractor along with the original
    input features (after normalization), combines them using a residual connection,
    and applies an MLP to generate the final classification logits.

    Args:
        c_in: Dimension of input node features
        hidden_size: Dimension of hidden graph features
        nc: Number of output classes
    '''
    def __init__(self,c_in, hidden_size, nc):
        super().__init__()
        self.bn_0 = gnn.BatchNorm(c_in)
        self.classifier1 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(hidden_size // 2, nc)
        )

    def forward(self, graph,h):
        x_normalization = self.bn_0(graph.x)
        logits = self.classifier1(h + x_normalization)
        return logits


class MLPNet_1(nn.Module):

    def __init__(self,c_in, hidden_size, nc):
        super().__init__()
        self.bn_0 = gnn.BatchNorm(c_in)
        self.classifier1 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(hidden_size // 2, nc)
        )

    def forward(self, graph,h):
        x_normalization = self.bn_0(graph.x)
        logits = self.classifier1(h + x_normalization)
        return logits


class Conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):

        x = x.unsqueeze(0)
        x = x.permute(0, 2, 1)
        x = x.unsqueeze(3)
        return self.conv(x).squeeze(0).squeeze(2).permute(1, 0)


class SuperConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(SuperConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class Backbone(nn.Module):
    def __init__(self,in_channels):
        super(Backbone, self).__init__()
        self.block1 = SuperConvBlock(in_channels, 128, kernel_size=3, stride=1, padding=1)
        self.block2 = SuperConvBlock(128, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = x.squeeze(0).permute(1, 2, 0)
        return x


class full_graphConv(nn.Module):
    def __init__(self, c_in, hidden_size1, hidden_size2):
        super().__init__()
        self.bn_0 = gnn.BatchNorm(c_in)
        self.gcn_1 = gnn.GCNConv(c_in, hidden_size1)
        self.bn_1 = gnn.BatchNorm(hidden_size1)
        self.gcn_2 = gnn.GraphConv(hidden_size1, hidden_size1)
        self.bn_2 = gnn.BatchNorm(hidden_size1)
        self.gcn_3 = gnn.GraphConv(hidden_size1, hidden_size2)
        self.bn_3 = gnn.BatchNorm(hidden_size2)

    def forward(self, x, edge_index,edge_weight):
        x_normalization = self.bn_0(x)
        h = self.bn_1(F.relu(self.gcn_1(x_normalization, edge_index, edge_weight=edge_weight)))
        h = self.bn_2(F.relu(self.gcn_2(h, edge_index, edge_weight=edge_weight)))
        h = self.bn_3(F.relu(self.gcn_3(h, edge_index, edge_weight=edge_weight)))

        return h


class KNNGraphNet(nn.Module):
    def __init__(self, c_in, hidden_size1,hidden_size2):
        super().__init__()
        self.bn_0 = gnn.BatchNorm(c_in)
        self.gcn_1 = gnn.GCNConv(c_in, hidden_size1)
        self.bn_1 = gnn.BatchNorm(hidden_size1)
        self.gcn_2 = gnn.GraphConv(hidden_size1, hidden_size1)
        self.bn_2 = gnn.BatchNorm(hidden_size1)
        self.gcn_3 = gnn.GraphConv(hidden_size1, hidden_size2)
        self.bn_3 = gnn.BatchNorm(hidden_size2)

    def forward(self, x,edge_index):

        x_normalization = self.bn_0(x)
        h = self.bn_1(F.relu(self.gcn_1(x_normalization, edge_index)))
        h = self.bn_2(F.relu(self.gcn_2(h, edge_index)))
        h = self.bn_3(F.relu(self.gcn_3(h, edge_index)))

        return h

class GATGraphNet(nn.Module):
    def __init__(self, c_in, hidden_size1, hidden_size2, heads=4):

        super().__init__()
        self.bn_0 = BatchNorm(c_in)

        self.gat_1 = GATConv(c_in, hidden_size1 // heads, heads=heads, concat=True)
        self.bn_1 = BatchNorm(hidden_size1)

        self.gat_2 = GATConv(hidden_size1, hidden_size1 // heads, heads=heads, concat=True)
        self.bn_2 = BatchNorm(hidden_size1)

        self.gat_3 = GATConv(hidden_size1, hidden_size2, heads=1, concat=False)
        self.bn_3 = BatchNorm(hidden_size2)

    def forward(self, x, edge_index):
        x_normalization = self.bn_0(x)

        h = self.bn_1(F.elu(self.gat_1(x_normalization, edge_index)))

        h = self.bn_2(F.elu(self.gat_2(h, edge_index)))

        h = self.bn_3(F.elu(self.gat_3(h, edge_index)))

        return h

class KD1(nn.Module):
    def __init__(self,c_in, hidden_size, nc):
        super().__init__()
        self.bn_0 = gnn.BatchNorm(c_in)
        self.classifier1 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, nc)
        )

        self.classifier2 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, nc)
        )

        self.fc_ensemble = nn.Linear(2 * hidden_size, nc)
        self.w1 = nn.Linear(hidden_size, hidden_size)
        self.w2 = nn.Linear(hidden_size, hidden_size)

        self.sigmoid = nn.Sigmoid()

    def forward(self, graph,h1,h2):

        x_normalization = self.bn_0(graph.x)
        logits1 = self.classifier1(h1 + x_normalization)    # student1's logit
        logits2 = self.classifier2(h2 + x_normalization)    # student2's logit


        '''Knowledge Distillation (KD)'''
        theta = self.sigmoid(self.w1(h1) + self.w2(h2))    # Gating coefficient

        out_d = theta * h1
        out_h = (1 - theta) * h2
        emseble_logit = self.sigmoid(self.fc_ensemble(torch.cat([out_d, out_h], -1)))  # teacher's logit

        logits1 = logits1/0.5
        logits2 = logits2/0.5
        emseble_logit = emseble_logit/0.5

        return logits1,logits2,emseble_logit

class PiexlsConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PiexlsConvBlock, self).__init__()

        self.conv1x1_1 = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.LeakyReLU(inplace=True)
        )
        self.conv1x1_2 = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.LeakyReLU(inplace=True)
        )

        # Cross Multi-Scale Convolution
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv5x5 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv7x7 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        x = self.conv1x1_1(x)
        # print("piexlsConvBlock-conv1x1_1",x.shape)
        x = self.conv1x1_2(x)
        # print("piexlsConvBlock-conv1x1_2",x.shape)

        # Cross Multi-Scale Convolution
        out3x3 = self.conv3x3(x)
        # print("piexlsConvBlock-conv3x3", x.shape)
        out5x5 = self.conv5x5(x)
        # print("piexlsConvBlock-conv5x5", x.shape)
        out7x7 = self.conv7x7(x)
        # print("piexlsConvBlock-conv7x7", x.shape)

        # Element-wise addition of all scales
        # out = out3x3 + out5x5 + out7x7

        out = out3x3 * out5x5 * out7x7

        # Cross Multi-Scale Convolution
        out3x3 = self.conv3x3(out)
        # print("piexlsConvBlock-conv3x3", x.shape)
        out5x5 = self.conv5x5(out)
        # print("piexlsConvBlock-conv5x5", x.shape)
        out7x7 = self.conv7x7(out)
        # print("piexlsConvBlock-conv7x7", x.shape)
        out = out3x3 * out5x5 * out7x7

        out = out.squeeze(0).permute(1, 2, 0)  # 调整维度为 (height, width, channels)

        return out

class FeatureFusionModel(nn.Module):
    def __init__(self, c_in, hidden_size, nc):
        super(FeatureFusionModel, self).__init__()
        self.bn_0 = gnn.BatchNorm(c_in)

        # Learnable parameters r1, r2, r3
        self.r1 = nn.Parameter(torch.tensor(0.5))
        self.r2 = nn.Parameter(torch.tensor(0.5))
        self.r3 = nn.Parameter(torch.tensor(0.5))

        self.classifier1 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, nc)
        )

        self.classifier2 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, nc)
        )

    def forward(self, graph, F1, F2, F3):
        # F1, F2, F3 are the three feature streams obtained from the DSGKD and MSFC modules.

        x_normalization = self.bn_0(graph.x)
        F4 = self.r1 * F1 + (1 - self.r1) * F2
        F5 = self.r2 * F4 + (1 - self.r2) * F3

        s4 = self.classifier1(F4 + x_normalization)
        s5 = self.classifier2(F5 + x_normalization)

        s = self.r3 * s5 + (1 - self.r3) * s4

        return s