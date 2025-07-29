import numpy as np
import torch
import cv2 as cv
from torch_scatter import scatter
from torch_geometric.data import Data
import copy
from torch import nn
import torch.nn.functional as F
# import GCL.augmentors as A
from sklearn.preprocessing import normalize


''' Getting adjacent relationship among nodes'''
def get_edge_index(segment):
    if isinstance(segment, torch.Tensor):
        segment = segment.numpy()
    # 扩张
    img = segment.astype(np.uint8)
    kernel = np.ones((3,3), np.uint8)
    expansion = cv.dilate(img, kernel)
    mask = segment == expansion
    mask = np.invert(mask)
    # 构图
    h, w = segment.shape
    edge_index = set()
    directions = ((-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1))
    indices = list(zip(*np.nonzero(mask)))
    for x, y in indices:
        for dx, dy in directions:
            adj_x, adj_y = x + dx, y + dy
            if -1 < adj_x < h and -1 < adj_y < w:
                source, target = segment[x, y], segment[adj_x, adj_y]
                if source != target:
                    edge_index.add((source, target))
                    edge_index.add((target, source))
    return torch.tensor(list(edge_index), dtype=torch.long).T, edge_index


''' Getting node features'''
def get_node(x, segment, mode='mean'):
    assert x.ndim == 3 and segment.ndim == 2
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if isinstance(segment, np.ndarray):
        segment = torch.from_numpy(segment).to(torch.long)
    c = x.shape[2]
    x = x.reshape((-1, c))
    mask = segment.flatten()
    nodes = scatter(x, mask, dim=0, reduce=mode)
    return nodes.to(torch.float32)


'''Constructing graphs by shifting'''
def get_grid_adj(grid):
    edge_index = list()
    # 上偏移
    a = np.full_like(grid, -1, dtype=np.int32)
    a[:-1] = grid[1:]
    adj = np.stack([grid, a], axis=-1)
    mask = adj != -1
    mask = np.logical_and(mask[..., 0], mask[..., 1])
    tmp = adj[mask]
    tmp = tmp.tolist()
    edge_index += tmp
    # 下偏移
    a = np.full_like(grid, -1, dtype=np.int32)
    a[1:] = grid[:-1]
    adj = np.stack([grid, a], axis=-1)
    mask = adj != -1
    mask = np.logical_and(mask[..., 0], mask[..., 1])
    tmp = adj[mask]
    tmp = tmp.tolist()
    edge_index += tmp
    # 左偏移
    a = np.full_like(grid, -1, dtype=np.int32)
    a[:, :-1] = grid[:, 1:]
    adj = np.stack([grid, a], axis=-1)
    mask = adj != -1
    mask = np.logical_and(mask[..., 0], mask[..., 1])
    tmp = adj[mask]
    tmp = tmp.tolist()
    edge_index += tmp
    # 右偏移
    a = np.full_like(grid, -1, dtype=np.int32)
    a[:, 1:] = grid[:, :-1]
    adj = np.stack([grid, a], axis=-1)
    mask = adj != -1
    mask = np.logical_and(mask[..., 0], mask[..., 1])
    tmp = adj[mask]
    tmp = tmp.tolist()
    edge_index += tmp
    return edge_index


''' Getting graph list'''
def get_graph_list(data, seg):
    graph_node_feature = []
    graph_edge_index = []
    for i in np.unique(seg):
        # 获取节点特征
        graph_node_feature.append(data[seg == i])
        # 获取邻接信息
        x, y = np.nonzero(seg == i)
        n = len(x)
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        grid = np.full((x_max - x_min + 1, y_max - y_min + 1), -1, dtype=np.int32)
        x_hat, y_hat = x - x_min, y - y_min
        grid[x_hat, y_hat] = np.arange(n)
        graph_edge_index.append(get_grid_adj(grid))
    graph_list = []
    # 数据变换
    for node, edge_index in zip(graph_node_feature, graph_edge_index):
        node = torch.tensor(node, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long).T
        graph_list.append(Data(node, edge_index=edge_index))
    return graph_list


def split(graph_list, gt, mask):
    indices = np.nonzero(gt)
    ans = []
    number = mask[indices]
    gt = gt[indices]
    for i, n in enumerate(number):
        graph = copy.deepcopy(graph_list[n])
        graph.y = torch.tensor([gt[i]], dtype=torch.long)
        ans.append(graph)
    return ans


def summary(net: nn.Module):
    single_dotted_line = '-' * 40
    double_dotted_line = '=' * 40
    star_line = '*' * 40
    content = []
    def backward(m: nn.Module, chain: list):
        children = m.children()
        params = 0
        chain.append(m._get_name())
        try:
            child = next(children)
            params += backward(child, chain)
            for child in children:
                params += backward(child, chain)
            # print('*' * 40)
            # print('{:>25}{:>15,}'.format('->'.join(chain), params))
            # print('*' * 40)
            if content[-1] is not star_line:
                content.append(star_line)
            content.append('{:>25}{:>15,}'.format('->'.join(chain), params))
            content.append(star_line)
        except:
            for p in m.parameters():
                if p.requires_grad:
                    params += p.numel()
            # print('{:>25}{:>15,}'.format(chain[-1], params))
            content.append('{:>25}{:>15,}'.format(chain[-1], params))
        chain.pop()
        return params
    # print('-' * 40)
    # print('{:>25}{:>15}'.format('Layer(type)', 'Param'))
    # print('=' * 40)
    content.append(single_dotted_line)
    content.append('{:>25}{:>15}'.format('Layer(type)', 'Param'))
    content.append(double_dotted_line)
    params = backward(net, [])
    # print('=' * 40)
    # print('-' * 40)
    content.pop()
    content.append(single_dotted_line)
    print('\n'.join(content))
    return params


def construt_nosimilarity_graph(features, nnodes):

    dist = F.cosine_similarity(features.detach().unsqueeze(1), features.detach().unsqueeze(0), dim=-1)
    weights_hp = torch.zeros((nnodes, nnodes))
    idx_ht = []

    dist_cpu = dist.cpu()
    dist_cpu_numpy = dist_cpu.numpy()

    k2 = 3
    for i in range(dist_cpu_numpy.shape[0]):
        idx = np.argpartition(dist_cpu_numpy[i, :], k2)[:k2]
        idx_ht.append(idx)

    for i, v in enumerate(idx_ht):
        for Nv in v:
            if Nv == i:
                pass
            else:
                weights_hp[i][Nv] = 1

    return weights_hp




def perturb_graph(features, edges, fea_mask_rate, edge_dropout_rate):
    # 删除节点特征
    feat_node = features.shape[1]
    device = features.device  # 获取 features 张量所在的设备
    mask = torch.zeros(features.shape, device=device)
    samples = torch.randperm(feat_node, device=device)[:int(feat_node * fea_mask_rate)]
    mask[:, samples] = 1
    features_1 = features * (1 - mask)

    # 删除边
    num_edges = edges.shape[1]
    edge_mask = torch.rand(num_edges, device=device) > edge_dropout_rate
    edges_1 = edges[:, edge_mask]

    return features_1, edges_1

''' mean teacher architecture'''
def update_teacher_params(teacher_net,student_net,alpha):
    for teacher_param, student_param in zip(teacher_net.parameters(), student_net.parameters()):
        teacher_param.data.mul_(alpha)
        teacher_param.data.add_((1 - alpha) * student_param.data)


def init_parameters(model, seed=42):
    torch.manual_seed(seed)
    for param in model.parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)

'''Compute the average within each superpixel to obtain its feature representation.'''
def average_superpixel(c,seg,data_normalization):
    num_superpixels = np.max(seg) + 1
    feature_dim = c

    superpixel_features = np.zeros((num_superpixels, feature_dim))

    for i in range(num_superpixels):

        mask = (seg == i)

        summed_features = torch.sum(data_normalization[mask], axis=0)

        averaged_features = summed_features / np.sum(mask)

        superpixel_features[i] = averaged_features.detach().cpu().numpy()
    return superpixel_features


'''Construct a KNN graph.'''
def KNN_graph(superpixel_features_tensor, k):

    dist_matrix = torch.cdist(superpixel_features_tensor, superpixel_features_tensor, p=2)  # 欧几里得距离

    top_k_values, top_k_indices = torch.topk(-dist_matrix, k + 1, dim=1)  # 负号表示寻找最小值

    edge_index = torch.cat([
        torch.arange(dist_matrix.size(0), device=superpixel_features_tensor.device).repeat_interleave(k).unsqueeze(0),
        top_k_indices[:, 1:].flatten().unsqueeze(0)
    ], dim=0)

    edge_weight = -top_k_values[:, 1:].flatten()

    num_edges = edge_index.size(1)
    shuffle_indices = torch.randperm(num_edges, device=superpixel_features_tensor.device)
    edge_index = edge_index[:, shuffle_indices]
    edge_weight = edge_weight[shuffle_indices]

    return edge_index, edge_weight


'''Construct a full connected graph.'''
def construct_full_connected_graph(features, sigma):

    dist_matrix = torch.cdist(features, features, p=2)

    weights = torch.exp(-dist_matrix ** 2 / sigma ** 2)

    col_sums = weights.sum(dim=0, keepdim=True)
    transition_matrix = weights / col_sums

    edge_index = torch.nonzero(transition_matrix, as_tuple=False).t()
    edge_weight = transition_matrix[edge_index[0], edge_index[1]]

    return edge_index, edge_weight


def KNN_graph_with_similarity(superpixel_features_tensor, spatial_positions_tensor, k, sigma_xy, sigma_rgb):

    N = superpixel_features_tensor.size(0)


    spatial_dist_matrix = torch.cdist(spatial_positions_tensor, spatial_positions_tensor, p=2)

    feature_dist_matrix = torch.cdist(superpixel_features_tensor, superpixel_features_tensor, p=2)

    spatial_similarity = torch.exp(-spatial_dist_matrix ** 2 / (2 * sigma_xy ** 2))

    feature_similarity = torch.exp(-feature_dist_matrix ** 2 / (2 * sigma_rgb ** 2))

    similarity_matrix = spatial_similarity * feature_similarity

    top_k_values, top_k_indices = torch.topk(similarity_matrix, k + 1, dim=1)

    edge_index = torch.cat([
        torch.arange(N, device=superpixel_features_tensor.device).repeat_interleave(k).unsqueeze(0),
        top_k_indices[:, 1:].flatten().unsqueeze(0)  # 排除自身节点，选择最相似的k个
    ], dim=0)

    edge_weight = top_k_values[:, 1:].flatten()

    num_edges = edge_index.size(1)
    shuffle_indices = torch.randperm(num_edges, device=superpixel_features_tensor.device)
    edge_index = edge_index[:, shuffle_indices]
    edge_weight = edge_weight[shuffle_indices]

    return edge_index, edge_weight


# def KNN_graph(superpixel_features_tensor, k):   # 双向KNN图
#     """
#     构造严格的双向KNN图，仅在两个节点互为最近邻时连边
#     :param superpixel_features_tensor: 节点特征张量，形状为 (N, F)，N是节点数，F是特征维度
#     :param k: 每个节点的K个最近邻
#     :return: edge_index (2, E), edge_weight (E,)
#     """
#     # 计算欧几里得距离矩阵
#     dist_matrix = torch.cdist(superpixel_features_tensor, superpixel_features_tensor, p=2)  # 欧几里得距离
#     # print("距离矩阵前5行:\n", dist_matrix[:5])
#
#     # 找到每个节点的 k 个最近邻
#     _, top_k_indices = torch.topk(-dist_matrix, k + 1, dim=1)  # 负号表示寻找最小值
#     top_k_indices = top_k_indices[:, 1:]  # 排除自身节点
#
#     # 构造邻接矩阵（每行表示一个节点的 k 近邻，置为 1 表示有连接）
#     N = dist_matrix.size(0)
#     adjacency_matrix = torch.zeros(N, N, device=superpixel_features_tensor.device)
#     for i in range(N):
#         adjacency_matrix[i, top_k_indices[i]] = 1
#
#     # 找到互为最近邻的节点对（相交的非零位置）
#     bidirectional_adjacency = adjacency_matrix * adjacency_matrix.T
#     edge_index = bidirectional_adjacency.nonzero(as_tuple=False).T
#
#     # 提取对应边的权重（欧几里得距离）
#     edge_weight = dist_matrix[edge_index[0], edge_index[1]]
#
#     return edge_index, edge_weight

def compute_transformation_matrix(superpixel_labels, num_superpixels):
        """
        计算像素到超像素的转换矩阵 Q。

        参数：
        - superpixel_labels: H × W 的二维数组，每个值是像素所属的超像素编号。
        - num_superpixels: 超像素的数量。

        返回：
        - Q: 转换矩阵 (H×W) × Z，其中 Z 是超像素数量。
        """
        height, width = superpixel_labels.shape
        num_pixels = height * width
        Q = np.zeros((num_pixels, num_superpixels), dtype=np.float32)
        print("compute_transformation_matrix")
        print("Q.shape", Q.shape)

        # 将二维像素索引映射到超像素
        for i in range(height):
            for j in range(width):
                pixel_index = i * width + j
                superpixel_index = superpixel_labels[i, j]
                Q[pixel_index, superpixel_index] = 1

        # 对列归一化
        Q_normalized = normalize(Q, norm='l1', axis=0)
        return Q_normalized