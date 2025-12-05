import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Optional, Dict, Union, Any, Tuple, Mapping
import random
from torch.utils.data import DataLoader, TensorDataset
import os
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm
import umap 

def set_random_seed(seed):
    random.seed(seed)       # 设置 Python 内置随机数生成器的种子
    np.random.seed(seed)    # 设置 NumPy 随机数生成器的种子
    torch.manual_seed(seed) # 设置 Torch 随机数生成器的种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # 将所有设备上的随机数生成器的种子设置为相同的

def set_device(dev):
    if dev is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif dev:
        device = torch.device(dev)
    # device = torch.device("cpu")
    return device

def create_project_folders(project_name: str) -> str:
    """创建项目文件夹并返回项目根目录路径

    Args:
        project_name: 项目名称

    Returns:
        str: 项目根目录的路径
    """
    project_dir = os.path.join(os.getcwd(), project_name)
    os.makedirs(project_dir, exist_ok=True)
    return project_dir

def normalize_spectra(data: np.ndarray) -> np.ndarray:
    """
    将光谱数据归一化到 [0, 1] 范围。

    Args:
        data: 输入的光谱数据

    Returns:
        normalized_data: 归一化后的光谱数据
    """
    # 初始化归一化后数据
    normalized_data = np.zeros_like(data)

    # 对每条光谱进行归一化
    for i in range(data.shape[0]):
        max_value = np.max(data[i])  # 找到当前光谱的最大值
        if max_value > 0:  # 确保最大值大于0以避免除以零
            normalized_data[i] = data[i] / max_value
        else:
            normalized_data[i] = 0  # 如果最大值为0，则归一化结果也直接为0

    return normalized_data

def prepare_data_loader(
    data: np.ndarray,
    labels: np.ndarray,
    batch_size: int = 128,
    device: Optional[str] = None,
    shuffle: bool = True
) -> Tuple[DataLoader, np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    准备数据加载器,将numpy数组转换为PyTorch Tensor,并将数据和标签移动到合适的设备上。

    Args:
        data: 形状为(n_samples, n_features)的输入数据数组
        labels: 长度为n_samples的标签数组
        batch_size: 批次大小,默认为128
        shuffle: 是否打乱数据,默认为True

    Returns:
        dataloader: PyTorch DataLoader对象
        unique_label: 唯一标签值的numpy数组
        tensor_data: 转换为Tensor的输入数据
        tensor_labels: 转换为Tensor的标签数据
        tensor_gpu_data: 移动到GPU的输入数据Tensor
        tensor_gpu_labels: 移动到GPU的标签数据Tensor
        device: 使用的设备,either 'cpu' or 'cuda'
    """
    # 确保标签为整数类型
    labels = labels.astype(int)

    # 获取唯一标签值
    unique_label = np.unique(labels)

    # norm
    # norm_data = normalize_spectra(data)

    # 将数据转换为PyTorch Tensor
    tensor_data = torch.tensor(data, dtype=torch.float32)
    tensor_labels = torch.tensor(labels, dtype=torch.int32)

    # 确定设备类型
    device = device

    # 将数据和标签移动到设备
    tensor_gpu_data = tensor_data.to(device)
    tensor_gpu_labels = tensor_labels.to(device)

    # 创建数据加载器
    dataloader = DataLoader(
        TensorDataset(tensor_gpu_data, tensor_gpu_labels),
        batch_size=batch_size,
        shuffle=shuffle
    )

    return (
        dataloader,
        unique_label,
        tensor_data,
        tensor_labels,
        tensor_gpu_data,
        tensor_gpu_labels
    )

def visualize_clusters(
    z: np.ndarray,
    labels: np.ndarray,
    gmm_labels: np.ndarray,
    leiden_labels: np.ndarray,
    ari_gmm: float,
    ari_leiden: float,
    save_path: str,
    random_state: int = 42
) -> None:
    """
    z visualization

    Args:
        z: (n_samples, n_features)
        labels: sample label
        save_path: save plot to path
        colors_map: colors dict
        random_state: t-SNE的随机种子，默认为42
        fig_size: 图像尺寸，默认为(10, 8)

    Returns:
        None
    """

    umap_reducer = umap.UMAP( n_components=2, n_neighbors=15,  min_dist=0.1, metric='euclidean') # , random_state=random_state
    z_umap = umap_reducer.fit_transform(z)

    def _build_cmap(num_colors):
        n = max(1, int(num_colors))
        palette = sns.color_palette('husl', n_colors=n)
        if n == 1:
            return ListedColormap(palette)
        return LinearSegmentedColormap.from_list('custom', palette)

    # 绘图
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 8))
    
    # 创建大调色板
    custom_cmap = _build_cmap(len(np.unique(labels)))
    scatter1 = ax1.scatter(z_umap[:, 0], z_umap[:, 1], c=labels, cmap=custom_cmap)
    ax1.set_title('True Labels')
    legend1 = ax1.legend(*scatter1.legend_elements(), title="Classes", bbox_to_anchor=(1.05, 1), loc='best', fontsize='small')
    ax1.add_artist(legend1)
    
    # 绘制GMM预测聚类结果的散点图
    custom_cmap = _build_cmap(len(np.unique(gmm_labels)))
    scatter2 = ax2.scatter(z_umap[:, 0], z_umap[:, 1], c=gmm_labels, cmap=custom_cmap)
    ax2.set_title(f'GMM Predicted Clusters\nARI: {ari_gmm:.3f}')
    legend2 = ax2.legend(*scatter2.legend_elements(num=len(np.unique(gmm_labels))), title="Clusters", bbox_to_anchor=(1.05, 1), loc='best', fontsize='small')
    ax2.add_artist(legend2)

    # 绘制Leiden预测聚类结果的散点图
    custom_cmap = _build_cmap(len(np.unique(leiden_labels)))
    scatter3 = ax3.scatter(z_umap[:, 0], z_umap[:, 1], c=leiden_labels, cmap=custom_cmap)
    ax3.set_title(f'Leiden Predicted Clusters\nARI: {ari_leiden:.3f}')
    legend3 = ax3.legend(*scatter3.legend_elements(num=len(np.unique(leiden_labels))), title="Clusters", bbox_to_anchor=(1.05, 1), loc='best', fontsize='small')
    ax3.add_artist(legend3)
    
    # 保存图像
    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=500)
    plt.close()

def plot_UMAP(
    X: np.ndarray,
    labels: np.ndarray,
    save_path: str,
    random_state: int = 42):

    umap_reducer = umap.UMAP( n_components=2, n_neighbors=15,  min_dist=0.1, metric='euclidean') # , random_state=random_state
    z_umap = umap_reducer.fit_transform(X)

    # 绘图
    uniq_labels = np.unique(labels)
    n_classes = len(uniq_labels)
    label_to_idx = {lbl: i for i, lbl in enumerate(uniq_labels)}
    labels_idx = np.array([label_to_idx[lbl] for lbl in labels])
    
    if n_classes == 2:
        palette = ['#F28E2B', '#9E9E9E']  # 橙 + 灰
    else:
        palette = sns.color_palette('husl', n_colors=n_classes)
    cmap = ListedColormap(palette)
    norm = BoundaryNorm(np.arange(n_classes+1)-0.5, n_classes)

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(z_umap[:, 0], z_umap[:, 1], c=labels_idx, cmap=cmap, norm=norm, s=10, linewidths=0, alpha=0.5)

    handles, _ = sc.legend_elements()
    for h, lbl in zip(handles, uniq_labels):
        h.set_label(str(lbl))
    ax.legend(handles=handles, title="Label", loc='best', fontsize='small', frameon=False)

    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=500)
    plt.close()

def plot_spectra(
        recon_data: np.ndarray,
        labels: np.ndarray,
        save_path: str,
        wavenumber: Optional[np.ndarray] = None
) -> None:
    """
    Args:
        X: (n_samples, n_features)
        labels: sample labels
        save_path: save plot to path
        wavenumber: spec wavenumber
    Returns:
        None
    """
    x = np.arange(recon_data.shape[1]) if wavenumber is None else wavenumber
    unique_labels = np.unique(labels)  
    stack_gap = float(np.mean(np.max(recon_data, axis=1))) * 0.6 

    palette = sns.color_palette('husl', n_colors=len(unique_labels))
    colors_map = {lbl: palette[i] for i, lbl in enumerate(unique_labels)}

    plt.figure(figsize=(14, 4 + 0.6 * len(unique_labels)))
    for i, lbl in enumerate(unique_labels):
        grp = recon_data[labels == lbl]
        if grp.size == 0:
            continue
        mean = grp.mean(axis=0)
        sd = grp.std(axis=0, ddof=1) if grp.shape[0] > 1 else np.zeros_like(mean)
        offset = -i * stack_gap
        color = colors_map[lbl]

        plt.fill_between(x, mean - sd + offset, mean + sd + offset, color=color, alpha=0.5, linewidth=0)
        plt.plot(x, mean + offset, color=color, lw=2, label=f'Cluster {lbl} (n={grp.shape[0]})')

    plt.xlabel('Wavenumber' if wavenumber is not None else 'Index')
    plt.ylabel('Intensity')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=500)
    plt.close()

def plot_S(S, matched_S, matched_chem, save_path, wavenumber):
    valid_idx = np.where((wavenumber >= 450) & (wavenumber <= 1800))[0]
    wn_valid = wavenumber[valid_idx]
    stack_gap = float(np.mean(np.max(matched_S, axis=1))) * 1.5

    S = S.detach().cpu().numpy()
    row_max_valid = np.max(S[:, valid_idx], axis=1, keepdims=True) + 1e-12
    S = S / row_max_valid

    plt.figure(figsize=(12, 8))
    n_components = S.shape[0]
    palette = sns.color_palette('husl', n_colors=n_components)

    for i in range(n_components):
        plt.plot(wn_valid, matched_S[i,:] -i * stack_gap, ls='--',color=palette[i], label=f'Component {i+1} : {matched_chem[i]}')
        plt.plot(wavenumber, S[i,:] -i * stack_gap, color=palette[i])
     
    plt.xlabel('Wavenumber')
    plt.ylabel('Intensity')
    plt.title('MCR Component Spectra')
    plt.legend()
    plt.grid(alpha=0.25)
    plt.savefig(save_path)

def leiden_clustering(spectra, n_neighbors=20, resolution=1.0, seed=42):
    """使用Leiden算法进行聚类
    
    Args:
        spectra: 输入数据
        n_neighbors: KNN图中的邻居数量，默认15
        resolution: 聚类分辨率参数，默认1.0
        seed: 随机数种子，默认42
    
    Returns:
        np.array: 聚类标签数组
    """
    try:
        import leidenalg
        import igraph as ig
        from sklearn.neighbors import kneighbors_graph
        import scipy.sparse as sparse
    except ImportError:
        raise ImportError("请安装必要的包：leidenalg, python-igraph, scikit-learn")
    
    # 构建KNN图
    knn_graph = kneighbors_graph(spectra, n_neighbors=n_neighbors, mode='distance')
    
    # 转换为igraph格式
    sources, targets = knn_graph.nonzero()
    weights = knn_graph.data
    edges = list(zip(sources, targets))
    
    g = ig.Graph(edges=edges, directed=False)
    g.es['weight'] = weights
    
    # 使用Leiden算法进行聚类，设置随机种子
    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights='weight',
        resolution_parameter=resolution,
        seed=seed  # 添加随机数种子
    )
    
    return np.array(partition.membership)

class WeightScheduler:
    def __init__(self, init_weights, max_weights, n_epochs, resolution):
        """
        Args:
            init_weights: 初始权重字典 {'lamb1': 1.0, 'lamb2': 0.1, ...}
            max_weights: 最终权重字典
            n_epochs: 总训练轮数
        """
        self.init_weights = init_weights
        self.max_weights = max_weights
        self.n_epochs = n_epochs
        self.warmup_epochs = n_epochs // 5  # 预热期为总轮数的1/5
        self.resolution = resolution
        
    def get_weights(self, epoch):
        """获取当前epoch的权重"""
        # 预热期：线性增加
        if epoch < self.warmup_epochs:
            ratio = epoch / self.warmup_epochs
        else:
            # 预热后：余弦退火
            ratio = 0.5 * (1 + np.cos(
                np.pi * (epoch - self.warmup_epochs) / 
                (self.n_epochs - self.warmup_epochs)
            ))
             
        weights = {}
        for key in self.init_weights:
            weights[key] = self.init_weights[key] + (
                self.max_weights[key] - self.init_weights[key]
            ) * ratio
            
        return weights 
