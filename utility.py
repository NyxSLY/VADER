import time
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import matplotlib
from sklearn.manifold import TSNE
from typing import Optional, Dict, Union, Any, Tuple, Mapping
from torch.distributions import Normal
import random
from torch.utils.data import DataLoader, TensorDataset
import os
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist, pdist
import datetime, pywt
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from scipy.sparse import spmatrix, csr_matrix
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import torch.nn.functional as F

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

def choose_kmeans(model,dataloader,num_classes):
    print("开始k-means方法比较实验...")
    # 选择并更新kmeans方法
    results = compare_kmeans_methods(
        model=model,
        dataloader=dataloader,
        num_classes=num_classes,
        n_runs=5
    )
    kmeans_method = ('k-means++' if np.mean(results['kmeans++']['silhouette']) >
                                    np.mean(results['kmeans']['silhouette']) else 'random')
    print(f"\n选择使用{kmeans_method}方法")
    return kmeans_method


def sample_plot(
        generated_samples: np.ndarray,
        generated_labels: np.ndarray,
        ncol: int = 2,
        title: Optional[str] = 'Generated Samples by Category',
        colors_map: Optional[Dict[int, str]] = None,
        save_path: Optional[str] = None
) -> None:
    """
    按照聚类类别展示生成的样本，每个类别占一行，通过ncol控制每行展示的样本数量。

    Args:
        generated_samples: 形状为(n_samples, n_features)的生成样本数组
        generated_labels: 长度为n_samples的标签数组
        ncol: 每行展示的样本数量，默认为2
        colors_map： colors dict
        title: 图像标题，默认为'Generated Samples by Category'
        save_path: 图像保存路径，若为None则不保存

    Returns:
        None
    """
    # 输入验证
    if len(generated_samples) != len(generated_labels):
        raise ValueError(
            f"Samples and labels length mismatch: {len(generated_samples)} != {len(generated_labels)}"
        )
    # 获取唯一标签
    unique_labels = np.unique(generated_labels)
    nrow = len(unique_labels)  # 行数等于类别数
    if colors_map is None:
        colors = sns.color_palette('husl', n_colors=num_classes)
        colors_list = LinearSegmentedColormap.from_list('custom', colors)
        colors_map = {label: colors_list(i) for i, label in enumerate(unique_labels)}
    width = ncol*10
    height = nrow * 3
    plt.figure(figsize=(width, height))

    if title:
        plt.suptitle(title, fontsize=12, y=0.95)

    # 为每个类别绘制样本
    for i, label in enumerate(unique_labels):
        # 获取当前类别的所有样本
        indices = np.where(generated_labels == label)[0]
        samples = generated_samples[indices]
        color = colors_map[label]
        # 确定要展示的样本数量
        n_samples = min(len(samples), ncol)

        # 在当前行绘制样本
        for j in range(n_samples):
            plt.subplot(nrow, ncol, i * ncol + j + 1)
            plt.plot(samples[j], c=color, linewidth=1.5)

            if j == 0:  # 只在每行第一个子图显示类别标签
                plt.title(f'Category {label}', fontsize=10, pad=5)


    # 调整布局
    plt.tight_layout()

    # 保存或显示图像
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=500)
        plt.close()
    else:
        plt.show()


def visualize_clusters(
    z: np.ndarray,
    labels: np.ndarray,
    gmm_labels: np.ndarray,
    leiden_labels: np.ndarray,
    save_path: str,
    colors_map: Optional[Mapping[int, Any]] = None,
    random_state: int = 42,
    fig_size: Tuple[int, int] = (10, 8),
    title: Optional[str] = None
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
        title: plot title

    Returns:
        None
    """
    # 输入验证
    if len(z) != len(labels):
        raise ValueError(
            f"Features and labels length mismatch: {len(z)} != {len(labels)}"
        )

    if title is None:
        title = " "

    tsne = TSNE(n_components=2, random_state=random_state)
    z_tsne = tsne.fit_transform(z)

    ari_gmm = adjusted_rand_score(labels, gmm_labels)
    ari_leiden = adjusted_rand_score(labels, leiden_labels)

    # 绘图
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 8))
    
    # 创建大调色板
    colors = sns.color_palette('husl', n_colors=len(np.unique(labels)))
    custom_cmap = LinearSegmentedColormap.from_list('custom', colors)
    scatter1 = ax1.scatter(z_tsne[:, 0], z_tsne[:, 1], c=labels, cmap=custom_cmap)
    ax1.set_title('True Labels')
    legend1 = ax1.legend(*scatter1.legend_elements(), title="Classes", bbox_to_anchor=(1.05, 1), loc='best', fontsize='small')
    ax1.add_artist(legend1)
    
    # 绘制GMM预测聚类结果的散点图
    colors = sns.color_palette('husl', n_colors=len(np.unique(gmm_labels)))
    custom_cmap = LinearSegmentedColormap.from_list('custom', colors)
    scatter2 = ax2.scatter(z_tsne[:, 0], z_tsne[:, 1], c=gmm_labels, cmap=custom_cmap)
    ax2.set_title(f'GMM Predicted Clusters\nARI: {ari_gmm:.3f}')
    legend2 = ax2.legend(*scatter2.legend_elements(num=len(np.unique(gmm_labels))), title="Clusters", bbox_to_anchor=(1.05, 1), loc='best', fontsize='small')
    ax2.add_artist(legend2)

    # 绘制Leiden预测聚类结果的散点图
    colors = sns.color_palette('husl', n_colors=len(np.unique(leiden_labels)))
    custom_cmap = LinearSegmentedColormap.from_list('custom', colors)
    scatter3 = ax3.scatter(z_tsne[:, 0], z_tsne[:, 1], c=leiden_labels, cmap=custom_cmap)
    ax3.set_title(f'Leiden Predicted Clusters\nARI: {ari_leiden:.3f}')
    legend3 = ax3.legend(*scatter3.legend_elements(num=len(np.unique(leiden_labels))), title="Clusters", bbox_to_anchor=(1.05, 1), loc='best', fontsize='small')
    ax3.add_artist(legend3)
    
    # 保存图像
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
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
    stack_gap = float(np.mean(np.max(matched_S, axis=1)))  

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

def compute_cluster_means(spectra, clusters):
    """计算每个簇的平均光谱"""
    unique_clusters = np.unique(clusters)
    cluster_means = []
    
    for cluster_id in unique_clusters:
        cluster_mask = clusters == cluster_id
        mean_spectrum = np.mean(spectra[cluster_mask], axis=0)
        cluster_means.append(mean_spectrum)
        
    return np.array(cluster_means)

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