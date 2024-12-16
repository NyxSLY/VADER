import time
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import matplotlib
from sklearn.manifold import TSNE
from typing import Optional, Dict, Union, Any,Tuple
from torch.distributions import Normal
import random
from torch.utils.data import DataLoader, TensorDataset
import os
from scipy.optimize import linear_sum_assignment
import datetime, pywt
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

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
    device : str = None,
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

def compare_kmeans_methods(model, dataloader, num_classes, n_runs=5):
    """比较普通k-means和k-means++的性能

    Args:
        model: VaDE模型实例
        dataloader: 数据加载器
        num_classes: 聚类数量
        n_runs: 运行次数
    """
    # 获取编码数据
    all_encoded_data = []
    for data in dataloader:
        x, _ = data
        with torch.no_grad():
            mean, _ = model.encoder(x)
        all_encoded_data.append(mean)
    encoded_data = torch.cat(all_encoded_data, dim=0).cpu().numpy()

    # 存储结果
    results = {
        'kmeans': {'inertia': [], 'silhouette': [], 'time': []},
        'kmeans++': {'inertia': [], 'silhouette': [], 'time': []}
    }

    # 多次运行实验
    for i in range(n_runs):
        print(f"\nRun {i + 1}/{n_runs}")

        # 普通k-means
        start_time = time.time()
        kmeans = KMeans(
            n_clusters=num_classes,
            init='random',
            n_init=1,
            random_state=42 + i
        )
        kmeans.fit(encoded_data)
        km_time = time.time() - start_time

        results['kmeans']['inertia'].append(kmeans.inertia_)
        results['kmeans']['silhouette'].append(
            silhouette_score(encoded_data, kmeans.labels_))
        results['kmeans']['time'].append(km_time)

        # k-means++
        start_time = time.time()
        kmeans_plus = KMeans(
            n_clusters=num_classes,
            init='k-means++',
            n_init=1,
            random_state=42 + i
        )
        kmeans_plus.fit(encoded_data)
        kmpp_time = time.time() - start_time

        results['kmeans++']['inertia'].append(kmeans_plus.inertia_)
        results['kmeans++']['silhouette'].append(
            silhouette_score(encoded_data, kmeans_plus.labels_))
        results['kmeans++']['time'].append(kmpp_time)

        # 打印当前运行的结果
        print(f"K-means    - Inertia: {kmeans.inertia_:.2f}, "
              f"Silhouette: {results['kmeans']['silhouette'][-1]:.4f}, "
              f"Time: {km_time:.2f}s")
        print(f"K-means++  - Inertia: {kmeans_plus.inertia_:.2f}, "
              f"Silhouette: {results['kmeans++']['silhouette'][-1]:.4f}, "
              f"Time: {kmpp_time:.2f}s")

    # 打印统计结果
    print("\nAverage Results:")
    for method in ['kmeans', 'kmeans++']:
        print(f"\n{method}:")
        print(f"Inertia: {np.mean(results[method]['inertia']):.2f} "
              f"± {np.std(results[method]['inertia']):.2f}")
        print(f"Silhouette: {np.mean(results[method]['silhouette']):.4f} "
              f"± {np.std(results[method]['silhouette']):.4f}")
        print(f"Time: {np.mean(results[method]['time']):.2f} "
              f"± {np.std(results[method]['time']):.2f}s")

    return results

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

def add_noise_to_signal(signal, noise_level=0.05):
    noise = np.random.randn(1,signal.shape[0])*noise_level
    return signal + noise

def shift_signal(signal):
    shift=random.choices([-3, 0, 3], weights=[0.1,8,0.1])[0]
    return np.roll(signal, shift)

def smooth_edges(signal, smooth_width=5):
    smoothed_data = signal.copy()
    # 平滑开头
    for i in range(smooth_width):
        smoothed_data[:,i] = np.mean(smoothed_data[:,:i+smooth_width],axis = 1)
    # 平滑结尾
    for i in range(smooth_width):
        smoothed_data[:,-i-1] = np.mean(smoothed_data[:,-smooth_width+i:],axis = 1)

    return smoothed_data

def scale_signal(signal, scale_range=(0.9, 1.1)):
    scale_factor = np.random.uniform(*scale_range)
    return signal * scale_factor

def perturb_signal(signal, perturb_level=0.01):
    perturbation = np.random.normal(0, perturb_level, signal.shape[0])
    return signal + perturbation

def interpolate_z(z1, z2, steps=10, mode='extrapolation'):
    interpolations = []
    arr = np.random.rand(steps)
    for alpha in arr:
        if mode == 'linear':
            z_interp = z1 * (1 - alpha) + z2 * alpha
        elif mode == 'extrapolation':
            z_interp = z1 + alpha * (z2 - z1)
        interpolations.append(z_interp)
    return interpolations

def feature_swap_z(z1,z2,num=5):
    random_integers = np.random.randint(0,z1.shape[0],size=(5))
    swap = torch.zeros(z1.size(), dtype=torch.bool)
    swap[random_integers] =True
    z_swap = z1 * (~swap) + z2 * swap
    return z_swap

def generate_spectra_from_means(means,model, num_samples_per_label=100, noise_level=0.01,num=3):
    model.eval()
    generated_samples = []
    generated_labels = []
    z_samples = []
    mean_labels = means.shape[0]
    
    for label in range(mean_labels):
        mean = means[label].unsqueeze(0)
        latent_samples = []
        
        # Step 1: Add noise to means
        for _ in range(num_samples_per_label):
            noisy_mean = add_noise_to_signal(mean, noise_level=noise_level)
            #shifted_mean = shift_signal(noisy_mean)
            scaled_mean = scale_signal(noisy_mean)
            latent_samples.append(scaled_mean)
        
        latent_samples = np.array(latent_samples)
        latent_samples_tensor = torch.from_numpy(latent_samples).float()

        # Step 3: Interpolation and Step 4: Feature Swap
        num_generated = 0
        while num_generated < num_samples_per_label:
            for i in range(0, len(latent_samples_tensor) - 1, 2):
                z1 = latent_samples_tensor[i]
                z2 = latent_samples_tensor[num_samples_per_label-1-i]
                
                # Interpolation
                interpolations = interpolate_z(z1, z2, steps=1)
                for z_interp in interpolations:
                    # Feature Swap
                    swaps = feature_swap_z(z1, z_interp,num=3)
                    for z_swap in swaps:
                        z_samples.append(z_swap)
                        x_spec = model.decoder(z_swap.unsqueeze(0))
                        x_spec_np = x_spec.detach().numpy()
                        x_spec_np_shift = shift_signal(x_spec_np)
                        x_spec_np_smooth = smooth_edges(x_spec_np_shift)
                        generated_samples.append(x_spec_np_smooth)
                        generated_labels.append(label)
                        num_generated += 1
                        if num_generated >= num_samples_per_label:
                            break
                if num_generated >= num_samples_per_label:
                    break
            if num_generated >= num_samples_per_label:
                break

    generated_samples = np.vstack(generated_samples)
    generated_labels = np.array(generated_labels)
    z_samples = np.vstack(z_samples)
    
    return generated_samples, generated_labels

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
        colors_list = matplotlib.colormaps.get_cmap('tab10')
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
    gmm_labels:np.ndarray,
    leiden_labels:np.ndarray,
    save_path: str,
    colors_map: Optional[Dict[int, str]] = None,
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
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 8))
    
    # 绘制真实标签的散点图
    scatter1 = ax1.scatter(z_tsne[:, 0], z_tsne[:, 1], c=labels, cmap=colors_map)
    ax1.set_title('True Labels')
    legend1 = ax1.legend(*scatter1.legend_elements(), title="Classes")
    ax1.add_artist(legend1)
    
    # 绘制GMM预测聚类结果的散点图
    scatter2 = ax2.scatter(z_tsne[:, 0], z_tsne[:, 1], c=gmm_labels, cmap='tab20')
    ax2.set_title(f'GMM Predicted Clusters\nARI: {ari_gmm:.3f}')
    legend2 = ax2.legend(*scatter2.legend_elements(), title="Classes")
    ax2.add_artist(legend2)

    # 绘制Leiden预测聚类结果的散点图
    scatter3 = ax3.scatter(z_tsne[:, 0], z_tsne[:, 1], c=leiden_labels, cmap='tab20')
    ax3.set_title(f'Leiden Predicted Clusters\nARI: {ari_leiden:.3f}')
    legend3 = ax3.legend(*scatter3.legend_elements(), title="Classes")
    ax3.add_artist(legend3)
    
    # 保存图像
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=500)
    plt.close()


def visualize_clusters1(
    z: np.ndarray,
    labels: np.ndarray,
    save_path: str,
    colors_map: Optional[Dict[int, str]] = None,
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

    unique_labels = np.unique(labels)
    if colors_map is None:
        colors_list = matplotlib.colormaps.get_cmap('tab10')
        colors_map = {label: colors_list(i) for i, label in enumerate(unique_labels)}

    plt.figure(figsize=fig_size)
    for label in unique_labels:
        mask = (labels == label)
        plt.scatter(
            z_tsne[mask, 0],
            z_tsne[mask, 1],
            c=[colors_map[label]],
            label=f'Cluster {label}',
        )

    plt.title(title)
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend(bbox_to_anchor=(1.08, 0.5), loc='center')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=500)
    plt.close()

def plot_reconstruction(
        recon_data: np.ndarray,
        labels: np.ndarray,
        save_path: str,
        wavenumber: Optional[np.ndarray] = None,
        colors_map: Optional[Dict[int, str]] = None
) -> None:
    """
    Args:
        recon_data: (n_samples, n_features)
        labels: sample labels
        save_path: save plot to path
        wavenumber: spec wavenumber
        colors_map: colors dic

    Returns:
        None
    """
    # 获取唯一标签及设置图形高度
    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)
    figure_height = 5 if num_classes <= 3 else 5 + 0.5 * num_classes

    if colors_map is None:
        colors_list = matplotlib.colormaps.get_cmap('tab10')
        colors_map = {label: colors_list(i) for i, label in enumerate(unique_labels)}
    # 设置波长范围
    if wavenumber is None:
        wavenumber = np.arange(recon_data.shape[1])

    # 创建图形
    plt.figure(figsize=(15, figure_height))

    # 绘制每个类别的谱线
    for i, label in enumerate(unique_labels):
        # print(i,label)
        indices = np.where(labels == label)[0]
        dis = -1 * i

        # 获取当前类别的颜色
        color = colors_map[label] if colors_map else None

        # 绘制当前类别的所有谱线
        for j, idx in enumerate(indices):
            spectrum = recon_data[idx] + dis
            if j == 0:
                plt.plot(wavenumber, spectrum,
                         label=f"Cluster{label}",
                         c=color)
            else:
                plt.plot(wavenumber, spectrum,
                         c=color)

    # 设置图形属性
    plt.title('Reconstruction Spectrum of Biological Cells')
    plt.legend(prop={'size': 9}, bbox_to_anchor=(1.04, 0.5), loc='center')
    plt.xlabel('Wavelength')
    plt.ylabel('Intensity')
    plt.tight_layout()

    # 保存并关闭图形
    plt.savefig(save_path,bbox_inches='tight', dpi=500)
    plt.close()

def leiden_clustering(spectra, n_neighbors=15, resolution=1.0, seed=42):
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


def wavelet_transform(data, wavelet='db4', level=3):
    """
    对数据进行小波变换
    
    参数:
    data: numpy array, 形状为 (n_samples, n_features) 的输入数据
    wavelet: str, 小波基函数类型，默认使用'db4'
    level: int, 分解层数
    
    返回:
    transformed_data: 小波变换后的数据
    """
    n_samples = data.shape[0]
    transformed_data = []
    
    for i in range(n_samples):
        # 对每个样本进行小波分解
        coeffs = pywt.wavedec(data[i], wavelet, level=level)
        # 将所有系数连接成一个向量
        transformed = np.concatenate([coeffs[0]] + [c for c in coeffs[1:]])
        transformed_data.append(transformed)
    
    return np.array(transformed_data)


def inverse_wavelet_transform(transformed_data, original_length, wavelet='db4', level=3):
    """
    进行小波逆变换
    
    参数:
    transformed_data: 小波变换后的数据
    original_length: 原始信号长度
    wavelet: str, 小波基函数类型
    level: int, 分解层数
    
    返回:
    reconstructed_data: 重构后的数据
    """
    n_samples = transformed_data.shape[0]
    reconstructed_data = []
    
    # 计算每层的系数长度
    coeffs_length = []
    dummy_data = np.zeros(original_length)
    dummy_coeffs = pywt.wavedec(dummy_data, wavelet, level=level)
    for coeff in dummy_coeffs:
        coeffs_length.append(len(coeff))
    
    for i in range(n_samples):
        # 分离系数
        pos = 0
        coeffs = []
        for length in coeffs_length:
            coeffs.append(transformed_data[i, pos:pos+length])
            pos += length
        
        # 重构信号
        reconstructed = pywt.waverec(coeffs, wavelet)
        reconstructed_data.append(reconstructed)
    
    return np.array(reconstructed_data)