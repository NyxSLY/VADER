import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
import numpy as np
from scipy import signal,sparse
import os
import torch.cuda
from utility import leiden_clustering, compute_cluster_means
import time
from collections import defaultdict
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from community import community_louvain
from utility import leiden_clustering
import math
from tqdm import tqdm
import itertools
from scipy.optimize import linear_sum_assignment


class Encoder(nn.Module):
    def __init__(self, input_dim, intermediate_dim, latent_dim, n_components, S=None):
        """
        Args:
            input_dim: 输入维度
            intermediate_dim: 中间维度
            latent_dim: 潜在空间维度
            n_components: MCR成分数
            S: MCR成分光谱矩阵 [n_components, input_dim]
        """
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_components = n_components

        # 注册MCR成分光谱矩阵S
        if S is not None:
            self.S = nn.Parameter(torch.tensor(S, dtype=torch.float32))
        else:
            # 如果没有提供S，随机初始化
            self.S = nn.Parameter(torch.randn(n_components, n_components, dtype=torch.float32))

        layers = []
        prev_dim = input_dim
        for dim in intermediate_dim:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
            ])
            prev_dim = dim

        self.net = nn.Sequential(*layers)

        # 修改输出层，输出浓度相关参数
        self.to_concentration = nn.Linear(intermediate_dim[-1], latent_dim)  # 浓度均值
        # self.to_concentration_logvar = nn.Linear(intermediate_dim[-1], latent_dim)  # 浓度方差, (c+σ)S
        self.to_concentration_logvar = nn.Linear(intermediate_dim[-1], n_components)  # 分解损失, cS+σ

    def forward(self, x):
        x = self.net(x)
        concentration = self.to_concentration(x)  # 浓度均值
        concentration_logvar = self.to_concentration_logvar(x)  # 浓度方差
        return concentration, concentration_logvar, self.S  


class SpectralConstraints(nn.Module):
    def __init__(self, method='poly', poly_degree=3, window_size=50):
        super().__init__()
        self.method = method
        self.poly_degree = poly_degree
        self.window_size = window_size
        self.dummy = nn.Parameter(torch.zeros(1))
        
    @torch.no_grad()
    def batch_als_baseline(self, x):
        if self.method == 'poly':
            return self._poly_baseline(x)
        # ... 其他方法保持不变 ...

    def _poly_baseline(self, x):
        """多项式拟合基线
        使用批量处理的多项式拟合
        """
        device = x.device
        batch_size, N = x.shape
        
        # 创建X矩阵 (多项式的次数项)
        x_range = torch.linspace(0, 1, N, device=device)
        X = torch.stack([x_range ** i for i in range(self.poly_degree + 1)]).T  # [N, degree+1]
        
        # 批量最小二乘拟合
        try:
            # 使用批量矩阵运算
            # X: [N, degree+1], x: [batch_size, N]
            # 计算 (X^T X)^(-1) X^T
            Xt = X.T
            XtX = torch.mm(Xt, X)
            XtX_inv = torch.linalg.inv(XtX + torch.eye(XtX.shape[0], device=device) * 1e-10)
            proj_matrix = torch.mm(XtX_inv, Xt)  # [(degree+1), N]
            
            # 批量计算系数
            coeffs = torch.mm(proj_matrix, x.T).T  # [batch_size, degree+1]
            
            # 计算拟合的基线
            baselines = torch.mm(coeffs, X.T)  # [batch_size, N]
            
        except Exception as e:
            print(f"Warning: Error in polynomial fitting: {str(e)}")
            # 如果失败，返回简单的均值基线
            baselines = torch.mean(x, dim=1, keepdim=True).expand(-1, N)
        
        return baselines

    def _adaptive_poly_baseline(self, x):
        """自适应多项式拟合基线
        将光谱分段处理，每段使用不同阶数的多项式
        """
        device = x.device
        batch_size, N = x.shape
        segment_size = min(self.window_size, N)
        num_segments = (N + segment_size - 1) // segment_size
        
        baselines = torch.zeros_like(x)
        
        for i in range(num_segments):
            start_idx = i * segment_size
            end_idx = min((i + 1) * segment_size, N)
            segment = x[:, start_idx:end_idx]
            
            # 对每段使用较低阶多项式
            segment_baseline = self._poly_baseline_segment(segment, degree=min(2, self.poly_degree))
            baselines[:, start_idx:end_idx] = segment_baseline
        
        # 平滑段之间的过渡
        if num_segments > 1:
            baselines = self._smooth_transitions(baselines, segment_size)
        
        return baselines
    
    def _poly_baseline_segment(self, x, degree):
        """对单个段进行多项式拟合"""
        device = x.device
        batch_size, seg_length = x.shape
        
        x_range = torch.linspace(0, 1, seg_length, device=device)
        X = torch.stack([x_range ** i for i in range(degree + 1)]).T
        
        try:
            Xt = X.T
            XtX = torch.mm(Xt, X)
            XtX_inv = torch.linalg.inv(XtX + torch.eye(XtX.shape[0], device=device) * 1e-10)
            proj_matrix = torch.mm(XtX_inv, Xt)
            coeffs = torch.mm(proj_matrix, x.T).T
            segment_baseline = torch.mm(coeffs, X.T)
            
        except Exception as e:
            segment_baseline = torch.mean(x, dim=1, keepdim=True).expand(-1, seg_length)
        
        return segment_baseline
    
    def _smooth_transitions(self, baselines, segment_size):
        """平滑段之间的过渡"""
        device = baselines.device
        batch_size, N = baselines.shape
        
        # 使用简单的移动平均来平滑过渡区域
        window_size = min(segment_size // 4, 20)  # 过渡区域的大小
        if window_size > 1:
            kernel = torch.ones(1, 1, window_size, device=device) / window_size
            padded = F.pad(baselines.unsqueeze(1), (window_size//2, window_size//2), mode='reflect')
            smoothed = F.conv1d(padded, kernel).squeeze(1)
            
            # 只在段的边界附近应用平滑
            mask = torch.ones_like(baselines)
            for i in range(1, (N + segment_size - 1) // segment_size):
                idx = i * segment_size
                if idx < N:
                    start_idx = max(0, idx - window_size//2)
                    end_idx = min(N, idx + window_size//2)
                    mask[:, start_idx:end_idx] = 0
            
            baselines = mask * baselines + (1 - mask) * smoothed
            
        return baselines

    def forward(self, x, recon_x):
        """计算光谱约束"""
        # 计算基线
        x_baseline = self.batch_als_baseline(x)
        recon_baseline = self.batch_als_baseline(recon_x)
        
        # 计算约束
        baseline_loss = F.mse_loss(recon_baseline, x_baseline)
        return baseline_loss

class PeakDetector(nn.Module):
    def __init__(self, height_factor=0.1, min_distance=10, prominence_factor=0.05):
        super().__init__()
        self.height_factor = height_factor
        self.min_distance = min_distance
        self.prominence_factor = prominence_factor
        # 添加一个虚拟参数以确保设备一致性
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """直接在 GPU 上计算峰值权重"""
        device = x.device
        batch_size, N = x.shape
        peaks = torch.zeros_like(x)  # 这样会自动在正确的设备上创建

        # 归一化到 [0, 1]
        x_normalized = (x - x.min(dim=-1, keepdim=True)[0]) / (
            x.max(dim=-1, keepdim=True)[0] - x.min(dim=-1, keepdim=True)[0] + 1e-5
        )

        # 批量检测峰值
        for i in range(batch_size):
            # 添加 detach() 来处理需要梯度的张量
            spectrum = x_normalized[i].detach().cpu().numpy()
            peak_indices, _ = signal.find_peaks(
                spectrum,
                height=self.height_factor,
                distance=self.min_distance,
                prominence=self.prominence_factor,
            )
            # 直接在正确的设备上索引
            peaks[i, peak_indices] = 1.0

        return peaks  # peaks已经在正确的设备上

class Decoder(nn.Module):
    def __init__(self, latent_dim,intermediate_dim, input_dim):
        super(Decoder, self).__init__()
        

        decoder_dims = intermediate_dim[::-1]

        layers = []
        prev_dim = latent_dim
        for dim in decoder_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),   
                # nn.BatchNorm1d(dim),
                nn.ReLU()
            ])
            prev_dim = dim
        layers.extend([
            nn.Linear(prev_dim, input_dim),
            # nn.BatchNorm1d(input_dim),
            nn.Sigmoid()
        ])
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)

class Gaussian(nn.Module):
    def __init__(self, num_clusters, latent_dim):
        super(Gaussian, self).__init__()
        self.num_clusters = num_clusters
        self.latent_dim = latent_dim
        
        # 初始化参数
        self.pi = nn.Parameter(torch.ones(num_clusters) / num_clusters)
        self.means = nn.Parameter(torch.zeros(num_clusters, latent_dim))
        self.log_variances = nn.Parameter(torch.zeros(num_clusters, latent_dim))

        
    def update_parameters(self, cluster_centers=None, variances=None, weights=None):
        """更新GMM参数"""
        print('Update gmm parameters..............\n')
        with torch.no_grad():
            if cluster_centers is not None:
                self.means.data.copy_(cluster_centers)
            if variances is not None:
                self.log_variances.data.copy_(torch.log(variances + 1e-10))
            if weights is not None:
                self.pi.data.copy_(weights)

    def forward(self, z):
        # 计算条件概率/可能性
        y = self.gaussian_log_prob(z)  # log p(z|c)
        
        # 计算后验概率
        gamma = F.softmax(y, dim=1)  # p(c|z)
        
        # 计算条件均值
        means = torch.sum(gamma.unsqueeze(2) * self.means.unsqueeze(0), dim=1)
        
        # 返回所有需要的值
        return self.means,self.log_variances, y, gamma, self.pi  # 输出PI

    def gaussian_log_prob(self, z):
        """计算log p(z|c)"""
        z_expanded = z.unsqueeze(1)  # [batch_size, 1, latent_dim]
        means_expanded = self.means.unsqueeze(0)  # [1, num_clusters, latent_dim]
        log_vars_expanded = self.log_variances.unsqueeze(0)  # [1, num_clusters, latent_dim]
        pi_expanded = self.pi.unsqueeze(0)  # [1, num_clusters]
    
        log_p_c = (
            torch.log(pi_expanded) * self.latent_dim                   # 混合权重项
            - 0.5 * torch.sum(torch.log(2*math.pi*torch.exp(log_vars_expanded)), dim=2)  # 常数项和方差项
            - torch.sum((z_expanded - means_expanded).pow(2)/(2*torch.exp(log_vars_expanded)), dim=2)  # 指数项
        )
        return log_p_c


class SpectralAnalyzer:
    def __init__(self, peak_detector, spectral_constraints):
        self.peak_detector = peak_detector
        self.spectral_constraints = spectral_constraints
        self.stats = {}  # 用于存储分析结果
        
    def get_dataset_stats(self):
        """获取数据集统计信息"""
        return self.stats
        
    @torch.no_grad()
    def analyze_dataset(self, dataloader):
        """分析数据集的光谱特征"""
        print("开始分析数据集...")
        
        # 1. 收集所有数据
        print("正在收集光谱数据...")
        all_spectra = []
        total_samples = 0
        for x, _ in dataloader:
            all_spectra.append(x.cpu().numpy())
            total_samples += x.shape[0]
        all_spectra = np.vstack(all_spectra)
        print(f"共收集到 {total_samples} 个光谱样本")
        
        # 2. 计算基本统计信息
        self.stats = {
            'mean_spectrum': torch.tensor(np.mean(all_spectra, axis=0)),
            'std_spectrum': torch.tensor(np.std(all_spectra, axis=0)),
            'intensity_range': {
                'min': float(np.min(all_spectra)),
                'max': float(np.max(all_spectra))
            }
        }
        
        # 3. 分析峰位置
        print("正在分峰位置...")
        peaks = self.peak_detector(torch.tensor(all_spectra))
        # 记录在至少10%光谱中出现过峰的位置
        peak_counts = torch.sum(peaks, dim=0)
        min_occurrences = 0.1 * peaks.shape[0]  # 10%的样本数
        self.stats['peak_positions'] = torch.where(peak_counts >= min_occurrences)[0]
        
        # 4. 分析基线特zheng
        print("正在分析基线特征...")
        baselines = self.spectral_constraints.batch_als_baseline(torch.tensor(all_spectra))
        self.stats['baseline_params'] = {
            'mean_baseline': torch.mean(baselines, dim=0),
            'std_baseline': torch.std(baselines, dim=0)
        }
        
        print("数据集分析完成!")
        return self.stats

    def save_analysis_results(self, save_dir='./spectral_analysis'):
        """保存分析结果"""
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'spectral_analysis.pt')
        torch.save(self.stats, save_path)
        print(f"分析结果已保存到: {save_path}")

    def load_analysis_results(self, save_dir='./spectral_analysis'):
        """加载分析结果"""
        load_path = os.path.join(save_dir, 'spectral_analysis.pt')
        if os.path.exists(load_path):
            self.stats = torch.load(load_path)
            print(f"已加载分析结果从: {load_path}")
            return True
        return False


class VaDE(nn.Module):
    def __init__(self, input_dim, intermediate_dim, latent_dim,  device, l_c_dim, n_components,
                 encoder_type="basic", batch_size=None, tensor_gpu_data=None,
                 lamb1=1.0, lamb2=1.0, lamb3=1.0, lamb4=1.0, lamb5=1.0, lamb6=1.0, lamb7=1.0, 
                 cluster_separation_method='cosine',
                 pretrain_epochs=50,
                 num_classes=0, resolution_1=1.0, resolution_2=0.9, clustering_method='leiden'):
        super(VaDE, self).__init__()
        self.device = device
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.tensor_gpu_data = tensor_gpu_data
        self.n_components = n_components
        self.encoder = self._init_encoder(input_dim, intermediate_dim, latent_dim, encoder_type, l_c_dim, n_components)
        self.decoder = Decoder(latent_dim, intermediate_dim, input_dim)
        self.gaussian = Gaussian(num_classes, latent_dim)
        self.cluster_centers = None
        self.lamb1 = lamb1
        self.lamb2 = lamb2
        self.lamb3 = lamb3
        self.lamb4 = lamb4
        self.lamb5 = lamb5
        self.lamb6 = lamb6
        self.lamb7 = lamb7
        self.cluster_separation_method = cluster_separation_method
        self.pretrain_epochs = pretrain_epochs
        self.num_classes = num_classes
        self.resolution_1 = resolution_1
        self.resolution_2 = resolution_2
        self.clustering_method = clustering_method
        self.input_dim = input_dim

        self.peak_detector = PeakDetector().to(device)
        self.spectral_constraints = SpectralConstraints(
            method='poly',  # 使用多项式拟合
            poly_degree=3,  # 多项式阶数
            window_size=200  # 分段大小
        ).to(device)
        self.spectral_analyzer = SpectralAnalyzer(self.peak_detector, self.spectral_constraints)
        
        # 初始化光谱统计数据
        self._init_spectral_stats()
        
        # 确保所有组件都在正确的设备上
        self.to(device)

    def _init_spectral_stats(self):
        """初始化光谱统计数据并确保在正确的设备上"""
        self.spectral_stats = {
            'peak_features': {
                'peak_count': torch.tensor(0, device=self.device),
                'peak_heights': torch.tensor([], device=self.device),
                'peak_positions': torch.tensor([], device=self.device),
                'peak_prominences': torch.tensor([], device=self.device)
            },
            'baseline_features': {
                'baseline_mean': torch.tensor(0.5, device=self.device),
                'baseline_std': torch.tensor(0.1, device=self.device),
                'baseline_slope': torch.tensor(0.0, device=self.device)
            },
            'spectral_features': {
                'total_intensity': torch.tensor(0.0, device=self.device),
                'max_intensity': torch.tensor(1.0, device=self.device),
                'min_intensity': torch.tensor(0.0, device=self.device),
                'mean_intensity': torch.tensor(0.5, device=self.device),
                'std_intensity': torch.tensor(0.1, device=self.device)
            }
        }
        # 更新SpectralAnalyzer的统计数据
        self.spectral_analyzer.stats = self.spectral_stats


    @staticmethod
    def _init_encoder(input_dim, intermediate_dim,latent_dim, encoder_type,l_c_dim, n_components):
        """初始化编码器"""
        if encoder_type == "basic":
            return Encoder(input_dim, intermediate_dim=intermediate_dim, latent_dim=latent_dim, 
                         n_components=n_components)
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

    def pretrain(self, dataloader,learning_rate=1e-3):
        pre_epoch=self.pretrain_epochs
        if  not os.path.exists('./nc9_pretrain_model_none_bn_.pk'):

            Loss=nn.MSELoss()
            opti=torch.optim.Adam(itertools.chain(self.encoder.parameters(),self.decoder.parameters()))

            print('Pretraining......')
            epoch_bar=tqdm(range(pre_epoch))
            for _ in epoch_bar:
                L=0
                for x,y in dataloader:
                    x=x.to(self.device)

                    mean,var,S = self.encoder(x)
                    z = self.reparameterize(mean, var, S)
                    x_=self.decoder(z)
                    loss=Loss(x,x_)

                    L+=loss.detach().cpu().numpy()

                    opti.zero_grad()
                    loss.backward()
                    opti.step()

                epoch_bar.write('L2={:.4f}'.format(L/len(dataloader)))

            # self.encoder.to_logvar.load_state_dict(self.encoder.to_mean.state_dict())

            # Z = []
            # Y = []
            # with torch.no_grad():
            #     for x, y in dataloader:
            #         if self.args.cuda:
            #             x = x.cuda()

            #         z1, z2 = self.encoder(x)
            #         assert F.mse_loss(z1, z2) == 0
            #         Z.append(z1)
            #         Y.append(y)

            # Z = torch.cat(Z, 0).detach().cpu().numpy()
            # Y = torch.cat([yy.cpu() for yy in Y], 0).detach().numpy()

            # gmm = GaussianMixture(n_components=self.args.nClusters, covariance_type='diag')

            # pre = gmm.fit_predict(Z)
            # print('Acc={:.4f}%'.format(cluster_acc(pre, Y)[0] * 100))

            # self.pi_.data = torch.from_numpy(gmm.weights_).cuda().float()
            # self.mu_c.data = torch.from_numpy(gmm.means_).cuda().float()
            # self.log_sigma2_c.data = torch.log(torch.from_numpy(gmm.covariances_).cuda().float())

            torch.save(self.state_dict(), './pretrain_model_50.pk')

        else:
            self.load_state_dict(torch.load('./nc9_pretrain_model_none_bn.pk'))
            


    def _apply_clustering(self, encoded_data):
        """应用选定的聚类方法"""
        print(f"\nClustering method: {self.clustering_method}")
        
        if self.clustering_method == 'kmeans':
            # K-means方法
            kmeans = KMeans(n_clusters=self.num_classes, random_state=0)
            labels = kmeans.fit_predict(encoded_data)
            cluster_centers = kmeans.cluster_centers_
            
        elif self.clustering_method == 'louvain':
            # Louvain方法
            # 构建 KNN 图
            nn = NearestNeighbors(n_neighbors=10)
            nn.fit(encoded_data)
            adj_matrix = nn.kneighbors_graph(encoded_data, mode='distance')
            
            G = nx.from_scipy_sparse_matrix(adj_matrix)
            partition = community_louvain.best_partition(G, resolution=self.resolution_1)
            labels = np.array(list(partition.values()))
            
            # 计算聚类中心
            unique_labels = np.unique(labels)
            cluster_centers = np.zeros((len(unique_labels), encoded_data.shape[1]))
            for i, label in enumerate(unique_labels):
                mask = labels == label
                cluster_centers[i] = encoded_data[mask].mean(axis=0)
        
        elif self.clustering_method == 'leiden':
            # Leiden方法
            labels = leiden_clustering(encoded_data,  resolution=self.resolution_1)
            
            # 计算聚类中心
            unique_labels = np.unique(labels)
            cluster_centers = np.zeros((len(unique_labels), encoded_data.shape[1]))
            for i, label in enumerate(unique_labels):
                mask = labels == label
                cluster_centers[i] = encoded_data[mask].mean(axis=0)
        
        else:
            raise ValueError(f"Unsupported clustering method: {self.clustering_method}")
        
        return labels, cluster_centers

    @torch.no_grad()
    def init_kmeans_centers(self, dataloader):
        encoded_data = []
        mean, var, S = self.encoder(self.tensor_gpu_data)
        encoded_data.append(mean.cpu())
        encoded_data = torch.cat(encoded_data, dim=0).numpy()

        # 使用选定的聚类方法
        print(f"Using clustering method: {self.clustering_method}")
        print(f'num_clusters: {self.num_classes}')
        labels, cluster_centers = self._apply_clustering(encoded_data)
        num_clusters = len(np.unique(labels))
    
        cluster_centers = torch.tensor(cluster_centers, device=self.device)
        # 如果聚类数量发生变化，需要重新初始化高斯分布
        if num_clusters != self.gaussian.num_clusters:
            print(f"Number of clusters changed from {self.gaussian.num_clusters} to {num_clusters}. Reinitializing Gaussian.")
            self.num_clusters = num_clusters
            self.gaussian = Gaussian(num_clusters, self.latent_dim).to(self.device)

        # 直接用聚类中心更新高斯分布参数
        self.gaussian.means.data.copy_(cluster_centers)
        self.cluster_centers = cluster_centers
    
    def optimal_transport(self,A, B, alpha):
        n, d = A.shape
        m, _ = B.shape

        # 计算成本矩阵
        cost_matrix = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)  # 解指派问题
        # print("row_ind:", row_ind) #A [0,1]
        # print("col_ind:", col_ind) #B [2,1]

        # 用匹配的值替换对应位置
        if n>= m:
            aligned_B = np.zeros_like(A)
            B_new = np.zeros((n,d))
            B_new[:m,:]=B
            for i, j in zip(row_ind, col_ind):
                aligned_B[j] = A[i]  # Match B[j] to A[i]
            
            unmatched_A_indices = set(range(n)) - set(row_ind)
            for idx in unmatched_A_indices:
                r = aligned_B.shape[0]
                aligned_B[r-1] = A[idx]  # Use unmatched A directly
                B_new[r-1] = A[idx]
            aligned_B = alpha * aligned_B + (1 - alpha) * B_new
        # Handle unmatched B (truncate if needed)
        elif n < m:
            aligned_B = np.zeros_like(B)
            for i, j in zip(row_ind, col_ind):
                aligned_B[j] = A[i]  # Match B[j] to A[i]

            unmatched_B_indices = set(range(m)) - set(col_ind)
            aligned_B = np.delete(aligned_B, list(unmatched_B_indices), axis=0)
            B = np.delete(B, list(unmatched_B_indices), axis=0)
            aligned_B = alpha * aligned_B + (1 - alpha) * B

        return aligned_B
    
    def change_var(self,array,n,w=1):
        if array.shape[0]>=n:
            array = array[:n]
        else:
            mean_value = np.mean(array, axis=0)*w
            padding = np.tile(mean_value, (n - array.shape[0], 1))
            array = np.vstack((array, padding))
        return array

    def  change_pi(self,array,n,w=1):

        if array.shape[0]>=n:
            array = array[:n]
        else:
            mean_value = np.mean(array)*w
            padding = np.full(n - array.shape[0], mean_value)
            array = np.concatenate((array, padding))
        return array



    @torch.no_grad()
    def update_kmeans_centers(self):
        print(f'Update clustering centers..........')
        encoded_data, _ = self.encoder(self.tensor_gpu_data)
        encoded_data_cpu = encoded_data.cpu().numpy()

        # update the clustering centers
        ml_labels, cluster_centers = self._apply_clustering(encoded_data_cpu)
        # leiden_centers = torch.tensor(leiden_centers, device=self.device).to(dtype=self.gaussian.means.data.dtype)
        # gaussian_centers = updated_centers = self.gaussian.means.data
        # gaussian_vars = updated_vars = self.gaussian.log_variances.data
        # gaussian_pi = updated_pi = self.gaussian.pi.data
        
        num_ml_centers = len(np.unique(ml_labels))

        """align"""
        if num_ml_centers != self.gaussian.num_clusters:
            print(f"Numberof clusters changed from {self.gaussian.num_clusters} to {num_ml_centers} .Reinitializing Gaussian.")
            gaussian_means = self.gaussian.means.cpu().numpy()
            cluster_centers = self.optimal_transport(cluster_centers, gaussian_means, 0.1)
            self.num_clusters = num_ml_centers
            array_var = self.gaussian.log_variances.cpu().numpy()
            array_pi = self.gaussian.pi.cpu().numpy()
            aligned_gaussian_var = self.change_var(array_var,self.num_clusters,w=1.0)
            aligned_gaussian_pi = self.change_pi(array_pi,self.num_clusters,w=0.5)
            self.gaussian = Gaussian(num_ml_centers, self.latent_dim).to(self.device)
            cluster_var = torch.tensor(aligned_gaussian_var,device = self.device)
            cluster_pi = torch.tensor(aligned_gaussian_pi,device = self.device)
            self.gaussian.log_variances.data.copy_(cluster_var)
            self.gaussian.pi.data.copy_(cluster_pi)
        cluster_centers_t = torch.tensor(cluster_centers,device = self.device)
        self.gaussian.means.data.copy_(cluster_centers_t)
 

    def reparameterize(self, concentration, log_var,spectra):
        """
        重参数化过程，将浓度参数转换为潜在空间表示
        Args:
            concentration: 浓度均值 [batch_size, n_components]
            log_var: 浓度方差 [batch_size, n_components]
            spectra: 光谱矩阵 [n_components, input_dim]
        Returns:
            z: 潜在空间表示 [batch_size, latent_dim]
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        # (C+σ)*S
        # return torch.matmul(concentration + eps * std, spectra)
        # CS+σ
        return torch.matmul(concentration, spectra) + eps * std

    def forward(self,x):
        """模型前向传播"""
        mean, log_var, S = self.encoder(x)
        z = self.reparameterize(mean, log_var, S)
        
        # 通过GMM获取所有需要的值
        means, log_variances, y, gamma, pi = self.gaussian(z)
        
        # 解码
        recon_x = self.decoder(z)
        
        return recon_x, mean, log_var, z, gamma, pi, S


    def train_step(self, x):
        """单步训练
        Args:
            x: 输入数据
        """
        # 前向传播
        recon_x, mean, log_var, z, gamma, pi, S = self(x)
        
        # 计算损失
        loss = self.loss_function(recon_x, x, mean, log_var, z, gamma, pi)
        
        return loss

    def compute_spectral_constraints(self, x, recon_x):
        """计算光谱约束时只使用数据集统计信息"""
        constraints = {}
        stats = self.spectral_analyzer.get_dataset_stats()
        
        # 1. 使用统计信息中的峰位置
        peak_positions = stats['peak_positions']
        distance = torch.diff(peak_positions, prepend=peak_positions[:1], append=peak_positions[-1:])
        variances = torch.max(distance[:-1], distance[1:])

        weights = torch.zeros_like(x)
        # plt.plot(self.tensor_gpu_data.mean(0).cpu().numpy(),linestyle='-')
        for i, peak in enumerate(peak_positions):
            gamma = variances[i]  # 洛伦兹分布的半宽
            lorentzian = gamma**2 / ((torch.arange(x.shape[1], device=x.device) - peak)**2 + gamma**2)
            # plt.plot( lorentzian.cpu().numpy(),  color=random_color(), linestyle='-')
            weights += lorentzian

        weightd_mse = F.mse_loss(recon_x, x, reduction='none') * weights

        return weightd_mse

        # ----------------- Before -----------------
        # if peak_positions is not None:
        #     peak_positions = peak_positions.to(x.device)
        #     constraints['peak'] = F.mse_loss(
        #         recon_x[:, peak_positions], 
        #         x[:, peak_positions], reduction='none'
        #     ).sum(-1)
        
        # # 2. 基线损失计算保持不变
        # mean_baseline = stats['baseline_params']['mean_baseline'].to(x.device)
        # baseline_std = stats['baseline_params']['std_baseline'].to(x.device)
        # x_baseline = self.spectral_constraints.batch_als_baseline(x)
        # recon_baseline = self.spectral_constraints.batch_als_baseline(recon_x)
        # baseline_diff = (recon_baseline - x_baseline) / (baseline_std + 1e-6)
        # constraints['baseline'] = (baseline_diff ** 2).mean(-1)
        
        # # 3. 强度范围约束
        # intensity_range = stats['intensity_range']
        # min_val = torch.tensor(intensity_range['min'], device=x.device)
        # max_val = torch.tensor(intensity_range['max'], device=x.device)
        # range_loss = (
        #     F.relu(min_val - recon_x).sum(-1) + 
        #     F.relu(recon_x - max_val).sum(-1)
        # )
        # constraints['range'] = range_loss
        
        # return constraints

    def compute_loss(self, x, recon_x, mean, log_var, z_prior_mean, y, S):

        """计算所有损失，严格按照原始VaDE论文"""
        batch_size = x.size(0)
        
        # 1. 重构损失
      
        recon_loss = self.lamb1 * F.mse_loss(recon_x, x, reduction='none').sum(-1)
       
        # 2. 从y计算gamma
        gamma = F.softmax(y, dim=-1)  # [batch_size, num_clusters]
        
        # 扩展gamma的维度用于广播
        gamma_t = gamma.unsqueeze(-1)  # [batch_size, num_clusters, 1]
        
        # 调整其他张量的维度
        mean = mean.unsqueeze(1)  # [batch_size, 1, latent_dim]
        log_var = log_var.unsqueeze(1)  # [batch_size, 1, latent_dim]
        
        # 调整高斯分布参数的维度
        gaussian_means = self.gaussian.means.unsqueeze(0)  # [1, n_clusters, latent_dim]
        gaussian_log_vars = self.gaussian.log_variances.unsqueeze(0)  # [1, n_clusters, latent_dim]
        
        # 3. GMM先验的KL散度
        try:
            kl_gmm = torch.sum(
                0.5 * gamma_t * (
                    self.latent_dim * math.log(2*math.pi) +
                    torch.log(torch.exp(gaussian_log_vars) + 1e-10) +
                    torch.exp(log_var) / (torch.exp(gaussian_log_vars) + 1e-10) +
                    (torch.matmul(mean, S) - gaussian_means).pow(2) / (torch.exp(gaussian_log_vars) + 1e-10)
                ),
                dim=(1,2)
            )
        except RuntimeError as e:
            print(f"Error in KL_GMM calculation:")
            print(f"gamma_t shape: {gamma_t.shape}")
            print(f"log_var shape: {log_var.shape}")
            print(f"mean shape: {mean.shape}")
            print(f"gaussian_means shape: {gaussian_means.shape}")
            print(f"gaussian_log_vars shape: {gaussian_log_vars.shape}")
            raise e
        
        # 4. 标准正态分布的KL散度
        kl_standard = -0.5 * torch.sum(1 + log_var, dim=2) # - mean.pow(2) - torch.exp(log_var)
        
        # 5. GMM熵项
        pi = self.gaussian.pi.unsqueeze(0)  # [1, n_clusters]
        entropy = (
            -torch.sum(torch.log(pi + 1e-10) * gamma, dim=-1) +
            torch.sum(torch.log(gamma + 1e-10) * gamma, dim=-1)
        )

        # 6. spectral constraints
        # spectral_constraints = self.lamb4 * torch.stack(list(self.compute_spectral_constraints(x, recon_x).values())).sum(0)
        spectral_constraints = self.lamb4 * self.compute_spectral_constraints(x, recon_x).sum(-1) * self.input_dim

        # 7. 总损失
        loss = recon_loss.mean() + kl_standard.mean() + kl_gmm.mean() +  entropy.mean() + spectral_constraints.mean()
        
        # 返回损失字典
        
        loss_dict = {
            'total_loss': loss,
            'recon_loss': recon_loss.mean().item(),
            'kl_gmm': kl_gmm.mean().item(),
            'kl_standard': kl_standard.mean().item(),
            'entropy': entropy.mean().item(),
            'spectral_loss': spectral_constraints.mean().item()
        }

        
        return loss_dict

    def _compute_cluster_separation(self, latent_vectors, cluster_assignments, method='cosine'):
        """计算类间分离损失，支持余弦相似度和欧氏距离两种方法
        
        Args:
            latent_vectors: 潜在空间向量 [batch_size, latent_dim]
            cluster_assignments: 聚类分配 [batch_size]
            method: 'cosine' 或 'dist'，选择计算方法
        """
        batch_size = latent_vectors.size(0)
        if batch_size < 2:
            return torch.tensor(0.0, device=latent_vectors.device)

        if method == 'cosine':
            # 原始的余弦相似度方法
            z_norm = F.normalize(latent_vectors, p=2, dim=1)
            sim_matrix = torch.mm(z_norm, z_norm.t())
            assignments_matrix = cluster_assignments.unsqueeze(0) == cluster_assignments.unsqueeze(1)
            same_class_sim = sim_matrix[assignments_matrix].mean()
            diff_class_sim = sim_matrix[~assignments_matrix].mean()
            return -same_class_sim + diff_class_sim

        elif method == 'dist':
            # 欧氏距离方法
            dist_matrix = torch.cdist(latent_vectors, latent_vectors)
            assignments_matrix = cluster_assignments.unsqueeze(0) == cluster_assignments.unsqueeze(1)
            same_class_mask = assignments_matrix.float()
            diff_class_mask = (~assignments_matrix).float()
            
            # 移除对角线
            diag_mask = ~torch.eye(batch_size, dtype=torch.bool, device=latent_vectors.device)
            same_class_mask = same_class_mask * diag_mask
            
            # 计算类内和类间距离
            intra_class_dist = (dist_matrix * same_class_mask).sum() / (same_class_mask.sum() + 1e-10)
            inter_class_dist = (dist_matrix * diff_class_mask).sum() / (diff_class_mask.sum() + 1e-10)
            
            return torch.relu(intra_class_dist - inter_class_dist + 1.0)  # margin=1.0
        
        else:
            raise ValueError(f"Unsupported method: {method}. Use 'cosine' or 'dist'.")

    def report(self, epoch, loss_dict, train_time):
        """确保所有损失值都正确显示"""
        report_str = (
            f"Epoch: {epoch:d}, "
            f"Loss: {loss_dict['total_loss'].item():.4f}, "  # 这里调用.item()
            f"Recon Loss: {loss_dict['recon_loss']:.4f}, "
            f"KL Loss: {loss_dict['kl_loss']:.4f}, "
            f"Spectral Loss: {loss_dict['spectral_loss']:.4f}, "
            f"Clustering Confidence Loss: {loss_dict['clustering_confidence_loss']:.4f}, "
            f"Cluster Separation Loss: {loss_dict['cluster_separation_loss']:.4f}, "
            f"Time: {train_time:.2f}s"
        )
        print(report_str)

