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


class Encoder(nn.Module):
    def __init__(self, input_dim, intermediate_dim,latent_dim):
        """
        Args:
            input_dim: 输入维度
            intermediate_dim: 中间维度
            latent_dim: 潜在空间
        """
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        layers = []
        prev_dim = input_dim
        for dim in intermediate_dim:
            layers.extend([
                nn.Linear(prev_dim, dim),
                # nn.BatchNorm1d(dim),
                nn.ReLU(),
            ])
            prev_dim = dim

        self.net = nn.Sequential(*layers)

        self.to_mean = nn.Linear(intermediate_dim[-1], latent_dim)
        self.to_logvar = nn.Linear(intermediate_dim[-1], latent_dim)

    def forward(self, x):
        x = self.net(x)
        mean = self.to_mean(x)
        log_var = self.to_logvar(x)
        return mean, log_var

#CNN
class CNNEncoder(nn.Module):
    def __init__(self, input_dim, cnn1,cnn2,cnn3,latent_dim):
        super(CNNEncoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # 残差块
        self.conv_layers = nn.ModuleList([
            self.make_res_block(1, cnn1, kernel_size=5),
            self.make_res_block(cnn1, cnn2, kernel_size=15),
            self.make_res_block(cnn2, cnn3, kernel_size=25),
        ])

        # 自适应归一化
        self.adaptive_norm = AdaptiveNormalization(alpha=0.1)

        # 输出层
        self.to_mean = nn.Linear(cnn3, latent_dim)
        self.to_logvar = nn.Linear(cnn3, latent_dim)

    def make_res_block(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(out_channels),
        )

    def forward(self, x):
        # 调整输入形状为 [batch_size, channels=1, sequence_length]
        x = x.unsqueeze(1)

        # 适应归一化
        x = self.adaptive_norm(x)

        # 遍历残差块
        for res_block in self.conv_layers:
            identity = x
            x = res_block(x)

            # 确保形状一致后再残差连接
            if x.shape == identity.shape:
                x = x + identity
            x = F.relu(x)

        # 展平为向量
        x = x.mean(dim=-1)  # 取序列均值作为特征向量
        mean = self.to_mean(x)
        log_var = self.to_logvar(x)

        return mean, log_var

# 多尺度
class AdvancedEncoder(nn.Module):
    def __init__(self, input_dim,cnn1,cnn2,cnn3, latent_dim):
        super(AdvancedEncoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # 多尺度空洞卷积块
        self.dilated_blocks = nn.ModuleList([
            self.make_dilated_block(1, cnn1, kernel_size=21, dilation=1),
            self.make_dilated_block(cnn1, cnn2, kernel_size=21, dilation=2),
            self.make_dilated_block(cnn2, cnn3, kernel_size=21, dilation=4),
        ])

        # 多尺度池化层
        self.multi_scale_pool = MultiScalePooling(cnn3)

        # 输出层
        self.to_mean = nn.Linear(cnn3*len(self.multi_scale_pool.scales), latent_dim)
        self.to_logvar = nn.Linear(cnn3*len(self.multi_scale_pool.scales), latent_dim)

    def forward(self, x):
        # 调整输入形状
        x = x.unsqueeze(1)

        # 空洞卷积块
        for dilated_block in self.dilated_blocks:
            identity = x
            x = dilated_block(x)

            # 确保形状一致后再残差连接
            if x.shape == identity.shape:
                x = x + identity
            x = F.relu(x)

        # 多尺度池化
        x = self.multi_scale_pool(x)

        # 展平为向量
        x = x.mean(dim=-1)
        mean = self.to_mean(x)
        log_var = self.to_logvar(x)

        return mean, log_var

    @staticmethod
    def make_dilated_block(in_channels, out_channels, kernel_size, dilation):
        padding = ((kernel_size - 1) * dilation) // 2
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation,stride=2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation,stride=1),
            nn.BatchNorm1d(out_channels),
        )

class MultiScalePooling(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.scales = [5, 10, 20, 40]  # 不同池化尺度

        self.pools = nn.ModuleList([
            nn.Sequential(
                nn.AvgPool1d(scale),
                nn.Conv1d(in_channels, in_channels, 1),  # 1x1卷积保持通道数
                nn.BatchNorm1d(in_channels),
                nn.ReLU()
            ) for scale in self.scales
        ])

    def forward(self, x):
        """对输入进行多尺度池化处理

        Args:
            x: 输入张量 [batch, channels, length]

        Returns:
            多尺度池化后的特征张量
        """
        original_size = x.size(-1)
        outputs = []

        # 对每个尺度进行池化
        for pool in self.pools:
            pooled = pool(x)
            # 上采样回原始大小
            pooled = F.interpolate(
                pooled,
                size=original_size,
                mode='linear',
                align_corners=True
            )
            outputs.append(pooled)
        concatenated = torch.cat(outputs, dim=1)
        # 合并所有尺度的特征
        return concatenated

class AdaptiveNormalization(nn.Module):
    def __init__(self, alpha=0.1):
        super(AdaptiveNormalization, self).__init__()
        self.alpha = alpha
    def forward(self, x):
        # 计算峰值区域
        peaks = self.find_peaks(x)
        # 计算权重
        weights = 1 + self.alpha * peaks
        # 归一化
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        return weights * (x - mean) / (std + 1e-5)

    def find_peaks(self, x):
        """使用scipy.signal.find_peaks进行峰值检测

        Args:
            x: 输入光谱张量 [batch, 1, spectrum_length]

        Returns:
            peaks: 峰值位置的二值张量 [batch, 1, spectrum_length]
        """
        # 将tensor转为numpy进行处理
        x_np = x.detach().cpu().numpy()
        batch_size = x_np.shape[0]
        spec_len = x_np.shape[2]
        peaks = torch.zeros_like(x)

        for i in range(batch_size):
            # 对每个光谱进行峰值检测
            spectrum = x_np[i, 0]
            max_intensity = np.max(spectrum)
            peak_indices, _ = signal.find_peaks(
                spectrum,
                height=0.1 * max_intensity,  # 最小峰高
                distance=10,  # 峰间最小距
                prominence=0.05 * max_intensity  # 小突出度
            )
            # 将峰值位置标记为1
            peaks[i, 0, peak_indices] = 1.0

        return peaks.to(x.device)

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

class SpectralConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 多尺度卷积捕获不同范围的光谱特征
        self.conv_small = nn.Conv1d(in_channels, out_channels//3, kernel_size=3, padding=1)
        self.conv_medium = nn.Conv1d(in_channels, out_channels//3, kernel_size=7, padding=3)
        self.conv_large = nn.Conv1d(in_channels, out_channels//3, kernel_size=11, padding=5)
        
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # 多尺度特征融合
        x_small = self.conv_small(x)
        x_medium = self.conv_medium(x)
        x_large = self.conv_large(x)
        x = torch.cat([x_small, x_medium, x_large], dim=1)
        return self.relu(self.bn(x))

class SpectralAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)  # 全局平均池化
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.ReLU(),
            nn.Linear(channels // 4, channels),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _ = x.size()
        # 计算通道注意力权重
        y = self.gap(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        # 应用注意力
        return x * y.expand_as(x)

class ImprovedSpectralEncoder(nn.Module):
    def __init__(self, input_dim, intermediate_dim, latent_dim):
        super().__init__()
        
        # 特征提取层
        self.feature_extractor = nn.Sequential(
            SpectralConvBlock(1, 32),
            SpectralAttention(32),
            SpectralConvBlock(32, 64),
            SpectralAttention(64),
            SpectralConvBlock(64, 128),
            SpectralAttention(128)
        )
        
        # 峰值检测分支
        self.peak_branch = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=1)
        )
        
        # 基线分支
        self.baseline_branch = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=15, padding=7),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=15, padding=7)
        )
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Linear(input_dim * 64, intermediate_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.2)
        )
        
        # 输出层
        self.to_mean = nn.Linear(intermediate_dim, latent_dim)
        self.to_logvar = nn.Linear(intermediate_dim, latent_dim)
        
    def forward(self, x):
        # 添加通道维度
        x = x.unsqueeze(1)
        
        # 特征提取
        features = self.feature_extractor(x)
        
        # 分支处理
        peak_features = self.peak_branch(features)
        baseline_features = self.baseline_branch(features)
        
        # 特征融合
        combined = torch.cat([peak_features, baseline_features], dim=1)
        batch_size = combined.size(0)
        combined = combined.view(batch_size, -1)
        fused = self.fusion(combined)
        
        # 生成均值和方差
        mean = self.to_mean(fused)
        logvar = self.to_logvar(fused)
        
        return mean, logvar

class VaDE(nn.Module):
    def __init__(self, input_dim, intermediate_dim, latent_dim,  device, l_c_dim, 
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
        self.encoder = self._init_encoder(input_dim, intermediate_dim, latent_dim, encoder_type, l_c_dim)
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

    # def init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         torch.nn.init.kaiming_normal_(m.weight)
    #         if m.bias is not None:
    #             m.bias.data.fill_(0.01)


    @staticmethod
    def _init_encoder(input_dim, intermediate_dim,latent_dim, encoder_type,l_c_dim):
        """初始化编码器"""
        if encoder_type == "basic":

            return Encoder(input_dim, intermediate_dim=intermediate_dim, latent_dim=latent_dim)
        elif encoder_type == "cnn":
            cnn1 = l_c_dim['cnn1']
            cnn2 = l_c_dim['cnn2']
            cnn3 = l_c_dim['cnn3']
            return CNNEncoder(input_dim, cnn1,cnn2,cnn3,latent_dim)

        elif encoder_type == "advanced":
            cnn1 = l_c_dim['cnn1']
            cnn2 = l_c_dim['cnn2']
            cnn3 = l_c_dim['cnn3']
            return AdvancedEncoder(input_dim,cnn1,cnn2,cnn3, latent_dim)
        elif encoder_type == "ImprovedSpectralEncoder":
            cnn1 = l_c_dim['cnn1']
            cnn2 = l_c_dim['cnn2']
            cnn3 = l_c_dim['cnn3']
            return AdvancedEncoder(input_dim,cnn1,cnn2,cnn3, latent_dim)
        else:
            raise ValueError(f"Unsupported encoder type: {encoder_type}")

    def pretrain(self, dataloader,learning_rate=1e-3):
        pre_epoch=self.pretrain_epochs
        if  not os.path.exists('./pretrain_model.pk'):

            Loss=nn.MSELoss()
            opti=torch.optim.Adam(itertools.chain(self.encoder.parameters(),self.decoder.parameters()))

            print('Pretraining......')
            epoch_bar=tqdm(range(pre_epoch))
            for _ in epoch_bar:
                L=0
                for x,y in dataloader:
                    x=x.to(self.device)

                    z,_=self.encoder(x)
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
            self.load_state_dict(torch.load('./pretrain_model_50.pk'))
            


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
        mean, _ = self.encoder(self.tensor_gpu_data)
        encoded_data.append(mean.cpu())
        encoded_data = torch.cat(encoded_data, dim=0).numpy()

        # 使用选定的聚类方法
        print(f"Using clustering method: {self.clustering_method}")
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


    @torch.no_grad()
    def update_kmeans_centers(self, dataloader):
        """更新聚类中心"""
        print(f'Update clustering centers..........')
        encoded_data = []
        for x, _ in dataloader:
            x = x.to(self.device)
            mean, _ = self.encoder(x)
            encoded_data.append(mean.cpu())
        encoded_data = torch.cat(encoded_data, dim=0).numpy()

        # 使用选定的方法更新聚类中心
        labels, cluster_centers = self._apply_clustering(encoded_data)
        num_clusters = len(np.unique(labels))
        cluster_centers = torch.tensor(cluster_centers, device=self.device)

        # 如果聚类数量发生变化，需要重新初始化高斯分布
        if num_clusters != self.gaussian.num_clusters:
            print(f"Number of clusters changed from {self.gaussian.num_clusters} to {num_clusters}. Reinitializing Gaussian.")
            self.num_clusters = num_clusters
            self.gaussian = Gaussian(num_clusters, self.latent_dim).to(self.device)


        # 更新高斯分布参数
        self.gaussian.means.data.copy_(cluster_centers)
        self.cluster_centers = cluster_centers


    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self,x):
        """模型前向传播"""
        mean, log_var = self.encoder(x)
        z = self.reparameterize(mean, log_var)
        
        # 通过GMM获取所有需要的值
        means, log_variances, y, gamma, pi = self.gaussian(z)
        
        # 解码
        recon_x = self.decoder(z)
        
        return recon_x, mean, log_var, z, gamma, pi


    def train_step(self, x):
        """单步训练
        Args:
            x: 输入数据
        """
        # 前向传播
        recon_x, mean, log_var, z, gamma, pi = self(x)
        
        # 计算损失
        loss = self.loss_function(recon_x, x, mean, log_var, z, gamma, pi)
        
        return loss

    def compute_spectral_constraints(self, x, recon_x):
        """计算光谱约束时只使用数据集统计信息"""
        constraints = {}
        stats = self.spectral_analyzer.get_dataset_stats()
        
        # 1. 使用统计信息中的峰位置
        peak_positions = stats['peak_positions']
        if peak_positions is not None:
            peak_positions = peak_positions.to(x.device)
            constraints['peak'] = F.mse_loss(
                recon_x[:, peak_positions], 
                x[:, peak_positions], reduction='none'
            ).sum(-1)
        
        # 2. 基线损失计算保持不变
        mean_baseline = stats['baseline_params']['mean_baseline'].to(x.device)
        baseline_std = stats['baseline_params']['std_baseline'].to(x.device)
        x_baseline = self.spectral_constraints.batch_als_baseline(x)
        recon_baseline = self.spectral_constraints.batch_als_baseline(recon_x)
        baseline_diff = (recon_baseline - x_baseline) / (baseline_std + 1e-6)
        constraints['baseline'] = (baseline_diff ** 2).sum(-1)
        
        # 3. 强度范围约束
        intensity_range = stats['intensity_range']
        min_val = torch.tensor(intensity_range['min'], device=x.device)
        max_val = torch.tensor(intensity_range['max'], device=x.device)
        range_loss = (
            F.relu(min_val - recon_x).sum(-1) + 
            F.relu(recon_x - max_val).sum(-1)
        )
        constraints['range'] = range_loss
        
        return constraints

    def compute_loss(self, x, recon_x, mean, log_var, z_prior_mean, y):

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
                    (mean - gaussian_means).pow(2) / (torch.exp(gaussian_log_vars) + 1e-10)
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
        spectral_constraints = self.lamb4 * torch.stack(list(self.compute_spectral_constraints(x, recon_x).values())).sum(0)

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

