import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
import numpy as np
from scipy import signal,sparse
import os
import torch.nn.functional as F
import torch.cuda
from utility import leiden_clustering, compute_cluster_means
import time
from collections import defaultdict

class Encoder(nn.Module):
    def __init__(self, input_dim, intermediate_dim,latent_dim):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        dim_1 = intermediate_dim*4
        dim_2 = intermediate_dim*2
        self.net = nn.Sequential(
            nn.Linear(input_dim, dim_1),
            nn.BatchNorm1d(dim_1),
            nn.LeakyReLU(),
            nn.Linear(dim_1, dim_2),
            nn.BatchNorm1d(dim_2),
            nn.LeakyReLU(),
            nn.Linear(dim_2, intermediate_dim),
            nn.BatchNorm1d(intermediate_dim),
            nn.LeakyReLU(),
        )
        self.to_mean = nn.Linear(intermediate_dim, latent_dim)
        self.to_logvar = nn.Linear(intermediate_dim, latent_dim)

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
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        dim_1 = intermediate_dim*4
        dim_2 = intermediate_dim*2
        self.net = nn.Sequential(
            nn.Linear(latent_dim, intermediate_dim),
            nn.BatchNorm1d(intermediate_dim),
            nn.LeakyReLU(),
            nn.Linear(intermediate_dim, dim_2),
            nn.BatchNorm1d(dim_2),
            nn.LeakyReLU(),
            nn.Linear(dim_2, dim_1),
            nn.BatchNorm1d(dim_1),
            nn.LeakyReLU(),
            nn.Linear(dim_1, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.net(z)

class Gaussian(nn.Module):
    def __init__(self, num_classes, latent_dim):
        super(Gaussian, self).__init__()
        self.num_classes = num_classes
        self.means = nn.Parameter(torch.zeros(num_classes, latent_dim))
        self.log_variances = nn.Parameter(torch.zeros(num_classes, latent_dim))

    def forward(self, z):
        log_p = self.gaussian_log_prob(z)
        means = torch.sum(torch.exp(log_p.unsqueeze(2) - log_p.max(dim=1, keepdim=True)[0].unsqueeze(1)) * self.means,
                          dim=1)
        y_pre = torch.exp(self.gaussian_log_prob(z))
        return means, y_pre

    def gaussian_log_prob(self, z):
        log_p_list = []
        for c in range(self.num_classes):
            log_p_list.append(
                self.gaussian_log_pdf(z, self.means[c:c + 1, :], self.log_variances[c:c + 1, :]).unsqueeze(1))
        return torch.cat(log_p_list, dim=1)

    @staticmethod
    def gaussian_log_pdf(x, mu, log_var):
        return -0.5 * (torch.sum(log_var + ((x - mu).pow(2) / torch.exp(log_var)), dim=1))

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
        self.relu = nn.LeakyReLU()
        
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
    def __init__(self, input_dim, intermediate_dim, latent_dim, num_classes, device, l_c_dim, encoder_type="basic", lamb1=1.0, lamb2=1.0, lamb3=1.0, lamb4=1.0, lamb5=1.0, lamb6=1.0, lamb7=1.0, cluster_separation_method='cosine', batch_size=None):
        super(VaDE, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.batch_size = batch_size
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

    @torch.no_grad()
    def init_kmeans_centers(self, dataloader):
        """初始化聚类中心"""
        encoded_data = []
        for x, _ in dataloader:
            x = x.to(self.device)
            mean, _ = self.encoder(x)
            encoded_data.append(mean.cpu())
        encoded_data = torch.cat(encoded_data, dim=0).numpy()

        # 使用 KMeans 初始化
        kmeans = KMeans(n_clusters=self.num_classes, random_state=0)
        kmeans.fit(encoded_data)
        cluster_centers = torch.tensor(kmeans.cluster_centers_, device=self.device)

        # 初始化高斯分布参数
        self.gaussian.means.data.copy_(cluster_centers)
        self.cluster_centers = cluster_centers
    # def init_kmeans_centers(self,x):
    #     """初始化聚类中心"""
    #     encoded_data, _ = self.encoder(x)
    #     encoded_data = encoded_data.to(x.device)
    #     kmeans = KMeans(n_clusters=self.num_classes)
    #     cluster_labels = kmeans.fit_predict(encoded_data.cpu().detach().numpy())
    #     cluster_centers = kmeans.cluster_centers_
    #     cluster_centers = torch.tensor(cluster_centers).to(x.device)
    #     self.gaussian.means.data.copy_(cluster_centers)
    #     self.cluster_centers = cluster_centers
    @torch.no_grad()
    def update_kmeans_centers(self, dataloader):
        """使用 KMeans 更新聚类中心"""
        encoded_data = []
        for x, _ in dataloader:
            x = x.to(self.device)
            mean, _ = self.encoder(x)
            encoded_data.append(mean.cpu())
        encoded_data = torch.cat(encoded_data, dim=0).numpy()

        # 使用 KMeans 更新中心
        kmeans = KMeans(n_clusters=self.num_classes, random_state=0)
        kmeans.fit(encoded_data)
        cluster_centers = torch.tensor(kmeans.cluster_centers_, device=self.device)

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
        recon_x = self.decoder(z)
        z_prior_mean, y = self.gaussian(z)
        return recon_x, mean, log_var, z, z_prior_mean, y

    def kl_div_loss(self, mean, log_var, z_prior_mean):
        z_mean = mean.unsqueeze(1)
        z_log_var = log_var.unsqueeze(1)
        kl_loss = -0.5 * torch.sum(1 + z_log_var - (z_mean - z_prior_mean) ** 2 - torch.exp(z_log_var), dim=-1)
        kl_loss = torch.mean(kl_loss)
        return kl_loss

    def vae_loss(self, recon_x, x):
        xent_loss = 0.5 * torch.mean((x - recon_x) ** 2)
        return xent_loss 

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
                x[:, peak_positions]
            )
        
        # 2. 基线损失计算保持不变
        mean_baseline = stats['baseline_params']['mean_baseline'].to(x.device)
        baseline_std = stats['baseline_params']['std_baseline'].to(x.device)
        x_baseline = self.spectral_constraints.batch_als_baseline(x)
        recon_baseline = self.spectral_constraints.batch_als_baseline(recon_x)
        baseline_diff = (recon_baseline - x_baseline) / (baseline_std + 1e-6)
        constraints['baseline'] = torch.mean(baseline_diff ** 2)
        
        # 3. 强度范围约束
        intensity_range = stats['intensity_range']
        min_val = torch.tensor(intensity_range['min'], device=x.device)
        max_val = torch.tensor(intensity_range['max'], device=x.device)
        range_loss = (
            F.relu(min_val - recon_x).mean() + 
            F.relu(recon_x - max_val).mean()
        )
        constraints['range'] = range_loss
        
        return constraints

    def compute_loss(self, x, recon_x, mean, log_var, z_prior_mean, y):
        """计算所有损失"""
        # 1. 基础损失计算
        recon_loss = self.vae_loss(recon_x, x) / (torch.mean(x**2) + 1e-8)
        kl_loss = self.kl_div_loss(mean, log_var, z_prior_mean) / self.latent_dim
        peak_loss = torch.mean(self.peak_detector(x) * (x - recon_x) ** 2)
        
        # 2. 光谱约束损失
        spectral_constraints = self.compute_spectral_constraints(x, recon_x)
        spectral_loss = sum(spectral_constraints.values())  # 合并所有光谱约束损失
        
        # 3. 聚类损失
        cluster_probs = F.softmax(y, dim=1)
        clustering_confidence_loss = -torch.mean(torch.max(cluster_probs, dim=1)[0])
        
        # 计算类间分离损失
        cluster_assignments = torch.argmax(cluster_probs, dim=1)
        cluster_separation_loss = self._compute_cluster_separation(
            mean, 
            cluster_assignments,
            method=self.cluster_separation_method
        )

        # 4. 合并所有损失
        total_loss = (
            self.lamb1 * recon_loss +
            self.lamb2 * kl_loss +
            self.lamb4 * spectral_loss +
            self.lamb5 * clustering_confidence_loss +
            self.lamb6 * cluster_separation_loss
        )

        # 5. 返回详细的损失字典
        loss_dict = {
            'total_loss': total_loss,  # 不要调用.item()，保持计算图
            'recon_loss': recon_loss.item(),  # 这些可以转换为标量
            'kl_loss': kl_loss.item(),
            'spectral_loss': spectral_loss.item(),
            'clustering_confidence_loss': clustering_confidence_loss.item(),
            'cluster_separation_loss': cluster_separation_loss.item()
        }
        
        # 添加详细的光谱约束损失
        for key, value in spectral_constraints.items():
            loss_dict[f'spectral_{key}'] = value.item()
        
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

