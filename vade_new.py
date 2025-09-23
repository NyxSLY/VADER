import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
import numpy as np
from scipy import signal
import os
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from community import community_louvain
from utility import leiden_clustering
import math
from scipy.optimize import linear_sum_assignment
from ramanbiolib.search import PeakMatchingSearch, SpectraSimilaritySearch


class Encoder(nn.Module):
    def __init__(self, input_dim, intermediate_dim, latent_dim, n_components, S):
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
        self.S = nn.Parameter(S.clone().detach().float())  

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
        self.to_concentration = nn.Linear(intermediate_dim[-1], n_components)  # 浓度均值
        self.to_concentration_logvar = nn.Linear(intermediate_dim[-1], n_components)  # 浓度方差, (c+σ)S
        # self.to_concentration_logvar = nn.Linear(intermediate_dim[-1], latent_dim)  # 分解损失, cS+σ

    def forward(self, x):
        x = self.net(x)
        concentration = F.softplus(self.to_concentration(x)) # 浓度均值，#仅非负约束：softplus
        concentration_logvar = self.to_concentration_logvar(x)  # 浓度方差
        S_pos = F.relu(self.S) # 非负，L2规范化（每个component平方和为1）
        return concentration, concentration_logvar, S_pos


class PeakDetector(nn.Module):
    def __init__(self, height_factor=0.1, min_distance=10, prominence_factor=0.05):
        super().__init__()
        self.height_factor = height_factor
        self.min_distance = min_distance
        self.prominence_factor = prominence_factor


    def forward(self, x):
        """直接在 GPU 上计算峰值权重"""
        device = x.device
        batch_size, N = x.shape
        peaks = torch.zeros_like(x)  

        # 归一化到 [0, 1]
        x_normalized = (x - x.min(dim=-1, keepdim=True)[0]) / (
            x.max(dim=-1, keepdim=True)[0] - x.min(dim=-1, keepdim=True)[0] + 1e-5
        )

        # 批量检测峰值
        for i in range(batch_size):
            spectrum = x_normalized[i].detach().cpu().numpy()
            peak_indices, _ = signal.find_peaks(
                spectrum,
                height=self.height_factor,
                distance=self.min_distance,
                prominence=self.prominence_factor,
            )
            peaks[i, peak_indices] = 1.0

        return peaks  

class Decoder(nn.Module):
    def __init__(self, latent_dim,intermediate_dim, input_dim, n_components):
        super(Decoder, self).__init__()
        
        self.S = nn.Parameter(torch.randn(n_components, latent_dim, dtype=torch.float32))

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
        return self.net(torch.matmul(z,self.S))

class Gaussian(nn.Module):
    def __init__(self, num_clusters, latent_dim, global_label_mapping=None):
        super(Gaussian, self).__init__()
        self.num_clusters = num_clusters
        self.latent_dim = latent_dim
        self.global_label_mapping = global_label_mapping # 全局标签映射
        
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
                self.log_variances.data.copy_(variances)
            if weights is not None:
                self.pi.data.copy_(weights)

    def forward(self, z, labels_batch=None):
        if labels_batch is not None: # 如果有批次标签，就使用它
            # labels_batch 预计是 1D 的标签张量
            y_labels_tensor = labels_batch.cpu().numpy() # Convert to numpy for mapping
            
            # 使用全局映射重映射批次标签
            if self.global_label_mapping is not None:
                remapped_labels_np = np.array([self.global_label_mapping[label.item()] for label in y_labels_tensor])
                y_labels_tensor_remapped = torch.tensor(remapped_labels_np, device=z.device, dtype=torch.long)
            else:
                # Fallback if no global mapping, though it should exist if prior_y was given
                unique_batch_labels = torch.unique(labels_batch)
                batch_label_mapping = {old_label.item(): new_label for new_label, old_label in enumerate(unique_batch_labels)}
                remapped_labels_np = np.array([batch_label_mapping[label.item()] for label in y_labels_tensor])
                y_labels_tensor_remapped = torch.tensor(remapped_labels_np, device=z.device, dtype=torch.long)

            # 为 gamma 创建独热编码，使用全局 num_clusters
            gamma = F.one_hot(y_labels_tensor_remapped, num_classes=self.num_clusters).float()
            y = y_labels_tensor_remapped # y 也应该是重映射后的标签

        else: # 如果没有批次标签，则回退到原来的GMM计算方式
            y = self.gaussian_log_prob(z)  # log p(c|z) # 计算条件概率/可能性
            gamma = F.softmax(y, dim=1)  # gamma(c)

        # 计算条件均值
        means = torch.sum(gamma.unsqueeze(2) * self.means.unsqueeze(0), dim=1)
        # print(self.means[1,:])

        # 返回所有需要的值
        return self.means,self.log_variances, y, gamma, self.pi  

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


class VaDE(nn.Module):
    def __init__(self, input_dim, intermediate_dim, latent_dim,  device, l_c_dim, n_components, S,wavenumber,
                 prior_y = None, encoder_type="basic", batch_size=None, tensor_gpu_data=None,
                 lamb1=1.0, lamb2=1.0, lamb3=1.0, lamb4=1.0, lamb5=1.0, lamb6=1.0, lamb7=1.0, 
                 pretrain_epochs=50,
                 num_classes=0, resolution_1=1.0, resolution_2=0.9, clustering_method='leiden'):
        super(VaDE, self).__init__()
        self.device = device
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.tensor_gpu_data = tensor_gpu_data
        self.n_components = n_components
        self.wavenumber = wavenumber
        self.prior_y = prior_y

        # Dynamically set self.num_classes and create global_label_mapping if prior_y is provided
        if self.prior_y is not None:
            if not isinstance(self.prior_y, np.ndarray):
                self.prior_y = np.array(self.prior_y)
            
            unique_prior_labels = np.unique(self.prior_y)
            self.num_classes = len(unique_prior_labels) # Update num_classes based on prior_y
            self.global_label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_prior_labels)}
        else:
            self.num_classes = num_classes # Use the provided num_classes if no prior_y
            self.global_label_mapping = None

        self.encoder = Encoder(input_dim, intermediate_dim=intermediate_dim, latent_dim=latent_dim, n_components=n_components, S=S)
        self.decoder = Decoder(latent_dim, intermediate_dim, input_dim, n_components)
        # Pass global_label_mapping to Gaussian
        self.gaussian = Gaussian(self.num_classes, n_components, global_label_mapping=self.global_label_mapping)
        self.cluster_centers = None
        self.lamb1 = lamb1
        self.lamb2 = lamb2
        self.lamb3 = lamb3
        self.lamb4 = lamb4
        self.lamb5 = lamb5
        self.lamb6 = lamb6
        self.lamb7 = lamb7
        self.pretrain_epochs = pretrain_epochs
        self.num_classes = num_classes
        self.resolution_1 = resolution_1
        self.resolution_2 = resolution_2
        self.clustering_method = clustering_method
        self.input_dim = input_dim

        self.peak_detector = PeakDetector().to(device)
        self.spectral_analyzer = self.get_peak_positions()
        self.spectra_search = SpectraSimilaritySearch(wavenumbers=wavenumber[np.where((wavenumber <= 1800) & (wavenumber >= 450) )[0]])  
        self.to(device)



    def pretrain(self, dataloader,learning_rate=1e-3):
        pre_epoch=self.pretrain_epochs
        if  not os.path.exists('./nc9_pretrain_model_none_bn_.pk'):

            Loss=nn.MSELoss()
            # opti=torch.optim.Adam(itertools.chain(self.encoder.parameters()))
            params = [p for n, p in self.encoder.named_parameters() if n != "S"]
            opti = torch.optim.Adam(params)

            # epoch_bar=tqdm(range(pre_epoch))
            for _ in range(pre_epoch):
                L=0
                for x,y in dataloader:
                    x=x.to(self.device)

                    mean,var, S = self.encoder(x)
                    z = self.reparameterize(mean, var)
                    x_=torch.matmul(z, S)
                    loss=Loss(x,x_)

                    L+=loss.detach().cpu().numpy()

                    opti.zero_grad()
                    loss.backward()
                    opti.step()

                # epoch_bar.write('L2={:.4f}'.format(L/len(dataloader)))

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
    def init_kmeans_centers(self, z):
        encoded_data = z.cpu().numpy()

        # 使用选定的聚类方法
        print(f"Using clustering method: {self.clustering_method}")
        print(f'num_clusters: {self.num_classes}')
    
        # update the clustering centers
        if self.prior_y is not None:
            labels = self.prior_y
            cluster_centers = np.array([encoded_data[labels == i].mean(axis=0) for i in np.unique(labels)])
        else:
            labels, cluster_centers = self._apply_clustering(encoded_data)
        
        cluster_centers = torch.tensor(cluster_centers, device=self.device)
        num_clusters = len(np.unique(labels))

        # 如果聚类数量发生变化，需要重新初始化高斯分布
        if num_clusters != self.gaussian.num_clusters:
            print(f"Number of clusters changed from {self.gaussian.num_clusters} to {num_clusters}. Reinitializing Gaussian.")
            self.num_clusters = num_clusters
            self.gaussian = Gaussian(num_clusters, self.n_components, global_label_mapping=self.global_label_mapping).to(self.device)
            self.gaussian.update_parameters(cluster_centers=cluster_centers)

        # 直接用聚类中心更新高斯分布参数
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
    def match_components(self,S, min_similarity):
        valid_idx = np.where((self.wavenumber <= 1800) & (self.wavenumber >= 450) )[0]
        wavenumber = self.wavenumber[valid_idx]
        S_valid = S[:,valid_idx]

        match_specs = []
        match_chems = []       
        spectra_search = self.spectra_search

        for i in range(0,S_valid.shape[0]):
            unknown_comp = S_valid[i].cpu().numpy()
            search_results = spectra_search.search(
                unknown_comp,
                class_filter=None,
                unique_components_in_results=True,
                similarity_method="cosine_similarity",
                similarity_params=25
                )
            top_100 = search_results.get_results(limit=100)

            top_similar = top_100[(top_100['similarity_score'] >= min_similarity) & (top_100['laser'] == 532.0)]
            if top_similar.empty:
                match_spec = unknown_comp
                match_chem = 'Unknown'
            else: 
                match_id = top_similar['id'].iloc[0]
                match_wave = np.array(search_results.database['wavenumbers'][match_id-1])
                match_spec = np.array(search_results.database['intensity'][match_id-1])
                match_chem = search_results.database['component'][match_id-1]
            match_specs.append(match_spec)
            match_chems.append(match_chem)

        match_specs = np.array(match_specs)
        return match_specs, match_chems

    @torch.no_grad()
    def update_kmeans_centers(self, z):
        print(f'Update clustering centers..........')
        encoded_data_cpu = z.cpu().numpy()

        # update the clustering centers
        if self.prior_y is not None:
            ml_labels = self.prior_y
            cluster_centers = np.array([encoded_data_cpu[ml_labels == i].mean(axis=0) for i in np.unique(ml_labels)])
            cluster_var = np.array([np.log(encoded_data_cpu[ml_labels == i].var(axis=0) + 1e-6) for i in np.unique(ml_labels)]) 
        else:
            ml_labels, cluster_centers = self._apply_clustering(encoded_data_cpu)
            cluster_var = self.gaussian.log_variances.data
        
        cluster_pi = self.gaussian.pi.data
        gaussian_means = self.gaussian.means.cpu().numpy()
        
        num_ml_centers = len(np.unique(ml_labels))

        """align"""
        if num_ml_centers != self.gaussian.num_clusters:
            print(f"Numberof clusters changed from {self.gaussian.num_clusters} to {num_ml_centers} .Reinitializing Gaussian.")
            cluster_centers = self.optimal_transport(cluster_centers, gaussian_means, 1)
            array_var = self.gaussian.log_variances.cpu().numpy()
            array_pi = self.gaussian.pi.cpu().numpy()
            aligned_gaussian_var = self.change_var(array_var,num_ml_centers,w=1.0)
            aligned_gaussian_pi = self.change_pi(array_pi,num_ml_centers,w=0.5)
            self.gaussian = Gaussian(num_ml_centers, self.n_components,global_label_mapping=self.global_label_mapping).to(self.device)
            cluster_var = torch.tensor(aligned_gaussian_var,device = self.device)
            cluster_pi = torch.tensor(aligned_gaussian_pi,device = self.device)
        else:
            cluster_centers = self.optimal_transport(cluster_centers, gaussian_means, 1)
            array_var = self.gaussian.log_variances.cpu().numpy()
            array_pi = self.gaussian.pi.cpu().numpy()
            aligned_gaussian_var = self.change_var(array_var,num_ml_centers,w=1.0)
            aligned_gaussian_pi = self.change_pi(array_pi,num_ml_centers,w=0.5)
            cluster_var = torch.tensor(aligned_gaussian_var,device = self.device)
            cluster_pi = torch.tensor(aligned_gaussian_pi,device = self.device)
        cluster_centers_t = torch.tensor(cluster_centers,device = self.device)
        self.gaussian.update_parameters(cluster_centers=cluster_centers_t, variances=cluster_var, weights=cluster_pi)
 

    def reparameterize(self, concentration, log_var):
        """
        重参数化过程，将浓度参数转换为潜在空间表示
        Args:
            concentration: 浓度均值 [batch_size, n_components]
            log_var: 浓度方差 [batch_size, n_components]
        Returns:
            z: 潜在空间表示 [batch_size, latent_dim]
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        # (C+σ)*S
        # return torch.matmul(concentration + eps * std, spectra)
        # CS+σ
        return concentration + eps * std

    def forward(self,x, labels_batch=None): # Add labels_batch parameter
        """模型前向传播"""
        mean, log_var, S= self.encoder(x)
        z = self.reparameterize(mean, log_var)
        
        # 通过GMM获取所有需要的值
        means, log_variances, y, gamma, pi = self.gaussian(z, labels_batch) # Pass labels_batch
        
        # 解码
        # recon_x = self.decoder(z)
        recon_x = torch.matmul(z, S)
        
        return recon_x, mean, means, log_var, z, gamma, pi, S

    @torch.no_grad()
    def constraint_angle(self, x, weight=0.05):
        """
        S: 要约束的矩阵（一组组分或浓度行/列），shape [n, dim]
        x: 原始输入矩阵
        """
        S_init = self.encoder.S
        m = x.mean(axis=0)  # 对列求均值
        m = m / (m.norm(p=2)+1e-8)  # 单位化
        S_init = S_init / (S_init.norm(p=2, dim=1, keepdim=True) + 1e-8)  # 行单位化
        return (1 - weight) * S_init + weight * m


    @torch.no_grad()
    def get_peak_positions(self):
        all_spectra = self.tensor_gpu_data.cpu()
        peaks = self.peak_detector(all_spectra)
        peak_counts = torch.sum(peaks, dim=0)
        min_occurrences = 0.1 * peaks.shape[0]  # 10%的样本数
        peak_positions = torch.where(peak_counts >= min_occurrences)[0]
        return peak_positions

    @torch.no_grad()
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



    def compute_loss(self, x, recon_x, mean, log_var, z_prior_mean, gamma, S, matched_S):

        """计算所有损失，严格按照原始VaDE论文"""
        batch_size = x.size(0)
        
        # 1. 重构损失
        recon_loss = self.lamb1 * F.mse_loss(recon_x, x, reduction='none').sum(-1)

        # 2. GMM先验的KL散度       
        # 扩展gamma的维度用于广播
        gamma_t = gamma.unsqueeze(-1)  # [batch_size, num_clusters, 1]
        
        # 调整其他张量的维度
        mean = mean.unsqueeze(1)  # [batch_size, 1, latent_dim]
        log_var = log_var.unsqueeze(1)  # [batch_size, 1, latent_dim]
        
        # 调整高斯分布参数的维度
        gaussian_means = self.gaussian.means.unsqueeze(0)  # [1, n_clusters, latent_dim]
        gaussian_log_vars = self.gaussian.log_variances.unsqueeze(0)  # [1, n_clusters, latent_dim]
        
        kl_gmm = torch.sum(
            0.5 * gamma_t * (
                self.latent_dim * math.log(2*math.pi) + gaussian_log_vars +
                torch.exp(log_var) / (torch.exp(gaussian_log_vars) + 1e-10) +
                (mean - gaussian_means).pow(2) / (torch.exp(gaussian_log_vars) + 1e-10)
            ),
            dim=(1,2)
        )

        
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
        # spectral_constraints = self.lamb4 * self.compute_spectral_constraints(x, recon_x).sum(-1) * self.input_dim

        # Unsimilar of S
        # SS = torch.matmul(S, S.t())
        # I = torch.eye(S.shape[0], device=self.device)
        # ortho_loss = ((SS - I) ** 2).sum()
        # spectral_constraints = self.lamb4 * ortho_loss

        # 6. Match Loss
        # matched_comp = torch.tensor(matched_S, dtype=torch.float32, device = self.device)
        # valid_idx = np.where((self.wavenumber <= 1800) & (self.wavenumber >= 450) )[0]
        # S_valid = S[:,valid_idx]
        # cos_sim = F.cosine_similarity(S_valid, matched_comp, dim=1) 
        # match_loss = 1 - cos_sim
        # match_loss_bioDB = self.lamb4 * match_loss

        # 7. 总损失
        loss = recon_loss.mean() + kl_standard.mean() + kl_gmm.mean() +  entropy.mean() #+ match_loss_bioDB.mean()
        
        # 返回损失字典
        
        loss_dict = {
            'total_loss': loss,
            'recon_loss': recon_loss.mean().item(),
            'kl_gmm': kl_gmm.mean().item(),
            'kl_standard': kl_standard.mean().item(),
            'entropy': entropy.mean().item(),
            #'spectral_loss': match_loss_bioDB.mean().item()
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
            f"Loss: {loss_dict['total_loss'].item():.4f}, "  
            f"Recon Loss: {loss_dict['recon_loss']:.4f}, "
            f"KL Loss: {loss_dict['kl_loss']:.4f}, "
            f"Spectral Loss: {loss_dict['spectral_loss']:.4f}, "
            f"Clustering Confidence Loss: {loss_dict['clustering_confidence_loss']:.4f}, "
            f"Cluster Separation Loss: {loss_dict['cluster_separation_loss']:.4f}, "
            f"Time: {train_time:.2f}s"
        )
        print(report_str)

