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


class Decoder(nn.Module):
    def __init__(self, latent_dim,intermediate_dim, input_dim, n_components):
        super(Decoder, self).__init__()
        
        self.S = nn.Parameter(torch.randn(n_components, latent_dim, dtype=torch.float64))

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


class VaDE(nn.Module):
    def __init__(self, input_dim, intermediate_dim, latent_dim,  device, l_c_dim, n_components, S,wavenumber,
                 prior_y = None, encoder_type="basic",  tensor_gpu_data=None,
                 pretrain_epochs=50,
                 num_classes=0, resolution=1.0,clustering_method='leiden'):
        super(VaDE, self).__init__()
        self.device = device
        self.latent_dim = latent_dim
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
        # self.decoder = Decoder(latent_dim, intermediate_dim, input_dim, n_components)
        self.pi_ = nn.Parameter(torch.full((self.n_components,), 1.0 / float(self.n_components), dtype=torch.float64, device=self.device),requires_grad=True)
        self.c_mean = nn.Parameter(torch.zeros(self.n_components, self.latent_dim, dtype=torch.float64, device=self.device),requires_grad=True)
        self.c_log_var = nn.Parameter(torch.zeros(self.n_components, self.latent_dim, dtype=torch.float64, device=self.device),requires_grad=True)
       
        self.cluster_centers = None
        self.pretrain_epochs = pretrain_epochs
        self.num_classes = num_classes
        self.resolution = resolution
        self.clustering_method = clustering_method
        self.input_dim = input_dim

        self.spectra_search = SpectraSimilaritySearch(wavenumbers=wavenumber[np.where((wavenumber <= 1800) & (wavenumber >= 450) )[0]])  
        

    def pretrain(self, dataloader,learning_rate=1e-3):
        pre_epoch=self.pretrain_epochs
        if  not os.path.exists('./nc9_pretrain_model_none_bn_.pk'):

            Loss=nn.MSELoss()
            params = [p for n, p in self.encoder.named_parameters() if n != "S"]
            opti = torch.optim.Adam(params, lr=learning_rate)

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

            # torch.save(self.state_dict(), './pretrain_model_50.pk')

        else:
            self.load_state_dict(torch.load('./nc9_pretrain_model_none_bn.pk'))
            
    def cal_gaussian_gamma(self,z):
        z_expanded = z.unsqueeze(1)  # [batch_size, 1, latent_dim]
        means_expanded = self.c_mean.unsqueeze(0)  # [1, num_clusters, latent_dim]
        log_vars_expanded = self.c_log_var.unsqueeze(0)  # [1, num_clusters, latent_dim]
        pi_expanded = self.pi_.unsqueeze(0)  # [1, num_clusters]
    
        gamma = (
            torch.log(pi_expanded) * self.latent_dim                   # 混合权重项
            - 0.5 * torch.sum(torch.log(2*math.pi*torch.exp(log_vars_expanded)), dim=2)  # 常数项和方差项
            - torch.sum((z_expanded - means_expanded).pow(2)/(2*torch.exp(log_vars_expanded)), dim=2)  # 指数项
        )

        return F.softmax(gamma,dim=1)


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
            nn = NearestNeighbors(n_neighbors=10)
            nn.fit(encoded_data)
            adj_matrix = nn.kneighbors_graph(encoded_data, mode='distance')
            
            G = nx.from_scipy_sparse_matrix(adj_matrix)
            partition = community_louvain.best_partition(G, resolution=self.resolution)
            labels = np.array(list(partition.values()))
            cluster_centers = np.array([encoded_data[labels == i].mean(axis=0) for i in np.unique(labels)])
        
        elif self.clustering_method == 'leiden':
            # Leiden方法
            labels = leiden_clustering(encoded_data,  resolution=self.resolution)
            cluster_centers = np.array([encoded_data[labels == i].mean(axis=0) for i in np.unique(labels)])
        
        else:
            raise ValueError(f"Unsupported clustering method: {self.clustering_method}")
        
        return labels, cluster_centers

    @torch.no_grad()
    def init_kmeans_centers(self, z):
        encoded_data = z.cpu().numpy()
        # update the clustering centers
        if self.prior_y is not None:
            labels = self.prior_y
            cluster_centers = np.array([encoded_data[labels == i].mean(axis=0) for i in np.unique(labels)])
        else:
            labels, cluster_centers = self._apply_clustering(encoded_data)
        
        cluster_centers = torch.tensor(cluster_centers, device=self.device)
        num_clusters = len(np.unique(labels))

        # 如果聚类数量发生变化，需要重新初始化高斯分布
        if num_clusters != self.n_components:
            print(f"Number of clusters changed from {self.n_components} to {num_clusters}. Reinitializing Gaussian.")
            self.num_classes = num_clusters
            self.pi_ = nn.Parameter(torch.full((self.num_classes,), 1.0 / float(self.num_classes), dtype=torch.float64, device=self.device),requires_grad=True)
            self.c_mean = nn.Parameter(torch.zeros(self.num_classes, self.latent_dim, dtype=torch.float64, device=self.device),requires_grad=True)
            self.c_log_var = nn.Parameter(torch.zeros(self.num_classes, self.latent_dim, dtype=torch.float64, device=self.device),requires_grad=True)

        # 直接用聚类中心更新高斯分布参数
        self.c_mean.data.copy_(cluster_centers)
    
    def update_kmeans_centers(self, z):
        print(f'Update clustering centers..........')
        encoded_data_cpu = z.detach().cpu().numpy()
        gaussian_means = self.c_mean.detach().cpu().numpy()
        gaussian_var = self.c_log_var.detach().cpu().numpy()
        gaussian_pi = self.pi_.detach().cpu().numpy()

        # update the clustering centers
        if self.prior_y is not None:
            ml_labels = self.prior_y
            unique_labels, counts = np.unique(ml_labels, return_counts=True)
            num_ml_centers = len(unique_labels)
            aligned_centers = np.array([encoded_data_cpu[ml_labels == i].mean(axis=0) for i in unique_labels])
            aligned_var = np.array([np.log(encoded_data_cpu[ml_labels == i].std(axis=0).pow(2)) for i in unique_labels])
            aligned_pi = counts / len(ml_labels)
        else:
            ml_labels, cluster_centers = self._apply_clustering(encoded_data_cpu)
            num_ml_centers = len(np.unique(ml_labels))
            aligned_centers = self.optimal_transport(cluster_centers, gaussian_means, 1)
            aligned_var = self.change_var(gaussian_var,self.num_clusters,w=1.0)
            aligned_pi = self.change_pi(gaussian_pi,self.num_clusters,w=0.5)  

        """align"""       
        if num_ml_centers != self.num_classes:
            print(f"Number of clusters changed from {self.num_classes} to {num_ml_centers} .Reinitializing Gaussian.")
            self.num_clusters = num_ml_centers
            self.pi_ = nn.Parameter(torch.tensor(aligned_pi, dtype=torch.float64, device=self.device),requires_grad=True)
            self.c_mean = nn.Parameter(torch.tensor(aligned_centers, dtype=torch.float64, device=self.device),requires_grad=True)
            self.c_log_var = nn.Parameter(torch.tensor(aligned_var, dtype=torch.float64, device=self.device),requires_grad=True)
        else:
            self.c_mean.data.copy_(torch.tensor(aligned_centers,device = self.device))
            self.c_log_var.data.copy_(torch.tensor(aligned_var,device = self.device))
            self.pi_.data.copy_(torch.tensor(aligned_pi,device = self.device))

    def optimal_transport(self,A, B, alpha):
        n, d = A.shape
        m, _ = B.shape

        # 计算成本矩阵
        cost_matrix = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)  # 解指派问题

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

    def change_pi(self,array,n,w=1):
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
                match_spec = np.array(search_results.database['intensity'][match_id-1])
                match_chem = search_results.database['component'][match_id-1]
            match_specs.append(match_spec)
            match_chems.append(match_chem)

        match_specs = np.array(match_specs)
        return match_specs, match_chems


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
        z_mean, z_log_var, S= self.encoder(x)
        z = self.reparameterize(z_mean, z_log_var)

        # 解码
        # recon_x = self.decoder(z)
        recon_x = torch.matmul(z, S)
        
        return recon_x, z_mean,z_log_var, z, S

    @torch.no_grad()
    def constraint_angle(self, x, weight=0.05):
        """
        S: 要约束的矩阵（一组组分或浓度行/列），shape [n, dim]
        x: 原始输入矩阵z_
        """
        S_init = self.encoder.S
        m = x.mean(axis=0)  # 对列求均值
        m = m / (m.norm(p=2)+1e-8)  # 单位化
        S_init = S_init / (S_init.norm(p=2, dim=1, keepdim=True) + 1e-8)  # 行单位化
        return (1 - weight) * S_init + weight * m

    @torch.no_grad()
    def get_peak_positions(self):
        x = self.tensor_gpu_data.detach().cpu()
        peaks = torch.zeros_like(x) 
        x_normalized = (x - x.min(dim=-1, keepdim=True)[0]) / (x.max(dim=-1, keepdim=True)[0] - x.min(dim=-1, keepdim=True)[0] + 1e-5)

        for i in range(x_normalized.shape[0]):
            spectrum = x_normalized[i].detach().cpu().numpy()
            peak_indices, _ = signal.find_peaks(
                spectrum,
                height=0.1,
                distance=10,
                prominence=0.05,
            )
            peaks[i, peak_indices] = 1.0
        peak_counts = torch.sum(peaks, dim=0)
        min_occurrences = 0.1 * peaks.shape[0]  
        peak_positions = torch.where(peak_counts >= min_occurrences)[0]
        return peak_positions

    @torch.no_grad()
    def compute_spectral_constraints(self, x, recon_x):
        peak_positions = self.get_peak_positions()
        distance = torch.diff(peak_positions, prepend=peak_positions[:1], append=peak_positions[-1:])
        variances = torch.max(distance[:-1], distance[1:])

        weights = torch.zeros_like(x)
        for i, peak in enumerate(peak_positions):
            gamma = variances[i]  # 洛伦兹分布的半宽
            lorentzian = gamma**2 / ((torch.arange(x.shape[1], device=x.device) - peak)**2 + gamma**2)
            weights += lorentzian

        weighted_mse = F.mse_loss(recon_x, x, reduction='none') * weights

        print(f"weighted_MSE = {weighted_mse.shape}")

        return weighted_mse


    def compute_loss(self, x, recon_x, z_mean, z_log_var, gamma, S, matched_S,lamb1,lamb2,lamb3,lamb4,lamb5,lamb6,lamb7):
        zero = torch.tensor(0.0, device=self.device)
        # 1. 重构损失
        if lamb1 > 0:
            recon_loss = lamb1 * F.mse_loss(recon_x, x, reduction='none').sum(-1)
        else:
            recon_loss = zero.expand(x.size(0))

        # 2. GMM先验的KL散度       
        gamma_t = gamma.unsqueeze(-1)  # [batch_size, num_clusters, 1]
        z_mean = z_mean.unsqueeze(1)  # [batch_size, 1, latent_dim]
        z_log_var = z_log_var.unsqueeze(1)  # [batch_size, 1, latent_dim]
        
        gaussian_means = self.c_mean.unsqueeze(0)  # [1, n_clusters, latent_dim]
        gaussian_log_vars = self.c_log_var.unsqueeze(0)  # [1, n_clusters, latent_dim]
        
        if lamb2 > 0:
            log2pi = torch.log(torch.tensor(2.0*math.pi, device=x.device, dtype=x.dtype))
            kl_gmm = torch.sum(
                0.5 * gamma_t * (
                    self.latent_dim * log2pi + gaussian_log_vars +
                    torch.exp(z_log_var) / (torch.exp(gaussian_log_vars) + 1e-10) +
                    (z_mean - gaussian_means).pow(2) / (torch.exp(gaussian_log_vars) + 1e-10)
                ),
                dim=(1,2)
            ) * lamb2
        else:
            kl_gmm = zero.expand(z_mean.size(0))

        # 3. VAE的KL散度
        if lamb3 > 0:
            kl_VAE = (-0.5 * torch.sum(1 + z_log_var, dim=2))* lamb3  # - z_mean.pow(2) - torch.exp(z_log_var)
        else:
            kl_VAE = zero.expand(gamma.size(0))

        # 4. GMM熵项
        if lamb4 > 0:
            pi = self.pi_.unsqueeze(0)  # [1, n_clusters]
            entropy = lamb4 * (
                -torch.sum(torch.log(pi + 1e-10) * gamma, dim=-1) +
                torch.sum(torch.log(gamma + 1e-10) * gamma, dim=-1)
            )
        else:
            entropy = zero.expand(gamma.size(0))

        # 5. 峰加权损失，替换Recon_Loss
        if lamb5 > 0:
            spectral_constraints = lamb5 * self.compute_spectral_constraints(x, recon_x).sum(-1) * self.input_dim
        else:
            spectral_constraints = zero.expand(recon_x.size(0))

        # 6. Match Loss of S
        if lamb6 > 0:
            matched_comp = torch.tensor(matched_S, dtype=torch.float64, device = self.device)
            valid_idx = np.where((self.wavenumber <= 1800) & (self.wavenumber >= 450) )[0]
            S_valid = S[:,valid_idx]
            cos_sim = F.cosine_similarity(S_valid, matched_comp, dim=1) 
            match_loss_bioDB = lamb6 * (1 - cos_sim)
        else:
            match_loss_bioDB = zero.expand(S.size(0))

        # 7. Unsimilarity between S
        if lamb7 > 0:
            SS = torch.matmul(S, S.t())
            I = torch.eye(S.shape[0], device=self.device)
            ortho_loss = ((SS - I) ** 2).sum()
            unsimilarity_S = lamb7 * ortho_loss
        else:
            unsimilarity_S = zero.expand(S.size(0))

        # 总损失
        loss = recon_loss.mean() + kl_VAE.mean() + kl_gmm.mean() +  entropy.mean()  + match_loss_bioDB.mean() + spectral_constraints.mean() + unsimilarity_S.mean()
        
        # 返回损失字典
        loss_dict = {
            'total_loss': loss,
            'recon_loss': recon_loss.mean().item(),
            'kl_gmm': kl_gmm.mean().item(),
            'kl_VAE': kl_VAE.mean().item(),
            'entropy': entropy.mean().item(),
            'weighted_spectral': spectral_constraints.mean().item(),
            'match_loss': match_loss_bioDB.mean().item(),
            'unsimilarity_loss': unsimilarity_S.mean().item()
        }
        
        return loss_dict