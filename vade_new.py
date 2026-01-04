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
from ramanbiolib.search import SpectraSimilaritySearch

from gmmot import GW2

def get_batch_gmm_stats(z, y_true, num_classes):
    mask = torch.nn.functional.one_hot(y_true, num_classes=num_classes).float()
    counts = mask.sum(dim=0)  # [num_classes]
    safe_counts = counts.unsqueeze(1) + 1e-9
    m_t = torch.matmul(mask.t(), z) / safe_counts
    
    m_t_sq = torch.matmul(mask.t(), z**2) / safe_counts
    v_t = m_t_sq - m_t**2
    v_t = torch.clamp(v_t, min=1e-6) # 数值保护
    
    w_t = counts / counts.sum()

    active_mask = counts > 1 # 至少 2 个样本才算方差
    return m_t[active_mask], v_t[active_mask], w_t[active_mask], active_mask

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
        concentration = F.softplus(self.to_concentration(x)) # 浓度均值
        concentration_logvar = self.to_concentration_logvar(x)  # 浓度方差
        S = F.relu(self.S) # 非负
        return concentration, concentration_logvar, S


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
        else:
            self.num_classes = num_classes # Use the provided num_classes if no prior_y

        self.cluster_centers = None
        self.pretrain_epochs = pretrain_epochs
        self.num_classes = num_classes
        self.max_clusters = 50
        self.activate_clusters = torch.arange(self.num_classes, device=self.device, dtype=torch.long)
        self.resolution = resolution
        self.clustering_method = clustering_method
        self.input_dim = input_dim

        self.encoder = Encoder(input_dim, intermediate_dim=intermediate_dim, latent_dim=latent_dim, n_components=n_components, S=S)
        # self.decoder = Decoder(latent_dim, intermediate_dim, input_dim, num_classes)
        self.pi_ = nn.Parameter(torch.full((self.max_clusters,), 1.0 / float(self.max_clusters), dtype=torch.float64, device=self.device),requires_grad=True)
        self.c_mean = nn.Parameter(torch.zeros(self.max_clusters, self.latent_dim, dtype=torch.float64, device=self.device),requires_grad=True)
        self.c_log_var = nn.Parameter(torch.zeros_like(self.c_mean),requires_grad=True)
       
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
        idx = self.activate_clusters
        z_expanded = z.unsqueeze(1)  # [batch_size, 1, latent_dim]
        means_expanded = F.softplus(self.c_mean[idx]).unsqueeze(0)  # [1, num_clusters, latent_dim]
        log_vars_expanded = self.c_log_var[idx].unsqueeze(0)  # [1, num_clusters, latent_dim]
        pi_expanded = self.pi_[idx].view(1, -1, 1).expand(z.shape[0], -1, z.shape[1])  # [1, num_clusters, 1]
    
        gamma = torch.sum(
            torch.log(pi_expanded)                   # 混合权重项
            - 0.5 * torch.log(2*math.pi*torch.exp(log_vars_expanded))  # 常数项和方差项
            - (z_expanded - means_expanded).pow(2)/(2*torch.exp(log_vars_expanded)), dim=2)  # 指数项

        return F.softmax(gamma,dim=1)

    def cal_desc_gamma(self, z, alpha=1.0):
        """
        根据 DESC 论文使用 Student's T-distribution 核计算软分配 Q (即 gamma)
        z: 编码后的潜在特征 (N, latent_dim)
        alpha: T 分布的自由度 (通常设为 1.0)
        """
        idx = self.activate_clusters
        
        # 只需要中心 c_mean，不需要 c_log_var
        means = F.softplus(self.c_mean[idx]) # [num_active_clusters, latent_dim]
        
        # 计算 z 和所有中心之间的距离平方 (N, K)
        # ||z_i - mu_j||^2
        z_expanded = z.unsqueeze(1)    # (N, 1, latent_dim)
        means_expanded = means.unsqueeze(0) # (1, K, latent_dim)
        
        # 计算欧氏距离平方
        dist_sq = torch.sum((z_expanded - means_expanded).pow(2), dim=2) # (N, K)

        # 计算 T-Kernel 相似度 Q_hat (未归一化)
        # Q_hat_ij = (1 + ||z_i - mu_j||^2 / alpha) ^ (-(alpha + 1) / 2)
        Q_hat = torch.pow((1.0 + dist_sq / alpha), -(alpha + 1.0) / 2.0) # (N, K)
        
        # 归一化，得到软分配 Q (即 gamma)
        gamma = Q_hat / torch.sum(Q_hat, dim=1, keepdim=True)
        
        return gamma # (N, K)
    
    @torch.no_grad()
    def cal_target_distribution(self, gamma):
        """
        根据当前的软分配 Q (gamma) 计算目标分布 P
        gamma: (N, K)
        """
        # 1. 计算 Q_ij^2 / sum_i Q_ij
        Q_sq = gamma.pow(2) # (N, K)
        f_j = torch.sum(gamma, dim=0) # sum_i Q_ij (K,)
        Q_norm = Q_sq / f_j # (N, K)
        
        # 2. 归一化
        sum_Q_norm = torch.sum(Q_norm, dim=1, keepdim=True) # sum_k (N, 1)
        
        # P_ij
        P = Q_norm / sum_Q_norm
        return P


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
        if self.prior_y is not None:
            labels = self.prior_y
            unique_labels = np.unique(labels) 
            self.label_map = {
                int(label): i for i, label in enumerate(unique_labels)
            }
            indexed_labels = np.array([self.label_map[int(l)] for l in labels])
            cluster_centers = np.array([encoded_data[indexed_labels == i].mean(axis=0) for i in range(len(unique_labels))])
        else:
            labels, cluster_centers = self._apply_clustering(encoded_data)
        
        cluster_centers = torch.tensor(cluster_centers, device=self.device, dtype=self.c_mean.dtype,)
        num_clusters = len(np.unique(labels))
        # 更新高斯聚类参数
        self.num_classes = num_clusters
        new_idx = torch.arange(num_clusters, device=self.device, dtype=torch.long)
        new_mean = self.c_mean.detach().clone()
        new_mean[new_idx] = cluster_centers
        self.activate_clusters = new_idx

        self.c_mean.data.copy_(new_mean)
    

    def optimal_transport(self,A, B, alpha):
        n, d = A.shape
        m, _ = B.shape

        if n > m:
            raise ValueError(f"Number of new centers ({n}) exceeds max clusters ({m}).")

        cost_matrix = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        aligned_B = B.copy()
        matched_indices = col_ind.copy()
        for a_idx, b_idx in zip(row_ind, col_ind):
            aligned_B[b_idx] = alpha * A[a_idx] + (1 - alpha) * B[b_idx]

        print(col_ind)
        return matched_indices, aligned_B
    
    def change_var_pi(self,var,pi,idx,w=1):
        if isinstance(idx, set):
            idx = np.array(sorted(idx), dtype=int)
        current_var = var[idx]
        current_pi = pi[idx]
        new_mask = np.isclose(current_pi, 1.0 / self.max_clusters)
        old_mask = ~new_mask
        if new_mask.any() and old_mask.any():
            mean_var = current_var[old_mask].mean(axis=0) * w
            mean_pi = current_pi[old_mask].mean() * w
            current_var[new_mask] = mean_var
            current_pi[new_mask] = mean_pi
        var[idx] = current_var
        pi[idx] = current_pi
        return var, pi

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

    def forward(self,x): # Add labels_batch parameter
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


    def compute_loss(self, x, y, recon_x, z_mean, z_log_var, gamma, S, matched_S,P,gamma_desc,
                     lamb1,lamb2,lamb3,lamb4,lamb5,lamb6,lamb7):
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
        
        idx = self.activate_clusters
        gaussian_means = F.softplus(self.c_mean[idx]).unsqueeze(0)  # [1, n_clusters, latent_dim]
        gaussian_log_vars = self.c_log_var[idx].unsqueeze(0)  # [1, n_clusters, latent_dim]
        
        if lamb2 > 0:
            log2pi = torch.log(torch.tensor(2.0*math.pi, device=x.device, dtype=x.dtype))
            # kl_distance_matrix = torch.sum(
            #     0.5 * gamma_t * (
            #         self.latent_dim * log2pi + gaussian_log_vars +
            #         torch.exp(z_log_var) / (torch.exp(gaussian_log_vars) + 1e-10) +
            #         (z_mean - gaussian_means).pow(2) / (torch.exp(gaussian_log_vars) + 1e-10)
            #     ),
            #     dim=-1
            # ) * lamb2
            kl_distance_matrix = torch.sum(
                0.5 * (
                    self.latent_dim * log2pi + gaussian_log_vars +
                    torch.exp(z_log_var) / (torch.exp(gaussian_log_vars) + 1e-10) +
                    (z_mean - gaussian_means) .pow(2) / (torch.exp(gaussian_log_vars) + 1e-10) - 
                    (1 + z_log_var)
                ),
                dim=(1,2)
            ) * lamb2 / gamma.shape[0]
            # OT Loss
            z = self.reparameterize(z_mean[:,0,:], z_log_var[:,0,:])
            gamma = self.cal_gaussian_gamma(z)
            with torch.no_grad():
                m_t, C_t, w_t, active_mask = get_batch_gmm_stats(z, y.long(), 30)
            m_s = gaussian_means[0,active_mask,:]
            C_s = torch.diag_embed(torch.exp(gaussian_log_vars[0,active_mask,:])+1e-4)
            w_s = self.pi_[idx][active_mask]
            w_s = w_s / w_s.sum()
            GW_loss = GW2(w_s,w_t.detach().double(), m_s,m_t.detach().double(), C_s, C_t.detach().double())
            kl_gmm = GW_loss*100 + kl_distance_matrix
           
            # if self.prior_y is not None:
            #     y_true = torch.tensor(np.array([self.label_map[int(label)] for label in y.cpu().numpy()], dtype=np.int64), device=self.device).long()
            #     chosen_kl = kl_distance_matrix.gather(1, y_true.unsqueeze(1)).squeeze()
            #     log_pi_chosen = torch.log(self.pi_[y_true] + 1e-10)
            #     kl_gmm = chosen_kl - log_pi_chosen
        else:
            kl_gmm = zero.expand(z_mean.size(0))

        # 3. VAE的正则化
        if lamb3 > 0 and self.prior_y is None:
            kl_VAE = (-0.5 * torch.sum(1 + z_log_var, dim=2))* lamb3  # - z_mean.pow(2) - torch.exp(z_log_var)
        else:
            kl_VAE = zero.expand(gamma.size(0))

        # 4. GMM熵项
        if lamb4 > 0 and self.prior_y is None:
            pi = self.pi_[idx].unsqueeze(0)  # [1, n_clusters]
            entropy = lamb4 * (
                -torch.sum(torch.log(pi + 1e-10) * gamma, dim=-1) +
                torch.sum(torch.log(gamma + 1e-10) * gamma, dim=-1)
            )
        else:
            entropy = zero.expand(gamma.size(0))

        # 5. 峰加权损失，替换Recon_Loss
        # if lamb5 > 0:
        #     spectral_constraints = lamb5 * self.compute_spectral_constraints(x, recon_x).sum(-1) * self.input_dim
        # else:
        #     spectral_constraints = zero.expand(recon_x.size(0))

        # 5.1. DESC KL Loss
        # weight = gamma ** 2 / gamma.sum(0)
        # p = (weight.T / weight.sum(1)).T
        # sum_p_norm = torch.sum(p, dim=1, keepdim=True) # sum_k (N, 1)
        
        # # P_ij
        # p_norm = p / sum_p_norm
        
        # # 计算p和gamma(q)之间的KL散度
        if lamb5 > 0:
            spectral_constraints = torch.sum(P * torch.log(P / (gamma_desc + 1e-10) + 1e-10), dim=1) * lamb5
        else:
            spectral_constraints = zero.expand(recon_x.size(0))
        
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
