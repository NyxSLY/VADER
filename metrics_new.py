import os
from typing import Optional, Dict, Union
import numpy as np
import torch
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment
from utility import plot_spectra, visualize_clusters,plot_S,plot_UMAP
from config import config
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from utility import leiden_clustering
import torch.nn.functional as F
import pandas as pd

class ModelEvaluator:
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        paths: Optional[Dict] = None,
        writer: Optional[SummaryWriter] = None,
        resolution: float = 1.0
    ):
        self.model = model
        self.device = device
        self.paths = paths
        self.writer = writer
        self.resolution = resolution

    def compute_clustering_metrics(
        self, y_pred: Union[torch.Tensor, np.ndarray], y_true: Optional[Union[torch.Tensor, np.ndarray]] = None
    ) -> Dict[str, float]:

        acc = self.calculate_acc(y_pred, y_true)
        nmi = normalized_mutual_info_score(y_pred, y_true)
        ari = adjusted_rand_score(y_pred, y_true)

        return {
            'acc': acc,
            'nmi': nmi,
            'ari': ari
        }

    @staticmethod
    def calculate_acc(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        assert y_pred.size == y_true.size
        D = int(max(y_pred.max(), y_true.max())) + 1
        w = np.zeros((D, D), dtype=np.int64)

        # 构建权重矩阵
        for i in range(y_pred.size):
            w[int(y_pred[i]), int(y_true[i])] += 1

        # 使用匈牙利算法找到最佳匹配
        ind = linear_sum_assignment(w.max() - w)
        ind = np.asarray(ind)
        ind = ind[:, w[ind[0], ind[1]] > 0]

        if ind.size == 0:
            return 0.0

        return float(w[ind[0], ind[1]].sum()) / y_pred.size

    def evaluate_epoch(
        self,
        recon_x,
        gmm_labels,
        z,
        labels: torch.Tensor,
        matched_S,
        matched_chem,
        epoch: int,
        lr: float,
        train_metrics: Dict[str, float],
        t_plot: bool,
        r_plot: bool
    ) -> Dict[str, float]:
        """
        评估一个 epoch 的结果,计算各种指标,并保存到文件和 TensorBoard。
        """
        
        self.model.eval()
        with torch.no_grad():
            # 转换数据到CPU
            z_cpu = z.detach().cpu().numpy()
            recon_x_cpu = recon_x.detach().cpu().numpy()
            y_true = labels.cpu().numpy()

            # 计算Leiden聚类标签
            z_leiden_labels = leiden_clustering(z_cpu, resolution=self.resolution)

            # 计算评估指标
            gmm_metrics = self.compute_clustering_metrics(gmm_labels, y_true)
            z_leiden_metrics = self.compute_clustering_metrics(z_leiden_labels, y_true)
            

            metrics = {
                'gmm_acc': gmm_metrics['acc'],
                'gmm_nmi': gmm_metrics['nmi'],
                'gmm_ari': gmm_metrics['ari'],
                'leiden_acc': z_leiden_metrics['acc'],
                'leiden_nmi': z_leiden_metrics['nmi'],
                'leiden_ari': z_leiden_metrics['ari']
            }

            # 解包训练指标
            train_metrics_names = ['total_loss', 'recon_loss', 'kl_gmm', 'kl_VAE', 'entropy', 'weighted_spectral', 'match_loss', 'unsimilarity_loss']
            train_metrics_dict = {name: train_metrics[name] for name in train_metrics_names}

            # 合并所有指标
            metrics.update(train_metrics_dict)

            self._save_to_tensorboard(epoch, metrics)
            self._print_metrics(epoch, lr, metrics)

            # 打印评估结果
            if (epoch+1) % config.get_model_params()['save_interval'] == 0:
                # ======== 保存Z ========
                # z_save_path = os.path.join(self.paths['plot'],f'epoch_{epoch}_z_value.txt')
                # np.savetxt(z_save_path,z_cpu)

                # ======== 保存gmm_labels ========
                # gmm_labels_path = os.path.join(self.paths['pth'], f'Epoch_{epoch+1}_gmm_labels.txt')
                # np.savetxt(gmm_labels_path, gmm_labels, fmt='%d')

                # ======== 保存S和匹配到的物质 ========
                S_plot_path = os.path.join(self.paths['plot'], f'epoch_{epoch}_spectra_comp.png')
                plot_S(self.model.encoder.S, matched_S, matched_chem, S_plot_path, self.model.wavenumber)
                np.savetxt(os.path.join(self.paths['plot'], f'S_values_epoch_{epoch+1}.txt'),  self.model.encoder.S.detach().cpu().numpy(), fmt='%.6f') 

                # ======== 保存t-SNE可视化 ========
                if t_plot :
                    UMAP_plot_path = os.path.join(self.paths['plot'], f'epoch_{epoch}_UMAP_plot.png')
                    visualize_clusters(z=z_cpu, labels=y_true, gmm_labels=gmm_labels, leiden_labels=z_leiden_labels,
                                        ari_gmm = metrics['gmm_ari'], ari_leiden = metrics['leiden_ari'], save_path=UMAP_plot_path)
                              
                # ======== 保存模型 ========
                # model_path = os.path.join(self.paths['pth'],f'epoch_{epoch}_gmm_acc_{metrics["gmm_acc"]:.2f}_gmm_nmi_{metrics["gmm_nmi"]:.2f}_gmm_ari_{metrics["gmm_ari"]:.2f}.pth')
                # torch.save(self.model.state_dict(), model_path)

                # ======== 保存重构可视化 ========
                if r_plot:
                    # 重构光谱
                    recon_plot_path = os.path.join(self.paths['plot'], f'epoch_{epoch}_recon_plot.png')
                    plot_spectra( recon_data=recon_x_cpu, labels=y_true, save_path=recon_plot_path, wavenumber=self.model.wavenumber)
                    # recon_txt_path = os.path.join(self.paths['plot'], f'epoch_{epoch}_recon_x_value.txt')
                    # np.savetxt(recon_txt_path, recon_x_cpu)
                    # 与原始数据的UMAP降维图
                    X = np.vstack([self.model.tensor_gpu_data.detach().cpu().numpy(), recon_x_cpu])
                    Y = np.array(['Raw'] * (X.shape[0]//2) + ['generate'] * (X.shape[0]//2))
                    plot_UMAP(X, Y, os.path.join(self.paths['plot'], f'epoch_{epoch}_recon_UMAP.png'))

        return metrics

    def _print_metrics(self, epoch: int, lr: float, metrics: Dict[str, float]) -> None:
        """
        打印评估指标。

        Args:
            epoch: 当前 epoch 编号
            lr: 学习率
            metrics: 评估指标字典
        """
        
        # 创建要打印的指标列表，匹配 vade_new.py 中的 loss_dict
        loss_items = [
            ('LR', lr, '.4f'),
            ('Total Loss', metrics.get('total_loss', 0.0), '.2f'),
            ('Recon Loss', metrics.get('recon_loss', 0.0), '.2f'),
            ('KL GMM', metrics.get('kl_gmm', 0.0), '.2f'),
            ('KL VAE', metrics.get('kl_VAE', 0.0), '.2f'),
            ('Entropy', metrics.get('entropy', 0.0), '.2f'),
            ('Weiggted Spectral', metrics.get('weighted_spectral', 0.0), '.2f'),
            ('Match Loss', metrics.get('match_loss', 0.0), 'f'),
            ('Unsimilarity Loss', metrics.get('unsimilarity_loss', 0.0), '.2f')
        ]

        gmm_items = [
            ('GMM ACC', metrics.get('gmm_acc', 0.0), '.4f'),
            ('GMM NMI', metrics.get('gmm_nmi', 0.0), '.4f'),
            ('GMM ARI', metrics.get('gmm_ari', 0.0), '.4f')
        ]
        z_leiden_items = [
            ('Z_Leiden ACC', metrics.get('leiden_acc', 0.0), '.4f'),
            ('Z_Leiden NMI', metrics.get('leiden_nmi', 0.0), '.4f'),
            ('Z_Leiden ARI', metrics.get('leiden_ari', 0.0), '.4f') 
        ]
        
        # 构建打印字符串
        loss_str = ', '.join([f'{name}: {value:{fmt}}' if fmt != 'd' else f'{name}: {value}' for name, value, fmt in loss_items])
        gmm_str = ', '.join([f'{name}: {value:{fmt}}' for name, value, fmt in gmm_items])
        z_leiden_str = ', '.join([f'{name}: {value:{fmt}}' for name, value, fmt in z_leiden_items])

        print(loss_str)
        print(gmm_str)
        print(z_leiden_str)


    def _save_to_tensorboard(self, epoch: int, metrics: Dict[str, float]) -> None:
        """
        将评估指标记录到 TensorBoard。

        Args:
            epoch: 当前 epoch 编号
            metrics: 评估指标字典
        """
        if self.writer:
            # GMM clustering metrics
            self.writer.add_scalar('GMM/ACC', metrics['gmm_acc'], epoch)
            self.writer.add_scalar('GMM/NMI', metrics['gmm_nmi'], epoch)
            self.writer.add_scalar('GMM/ARI', metrics['gmm_ari'], epoch)
            
            # Leiden clustering metrics
            self.writer.add_scalar('Leiden/ACC', metrics['leiden_acc'], epoch)
            self.writer.add_scalar('Leiden/NMI', metrics['leiden_nmi'], epoch)
            self.writer.add_scalar('Leiden/ARI', metrics['leiden_ari'], epoch)