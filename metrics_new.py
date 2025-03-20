import os
from typing import Optional, Tuple, Dict, Union, Callable
import numpy as np
import torch
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment
from utility import visualize_clusters, plot_reconstruction, generate_spectra_from_means
from config import config
from torch.utils.tensorboard import SummaryWriter
from utility import leiden_clustering,inverse_wavelet_transform
class ModelEvaluator:
    """
    模型评估器,用于计算模型在验证集或测试集上的各种指标,并保存结果。

    Args:
        model: 待评估的模型
        dataloader:数据加载
        device: 使用的设备,either 'cpu' or 'cuda'
        project_dir: 项目根目录路径,用于保存评估结果
        writer: TensorBoard日志记录器,用于记录评估指标

    Attributes:
        model: 待评估的模型
        device: 使用的设备
        project_dir: 项目根目录路径
        writer: TensorBoard日志记录器
        paths: 保存评估结果的路径
    """
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        paths: Optional[Dict] = None,
        writer: Optional[SummaryWriter] = None,
        resolution_2: float = 1.0
    ):
        self.model = model
        self.device = device
        self.paths = paths
        self.writer = writer
        self.resolution_2 = resolution_2
        if self.paths and not os.path.exists(os.path.join(self.paths['training_log'], "training_log.txt")):
            with open(os.path.join(self.paths['training_log'], "training_log.txt"), "w") as f:
                f.write("Epoch\tTotal_loss\tRecon_loss\tKL_GMM\tKL_Standard\tEntropy\t"
                        "Peak_loss\tSpectral_loss\t"
                        "gmm_acc\tgmm_nmi\tgmm_ari\t"
                        "z_leiden_acc\tz_leiden_nmi\tz_leiden_ari\t"
                        "Learning_Rate\t"
                        'SNR\n')

    def compute_reconstruction_metrics(
        self, x: torch.Tensor, recon_x: torch.Tensor
    ) -> Dict[str, float]:
        """
        计算重构相关指标。

        Args:
            x: 原始输入数据
            recon_x: 重构后的数据

        Returns:
            字典,包含 'mse', 'mae', 'max_error' 等重构质量相关指标
        """
        mse = ((x - recon_x) ** 2).mean().item()
        mae = (x - recon_x).abs().mean().item()
        max_error = (x - recon_x).abs().max().item()

        return {
            'mse': mse,
            'mae': mae,
            'max_error': max_error
        }

    def compute_clustering_metrics(
        self, y_pred: Union[torch.Tensor, np.ndarray], y_true: Optional[Union[torch.Tensor, np.ndarray]] = None
    ) -> Dict[str, float]:
        """
        计算聚类相关指标,包括准确率(ACC)、标准化互信息(NMI)和调整兰德系数(ARI)。

        Args:
            y_pred: 预测的聚类标签
            y_true: 真实的聚类标签,可选

        Returns:
            字典,包含 'acc', 'nmi', 'ari' 等聚类质量相关指标
        """
        # 如果没有真实标签,返回默认值
        if y_true is None:
            return {
                'acc': 0.0,
                'nmi': 0.0,
                'ari': 0.0
            }

        # 确保数据在CPU上并转换为numpy数组
        y_pred = self._to_numpy(y_pred)
        y_true = self._to_numpy(y_true)

        acc = self.calculate_acc(y_pred, y_true)
        nmi = normalized_mutual_info_score(y_pred, y_true)
        ari = adjusted_rand_score(y_pred, y_true)

        return {
            'acc': acc,
            'nmi': nmi,
            'ari': ari
        }

    @staticmethod
    def cal_SNR(X: np.ndarray) -> float:
        max_signals = np.max(X, axis=1)
        noise_region = X[:, 784:843]
        noise_means = np.mean(noise_region, axis=1)
        noise_stds = np.std(noise_region, axis=1, ddof=1) 
        snr_values = (max_signals - noise_means) / noise_stds
        valid_snr = snr_values[np.isfinite(snr_values)]
        return np.mean(valid_snr)
   
    @staticmethod
    def calculate_acc(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        计算聚类准确率。

        Args:
            y_pred: 预测标签
            y_true: 真实标签

        Returns:
            聚类准确率
        """
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
        tensor_gpu_data: torch.Tensor,
        labels: torch.Tensor,
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
            # 获取模型输出
            recon_x, mean, log_var, z, gamma, pi = self.model(tensor_gpu_data)
            
            # 转换数据到CPU
            z_cpu = z.detach().cpu().numpy()
            gmm_probs = gamma.detach().cpu().numpy()
            gmm_labels = np.argmax(gmm_probs, axis=1)
            recon_x_cpu = recon_x.detach().cpu().numpy()
            y_true = labels.cpu().numpy()
            mean_SNR = self.cal_SNR(recon_x_cpu)

            # 计算Leiden聚类标签
            # z_leiden_labels = leiden_clustering(z_cpu, resolution=self.resolution_2)

            # 计算评估指标
            gmm_metrics = self.compute_clustering_metrics(gmm_labels, y_true)
            # z_leiden_metrics = self.compute_clustering_metrics(z_leiden_labels, y_true)

            metrics = {
                'gmm_acc': gmm_metrics['acc'],
                'gmm_nmi': gmm_metrics['nmi'],
                'gmm_ari': gmm_metrics['ari'],
                # 'leiden_acc': z_leiden_metrics['acc'],
                # 'leiden_nmi': z_leiden_metrics['nmi'],
                # 'leiden_ari': z_leiden_metrics['ari']
            }

            # 解包训练指标
            train_metrics_names = ['total_loss', 'recon_loss', 'kl_gmm', 'kl_standard', 'entropy', 'peak_loss', 'spectral_loss', 'clustering_confidence_loss', 'cluster_separation_loss']
            train_metrics_dict = {name: train_metrics[name] for name in train_metrics_names}

            # 合并所有指标
            metrics.update(train_metrics_dict)
            metrics.update({'SNR': mean_SNR})

            self._save_log(epoch, metrics, lr)
            self._save_to_tensorboard(epoch, metrics)
            # self._print_metrics(epoch, lr, metrics)

            # 打印评估结果
            # if (epoch+1) % 10 == 0:
            #     self._save_results(
            #         epoch, 
            #         metrics, 
            #         lr,
            #         z_cpu, 
            #         recon_x_cpu, 
            #         y_true,
            #         gmm_labels,
            #         # z_leiden_labels,
            #         t_plot,
            #         r_plot
            #     )

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
            ('KL Standard', metrics.get('kl_standard', 0.0), '.2f'),
            ('Entropy', metrics.get('entropy', 0.0), '.2f'),
            ('Peak Loss', metrics.get('peak_loss', 0.0), '.2f'),
            ('Spectral Loss', metrics.get('spectral_loss', 0.0), 'f'),
            ('Clustering Confidence Loss', metrics.get('clustering_confidence_loss', 0.0), '.2f'),
            ('Cluster Separation Loss', metrics.get('cluster_separation_loss', 0.0), '.2f')
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

        # print(loss_str)
        # print(gmm_str)
        # print(z_leiden_str)

    def _save_results(
        self,
        epoch: int,
        metrics: Dict[str, float],
        lr: float,
        z_cpu: np.ndarray,
        recon_x_cpu: np.ndarray,
        labels:np.ndarray,
        gmm_labels: np.ndarray,
        leiden_labels: np.ndarray,
        t_plot: bool,
        r_plot: bool,
        wavenumber:Optional[np.ndarray] = None
    ) -> None:
        """
        保存评估结果,包括日志文件、TensorBoard 记录和可视化。

        Args:
            epoch: 当前 epoch 编号
            metrics: 评估指标字典
            z_cpu: 编码后的特征
            recon_x_cpu: 重构后的数据
            colors_map: 类别到颜色的映射字典
            labels: 标签数据
            # num_classes: 类别数量
            # unique_label: 一标签值数组
        """
        if not self.paths:
            print("Warning: No paths specified for saving results")
            return

        # # 记录到文件
        # self._save_log(epoch, metrics, lr)

        # # 记录到TensorBoard
        # self._save_to_tensorboard(epoch, metrics)

        # 保存t-SNE可视化
        self._save_tsne_plot(epoch, z_cpu, labels, gmm_labels, leiden_labels, t_plot)

        # 保存模型
        self._save_model(epoch, metrics)

        # 保存重构可视化
        self._save_recon_plot(epoch, recon_x_cpu, labels, wavenumber,r_plot)

    @staticmethod
    def _to_numpy(tensor: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """
        将输入转换为numpy数组。

        Args:
            tensor: 输入数据,可以是 PyTorch Tensor 或 numpy 数组

        Returns:
            numpy 数组
        """
        if torch.is_tensor(tensor):
            return tensor.detach().cpu().numpy()
        else:
            return np.asarray(tensor)

    def _save_log(self, epoch: int, metrics: Dict[str, float], lr: float) -> None:
        """
        将评估指标保存到日志文件。

        Args:
            epoch: 当前 epoch 编号
            metrics: 评估指标字典
        """
        try:
            with open(os.path.join(self.paths['training_log'], "training_log.txt"), "a") as f:
                f.write(
                    f'{epoch}\t{metrics["total_loss"]:.4f}\t{metrics["recon_loss"]:.4f}\t'
                    f'{metrics["kl_gmm"]:.4f}\t{metrics["kl_standard"]:.4f}\t{metrics["entropy"]:.4f}\t'
                    f'{metrics["peak_loss"]:.4f}\t{metrics["spectral_loss"]:.4f}\t'
                    f'{metrics["gmm_acc"]:.4f}\t{metrics["gmm_nmi"]:.4f}\t{metrics["gmm_ari"]:.4f}\t'
                    # f'{metrics["leiden_acc"]:.4f}\t{metrics["leiden_nmi"]:.4f}\t{metrics["leiden_ari"]:.4f}\t'
                    f'{lr:.4f}\t'
                    f'{metrics["SNR"]:.4f}\n'
                )
        except Exception as e:
            print(f"Error saving log file: {e}")

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
            self.writer.add_scalar('Recon/SNR', metrics['SNR'], epoch)
            
            # Leiden clustering metrics
            # self.writer.add_scalar('Leiden/ACC', metrics['leiden_acc'], epoch)
            # self.writer.add_scalar('Leiden/NMI', metrics['leiden_nmi'], epoch)
            # self.writer.add_scalar('Leiden/ARI', metrics['leiden_ari'], epoch)

    def _save_tsne_plot(
        self,
        epoch: int,
        z_cpu: np.ndarray,
        labels: np.ndarray,
        gmm_labels:np.ndarray,  
        leiden_labels:np.ndarray,
        plot: bool
    ) -> None:
        """
        保存 t-SNE 可视化。

        Args:
            epoch: 当前 epoch 编号
            z_cpu: 编码后的特征
            labels: 标签数据
            colors_map: 类别到颜色的映射字典
        """
        tsne_plot_path = os.path.join(self.paths['plot'], f'epoch_{epoch}_tsne_plot.png')
        tsne_txt_pth = os.path.join(self.paths['plot'],f'epoch_{epoch}_z_value.txt')
        np.savetxt(tsne_txt_pth,z_cpu)
        if plot :
            visualize_clusters(z=z_cpu,gaussian_centers=self.model.gaussian.means.data.cpu().numpy(),labels=labels, gmm_labels=gmm_labels, leiden_labels=leiden_labels, save_path=tsne_plot_path)

    def _save_model(self, epoch: int, metrics: Dict[str, float]) -> None:
        """
        保存模型权重。

        Args:
            epoch: 当前 epoch 编号
            metrics: 评估指标字典
        """
        model_path = os.path.join(
            self.paths['pth'],
            f'epoch_{epoch}_gmm_acc_{metrics["gmm_acc"]:.2f}_gmm_nmi_{metrics["gmm_nmi"]:.2f}_gmm_ari_{metrics["gmm_ari"]:.2f}.pth'
        )
        torch.save(self.model.state_dict(), model_path)

    def _save_recon_plot(
        self,
        epoch: int,
        recon_x_cpu: np.ndarray,
        labels: np.ndarray,
        wavenumber: np.ndarray,
        plot: bool
    ) -> None:
        """
        保存重构可视化。
        """
        recon_plot_path = os.path.join(self.paths['plot'], f'epoch_{epoch}_recon_plot.png')
        recon_txt_path = os.path.join(self.paths['plot'], f'epoch_{epoch}_recon_x_value.txt')
        np.savetxt(recon_txt_path, recon_x_cpu)
        
        if plot:
            plot_reconstruction(
                recon_data=recon_x_cpu,
                labels=labels,
                save_path=recon_plot_path,
                wavenumber=wavenumber
            )
    