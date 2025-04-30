from collections import defaultdict
from metrics_new import ModelEvaluator
import torch
import numpy as np
from config import config
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Queue
from contextlib import contextmanager
import time
import os
from itertools import chain
import sys
from utility import generate_spectra_from_means
# 添加这行来设置多进程启动方法
mp.set_start_method('spawn', force=True)

class WeightScheduler:
    def __init__(self, init_weights, max_weights, n_epochs, resolution_2):
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
        self.resolution_2 = resolution_2
        
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
    

def train_epoch(model, data_loader, optimizer_nn, optimizer_gmm, epoch, writer):
    """训练一个epoch"""
    model.train()
    total_metrics = defaultdict(float)
    
    for batch_idx, x in enumerate(data_loader):
        # 数据准备
        x = x[0].to(model.device)
        
        # 前向传播
        recon_x, mean, log_var, z, gamma, pi = model(x)
        
        # 获取GMM的输出
        gmm_means, gmm_log_variances, y, gamma, pi = model.gaussian(z)
        
        # 损失计算
        loss_dict = model.compute_loss(x, recon_x, mean, log_var, z, y)

        
        # 反向传播
        optimizer_nn.zero_grad()
        optimizer_gmm.zero_grad()
        loss_dict['total_loss'].backward()
        optimizer_nn.step()
        optimizer_gmm.step()
        
        # 更新总指标
        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                total_metrics[key] += value.item()
            else:
                total_metrics[key] += value
            
        # 记录到tensorboard
        if writer is not None and batch_idx % 10 == 0:
            step = epoch * len(data_loader) + batch_idx
            for key, value in total_metrics.items():
                writer.add_scalar(
                    f'Batch/{key}', 
                    value / (batch_idx + 1), 
                    step
                )
            
                
    # 计算平均指标
    for key in total_metrics:
        total_metrics[key] /= len(data_loader)
        
    return total_metrics


def train_manager(model, dataloader, tensor_gpu_data, labels, num_classes, paths):
    """管理整个训练流程"""
    # 初始化配置和组件
    train_config = config.get_train_config()
    model_params = config.get_model_params()
    vis_config = config.get_vis_config()
    weight_config = config.get_weight_scheduler_config()
    t_plot = train_config['tsne_plot']
    r_plot = train_config['recon_plot']

    # 初始化优化器和其他组件
    # optimizer = optim.Adam(model.parameters(), lr=model_params['learning_rate'])
    optimizer_nn = optim.Adam(chain(model.encoder.parameters(), model.decoder.parameters()), lr=model.learning_rate)
    optimizer_gmm = optim.Adam(model.gaussian.parameters(), lr=model.learning_rate)
    
    if model.use_lr_scheduler:
        print("使用学习率调度器")
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer,
        #     T_max=model_params['epochs'],
        #     eta_min=model_params['learning_rate'] * 0.01
        # )
        scheduler_nn = optim.lr_scheduler.CosineAnnealingLR(
            optimizer_nn,
            T_max=model.epochs,
            eta_min=model.learning_rate * 0.01
        )
        scheduler_gmm = optim.lr_scheduler.CosineAnnealingLR(
            optimizer_gmm,
            T_max=model.epochs,
            eta_min=model.learning_rate * 0.01
        )
    else:
        print("使用固定学习率")
        scheduler_nn = None
        scheduler_gmm = None
        
    device = model.device
    model_name = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
    writer = SummaryWriter(log_dir=paths['tensorboard_log'])
    evaluator = ModelEvaluator(
        model=model,
        device=device,
        paths=paths,
        writer=writer,
        resolution_2=model_params['resolution_2']
    )

    # 初始化权重调度器
    weight_scheduler = WeightScheduler(
        init_weights=weight_config['init_weights'],
        max_weights=weight_config['max_weights'],
        n_epochs=model.epochs,
        resolution_2=model_params['resolution_2']
    )
    model.init_kmeans_centers(dataloader)

    # 在训练开始前分析数据集
    # print("正在分析数据集特征...")
    model.spectral_analyzer.analyze_dataset(dataloader)
    
    # 可选：保存分析结果
    # model.spectral_analyzer.save_analysis_results()
    
    for epoch in range(train_config['start_epoch'], 
                      train_config['start_epoch'] + model.epochs):
        # 更新权重
        weights = weight_scheduler.get_weights(epoch)
        # print(weights)
        for key, value in weights.items():
            setattr(model, key, value)
        
        # 训练一个epoch
        train_metrics = train_epoch(
            model=model,
            data_loader=dataloader,
            optimizer_nn=optimizer_nn,
            optimizer_gmm=optimizer_gmm,
            epoch=epoch,
            writer=writer,
        )
        
        # 添加进度打印
        # print(f"\nEpoch [{epoch+1}/{model_params['epochs']}]")

        recon_x, mean, log_var, z, gamma, pi = model(tensor_gpu_data)
        gmm_probs = gamma.detach().cpu().numpy()
        
        # skip update kmeans centers
        if (epoch + 1) % 100 == 0:
            model.update_kmeans_centers()
            evaluator._save_results(epoch,train_metrics,0.0001,z.cpu().detach().numpy(),recon_x.cpu().detach().numpy(),labels,np.argmax(gmm_probs, axis=1),np.argmax(gmm_probs, axis=1),False,False)
            # generated_samples, generated_labels = generate_spectra_from_means(gmm_means.detach().cpu(), model,num_samples_per_label=int(len(labels)/gmm_means.shape[0]), noise_level=0.01)
            # gene_txt_path = os.path.join(paths['plot'], f'epoch_{epoch+1}_generate_x_value.txt')
            # np.savetxt(gene_txt_path, generated_samples)
            # gene_label_path = os.path.join(paths['plot'], f'epoch_{epoch+1}_generate_y_value.txt')
            # np.savetxt(gene_label_path, generated_labels)
            
        # 更新学习率
        lr_nn = model_params['learning_rate'] if scheduler_nn is None else scheduler_nn.get_last_lr()[0]
        lr_gmm = model_params['learning_rate'] if scheduler_gmm is None else scheduler_gmm.get_last_lr()[0]
        if scheduler_nn is not None:
            scheduler_nn.step()
        if scheduler_gmm is not None:
            scheduler_gmm.step()

        # 记录学习率
        if writer is not None:
            writer.add_scalar('Learning_rate_nn', lr_nn, epoch)
            writer.add_scalar('Learning_rate_gmm', lr_gmm, epoch)

            gmm_labels = np.argmax(gmm_probs, axis=1)
            unique_labels, counts = np.unique(gmm_labels, return_counts=True)
            proportions = counts / len(gmm_labels)
            writer.add_scalar('GMM/number_of_clusters', len(unique_labels), epoch)
        
        # 同步评估
        metrics = evaluator.evaluate_epoch(
            recon_x,
            z,
            gamma,
            labels, 
            epoch, 
            lr_nn, 
            train_metrics, 
            t_plot, 
            r_plot
        )
            
        # 检查早停条件
        if check_early_stopping(metrics, train_config['min_loss_threshold']):
            print(f'达到最小损失阈值,提前停止训练。总损失:{metrics["total_loss"]:.6f}, '
                  f'重建损失:{metrics["recon_loss"]:.6f}')
            return model
    
    return model

def check_early_stopping(metrics, thresholds):
    """检查是否需要早停"""
    return (metrics.get('total_loss') is not None and 
            metrics.get('recon_loss') is not None and 
            metrics['total_loss'] < thresholds['total'] and 
            metrics['recon_loss'] < thresholds['recon'])
