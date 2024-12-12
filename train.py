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

# 添加这行来设置多进程启动方法
mp.set_start_method('spawn', force=True)

class WeightScheduler:
    def __init__(self, init_weights, max_weights, n_epochs):
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
    

def train_epoch(model, data_loader, optimizer, epoch, writer):
    """训练一个epoch"""
    model.train()
    total_metrics = defaultdict(float)
    
    for batch_idx, x in enumerate(data_loader):
        # 数据准备
        x = x[0].to(model.device)
        
        # 前向传播
        recon_x, mean, log_var, z, z_prior_mean, y_pred = model(x)
        
        # 损失计算
        loss_dict = model.compute_loss(x, recon_x, mean, log_var, z_prior_mean, y_pred)
        
        # 反向传播
        optimizer.zero_grad()
        loss_dict['total_loss'].backward()
        optimizer.step()
        
        # 更新总指标
        for key, value in loss_dict.items():
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
    colors_map = config.get_color_map(num_classes)
    t_plot = train_config['tsne_plot']
    r_plot = train_config['recon_plot']

    # 初始化优化器和其他组件
    optimizer = optim.Adam(model.parameters(), lr=model_params['learning_rate'])
    
    if model_params.get('use_lr_scheduler', False):
        print("使用学习率调度器")
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=model_params['epochs'],
            eta_min=model_params['learning_rate'] * 0.01
        )
    else:
        print("使用固定学习率")
        scheduler = None
        
    device = model.device
    writer = SummaryWriter(paths['tensorboard_log'])
    evaluator = ModelEvaluator(
        model=model,
        device=device,
        paths=paths,
        writer=writer
    )

    # 初始化权重调度器
    weight_scheduler = WeightScheduler(
        init_weights=weight_config['init_weights'],
        max_weights=weight_config['max_weights'],
        n_epochs=model_params['epochs']
    )
    model.init_kmeans_centers(dataloader)

    # 在训练开始前分析数据集
    print("正在分析数据集特征...")
    model.spectral_analyzer.analyze_dataset(dataloader)
    
    # 可选：保存分析结果
    model.spectral_analyzer.save_analysis_results()
    
    for epoch in range(train_config['start_epoch'], 
                      train_config['start_epoch'] + model_params['epochs']):
        # 更新权重
        weights = weight_scheduler.get_weights(epoch)
        for key, value in weights.items():
            setattr(model, key, value)
        
        # 训练一个epoch
        train_metrics = train_epoch(
            model=model,
            data_loader=dataloader,
            optimizer=optimizer,
            epoch=epoch,
            writer=writer,
        )
        
        # 添加进度打印
        print(f"\nEpoch [{epoch+1}/{model_params['epochs']}]")
        
        if (epoch + 1) % 10 == 0:
            model.update_kmeans_centers(dataloader)
            
        # 更新学习率
        lr = model_params['learning_rate'] if scheduler is None else scheduler.get_last_lr()[0]
        if scheduler is not None:
            scheduler.step()

        # 记录学习率
        if writer is not None:
            writer.add_scalar('Learning_rate', lr, epoch)
        
        # 同步评估（每隔一定间隔进行）
        if epoch % train_config['save_interval'] == 0:
            metrics = evaluator.evaluate_epoch(
                tensor_gpu_data, 
                labels, 
                epoch, 
                colors_map,
                lr, 
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