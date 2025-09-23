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
# 添加这行来设置多进程启动方法
mp.set_start_method('spawn', force=True)

class WeightScheduler:
    def __init__(self, init_weights, max_weights, n_epochs, resolution):
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
        self.resolution = resolution
        
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
    

def train_epoch(model, data_loader, optimizer_nn, optimizer_gmm, epoch, writer, matched_S):
    """训练一个epoch"""
    model.train()
    total_metrics = defaultdict(float)
    
    for batch_idx, x in enumerate(data_loader):
        # 数据准备
        data_x = x[0].to(model.device)
        
        # 前向传播   
        recon_x, mean, gaussian_means, log_var, z, gamma, pi, S = model(data_x,  labels_batch = None if model.prior_y is None else x[1].to(model.device))
        
        # 获取GMM的输出
        # gmm_means, gmm_log_variances, y, gamma, pi = model.gaussian(z, labels_batch = None if model.prior_y is None else x[1].to(model.device))
        
        # 损失计算

        loss_dict = model.compute_loss(data_x, recon_x, mean, log_var, z, gamma, S, matched_S)

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


def train_manager(model, dataloader, tensor_gpu_data, labels, num_classes, paths, epochs):
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
    optimizer_nn = optim.Adam(chain(model.encoder.parameters()), lr=model_params['learning_rate'])
    optimizer_gmm = optim.Adam(model.gaussian.parameters(), lr=model_params['learning_rate'])
    
    if model_params.get('use_lr_scheduler', False):
        print("使用学习率调度器")
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer,
        #     T_max=model_params['epochs'],
        #     eta_min=model_params['learning_rate'] * 0.01
        # )
        scheduler_nn = optim.lr_scheduler.CosineAnnealingLR(
            optimizer_nn,
            T_max=epochs,
            eta_min=model_params['learning_rate'] * 0.01
        )
        scheduler_gmm = optim.lr_scheduler.CosineAnnealingLR(
            optimizer_gmm,
            T_max=epochs,
            eta_min=model_params['learning_rate'] * 0.01
        )
    else:
        print("使用固定学习率")
        scheduler_nn = None
        scheduler_gmm = None
        
    device = model.device
    model_name = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
    writer = SummaryWriter(log_dir=paths['tensorboard_log']+f'/{model_name}')
    evaluator = ModelEvaluator(
        model=model,
        device=device,
        paths=paths,
        writer=writer,
        resolution=model_params['resolution']
    )

    # 初始化权重调度器
    weight_scheduler = WeightScheduler(
        init_weights=weight_config['init_weights'],
        max_weights=weight_config['max_weights'],
        n_epochs=epochs,
        resolution=model_params['resolution']
    )
    recon_x, mean, gaussian_means, log_var, z, gamma, pi, S = model(tensor_gpu_data,  labels_batch = None if model.prior_y is None else labels.to(model.device))
    model.init_kmeans_centers(z)

    # 初始化 best_gmm_acc 和 best_epoch
    best_leiden_acc = -1.0
    best_epoch = -1

    for epoch in range(train_config['start_epoch'], 
                      train_config['start_epoch'] + epochs):
        # 更新权重
        weights = weight_scheduler.get_weights(epoch)
        for key, value in weights.items():
            setattr(model, key, value)
        
        # 训练一个epoch
        recon_x, mean, gaussian_means, log_var, z, gamma, pi, S = model(tensor_gpu_data,  labels_batch = None if model.prior_y is None else labels.to(model.device))
        matched_comp, matched_chems = model.match_components(S,0.7)

        train_metrics = train_epoch(
            model=model,
            data_loader=dataloader,
            optimizer_nn=optimizer_nn,
            optimizer_gmm=optimizer_gmm,
            epoch=epoch,
            writer=writer,
            matched_S = matched_comp
        )

        print(S[:,1])
        
        # model.constraint_angle(tensor_gpu_data, weight=0.05) # 角度约束，保证峰形
        # gmm_means, gmm_log_variances, y, gamma, pi = model.gaussian(z)
        # 添加进度打印
        print(f"\nEpoch [{epoch+1}/{epochs}]")
        
        # skip update kmeans centers
        if (epoch + 1) % 10 == 0:
            model.update_kmeans_centers(z)

        if (epoch + 1) % 50 == 0:
           np.savetxt(os.path.join(paths['plot'], f'S_values_epoch_{epoch+1}.txt'),  S.detach().cpu().numpy(), fmt='%.6f') 

        # 保存物质匹配情况
        if not os.path.exists(os.path.join(paths['training_log'], "matched_chems.txt")):
            with open(os.path.join(paths['training_log'], "matched_chems.txt"), "w") as f:
                f.write("Epoch\tMatched_Chems\n")  
        
        with open(os.path.join(paths['training_log'], "matched_chems.txt"), "a") as f:
            f.write(f'{epoch+1}\t{ matched_chems}\n')
            
        # 更新学习率
        lr_nn = model_params['learning_rate'] if scheduler_nn is None else scheduler_nn.get_last_lr()[0]
        lr_gmm = model_params['learning_rate'] if scheduler_gmm is None else scheduler_gmm.get_last_lr()[0]
        if scheduler_nn is not None:
            scheduler_nn.step()
        if scheduler_gmm is not None:
            scheduler_gmm.step()

        
        if writer is not None:
            writer.add_scalar('Learning_rate_nn', lr_nn, epoch)
            writer.add_scalar('Learning_rate_gmm', lr_gmm, epoch)

            gmm_probs = gamma.detach().cpu().numpy()
            gmm_labels = np.argmax(gmm_probs, axis=1)
            unique_labels, counts = np.unique(gmm_labels, return_counts=True)
            proportions = counts / len(gmm_labels)
            writer.add_scalar('GMM/number_of_clusters', len(unique_labels), epoch)
        
        # 同步评估
        metrics = evaluator.evaluate_epoch(
            recon_x,
            gamma,
            z,
            labels, 
            epoch, 
            lr_nn, 
            train_metrics, 
            t_plot, 
            r_plot
        )

        # 在训练了100个epoch以后，总是记录gmm_acc最大的epoch的pth
        if epoch >= 100:
            current_leiden_acc = metrics['leiden_acc']
            if current_leiden_acc > best_leiden_acc:
                best_leiden_acc = current_leiden_acc
                best_epoch = epoch
                print(f"新的最佳GMM准确率: {best_leiden_acc:.4f} 在 epoch {best_epoch + 1}")
                torch.save(model.state_dict(), os.path.join(paths['training_log'], f'Epoch_{best_epoch + 1}_Acc={best_leiden_acc:.2f}_model.pth'))
                print(f"模型已保存到 {os.path.join(paths['training_log'], f'Epoch_{best_epoch + 1}_Acc={best_leiden_acc:.2f}_model.pth')}")
            
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
