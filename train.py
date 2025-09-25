from collections import defaultdict
from metrics_new import ModelEvaluator
from utility import WeightScheduler
import torch
import numpy as np
from config import config
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
from contextlib import contextmanager
import os
from itertools import chain

# 添加这行来设置多进程启动方法
mp.set_start_method('spawn', force=True)

def train_epoch(model, weights, data_loader, optimizer, epoch, writer, matched_S):
    """训练一个epoch"""
    model.train()
    total_metrics = defaultdict(float)
    
    for batch_idx, x in enumerate(data_loader):
        # 数据准备
        data_x = x[0].to(model.device)
        
        # 前向传播   
        recon_x, z_mean,z_log_var, z,  S = model(data_x,  labels_batch = None if model.prior_y is None else x[1].to(model.device))
        gamma = torch.exp(torch.log(model.pi_.unsqueeze(0)) + model.gaussian_pdfs_log(z, model.c_mean, model.c_log_var)) + 1e-10
        gamma=gamma/(gamma.sum(1).view(-1,1))
        
        # 损失计算

        loss_dict = model.compute_loss(data_x, recon_x, z_mean, z_log_var, gamma, S, matched_S,
                                       weights['lamb1'], weights['lamb2'], weights['lamb3'], weights['lamb4'],
                                       weights['lamb5'], weights['lamb6'], weights['lamb7'])

        # 反向传播
        optimizer.zero_grad()
        loss_dict['total_loss'].backward()
        optimizer.step()
        
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
                writer.add_scalar( f'Batch/{key}',  value / (batch_idx + 1), step )
            
                
    # 计算平均指标
    for key in total_metrics:
        total_metrics[key] /= len(data_loader)
        
    return total_metrics


def train_manager(model, dataloader, tensor_gpu_data, labels, paths, epochs):
    """管理整个训练流程"""
    # 初始化配置和组件
    model_params = config.get_model_params() 
    weight_config = config.get_weight_scheduler_config()
    t_plot = model_params['tsne_plot']
    r_plot = model_params['recon_plot']
    optimizer = optim.Adam(model.parameters(), lr=model_params['learning_rate'])

    if model_params.get('use_lr_scheduler', False):
        print("使用学习率调度器")
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=model_params['learning_rate'] * 0.01
        )
    else:
        print("使用固定学习率")
        scheduler = None
        
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
    recon_x, z_mean, z_log_var, z, S = model(tensor_gpu_data,  labels_batch = None if model.prior_y is None else labels.to(model.device))
    model.init_kmeans_centers(z)

    for epoch in range(0, epochs):
        print(f"\nEpoch [{epoch+1}/{epochs}]")
        # 更新权重
        weights = weight_scheduler.get_weights(epoch)

        # 训练一个epoch
        recon_x, z_mean, z_log_var, z, S = model(tensor_gpu_data,   labels_batch = None if model.prior_y is None else labels.to(model.device))
        matched_comp, matched_chems = model.match_components(S,0.7)
        print(model.c_mean[:,1])
        print(model.c_log_var[:,1])
        print(model.pi_)

        train_metrics = train_epoch(
            model=model, weights=weights,
            data_loader=dataloader,
            optimizer=optimizer,
            epoch=epoch,
            writer=writer,
            matched_S = matched_comp
        )
        
        # model.constraint_angle(tensor_gpu_data, weight=0.05) # 角度约束，保证峰形
        
        # skip update kmeans centers
        # if (epoch + 1) % model_params['update_interval'] == 0:
        #     model.update_kmeans_centers(z)

        # 更新学习率
        lr = model_params['learning_rate'] if scheduler is None else scheduler.get_last_lr()[0]
        if scheduler is not None:
            scheduler.step()

        
        if writer is not None:
            writer.add_scalar('Learning_raten', lr, epoch)
            gmm_labels = model.predict(tensor_gpu_data)
            unique_labels, counts = np.unique(gmm_labels, return_counts=True)
            writer.add_scalar('GMM/number_of_clusters', len(unique_labels), epoch)
        
        # 同步评估
        metrics = evaluator.evaluate_epoch(
            recon_x,
            gmm_labels,
            z,
            labels, 
            matched_S = matched_comp,
            matched_chem = matched_chems,
            epoch = epoch, 
            lr = lr, 
            train_metrics = train_metrics, 
            t_plot = t_plot, 
            r_plot = r_plot
        )

        # 检查早停条件
        if check_early_stopping(metrics, model_params['min_loss_threshold']):
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
