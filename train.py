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
from torchviz import make_dot
import torch.nn.functional as F

# 添加这行来设置多进程启动方法
mp.set_start_method('spawn', force=True)

# train.py (新增辅助函数)

def get_full_dataset_P(model, data_loader, alpha=1.0):
    """
    遍历整个数据集，计算所有样本的软分配 Q (gamma)，并计算目标分布 P。
    """
    model.eval() # 切换到评估模式 (不更新权重, 禁用 dropout/batchnorm)
    all_z = []
    all_gamma = []
    
    with torch.no_grad(): # 禁用梯度计算
        for x in data_loader:
            data_x = x[0].to(model.device)
            
            # 1. 前向传播获取 z
            _, _, _, z, _ = model(data_x) 
            all_z.append(z)
            
            # 2. 计算软分配 Q (gamma)
            gamma_batch = model.cal_desc_gamma(z) # 使用T-Kernel
            all_gamma.append(gamma_batch)
            
    # 合并结果
    z_all = torch.cat(all_z, dim=0)
    gamma_all = torch.cat(all_gamma, dim=0) # 整个数据集的 Q (N_total, K)
    
    # 3. 计算目标分布 P
    P_all = model.cal_target_distribution(gamma_all) # (N_total, K)
    
    return P_all.cpu().numpy(), z_all.cpu().numpy()

def train_epoch(model, weights, data_loader, optimizer, epoch, writer, matched_S, P_target=None):
    """训练一个epoch"""
    model.train()
    total_metrics = defaultdict(float)

    for batch_idx, x in enumerate(data_loader):
        # 数据准备
        data_x = x[0].to(model.device)
        data_y = x[1]
        
        # 前向传播   
        recon_x, z_mean,z_log_var, z,  S = model( data_x )
        z = model.reparameterize(z_mean, z_log_var)
        gamma = model.cal_gaussian_gamma(z)
        gamma_desc = model.cal_desc_gamma(z)
        P_batch = None
        if P_target is not None:
             start_idx = batch_idx * data_x.size(0)
             end_idx = start_idx + data_x.size(0)
             P_batch = P_target[start_idx:end_idx]
        
        # 损失计算
        loss_dict = model.compute_loss(data_x,data_y, recon_x, z_mean, z_log_var, gamma, S, matched_S, P_batch,gamma_desc,
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
    
    recon_x, z_mean, z_log_var, z, S = model(tensor_gpu_data)
    model.init_kmeans_centers(z)
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
        
    model_name = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
    writer = SummaryWriter(log_dir=paths['tensorboard_log']+f'/{model_name}')
    evaluator = ModelEvaluator(
        model=model,
        device=model.device,
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
    # 更新权重
    weights = weight_scheduler.get_weights(1)

    for epoch in range(0, epochs):
        print(f"\nEpoch [{epoch+1}/{epochs}]")
        
        # 训练一个epoch
        recon_x, z_mean, z_log_var, z, S = model(tensor_gpu_data)
        matched_comp, matched_chems = model.match_components(S,0.7)
        P_target_np, z_all_np = get_full_dataset_P(model, dataloader)
        P_target = torch.tensor(P_target_np, dtype=torch.float32).to(model.device)

        train_metrics = train_epoch(
            model=model, weights=weights,
            data_loader=dataloader,
            optimizer=optimizer,
            epoch=epoch,
            writer=writer,
            matched_S = matched_comp,
            P_target=P_target
        )
        recon_x, z_mean, z_log_var, z, S = model(tensor_gpu_data)
        
        # target_lamb1 = 20 * train_metrics['kl_gmm'] * weights['lamb1'] / train_metrics['recon_loss']
        # weights['lamb1'] =  target_lamb1 * 0.1 + weights['lamb1'] * 0.9
        
        # model.constraint_angle(tensor_gpu_data, weight=0.05) # 角度约束，保证峰形

        # 更新学习率
        lr = model_params['learning_rate'] if scheduler is None else scheduler.get_last_lr()[0]
        if scheduler is not None:
            scheduler.step()

        if writer is not None:
            writer.add_scalar('Learning_raten', lr, epoch)
            gamma = model.cal_gaussian_gamma(z)   # z是以前的，gamma是根据更新后的Gaussian计算的
            gmm_labels = np.argmax(gamma.detach().cpu().numpy(), axis=1)
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


        gaussian_save_path = os.path.join(paths['training_log'],f"epoch_{epoch}_GMM_ARI={metrics['gmm_ari']}_Gaussian.txt")
        gaussian_para = np.hstack((model.c_mean.detach().cpu().numpy(), model.c_log_var.detach().cpu().numpy(), model.pi_.detach().cpu().numpy().reshape(-1, 1)))
        np.savetxt(gaussian_save_path,gaussian_para)

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
