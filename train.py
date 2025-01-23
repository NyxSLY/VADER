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
    """管理整个训练流程的三阶段训练"""
    # 初始化配置和组件
    train_config = config.get_train_config()
    model_params = config.get_model_params()
    vis_config = config.get_vis_config()
    weight_config = config.get_weight_scheduler_config()
    t_plot = train_config['tsne_plot']
    r_plot = train_config['recon_plot']
    
    # 设置tensorboard
    model_name = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
    writer = SummaryWriter(log_dir=paths['tensorboard_log']+f'/{model_name}')
    
    # 初始化评估器
    evaluator = ModelEvaluator(
        model=model,
        device=model.device,
        paths=paths,
        writer=writer,
        resolution_2=model_params['resolution_2']
    )
    
    # 数据集分析
    print("正在分析数据集特征...")
    model.spectral_analyzer.analyze_dataset(dataloader)
    model.spectral_analyzer.save_analysis_results()
    
    print("=== 阶段 1: AE 预训练 ===")
    writer.add_text('Training', 'Stage 1: AE Pretraining', 0)
    model.pretrain(dataloader=dataloader, learning_rate=1e-3)
    torch.save(model.state_dict(), f'stage1_ae_pretrain.pt')
    
    print("=== 阶段 2: DEC式聚类预训练 ===")
    writer.add_text('Training', 'Stage 2: DEC Pretraining', model_params['epochs'] // 3)
    
    # 关闭attention
    model.encoder.use_attention = False
    model.decoder.use_attention = False
    
    # 冻结 Decoder
    for param in model.decoder.parameters():
        param.requires_grad = False
    
    # 初始化聚类中心
    model.init_kmeans_centers(dataloader)  # 添加这行，确保聚类中心正确初始化

    # 分离Encoder和GMM参数，设置不同学习率
    optimizer_stage2 = optim.Adam(
        [
            {'params': model.encoder.parameters(), 'lr': model_params['learning_rate']},  # Encoder微调
            {'params': model.gaussian.means, 'lr': 1e-4},        # 聚类中心快速调整
            {'params': model.gaussian.log_variances, 'lr': 1e-4},
            {'params': model.gaussian.pi, 'lr': 1e-4}
        ]
)
    
    # 阶段2训练循环
    n_epochs_stage2 = model_params['epochs'] // 3
    
    for epoch in range(n_epochs_stage2):
        model.train()
        total_loss = 0
        
        for batch_idx, (x, _) in enumerate(dataloader):
            x = x.to(model.device)
            # 前向传播
            mean, log_var = model.encoder(x)
            z = model.reparameterize(mean, log_var)
            
            # 计算DEC损失
            dec_loss = model._dec_loss(z)
            
            # 反向传播
            optimizer_stage2.zero_grad()
            dec_loss.backward()
            optimizer_stage2.step()
            
            total_loss += dec_loss.item()

        # 更新聚类中心
        if (epoch + 1) % 100 == 0:
            model.update_kmeans_centers()

        # 记录训练信息
        avg_loss = total_loss / len(dataloader)
        print(f"Stage 2 Epoch [{epoch+1}/{n_epochs_stage2}], Loss: {avg_loss:.4f}")
        writer.add_scalar('Stage2/dec_loss', avg_loss, epoch)
        
        # 定期评估
        if (epoch + 1) % 50 == 0:
            metrics = evaluator.evaluate_epoch(
                tensor_gpu_data, 
                labels, 
                epoch,
                model_params['learning_rate'],  # 直接使用固定学习率
                {'dec_loss': avg_loss},
                t_plot,
                r_plot
            )
    
    # 保存阶段2模型
    torch.save(model.state_dict(), f'stage2_dec_pretrain.pt')
    
    print("=== 阶段 3: 联合训练 ===")
    writer.add_text('Training', 'Stage 3: Joint Training', model_params['epochs'] * 2 // 3)
    
    # 重新开启attention
    model.encoder.use_attention = True
    model.decoder.use_attention = True

    # 解冻 Decoder
    for param in model.decoder.parameters():
        param.requires_grad = True
    
    # 初始化优化器 - 这里的LR变成10%
    optimizer_nn = optim.Adam(
        chain(model.encoder.parameters(), model.decoder.parameters()),
        lr=model_params['learning_rate'] * 0.1
    )
    optimizer_gmm = optim.Adam(
        model.gaussian.parameters(),
        lr=model_params['learning_rate'] * 0.1
    )
    
    # 初始化权重调度器
    weight_scheduler = WeightScheduler(
        init_weights=weight_config['init_weights'],
        max_weights=weight_config['max_weights'],
        n_epochs=model_params['epochs'] - n_epochs_stage2,
        resolution_2=model_params['resolution_2']
    )
    
    # 阶段3训练循环
    for epoch in range(model_params['epochs'] - n_epochs_stage2):
        # 更新权重
        weights = weight_scheduler.get_weights(epoch)
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
        
        # 更新聚类中心
        if (epoch + 1) % 100 == 0:
            model.update_kmeans_centers()
        
        # 记录训练信息
        lr_nn = model_params['learning_rate']
        recon_x, mean, log_var, z, gamma, pi = model(tensor_gpu_data)
        gmm_means, gmm_log_variances, y, gamma, pi = model.gaussian(z)
        
        if writer is not None:
            writer.add_scalar('Learning_rate_nn', lr_nn, epoch)
            writer.add_scalar('Learning_rate_gmm', 
                            model_params['learning_rate'], 
                            epoch)
            
            # 记录聚类信息
            gmm_probs = gamma.detach().cpu().numpy()
            gmm_labels = np.argmax(gmm_probs, axis=1)
            unique_labels, counts = np.unique(gmm_labels, return_counts=True)
            writer.add_scalar('GMM/number_of_clusters', len(unique_labels), epoch)
        
        # 评估
        metrics = evaluator.evaluate_epoch(
            tensor_gpu_data, 
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
            break
    
    return model

def check_early_stopping(metrics, thresholds):
    """检查是否需要早停"""
    return (metrics.get('total_loss') is not None and 
            metrics.get('recon_loss') is not None and 
            metrics['total_loss'] < thresholds['total'] and 
            metrics['recon_loss'] < thresholds['recon'])
