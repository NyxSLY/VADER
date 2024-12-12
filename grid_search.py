import itertools
import numpy as np
from train import train_manager
import torch
from utility import prepare_data_loader
import json
from datetime import datetime
import os

class GridSearch:
    def __init__(self, model_class, data, labels, num_classes, base_params, device):
        """
        初始化网格搜索
        
        Args:
            model_class: 模型类
            data: 训练数据
            labels: 标签
            num_classes: 类别数量
            base_params: 基础模型参数
            device: 设备
        """
        self.model_class = model_class
        self.data = data
        self.labels = labels
        self.num_classes = num_classes
        self.base_params = base_params
        self.device = device
        
        # 设置网格搜索参数范围
        self.param_grid = {
            'lamb1': [0, 1, 1e3, 1e4, 1e5, 1e6],
            'lamb2': [0.1, 1.0, 10.0],
            'lamb3': [0.0, 0.1, 1.0],
            'lamb4': [0, 1, 1e3, 1e4, 1e5, 1e6],
            'lamb5': [0, 0.1, 0.5, 1.0],
            'lamb6': [0, 0.1, 0.5, 1.0]
        }
        
    def create_model(self, params):
        """使用给定参数创建模型"""
        model_params = self.base_params.copy()
        model_params.update(params)
        return self.model_class(**model_params).to(self.device)
    
    def evaluate_params(self, params, dataloader, tensor_gpu_data, tensor_gpu_labels, paths):
        """评估单个参数组合"""
        model = self.create_model(params)
        
        try:
            trained_model = train_manager(
                model=model,
                dataloader=dataloader,
                tensor_gpu_data=tensor_gpu_data,
                labels=tensor_gpu_labels,
                num_classes=self.num_classes,
                paths=paths
            )
            
            # 获取最终评估指标
            with torch.no_grad():
                trained_model.eval()
                _, _, _, _, _, y_pred = trained_model(tensor_gpu_data)
                metrics = trained_model.compute_metrics(tensor_gpu_labels, y_pred)
            
            return {
                'params': params,
                'acc': metrics['acc'],
                'nmi': metrics['nmi'],
                'ari': metrics['ari']
            }
            
        except Exception as e:
            print(f"参数组合训练失败: {params}")
            print(f"错误信息: {str(e)}")
            return {
                'params': params,
                'acc': 0,
                'nmi': 0,
                'ari': 0
            }
    
    def search(self, project_dir):
        """执行网格搜索"""
        # 准备数据加载器
        dataloader, _, _, _, tensor_gpu_data, tensor_gpu_labels = prepare_data_loader(
            self.data, self.labels, self.base_params['batch_size'], self.device
        )
        
        # 生成所有参数组合
        param_combinations = [dict(zip(self.param_grid.keys(), v)) 
                            for v in itertools.product(*self.param_grid.values())]
        
        results = []
        best_result = {'acc': 0, 'params': None}
        
        # 创建结果保存目录
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        results_dir = os.path.join(project_dir, 'grid_search_results', timestamp)
        os.makedirs(results_dir, exist_ok=True)
        
        for i, params in enumerate(param_combinations):
            print(f"\n测试参数组合 {i+1}/{len(param_combinations)}:")
            print(params)
            
            # 为每个参数组合创建独特的路径
            param_str = '_'.join([f"{k}_{v}" for k, v in params.items()])
            paths = {
                'train_path': os.path.join(results_dir, param_str),
                'tensorboard_log': os.path.join(results_dir, param_str, 'logs'),
                'plot': os.path.join(results_dir, param_str, 'plot'),
                'pth': os.path.join(results_dir, param_str, 'pth'),
                'training_log': os.path.join(results_dir, param_str, 'txt')
            }
            # 创建所有必要的目录
            print("Creating directories:")
            for path in paths.values():
                if path.endswith('.txt'):  # 如果是文件路径
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                else:  # 如果是目录路径
                    os.makedirs(path, exist_ok=True)
            
            # 评估参数组合
            result = self.evaluate_params(
                params, dataloader, tensor_gpu_data, tensor_gpu_labels, paths
            )
            results.append(result)
            
            # 更新最佳结果
            if result['acc'] > best_result['acc']:
                best_result = result
            
            # 保存当前结果
            with open(os.path.join(results_dir, 'results.json'), 'w') as f:
                json.dump({
                    'all_results': results,
                    'best_result': best_result
                }, f, indent=4)
            
        return results, best_result 