"""
配置文件：负责读取和管理配置参数
"""
import os
import yaml
from typing import Dict, Any
import datetime

class ProjectConfig:
    """项目全局配置类"""
    
    def __init__(self, config_path: str = "model_config.yaml"):
        # 加载YAML配置
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        # 从YAML加载各项配置
        self.BASE_PATHS = config['base_paths']
        
        # 确保MODEL_PARAMS中的数值类型正确
        self.MODEL_PARAMS = {
            'latent_dim': int(config['model_params']['basic_parameters']['latent_dim']),
            'intermediate_dim': int(config['model_params']['basic_parameters']['intermediate_dim']),
            'learning_rate': float(config['model_params']['learning_rate']),
            'batch_size': int(config['model_params']['batch_size']),
            'epochs': int(config['model_params']['epochs']),
            'device':str(config['model_params']['device']),
            'encoder_type': str(config['model_params']['encoder_type']),
            'pretrain_epochs': int(config['model_params']['pretrain_epochs']),
            'clustering_method': str(config['model_params']['clustering_method']),
            'resolution_1': float(config['model_params']['resolution_1']),
            'resolution_2': float(config['model_params']['resolution_2']),
            'cnn':{
                'cnn1': int(config['model_params']['cnn_parameters']['cnn1']),
                'cnn2': int(config['model_params']['cnn_parameters']['cnn2']),
                'cnn3': int(config['model_params']['cnn_parameters']['cnn3']),
            },
            'advanced':{
                'cnn1': int(config['model_params']['advanced_parameters']['cnn1']),
                'cnn2': int(config['model_params']['advanced_parameters']['cnn2']),
                'cnn3': int(config['model_params']['advanced_parameters']['cnn3']),
            }

        }

        # 确保WEIGHT_SCHEDULER中的数值类型正确
        self.WEIGHT_SCHEDULER = {
            'init_weights': {
                k: float(v) for k, v in config['weight_scheduler']['init_weights'].items()
            },
            'max_weights': {
                k: float(v) for k, v in config['weight_scheduler']['max_weights'].items()
            }
        }
        
        # 确保TRAIN_CONFIG中的数值类型正确
        self.TRAIN_CONFIG = {
            'start_epoch': int(config['train_config']['start_epoch']),
            'save_interval': int(config['train_config']['save_interval']),
            'early_stop_patience': int(config['train_config']['early_stop_patience']),
            'min_loss_threshold': {
                'total': float(config['train_config']['min_loss_threshold']['total']),
                'recon': float(config['train_config']['min_loss_threshold']['recon'])},
            'tsne_plot': bool(config['train_config']['tsne_plot']),
            'recon_plot': bool(config['train_config']['recon_plot'])
        }
        
        # 确保VIS_CONFIG中的数值类型正确
        self.VIS_CONFIG = {
            # 'figsize': config['vis_config']['figsize'],
            'dpi': int(config['vis_config']['dpi']),
            'wavenumber_ticks': [int(x) for x in config['vis_config']['wavenumber_ticks']]
        }
        
        # 确保COLOR_MAPS中的键为整数
        self.COLOR_MAPS = {
            str_k: {int(k): v for k, v in color_map.items()}
            for str_k, color_map in config['color_maps'].items()
        }

    def encoder_type(self, encoder_type, train_path):
        if encoder_type == "basic":
            dim = {
                'latent_dim': self.MODEL_PARAMS['latent_dim'],
                'intermediate_dim': self.MODEL_PARAMS['intermediate_dim']
             }

            with open(f'{train_path}/encoder_type_{encoder_type}_latent_dim_{dim["latent_dim"]}_intermediate_dim_{dim["intermediate_dim"]}.txt', 'w') as file:
                pass

        elif encoder_type == "cnn":
            dim = {
                'cnn1': self.MODEL_PARAMS['cnn']['cnn1'],
                'cnn2': self.MODEL_PARAMS['cnn']['cnn2'],
                'cnn3': self.MODEL_PARAMS['cnn']['cnn3']
            }

            with open(
                    f'{train_path}/encoder_type_{encoder_type}_cnn_channels1_{dim["cnn1"]}_cnn_channels2_{dim["cnn2"]}_cnn_channels3_{dim["cnn3"]}.txt','w') as file:
                pass

        elif encoder_type in ["advanced", "ImprovedSpectralEncoder"]:
            dim = {
                'cnn1': self.MODEL_PARAMS['advanced']['cnn1'],
                'cnn2': self.MODEL_PARAMS['advanced']['cnn2'],
                'cnn3': self.MODEL_PARAMS['advanced']['cnn3']
            }
            with open(
                    f'{train_path}/encoder_type_{encoder_type}_cnn_channels1_{dim["cnn1"]}_cnn_channels2_{dim["cnn2"]}_cnn_channels3_{dim["cnn3"]}.txt','w') as file:
                pass
        else:
            raise ValueError(f"Invalid encoder type specified: {encoder_type}")

        return dim


    def get_project_paths(self, project_dir: str, num_classes: int,
                         lamb1, lamb2, lamb3, lamb4, lamb5=None, lamb6=None, memo='test') -> Dict[str, str]:
        """获取项目所有路径配置"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        base_folder_name = f"class{num_classes}"
        
        # 构建文件夹名称，只包含非None的参数
        params = []
        if lamb1 is not None: params.append(str(lamb1))
        if lamb2 is not None: params.append(str(lamb2))
        if lamb3 is not None: params.append(str(lamb3))
        if lamb4 is not None: params.append(str(lamb4))
        if lamb5 is not None: params.append(str(lamb5))
        if lamb6 is not None: params.append(str(lamb6))
        
        folder_params = '_'.join(params)
        #train_path = os.path.join(project_dir, f'{folder_params}_{base_folder_name}_{timestamp}')
        
        train_path = os.path.join(project_dir, f'{memo}_{timestamp}')
        
        paths = {
            'train_path':train_path,
            'pth': os.path.join(train_path, self.BASE_PATHS['pth_dir']),
            'plot': os.path.join(train_path, self.BASE_PATHS['plot_dir']),
            # 'logs': os.path.join(train_path, self.BASE_PATHS['log_dir'], base_folder_name),
            'tensorboard_log': os.path.join(train_path, self.BASE_PATHS['tensorboard_dir']),
            'training_log': os.path.join(train_path, self.BASE_PATHS['txt_dir'])
        }

        # 创建所需的文件夹
        for path in paths.values():
            os.makedirs(path, exist_ok=True)

        config_path = os.path.join(os.path.dirname(__file__), "model_config.yaml")
        print(f"配置文件路径: {config_path}")
        if os.path.exists(config_path):
            import shutil
            shutil.copy2(config_path, train_path)
            print(f"成功复制配置文件到: {train_path}")
        
        return paths
    
    def get_color_map(self, num_classes: int) -> Dict[int, str]:
        """获取指定类别数量的颜色映射
        
        Args:
            num_classes: 类别数量
            
        Returns:
            Dict[int, str]: 整数索引到颜色字符串的映射
        """
        str_num = str(num_classes)
        if str_num not in self.COLOR_MAPS:
            raise ValueError(f"不支持的类别数量: {num_classes}")
        
        return self.COLOR_MAPS[str_num]
    
    def get_model_params(self) -> Dict[str, Any]:
        """获取模型参数"""
        return self.MODEL_PARAMS.copy()
    
    def get_train_config(self) -> Dict[str, Any]:
        """获取训练配置"""
        return self.TRAIN_CONFIG.copy()
    
    def get_vis_config(self) -> Dict[str, Any]:
        """获取可视化配置"""
        return self.VIS_CONFIG.copy()
    
    def get_weight_scheduler_config(self) -> Dict[str, Dict[str, float]]:
        """获取权重调度器配置"""
        return self.WEIGHT_SCHEDULER.copy()

# 创建全局配置实例
config = ProjectConfig()