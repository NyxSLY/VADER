"""
配置文件：负责读取和管理配置参数
"""

import os
import yaml
import datetime
import shutil
from typing import Dict, Any

class ProjectConfig:
    """项目全局配置类"""

    def __init__(self, config_path: str = "model_config.yaml"):
        # 加载 YAML 配置
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # 路径配置
        self.BASE_PATHS = config['base_paths']

        # 模型参数
        self.MODEL_PARAMS = {
            'latent_dim': int(config['model_params']['basic_parameters']['latent_dim']),
            'intermediate_dim': config['model_params']['basic_parameters']['intermediate_dim'],
            'learning_rate': float(config['model_params']['learning_rate']),
            'use_lr_scheduler': bool(config['model_params']['use_lr_scheduler']),
            'batch_size': int(config['model_params']['batch_size']),
            'epochs': int(config['model_params']['epochs']),
            'device': str(config['model_params']['device']),
            'encoder_type': str(config['model_params']['encoder_type']),
            'pretrain_epochs': int(config['model_params']['pretrain_epochs']),
            'clustering_method': str(config['model_params']['clustering_method']),
            'resolution': float(config['model_params']['resolution']),
            'cnn': {k: int(v) for k, v in config['model_params']['cnn_parameters'].items()},
            'advanced': {k: int(v) for k, v in config['model_params']['advanced_parameters'].items()},

            'save_interval': int(config['model_params']['train_config']['save_interval']),
            'update_interval':int(config['model_params']['train_config']['update_interval']),
            'early_stop_patience': int(config['model_params']['train_config']['early_stop_patience']),
            'min_loss_threshold': {
                'total': float(config['model_params']['train_config']['min_loss_threshold']['total']),
                'recon': float(config['model_params']['train_config']['min_loss_threshold']['recon']),
            },
            'tsne_plot': bool(config['model_params']['train_config']['tsne_plot']),
            'recon_plot': bool(config['model_params']['train_config']['recon_plot'])
        }

        # 权重调度器
        self.WEIGHT_SCHEDULER = {
            'init_weights': {k: float(v) for k, v in config['weight_scheduler']['init_weights'].items()},
            'max_weights': {k: float(v) for k, v in config['weight_scheduler']['max_weights'].items()},
        }


    # ------------------- 模型参数相关 -------------------

    def get_model_params(self) -> Dict[str, Any]:
        """获取模型参数"""
        return self.MODEL_PARAMS.copy()

    def get_weight_scheduler_config(self) -> Dict[str, Dict[str, float]]:
        """获取权重调度器配置"""
        return self.WEIGHT_SCHEDULER.copy()

    # ------------------- 编码器配置 -------------------

    def encoder_type(self, encoder_type: str, train_path: str) -> Dict[str, Any]:
        """根据编码器类型生成参数并记录文件"""
        if encoder_type == "basic":
            dim = {
                'latent_dim': self.MODEL_PARAMS['latent_dim'],
                'intermediate_dim': self.MODEL_PARAMS['intermediate_dim'],
            }
            filename = f"encoder_type_basic_latent_{dim['latent_dim']}_intermediate_{dim['intermediate_dim']}.txt"

        elif encoder_type == "cnn":
            dim = self.MODEL_PARAMS['cnn']
            filename = f"encoder_type_cnn_c1_{dim['cnn1']}_c2_{dim['cnn2']}_c3_{dim['cnn3']}.txt"

        elif encoder_type in ["advanced", "ImprovedSpectralEncoder"]:
            dim = self.MODEL_PARAMS['advanced']
            filename = f"encoder_type_{encoder_type}_c1_{dim['cnn1']}_c2_{dim['cnn2']}_c3_{dim['cnn3']}.txt"

        else:
            raise ValueError(f"Invalid encoder type specified: {encoder_type}")

        open(os.path.join(train_path, filename), 'w').close()
        return dim

    # ------------------- 项目路径配置 -------------------

    def get_project_paths(
        self,
        project_dir: str,
        num_classes: int,
        lamb1=None, lamb2=None, lamb3=None, lamb4=None, lamb5=None, lamb6=None,
        memo: str = "test"
    ) -> Dict[str, str]:
        """生成并创建项目路径"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        # 参数标记
        params = [str(v) for v in (lamb1, lamb2, lamb3, lamb4, lamb5, lamb6) if v is not None]
        folder_params = "_".join(params) if params else ""
        base_folder_name = f"class{num_classes}"

        # 最终训练路径
        train_path = os.path.join(project_dir, memo)

        # 路径字典
        paths = {
            'train_path': train_path,
            'pth': os.path.join(train_path, self.BASE_PATHS['pth_dir']),
            'plot': os.path.join(train_path, self.BASE_PATHS['plot_dir']),
            'tensorboard_log': os.path.join(train_path, self.BASE_PATHS['tensorboard_dir']),
            'training_log': os.path.join(train_path, self.BASE_PATHS['txt_dir']),
        }

        # 创建文件夹
        for p in paths.values():
            os.makedirs(p, exist_ok=True)

        # 复制配置文件
        config_path = os.path.join(os.path.dirname(__file__), "model_config.yaml")
        if os.path.exists(config_path):
            shutil.copy2(config_path, train_path)
            print(f"成功复制配置文件到: {train_path}")

        return paths


# ------------------- 全局配置实例 -------------------
config = ProjectConfig()