import numpy as np
import warnings  
warnings.filterwarnings('ignore', category=FutureWarning)  

from vade_new import VaDE
from utility import create_project_folders,prepare_data_loader, set_random_seed,choose_kmeans,set_device
from config import config
from train import train_manager
import torch
from utility import wavelet_transform
from grid_search import GridSearch
import os
from datetime import datetime

set_random_seed(123)


def main():
    # 读取数据
    # oc_train_data = np.loadtxt("/mnt/mt3/wangmc/lvfy/plotdata/oc_data_fil_191.txt", delimiter=" ")
    # oc_train_label = np.loadtxt("/mnt/mt3/wangmc/lvfy/plotdata/oc_labels_to_confmatrix.txt", delimiter=" ").astype(int)
    nc_data_org = np.load("/mnt/d/BaiduNetdiskWorkspace/OneDrive/work/VADER/VADERdata/X_reference.npy")
    nc_labels_org = np.load("/mnt/d/BaiduNetdiskWorkspace/OneDrive/work/VADER/VADERdata/y_reference.npy").astype(int)
    #nc_data_org = np.load("/mnt/c/Users/ASUS/OneDrive/work/VADER/VADERdata/X_reference.npy")
    #nc_labels_org = np.load("/mnt/c/Users/ASUS/OneDrive/work/VADER/VADERdata/y_reference.npy").astype(int)
    # nc_data_org = np.load("/home/zym/DESC/Datasets/NC-30-species/X_reference.npy")
    # nc_labels_org = np.load("/home/zym/DESC/Datasets/NC-30-species/y_reference.npy").astype(int)
    keep_indices = np.where((nc_labels_org == 2) | (
                nc_labels_org == 9) |  # (nc_labels ==25) | (nc_labels ==26) | (nc_labels ==27) | (nc_labels ==29)|\n",
                            (nc_labels_org == 18) | (nc_labels_org == 21) |
                            (nc_labels_org == 1) | (nc_labels_org == 5) | (nc_labels_org == 13) | (
                                        nc_labels_org == 20) | (nc_labels_org == 24))
    oc_train_data = nc_data_org[keep_indices]
    oc_train_label = nc_labels_org[keep_indices]


    # 准备数据
    model_params = config.get_model_params()
    device = set_device(model_params['device'])
    batch_size = model_params['batch_size']
    dataloader, unique_label, tensor_data, tensor_labels, tensor_gpu_data, tensor_gpu_labels = prepare_data_loader(oc_train_data, oc_train_label,batch_size,device)

    # 获取模型配置
    input_dim = tensor_data.shape[1]
    num_classes = len(unique_label)
    project_dir = create_project_folders("nc_test_gridSearch")
    
    weight_scheduler_config = config.get_weight_scheduler_config()
    paths = config.get_project_paths(project_dir, num_classes,
                                     lamb1=weight_scheduler_config['init_weights']['lamb1'],
                                     lamb2=weight_scheduler_config['init_weights']['lamb2'],
                                     lamb3=weight_scheduler_config['init_weights']['lamb3'],
                                     lamb4=weight_scheduler_config['init_weights']['lamb4'])
    l_c_dim = config.encoder_type(model_params['encoder_type'], paths['train_path'])

    # 初始化基础模型参数
    base_params = {
        'input_dim': input_dim,
        'intermediate_dim': model_params['intermediate_dim'],
        'latent_dim': model_params['latent_dim'],
        'num_classes': num_classes,
        'device': device,
        'encoder_type': model_params['encoder_type'],
        'l_c_dim': l_c_dim,
        'batch_size': batch_size
    }
    
    # 初始化并执行网格搜索
    grid_search = GridSearch(
        model_class=VaDE,
        data=oc_train_data,
        labels=oc_train_label,
        num_classes=num_classes,
        base_params=base_params,
        device=device
    )
    
    results, best_result = grid_search.search(project_dir)
    
    print("\n网格搜索完成!")
    print("最佳参数组合:", best_result['params'])
    print(f"最佳准确率: {best_result['acc']:.4f}")
    print(f"对应的NMI: {best_result['nmi']:.4f}")
    print(f"对应的ARI: {best_result['ari']:.4f}")
    
    return best_result

if __name__ == "__main__":
    main()
