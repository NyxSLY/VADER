import numpy as np
import warnings  
warnings.filterwarnings('ignore', category=FutureWarning)  

from vade_new import VaDE
from utility import create_project_folders,prepare_data_loader, set_random_seed,choose_kmeans,set_device
from config import config
from train import train_manager
import torch
from utility import wavelet_transform
import sys
import random
import os


try:
    memo = sys.argv[1]
    if not memo or memo.isspace():
        memo = 'test'
except IndexError:
    memo = 'test'

def get_dataset_params(dataset_name):
    if dataset_name == 'NC_all':
        data = np.load(r"/mnt/sda/zhangym/VADER/Data/X_reference.npy")
        label = np.load(r"/mnt/sda/zhangym/VADER/Data/y_reference.npy").astype(int)
        epoch = 300
    
    elif dataset_name == 'NC_9':
        data_all = np.load(r"/mnt/sda/zhangym/VADER/Data/X_reference.npy")
        label_all = np.load(r"/mnt/sda/zhangym/VADER/Data/y_reference.npy").astype(int)
        keep_indices = np.where(np.isin(label_all, [1,2,5,9,13,18,20,21,24]))
        data = data_all[keep_indices]
        label = label_all[keep_indices]
        epoch = 1000
    
    elif dataset_name == 'Ocean':
        data = np.load(r"/mnt/sda/zhangym/VADER/Data/Ocean_train_process.npy")
        label = np.repeat([0,1,2],50)
        epoch = 70000


    elif dataset_name == 'HP_15':
        data = np.load(r"/mnt/sda/zhangym/VADER/Data/HP_X_processed.npy")
        label = np.load(r"/mnt/sda/zhangym/VADER/Data/HP_Y_processed.npy").astype(int)
        epoch = 2000
    
    elif dataset_name == 'Algae':
        data = np.load(r"/mnt/sda/zhangym/VADER/Data/Algae_process.npy")
        label = np.load(r"/mnt/sda/zhangym/VADER/Data/Algae_label.npy")[:,0].astype(int)
        epoch = 15000

    
    return data, label, epoch
    
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools

def train_wrapper(args):
    """包装训练函数以便并行化"""
    dataset_params, latent_dim, learning_rate, lr_scheduler, resolution, batch_size, work_path = args
    data, label, epoch = dataset_params
    device = set_device(f'cuda:{random.randint(0, 4)}')
    
    # 原有训练逻辑
    dataloader, unique_label, tensor_data, tensor_labels, tensor_gpu_data, tensor_gpu_labels = prepare_data_loader(data, label, batch_size, device)
    input_dim = tensor_data.shape[1]
    num_classes = len(np.unique(label))
    paths = {
            'train_path':work_path,
            'pth': work_path,
            'plot': work_path,
            'tensorboard_log': work_path,
            'training_log': work_path}
    
    # 创建模型并训练
    model = VaDE(
        input_dim=input_dim,
        intermediate_dim=[512,1024,2048],
        latent_dim=latent_dim,
        tensor_gpu_data=tensor_gpu_data,
        lamb1=0, lamb2=1, lamb3=0, lamb4=1, lamb5=0, lamb6=0,
        device=device,
        batch_size=batch_size,
        encoder_type='basic',
        pretrain_epochs=0,
        epochs=epoch,
        learning_rate=learning_rate,
        num_classes=num_classes,
        clustering_method='leiden',
        resolution_1=resolution,
        resolution_2=resolution
    ).to(device)
    
    model.kmeans_init = 'random'
    # model.pretrain(
    #     dataloader=dataloader,
    #     learning_rate=1e-3
    # )
    model = train_manager( model=model,
                          dataloader=dataloader,
                          tensor_gpu_data=tensor_gpu_data,
                          labels=tensor_gpu_labels,
                          num_classes=num_classes, 
                          paths=paths)
    return args

def main():
    datasets = ['Algae'] # 'Ocean', 'Algae', 'NC_9', 'HP_15', 'NC_all'
    pretrain = 0
    
    # 生成所有参数组合（保留dataset和latent_dim的循环）
    all_args = []
    for dataset in datasets:
        data, label, epoch = get_dataset_params(dataset)
        path = os.path.join('home_pc','para_test', f'{dataset}')
        os.makedirs(path, exist_ok=True)
        
        for latent_dim in [10, 20]:
            # 生成其他参数的笛卡尔积
            other_params = itertools.product(
                [1e-4, 1e-3],  # learning_rate
                [True, False], # lr_scheduler
                [1, 0.6, 0.8], # resolution
                [128, 256, 512] # batch_size
            )
            
            for lr, scheduler, res, bs in other_params:
                work_path = os.path.join(path, f'{dataset}_pretrain={pretrain}_latent={latent_dim}_{lr}_{scheduler}_{res}_{bs}')
                os.makedirs(work_path, exist_ok=True)
                all_args.append(((data, label, epoch), latent_dim, lr, scheduler, res, bs, work_path))

    # 控制并行数量（根据GPU数量调整）
    max_workers = 3
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(train_wrapper, args) for args in all_args]
        
        # 可以添加进度监控
        for future in as_completed(futures):
            try:
                result = future.result()
                print(f"Completed: {result[-1]}")
            except Exception as e:
                print(f"Error occurred: {e}")
                  
if __name__ == "__main__":
    main()
