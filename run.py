import numpy as np
import warnings

from sympy.geometry.plane import x  
warnings.filterwarnings('ignore', category=FutureWarning)  

from vade_new import VaDE
from utility import create_project_folders,prepare_data_loader, set_random_seed,set_device
from config import config
from train import train_manager
import torch
import sys

set_random_seed(123)


try:
    memo = sys.argv[1]
    if not memo or memo.isspace():
        memo = 'test'
except IndexError:
    memo = 'test'

def main():
    # NC
    # oc_train_data = np.load(r"/mnt/sda/gene/zhangym/VADER/Data/X_reference.npy")
    # oc_train_label = np.load(r"/mnt/sda/gene/zhangym/VADER/Data/y_reference.npy").astype(int)
    
    # keep_indices = np.where(np.isin(oc_train_label, [1,2,5,9,13,18,20,21,24]))
    # oc_train_data = oc_train_data[keep_indices]
    # oc_train_label = oc_train_label[keep_indices]
    
    # S = np.load(r"/mnt/sda/gene/zhangym/VADER/Data/MCR_NC9_S_20.npy")

    # HP_15
    # oc_train_data = np.load(r"/mnt/sda/gene/zhangym/VADER/Data/HP_X_processed.npy")
    # oc_train_label = np.load(r"/mnt/sda/gene/zhangym/VADER/Data/HP_Y_processed.npy").astype(int) 
    # S = np.load(r"/mnt/sda/gene/zhangym/VADER/Data/MCR_HP_S_10.npy").T


    # # Algae
    train_data = np.load(r"/mnt/sda/gene/zhangym/VADER/Data/Algae/Algae_process.npy")
    train_label = np.load(r"/mnt/sda/gene/zhangym/VADER/Data/Algae/Algae_label.npy")[:,0].astype(int)
    S = np.load(r"/mnt/sda/gene/zhangym/VADER/Data/Algae/MCR_Algae_S_10.npy")
    Wavenumber = np.load(r'/mnt/sda/gene/zhangym/VADER/Data/Algae/Algae_wave.npy')
    device = 'cuda:2'
    pretrain_epochs = 1
    epochs = 600
    project_tag = 'Test_MCREC/1211_cVADER_softplus_KLpq_update100'
    memo = 'Algae'
    

    # 准备数据
    model_params = config.get_model_params()
    device = set_device(device)
    batch_size = model_params['batch_size']
    dataloader, unique_label, tensor_data, tensor_labels, tensor_gpu_data, tensor_gpu_labels = prepare_data_loader(
        train_data, train_label, batch_size, device
    )
    # 获取模型配置
    input_dim = tensor_data.shape[1]
    n_component = S.shape[0]
    project_dir = create_project_folders(project_tag)
    
    paths = config.get_project_paths( project_dir, memo=memo)
    l_c_dim = config.encoder_type(model_params['encoder_type'], paths['train_path'])

    # 初始化模型
    model = VaDE(
        input_dim=input_dim,
        intermediate_dim=model_params['intermediate_dim'],
        latent_dim=n_component,
        tensor_gpu_data=tensor_gpu_data,
        n_components=n_component,
        S=torch.tensor(S).float().to(device),
        wavenumber = Wavenumber,
        prior_y=train_label,
        device=device,
        l_c_dim=l_c_dim,
        encoder_type=model_params['encoder_type'],
        pretrain_epochs=pretrain_epochs,
        num_classes=n_component,
        clustering_method=model_params['clustering_method'],
        resolution=model_params['resolution']
    ).to(device)

    model.pretrain(dataloader=dataloader, learning_rate=1e-5)

    model = train_manager(
        model=model,
        dataloader=dataloader,
        tensor_gpu_data=tensor_gpu_data,
        labels=tensor_gpu_labels,
        paths=paths,
        epochs = epochs
    )

    return model

if __name__ == "__main__":
    main()
