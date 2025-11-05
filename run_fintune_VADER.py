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
import asyncio
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

try:
    memo = sys.argv[1]
    if not memo or memo.isspace():
        memo = 'test'
except IndexError:
    memo = 'test'

def main():
    model_file = '/mnt/sda/gene/zhangym/VADER/Augmentation/Gene_spectra/Model/NC_All_cVADER_100_0.76.pk'
    X_fn = '/mnt/sda/gene/zhangym/VADER/Data/NC_9/X_finetune.npy'
    y_fn = '/mnt/sda/gene/zhangym/VADER/Data/NC_9/y_finetune.npy'
    X = np.flip(np.load(X_fn), axis=1)
    y = np.load(y_fn).astype(int)
    Wavenumber = np.flip(np.load(r'/mnt/sda/gene/zhangym/VADER/Data/NC_9/wavenumbers.npy'), axis=0)
    S = np.flip(np.load(r"/mnt/sda/gene/zhangym/VADER/Data/NC_All/MCR_NCAll_Raw_30_component.npy"),axis=1)
    np.flip(np.load(r"/mnt/sda/gene/zhangym/VADER/Data/NC_All/MCR_NCAll_Raw_30_component.npy"),axis=1)
    print(f'Finetune VADER on dataset of {X.shape}')

    epochs = 300
    batch_size = 128
    n_cluster = 30
    device = "cuda:1"
    save_file = '/mnt/sda/gene/zhangym/VADER/Augmentation/Gene_spectra/Model/NC_All_cVADER_Finetune_300.pk'

    set_random_seed(123)
    dataloader, unique_label, tensor_data, tensor_labels, tensor_gpu_data, tensor_gpu_labels = prepare_data_loader( X, y, batch_size, device)
    input_dim = tensor_data.shape[1]


    model = VaDE( input_dim= X_fn.shape[1], intermediate_dim=[512,1024,2048], latent_dim=S.shape[0], tensor_gpu_data=X, n_components=n_component,
            S=torch.tensor(S.copy()).float().to(device), wavenumber = Wavenumber, prior_y=y,
            device=device, l_c_dim='', encoder_type='basic', pretrain_epochs=50, num_classes=n_cluster, clustering_method='leiden', resolution=0.8 ).to(device)
    model.load_state_dict(torch.load(model_file))

    model_params = config.get_model_params()
    device = set_device(device)

    model = VaDE(
        input_dim=input_dim,
        intermediate_dim=model_params['intermediate_dim'],
        latent_dim=n_component,
        tensor_gpu_data=tensor_gpu_data,
        n_components=n_component,
        S=torch.tensor(S).float().to(device),
        wavenumber = Wavenumber,
        # prior_y=train_label,
        device=device,
        l_c_dim=l_c_dim,
        encoder_type=model_params['encoder_type'],
        pretrain_epochs=Pretrain_epochs,
        num_classes=n_component,
        clustering_method=model_params['clustering_method'],
        resolution=model_params['resolution']
    ).to(device)

    model.kmeans_init = 'random'
    model.pretrain(dataloader=dataloader, learning_rate=1e-5)
    model = train_manager(
        model=model,
        dataloader=dataloader,
        tensor_gpu_data=tensor_gpu_data,
        labels=tensor_gpu_labels,
        paths=paths,
        epochs = epochs
    )

    # torch.save(model.state_dict(), f'/mnt/sda/gene/zhangym/VADER/Augmentation/Gene_spectra/Generated_Spectra/{memo}_VADER_{epochs}.pk')
    # torch.save(model.state_dict(), f'/mnt/sda/gene/zhangym/VADER/VADER/Test_MCREC/1015_cVADER_lamb20/{memo}_cVADER_{epochs}.pk')

    print(f"[{project_tag}] 训练完成。\n")






if __name__ == "__main__":
    main()
