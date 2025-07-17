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
import asyncio


set_random_seed(123)


try:
    memo = sys.argv[1]
    if not memo or memo.isspace():
        memo = 'test'
except IndexError:
    memo = 'test'

def train_on_dataset(
    train_data, train_label, S, device, project_tag, Pretrain_epochs, epochs, batch_size, memo="test"):

    model_params = config.get_model_params()
    device = set_device(device)
    dataloader, unique_label, tensor_data, tensor_labels, tensor_gpu_data, tensor_gpu_labels = prepare_data_loader(
        train_data, train_label, batch_size, device
    )
    input_dim = tensor_data.shape[1]
    project_dir = create_project_folders(project_tag)
    weight_scheduler_config = config.get_weight_scheduler_config()
    n_component = S.shape[0]
    paths = config.get_project_paths(
        project_dir,
        n_component,
        lamb1=weight_scheduler_config['init_weights']['lamb1'],
        lamb2=weight_scheduler_config['init_weights']['lamb2'],
        lamb3=weight_scheduler_config['init_weights']['lamb3'],
        lamb4=weight_scheduler_config['init_weights']['lamb4'],
        lamb5=weight_scheduler_config['init_weights']['lamb5'],
        lamb6=weight_scheduler_config['init_weights']['lamb6'],
        memo=memo,
    )
    l_c_dim = config.encoder_type(model_params['encoder_type'], paths['train_path'])
    model = VaDE(
        input_dim=input_dim,
        intermediate_dim=model_params['intermediate_dim'],
        latent_dim=n_component,
        tensor_gpu_data=tensor_gpu_data,
        n_components=n_component,
        S=torch.tensor(S).float().to(device),
        lamb1=weight_scheduler_config['init_weights']['lamb1'],
        lamb2=weight_scheduler_config['init_weights']['lamb2'],
        lamb3=weight_scheduler_config['init_weights']['lamb3'],
        lamb4=weight_scheduler_config['init_weights']['lamb4'],
        lamb5=weight_scheduler_config['init_weights']['lamb5'],
        lamb6=weight_scheduler_config['init_weights']['lamb6'],
        device=device,
        l_c_dim=l_c_dim,
        batch_size=batch_size,
        encoder_type=model_params['encoder_type'],
        pretrain_epochs=model_params['pretrain_epochs'],
        num_classes=n_component,
        clustering_method=model_params['clustering_method'],
        resolution_1=model_params['resolution_1'],
        resolution_2=model_params['resolution_2']
    ).to(device)

    model.kmeans_init = 'random'
    model.pretrain(dataloader=dataloader, learning_rate=1e-5)
    model = train_manager(
        model=model,
        dataloader=dataloader,
        tensor_gpu_data=tensor_gpu_data,
        labels=tensor_gpu_labels,
        num_classes=n_component,
        paths=paths,
        epochs = epochs
    )
    print(f"[{project_tag}] 训练完成。\n")



async def run_all_datasets_async(datasets):
    tasks = []
    for ds in datasets:
        tasks.append(train_on_dataset(**ds))

    await asyncio.gather(*tasks)

def main():
    datasets = [
        # {
        #     'train_data': np.load(r"/mnt/sda/gene/zhangym/VADER/Data/Algae_process.npy"),
        #     "train_label": np.load(r"/mnt/sda/gene/zhangym/VADER/Data/Algae_label.npy")[:,0].astype(int),
        #     "S": np.load(r"/mnt/sda/gene/zhangym/VADER/Data/MCR_Algae_S_10.npy"),
        #     "device": "cuda:1",
        #     "project_tag": "Test_MCREC/0717_base",
        #     'Pretrain_epochs': 1000,
        #     'epochs': 3000,
        #     'batch_size': 128,
        #     "memo": "Algae"
        # },
        {
            "train_data":  np.load(r"/mnt/sda/gene/zhangym/VADER/Data/HP_X_processed.npy"),
            "train_label": np.load(r"/mnt/sda/gene/zhangym/VADER/Data/HP_Y_processed.npy").astype(int),
            "S": np.load(r"/mnt/sda/gene/zhangym/VADER/Data/MCR_HP_S_10.npy"),
            "device": "cuda:3",
            "project_tag": "Test_MCREC/0717_base",
            'Pretrain_epochs': 100,
            'epochs':   600,
            'batch_size':   128,
            "memo": "HP_15"
        },
        # {
        #     "train_data":  np.load(r"/mnt/sda/gene/zhangym/VADER/Data/X_reference_9.npy"),
        #     "train_label": np.load(r"/mnt/sda/gene/zhangym/VADER/Data/y_reference_9.npy").astype(int),
        #     "S": np.load(r"/mnt/sda/gene/zhangym/VADER/Data/MCR_NC9_S_20.npy"),
        #     "device": "cuda:4",
        #     "project_tag": "Test_MCREC/0717_base",
        #     'Pretrain_epochs': 50,
        #     'epochs':   300,
        #     'batch_size':   128,
        #     "memo": "NC_9"
        # },
        # {
        #     "train_data":  np.load(r"/mnt/sda/gene/zhangym/VADER/Data/Ocean_3/Ocean_train_process.npy"),
        #     "train_label": np.repeat([0,1,2],50),
        #     "S": np.load(r"/mnt/sda/gene/zhangym/VADER/Data/Ocean_3/MCR_Ocean3_10_component.npy"),
        #     "device": "cuda:2",
        #     "project_tag": "Test_MCREC/0717_base",
        #     'Pretrain_epochs': 5000,
        #     'epochs':   30000,
        #     'batch_size':   128,
        #     "memo": "Ocean_3"
        # },
        # {
        #     "train_data":  np.load(r"/mnt/sda/gene/zhangym/VADER/Data/Marine_7/Marine_7.npy"),
        #     "train_label": np.load(r"/mnt/sda/gene/zhangym/VADER/Data/Marine_7/Marine_7_label.npy").astype(int),
        #     "S": np.load(r'/mnt/sda/gene/zhangym/VADER/Data/Marine_7/MCR_Marine7_10_component.npy'),
        #     "device": "cuda:3",
        #     "project_tag": "Test_MCREC/0717_base",
        #     'Pretrain_epochs': 100,
        #     'epochs':   600,
        #     'batch_size':   128,
        #     "memo": "Ocean_7"
        # },
        # {
        #     "train_data":  np.load(r"/mnt/sda/gene/zhangym/VADER/Data/Neuron/X_Neuron.npy"),
        #     "train_label": np.load(r"/mnt/sda/gene/zhangym/VADER/Data/Neuron/Y_Neuron.npy").astype(int),
        #     "S": np.load(r"/mnt/sda/gene/zhangym/VADER/Data/Neuron/MCR_Neuron_20_component.npy"),
        #     "device": "cuda:2",
        #     "project_tag": "Test_MCREC/0717_base",
        #     'Pretrain_epochs': 300,
        #     'epochs':   1000,
        #     'batch_size':   128,
        #     "memo": "Neuron"
        # },
        # {
        #     "train_data":  np.load(r"/mnt/sda/gene/zhangym/VADER/Data/Probiotics/X_traindata.npy"),
        #     "train_label": np.load(r"/mnt/sda/gene/zhangym/VADER/Data/Probiotics/Y_traindata.npy").astype(int),
        #     "S": np.load(r"/mnt/sda/gene/zhangym/VADER/Data/Probiotics/MCR_Probiotics_20_component.npy"),
        #     "device": "cuda:0",
        #     "project_tag": "Test_MCREC/0717_base",
        #     'Pretrain_epochs': 300,
        #     'epochs':   1000,
        #     'batch_size':   128,
        #     "memo": "Probiotics"
        # }
    ]

    all_models = asyncio.run(run_all_datasets_async(datasets))



if __name__ == "__main__":
    main()
