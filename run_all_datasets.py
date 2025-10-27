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

def train_on_dataset(
    train_data, train_label, S, Wavenumber, device, project_tag, Pretrain_epochs, epochs, batch_size, memo="test"):
    set_random_seed(123)

    model_params = config.get_model_params()
    device = set_device(device)
    dataloader, unique_label, tensor_data, tensor_labels, tensor_gpu_data, tensor_gpu_labels = prepare_data_loader(
        train_data, train_label, batch_size, device
    )
    input_dim = tensor_data.shape[1]
    project_dir = create_project_folders(project_tag)
    weight_scheduler_config = config.get_weight_scheduler_config()
    n_component = S.shape[0]
    paths = config.get_project_paths( project_dir, memo=memo)
    l_c_dim = config.encoder_type(model_params['encoder_type'], paths['train_path'])
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

def train_on_dataset_wrapper(args):
    return train_on_dataset(**args)

async def run_all_datasets_async(datasets):
    loop = asyncio.get_event_loop()
    results = []

    with ProcessPoolExecutor(max_workers=min(len(datasets), multiprocessing.cpu_count())) as executor:
        tasks = [
            loop.run_in_executor(
                executor,
                train_on_dataset_wrapper,
                ds
            )
            for ds in datasets
        ]
        for f in asyncio.as_completed(tasks):
            result = await f
            results.append(result)
    return results

def main():
    project_tag = 'Test_MCREC/1027_VADER_match_loss'
    datasets = [
        {
            'train_data': np.load(r"/mnt/sda/gene/zhangym/VADER/Data/Algae/Algae_process.npy"),
            "train_label": np.load(r"/mnt/sda/gene/zhangym/VADER/Data/Algae/Algae_label.npy")[:,0].astype(int),
            "S": np.load(r"/mnt/sda/gene/zhangym/VADER/Data/Algae/MCR_Algae_S_10.npy"),
            "Wavenumber": np.load(r'/mnt/sda/gene/zhangym/VADER/Data/Algae/Algae_wave.npy'),
            "device": "cuda:1",
            "project_tag": project_tag,
            'Pretrain_epochs': 300,
            'epochs': 600,
            'batch_size': 128,
            "memo": "Algae"
        },
        {
            "train_data":  np.load(r"/mnt/sda/gene/zhangym/VADER/Data/HP/HP_X_processed.npy"),
            "train_label": np.load(r"/mnt/sda/gene/zhangym/VADER/Data/HP/HP_Y_processed.npy").astype(int),
            "S": np.load(r"/mnt/sda/gene/zhangym/VADER/Data/HP/MCR_HP_S_10.npy"),
            "Wavenumber": np.load(r'/mnt/sda/gene/zhangym/VADER/Data/HP/HP_wave.npy'),
            "device": "cuda:3",
            "project_tag": project_tag,
            'Pretrain_epochs': 200,
            'epochs':   600,
            'batch_size':   128,
            "memo": "HP_15"
        },
        {
            "train_data":  np.flip(np.load(r"/mnt/sda/gene/zhangym/VADER/Data/NC_9/X_reference_9.npy"), axis=1),
            "train_label": np.load(r"/mnt/sda/gene/zhangym/VADER/Data/NC_9/y_reference_9.npy").astype(int),
            "S": np.flip(np.load(r"/mnt/sda/gene/zhangym/VADER/Data/NC_9/MCR_NC9_S_20.npy"),axis=1),
            "Wavenumber": np.flip(np.load(r'/mnt/sda/gene/zhangym/VADER/Data/NC_9/wavenumbers.npy'),axis=0),
            "device": "cuda:0",
            "project_tag": project_tag,
            'Pretrain_epochs': 100,
            'epochs':   300,
            'batch_size':   128,
            "memo": "NC_9"
        },
        {
            "train_data":  np.flip(np.load(r"/mnt/sda/gene/zhangym/VADER/Data/NC_9/X_reference.npy"), axis=1),
            "train_label": np.load(r"/mnt/sda/gene/zhangym/VADER/Data/NC_9/y_reference.npy").astype(int), 
            "S": np.flip(np.load(r"/mnt/sda/gene/zhangym/VADER/Data/NC_All/MCR_NCAll_Raw_30_component.npy"),axis=1),
            "Wavenumber": np.flip(np.load(r'/mnt/sda/gene/zhangym/VADER/Data/NC_9/wavenumbers.npy'), axis=0),
            "device": "cuda:1",
            "project_tag": project_tag,
            'Pretrain_epochs': 100,
            'epochs':   300,
            'batch_size':   128,
            "memo": "NC_All"
        },

        {
            "train_data":  np.load(r"/mnt/sda/gene/zhangym/VADER/Data/Ocean_3/Ocean_train_process.npy"),
            "train_label": np.repeat([0,1,2],50),
            "S": np.load(r"/mnt/sda/gene/zhangym/VADER/Data/Ocean_3/MCR_Ocean3_10_component.npy"),
            "Wavenumber": np.arange(600, 1801),
            "device": "cuda:0",
            "project_tag": project_tag,
            'Pretrain_epochs': 100,
            'epochs':   500,
            'batch_size':   128,
            "memo": "Ocean_3"
        },
        {
            "train_data":  np.load(r"/mnt/sda/gene/zhangym/VADER/Data/Marine_7/Marine_7.npy"),
            "train_label": np.load(r"/mnt/sda/gene/zhangym/VADER/Data/Marine_7/Marine_7_label.npy").astype(int),
            "S": np.load(r'/mnt/sda/gene/zhangym/VADER/Data/Marine_7/MCR_Marine7_10_component.npy'),
            "Wavenumber": np.load(r'/mnt/sda/gene/zhangym/VADER/Data/Marine_7/Marine_7_wave.npy'),
            "device": "cuda:3",
            "project_tag": project_tag,
            'Pretrain_epochs': 200,
            'epochs':   600,
            'batch_size':   128,
            "memo": "Ocean_7"
        },
        {
            "train_data":  np.load(r"/mnt/sda/gene/zhangym/VADER/Data/Neuron/X_Neuron.npy"),
            "train_label": np.load(r"/mnt/sda/gene/zhangym/VADER/Data/Neuron/Y_Neuron.npy").astype(int),
            "S": np.load(r"/mnt/sda/gene/zhangym/VADER/Data/Neuron/MCR_Neuron_20_component.npy"),
            "Wavenumber": np.load(r'/mnt/sda/gene/zhangym/VADER/Data/Neuron/Neuron_wave.npy'),
            "device": "cuda:3",
            "project_tag": project_tag,
            'Pretrain_epochs': 100,
            'epochs':   300,
            'batch_size':   128,
            "memo": "Neuron"
        },
        {
            "train_data":  np.load(r"/mnt/sda/gene/zhangym/VADER/Data/Probiotics/X_probiotics.npy"),
            "train_label": np.load(r"/mnt/sda/gene/zhangym/VADER/Data/Probiotics/Y_probiotics.npy").astype(int),
            "S": np.load(r"/mnt/sda/gene/zhangym/VADER/Data/Probiotics/MCR_Probiotics_20_component.npy"),
            "Wavenumber": np.linspace(500, 1800, 593),
            "device": "cuda:0",
            "project_tag": project_tag,
            'Pretrain_epochs': 300,
            'epochs':   1000,
            'batch_size':   128,
            "memo": "Probiotics"
        },
        {
            'train_data': np.load(r"/mnt/sda/gene/zhangym/VADER/Data/MTB_drug/MTB_Drug_scientific_X.npy"),
            "train_label": np.load(r"/mnt/sda/gene/zhangym/VADER/Data/MTB_drug/MTB_Drug_scientific_Y.npy").astype(int),
            "S": np.load(r"/mnt/sda/gene/zhangym/VADER/Data/MTB_drug/MTB_Drug_scientific_S_8.npy"),
            "Wavenumber": np.load(r'/mnt/sda/gene/zhangym/VADER/Data/MTB_drug/MTB_Drug_scientific_wave.npy'),
            "device": "cuda:1",
            "project_tag": project_tag,
            'Pretrain_epochs': 100,
            'epochs': 500,
            'batch_size': 128,
            "memo": "MTB_Scitific"
        },

        ## ATCC Datasets
        # {
        #     'train_data': np.load(r"/mnt/sda/gene/zhangym/VADER/Data/ATCC_7/Noise_0.01s.npy"),  
        #     "train_label": np.load(r"/mnt/sda/gene/zhangym/VADER/Data/ATCC_7/Noise_0.01s_y.npy")[:,0].astype(int),
        #     "S": np.load(r'/mnt/sda/gene/zhangym/VADER/Data/ATCC_7/MCR_0.01s_20_component.npy'),
        #     "Wavenumber": np.load(r'/mnt/sda/gene/zhangym/VADER/Data/ATCC_7/Wavenumber.npy'),
        #     "device": "cuda:1",
        #     "project_tag": project_tag,
        #     'Pretrain_epochs': 100,
        #     'epochs':   300,
        #     'batch_size':   128,
        #     "memo": "ATCC_7_0.01s"
        # },
        # {
        #     'train_data': np.load(r"/mnt/sda/gene/zhangym/VADER/Data/ATCC_7/Noise_0.1s.npy"),  
        #     "train_label": np.load(r"/mnt/sda/gene/zhangym/VADER/Data/ATCC_7/Noise_0.1s_y.npy")[:,0].astype(int),
        #     "S": np.load(r'/mnt/sda/gene/zhangym/VADER/Data/ATCC_7/MCR_0.1s_20_component.npy'),
        #     "Wavenumber": np.load(r'/mnt/sda/gene/zhangym/VADER/Data/ATCC_7/Wavenumber.npy'),
        #     "device": "cuda:0",
        #     "project_tag": project_tag,
        #     'Pretrain_epochs': 100,
        #     'epochs':   300,
        #     'batch_size':   128,
        #     "memo": "ATCC_7_0.1s"
        # },
        # {
        #     'train_data': np.load(r"/mnt/sda/gene/zhangym/VADER/Data/ATCC_7/Noise_1s.npy"),  
        #     "train_label": np.load(r"/mnt/sda/gene/zhangym/VADER/Data/ATCC_7/Noise_1s_y.npy")[:,0].astype(int),
        #     "S": np.load(r'/mnt/sda/gene/zhangym/VADER/Data/ATCC_7/MCR_1s_20_component.npy'),
        #     "Wavenumber": np.load(r'/mnt/sda/gene/zhangym/VADER/Data/ATCC_7/Wavenumber.npy'),
        #     "device": "cuda:2",
        #     "project_tag": project_tag,
        #     'Pretrain_epochs': 100,
        #     'epochs':   300,
        #     'batch_size':   128,
        #     "memo": "ATCC_7_1s"
        # },
        # {
        #     'train_data': np.load(r"/mnt/sda/gene/zhangym/VADER/Data/ATCC_7/Noise_10s.npy"),  
        #     "train_label": np.load(r"/mnt/sda/gene/zhangym/VADER/Data/ATCC_7/Noise_10s_y.npy")[:,0].astype(int),
        #     "S": np.load(r'/mnt/sda/gene/zhangym/VADER/Data/ATCC_7/MCR_10s_20_component.npy'),
        #     "Wavenumber": np.load(r'/mnt/sda/gene/zhangym/VADER/Data/ATCC_7/Wavenumber.npy'),
        #     "device": "cuda:3",
        #     "project_tag": project_tag,
        #     'Pretrain_epochs': 100,
        #     'epochs':   300,
        #     'batch_size':   128,
        #     "memo": "ATCC_7_10s"
        # },
        # {
        #     'train_data': np.load(r"/mnt/sda/gene/zhangym/VADER/Data/ATCC_7/Noise_15s.npy"),  
        #     "train_label": np.load(r"/mnt/sda/gene/zhangym/VADER/Data/ATCC_7/Noise_15s_y.npy")[:,0].astype(int),
        #     "S": np.load(r'/mnt/sda/gene/zhangym/VADER/Data/ATCC_7/MCR_15s_20_component.npy'),
        #     "Wavenumber": np.load(r'/mnt/sda/gene/zhangym/VADER/Data/ATCC_7/Wavenumber.npy'),
        #     "device": "cuda:1",
        #     "project_tag": project_tag,
        #     'Pretrain_epochs': 100,
        #     'epochs':   300,
        #     'batch_size':   128,
        #     "memo": "ATCC_7_15s"
        # }
    ]

    all_models = asyncio.run(run_all_datasets_async(datasets))



if __name__ == "__main__":
    main()
