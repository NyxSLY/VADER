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
import pandas as pd
from utility import plot_spectra, plot_UMAP

set_random_seed(123)

def generate_from_gaussian(model, n, method = 'Raw', gamma_thr = 0.9, seed=42):
    means = model.c_mean.detach().cpu().numpy()
    vars_ = np.exp(model.c_log_var.detach().cpu().numpy())
    S = model.encoder.S.detach().cpu().numpy()

    C, D = means.shape
    K = S.shape[1]

    rng = np.random.default_rng(seed)
    std = np.sqrt(vars_)

    collected_X = []
    collected_Y = []
    
    for c in range(C):
        accepted_out_c = []

        mu_c = means[c]           # (D,)
        std_c = std[c]            # (D,)

        trials = 0
        total = 0
        while total < n and trials < 100:
            trials += 1
            z_c = rng.normal(loc=mu_c, scale=std_c, size=(100, D))
            out_c = np.matmul(z_c, S)     # (batch_size, K)

            if method == 'Raw':
                keep_idx = np.arange(0, 100)

            if method == 'Prob':
                gamma = model.cal_gaussian_gamma(torch.tensor(z_c).float().to(model.device)).detach().cpu().numpy()  
                gamma_c = gamma[:, c]  # (batch_size,)
                keep_idx = np.where(gamma_c > gamma_thr)[0]   # 通过阈值的索引
            
            if method == 'Max':
                gamma = model.cal_gaussian_gamma(torch.tensor(z_c).float().to(model.device)).detach().cpu().numpy() 
                argmax_cls = np.argmax(gamma, axis=1)
                keep_idx = np.where(argmax_cls == c)[0]   # 通过最大值的索引

            if keep_idx.size > 0:
                accepted_out_c.append(out_c[keep_idx,:])
                total += out_c[keep_idx,:].shape[0]

        # 合并
        if len(accepted_out_c) > 0:
            accepted_out_c = np.vstack(accepted_out_c)  # (m_ok, K)
        else:
            accepted_out_c = np.empty((0, K), dtype=float)
    
        collected_X.append(accepted_out_c)          # (n, K)
        collected_Y.append(np.full(accepted_out_c.shape[0], c, dtype=int))  # (n,)
        
    X = np.vstack(collected_X)       # (C*n, K)
    Y = np.concatenate(collected_Y)  # (C*n,)

    print(X.shape)
    return X, Y


try:
    memo = sys.argv[1]
    if not memo or memo.isspace():
        memo = 'test'
except IndexError:
    memo = 'test'

def train_on_dataset(
    train_data, train_label, S, Wavenumber, device, project_tag, Pretrain_epochs, epochs, batch_size, memo="test", n_gene = None):

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

    torch.save(model.state_dict(), f'/mnt/sda/gene/zhangym/VADER/Augmentation/Gene_spectra/Generated_Spectra/{memo}_clustering_model_{epochs}.pk')

    if n_gene is not None:
        # labels_batch = None if model.prior_y is None else labels.to(model.device)
        # Method: 'Raw', 'Prob', 'Max'
        gene_samples, gene_labels = generate_from_gaussian(model, method = 'Raw',n=n_gene, gamma_thr = 0.7)
        np.save(f'/mnt/sda/gene/zhangym/VADER/Augmentation/Gene_spectra/Generated_Spectra/{memo}_X_gene_VADER_Raw_{n_gene}.npy', gene_samples)
        np.save(f'/mnt/sda/gene/zhangym/VADER/Augmentation/Gene_spectra/Generated_Spectra/{memo}_Y_gene_VADER_Raw_{n_gene}.npy', gene_labels)
        plot_spectra( recon_data=gene_samples, labels=gene_labels, save_path=f'/mnt/sda/gene/zhangym/VADER/Augmentation/Gene_spectra/Generated_Spectra/{memo}_VADER_Raw_{n_gene}_Spectra.png', wavenumber=model.wavenumber)
        plot_UMAP(gene_samples, gene_labels,f'/mnt/sda/gene/zhangym/VADER/Augmentation/Gene_spectra/Generated_Spectra/{memo}_VADER_Raw_{n_gene}_UMAP.png')

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
    project_tag = 'Test_MCREC/1013_Generate_lamb20'
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
            "train_data":  np.load(r"/mnt/sda/gene/zhangym/VADER/Data/Fungi_7/Fungi7_Horiba_X.npy"),
            "train_label": np.load(r"/mnt/sda/gene/zhangym/VADER/Data/Fungi_7/Fungi7_Horiba_Y.npy").astype(int),
            "S": np.load(r"/mnt/sda/gene/zhangym/VADER/Data/Fungi_7/MCR_Horiba_20_component.npy"),
            "Wavenumber": np.load(r'/mnt/sda/gene/zhangym/VADER/Data/Fungi_7/Fungi7_Wave.npy'),
            "device": "cuda:2",
            "project_tag": project_tag,
            'Pretrain_epochs': 100,
            'epochs':   100,
            'batch_size':   128,
            "memo": "Fungi_7",
            'n_gene': None
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
            "memo": "NC_9",
            'n_gene': None
        },
        {
            "train_data":  np.flip(np.load(r"/mnt/sda/gene/zhangym/VADER/Data/NC_9/X_reference.npy"), axis=1),
            "train_label": np.load(r"/mnt/sda/gene/zhangym/VADER/Data/NC_9/y_reference.npy").astype(int), 
            "S": np.flip(np.load(r"/mnt/sda/gene/zhangym/VADER/Data/NC_All/MCR_NCAll_Raw_30_component.npy"),axis=1),
            "Wavenumber": np.flip(np.load(r'/mnt/sda/gene/zhangym/VADER/Data/NC_9/wavenumbers.npy'), axis=0),
            "device": "cuda:1",
            "project_tag": project_tag,
            'Pretrain_epochs': 100,
            'epochs':   100,
            'batch_size':   128,
            "memo": "NC_All",
            'n_gene': None
        },

        {
            "train_data":  np.load(r"/mnt/sda/gene/zhangym/VADER/Data/Ocean_3/Ocean_train_process.npy"),
            "train_label": np.repeat([0,1,2],50),
            "S": np.load(r"/mnt/sda/gene/zhangym/VADER/Data/Ocean_3/MCR_Ocean3_10_component.npy"),
            "Wavenumber": np.arange(600, 1801),
            "device": "cuda:2",
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
        #     "memo": "ATCC_7_0.01s",
        #     "n_gene": 10000
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
        #     "memo": "ATCC_7_0.1s",
        #     "n_gene": 10000
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
        #     "memo": "ATCC_7_1s",
        #     "n_gene": 10000
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
        #     "memo": "ATCC_7_10s",
        #     "n_gene": 10000
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
        #     "memo": "ATCC_7_15s",
        #     "n_gene": 10000
        # }
    ]

    all_models = asyncio.run(run_all_datasets_async(datasets))



if __name__ == "__main__":
    main()
