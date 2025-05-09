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
set_random_seed(123)


try:
    memo = sys.argv[1]
    if not memo or memo.isspace():
        memo = 'test'
except IndexError:
    memo = 'test'

def main():
    # 读取数据
    # oc_train_data = np.loadtxt("/mnt/mt3/wangmc/lvfy/plotdata/oc_data_fil_191.txt", delimiter=" ")
    # oc_train_label = np.loadtxt("/mnt/mt3/wangmc/lvfy/plotdata/oc_labels_to_confmatrix.txt", delimiter=" ").astype(int)
    # # nc_data_org = np.load(r"/mnt/sda/zhangym/VADER/Data/processed_NC_9.npy")
    # nc_labels_org = np.load(r"/mnt/sda/zhangym/VADER/Data/processed_NC_9_label.npy").astype(int)

    # nc_data_org = np.load(r"/mnt/sda/zhangym/VADER/Data/X_reference.npy")
    # nc_labels_org = np.load(r"/mnt/sda/zhangym/VADER/Data/y_reference.npy").astype(int)
    # # home pc
    # # nc_data_org = np.load("/mnt/c/Users/ASUS/OneDrive/work/VADER/VADERdata/processed_NC_9.npy")
    # # nc_labels_org = np.load("/mnt/c/Users/ASUS/OneDrive/work/VADER/VADERdata/processed_NC_9_label.npy").astype(int)
    

    
    # keep_indices = np.where((nc_labels_org == 2) | (
    #             nc_labels_org == 9) |  # (nc_labels ==25) | (nc_labels ==26) | (nc_labels ==27) | (nc_labels ==29)|\n",
    #                         (nc_labels_org == 18) | (nc_labels_org == 21) |
    #                         (nc_labels_org == 1) | (nc_labels_org == 5) | (nc_labels_org == 13) | (
    #                                     nc_labels_org == 20) | (nc_labels_org == 24))
    # oc_train_data = nc_data_org[keep_indices]
    # oc_train_label = nc_labels_org[keep_indices]
    nc_data_org = np.load(r"D:\Scientific research\deep learning\DESC\DataSet\Downloaded Datasets\NC-30-species/X_reference.npy")
    nc_labels_org = np.load(r"D:\Scientific research\deep learning\DESC\DataSet\Downloaded Datasets\NC-30-species/y_reference.npy").astype(int)
    keep_indices = np.where(np.isin(nc_labels_org, [1,2,5,9,13,18,20,21,24]))
    oc_train_data = nc_data_org[keep_indices]
    oc_train_label = nc_labels_org[keep_indices]


    # 准备数据
    model_params = config.get_model_params()
    device = set_device(model_params['device'])
    batch_size = model_params['batch_size']
    dataloader, unique_label, tensor_data, tensor_labels, tensor_gpu_data, tensor_gpu_labels = prepare_data_loader(oc_train_data, oc_train_label,batch_size,device)

    # 获取模型配置
    input_dim = tensor_data.shape[1]
    num_classes = 10 # len(unique_label)
    project_dir = create_project_folders("Test_MCR")
    
    weight_scheduler_config = config.get_weight_scheduler_config()
    paths = config.get_project_paths(project_dir, num_classes,
                                     lamb1=weight_scheduler_config['init_weights']['lamb1'],
                                     lamb2=weight_scheduler_config['init_weights']['lamb2'],
                                     lamb3=weight_scheduler_config['init_weights']['lamb3'],
                                     lamb4=weight_scheduler_config['init_weights']['lamb4'],
                                     lamb5=weight_scheduler_config['init_weights']['lamb5'],
                                     lamb6=weight_scheduler_config['init_weights']['lamb6'], 
                                     memo=memo)
    l_c_dim = config.encoder_type(model_params['encoder_type'], paths['train_path'])

    # 初始化模型
    model = VaDE(
        input_dim=input_dim,
        intermediate_dim=model_params['intermediate_dim'],
        latent_dim=model_params['latent_dim'],
        tensor_gpu_data=tensor_gpu_data,
        n_components=20,
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
        num_classes=num_classes,
        clustering_method=model_params['clustering_method'],
        resolution_1=model_params['resolution_1'],
        resolution_2=model_params['resolution_2']
    ).to(device)

    # model.eval()
    #choose_kmeans_method = choose_kmeans(model,dataloader,num_classes)
    # 更新模型的kmeans初始化方法
    #model.kmeans_init = choose_kmeans_method
    model.kmeans_init = 'random'
    # 训练模型
    print("\n开始预训练...  ")
    model.pretrain(
        dataloader=dataloader,
        learning_rate=1e-3
    )

    

    print("\n开始模型训练...")
    # model.state_dict(torch.load("/mnt/d/BaiduNetdiskWorkspace/OneDrive/work/VADER/Vader-11.21/Vader-11.21/nc/100000.0_1.0_0.0_0.0_class9_20241127-154315/pth/epoch_60_acc_0.49_nmi_0.59_ari_0.38.pth"))
    model = train_manager(
        model=model,
        dataloader=dataloader,
        tensor_gpu_data=tensor_gpu_data,
        labels=tensor_gpu_labels,
        num_classes=num_classes,
        paths=paths,
    )

    return model

if __name__ == "__main__":
    main()
