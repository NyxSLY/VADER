import numpy as np
import warnings  
warnings.filterwarnings('ignore', category=FutureWarning)  

from vade_new import VaDE
from utility import create_project_folders,prepare_data_loader, set_random_seed,choose_kmeans,set_device
from config import config
from train import train_manager
import torch
from utility import visualize_clusters, plot_reconstruction

trained_model = '/mnt/d/BaiduNetdiskWorkspace/OneDrive/work/VADER/Vader-11.21/Vader-11.21/nc_test/100000.0_1.0_0.0_0.1_class9_20241129-085948/pth/epoch_670_acc_0.58_nmi_0.65_ari_0.46.pth'

def evaluate_model(trained_model: str):
    # 加载数据
    nc_data_org = np.load("/mnt/d/BaiduNetdiskWorkspace/OneDrive/work/VADER/VADERdata/X_reference.npy")
    nc_labels_org = np.load("/mnt/d/BaiduNetdiskWorkspace/OneDrive/work/VADER/VADERdata/y_reference.npy").astype(int)

    keep_indices = np.where((nc_labels_org == 2) | (
                nc_labels_org == 9) |  # (nc_labels ==25) | (nc_labels ==26) | (nc_labels ==27) | (nc_labels ==29)|\n",
                            (nc_labels_org == 18) | (nc_labels_org == 21) |
                            (nc_labels_org == 1) | (nc_labels_org == 5) | (nc_labels_org == 13) | (
                                        nc_labels_org == 20) | (nc_labels_org == 24))
    oc_train_data = nc_data_org[keep_indices]
    oc_train_label = nc_labels_org[keep_indices]
    print(f"数据大小: {oc_train_data.shape}, 标签大小: {oc_train_label.shape}")

    # 准备数据
    model_params = config.get_model_params()
    device = set_device(model_params['device'])
    batch_size = model_params['batch_size']
    dataloader, unique_label, tensor_data, tensor_labels, tensor_gpu_data, tensor_gpu_labels = prepare_data_loader(oc_train_data, oc_train_label,batch_size,device)

    # 获取模型配置
    input_dim = tensor_data.shape[1]
    num_classes = len(unique_label)
    project_dir = create_project_folders("nc_test")
    weight_scheduler_config = config.get_weight_scheduler_config()
    paths = config.get_project_paths(project_dir, num_classes,
                                        lamb1=weight_scheduler_config['init_weights']['lamb1'],
                                        lamb2=weight_scheduler_config['init_weights']['lamb2'],
                                        lamb3=weight_scheduler_config['init_weights']['lamb3'],
                                        lamb4=weight_scheduler_config['init_weights']['lamb4'], )
    l_c_dim = config.encoder_type(model_params['encoder_type'], paths['train_path'])

    # 初始化模型
    model = VaDE(
        input_dim=input_dim,
        intermediate_dim=model_params['intermediate_dim'],
        latent_dim=model_params['latent_dim'],
        num_classes=num_classes,
        lamb1=weight_scheduler_config['init_weights']['lamb1'],
        lamb2=weight_scheduler_config['init_weights']['lamb2'],
        lamb3=weight_scheduler_config['init_weights']['lamb3'],
        lamb4=weight_scheduler_config['init_weights']['lamb4'],
        device=device,
        encoder_type=model_params['encoder_type'],
        l_c_dim=l_c_dim
    ).to(device)

    # 加载预训练模型权重
    model.load_state_dict(torch.load(trained_model))

    # 从trained_model路径中提取信息
    model_info = trained_model.split('/')[-1].replace('.pth', '')  # 获取文件名并移除.pth后缀

    # 设置为评估模式
    model.eval()

    # 无梯度推理
    with torch.no_grad():
        recon_x, mean, log_var, z, z_prior_mean, y_pred = model(tensor_gpu_data)
        
        # 将结果转移到CPU
        recon_x_cpu = recon_x.cpu().numpy()
        z_cpu = z.cpu().numpy()
        y_pred_cpu = y_pred.cpu().numpy()
        y_pred_labels = np.argmax(y_pred_cpu, axis=1)  # 获取预测的类别标签
        
        # 保存结果（文件名包含模型信息）
        np.save(f'recon_data_{model_info}.npy', recon_x_cpu)
        np.save(f'latent_vectors_{model_info}.npy', z_cpu)
        np.save(f'pred_labels_{model_info}.npy', y_pred_labels)
        
    print(f"重构数据形状: {recon_x_cpu.shape}")
    print(f"潜在向量形状: {z_cpu.shape}")
    print(f"预测标签形状: {y_pred_labels.shape}")
    print(f"保存的文件名包含模型信息: {model_info}")


evaluate_model(trained_model)
nc_labels_org = np.load("/mnt/d/BaiduNetdiskWorkspace/OneDrive/work/VADER/VADERdata/y_reference.npy").astype(int)

keep_indices = np.where((nc_labels_org == 2) | (
            nc_labels_org == 9) |  # (nc_labels ==25) | (nc_labels ==26) | (nc_labels ==27) | (nc_labels ==29)|\n",
                        (nc_labels_org == 18) | (nc_labels_org == 21) |
                        (nc_labels_org == 1) | (nc_labels_org == 5) | (nc_labels_org == 13) | (
                                    nc_labels_org == 20) | (nc_labels_org == 24))
oc_train_label = nc_labels_org[keep_indices]

model_info = trained_model.split('/')[-1].replace('.pth', '')  # 获取文件名并移除.pth后缀
recon_x_cpu = np.load(f'recon_data_{model_info}.npy')
z_cpu = np.load(f'latent_vectors_{model_info}.npy')
y_pred_labels = np.load(f'pred_labels_{model_info}.npy')

# 使用Leiden算法进行聚类
from utility import leiden_clustering  # 假设utility.py中已经定义了这个函数
LDN = 20
leiden_labels = leiden_clustering(z_cpu, resolution=0.9, n_neighbors=LDN)

# 可视化聚类结果（如果需要）
visualize_clusters(z=z_cpu, labels=oc_train_label, pred_labels=y_pred_labels, 
                  save_path=f'model_clusters_{model_info}_LeidenN{LDN}.png')

#visualize_clusters(z=z_cpu, labels=oc_train_label, pred_labels=leiden_labels, 
#                  save_path=f'leiden_clusters_{model_info}.png')

# 保存Leiden聚类标签
np.save(f'leiden_labels_{model_info}.npy', leiden_labels)