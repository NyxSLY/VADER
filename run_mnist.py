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
import gzip
set_random_seed(123)


try:
    memo = sys.argv[1]
    if not memo or memo.isspace():
        memo = 'test'
except IndexError:
    memo = 'test'

def main():

    f = gzip.open(r"/mnt/sda/zhangym/VADER/VADE/dataset/mnist/mnist.pkl.gz", 'rb')
    (x_train, y_train), (x_test, y_test) = cPickle.load(f, encoding="bytes")
    f.close()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    X = np.concatenate((x_train,x_test))
    Y = np.concatenate((y_train,y_test))


    # 准备数据
    
    model_params = config.get_model_params()
    device = torch.device('cuda:4')
    batch_size = 100
    dataloader, unique_label, tensor_data, tensor_labels, tensor_gpu_data, tensor_gpu_labels = prepare_data_loader(X, Y,batch_size,device)

    # 获取模型配置
    input_dim = tensor_data.shape[1]
    num_classes = len(unique_label)
    project_dir = create_project_folders("home_pc")
    
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
