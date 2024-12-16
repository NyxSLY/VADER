import numpy as np

'''
# 加载原始数据
nc_data_org = np.load("/mnt/d/BaiduNetdiskWorkspace/OneDrive/work/VADER/VADERdata/X_reference.npy")
nc_labels_org = np.load("/mnt/d/BaiduNetdiskWorkspace/OneDrive/work/VADER/VADERdata/y_reference.npy").astype(int)

# 选择需要的标签
keep_indices = np.where((nc_labels_org == 2) | (
            nc_labels_org == 9) |  # (nc_labels ==25) | (nc_labels ==26) | (nc_labels ==27) | (nc_labels ==29)|\n",
                        (nc_labels_org == 18) | (nc_labels_org == 21) |
                        (nc_labels_org == 1) | (nc_labels_org == 5) | (nc_labels_org == 13) | (
                                    nc_labels_org == 20) | (nc_labels_org == 24))
oc_train_data = nc_data_org[keep_indices]
oc_train_label = nc_labels_org[keep_indices]

# 保存数据和标签到txt文件
np.savetxt('./raw_data_qc/filtered_train_data.txt', oc_train_data, delimiter=',', fmt='%.6f')
np.savetxt('./raw_data_qc/filtered_train_label.txt', oc_train_label, delimiter=',', fmt='%d')

'''
# plot the spectra
from utility import plot_reconstruction
oc_train_data = np.load("/mnt/c/Users/ASUS/OneDrive/work/VADER/VADERdata/processed_NC_9.npy")
oc_train_label = np.load("/mnt/c/Users/ASUS/OneDrive/work/VADER/VADERdata/processed_NC_9_label.npy").astype(int)


#plot_reconstruction(recon_data=oc_train_data, labels=oc_train_label, save_path='./raw_data_qc/raw_spectra.png')

# 随机从每个label取100条光谱，再画图
unique_labels = np.unique(oc_train_label)
selected_indices = []

for label in unique_labels:
    label_indices = np.where(oc_train_label == label)[0]
    if len(label_indices) > 100:
        selected = np.random.choice(label_indices, size=100, replace=False)
    else:
        selected = label_indices
    selected_indices.extend(selected)

selected_indices = np.array(selected_indices)
selected_data = oc_train_data[selected_indices]
selected_labels = oc_train_label[selected_indices]

plot_reconstruction(recon_data=selected_data, labels=selected_labels, save_path='./raw_data_qc/QCed_spectra_sampled.png')
