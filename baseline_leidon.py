# 使用leiden算法进行聚类
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import Birch
from utility import visualize_clusters
from matplotlib import cm  # 显式导入 cm 模块
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from utility import leiden_clustering
from metrics_new import ModelEvaluator


nc_data_org = np.load("/mnt/d/BaiduNetdiskWorkspace/OneDrive/work/VADER/VADERdata/X_reference.npy")
nc_labels_org = np.load("/mnt/d/BaiduNetdiskWorkspace/OneDrive/work/VADER/VADERdata/y_reference.npy").astype(int)

keep_indices = np.where((nc_labels_org == 2) | (
            nc_labels_org == 9) |  # (nc_labels ==25) | (nc_labels ==26) | (nc_labels ==27) | (nc_labels ==29)|\n",
                        (nc_labels_org == 18) | (nc_labels_org == 21) |
                        (nc_labels_org == 1) | (nc_labels_org == 5) | (nc_labels_org == 13) | (
                                    nc_labels_org == 20) | (nc_labels_org == 24))
oc_train_data = nc_data_org[keep_indices]
oc_train_label = nc_labels_org[keep_indices]

# 1. SVD分解取top50
from sklearn.decomposition import TruncatedSVD

# 进行SVD分解
svd = TruncatedSVD(n_components=50, random_state=42)
X_svd = svd.fit_transform(oc_train_data)

# 2. 使用Leiden算法进行聚类
# leiden_labels = leiden_clustering(X_svd, n_neighbors=15, resolution=1.0)

def compare_clustering_methods(data, true_labels, n_clusters=10, outputfile='clustering_comparison_results.csv'):
    """比较不同聚类方法的性能
    
    Args:
        data: 输入数据
        true_labels: 真实标签
        n_clusters: 聚类数量,默认为10
        
    Returns:
        results_df: 包含各种聚类方法评估指标的表格
    """
    # 定义评估器
    evaluator = ModelEvaluator(None, None)
    
    # 定义聚类方法
    methods = {
        'Leiden': lambda x: leiden_clustering(x, n_neighbors=15, resolution=1.0, seed=42),
        'KMeans': lambda x: KMeans(n_clusters=n_clusters, random_state=42).fit_predict(x),
        'Spectral': lambda x: SpectralClustering(n_clusters=n_clusters, random_state=42).fit_predict(x),
        'Agglomerative': lambda x: AgglomerativeClustering(n_clusters=n_clusters).fit_predict(x),
        'DBSCAN': lambda x: DBSCAN(eps=0.5, min_samples=5).fit_predict(x),
        'Birch': lambda x: Birch(n_clusters=n_clusters).fit_predict(x)
    }
    
    # 存储结果
    results = []
    
    # 评估每种方法
    for name, method in methods.items():
        pred_labels = method(data)
        acc = evaluator.calculate_acc(pred_labels, true_labels)
        ari = adjusted_rand_score(true_labels, pred_labels)
        nmi = normalized_mutual_info_score(true_labels, pred_labels)
        
        results.append({
            'Method': name,
            'ACC': f"{acc:.4f}",
            'ARI': f"{ari:.4f}", 
            'NMI': f"{nmi:.4f}"
        })
    
    # 打印结果表格
    print("\nClustering Methods Comparison:")
    print("-" * 50)
    print(f"{'Method':<15} {'ACC':<10} {'ARI':<10} {'NMI':<10}")
    print("-" * 50)
    for r in results:
        print(f"{r['Method']:<15} {r['ACC']:<10} {r['ARI']:<10} {r['NMI']:<10}")
    print("-" * 50)
    
    # 保存结果到CSV文件
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv('./baseline/' + outputfile, index=False)
    
    return results

compare_clustering_methods(X_svd, oc_train_label, n_clusters=9, outputfile='results_svd.csv')
compare_clustering_methods(oc_train_data, oc_train_label, n_clusters=9, outputfile='results_original.csv')

# 可视化聚类结果
#visualize_clusters(
#    z=X_svd,
#    labels=oc_train_label,
#    pred_labels=leiden_labels,
#    save_path='./baseline/leiden_clustering_results.png'
#    )