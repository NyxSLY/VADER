import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

class SimpleAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, dim),
            nn.Sigmoid()  # 改回Sigmoid，因为我们不需要权重和为1
        )
    
    def forward(self, x):
        # 计算特征的重要性分数（使用更合适的方法）
        feature_importance = torch.abs(x)  # 直接使用绝对值作为重要性指标
        
        # 生成基础注意力权重
        base_weights = self.attention(x)
        
        # 结合特征重要性和基础权重
        weights = base_weights * feature_importance
        
        # 不需要归一化，让权重自然反映特征重要性
        return x * weights, weights

class SimpleAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            SimpleAttention(256),
            nn.Linear(256, 64),
            nn.ReLU()
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            SimpleAttention(256),
            nn.Linear(256, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # 记录中间的注意力权重
        self.attention_weights = []
        
        # 编码
        x_encoded = x
        for layer in self.encoder:
            if isinstance(layer, SimpleAttention):
                x_encoded, weights = layer(x_encoded)
                self.attention_weights.append(weights)
            else:
                x_encoded = layer(x_encoded)
        
        # 解码
        x_decoded = x_encoded
        for layer in self.decoder:
            if isinstance(layer, SimpleAttention):
                x_decoded, weights = layer(x_decoded)
                self.attention_weights.append(weights)
            else:
                x_decoded = layer(x_decoded)
                
        return x_decoded

def generate_spectrum(n_points=1000):
    """生成单个模拟光谱数据，使峰更明显"""
    x = np.linspace(0, 999, n_points)
    # 创建更平滑的基线
    baseline = 0.1 * np.sin(x / 300) + 0.3
    
    # 创建三个明显的高斯峰
    peak1 = 0.9 * np.exp(-(x - 250)**2 / 50)  # 更窄更高的峰
    peak2 = 1.0 * np.exp(-(x - 500)**2 / 50)
    peak3 = 0.8 * np.exp(-(x - 750)**2 / 50)
    
    # 添加较小的噪声
    noise = np.random.normal(0, 0.02, n_points)
    
    # 组合所有成分
    spectrum = baseline + peak1 + peak2 + peak3 + noise
    return torch.FloatTensor(spectrum)

def generate_batch_spectra(batch_size=32, n_points=1000):
    """生成一批光谱数据，每个都略有变化"""
    spectra = []
    for _ in range(batch_size):
        spectrum = generate_spectrum(n_points)
        spectra.append(spectrum)
    return torch.stack(spectra)

def visualize_attention_effect(model, test_spectrum, epoch):
    """可视化注意力机制的效果"""
    model.eval()
    device = next(model.parameters()).device
    test_spectrum = test_spectrum.to(device)
    
    with torch.no_grad():
        # 获取重建结果
        reconstructed = model(test_spectrum.unsqueeze(0))
        
        # 直接在输入层添加一个注意力层来观察
        attention = SimpleAttention(1000).to(device)
        weighted_spectrum, attention_weights = attention(test_spectrum.unsqueeze(0))
        
        # 将数据移回CPU进行绘图
        test_spectrum = test_spectrum.cpu()
        reconstructed = reconstructed.cpu()
        attention_weights = attention_weights.cpu()
        
        plt.figure(figsize=(15, 12))
        
        # 1. 原始光谱
        plt.subplot(4, 1, 1)
        plt.plot(test_spectrum.numpy(), 'b', label='原始光谱')
        plt.title(f'Epoch {epoch}: 原始光谱')
        plt.legend()
        plt.grid(True)
        
        # 2. 注意力权重
        plt.subplot(4, 1, 2)
        plt.plot(attention_weights[0].numpy(), 'r', label='注意力权重')
        plt.title('注意力权重分布')
        plt.legend()
        plt.grid(True)
        
        # 3. 重建结果
        plt.subplot(4, 1, 3)
        plt.plot(test_spectrum.numpy(), 'b', label='原始光谱', alpha=0.5)
        plt.plot(reconstructed[0].numpy(), 'r--', label='重建光谱')
        plt.title('重建结果对比')
        plt.legend()
        plt.grid(True)
        
        # 4. 损失值
        plt.subplot(4, 1, 4)
        mse = nn.MSELoss()(reconstructed[0], test_spectrum).item()
        plt.text(0.5, 0.5, f'MSE Loss: {mse:.6f}', 
                horizontalalignment='center',
                verticalalignment='center',
                transform=plt.gca().transAxes,
                fontsize=12)
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

def attention_loss(attention_weights):
    """
    修改注意力损失，更适合光谱数据特点
    """
    # 鼓励权重关注高强度区域
    intensity_loss = -torch.mean(attention_weights)
    
    # 鼓励权重的平滑性（相邻位置权重应该相近）
    smoothness_loss = torch.mean(torch.abs(attention_weights[:, 1:] - attention_weights[:, :-1]))
    
    return intensity_loss + 0.1 * smoothness_loss

def train_model(n_epochs=30):
    """训练模型并展示过程"""
    input_dim = 1000
    batch_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建模型和优化器
    model = SimpleAutoencoder(input_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # 生成固定的测试样本
    test_spectrum = generate_spectrum()
    
    # 训练循环
    losses = []
    print("开始训练...")
    for epoch in tqdm(range(n_epochs)):
        model.train()
        epoch_losses = []
        
        # 每个epoch训练50个批次
        for _ in range(50):
            spectra = generate_batch_spectra(batch_size).to(device)
            
            optimizer.zero_grad()
            reconstructed = model(spectra)
            
            # 只使用重建损失
            loss = criterion(reconstructed, spectra)
            
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        
        # 每5个epoch展示一次结果
        if (epoch + 1) % 5 == 0:
            print(f"\nEpoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.6f}")
            print(f"注意力权重: {model.attention_weights[0][0]}")
            visualize_attention_effect(model, test_spectrum, epoch+1)
    
    # 显示损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()
    
    return model, losses

# 运行训练
if __name__ == "__main__":
    model, losses = train_model()
