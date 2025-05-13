import argparse
import os
from visualize_components import visualize_component_spectra

def main():
    parser = argparse.ArgumentParser(description="可视化VADE模型的MCR成分光谱")
    parser.add_argument("--model_path", type=str, required=True, 
                        help="训练好的模型路径，例如 './xxx/pth/epoch_50_acc_0.85_nmi_0.75_ari_0.80.pth'")
    parser.add_argument("--save_dir", type=str, default="./component_spectra",
                        help="保存可视化结果的目录")
    parser.add_argument("--wavenumber_path", type=str, default=None,
                        help="波数数据的路径（可选）")
    
    args = parser.parse_args()
    
    # 检查模型路径是否存在
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"模型文件不存在: {args.model_path}")
    
    # 运行可视化
    print(f"正在可视化模型 {args.model_path} 的成分光谱...")
    visualize_component_spectra(
        model_path=args.model_path,
        save_dir=args.save_dir,
        wavenumber_path=args.wavenumber_path
    )
    print("可视化完成！")

if __name__ == "__main__":
    main() 