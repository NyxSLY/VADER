import os
import fnmatch
from datetime import datetime

def find_and_delete_in_subdirs(pattern="your_pattern.txt", preview=True):
    current_dir = os.getcwd()
    print(f"当前目录: {current_dir}")
    print(f"查找模式: {pattern}\n")

    # 存储目录和匹配文件的信息
    directories = []
    matched_files = []

    # 遍历所有子目录
    for root, dirs, files in os.walk(current_dir):
        # 记录目录信息
        level = root.replace(current_dir, '').count(os.sep)
        dir_name = os.path.basename(root) or root
        directories.append({
            'path': root,
            'name': dir_name,
            'level': level
        })

        # 在当前目录中查找匹配文件
        for filename in files:
            if pattern in filename:
                file_path = os.path.join(root, filename)
                matched_files.append({
                    'path': file_path,
                    'size': os.path.getsize(file_path),
                    'time': datetime.fromtimestamp(os.path.getmtime(file_path)),
                    'dir_level': level
                })

    # 打印目录结构和匹配文件
    print("目录结构:")
    for dir_info in directories:
        indent = '  ' * dir_info['level']
        print(f"{indent}📁 {dir_info['name']}")

        # 打印该目录下的匹配文件
        dir_files = [f for f in matched_files if os.path.dirname(f['path']) == dir_info['path']]
        if dir_files:
            for file_info in dir_files:
                sub_indent = '  ' * (dir_info['level'] + 1)
                size_mb = file_info['size'] / (1024 * 1024)
                print(f"{sub_indent}📄 {os.path.basename(file_info['path'])}")
                print(f"{sub_indent}   大小: {size_mb:.2f} MB")
                print(f"{sub_indent}   修改时间: {file_info['time']}")

    # 统计信息
    if matched_files:
        total_size = sum(f['size'] for f in matched_files)
        print(f"\n找到 {len(matched_files)} 个匹配文件")
        print(f"总大小: {total_size / (1024 * 1024):.2f} MB")

        # 预览模式确认
        if preview:
            print("\n这是预览模式，不会删除文件")
            response = input("是否切换到删除模式? (y/n): ").lower()
            if response != 'y':
                return

        # 删除确认
        print("\n警告: 此操作将永久删除上述文件!")
        response = input("确定要删除这些文件吗? (y/n): ").lower()

        if response == 'y':
            deleted_count = 0
            failed_files = []

            for file_info in matched_files:
                try:
                    os.remove(file_info['path'])
                    deleted_count += 1
                    print(f"已删除: {file_info['path']}")
                except Exception as e:
                    failed_files.append((file_info['path'], str(e)))

            # 打印结果
            print(f"\n删除完成:")
            print(f"成功: {deleted_count} 个文件")

            if failed_files:
                print(f"失败: {len(failed_files)} 个文件")
                print("\n失败详情:")
                for path, error in failed_files:
                    print(f"- {path}")
                    print(f"  错误: {error}")
        else:
            print("操作已取消")
    else:
        print("\n未找到匹配的文件")

if __name__ == "__main__":
    try:
        # 设置要查找的文件模式
        pattern = "recon_x_value.txt"  # 修改为你要查找的文件模式

        # 默认使用预览模式
        preview_mode = False

        find_and_delete_in_subdirs(pattern, preview_mode)

    except KeyboardInterrupt:
        print("\n操作被用户中断")
    except Exception as e:
        print(f"发生错误: {str(e)}")
