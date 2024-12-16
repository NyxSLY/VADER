import json
from grid_search import generate_combinations


param_grid = {
    'lamb1': [0, 1, 1e3, 1e5],
    'lamb2': [0.1, 1.0, 10.0],
    'lamb4': [0, 1, 1e3, 1e5],
    'lamb5': [0, 0.1, 0.5],
    'lamb6': [0, 0.1, 0.5]
}
combinations_file = "param_combinations.json"
total_combinations = generate_combinations(param_grid, combinations_file)
print(f"总共生成了 {total_combinations} 个参数组合")


'''
# 在主控PC上运行：
param_grid = {
    'lamb1': [0, 1, 1e3, 1e5],
    'lamb2': [0.1, 1.0, 10.0],
    'lamb4': [0, 1, 1e3, 1e5],
    'lamb5': [0, 0.1, 0.5],
    'lamb6': [0, 0.1, 0.5]
}
combinations_file = "param_combinations.json"
total_combinations = generate_combinations(param_grid, combinations_file)
print(f"总共生成了 {total_combinations} 个参数组合")

# 在PC1上运行：
grid_search = GridSearch(
    model_class=YourModel,
    data=data,
    labels=labels,
    num_classes=num_classes,
    base_params=base_params,
    device=device,
    pc_id=0,  # PC1的ID为0
    total_pcs=2,  # 总共2台PC
    combinations_file=combinations_file
)
results, best_result = grid_search.search(project_dir)

# 在PC2上运行：
grid_search = GridSearch(
    # ... 其他参数相同
    pc_id=1,  # PC2的ID为1
    total_pcs=2,
    combinations_file=combinations_file
)
results, best_result = grid_search.search(project_dir)
'''
