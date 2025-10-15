import ast
from pathlib import Path

def load_results_from_files(file_paths):
    """
    从多个txt文件中加载实验结果，并存入一个嵌套字典中以便快速查询。

    参数:
    file_paths (list): 一个包含所有txt文件路径的列表。

    返回:
    dict: 一个嵌套字典，结构为 {fold_str: {alpha_str: {trial_str: iou_value}}}
          例如: {'Fold-0': {'alpha-0.16': {'trial-1': 0.6964}}}
    """
    results_cache = {}
    
    for file_path in file_paths:
        path = Path(file_path)
        if not path.is_file():
            print(f"警告：文件 '{file_path}' 不存在，已跳过。")
            continue
            
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    parts = line.split('\t ')
                    if len(parts) == 4:
                        fold_str, alpha_str, trial_str, metrics_str = parts
                        
                        # 解析iou值
                        metrics_dict = ast.literal_eval(metrics_str)
                        iou_value = metrics_dict.get('iou')

                        # 初始化嵌套字典
                        if fold_str not in results_cache:
                            results_cache[fold_str] = {}
                        if alpha_str not in results_cache[fold_str]:
                            results_cache[fold_str][alpha_str] = {}
                        
                        # 存储iou结果
                        results_cache[fold_str][alpha_str][trial_str] = iou_value

                except (ValueError, SyntaxError) as e:
                    print(f"警告：无法解析文件 '{file_path}' 中的行: '{line}'。错误: {e}")

    return results_cache

def check_experiment_exists(fold, alpha, trial, results_cache):
    """
    校验指定的 fold, alpha, trial 组合是否已经存在于加载的结果中。

    参数:
    fold (int or str): Fold 的编号 (例如: 0 or "0")。
    alpha (float or str): Alpha 的值 (例如: 0.16 or "0.16")。
    trial (int or str): Trial 的编号 (例如: 1 or "1")。
    results_cache (dict): 由 load_results_from_files 函数生成的结果缓存。

    返回:
    tuple: 一个元组 (bool, float or None)。
           如果存在，返回 (True, iou_value)。
           如果不存在，返回 (False, None)。
    """
    # 将传入的参数格式化为与文件中一致的字符串
    fold_str = f"Fold-{fold}"
    alpha_str = f"alpha-{alpha}"
    trial_str = f"trial-{trial}"

    # 使用 .get() 方法安全地查询嵌套字典
    # .get(key, {}) 如果键不存在，会返回一个空字典，避免了KeyError
    iou_result = results_cache.get(fold_str, {}).get(alpha_str, {}).get(trial_str)

    if iou_result is not None:
        return (True, iou_result)
    else:
        return (False, None)

# --- 主程序入口和使用示例 ---
if __name__ == "__main__":
    
    # 为了演示，我们先创建一些示例文件
    print("--- 正在设置演示环境 ---")
    Path("results").mkdir(exist_ok=True)
    
    data1 = """Fold-0	 alpha-0.16	 trial-1	 {'iou': 0.6964724384173749, 'color_blind_iou': 0.6964724384173749, 'accuracy': 0.9432388624865622}
Fold-0	 alpha-0.16	 trial-4	 {'iou': 0.6992522188127035, 'color_blind_iou': 0.6992522188127035, 'accuracy': 0.9437730344065963}
Fold-0	 alpha-0.16	 trial-5	 {'iou': 0.693249968182294, 'color_blind_iou': 0.693249968182294, 'accuracy': 0.9442285411042217}
"""
    data2 = """Fold-1	 alpha-0.16	 trial-1	 {'iou': 0.7487348372764256, 'color_blind_iou': 0.7487348372764256, 'accuracy': 0.9520470118831666}
Fold-1	 alpha-0.16	 trial-2	 {'iou': 0.7427089968508827, 'color_blind_iou': 0.7427089968508827, 'accuracy': 0.9499891455103449}
Fold-1	 alpha-0.32	 trial-1	 {'iou': 0.8100000000000000, 'color_blind_iou': 0.81, 'accuracy': 0.96}
"""
    
    with open("results/log1.txt", "w") as f: f.write(data1)
    with open("results/log2.txt", "w") as f: f.write(data2)
    print("创建了示例文件: results/log1.txt, results/log2.txt")
    print("--- 演示环境设置完毕 ---\n")


    # 1. 定义你的结果文件路径列表
    file_paths = ["results/log1.txt", "results/log2.txt", "results/non_existent_file.txt"]
    
    # 2. 一次性加载所有结果到缓存
    print(f"正在从 {file_paths} 加载结果...")
    existing_results = load_results_from_files(file_paths)
    print("结果加载完成。\n")

    # 3. 进行校验
    print("--- 开始校验 ---")

    # 示例1：校验一个存在的结果
    fold_to_check, alpha_to_check, trial_to_check = 1, 0.16, 2
    exists, iou = check_experiment_exists(fold_to_check, alpha_to_check, trial_to_check, existing_results)
    if exists:
        print(f"检查 (Fold-{fold_to_check}, alpha-{alpha_to_check}, trial-{trial_to_check}): ✅ 存在，IOU值为 {iou:.6f}")
    else:
        print(f"检查 (Fold-{fold_to_check}, alpha-{alpha_to_check}, trial-{trial_to_check}): ❌ 不存在")

    # 示例2：校验另一个存在的结果
    fold_to_check, alpha_to_check, trial_to_check = 1, 0.32, 1
    exists, iou = check_experiment_exists(fold_to_check, alpha_to_check, trial_to_check, existing_results)
    if exists:
        print(f"检查 (Fold-{fold_to_check}, alpha-{alpha_to_check}, trial-{trial_to_check}): ✅ 存在，IOU值为 {iou:.6f}")
    else:
        print(f"检查 (Fold-{fold_to_check}, alpha-{alpha_to_check}, trial-{trial_to_check}): ❌ 不存在")

    # 示例3：校验一个不存在的结果 (trial-10不存在)
    fold_to_check, alpha_to_check, trial_to_check = 0, 0.16, 10
    exists, iou = check_experiment_exists(fold_to_check, alpha_to_check, trial_to_check, existing_results)
    if exists:
        print(f"检查 (Fold-{fold_to_check}, alpha-{alpha_to_check}, trial-{trial_to_check}): ✅ 存在，IOU值为 {iou:.6f}")
    else:
        print(f"检查 (Fold-{fold_to_check}, alpha-{alpha_to_check}, trial-{trial_to_check}): ❌ 不存在")

    # 示例4：校验一个不存在的结果 (alpha-0.5不存在)
    fold_to_check, alpha_to_check, trial_to_check = 1, 0.5, 1
    exists, iou = check_experiment_exists(fold_to_check, alpha_to_check, trial_to_check, existing_results)
    if exists:
        print(f"检查 (Fold-{fold_to_check}, alpha-{alpha_to_check}, trial-{trial_to_check}): ✅ 存在，IOU值为 {iou:.6f}")
    else:
        print(f"检查 (Fold-{fold_to_check}, alpha-{alpha_to_check}, trial-{trial_to_check}): ❌ 不存在")