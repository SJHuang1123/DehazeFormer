import os
import re

def numeric_key(filename):
    # 從 q_123.mat 取出 123 作為排序依據
    match = re.search(r'q_(\d+)\.mat', filename)
    return int(match.group(1)) if match else float('inf')

def rename_to_sequential(dir_path):
    files = [f for f in os.listdir(dir_path) if f.endswith('.mat')]
    files.sort(key=numeric_key)  # 依數字排序

    for i, filename in enumerate(files):
        new_name = f"q_{i}.mat"
        src = os.path.join(dir_path, filename)
        dst = os.path.join(dir_path, new_name)
        if src != dst:
            os.rename(src, dst)
            print(f"Renamed: {filename} -> {new_name}")

# 使用範例
rename_to_sequential('/home/q36131207/HSID_dataset/AVIRIS/qval/trans')
