import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

def create_BUSI_dataset():
    # 1. 定义文件路径
    root_dir = Path("/IVQ-CBM/Data/Dataset_BUSI_with_GT") 
    output_dir = Path("/IVQ-CBM/dataset/BUSI")

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出目录已确认/创建: {output_dir}")

    subfolders = sorted([d for d in root_dir.iterdir() if d.is_dir()])
    label_map = {folder.name: i for i, folder in enumerate(subfolders)}
    
    print(label_map)
    
    dataList = []
    labelList = []
    
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
    
    for folder_name, label in label_map.items():
        subfolder_path = root_dir / folder_name
        
        print(f"\n--- 正在处理文件夹: {folder_name} (标签: {label}) ---")
        
        image_paths = [p for p in subfolder_path.glob('*') if p.suffix.lower() in image_extensions]
        
        if not image_paths:
            print(f"警告：在 {subfolder_path} 中未找到任何图像文件。")
            continue

        for image_path in tqdm(image_paths, desc=f"读取 {folder_name}"):
            try:
                image = cv2.imread(str(image_path))
                if image is not None:
                    dataList.append(image)
                    labelList.append(label)
                else:
                    print(f"警告：无法读取文件 {image_path}，已跳过。")
            except Exception as e:
                print(f"读取文件 {image_path} 时发生错误: {e}")

    if not dataList:
        print("\n错误：未能加载任何数据，程序终止。")
        return

    dataList_np = np.array(dataList, dtype=np.uint8)
    labelList_np = np.array(labelList, dtype=np.uint8)


    data_save_path = output_dir / 'dataList.npy'
    label_save_path = output_dir / 'labelList.npy'
    
    np.save(data_save_path, dataList_np)
    np.save(label_save_path, labelList_np)

    print(f"  - {data_save_path}")
    print(f"  - {label_save_path}")


if __name__ == "__main__":
    BUSI_dataset = create_BUSI_dataset()