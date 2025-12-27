import os
import shutil
from sklearn.model_selection import train_test_split

# 設置原始資料集目錄和目標訓練/測試目錄
src_dataset_dir = 'CarDatasets_20'
train_dir = 'train'
test_dir = 'test'

# 為每個類別創建訓練和測試資料夾
for class_dir in os.listdir(src_dataset_dir):
    os.makedirs(os.path.join(train_dir, class_dir), exist_ok=True)
    os.makedirs(os.path.join(test_dir, class_dir), exist_ok=True)

    # 獲取每個類別資料夾內的所有圖片
    images = os.listdir(os.path.join(src_dataset_dir, class_dir))
    
    # 切分圖片為訓練集和測試集
    train_imgs, test_imgs = train_test_split(images, test_size=0.2, random_state=42)
    
    # 複製圖片到訓練集和測試集資料夾
    for img in train_imgs:
        shutil.copy(os.path.join(src_dataset_dir, class_dir, img),
                    os.path.join(train_dir, class_dir, img))
        
    for img in test_imgs:
        shutil.copy(os.path.join(src_dataset_dir, class_dir, img),
                    os.path.join(test_dir, class_dir, img))
