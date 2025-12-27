from PIL import Image
import os

def resize_and_pad(img, size, fill=0):
    img.thumbnail((size, size), Image.Resampling.LANCZOS)
    new_img = Image.new("RGB", (size, size), fill)
    x = (size - img.size[0]) // 2
    y = (size - img.size[1]) // 2
    new_img.paste(img, (x, y))
    return new_img

def process_images_from_folder(folder_path, output_folder, output_size):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg')):
                try:
                    img_path = os.path.join(root, file)
                    img = Image.open(img_path)
                    new_img = resize_and_pad(img, output_size)
                    # 構建輸出圖片的完整路徑
                    output_path = os.path.join(output_folder, file)
                    new_img.save(output_path)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")

# 指定原始圖片的資料夾路徑和輸出圖片的大小
folder_path = 'C:/講義與作業/人工智慧總整與實作/hw/hw1/CarDatasets_20'
output_folder = 'C:/講義與作業/人工智慧總整與實作/hw/hw1/resized_data'  # 調整後的圖片保存的位置
output_size = 256  # 想要縮放到的正方形圖片大小

process_images_from_folder(folder_path, output_folder, output_size)
