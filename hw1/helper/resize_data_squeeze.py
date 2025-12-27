from PIL import Image
import cv2 as cv
import os

def process_images_from_folder(folder_path, output_folder, output_size):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg')):
                try:
                    img_path = os.path.join(root, file)
                    img = Image.open(img_path)
                    img = img.convert('RGB')  # 添加这行来转换图像模式到RGB
                    img = img.resize((output_size, output_size), Image.Resampling.LANCZOS)
                    
                    # 获取文件的相对路径，并在输出目录中创建相应的目录结构
                    relative_path = os.path.relpath(root, folder_path)
                    output_dir = os.path.join(output_folder, relative_path)
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    
                    # 构建输出图片的完整路径
                    output_path = os.path.join(output_dir, file)
                    img.save(output_path, 'JPEG')  # 使用PIL的save方法保存图片
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")

# 指定原始图片的文件夹路径和输出图片的大小
folder_path = 'C:/講義與作業/人工智慧總整與實作/hw/hw1/CarDatasets_20_origin'
output_folder = 'C:/講義與作業/人工智慧總整與實作/hw/hw1/resized_data'  # 调整后的图片保存的位置
output_size = 256  # 想要缩放到的正方形图片大小

process_images_from_folder(folder_path, output_folder, output_size)
