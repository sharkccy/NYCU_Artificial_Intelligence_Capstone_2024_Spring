from PIL import Image, ImageOps
import os

def resize_and_crop(img, target_size):
    
    ratio = target_size / min(img.size)
    new_size = (int(round(img.size[0] * ratio)), int(round(img.size[1] * ratio)))
    img = img.resize(new_size, Image.LANCZOS)
    
    
    left = (img.width - target_size) / 2
    top = (img.height - target_size) / 2
    right = (img.width + target_size) / 2
    bottom = (img.height + target_size) / 2

    
    img = img.crop((left, top, right, bottom))
    return img

def process_images_from_folder(folder_path, output_folder, output_size=224):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg')):
                try:
                    img_path = os.path.join(root, file)
                    img = Image.open(img_path)
                    img = img.convert('RGB')  
                    new_img = resize_and_crop(img, output_size)
                    # 构建输出图片的完整路径
                    relative_path = os.path.relpath(root, folder_path)
                    output_dir = os.path.join(output_folder, relative_path)
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    
                    # 构建输出图片的完整路径
                    output_path = os.path.join(output_dir, file)
                    new_img.save(output_path, 'JPEG')  
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")


folder_path = 'C:/講義與作業/人工智慧總整與實作/hw/hw1/CarDatasets_20_origin'
output_folder = 'C:/講義與作業/人工智慧總整與實作/hw/hw1/resized_data'
output_size = 224  

process_images_from_folder(folder_path, output_folder, output_size)
