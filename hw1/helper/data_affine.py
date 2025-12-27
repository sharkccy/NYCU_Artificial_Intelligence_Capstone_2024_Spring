from imgaug import augmenters as iaa
from PIL import Image, ImageOps
import numpy as np
import imageio
import os


seq = iaa.Sequential([ 
    iaa.flip.Fliplr(1.0),
])

def process_images_with_augmentation(folder_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg')):
                try:
                    img_path = os.path.join(root, file)
                    img = imageio.imread(img_path)
                    img_aug = seq.augment_image(img)  
                    
                    
                    relative_path = os.path.relpath(root, folder_path)
                    output_dir = os.path.join(output_folder, relative_path)
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)

                    output_path = os.path.join(output_dir, file)
                    imageio.imwrite(output_path, img)  

                    aug_output_path = os.path.join(output_dir, f"aug_" + file)
                    imageio.imwrite(aug_output_path, img_aug)  
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")

folder_path = 'C:/講義與作業/人工智慧總整與實作/hw/hw1/CarDatasets_20_squeeze'
output_folder = 'C:/講義與作業/人工智慧總整與實作/hw/hw1/resized_data'
process_images_with_augmentation(folder_path, output_folder)
