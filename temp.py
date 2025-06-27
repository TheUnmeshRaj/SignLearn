import os
import shutil

source_folders = [
    'alphabets(Learn A-F)',
    'alphabets(Learn G-L)',
    'alphabets(Learn M-S)',
    'alphabets(Learn T-Z)'
]

target_dir = 'ref_imgs'
os.makedirs(target_dir, exist_ok=True)

for source_dir in source_folders:
    if not os.path.exists(source_dir):
        continue
    for folder in os.listdir(source_dir):
        folder_path = os.path.join(source_dir, folder)
        if os.path.isdir(folder_path):
            img_files = sorted(os.listdir(folder_path))
            if len(img_files) > 4:
                src_img = os.path.join(folder_path, img_files[4])
                dest_img = os.path.join(target_dir, f"{folder}.jpg")
                shutil.copyfile(src_img, dest_img)
                print(f"Copied: {dest_img}")
