import os

from PIL import Image
import numpy as np
import cv2

from torch.utils.data import Dataset
from barcode_extractor import find_barcode


class BarcodeDataset(Dataset):
    def __init__(self, root_dir, class_folders, train = True, random_seed=42):
        """
        Args:
            root_dir (string): Directory with all the barcode folders (e.g., '/content/drive/MyDrive/barcode/').
            class_folders (list): List of class folder names (e.g., ['strap', 'reflection']).
            transform (callable, optional): Optional transform to be applied on a sample.
            random_seed (int): Seed for reproducible data collection (e.g., shuffling for potential future use).
        """
        self.root_dir = root_dir
        self.class_folders = class_folders
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {folder: i for i, folder in enumerate(class_folders)}

        for class_folder in class_folders:
            class_path = os.path.join(self.root_dir, class_folder)
            if not os.path.isdir(class_path):
                print(f"Warning: Directory not found: {class_path}. Skipping.")
                continue

            all_items = os.listdir(class_path)
            if train:
              all_images_in_class = [
                  os.path.join(class_path, img_name)
                  for img_name in all_items[:len(all_items)//2]
                  if img_name.lower().endswith(('.png'))
              ]

              repeat = 3
              for _ in range(repeat):
                self.image_paths.extend(all_images_in_class)

              self.labels.extend([self.class_to_idx[class_folder]] * len(all_images_in_class)*repeat)

            else:
              all_images_in_class = [
                  os.path.join(class_path, img_name)
                  for img_name in all_items[len(all_items)//2:]
                  if img_name.lower().endswith(('.png'))
              ]

              self.image_paths.extend(all_images_in_class)
              self.labels.extend([self.class_to_idx[class_folder]] * len(all_images_in_class))

        if not self.image_paths:
            raise RuntimeError(f"No images found in {root_dir} with folders {class_folders}. Please check paths and image extensions.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path, 0)
        barcode, _ = find_barcode(image) # Get the barcode crop
        barcode = Image.fromarray(barcode).convert('RGB')

        # barcode = cv2.cvtColor(barcode, cv2.COLOR_GRAY2RGB) # Ensure 3 channels for model input
        image = barcode

        label = self.labels[idx]
        return image, label

