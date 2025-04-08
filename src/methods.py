''' Some supportive methods'''
from datetime import datetime
import os
import numpy as np
import torch
import random

def create_timestamped_folder(base_path):
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")  # Формат YYYY_MM_DD_HH_MM
    folder_path = os.path.join(base_path, timestamp)
    os.makedirs(folder_path, exist_ok=True)  # Создаст все родительские папки, если их нет
    return folder_path

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    return seed   

if __name__ == '__main__':
    pass