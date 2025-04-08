'''
A few classes to build datasets from the folders with images. 
'''
import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

def build_dataset(conf):
    ds = []
    fractions = {}
    total_samples = 0
    
    for key in conf:
        fractions[key] = 0
        for path in conf[key]:
            ds.append(ImageLoader(path, key))
            fractions[key] += len(ds[-1])
        total_samples += fractions[key]
    
    sorted_weights = [total_samples/fractions[key] for key in sorted(fractions.keys())]
    
    return ds, sorted_weights

def FTT_preproc(img, size):
    ft_sample = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return ft_sample

def generate_FT(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    fimg = np.log(np.abs(fshift)+1)
    maxx = np.max(fimg)
    minn = np.min(fimg)
    fimg = (fimg - minn +1) / (maxx - minn +1)
    return fimg

class ImageLoader(Dataset):
    '''
    читает картинки в папке и загружает их. Чекает annotations.pickle в папке, если есть, то лэйблы могут быть закружены как "мягкие"
    '''
    def __init__(self, folder_path, class_label, image_preproc=None, label_preproc=None):
        self.folder_path = folder_path
        self.class_label = class_label
        self.image_preproc = image_preproc
        self.label_preproc = label_preproc
        self.image_files = [f for f in os.listdir(folder_path) if os.path.splitext(f)[1].lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.gif'}]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        file_name = self.image_files[idx]
        file_path = os.path.join(self.folder_path, file_name)
        image = cv2.imread(file_path)
        if image is None:
            raise RuntimeError(f"Failed to load image: {file_path}")
        
        image = self._image_preproc(image)
        label = self._label_preproc(self.class_label)
        
        return image, label

    def _image_preproc(self, image):
        if self.image_preproc is not None:
            return self.image_preproc(image)
        else:
            return image
    
    def _label_preproc(self, label):
        if self.label_preproc is not None:
            return self.label_preproc(label)
        else:
            return torch.tensor(label, dtype=torch.long)

class ImageDataset(Dataset):
    '''
    формирует датасет из объектов ImageLoader
    '''
    def __init__(self, image_loaders, img_preproc = None, lbl_preproc = None, f_preproc = None):
        self.image_loaders = image_loaders
        self.loader_indices = []
        self.img_preproc = img_preproc
        self.lbl_preproc = lbl_preproc
        self.f_preproc = f_preproc
        # Формируем маппинг индексов к конкретному ImageLoader и его локальному индексу
        for loader_idx, loader in enumerate(self.image_loaders):
            self.loader_indices.extend([(loader_idx, i) for i in range(len(loader))])
    
    def __len__(self):
        return len(self.loader_indices)
    
    def __getitem__(self, idx):
        loader_idx, local_idx = self.loader_indices[idx]
        im, lb = self.image_loaders[loader_idx][local_idx]
        lb = self.lbl_preproc(lb) if self.lbl_preproc is not None else lb
        im = self.img_preproc(im) if self.img_preproc is not None else im
        fim = generate_FT(im)
        fim = self.f_preproc(fim) if self.f_preproc is not None else fim
        
        fim = torch.from_numpy(fim).float().unsqueeze(0)
        timage = torch.tensor(im[..., [2, 1, 0]], dtype=torch.float32).permute(2, 0, 1) / 255.0
        
        return timage, lb, fim
    
    
    
if __name__ == '__main__':
    
    pass
    