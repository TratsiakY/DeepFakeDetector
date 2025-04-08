'''
The list with augmentations we would like to apply to images durint the train
'''
import albumentations as A
import cv2 

def preprocess(img, size, augm=None):
    img = cv2.resize(img, size)
    aimg = apl_augmentation(img, augm) if augm is not None else img
    return aimg

def apl_augmentation(img, augmentation):
    '''
    Применяет аугментацию к изображению
    '''
    return augmentation(image=img)['image']
    
augmentations = A.Compose([
    # Цветовые трансформации
    A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.15, p=0.7),
    A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.3),
    A.ToGray(p=0.1),

    # Геометрические трансформации
    A.Rotate(limit=15, border_mode=0, p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.1),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, border_mode=0, p=0.5),

    # # Искажения формы
    # A.ElasticTransform(alpha=0.5, sigma=30, alpha_affine=10, p=0.2),
    # A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.2),
    # A.OpticalDistortion(distort_limit=0.1, shift_limit=0.1, p=0.2),

    # Регуляризация (шумы, дропы и т.п.)
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    A.MotionBlur(blur_limit=3, p=0.2),
    A.CoarseDropout(max_holes=8, max_height=8, max_width=8, fill_value=0, p=0.3),

    # Приведение к нужному размеру (если нужно)
    # A.Resize(height=224, width=224)
])