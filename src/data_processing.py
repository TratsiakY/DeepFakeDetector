'''
The cript to preprocess datasets to unify folders structure for subsequent use in dataset creation
'''
import os
import cv2
from tqdm import tqdm
import argparse

def get_all_images_in_folder(folder_path):
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp', 'mp4', 'avi', 'mov']  # Расширения изображений
    image_files = []

    for root, dirs, files in os.walk(folder_path):  # Обход всех папок и подпапок
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):  # Проверка на расширение изображения
                image_files.append(os.path.join(root, file))  # Добавление полного пути к файлу

    return image_files


def process_directories(A, B):
    '''The methot for getting and investigate all subfolders in A forlder and find those mathing to B'''
    all_images = []
    dirs = []
    for directory in A:
        dirlist = [os.path.join(directory, item) for item in os.listdir(directory) if os.path.isdir(os.path.join(directory, item))]
        dirs.extend(dirlist)
    
    while dirs:
        folder = dirs.pop(0)
        if os.path.basename(folder) in B:
            all_images.extend(get_all_images_in_folder(folder))
            # print(len(all_images))
        else:
            dirlist = [os.path.join(folder, item) for item in os.listdir(folder) if os.path.isdir(os.path.join(folder, item))]
            dirs.extend(dirlist)
    return all_images


def read_image(img, output_size, i, out, preprocess):
    if img is not None:
        img = cv2.resize(img, output_size)
        img = preprocess(img) if preprocess is not None else img
        out_path = os.path.join(out, i +'.jpg')
        cv2.imwrite(out_path, img)
                
                
def is_image(file_path):
    """Проверка, является ли файл изображением"""
    img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']
    _, ext = os.path.splitext(file_path)
    return ext.lower() in img_extensions


def process_data(images, out, output_size, preprocess = None, every = 10): #обработка или картинки или видео
    for i, im_path in enumerate(tqdm(images)):
        if is_image(im_path):
            img = cv2.imread(im_path)
            read_image(img, output_size, str(i), out, preprocess)
        else:
            stream = cv2.VideoCapture(im_path)
            total_frames = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Читаем каждый кадр в цикле for
            for frame_index in range(total_frames):
                ret, frame = stream.read()
                if not ret:
                    continue  # Если кадры закончились, завершаем цикл
                if frame_index % every == 0:
                # Обрабатываем каждый кадр видео
                    read_image(frame, output_size, str(i) + '_' + str(frame_index), out, preprocess)
            stream.release()

def data_preprocess_pipeine(A, B, out_dir, preprocess = None, output_size = (224,224), process_every = 10):
    os.makedirs(out_dir, exist_ok=True)
    images = process_directories(A, B)
    process_data(images, out=out_dir, output_size = output_size, preprocess=preprocess,every = process_every)
    
def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--root_folder', type=str, default='', required=True, help="path to a folder where a target folder(s) should be")
    parser.add_argument('--target_folder', type=str, default='', required=True, help="target folder name we should find in the root folder")
    parser.add_argument('--output_folder', type=str, default='', required=True, help="path to the output folder")
    parser.add_argument('--out_size', type=int, default=128, required=True, help="The AxA size of the output image")
    parser.add_argument('--process_every', type=int, default=1, required=False, help="Parameter is used only for a video. It indicates every frame should be processed")
    return parser.parse_args()

def main(args):
    # Пример использования
    root_folder = [args.root_folder]
    target_folder = [args.target_folder]
    data_preprocess_pipeine(root_folder, target_folder, args.output_folder, preprocess = None, output_size=(args.out_size, args.out_size),process_every = args.process_every)
    
if __name__ == '__main__':
    main(parse_args())