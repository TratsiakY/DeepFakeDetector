'''OpenVino model inference'''
import multiprocessing as mp
if mp.get_start_method(allow_none=True) != 'spawn':
    mp.set_start_method('spawn')
    
import cv2
from openvino.runtime import Core
import numpy as np
import argparse

class OVINEModel():
    """ Class that allows working with OpenVINO Runtime model. """

    def __init__(self, model_path, device="AUTO"):
        self.core = Core()
        self.model = self.core.read_model(model_path)
        self.compiled_model = self.core.compile_model(self.model, device)
        
        # Получение входных и выходных слотов
        self.input_layer = self.compiled_model.input(0)
        
        # Получаем входную форму
        self.input_shape = self.input_layer.shape
        # print(self.input_shape)
        # print("=== Информация о модели ===")
        # print(f"Входной слой: {self.input_layer.get_any_name()}")
        # print(f"Форма входного слоя: {self.input_layer.shape}")
        # for i, output in enumerate(self.compiled_model.outputs):
        #     print(f"Выходной слой {i}: {output.get_any_name()}")
        #     print(f"Форма выхода {i}: {output.shape}")
    
    def preprocess(self, image):
        image = cv2.resize(image, (self.input_shape[2], self.input_shape[3]), interpolation=cv2.INTER_AREA) 
        image = np.transpose(image, (2, 0, 1))[None]
        return image
    
    def predict(self, image):
        image = self.preprocess(image)
        result =  self.compiled_model(image)
        return result[0][0]


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--ovine_model', type=str, default='', required=True, help="path to ovine model")
    parser.add_argument('--img', type=str, default='', required=True, help="path to test image")
    return parser.parse_args()

def main(args):
    try:
        model = OVINEModel(args.ovine_model)
    except:
        raise ImportError(f'Problem loading {args.ovine_model}')
    
    try:
        img = cv2.imread(args.img)
        if img is None:
            raise ValueError(f"Could not read image from {args.img}")
    except:
        raise ImportError(f'Problem loading {args.img}')
    
    print(model.predict(img))
    
if __name__ == '__main__':
    main(parse_args())