'''Test pytorch models using test dataset. Single model or multiply models can be tested'''
from validation import validation
from data_to_dataset import ImageLoader, ImageDataset
from model import MultiFTNet, load_config
import torch
from torch.utils.data import DataLoader
import cv2
import argparse
import os

def parse_args():
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_path_live', type=str, required=True)
    parser.add_argument('--data_path_fake', type=str, required=True)
    return parser.parse_args()

def main(args):
    model_list = []
    if os.path.isfile(args.model_path):
        conf = load_config(os.path.join(os.path.dirname(args.model_path), 'config.yaml'))
        model_list.append(args.model_path)
    else:
        model_list = [os.path.join(args.model_path, f) for f in os.listdir(args.model_path) if f.endswith('.pth')]
        conf = load_config(os.path.join(args.model_path, 'config.yaml'))

    vld = ImageLoader(args.data_path_live, 0)
    vsd = ImageLoader(args.data_path_fake, 1)
    vds = ImageDataset([vld, vsd,], img_preproc= lambda x : cv2.resize(x, (conf['model_params']['embedding_size'], conf['model_params']['embedding_size'])), f_preproc= None)
    data = DataLoader(vds, batch_size=conf['batch_size'], shuffle=False, pin_memory = True)
    
    for mdl in model_list:
        model = MultiFTNet(**conf['model_params'])
        model.load_state_dict(torch.load(mdl, map_location=torch.device('cpu'))) 
        metrics = validation(model, data, None, 'cpu')
        print(mdl)
        print(metrics)

if __name__ == '__main__':
    main(parse_args())
