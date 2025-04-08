'''Convert pytorch model to onnx and openvino formats'''
import multiprocessing as mp
if mp.get_start_method(allow_none=True) != 'spawn':
    mp.set_start_method('spawn')

import torch
import torch.nn as nn
from openvino import convert_model, serialize
import argparse
import os
from model import MultiFTNet, load_config


class Cover(nn.Module):
    def __init__(self, model):
        super(Cover, self).__init__()
        self.model = model

    def preprocess(self, x):
        x = x / 255.0
        return x

    def postprocess(self, x):
        sm = torch.nn.Softmax(dim=1)
        return sm(x)

    def forward(self, x):
        x = self.preprocess(x)
        y = self.model(x)
        prob = self.postprocess(y)
        return prob


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_path', type=str, default='', required=True, help="path to pytorch model")
    parser.add_argument('--out_folder', type=str, default='ovine_models/', required=True, help="path to the model output")
    return parser.parse_args()


def main(args):

    if not os.path.isfile(args.model_path):
        raise ValueError(f'Model path {args.model_path} doesn’t point to a valid file')
    
    conf = load_config(os.path.join(os.path.dirname(args.model_path), 'config.yaml'))
    
    m_name = os.path.splitext(os.path.basename(args.model_path))[0]
    model = MultiFTNet(**conf['model_params'])
    model.load_state_dict(torch.load(args.model_path, map_location='cpu'))

    model = Cover(model).eval()

    input_example = torch.randint(0, 255, (1, 3, conf['model_params']['embedding_size'], conf['model_params']['embedding_size']), dtype=torch.uint8)

    os.makedirs(args.out_folder, exist_ok=True)
    onnx_path = os.path.join(args.out_folder, m_name + ".onnx")
    ovine_path = os.path.join(args.out_folder, m_name + ".xml")
    
    torch.onnx.export(model, input_example,
                    onnx_path,
                    input_names=("image", ),
                    output_names=("classifier",),
                    opset_version=16,
                    )

    model = convert_model(onnx_path)
    serialize(model, ovine_path)

if __name__ == "__main__":
    main(parse_args())