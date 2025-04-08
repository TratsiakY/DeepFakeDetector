'''Main method to run the train process'''
import torch
from model import MultiFTNet, save_config, load_config
from data_to_dataset import ImageDataset, FTT_preproc, build_dataset
from torch.utils.data import DataLoader
from train import train
from aim import Run
from augmentations import augmentations, preprocess
import os
import argparse
from methods import create_timestamped_folder, seed_everything
import cv2

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config_path', default = 'src/config.yaml', type=str, required=False, help="path to config YAML file")
    return parser.parse_args()

def main(args):
    run = None
    conf = load_config(args.config_path)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    conf['path'] = create_timestamped_folder(conf['path'])
    conf['model'] = os.path.join(conf['path'], conf['model'])
    print(conf['model'])
    
    seed_everything(conf['seed'])
    
    model = MultiFTNet(**conf['model_params'])
    if conf['fine_tune'] is not None:
        model.load_state_dict(torch.load(conf['fine_tune'], map_location=torch.device(device))) 
    ftt_size = [2*s for s in conf['model_params']['conv6_kernel']]
    
    train_data, weights = build_dataset(conf['train_dataset'])
    train_dataset = ImageDataset(train_data, img_preproc= lambda x : preprocess(x, (conf['model_params']['embedding_size'], conf['model_params']['embedding_size']), augmentations), f_preproc= lambda x: FTT_preproc(x, ftt_size))
    conf['cls_loss']['loss_weight'] = weights
    
    cls_loss = getattr(torch.nn, conf['cls_loss']['loss'])(reduction=conf['cls_loss']['loss_reduction'], **({'weight': torch.tensor(conf['cls_loss']['loss_weight']).to(device)} if 'loss_weight' in conf['cls_loss'] and conf['cls_loss']['loss_weight'] is not None else {}))
    ftt_loss = getattr(torch.nn, conf['ftt_loss']['loss'])(reduction=conf['ftt_loss']['loss_reduction'])
    val_loss = getattr(torch.nn, conf['cls_loss']['loss'])(reduction=conf['cls_loss']['loss_reduction'],)
    optimizer = getattr(torch.optim, conf['optimizer']['optimizer'])(model.parameters(), **{k: v for k, v in conf['optimizer'].items() if k != "optimizer"})
    scheduler_class = getattr(torch.optim.lr_scheduler, conf['scheduler']['type'])
    scheduler = scheduler_class(optimizer, **{k: v for k, v in conf['scheduler'].items() if k != "type"})
    
    valid_data, _ = build_dataset(conf['valid_dataset'])
    val_dataset = ImageDataset(valid_data, img_preproc= lambda x : cv2.resize(x, (conf['model_params']['embedding_size'], conf['model_params']['embedding_size']), interpolation=cv2.INTER_AREA), f_preproc= lambda x: FTT_preproc(x, ftt_size))
    
    train_dataloader = DataLoader(train_dataset, batch_size=conf['batch_size'], shuffle=conf['shuffle'], pin_memory = True)
    valid_dataloader = DataLoader(val_dataset, batch_size=conf['batch_size'], shuffle=False, pin_memory = True)
    
    run = Run(experiment="spoofing dataset and model debug")
    run['hparams'] = conf
    save_config(conf['path'], conf)
    train(model, optimizer, train_dataloader, conf['n_epoch'], cls_loss, ftt_loss, val_loss, valid_dataloader, scheduler = scheduler, m_name = conf['model'], aim_run=run, device = device, parallel=False)

if __name__ == '__main__':
    main(parse_args())