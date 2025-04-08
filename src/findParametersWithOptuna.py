'''Finding optimal parameters using optuna'''

import torch
from model import MultiFTNet, save_config, load_config
from data_to_dataset import ImageDataset, FTT_preproc, build_dataset
from torch.utils.data import DataLoader
from train import train
from augmentations import augmentations, preprocess
import os
from methods import create_timestamped_folder, seed_everything
import cv2
import optuna
from functools import partial


def objective(trial, conf, train_set, val_set, weights, device):

    seed_everything(conf['seed'])
    
    conf['optimizer']['lr'] =  trial.suggest_float('learning_rate', 1e-8, 1e-2, log = True)
    # conf['optimizer']['optimizer'] = trial.suggest_categorical("optimizer", ["RAdam","Adam", "RMSprop", "SGD"])
    conf['batch_size'] = trial.suggest_int('batch_size', 16, 128)
    conf['optimizer']['weight_decay'] = trial.suggest_float('weight_decay', 1e-5, 1e-2, log = True)
    
    model = MultiFTNet(**conf['model_params'])
    if conf['fine_tune'] is not None:
        model.load_state_dict(torch.load(conf['fine_tune'], map_location=torch.device(device))) 
    ftt_size = [2*s for s in conf['model_params']['conv6_kernel']]

    cls_loss = getattr(torch.nn, conf['cls_loss']['loss'])(reduction=conf['cls_loss']['loss_reduction'], **({'weight': torch.tensor(conf['cls_loss']['loss_weight']).to(device)} if 'loss_weight' in conf['cls_loss'] and conf['cls_loss']['loss_weight'] is not None else {}))
    ftt_loss = getattr(torch.nn, conf['ftt_loss']['loss'])(reduction=conf['ftt_loss']['loss_reduction'])
    val_loss = getattr(torch.nn, conf['cls_loss']['loss'])(reduction=conf['cls_loss']['loss_reduction'],)
    optimizer = getattr(torch.optim, conf['optimizer']['optimizer'])(model.parameters(), **{k: v for k, v in conf['optimizer'].items() if k != "optimizer"})
    scheduler_class = getattr(torch.optim.lr_scheduler, conf['scheduler']['type'])
    scheduler = scheduler_class(optimizer, **{k: v for k, v in conf['scheduler'].items() if k != "type"})
    
    train_dataset = ImageDataset(train_set, img_preproc= lambda x : preprocess(x, (conf['model_params']['embedding_size'], conf['model_params']['embedding_size']), augmentations), f_preproc= lambda x: FTT_preproc(x, ftt_size))
    conf['cls_loss']['loss_weight'] = weights
    
    val_dataset = ImageDataset(val_set, img_preproc= lambda x : cv2.resize(x, (conf['model_params']['embedding_size'], conf['model_params']['embedding_size']), interpolation=cv2.INTER_AREA), f_preproc= lambda x: FTT_preproc(x, ftt_size))
    train_dataloader = DataLoader(train_dataset, batch_size=conf['batch_size'], shuffle=conf['shuffle'], pin_memory = True)
    valid_dataloader = DataLoader(val_dataset, batch_size=conf['batch_size'], shuffle=False, pin_memory = True)
    
    m_path = create_timestamped_folder(conf['path'])
    m_name = os.path.join(m_path, conf['model'])
    save_config(m_path, conf)

    f1 = train(model, optimizer, train_dataloader, conf['n_epoch'], cls_loss, ftt_loss, val_loss, valid_dataloader, scheduler = scheduler, m_name = m_name, aim_run=None, device = device, parallel=False) 
    
    print(f'The best f1 value {f1} is for the {conf['model']}')
    return f1

if __name__ == '__main__':
    storage_path = None#"sqlite:///path_to_db.db"
    study_name = "hyperparam_search"

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    conf = load_config('src/config.yaml')
    
    train_data, weights = build_dataset(conf['train_dataset'])
    valid_data, _ = build_dataset(conf['valid_dataset'])
    
    func = partial(objective, conf=conf, train_set = train_data, val_set = valid_data, weights = weights, device = device)
    
    if storage_path is not None:
        study = optuna.create_study(direction ='maximize', study_name=study_name, storage=storage_path, load_if_exists=True,sampler=optuna.samplers.TPESampler())
    else:
        study = optuna.create_study(direction ='maximize', study_name=study_name, sampler=optuna.samplers.TPESampler())
    study.optimize(func, n_trials= 30, show_progress_bar=True)

    params = study.best_params
    print('Fitted values', params)


