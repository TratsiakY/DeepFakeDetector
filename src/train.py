'''Train method'''
from validation import validation
import torch
from tqdm import tqdm, trange

def train(model, optimizer, trainds, n_epochs, cls_loss, ftt_loss, valloss, validds, scheduler, m_name, aim_run = None, device = 'cuda', parallel = False, clip = 1):
    '''
    todo: Add documentation
    '''
    model = model.to(device)
    if parallel:
        model = torch.nn.DataParallel(model)
    f1 = 0
    for epoch in trange(n_epochs):
        run_loss = 0.
        run_loss_cls = 0.
        run_loss_ft = 0.
        
        model.train()

        for sample, labels, ft_sample in tqdm(trainds):

            optimizer.zero_grad()
            labels = labels.to(device, non_blocking=True)
            embeddings, feature_map = model(sample.to(device, non_blocking=True))
            # print(embeddings.shape, labels.shape)
            loss_cls = cls_loss(embeddings, labels)
            loss_fea = ftt_loss(feature_map, ft_sample.to(device, non_blocking=True))

            loss = 0.5*loss_cls + 0.5*loss_fea
            
            loss.backward()
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         aim_run.track(
            #             aim_run.Distribution(param.grad.detach().cpu().numpy()), 
            #             name=f"gradients/{name}", 
            #             step=epoch
            #         )
            total_norm = torch.norm(torch.stack([p.grad.norm() for p in model.parameters() if p.grad is not None ]), p=2)

            # Логируем в AIM
            if aim_run:
                aim_run.track(total_norm.item(), name="gradient_norm", step=epoch) 

            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            
            optimizer.step()
            

            run_loss += loss.detach().item()
            run_loss_cls += loss_cls.detach().item()
            run_loss_ft += loss_fea.detach().item()
        
        torch.cuda.empty_cache()
        if scheduler is not None:
            scheduler.step()   
        
        print('learning rate is ', optimizer.param_groups[0]['lr'])
        metrics = validation(model, validds, valloss, device)
        print(metrics)
        print(f'Epoch {epoch +1} done. ')

        if metrics['f1_score'] > f1:
            f1 = metrics['f1_score']
            torch.save(model.state_dict(), m_name + '__' + str(epoch+1) + '.pth')
        if aim_run:
            aim_run.track( run_loss / len(trainds), step= epoch, name = 'Epoch loss', context={ "subset":"train" })
            aim_run.track( run_loss_cls / len(trainds), step= epoch, name = 'Epoch classification loss', context={ "subset":"train" })
            aim_run.track( run_loss_ft / len(trainds), step= epoch, name = 'Epoch FT loss', context={ "subset":"train" })
            for key in metrics:
                if key in ['accuracy', 'precision', 'recall', 'f1_score', 'loss']:
                    aim_run.track(metrics[key], step= epoch, name = key, context={ f"subset":"Validation" })
                else:
                    for i, val in enumerate(metrics[key]):
                        aim_run.track(val, step= epoch, name = f'{key} score for class {i}', context={ f"subset":"Validation" })
                        
    return f1                   
                        
if __name__ == '__main__':
    pass