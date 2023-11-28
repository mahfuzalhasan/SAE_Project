import torch
import os

def save_model(model, opt, e, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_file_path = os.path.join(save_dir, 'model_best.pth')
    states = {
        'epoch': e,
        'model': model.state_dict(),
        'optimizer': opt.state_dict()
    }
    torch.save(states, save_file_path)