
import models.train as mod
import torch
import configs
from torchvision.utils import save_image

model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
model.num_features = 384

def forward_pass(model, x, device):
    with torch.no_grad():
        return model(x.to(device), is_training=True)['x_norm_patchtokens'].detach().cpu()
model.forward_pass = forward_pass
dataset = 'orgaquant'
output = mod.train(model,
                    dataset,
                    split=0,
                    seed=0,
                    download=False,
                    batch_size=4,
                    max_epochs=1,
                    optimizer_params={'lr': 0.00001},
                    task_specific={'num_patches': 1,
                                    'num_queries': 50,
                                    'dataset': dataset,
                                    'cost_giou': 2,
                                    'eos_coef': 0.1,
                                    'cost_class': 1,
                                    'cost_bbox': 5,
                                    'img_size': configs.dataset_config[dataset].size[0],
                                    'num_decoder_heads': 8,
                                    'num_decoder_layers': 6})

print(output['metrics'])
