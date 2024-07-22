import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4" 
os.environ['http_proxy'] = 'http://10.16.35.10:13390' 
os.environ['https_proxy'] = 'http://10.16.35.10:13390' 

import torch
from torch import nn
import torch.nn.functional as F
from prior_networks import UViT_Clip
from prior_pipe import PriorPipe

def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    from torch.utils.data import DataLoader
    path_data = '../data'
    image_features = torch.load(os.path.join(path_data, 'openclip_emb/emb_imgnet.pt')) # 'emb_imgnet' or 'image_features'
    h_embeddings = image_features['image_features']
    c_embeddings = torch.load(os.path.join(path_data, 'openclip_emb/embedding_concept_imagenet.pt')) # 'embedding_concept_imagenet' or 'embedding_pre_emb_img'
    
    from torch.utils.data import TensorDataset
    dataset = TensorDataset(h_embeddings, c_embeddings)
    # dataset = EmbeddingDatasetVICE(path_data)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=128)
    print(len(dataset))

    diffusion_prior = UViT_Clip(embed_dim=512, num_heads=8, mlp_ratio=4)
    # number of parameters
    sum(p.numel() for p in diffusion_prior.parameters() if p.requires_grad)

    pipe = PriorPipe(diffusion_prior, device=device)

    # load pretrained model
    model_name = 'uvit_vice_pred_imagenet' 
    path = f'./ckpts/{model_name}'
    pipe.diffusion_prior.load_state_dict(torch.load(f'{path}.pt'))
    pipe.ema.load_state_dict(torch.load(f'{path}_ema.pt'))
    pipe.train(dataloader, path=path, num_epochs=32, learning_rate=1e-4, uncondition_rate=0.1) # to 0.086 (uncond/7.83h) / 

if __name__ == '__main__':

    main()