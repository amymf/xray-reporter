import torch
import torchxrayvision as xrv
import torch.nn as nn
from transformers import GPT2LMHeadModel

class CheXNetEncoder(torch.nn.Module):
    def __init__(self):
        super(CheXNetEncoder, self).__init__()
        chexnet = xrv.models.DenseNet(weights="densenet121-res224-chex")    
        
        # Freeze encoder
        for param in chexnet.parameters():
            param.requires_grad = False

        self.layers = nn.Sequential(
            chexnet.features,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

    def forward(self, x):
        x = self.layers(x)
        return x # (batch_size, img_embed_dim) which is 1024 for CheXNet


class CheXNetReportDecoder(torch.nn.Module):
    def __init__(self, gpt2_model, img_embed_dim, prefix_len=10):
        super(CheXNetReportDecoder, self).__init__()
        self.prefix_len = prefix_len
        self.gpt2 = gpt2_model
        self.gpt2_embed_dim = self.gpt2.config.n_embd
        self.image_projection = nn.Linear(img_embed_dim, prefix_len*self.gpt2_embed_dim)
    
    def forward(self, input_ids, attn_mask, image_embeds):
        # input_ids: (batch_size, seq_len)
        # attn_mask: (batch_size, seq_len)
        # image_embeds: (batch_size, img_embed_dim)
        batch_size = input_ids.size(0)

        image_embeds = self.image_projection(image_embeds)                      # (batch_size, prefix_len * gpt2_embed_dim)
        image_embeds = image_embeds.view(batch_size, -1, self.gpt2_embed_dim)   # (batch_size, prefix_len, gpt2_embed_dim)
        
        token_embeds = self.gpt2.transformer.wte(input_ids)                     # (batch_size, seq_len, gpt2_embed_dim)

        input = torch.cat([image_embeds, token_embeds], dim=1)                  # (batch_size, prefix_len + seq_len, gpt2_embed_dim)
        
        # update attn mask 
        prefix_mask = torch.ones(batch_size, self.prefix_len, dtype=torch.long).to(attn_mask.device)
        attn_mask = torch.cat([prefix_mask, attn_mask], dim=1)                 # (batch_size, prefix_len + seq_len)

        outputs = self.gpt2(inputs_embeds=input, attention_mask=attn_mask)
        logits = outputs.logits[:, self.prefix_len:, :]                        # (batch_size, seq_len, vocab_size)
        return logits


class CheXNetReportModel(torch.nn.Module):
    def __init__(self, gpt2_model):
        super(CheXNetReportModel, self).__init__()
        self.encoder = CheXNetEncoder()
        self.decoder = CheXNetReportDecoder(gpt2_model=gpt2_model, img_embed_dim=1024, prefix_len=10)

    def forward(self, images, input_ids, attn_mask):
        batch_size, N, _, _, _ = images.shape
        image_embeds = self.encoder(images)                             # (batch_size * N, img_embed_dim)
        # avg embedding across N images
        image_embeds = image_embeds.view(batch_size, N, -1)             # (batch_size, N, img_embed_dim)
        image_embeds = image_embeds.mean(dim=1)                         # (batch_size, img_embed_dim)
        logits = self.decoder(input_ids, attn_mask, image_embeds)       # (batch_size, seq_len, vocab_size)
        return logits


print(CheXNetReportModel(GPT2LMHeadModel.from_pretrained("gpt2")))