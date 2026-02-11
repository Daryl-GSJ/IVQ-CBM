import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from open_clip import create_model_from_pretrained, get_tokenizer, create_model_and_transforms
from open_clip.factory import HF_HUB_PREFIX, _MODEL_CONFIGS
from torchvision import transforms
from .utils import FFN
import json
import pdb

class IVQ_CBM(nn.Module):  
    def __init__(self, concept_list, model_name='biomedclip', config=None):
        super().__init__()
        self.concept_list = concept_list
        self.model_name = model_name
        self.config = config
        self.concept_dim = 34 # change according to the number of attributes (here is 34 for isic2018)
        self.concept_num = 7 # change according to the number of concepts (here is 7 for isic2018)
       
        if self.model_name in ['biomedclip', 'openclip']:
            if self.model_name == 'biomedclip':
                model_name = "biomedclip_local"

                with open ('IVQ-CBM/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/open_clip_config.json') as f:
                    config = json.load(f)
                    model_cfg = config['model_cfg']
                    preprocess_cfg = config['preprocess_cfg']

                if (not model_name.startswith(HF_HUB_PREFIX)
                    and model_name not in _MODEL_CONFIGS
                    and config is not None):
                    _MODEL_CONFIGS[model_name] = model_cfg

                model_path = 'IVQ-CBM/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/open_clip_pytorch_model.bin'
                self.tokenizer = get_tokenizer(model_name)
                self.model, preprocess = create_model_from_pretrained(model_name, pretrained=model_path)
            elif self.model_name == 'openclip':
                self.model, preprocess = create_model_from_pretrained('hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K')
                self.tokenizer = get_tokenizer('hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K')
            
            self.config.preprocess = preprocess
            
            self.model.cuda()
            
            concept_keys = list(concept_list.keys())

            self.concept_token_dict = {}
            self.concept_emb_list = [] 
            self.group_starts = {}
            

            for key in concept_keys:
                if self.config.dataset == 'isic2018':
                    prefix = f"this is a dermoscopic image, the {key} of the lesion is "
                attr_concept_list = concept_list[key]
                prefix_attr_concept_list = [prefix + concept for concept in attr_concept_list]
                tmp_concept_text = self.tokenizer(prefix_attr_concept_list).cuda()
                _, tmp_concept_feats, logit_scale = self.model(None, tmp_concept_text)
                start = len(self.concept_emb_list)
                self.concept_emb_list.append(tmp_concept_feats.detach())
                self.group_starts[key] = (start, tmp_concept_feats.size(0))
                self.concept_token_dict[key] = tmp_concept_feats.detach()
            self.logit_scale = logit_scale.detach()

        self.visual_features = []
        self.hook_list = []
        def hook_fn(module, input, output):
            self.visual_features.append(output) # detach to aboid saving computation graph
        layers = [self.model.visual.trunk.blocks[11]]
        for layer in layers:
            self.hook_list.append(layer.register_forward_hook(hook_fn))
        
        self.visual_tokens = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(7, 768)))

        self.vector_quantizer = VectorQuantizer(
            num_embeddings=7,
            embedding_dim=768,
            commitment_cost=1
        )

        self.cross_attn = nn.MultiheadAttention(embed_dim=768, num_heads=12, batch_first=True) 
        self.vq_aggregator = VectorQuantizeAggregator(num_vectors=7, dim=768)

        self.ffn = FFN(768, 768*4)
        self.norm = nn.LayerNorm(768)
        self.proj = nn.Linear(in_features=768, out_features=512, bias=False)

        self.cls_head = nn.Linear(in_features=self.concept_dim, out_features=self.config.num_class)
        

        for param in self.model.text.parameters():
            param.requires_grad = False
        for param in self.model.visual.trunk.parameters():
            param.requires_grad = True
        
        self.visual_tokens.requires_grad = True
    
    def get_backbone_params(self):
        return self.model.visual.trunk.parameters()
    def get_bridge_params(self):
        param_list = []
        
        param_list.append(self.visual_tokens)
        for param in self.cross_attn.parameters():
            param_list.append(param)
        for param in self.ffn.parameters():
            param_list.append(param)
        for param in self.norm.parameters():
            param_list.append(param)
        for param in self.proj.parameters():
            param_list.append(param)
        for param in self.cls_head.parameters():
            param_list.append(param)
        return param_list

    def compute_effective_rank(self, matrix, threshold=0.99):
        U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
        energy = torch.cumsum(S, dim=0) / torch.sum(S)
        rank = torch.sum(energy <= threshold).item()
        return rank

    def compute_average_rank(self, batch_feat_map, threshold=0.99): 
        ranks = []
        B, N, D = batch_feat_map.shape
        for i in range(B):
            feat = batch_feat_map[i]
            rank = self.compute_effective_rank(feat.detach(), threshold)
            ranks.append(rank)
        
        avg_rank = np.mean(ranks)
        return avg_rank, ranks


    def forward(self, imgs):
        B = imgs.size(0)
        self.visual_features.clear()
        img_feats, _, _ = self.model(imgs, None)
        img_feat_map = self.visual_features[0][:, 1:, :]
        avg_rank, rank_list = self.compute_average_rank(img_feat_map)

        B, _, _ = img_feat_map.shape
        visual_tokens_from_img = self.vq_aggregator(img_feat_map)
        quantized_features, vq_loss, indice = self.vector_quantizer(img_feat_map)
        
        agg_visual_tokens = self.proj(self.norm(self.ffn(visual_tokens_from_img)))
        agg_visual_tokens = F.normalize(agg_visual_tokens, dim=-1)

        image_logits_dict = {}
        idx = 0
        for key in self.concept_token_dict.keys():
            image_logits_dict[key] = (self.logit_scale * agg_visual_tokens[:, idx:idx+1, :] @ self.concept_token_dict[key].repeat(B, 1, 1).permute(0, 2, 1)).squeeze(1)
            idx += 1
        

        image_logits_list = []
        for key in image_logits_dict.keys():
            image_logits_list.append(image_logits_dict[key])
        
        image_logits = torch.cat(image_logits_list, dim=-1)
        cls_logits = self.cls_head(image_logits)

        return cls_logits, image_logits_dict, vq_loss, avg_rank


class VectorQuantizeAggregator(nn.Module):
    def __init__(self, num_vectors: int, dim: int):
        super().__init__()
        self.num_vectors = num_vectors
        self.dim = dim

        self.codebook = nn.Parameter(torch.randn(num_vectors, dim))

    def forward(self, img_feat_map: torch.Tensor) -> torch.Tensor:
        B, N, D = img_feat_map.shape

        img_feat_map_sq = (img_feat_map**2).sum(dim=2, keepdim=True)  # (B, N, 1)
        codebook_sq = (self.codebook**2).sum(dim=1, keepdim=True).t() # (1, K)
        
        dot_product = torch.bmm(img_feat_map, self.codebook.t().unsqueeze(0).expand(B, -1, -1))
        
        distances = img_feat_map_sq + codebook_sq - 2 * dot_product

        soft_assignments = F.softmax(-distances, dim=2)

        aggregated_vectors = torch.bmm(soft_assignments.transpose(1, 2), img_feat_map)

        return aggregated_vectors



class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float):
        super().__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._codebook = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._codebook.weight.data.uniform_(-1.0 / self._num_embeddings, 1.0 / self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs: torch.Tensor):
        b, n, _ = inputs.shape
        
        inputs_flat = inputs.reshape(-1, self._embedding_dim)
        
        distances = (
            torch.sum(inputs_flat**2, dim=1, keepdim=True) 
            + torch.sum(self._codebook.weight**2, dim=1)
            - 2 * torch.matmul(inputs_flat, self._codebook.weight.t())
        )
            
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        
        indices = encoding_indices.view(b, n)
        
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        quantized_flat = torch.matmul(encodings, self._codebook.weight)
        quantized = quantized_flat.view(inputs.shape)
        
        codebook_loss = F.mse_loss(quantized, inputs.detach())
        commitment_loss = F.mse_loss(quantized.detach(), inputs)
        loss = codebook_loss + self._commitment_cost * commitment_loss
        
        quantized = inputs + (quantized - inputs).detach()
        
        return quantized, loss, indices
