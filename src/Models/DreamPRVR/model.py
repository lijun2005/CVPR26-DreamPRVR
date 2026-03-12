import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from easydict import EasyDict as edict
from Models.DreamPRVR.model_components import EuclideanAttentionBlock, LinearLayer, \
                                            TrainablePositionalEncoding,QueryFormer,DreamPRVRBlock
from Models.DreamPRVR.latent_sampler import VideoLatentVariableSampler,TextSampler
from Models.DreamPRVR.diffusion_model import DiffusionMLP,GaussianDiffusion

class DreamPRVR_Net(nn.Module):
    def __init__(self, config):
        super(DreamPRVR_Net, self).__init__()
        self.config = config
        self.query_pos_embed = TrainablePositionalEncoding(max_position_embeddings=config.max_desc_l,
                                                           hidden_size=config.hidden_size, dropout=config.input_drop)
        self.clip_pos_embed = TrainablePositionalEncoding(max_position_embeddings=config.max_ctx_l,
                                                         hidden_size=config.hidden_size, dropout=config.input_drop)
        self.frame_pos_embed = TrainablePositionalEncoding(max_position_embeddings=config.max_ctx_l,
                                                          hidden_size=config.hidden_size, dropout=config.input_drop)
        #
        self.query_input_proj = LinearLayer(config.query_input_size, config.hidden_size, layer_norm=True,
                                            dropout=config.input_drop, relu=True)
        self.query_encoder = EuclideanAttentionBlock(edict(hidden_size=config.hidden_size, intermediate_size=config.hidden_size,
                                                 hidden_dropout_prob=config.drop, num_attention_heads=config.n_heads,frame_len=None,
                                                 attention_probs_dropout_prob=config.drop))
        self.clip_input_proj = LinearLayer(config.visual_input_size, config.hidden_size, layer_norm=True,
                                            dropout=config.input_drop, relu=True)
        self.clip_encoder = DreamPRVRBlock(edict(hidden_size=config.hidden_size, intermediate_size=config.hidden_size,
                                                 hidden_dropout_prob=config.drop, num_attention_heads=config.n_heads,
                                                 attention_probs_dropout_prob=config.drop, frame_len=32, sft_factor=config.sft_factor,drop=config.drop,attention_num=config.attention_num,
                                                 map_size = config.map_size,num_registers = config.num_registers))

        self.frame_input_proj = LinearLayer(config.visual_input_size, config.hidden_size, layer_norm=True,
                                             dropout=config.input_drop, relu=True)
        self.frame_encoder = DreamPRVRBlock(edict(hidden_size=config.hidden_size, intermediate_size=config.hidden_size,
                                                 hidden_dropout_prob=config.drop, num_attention_heads=config.n_heads,
                                                 attention_probs_dropout_prob=config.drop, frame_len=128, sft_factor=config.sft_factor,drop=config.drop,attention_num=config.attention_num,
                                                 num_registers = config.num_registers))
                    
        self.modular_vector_mapping = nn.Linear(config.hidden_size, out_features=1, bias=False)

        self.weight_token = None




        self.num_registers = config.num_registers

        self.video_probabilistic_variational_sampler = VideoLatentVariableSampler(hidden_size=config.hidden_size)
        self.textual_perturbation_sampler = TextSampler(feature_size = config.hidden_size)

        ##diffusion register estimator
        self.diffusion_mlp = DiffusionMLP(
            input_dim=config.hidden_size,
            hidden_dim=config.hidden_size,
            output_dim= config.hidden_size,
            T = config.timesteps,
            num_layers = config.num_diffusion_layers)
        self.guassian_diffusion = GaussianDiffusion(timesteps=config.timesteps)


        ## produce the  condition for register generation
        self.mean_clip_encoder = EuclideanAttentionBlock(edict(hidden_size=config.hidden_size, intermediate_size=config.hidden_size,
                                                 hidden_dropout_prob=config.drop, num_attention_heads=config.n_heads,frame_len=None,
                                                 attention_probs_dropout_prob=config.drop))
        self.video_qformer = QueryFormer(edict(hidden_size=config.hidden_size, intermediate_size=config.hidden_size,
                                                 hidden_dropout_prob=config.drop, num_attention_heads=config.n_heads,
                                                 attention_probs_dropout_prob=config.drop, drop=config.drop,
                                                 num_registers = config.num_registers,frame_len=None
                                                 ))  

        self.reset_parameters()

    def reset_parameters(self):
        """ Initialize the weights."""

        def re_init(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            elif isinstance(module, nn.Conv1d):
                module.reset_parameters()
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

        self.apply(re_init)

    def set_hard_negative(self, use_hard_negative, hard_pool_size):
        """use_hard_negative: bool; hard_pool_size: int, """
        self.config.use_hard_negative = use_hard_negative
        self.config.hard_pool_size = hard_pool_size

    def reparameterize(self, mu, logvar,num_samples,deterministic=False):
        eps = torch.randn(mu.size(0), num_samples, mu.size(1), dtype=mu.dtype, device=mu.device)

        samples = eps.mul(torch.exp(0.5*logvar.unsqueeze(1))).add_(
        mu.unsqueeze(1)) if not deterministic else mu.unsqueeze(1).expand(-1,num_samples,-1)
        return samples

    def kld_loss(self, mu, logvar):
        KLD = - 0.5 * torch.sum(1 + logvar - mu.pow(2) -
                                logvar.exp()) / mu.shape[0]
        return KLD

    def forward(self, batch):

        clip_video_feat = batch['clip_video_features']
        query_feat = batch['text_feat']
        query_mask = batch['text_mask']
        query_labels = batch['text_labels']

        frame_video_feat = batch['frame_video_features']
        frame_video_mask = batch['videos_mask']


        label_dict = {}
        for index, label in enumerate(query_labels):
            if label in label_dict:
                label_dict[label].append(index)
            else:
                label_dict[label] = []
                label_dict[label].append(index)

        video_query = self.encode_query(query_feat, query_mask)
        unique_labels = sorted(label_dict.keys())
        mean_queries_list = []
        for label in unique_labels:
            indices = label_dict[label]
            queries_for_label = video_query[indices]
            mean_query = torch.mean(queries_for_label, dim=0)
            mean_queries_list.append(mean_query)
        mean_video_queries = torch.stack(mean_queries_list, dim=0)

        encoded_frame_feat,encoded_clip_feat,video_mu,video_logvar,sample_init_video,all_original_noise,all_predicted_noise,all_x_recon_list = self.encode_context(
            clip_video_feat, frame_video_feat, frame_video_mask,train=True,mean_video_queries=mean_video_queries)

        clip_scale_scores, clip_scale_scores_, frame_scale_scores, frame_scale_scores_ \
            = self.get_pred_from_raw_query(
            query_feat, query_mask, query_labels, encoded_clip_feat, encoded_frame_feat, return_query_feats=True)


        return [clip_scale_scores, clip_scale_scores_, label_dict, frame_scale_scores, frame_scale_scores_, 
                None,None,None,None,video_mu,video_logvar,sample_init_video,all_original_noise,all_predicted_noise,all_x_recon_list,mean_video_queries,video_query]




    def encode_query(self, query_feat, query_mask):
        encoded_query = self.encode_input(query_feat, query_mask, self.query_input_proj, self.query_encoder,
                                          self.query_pos_embed)  # (N, Lq, D)
        if query_mask is not None:
            mask = query_mask.unsqueeze(1)
        
        video_query = self.get_modularized_queries(encoded_query, query_mask)  # (N, D) * 1

        return video_query


    def encode_context(self, clip_video_feat, frame_video_feat, video_mask=None,train=False,mean_video_queries=None):


        feat_clip = self.clip_input_proj(clip_video_feat)
        feat_clip = self.clip_pos_embed(feat_clip)

        mean_encoded_clip_feat = self.mean_clip_encoder(feat_clip)
        video_global_register = self.video_qformer(mean_encoded_clip_feat)
        video_mu,video_logvar = self.video_probabilistic_variational_sampler(mean_encoded_clip_feat)
        if frame_video_feat.shape[1] != 128:
            fix = 128 - frame_video_feat.shape[1]
            temp_feat = 0.0 * frame_video_feat.mean(dim=1, keepdim=True).repeat(1, fix, 1)
            frame_video_feat = torch.cat([frame_video_feat, temp_feat], dim=1)

            temp_mask = 0.0 * video_mask.mean(dim=1, keepdim=True).repeat(1, fix).type_as(video_mask)
            video_mask = torch.cat([video_mask, temp_mask], dim=1)
        if train:
            noise = self.reparameterize(video_mu,video_logvar,num_samples = self.num_registers,deterministic=True)
            sample_queries = self.textual_perturbation_sampler(mean_video_queries,num_samples = self.num_registers)

            all_original_noise,all_predicted_noise,all_x_recon_list,_ = self.guassian_diffusion.ddpm_forward_oversample(self.diffusion_mlp,sample_queries,video_global_register,noise = noise)
            
            noise = self.reparameterize(video_mu,video_logvar,num_samples = self.num_registers,deterministic=True)
            all_x_noisy_list = self.guassian_diffusion.sample(self.diffusion_mlp,video_global_register,noise,middle_noise=noise).unsqueeze(0)


            clip_output_list = []
            for t in range(len(all_x_noisy_list)):
                noisy_clip = all_x_noisy_list[t]
                clip_output_list.append(self.clip_encoder(input_tensor = feat_clip,register_token = noisy_clip))
            encoded_clip_feat = torch.stack(clip_output_list,dim=0)

            encoded_clip_feat = encoded_clip_feat.squeeze(0)#[T,B,L,D]
            encoded_frame_feat = self.encode_input(frame_video_feat, video_mask, self.frame_input_proj,
                                                    self.frame_encoder,
                                                    self.frame_pos_embed, self.weight_token,register_token=all_x_noisy_list[0])
        else:
            noise = self.reparameterize(video_mu,video_logvar,num_samples = self.num_registers,deterministic=True)
            clean_register = self.guassian_diffusion.sample(self.diffusion_mlp,video_global_register,noise,middle_noise=noise)
            encoded_clip_feat = self.clip_encoder(input_tensor = feat_clip,register_token = clean_register)#[B,L,D]
            encoded_frame_feat = self.encode_input(frame_video_feat, video_mask, self.frame_input_proj,
                                                    self.frame_encoder,
                                                    self.frame_pos_embed, self.weight_token,register_token=clean_register)
        encoded_frame_feat = torch.where(video_mask.unsqueeze(-1).repeat(1, 1, encoded_frame_feat.shape[-1]) == 1.0, \
                                                                        encoded_frame_feat, 0. * encoded_frame_feat)

        if train:
            return encoded_frame_feat,encoded_clip_feat,video_mu,video_logvar,None,all_original_noise,all_predicted_noise,all_x_recon_list
        else:
            return encoded_frame_feat,encoded_clip_feat

    @staticmethod
    def encode_input(feat, mask, input_proj_layer, encoder_layer, pos_embed_layer, weight_token=None,register_token=None):
        """
        Args:
            feat: (N, L, D_input), torch.float32
            mask: (N, L), torch.float32, with 1 indicates valid query, 0 indicates mask
            input_proj_layer: down project input
            encoder_layer: encoder layer
            pos_embed_layer: positional embedding layer
        """

        feat = input_proj_layer(feat)
        feat = pos_embed_layer(feat)
        if mask is not None:
            mask = mask.unsqueeze(1)  # (N, 1, L), torch.FloatTensor
        if weight_token is not None:
            return encoder_layer(feat, mask, weight_token,register_token=register_token)  # (N, L, D_hidden)
        else:
            return encoder_layer(feat, mask,register_token=register_token)  # (N, L, D_hidden)

    def get_modularized_queries(self, encoded_query, query_mask):
        """
        Args:
            encoded_query: (N, L, D)
            query_mask: (N, L)
            return_modular_att: bool
        """
        modular_attention_scores = self.modular_vector_mapping(encoded_query)  # (N, L, 2 or 1)
        modular_attention_scores = F.softmax(mask_logits(modular_attention_scores, query_mask.unsqueeze(2)), dim=1)
        modular_queries = torch.einsum("blm,bld->bmd", modular_attention_scores, encoded_query)  # (N, 2 or 1, D)
        return modular_queries.squeeze()

    @staticmethod
    def get_clip_scale_scores(modularied_query, context_feat):

        modularied_query = F.normalize(modularied_query, dim=-1)
        context_feat = F.normalize(context_feat, dim=-1)

        if context_feat.dim()==3:
            clip_level_query_context_scores = torch.matmul(context_feat, modularied_query.t()).permute(2, 1, 0)
            query_context_scores, indices = torch.max(clip_level_query_context_scores,
                                                    dim=1)  # (N, N) diagonal positions are positive pairs
        elif context_feat.dim()==4:
            clip_level_query_context_scores = torch.matmul(context_feat, modularied_query.t()).permute(0,3, 2, 1)
            query_context_scores, indices = torch.max(clip_level_query_context_scores,
                                                    dim=2)  # (N, N) diagonal positions are positive pairs
        return query_context_scores


    @staticmethod
    def get_unnormalized_clip_scale_scores(modularied_query, context_feat):

        if context_feat.dim()==3:

            clip_level_query_context_scores = torch.matmul(context_feat, modularied_query.t()).permute(2, 1, 0)
            query_context_scores, indices = torch.max(clip_level_query_context_scores,
                                                    dim=1)  # (N, N) diagonal positions are positive pairs

        elif context_feat.dim()==4:

            clip_level_query_context_scores = torch.matmul(context_feat, modularied_query.t()).permute(0,3, 2, 1)
            query_context_scores, indices = torch.max(clip_level_query_context_scores,
                                                    dim=2)  # (N, N) diagonal positions are positive pairs

        return query_context_scores


    def get_pred_from_raw_query(self, query_feat, query_mask, query_labels=None,
                                video_proposal_feat=None, encoded_frame_feat=None,
                                return_query_feats=False):

        video_query = self.encode_query(query_feat, query_mask)


        clip_scale_scores = self.get_clip_scale_scores(
            video_query, video_proposal_feat)

        frame_scale_scores = self.get_clip_scale_scores(
            video_query, encoded_frame_feat)

        if return_query_feats:
            clip_scale_scores_ = self.get_unnormalized_clip_scale_scores(video_query, video_proposal_feat)
            frame_scale_scores_ = self.get_unnormalized_clip_scale_scores(video_query, encoded_frame_feat)

            return clip_scale_scores, clip_scale_scores_, frame_scale_scores, frame_scale_scores_
        else:

            return clip_scale_scores, frame_scale_scores


def mask_logits(target, mask):
    return target * mask + (1 - mask) * (-1e10)
