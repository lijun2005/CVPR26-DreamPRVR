# coding=utf-8
import numpy as np
import torch
import torch.nn as nn
import math
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
def l2_normalize(tensor, axis=-1):
    
    """L2-normalize columns of tensor"""
    return torch.nn.functional.normalize(tensor, p=2, dim=axis)

def positional_encoding_1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                          -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe


class GPO(nn.Module):
    def __init__(self, d_pe, d_hidden):
        super(GPO, self).__init__()
        self.d_pe = d_pe
        self.d_hidden = d_hidden

        self.pe_database = {}
        self.gru = nn.GRU(self.d_pe, d_hidden, 1, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(self.d_hidden, 1, bias=False)

    def compute_pool_weights(self, lengths, features):
        max_len = int(lengths.max())
        pe_max_len = self.get_pe(max_len)
        pes = pe_max_len.unsqueeze(0).repeat(lengths.size(0), 1, 1).to(lengths.device)
        mask = torch.arange(max_len).expand(lengths.size(0), max_len).to(lengths.device)
        mask = (mask < lengths.long().unsqueeze(1)).unsqueeze(-1)
        pes = pes.masked_fill(mask == 0, 0)

        self.gru.flatten_parameters()
        packed = pack_padded_sequence(pes, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.gru(packed)
        padded = pad_packed_sequence(out, batch_first=True)
        out_emb, out_len = padded
        out_emb = (out_emb[:, :, :out_emb.size(2) // 2] + out_emb[:, :, out_emb.size(2) // 2:]) / 2
        scores = self.linear(out_emb)
        scores[torch.where(mask == 0)] = -10000

        weights = torch.softmax(scores / 0.1, 1)
        return weights, mask

    def forward(self, features, lengths):
        """
        :param features: features with shape B x K x D
        :param lengths: B x 1, specify the length of each data sample.
        :return: pooled feature with shape B x D
        """
        pool_weights, mask = self.compute_pool_weights(lengths, features)

        features = features[:, :int(lengths.max()), :]
        sorted_features = features.masked_fill(mask == 0, -10000)
        sorted_features = sorted_features.sort(dim=1, descending=True)[0]
        sorted_features = sorted_features.masked_fill(mask == 0, 0)

        pooled_features = (sorted_features * pool_weights).sum(1)
        return pooled_features

    def get_pe(self, length):
        """

        :param length: the length of the sequence
        :return: the positional encoding of the given length
        """
        length = int(length)
        if length in self.pe_database:
            return self.pe_database[length]
        else:
            pe = positional_encoding_1d(self.d_pe, length)
            self.pe_database[length] = pe
            return pe

class MultiHeadSelfAttention(nn.Module):

    """ Self-attention module """

    def __init__(self, n_head, d_in, d_hidden):

        super(MultiHeadSelfAttention, self).__init__()

        self.n_head = n_head
        self.w_1 = nn.Linear(d_in, d_hidden, bias=False)
        self.w_2 = nn.Linear(d_hidden, n_head, bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.w_1.weight)
        nn.init.xavier_uniform_(self.w_2.weight)

    def forward(self, x, mask=None):

        # This expects input x to be of size (b x seqlen x d_feat)

        attn = self.w_2(self.tanh(self.w_1(x)))
        if mask is not None:
            mask = mask.repeat(self.n_head, 1, 1).permute(1, 2, 0)
            attn.masked_fill_(mask, -np.inf)
        attn = self.softmax(attn)

        output = torch.bmm(attn.transpose(1, 2), x)
        if output.shape[1] == 1:
            output = output.squeeze(1)
        return output, attn


class VideoMean(nn.Module):
    def __init__(self, n_embeds, d_in, d_out, d_h):
        super(VideoMean, self).__init__()

        # 1, embed_dim, embed_dim, embed_dim // 2

        self.num_embeds = n_embeds
        self.attention = MultiHeadSelfAttention(n_embeds, d_in, d_h)
        self.fc = nn.Linear(d_in, d_out)
        self.sigmoid = nn.Sigmoid()
        self.layer_norm = nn.LayerNorm(d_out)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, out, x, pad_mask=None):

        residual, _ = self.attention(x, pad_mask)
        residual = self.sigmoid(self.fc(residual))
        if self.num_embeds > 1:
            out = out.unsqueeze(1).repeat(1, self.num_embeds, 1)
        out = self.layer_norm(out + residual)

        return out


class VideoVar(nn.Module):
    def __init__(self, d_in, d_out, d_h):
        super().__init__()

        self.attention = MultiHeadSelfAttention(1, d_in, d_h)

        self.fc = nn.Linear(d_in, d_out)
        self.init_weights()

        self.fc2 = nn.Linear(d_in, d_out)
        self.embed_dim = d_in

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, out, x, pad_mask=None):
        residual, _ = self.attention(x, pad_mask)

        fc_out = self.fc2(out)
        out = self.fc(residual) + fc_out

        return out


class VideoLatentVariableSampler(nn.Module):
    def __init__(self,hidden_size):
        super().__init__()
        self.gpo = GPO(32,32)
        self.pie_net_video = VideoMean(1, hidden_size, hidden_size, hidden_size // 2)
        self.uncertain_net_video = VideoVar(hidden_size, hidden_size, hidden_size // 2)
        self._init_gpo_weights()
    def _init_gpo_weights(self):
        # Initialize GPO components properly
        # GRU initialization (PyTorch default is usually fine, but ensure it's set properly)
        for name, param in self.gpo.gru.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.zeros_(param.data)
        
        # GPO linear layer initialization
        nn.init.xavier_uniform_(self.gpo.linear.weight)
        # Note: GPO linear layer has no bias (bias=False)
    def forward(self,x):
        batch_size, seq_len, _ = x.shape
        lengths = torch.full((batch_size,), seq_len, dtype=torch.long, device=x.device) 
        x_pooled= self.gpo(x, lengths)  # [B, D] 
        mean_out = l2_normalize(self.pie_net_video(x_pooled,x))
        var_out = self.uncertain_net_video(x_pooled,x)
        return mean_out,var_out
    
    def reparameterize(self, mu, logvar,num_samples):
        eps = torch.randn(mu.size(0), num_samples, mu.size(1), dtype=mu.dtype, device=mu.device)

        samples = eps.mul(torch.exp(0.5*logvar.unsqueeze(1))).add_(
        mu.unsqueeze(1))
        return samples

    def kld_loss(self, mu, logvar):
        KLD = - 0.5 * torch.sum(1 + logvar - mu.pow(2) -
                                logvar.exp()) / mu.shape[0]
        return KLD

class TextSampler(nn.Module):

    def __init__(self, feature_size, alpha_scale=1, beta_scale=1,perturbation_scale=0.1):
        super().__init__()
        self.instance_norm = nn.InstanceNorm1d(feature_size)
        self.alpha_scale = alpha_scale
        self.beta_scale = beta_scale
        self.perturbation_scale = perturbation_scale


    
    def forward(self, features, num_samples=1):
        if num_samples <= 1:
            return self._original_forward(features).unsqueeze(1)

        batch_size, feature_dim = features.shape

        std, mean = torch.std_mean(features, dim=1) 


        perturb_std = std * self.perturbation_scale

        normal_alpha = torch.distributions.Normal(loc=1, scale=perturb_std)
        normal_beta = torch.distributions.Normal(loc=mean, scale=perturb_std)

        sample_shape = torch.Size([num_samples, feature_dim]) 
        

        alpha = normal_alpha.sample(sample_shape)
        beta = normal_beta.sample(sample_shape)

        alpha = alpha.permute(2, 0, 1) # (N, D, B) -> (B, N, D)
        beta = beta.permute(2, 0, 1)   # (N, D, B) -> (B, N, D)

        alpha = self.alpha_scale * alpha
        beta = self.beta_scale * beta

        expanded_features = features.unsqueeze(1).expand(-1, num_samples, -1)
        flat_features = expanded_features.reshape(-1, feature_dim)
        normed_flat_features = self.instance_norm(flat_features)
        normed_features = normed_flat_features.view(batch_size, num_samples, feature_dim)
        x = alpha * normed_features + beta

        return x
