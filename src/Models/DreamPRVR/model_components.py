import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def onehot(indexes, N=None):
    """
    Creates a one-representation of indexes with N possible entries
    if N is not specified, it will suit the maximum index appearing.
    indexes is a long-tensor of indexes
    """
    if N is None:
        N = indexes.max() + 1
    sz = list(indexes.size())
    output = indexes.new().long().resize_(*sz, N).zero_()
    output.scatter_(-1, indexes.unsqueeze(-1), 1)
    return output


class TrainablePositionalEncoding(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""
    def __init__(self, max_position_embeddings, hidden_size, dropout=0.1):
        super(TrainablePositionalEncoding, self).__init__()
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_feat):
        bsz, seq_length = input_feat.shape[:2]
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_feat.device)
        position_ids = position_ids.unsqueeze(0).repeat(bsz, 1)  # (N, L)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = self.LayerNorm(input_feat + position_embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def add_position_emb(self, input_feat):
        bsz, seq_length = input_feat.shape[:2]
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_feat.device)
        position_ids = position_ids.unsqueeze(0).repeat(bsz, 1)  # (N, L)
        position_embeddings = self.position_embeddings(position_ids)
        return input_feat + position_embeddings


class LinearLayer(nn.Module):
    """linear layer configurable with layer normalization, dropout, ReLU."""
    def __init__(self, in_hsz, out_hsz, layer_norm=True, dropout=0.1, relu=True):
        super(LinearLayer, self).__init__()
        self.relu = relu
        self.layer_norm = layer_norm
        if layer_norm:
            self.LayerNorm = nn.LayerNorm(in_hsz)
        layers = [nn.Dropout(dropout), nn.Linear(in_hsz, out_hsz)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """(N, L, D)"""
        if self.layer_norm:
            x = self.LayerNorm(x)
        x = self.net(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x  # (N, L, D)


class HLFormerBlock(nn.Module):
    def __init__(self, config):
        super(HLFormerBlock, self).__init__()

        self.num_block = config.attention_num
        self.e_attns = nn.ModuleList()
        self.e_attns.append(EuclideanAttentionBlock(config))
        for i in range(-2,self.num_block-3):
            wid = 2 ** i 
            self.e_attns.append(EuclideanAttentionBlock(config, wid=wid))


        self.ca = CrossAttention(config)
        self.layer1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layer2 = nn.Linear(config.hidden_size, config.frame_len)

        self.sft_factor = config.sft_factor

    def forward(self, input_tensor, attention_mask=None, weight_token=None):

        outputs = []
        for i in range(len(self.e_attns)):
            o = self.e_attns[i](input_tensor, attention_mask).unsqueeze(-1)
            outputs.append(o) 
        oo = torch.cat(outputs, dim=-1)


        if weight_token is None:
            mean_oo = torch.mean(oo, dim=-1)#.squeeze())
            weight_token =torch.mean(mean_oo,dim=1,keepdim=True)

        else:
            weight_token = weight_token.to(oo.device).type_as(oo).repeat(oo.shape[0], 1, 1)
        weight = []
        for i in range(oo.shape[-1]):
            temp_token = self.ca(weight_token, oo[..., i], attention_mask)
            weight.append(temp_token)

        weight = torch.cat(weight, dim=1)    
        weight = self.layer1(weight)
        weight = self.dropout(F.relu(weight))
        weight = self.layer2(weight)
        
        weight = F.softmax(weight.permute(0, 2, 1) / self.sft_factor, dim=-1)
        out = torch.sum(oo * weight.unsqueeze(2).repeat(1, 1, oo.shape[2], 1), dim=-1)

        return out


class FeedForward(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.layer1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.layer2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.layer1(x)
        x = self.dropout(F.relu(x))
        x = self.layer2(x)
        return x



class EuclideanAttentionBlock(nn.Module):
    def __init__(self, config, wid=None):
        super(EuclideanAttentionBlock, self).__init__()
        self.self = EuclideanGaussianAttention(config, wid=wid)
        self.output = FeedForward(config.hidden_size, int(4*config.hidden_size), config.hidden_dropout_prob)

        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)
        self.dropout1 = nn.Dropout(config.hidden_dropout_prob)
        self.dropout2 = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_tensor, attention_mask=None,register_token=None):
        """
        Args:
            input_tensor: (N, L, D)
            attention_mask: (N, L)
        """
        self_output = self.self(input_tensor, input_tensor, input_tensor, attention_mask)
        self_output = self.dropout1(self_output)
        input_tensor = self.norm1(input_tensor + self_output)
        tmp = self.output(input_tensor)
        tmp = self.dropout2(tmp)
        input_tensor = self.norm2(input_tensor + tmp)
        return input_tensor


class CrossAttention(nn.Module):
    def __init__(self, config):
        super(CrossAttention, self).__init__()
        self.self = EuclideanGaussianAttention(config)
        self.output = FeedForward(config.hidden_size, int(1*config.hidden_size), config.hidden_dropout_prob)

        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)
        self.dropout1 = nn.Dropout(config.hidden_dropout_prob)
        self.dropout2 = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, query, input_tensor, attention_mask=None):
        """
        Args:
            input_tensor: (N, L, D)
            attention_mask: (N, L)
        """
        self_output = self.self(query, input_tensor, input_tensor, attention_mask)

        self_output = self.dropout1(self_output)
        query = self.norm1(query + self_output)
        tmp = self.output(query)
        tmp = self.dropout2(tmp)
        query = self.norm2(query + tmp)
        return query


class EuclideanGaussianAttention(nn.Module):
    def __init__(self, config, wid=None):
        super(EuclideanGaussianAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError("The hidden size (%d) is not a multiple of the number of attention heads (%d)" % (
                config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.wid = wid
        self.frame_len =  config.frame_len

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)  # (N, L, nh, dh)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # (N, nh, L, dh)

    def generate_gauss_weight(self, props_len, width):

        center = torch.arange(props_len).cuda() / props_len
        width = width*torch.ones(props_len).cuda()
        weight = torch.linspace(0, 1, props_len)
        weight = weight.view(1, -1).expand(center.size(0), -1).to(center.device)
        center = center.unsqueeze(-1)
        width = width.unsqueeze(-1).clamp(1e-2) / 9

        w = 0.3989422804014327

        weight = w/width*torch.exp(-(weight-center)**2/(2*width**2))

        return weight/weight.max(dim=-1, keepdim=True)[0]

    def forward(self, query_states, key_states, value_states, attention_mask=None):
        """
        Args:
            query_states: (N, Lq, D)
            key_states: (N, L, D)
            value_states: (N, L, D)
            attention_mask: (N, Lq, L)
        """

        mixed_query_layer = self.query(query_states)
        mixed_key_layer = self.key(key_states)
        mixed_value_layer = self.value(value_states)
        # transpose
        query_layer = self.transpose_for_scores(mixed_query_layer)  # (N, nh, Lq, dh)
        key_layer = self.transpose_for_scores(mixed_key_layer)  # (N, nh, L, dh)
        value_layer = self.transpose_for_scores(mixed_value_layer)  # (N, nh, L, dh)
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores_ori = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # (N, nh, Lq, L)

        attention_scores_ori = attention_scores_ori / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores_ori
        if self.wid is not None:
            gmm_mask = torch.ones(attention_scores.shape[-1], attention_scores.shape[-1], device=attention_scores.device)
            gauss_weight_part = self.generate_gauss_weight(self.frame_len, self.wid)
            gmm_mask[:self.frame_len, :self.frame_len] = gauss_weight_part

            gmm_mask = gmm_mask.unsqueeze(0).unsqueeze(0)
            attention_scores = attention_scores_ori * gmm_mask


        if attention_mask is not None:
            attention_mask = (1 - attention_mask.unsqueeze(1)) * -10000.  # (N, 1, Lq, L)
            attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)


        attention_probs = self.dropout(attention_probs)


        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer
    


class QueryFormer(nn.Module):
    def __init__(self, config):
        super(QueryFormer, self).__init__()
        self.num_queries = config.num_registers
        self.hidden_size = config.hidden_size


        self.query_tokens = nn.Parameter(torch.randn(1, self.num_queries, self.hidden_size))
        self.cross_attention = CrossAttention(config)

    def forward(self, video_embedding, attention_mask=None):
        batch_size = video_embedding.shape[0]

        queries = self.query_tokens.expand(batch_size, -1, -1)

        updated_queries = self.cross_attention(queries, video_embedding, attention_mask)

        return updated_queries




class DreamPRVRBlock(nn.Module):
    def __init__(self, config):
        super(DreamPRVRBlock, self).__init__()

        self.num_block = config.attention_num
        self.e_attns = nn.ModuleList()
        self.e_attns.append(EuclideanAttentionBlock(config))
        for i in range(-2,self.num_block-3):
            wid = 2 ** i 
            self.e_attns.append(EuclideanAttentionBlock(config, wid=wid))




        self.ca = CrossAttention(config)
        self.layer1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layer2 = nn.Linear(config.hidden_size, config.frame_len)

        self.sft_factor = config.sft_factor
        self.clip_num = config.frame_len
        self.register_num = config.num_registers
        self.noise_input_layer = LinearLayer(config.hidden_size, config.hidden_size, layer_norm=True,
                                            dropout=config.hidden_dropout_prob, relu=True)
        
    def forward(self, input_tensor, attention_mask=None, weight_token=None,register_token=None):

        Lk = self.clip_num+self.register_num
        mask_clip = torch.ones(self.clip_num, Lk, dtype=torch.bool)
        
        mask_register = torch.zeros(self.register_num, Lk, dtype=torch.bool)
        mask_register[:, :self.clip_num] = True

        attention_mask_custom = torch.cat([mask_clip, mask_register], dim=0).to(input_tensor.device)
        attention_mask_custom = attention_mask_custom.unsqueeze(0) # (1, Lq, Lk)




        if attention_mask is not None:
            batch_size = attention_mask.shape[0]
            
            register_mask = torch.ones(batch_size, 1, self.register_num, dtype=torch.bool, device=input_tensor.device)
            
            valid_keys_mask = torch.cat([attention_mask.bool(), register_mask], dim=2) # (B, 1, Lk)
            
            final_attention_mask = attention_mask_custom & valid_keys_mask
        else:
            final_attention_mask = attention_mask_custom

        register_token = self.noise_input_layer(register_token)
        input_tensor_all = torch.cat([input_tensor,register_token],dim=1)
        outputs = []
        for i in range(len(self.e_attns)):
            o = self.e_attns[i](input_tensor_all, final_attention_mask.float())[:,:self.clip_num,:].unsqueeze(-1)
            outputs.append(o) 

        oo = torch.cat(outputs, dim=-1)


        if weight_token is None:
            mean_oo = torch.mean(oo, dim=-1)#.squeeze())
            weight_token =torch.mean(mean_oo,dim=1,keepdim=True)

        else:
            weight_token = weight_token.to(oo.device).type_as(oo).repeat(oo.shape[0], 1, 1)
        weight = []
        for i in range(oo.shape[-1]):
            temp_token = self.ca(weight_token, oo[..., i], attention_mask)
            weight.append(temp_token)

        weight = torch.cat(weight, dim=1)    
        weight = self.layer1(weight)
        weight = self.dropout(F.relu(weight))
        weight = self.layer2(weight)
        
        weight = F.softmax(weight.permute(0, 2, 1) / self.sft_factor, dim=-1)
        out = torch.sum(oo * weight.unsqueeze(2).repeat(1, 1, oo.shape[2], 1), dim=-1)

        return out
