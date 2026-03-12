import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from easydict import EasyDict as edict


class query_similarity_preservation_loss(nn.Module):
    def __init__(self,config):
        super(query_similarity_preservation_loss, self).__init__()
        self.tau = config['deterministic_tau']
    def forward(self, query, query_labels):
        batch_size = query.shape[0]
        device = query.device

        features = F.normalize(query, p=2, dim=1)

        labels_tensor = torch.tensor(np.array(query_labels), device=device)
        pos_mask = (labels_tensor.unsqueeze(1) == labels_tensor.unsqueeze(0)).float()

        

        identity_mask = torch.eye(batch_size, device=device, dtype=torch.bool)
        pos_mask.masked_fill_(identity_mask, 0)


        logits = torch.matmul(features, features.t()) / self.tau

        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()


        neg_mask = ~identity_mask
        exp_logits = torch.exp(logits) * neg_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        num_positives_per_row = pos_mask.sum(1)
        has_positives = num_positives_per_row > 0
        
        if not has_positives.any():
            return torch.tensor(0.0, device=device)


        mean_log_prob_pos = (pos_mask * log_prob).sum(1)[has_positives] / num_positives_per_row[has_positives]


        loss = -mean_log_prob_pos.mean()

        return loss

        
class query_diverse_loss(nn.Module):
    def __init__(self, config):
        torch.nn.Module.__init__(self)
        self.mrg = config['neg_factor'][0]
        self.alpha = config['neg_factor'][1]
        self.lamda = config['neg_factor'][2]
        
    def forward(self, x, label_dict):

        bs = x.shape[0]
        x = F.normalize(x, dim=-1)
        cos = torch.matmul(x, x.t())

        N_one_hot = torch.zeros((bs, bs))
        for i, label in label_dict.items():
            N_one_hot[label[0]:(label[-1]+1), label[0]:(label[-1]+1)] = torch.ones((len(label), len(label)))
        N_one_hot = N_one_hot - torch.eye(bs)
        N_one_hot = N_one_hot.cuda()
    
        neg_exp = torch.exp(self.alpha * (cos + self.mrg))
        
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp))
        focal = torch.where(N_one_hot == 1, cos, torch.zeros_like(cos))
    
        neg_term = (((1 + focal) ** self.lamda) * torch.log(1 + N_sim_sum)).sum(dim=0).sum() / bs
        
        return neg_term



class clip_nce(nn.Module):
    def __init__(self, reduction='mean'):
        super(clip_nce, self).__init__()
        self.reduction = reduction

    def forward(self, labels, label_dict, q2ctx_scores=None):

        if q2ctx_scores.dim()==2:
            query_bsz = q2ctx_scores.shape[0]
            vid_bsz = q2ctx_scores.shape[1]
            diagnoal = torch.arange(query_bsz).to(q2ctx_scores.device)
            t2v_nominator = q2ctx_scores[diagnoal, labels]

            t2v_nominator = torch.logsumexp(t2v_nominator.unsqueeze(1), dim=1)
            t2v_denominator = torch.logsumexp(q2ctx_scores, dim=1)

            v2t_nominator = torch.zeros(vid_bsz).to(q2ctx_scores)
            v2t_denominator = torch.zeros(vid_bsz).to(q2ctx_scores)

            for i, label in label_dict.items():
                v2t_nominator[i] = torch.logsumexp(q2ctx_scores[label, i], dim=0)

                v2t_denominator[i] = torch.logsumexp(q2ctx_scores[:, i], dim=0)
            if self.reduction:
                return torch.mean(t2v_denominator - t2v_nominator) + torch.mean(v2t_denominator - v2t_nominator)

        elif q2ctx_scores.dim()==3:
            loss = 0.
            for k in range(q2ctx_scores.shape[0]):
                q2ctx_scores_i = q2ctx_scores[k]
                query_bsz = q2ctx_scores_i.shape[0]
                vid_bsz = q2ctx_scores_i.shape[1]
                diagnoal = torch.arange(query_bsz).to(q2ctx_scores_i.device)
                t2v_nominator = q2ctx_scores_i[diagnoal, labels]

                t2v_nominator = torch.logsumexp(t2v_nominator.unsqueeze(1), dim=1)
                t2v_denominator = torch.logsumexp(q2ctx_scores_i, dim=1)

                v2t_nominator = torch.zeros(vid_bsz).to(q2ctx_scores_i)
                v2t_denominator = torch.zeros(vid_bsz).to(q2ctx_scores_i)

                for i, label in label_dict.items():
                    v2t_nominator[i] = torch.logsumexp(q2ctx_scores_i[label, i], dim=0)

                    v2t_denominator[i] = torch.logsumexp(q2ctx_scores_i[:, i], dim=0)
                if self.reduction:
                    loss+= torch.mean(t2v_denominator - t2v_nominator) + torch.mean(v2t_denominator - v2t_nominator)
            return loss/q2ctx_scores.shape[0]



class loss(nn.Module):
    def __init__(self, cfg):
        super(loss, self).__init__()
        self.cfg = cfg
        self.clip_nce_criterion = clip_nce(reduction='mean')
        self.video_nce_criterion = clip_nce(reduction='mean')

        self.qdl = query_diverse_loss(cfg)
        self.qsp = query_similarity_preservation_loss(cfg)


    def kld_loss(self, mu, logvar):
        KLD = - 0.5 * torch.sum(1 + logvar - mu.pow(2) -
                                logvar.exp()) / mu.shape[0]
        return KLD
    def forward(self, input_list, batch):
        '''
        param: query_labels: List[int]
        param: clip_scale_scores.shape = [5*bs,bs]
        param: frame_scale_scores.shape = [5*bs,5*bs]
        param: clip_scale_scores_.shape = [5*bs,bs]
        param: frame_scale_scores_.shape = [5*bs,5*bs]
        param: label_dict: Dict[List]
        '''

        query_labels = batch['text_labels']
        
        clip_scale_scores = input_list[0]
        clip_scale_scores_ = input_list[1]
        label_dict = input_list[2]
        frame_scale_scores = input_list[3]
        frame_scale_scores_ = input_list[4]

        video_mu = input_list[9]
        video_logvar= input_list[10]
        all_original_noise= input_list[12]
        all_predicted_noise= input_list[13]



        loss_pvs = self.kld_loss(video_mu,video_logvar)*self.cfg['loss_factor'][4]
        loss_dre = F.mse_loss(all_original_noise,all_predicted_noise)*self.cfg['loss_factor'][5]



        query = input_list[-1]


        clip_nce_loss = self.cfg['loss_factor'][0] * self.clip_nce_criterion(query_labels, label_dict, clip_scale_scores_)
        clip_trip_loss = self.get_clip_triplet_loss(clip_scale_scores, query_labels)

        frame_nce_loss = self.cfg['loss_factor'][1] * self.video_nce_criterion(query_labels, label_dict, frame_scale_scores_)
        frame_trip_loss = self.get_clip_triplet_loss(frame_scale_scores, query_labels)

        loss_sim = clip_nce_loss + clip_trip_loss + frame_nce_loss + frame_trip_loss
        loss_div = self.cfg['loss_factor'][2] * self.qdl(query, label_dict) 
        loss_qsp = self.cfg['loss_factor'][3]*self.qsp(query, query_labels)
        loss_tssl = loss_div+loss_qsp
        loss_total = loss_sim+loss_tssl+loss_pvs+loss_dre

        return loss_total,loss_dre/(self.cfg['loss_factor'][5]+1e-8)


    def get_clip_triplet_loss(self, query_context_scores_in, labels):
        if len(query_context_scores_in.shape) == 2:
            return self.get_single_timestep_clip_triplet_loss(query_context_scores_in,labels)
        elif len(query_context_scores_in.shape)==3:
                loss = 0.
                for k in range(query_context_scores_in.shape[0]):
                    loss += self.get_single_timestep_clip_triplet_loss(query_context_scores_in[k],labels)
                return loss/query_context_scores_in.shape[0]       
        
    def get_single_timestep_clip_triplet_loss(self,query_context_scores_in,labels):
        v2t_scores = query_context_scores_in.t()
        t2v_scores = query_context_scores_in
        labels = np.array(labels)

        # cal_v2t_loss
        v2t_loss = 0
        for i in range(v2t_scores.shape[0]):
            pos_pair_scores = torch.mean(v2t_scores[i][np.where(labels == i)])


            neg_pair_scores, _ = torch.sort(v2t_scores[i][np.where(labels != i)[0]], descending=True)
            if self.cfg['use_hard_negative']:
                sample_neg_pair_scores = neg_pair_scores[0]
            else:
                v2t_sample_max_idx = neg_pair_scores.shape[0]
                sample_neg_pair_scores = neg_pair_scores[
                    torch.randint(0, v2t_sample_max_idx, size=(1,)).to(v2t_scores.device)]

            v2t_loss += (self.cfg['margin'] + sample_neg_pair_scores - pos_pair_scores).clamp(min=0).sum()

        # cal_t2v_loss
        text_indices = torch.arange(t2v_scores.shape[0]).to(t2v_scores.device)
        t2v_pos_scores = t2v_scores[text_indices, labels]
        mask_score = copy.deepcopy(t2v_scores.data)
        mask_score[text_indices, labels] = 999
        _, sorted_scores_indices = torch.sort(mask_score, descending=True, dim=1)
        t2v_sample_max_idx = min(1 + self.cfg['hard_pool_size'],
                                t2v_scores.shape[1]) if self.cfg['use_hard_negative'] else t2v_scores.shape[1]
        sample_indices = sorted_scores_indices[
            text_indices, torch.randint(1, t2v_sample_max_idx, size=(t2v_scores.shape[0],)).to(t2v_scores.device)]

        t2v_neg_scores = t2v_scores[text_indices, sample_indices]

        t2v_loss = (self.cfg['margin'] + t2v_neg_scores - t2v_pos_scores).clamp(min=0)

        return t2v_loss.sum() / len(t2v_scores) + v2t_loss / len(v2t_scores)