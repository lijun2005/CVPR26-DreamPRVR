# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import defaultdict

import ipdb

import numpy as np
import logging
import torch.backends.cudnn as cudnn
import os
import pickle

from tqdm import tqdm
import torch

from Utils.utils import gpu
import json


def get_text_gt(query_metas):
  
    vid_to_query_indices = defaultdict(list)
    for i, query_id in enumerate(query_metas):
        vid_id = query_id.split('#', 1)[0]
        vid_to_query_indices[vid_id].append(i)


    t2t_gt = {}
    for i, query_id in enumerate(query_metas):
        vid_id = query_id.split('#', 1)[0]
        all_indices_for_vid = vid_to_query_indices[vid_id]
        
        t2t_gt[i] = [idx for idx in all_indices_for_vid if idx != i]
        
    return t2t_gt

def get_gt(video_metas, query_metas):
    v2t_gt = []
    for vid_id in video_metas:
        v2t_gt.append([])
        for i, query_id in enumerate(query_metas):
            if query_id.split('#', 1)[0] == vid_id:
                v2t_gt[-1].append(i)

    t2v_gt = {}
    for i, t_gts in enumerate(v2t_gt):
        for t_gt in t_gts:
            t2v_gt.setdefault(t_gt, [])
            t2v_gt[t_gt].append(i)

    return v2t_gt, t2v_gt


def eval_q2m(scores, q2m_gts):

    n_q, n_m = scores.shape

    gt_ranks = torch.zeros((n_q), dtype=torch.int32).cuda()
    aps = torch.zeros(n_q).cuda()
    for i in range(n_q):
        s = scores[i]
        sorted_idxs = torch.argsort(s)
        rank = n_m + 1
        tmp_set = []
        for k in q2m_gts[i]:
            tmp = torch.where(sorted_idxs == k)[0][0] + 1
            if tmp < rank:
                rank = tmp

        gt_ranks[i] = rank

    # compute metrics
    r1 = 100.0 * len(torch.where(gt_ranks <= 1)[0]) / n_q
    r5 = 100.0 * len(torch.where(gt_ranks <= 5)[0]) / n_q
    r10 = 100.0 * len(torch.where(gt_ranks <= 10)[0]) / n_q
    r100 = 100.0 * len(torch.where(gt_ranks <= 100)[0]) / n_q

    return (r1, r5, r10, r100)


def cal_perf(t2v_all_errors, t2v_gt):

    # video retrieval
    (t2v_r1, t2v_r5, t2v_r10, t2v_r100) = eval_q2m(t2v_all_errors, t2v_gt)

    return (t2v_r1, t2v_r5, t2v_r10, t2v_r100)


class validations(nn.Module):
    def __init__(self, cfg):
        super(validations, self).__init__()

        self.cfg = cfg


    def forward(self, model, context_dataloader, query_eval_loader,text_t2t=False):

        model.eval()

        context_info = self.compute_context_info(model, context_dataloader)
        score_sum, query_metas = self.compute_query2ctx_info(model,
                                                             query_eval_loader,
                                                             context_info)
        video_metas = context_info['video_metas']

        v2t_gt, t2v_gt = get_gt(video_metas, query_metas)

        t2v_r1, t2v_r5, t2v_r10, t2v_r100 = cal_perf(-1 * score_sum, t2v_gt)
        t2v_rsum = 0
        t2v_rsum += (t2v_r1 + t2v_r5 + t2v_r10 + t2v_r100)
        if text_t2t:
            t2t_result = self.forward_text_recall(model, query_eval_loader)
        else:
            t2t_result = [0,0,0,0,0]
        return [t2v_r1, t2v_r5, t2v_r10, t2v_r100, t2v_rsum,t2t_result]


    def compute_query2ctx_info(self, model, query_eval_loader, ctx_info):

        query_metas = []#每个查询的 cap id
        score_sum = []
        for idx, batch in tqdm(enumerate(query_eval_loader), desc="Computing q embedding", total=len(query_eval_loader)):

            batch = gpu(batch)
            query_metas.extend(batch[-1])
            query_feat = batch[0]
            query_mask = batch[1]
            _clip_scale_scores, _frame_scale_scores = model.get_pred_from_raw_query(
                query_feat, query_mask, None, ctx_info["video_proposal_feat"], ctx_info["video_feat"])
            _score_sum = self.cfg['clip_scale_w'] * _clip_scale_scores + self.cfg['frame_scale_w'] * _frame_scale_scores

            score_sum.append(_score_sum)

        score_sum = torch.cat(score_sum, dim=0)

        return score_sum, query_metas


    def compute_context_info(self, model, context_dataloader):

        n_total_vid = len(context_dataloader.dataset)
        bsz = self.cfg['eval_context_bsz']
        metas = []  # list(dicts)
        vid_proposal_feat = []
        frame_feat, frame_mask = [], []
        for idx, batch in tqdm(enumerate(context_dataloader), desc="Computing query2video scores",
                            total=len(context_dataloader)):

            batch = gpu(batch)
            metas.extend(batch[-1])
            clip_video_feat_ = batch[0]
            frame_video_feat_ = batch[1]
            frame_mask_ = batch[2]
            _frame_feat, _video_proposal_feat = model.encode_context(clip_video_feat_, frame_video_feat_, frame_mask_,train=False)

            frame_feat.append(_frame_feat)
            frame_mask.append(frame_mask_)

            vid_proposal_feat.append(_video_proposal_feat)

        vid_proposal_feat = torch.cat(vid_proposal_feat, dim=0)

        def cat_tensor(tensor_list):
            if len(tensor_list) == 0:
                return None
            else:
                seq_l = [e.shape[1] for e in tensor_list]
                b_sizes = [e.shape[0] for e in tensor_list]
                b_sizes_cumsum = np.cumsum([0] + b_sizes)
                if len(tensor_list[0].shape) == 3:
                    hsz = tensor_list[0].shape[2]
                    res_tensor = tensor_list[0].new_zeros(sum(b_sizes), max(seq_l), hsz)
                elif len(tensor_list[0].shape) == 2:
                    res_tensor = tensor_list[0].new_zeros(sum(b_sizes), max(seq_l))
                else:
                    raise ValueError("Only support 2/3 dimensional tensors")
                for i, e in enumerate(tensor_list):
                    res_tensor[b_sizes_cumsum[i]:b_sizes_cumsum[i+1], :seq_l[i]] = e
                return res_tensor
                
        return dict(
            video_metas=metas,  # list(dict) (N_videos)
            video_proposal_feat=vid_proposal_feat,
            video_feat=cat_tensor(frame_feat),
            video_mask=cat_tensor(frame_mask)
            )
    def forward_text_recall(self, model, query_eval_loader,num_videos=10000,num_queries=40000):

        model.eval()
        query_embeddings, query_metas = self.compute_query_embeddings(model, query_eval_loader)


        vid_to_indices = defaultdict(list)
        for i, meta in enumerate(query_metas):
            vid_id = meta.split('#', 1)[0]
            vid_to_indices[vid_id].append(i)
        
        unique_vid_ids = list(vid_to_indices.keys())
        n_queries = len(query_metas)
        n_videos = len(unique_vid_ids)
        print(f"Total queries: {n_queries}, total videos: {n_videos}")
        
        # 随机选择 num_videos 个视频
        if len(unique_vid_ids) > num_videos:
            selected_vids = np.random.choice(unique_vid_ids, num_videos, replace=False)
        else:
            selected_vids = unique_vid_ids

        selected_indices = []
        for vid_id in selected_vids:
            indices = vid_to_indices[vid_id]
            if len(selected_indices) + len(indices) > num_queries:
                needed = num_queries - len(selected_indices)
                selected_indices.extend(np.random.choice(indices, needed, replace=False))
                break
            else:
                selected_indices.extend(indices)
        print("Selected {} queries from {} videos.".format(len(selected_indices), len(selected_vids)))
        selected_indices = sorted(selected_indices)
        query_embeddings = query_embeddings[selected_indices]
        query_metas = [query_metas[i] for i in selected_indices]



        query_embeddings = F.normalize(query_embeddings, p=2, dim=1)

        text_sim_scores = query_embeddings @ query_embeddings.T


        t2t_gt = get_text_gt(query_metas)


        self_mask = torch.eye(text_sim_scores.shape[0], dtype=torch.bool, device=text_sim_scores.device)
        text_sim_scores.masked_fill_(self_mask, -1e9)
        
        (t2t_r1, t2t_r5, t2t_r10, t2t_r100) = eval_q2m(-1 * text_sim_scores, t2t_gt)
        t2t_rsum = t2t_r1+t2t_r5+t2t_r10+t2t_r100

        return [t2t_r1, t2t_r5, t2t_r10, t2t_r100, t2t_rsum]


    def compute_query_embeddings(self, model, query_eval_loader):
        model.eval()
        query_metas = []
        all_query_embs = []
        for idx, batch in tqdm(enumerate(query_eval_loader), desc="Computing query embeddings", total=len(query_eval_loader)):
            batch = gpu(batch)
            query_metas.extend(batch[-1])
            query_feat = batch[0]
            query_mask = batch[1]
            
            query_emb = model.encode_query(query_feat, query_mask)
            all_query_embs.append(query_emb)

        all_query_embs = torch.cat(all_query_embs, dim=0)
        return all_query_embs, query_metas
