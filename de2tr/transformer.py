# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor
import math
import numpy as np

from .attention import MultiheadAttention


def masked_softmax(scores: torch.Tensor, mask: torch.Tensor, dim: int = -1, tau: float = 1.0):
    """
    scores: (B, L)
    mask:   (B, L) bool, True=valid
    """
    scores = scores / max(tau, 1e-6)
    scores = scores.masked_fill(~mask, -1e4)
    return torch.softmax(scores, dim=dim)

def saliency_evidence_pool(
    vid_mem_bld: torch.Tensor,         # (B, V_L, D)
    saliency_scores_bl: torch.Tensor,  # (B, V_L)
    video_valid_mask: torch.Tensor,    # (B, V_L) bool, True=valid
    tau: float = 0.3,
    topk: int = 0,
):
    """
    vid_mem: (B, V_L, D)
    saliency_scores: (B, V_L)
    video_mask: (B, V_L) bool, True=valid
    return: evidence (B, D)
    """
    if topk is not None and topk > 0:
        scores = saliency_scores_bl.masked_fill(~video_valid_mask, -1e4)
        k = min(int(topk), scores.size(1))
        idx = scores.topk(k, dim=1).indices  # (B,K)
        gather_mem = vid_mem_bld.gather(1, idx.unsqueeze(-1).expand(-1, -1, vid_mem_bld.size(-1)))  # (B,K,D)
        gather_scores = saliency_scores_bl.gather(1, idx)  # (B,K)
        w = torch.softmax(gather_scores / max(tau, 1e-6), dim=1)  # (B,K)
        return (gather_mem * w.unsqueeze(-1)).sum(dim=1)

    w = masked_softmax(saliency_scores_bl, video_valid_mask, dim=1, tau=tau)  # (B,V_L)
    return (vid_mem_bld * w.unsqueeze(-1)).sum(dim=1)  # (B,D)

class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)."""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        # ReLU on every layer but the last
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

def inverse_sigmoid(x, eps=1e-3):
    # Keeps the log finite at the boundaries
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)

def gen_sineembed_for_position(pos_tensor):
    # Sinusoidal embedding of a normalised (center, width) pair
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / 128)
    center_embed = pos_tensor[:, :, 0] * scale
    pos_x = center_embed[:, :, None] / dim_t
    # Interleave sin and cos
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)

    span_embed = pos_tensor[:, :, 1] * scale
    pos_w = span_embed[:, :, None] / dim_t
    pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

    pos = torch.cat((pos_x, pos_w), dim=2)  # (center, width) embeddings
    return pos


class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_queries=2, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, query_dim=2,
                 keep_query_pos=False, query_scale_type='cond_elewise',
                 num_patterns=0,
                 modulate_t_attn=True,
                 bbox_embed_diff_each_layer=False,
                 use_bnd_priori=True, use_sal_priori=False, sal_priori_film=True,
                 saliency_in_transformer=True,
                 ):
        super().__init__()
        # Both released checkpoints were trained with the boundary prior ON and the
        # saliency prior OFF, so these defaults reproduce them. The flags only control
        # whether the priors are injected into the decoder; `sal_priori_film` controls
        # whether the (otherwise unused) saliency FiLM projections are instantiated at
        # all, which the older released checkpoint does not contain.
        self.use_bnd_priori = use_bnd_priori
        self.use_sal_priori = use_sal_priori
        self.sal_priori_film = sal_priori_film or use_sal_priori
        # In the newer layout the saliency scores are computed inside this class; in the
        # older one they are computed in model.py. The two are numerically identical but
        # accumulate gradients in a different order.
        self.saliency_in_transformer = saliency_in_transformer
        # Text-to-video cross-attention encoder
        t2v_encoder_layer = T2V_TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.t2v_encoder = TransformerEncoder(t2v_encoder_layer, num_encoder_layers, encoder_norm)

        # Self-attention encoder over [global token, video tokens]
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        self.boundary_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 2)
        )
        self.boundary_bias_scale = 0.1
        self.boundary_feat_proj_s = nn.Linear(1, d_model)
        self.boundary_feat_proj_e = nn.Linear(1, d_model)

        self.saliency_proj1 = None
        self.saliency_proj2 = None
        # Saliency -> evidence pooling -> class FiLM
        self.saliency_pool_tau = 0.3   # softmax temperature
        self.saliency_topk = 0         # 0 = no top-k; >0 = pool only the top-k clips
        # Decoder
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before, keep_query_pos=keep_query_pos,
                                                sal_priori_film=self.sal_priori_film)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec,
                                          d_model=d_model, query_dim=query_dim, keep_query_pos=keep_query_pos, query_scale_type=query_scale_type,
                                          modulate_t_attn=modulate_t_attn,
                                          bbox_embed_diff_each_layer=bbox_embed_diff_each_layer)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead
        self.dec_layers = num_decoder_layers
        self.num_queries = num_queries
        self.num_patterns = num_patterns

    def _reset_parameters(self):
        # Xavier-init every parameter with dim > 1
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # `video_length` is only used by the tvsum pipeline
    def forward(self, src, mask, query_embed, pos_embed, video_length=None):
        """
        Args:
            src: (batch_size, L, d) input features, L = 1 + L_vid + L_txt
            mask: (batch_size, L), 1 on padded positions
            query_embed: (#queries, d) learnable moment queries
            pos_embed: (batch_size, L, d), aligned with src

        Returns:
            hs, references, memory_local, memory_global, src_txt, src_vid,
            boundary_logits, saliency_scores
        """
        # src is [global token, video tokens, text tokens]
        bs, l, d = src.shape
        src = src.permute(1, 0, 2)  # (L, batch_size, d)
        pos_embed = pos_embed.permute(1, 0, 2)   # (L, batch_size, d)
        refpoint_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)  # (#queries, batch_size, 2)

        src = self.t2v_encoder(src, src_key_padding_mask=mask, pos=pos_embed, video_length=video_length)  # (L, batch_size, d)
        src_txt = src[video_length + 1:].transpose(0, 1)  # text features
        src_vid = src[1:video_length + 1].transpose(0, 1)  # video features
        src = src[:video_length + 1]  # keep [global, video] for the encoder
        mask = mask[:, :video_length + 1]
        pos_embed = pos_embed[:video_length + 1]

        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)  # (L, batch_size, d)
        memory_global, memory_local = memory[0], memory[1:]  # global token, local (video) tokens
        mask_local = mask[:, 1:]
        pos_embed_local = pos_embed[1:]


        saliency_scores = None
        if self.saliency_in_transformer:
            proj1 = self.saliency_proj1 if self.saliency_proj1 is not None else getattr(self.decoder, "saliency_proj1", None)
            proj2 = self.saliency_proj2 if self.saliency_proj2 is not None else getattr(self.decoder, "saliency_proj2", None)
            assert proj1 is not None and proj2 is not None, "saliency_proj1/2 not set (assign in DE2TR.__init__)."

            vid_mem_bld = memory_local.transpose(0, 1)  # (B,V_L,D)
            saliency_scores = (
                torch.sum(proj1(vid_mem_bld) * proj2(memory_global).unsqueeze(1), dim=-1) / np.sqrt(self.d_model)
            )   # (B,V_L)
        saliency_evidence = None
        if self.use_sal_priori:
            video_valid_mask = ~mask_local  # True=valid
            saliency_evidence = saliency_evidence_pool(
                vid_mem_bld=vid_mem_bld,
                saliency_scores_bl=saliency_scores.detach(),  # detached: teacher-style, more stable
                video_valid_mask=video_valid_mask,
                tau=self.saliency_pool_tau,
                topk=self.saliency_topk,
            )  # (B,D)

        # Boundary prior logits over the video memory
        boundary_logits = self.boundary_head(memory_local.transpose(0, 1))

        # Zero out the padding positions before building the prior features
        valid = (~mask_local).float()  # (B, V_L), mask_local True=pad
        b_s = boundary_logits[..., 0:1] * valid.unsqueeze(-1)  # (B,V_L,1)
        b_e = boundary_logits[..., 1:2] * valid.unsqueeze(-1)  # (B,V_L,1)
        if self.use_bnd_priori:
            boundary_feat_s = self.boundary_feat_proj_s(b_s).detach()  # (B,V_L,d)
            boundary_feat_e = self.boundary_feat_proj_e(b_e).detach()  # (B,V_L,d)
        else:
            boundary_feat_s = None
            boundary_feat_e = None

        tgt = torch.zeros(refpoint_embed.shape[0], bs, 3 * d, device=src.device)

        hs, references = self.decoder(tgt, memory_local, memory_local, memory_key_padding_mask=mask_local,
                          pos=pos_embed_local, refpoints_unsigmoid=refpoint_embed, boundary_score=None, boundary_bias_scale=self.boundary_bias_scale,
                          boundary_feat_s=boundary_feat_s, boundary_feat_e=boundary_feat_e, saliency_evidence=saliency_evidence,)  # (#layers, #queries, batch_size, d)

        memory_local = memory_local.transpose(0, 1)  # (batch_size, L, d)
        return hs, references, memory_local, memory_global, src_txt, src_vid, boundary_logits, saliency_scores

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    # `**kwargs` is only used by the tvsum pipeline
    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                **kwargs):
        output = src
        intermediate = []

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos, **kwargs)
            if self.return_intermediate:
                intermediate.append(output)

        if self.norm is not None:
            output = self.norm(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False,
                 d_model=256, query_dim=2, keep_query_pos=False, query_scale_type='cond_elewise',
                 modulate_t_attn=False,
                 bbox_embed_diff_each_layer=False,
                 ):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        assert return_intermediate
        self.query_dim = query_dim

        assert query_scale_type in ['cond_elewise', 'cond_scalar', 'fix_elewise']
        self.query_scale_type = query_scale_type
        if query_scale_type == 'cond_elewise':
            self.query_scale_left = MLP(d_model, d_model, d_model, 2)
            self.query_scale_right = MLP(d_model, d_model, d_model, 2)
            self.query_scale_cls  = MLP(d_model, d_model, d_model, 2)
        elif query_scale_type == 'cond_scalar':
            self.query_scale_left = MLP(d_model, d_model, 1, 2)
            self.query_scale_right = MLP(d_model, d_model, 1, 2)
            self.query_scale_cls  = MLP(d_model, d_model, 1, 2)
        elif query_scale_type == 'fix_elewise':
            self.query_scale_left = nn.Embedding(num_layers, d_model)
            self.query_scale_right = nn.Embedding(num_layers, d_model)
            self.query_scale_cls  = nn.Embedding(num_layers, d_model)
        else:
            raise NotImplementedError("Unknown query_scale_type: {}".format(query_scale_type))

        self.ref_point_head_span_left = MLP(d_model, d_model, d_model, 2)  # (nq,bs,2d)
        self.ref_point_head_span_right = MLP(d_model, d_model, d_model, 2)  # (nq,bs,2d)
        self.ref_point_head_cls  = MLP(d_model, d_model, d_model, 2)      # (nq,bs,d)

        # for DAB-DETR
        if bbox_embed_diff_each_layer:
            self.bbox_embed = nn.ModuleList([
                nn.ModuleList([MLP(d_model, d_model, 1, 3) for _ in range(num_layers)]),
                nn.ModuleList([MLP(d_model, d_model, 1, 3) for _ in range(num_layers)]),
            ])
        else:
            self.bbox_embed = nn.ModuleList([
                MLP(d_model, d_model, 1, 3),
                MLP(d_model, d_model, 1, 3),
            ])

        # init bbox_embed
        if bbox_embed_diff_each_layer:
            for i in range(2):
                for bbox_embed in self.bbox_embed[i]:
                    nn.init.constant_(bbox_embed.layers[-1].weight.data, 0)
                    nn.init.constant_(bbox_embed.layers[-1].bias.data, 0)
        else:
            for i in range(2):
                nn.init.constant_(self.bbox_embed[i].layers[-1].weight.data, 0)
                nn.init.constant_(self.bbox_embed[i].layers[-1].bias.data, 0)
        self.d_model = d_model
        self.modulate_t_attn = modulate_t_attn
        self.bbox_embed_diff_each_layer = bbox_embed_diff_each_layer

        if modulate_t_attn:
            self.ref_anchor_head_left = MLP(d_model, d_model, 1, 2)
            self.ref_anchor_head_right = MLP(d_model, d_model, 1, 2)
            self.ref_anchor_head_cls = MLP(d_model, d_model, 1, 2)

        # Only the first decoder layer keeps positional query projections
        if not keep_query_pos:
            for layer_id in range(num_layers - 1):
                self.layers[layer_id + 1].ca_qpos_proj = None

        self.span_embed = None
        self.class_embed = None
        self.iou_head = None


    def _build_query_gated_bias_1d(
        self,
        score_1d: torch.Tensor,      # (B, V_L) logits-like
        center: torch.Tensor,        # (B, nq, 1) in [0,1]
        sigma: torch.Tensor,         # (B, nq, 1) in [0,1]
        nhead: int,
        bias_scale: float,
    ) -> torch.Tensor:
        """
        return: (B*nhead, nq, V_L) additive bias
        """
        B, V_L = score_1d.shape
        t = (torch.arange(V_L, device=score_1d.device, dtype=torch.float32) + 0.5) / float(V_L)  # (V_L,)
        t = t.view(1, 1, V_L)  # (1,1,V_L)

        sigma = sigma.clamp(min=1e-6)
        gate = torch.exp(-0.5 * ((t - center) / sigma) ** 2)  # (B,nq,V_L)

        bias = (bias_scale * score_1d.unsqueeze(1)) * gate     # (B,nq,V_L)
        bias = bias.repeat(nhead, 1, 1)                        # (B*nhead,nq,V_L)
        return bias
    def _refine_center_softargmax_1d(
        self,
        score_1d: torch.Tensor,   # (B, V_L) logits-like
        center: torch.Tensor,     # (B, nq, 1) in [0,1]
        sigma: torch.Tensor,      # (B, nq, 1) in [0,1]
        tau: float = 0.2,
        win_mul: float = 3.0,
    ) -> torch.Tensor:
        """
        Use boundary prior to refine center via soft-argmax within a local window.
        return refined_center: (B, nq, 1)
        """
        B, V_L = score_1d.shape
        t = (torch.arange(V_L, device=score_1d.device, dtype=torch.float32) + 0.5) / float(V_L)  # (V_L,)
        t = t.view(1, 1, V_L)  # (1,1,V_L)

        # window radius proportional to sigma
        rad = (win_mul * sigma).clamp(min=1.0 / V_L, max=0.5)  # (B,nq,1)
        # mask outside window
        in_win = (t - center).abs() <= rad                     # (B,nq,V_L) bool
        masked = score_1d.unsqueeze(1).to(torch.float32)       # (B,1,V_L)
        masked = masked.expand(B, center.shape[1], V_L)        # (B,nq,V_L)
        masked = masked.masked_fill(~in_win, -1e4)

        # softmax -> distribution over positions
        p = torch.softmax(masked / tau, dim=-1)                # (B,nq,V_L)
        refined = (p * t).sum(dim=-1, keepdim=True)            # (B,nq,1)
        return refined

    def forward(self, tgt, memory, memory_local,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                refpoints_unsigmoid: Optional[Tensor] = None,  # num_queries, bs, 2
                boundary_score: Optional[Tensor] = None,        # (B,V_L,2) or (B,V_L)
                boundary_bias_scale: float = 1.0,
                boundary_feat_s: Optional[Tensor] = None,  # (B,V_L,d)
                boundary_feat_e: Optional[Tensor] = None,  # (B,V_L,d)
                saliency_evidence: Optional[Tensor] = None,  # (B,D)
                ):
        output = tgt  # (num_queries, batch_size, 3* d_model)
        nq, bs, _ = output.size()
        d0 = self.d_model

        intermediate = []
        reference_points = refpoints_unsigmoid.sigmoid()  # (num_queries, batch_size, 2)
        ref_points = [reference_points]

        for layer_id, layer in enumerate(self.layers):
            obj_center = reference_points[..., :self.query_dim]  # (center, width)
            query_sine_embed_cls = gen_sineembed_for_position(obj_center)
            left_point = (obj_center[..., 0] - 0.5 * reference_points[..., 1]).clamp(min=0.0)
            right_point = (obj_center[..., 0] + 0.5 * reference_points[..., 1]).clamp(max=1.0)
            # The boundary window is capped so it does not grow with the span
            bnd = (0.5 * reference_points[..., 1]).clamp(min=0.05, max=0.2)
            left_bnd = torch.stack([left_point, bnd.clone()], dim=-1)  # (num_queries, batch_size, 2)
            right_bnd = torch.stack([right_point, bnd.clone()], dim=-1)  # (num_queries, batch_size, 2)
            query_sine_embed_left = gen_sineembed_for_position(left_bnd)
            query_sine_embed_right = gen_sineembed_for_position(right_bnd)

            query_pos_span_left = self.ref_point_head_span_left(query_sine_embed_left)  # (nq,bs,d)
            query_pos_span_right = self.ref_point_head_span_right(query_sine_embed_right)  # (nq,bs,d)
            query_pos_span = torch.cat([query_pos_span_left, query_pos_span_right], dim=-1)  # (nq,bs,2d)
            query_pos_cls  = self.ref_point_head_cls(query_sine_embed_cls)   # (nq,bs,d)

            query_pos = torch.cat([query_pos_span, query_pos_cls], dim=-1)  # (nq,bs,3d)

            
            # For the first decoder layer, we do not apply transformation over p_s
            if self.query_scale_type != 'fix_elewise':
                if layer_id == 0:
                    pos_transformation_left = 1
                    pos_transformation_right = 1
                    pos_transformation_cls  = 1
                else:
                    pos_transformation_left = self.query_scale_left(output[..., :d0])
                    pos_transformation_right = self.query_scale_right(output[..., d0:2*d0])
                    pos_transformation_cls  = self.query_scale_cls (output[..., 2*d0:])
            else:
                pos_transformation_left = self.query_scale_left.weight[layer_id]
                pos_transformation_right = self.query_scale_right.weight[layer_id]
                pos_transformation_cls  = self.query_scale_cls.weight[layer_id]

            query_sine_embed_left = query_sine_embed_left * pos_transformation_left
            query_sine_embed_right = query_sine_embed_right * pos_transformation_right
            query_sine_embed_cls = query_sine_embed_cls * pos_transformation_cls

            if self.modulate_t_attn:
                reft_cond_left = self.ref_anchor_head_left(output[..., :d0]).sigmoid()  # nq, bs, 1
                query_sine_embed_left *= (reft_cond_left[..., 0] / left_bnd[..., 1]).unsqueeze(-1)
                reft_cond_right = self.ref_anchor_head_right(output[..., d0:2*d0]).sigmoid()  # nq, bs, 1
                query_sine_embed_right *= (reft_cond_right[..., 0] / right_bnd[..., 1]).unsqueeze(-1)
                reft_cond_cls = self.ref_anchor_head_cls(output[..., 2*d0:]).sigmoid()  # nq, bs, 1
                query_sine_embed_cls *= (reft_cond_cls[..., 0] / reference_points[..., 1]).unsqueeze(-1)
            
            query_sine_embed = torch.cat([query_sine_embed_left, query_sine_embed_right, query_sine_embed_cls], dim=-1)  # (nq,bs,3d)

            if layer_id == 0:
                r = torch.zeros(nq, nq, bs, device=output.device)
            else:
                cls_score = self.class_embed(output[..., 2*d0:])
                cls_score = F.softmax(cls_score, dim=-1)[..., 0] # nq, bs
                cls_score_row = cls_score.unsqueeze(1).repeat(1, nq, 1)  # nq, nq, bs
                cls_score_col = cls_score.unsqueeze(0).repeat(nq, 1, 1)  # nq, nq, bs

                iou_score = self.iou_head(output[..., :2*d0])[..., 0].sigmoid()  # nq, bs
                iou_score_row = iou_score.unsqueeze(1).repeat(1, nq, 1)  # nq, nq, bs
                iou_score_col = iou_score.unsqueeze(0).repeat(nq, 1, 1)  # nq, nq, bs

                score_row = cls_score_row * iou_score_row
                score_col = cls_score_col * iou_score_col
                r_rank = (score_row >= score_col).float() * 2 - 1  # nq, nq, bs

                spans_log_ds = self.span_embed[0](output[..., :d0]) # nq, bs, 1
                spans_log_de = self.span_embed[1](output[..., d0:2*d0]) # nq, bs, 1
                spans = torch.cat([spans_log_ds, spans_log_de], dim=-1)  # nq, bs, 2

                st = reference_points[..., 0] - 0.5 * reference_points[..., 1]
                ed = reference_points[..., 0] + 0.5 * reference_points[..., 1]
                reference_se = torch.stack([st, ed], dim=-1)  # (num_queries, batch_size, 2)
                spans += inverse_sigmoid(reference_se)
                spans = spans.sigmoid()
                spans_st = spans[..., 0]
                spans_ed = spans[..., 1]
                spans = torch.stack([spans_st, spans_ed], dim=-1)  # nq, bs, 2
                spans_row = spans.unsqueeze(1).repeat(1, nq, 1, 1)
                spans_col = spans.unsqueeze(0).repeat(nq, 1, 1, 1)
                r_spt = 1 - torch.mean((spans_row - spans_col) ** 2, dim=-1)  # nq, nq, bs

                r = r_rank * r_spt

            # ===== build query-conditioned gated bias (start->left, end->right) =====
            span_attn_bias_left = None
            span_attn_bias_right = None
            if boundary_score is not None and layer_id >= 2:
                # boundary_score: (B,V_L,2) logits
                if boundary_score.dim() == 3 and boundary_score.size(-1) == 2:
                    score_s = boundary_score[..., 0]  # (B,V_L)
                    score_e = boundary_score[..., 1]  # (B,V_L)
                else:
                    # fallback: single-channel -> use same for both sides
                    score_s = boundary_score
                    score_e = boundary_score

                # gate centers/sigmas: (B,nq,1)
                center_left = left_point.permute(1, 0).unsqueeze(-1)   # (B,nq,1)
                center_right = right_point.permute(1, 0).unsqueeze(-1) # (B,nq,1)
                sigma = bnd.permute(1, 0).unsqueeze(-1)                # (B,nq,1)
                # Refine centers using the boundary peaks; detached so it stays a prior.
                score_s_det = score_s.detach()
                score_e_det = score_e.detach()
                center_left = self._refine_center_softargmax_1d(score_s_det, center_left, sigma, tau=0.2, win_mul=3.0)
                center_right = self._refine_center_softargmax_1d(score_e_det, center_right, sigma, tau=0.2, win_mul=3.0)

                span_attn_bias_left = self._build_query_gated_bias_1d(
                    score_1d=score_s.to(dtype=torch.float32),
                    center=center_left.to(dtype=torch.float32),
                    sigma=sigma.to(dtype=torch.float32),
                    nhead=self.layers[0].nhead,
                    bias_scale=boundary_bias_scale,
                )
                span_attn_bias_right = self._build_query_gated_bias_1d(
                    score_1d=score_e.to(dtype=torch.float32),
                    center=center_right.to(dtype=torch.float32),
                    sigma=sigma.to(dtype=torch.float32),
                    nhead=self.layers[0].nhead,
                    bias_scale=boundary_bias_scale,
                )
            if span_attn_bias_left is not None:
                span_attn_bias_left = span_attn_bias_left.clamp(-5.0, 5.0)
                span_attn_bias_left = span_attn_bias_left - span_attn_bias_left.max(dim=-1, keepdim=True)[0]
            if span_attn_bias_right is not None:
                span_attn_bias_right = span_attn_bias_right.clamp(-5.0, 5.0)
                span_attn_bias_right = span_attn_bias_right - span_attn_bias_right.max(dim=-1, keepdim=True)[0]

            output = layer(output, memory, memory_local, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos, query_sine_embed=query_sine_embed,
                           is_first=(layer_id == 0),
                           competition_matrix=r, 
                           span_attn_bias_left=span_attn_bias_left,
                           span_attn_bias_right=span_attn_bias_right,
                            boundary_feat_s=boundary_feat_s,
                            boundary_feat_e=boundary_feat_e,
                            saliency_evidence=saliency_evidence, 
                           )

            # iter update reference points and hd
            if self.bbox_embed is not None:
                if self.bbox_embed_diff_each_layer:
                    tmp_log_s = self.bbox_embed[0][layer_id](output[..., :d0])
                    tmp_log_e = self.bbox_embed[1][layer_id](output[..., d0:2*d0])
                else:
                    tmp_log_s = self.bbox_embed[0](output[..., :d0])
                    tmp_log_e = self.bbox_embed[1](output[..., d0:2*d0])
                tmp_log_s = tmp_log_s.squeeze(-1)
                tmp_log_e = tmp_log_e.squeeze(-1)
                
                tmp = torch.stack([tmp_log_s, tmp_log_e], dim=-1)  # (num_queries, batch_size, 2)
                st = reference_points[..., 0] - 0.5 * reference_points[..., 1]
                ed = reference_points[..., 0] + 0.5 * reference_points[..., 1]
                reference_se = torch.stack([st, ed], dim=-1)  # (num_queries, batch_size, 2)
                tmp[..., :self.query_dim] += inverse_sigmoid(reference_se)
                new_reference_se = tmp[..., :self.query_dim].sigmoid()

                st = torch.min(new_reference_se[..., 0], new_reference_se[..., 1])
                ed = torch.max(new_reference_se[..., 0], new_reference_se[..., 1])
                new_reference_se = torch.stack([st, ed], dim=-1)

                new_c = (new_reference_se[..., 0] + new_reference_se[..., 1]) * 0.5
                new_w = (new_reference_se[..., 1] - new_reference_se[..., 0]).clamp(min=1e-3)
                new_reference_points = torch.stack([new_c, new_w], dim=-1)  # (num_queries, batch_size, 2)
                if layer_id != self.num_layers - 1:
                    ref_points.append(new_reference_points)
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                if self.norm is not None:
                    norm_output = [self.norm(output[..., idx*d0:(idx+1)*d0]) for idx in range(3)]
                    norm_output = torch.cat(norm_output, dim=-1)
                    intermediate.append(norm_output)
                else:
                    intermediate.append(output)

        if self.norm is not None:
            norm_output = [self.norm(output[..., idx*d0:(idx+1)*d0]) for idx in range(3)]
            norm_output = torch.cat(norm_output, dim=-1)
            output = norm_output
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            if self.bbox_embed is not None:
                return [
                    torch.stack(intermediate).transpose(1, 2),
                    torch.stack(ref_points).transpose(1, 2),
                ]
            else:
                return [
                    torch.stack(intermediate).transpose(1, 2),
                    reference_points.unsqueeze(0).transpose(1, 2),
                ]

        return output.unsqueeze(0)

class TransformerEncoderLayerThin(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src2 = self.linear(src2)
        src = src + self.dropout(src2)
        src = self.norm(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        """not used"""
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)

class T2V_TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.nhead = nhead

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     video_length=None):
        assert video_length is not None

        pos_src = self.with_pos_embed(src, pos)
        global_token, q, k, v = src[0].unsqueeze(0), pos_src[1:video_length + 1], pos_src[video_length + 1:], src[video_length + 1:]

        qmask, kmask = src_key_padding_mask[:, 1:video_length + 1].unsqueeze(2), src_key_padding_mask[:, video_length + 1:].unsqueeze(1)
        attn_mask = torch.matmul(qmask.float(), kmask.float()).bool().repeat(self.nhead, 1, 1)

        src2 = self.self_attn(q, k, value=v, attn_mask=attn_mask,
                              key_padding_mask=src_key_padding_mask[:, video_length + 1:])[0]
        src2 = src[1:video_length + 1] + self.dropout1(src2)
        src3 = self.norm1(src2)
        src3 = self.linear2(self.dropout(self.activation(self.linear1(src3))))
        src2 = src2 + self.dropout2(src3)
        src2 = self.norm2(src2)
        src2 = torch.cat([global_token, src2], dim=0)
        src = torch.cat([src2, src[video_length + 1:]])
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        pos_src = self.with_pos_embed(src2, pos)
        global_token, q, k, v = src[0].unsqueeze(0), pos_src[1:76], pos_src[76:], src2[76:]

        src2 = self.self_attn(q, k, value=v, attn_mask=src_key_padding_mask[:, 1:76].permute(1,0),
                              key_padding_mask=src_key_padding_mask[:, 76:])[0]
        src2 = src[1:76] + self.dropout1(src2)
        src3 = self.norm1(src2)
        src3 = self.linear2(self.dropout(self.activation(self.linear1(src3))))
        src2 = src2 + self.dropout2(src3)
        src2 = self.norm2(src2)
        src2 = torch.cat([global_token, src2], dim=0)
        src = torch.cat([src2, src[76:]])
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                **kwargs):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        # For tvsum, add kwargs
        return self.forward_post(src, src_mask, src_key_padding_mask, pos, **kwargs)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


def _film_block(d_model):
    """FiLM head shared by the boundary and saliency priors: d -> 2d."""
    return nn.Sequential(
        nn.Linear(d_model, d_model),
        nn.ReLU(inplace=True),
        nn.Linear(d_model, 2 * d_model),
    )


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, keep_query_pos=False,
                 rm_self_attn_decoder=False, sal_priori_film=True):
        super().__init__()
        # Decoder Self-Attention
        d_model_3 = d_model * 3
        self.d_model = d_model
        if not rm_self_attn_decoder:
            self.sa_qcontent_proj = nn.Linear(d_model_3, d_model_3)
            self.sa_qpos_proj = nn.Linear(d_model_3, d_model_3)
            self.sa_kcontent_proj = nn.Linear(d_model_3, d_model_3)
            self.sa_kpos_proj = nn.Linear(d_model_3, d_model_3)
            self.sa_v_proj = nn.Linear(d_model_3, d_model_3)
            self.self_attn = MultiheadAttention(d_model_3, nhead, dropout=dropout, vdim=d_model_3)

            self.norm1 = nn.LayerNorm(d_model_3)
            self.dropout1 = nn.Dropout(dropout)

        # Decoder Cross-Attention for span(left and right).
        # The names in these two loops are the checkpoint keys and the iteration
        # order is the parameter order that `Transformer._reset_parameters` walks,
        # so neither may be reordered or merged into a single loop.
        for name in ("ca_qcontent_proj", "ca_qpos_proj", "ca_kcontent_proj",
                     "ca_kpos_proj", "ca_v_proj", "ca_qpos_sine_proj"):
            setattr(self, name, nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(2)]))
        self.cross_attn = nn.ModuleList([MultiheadAttention(d_model * 2, nhead, dropout=dropout, vdim=d_model) for _ in range(2)])

        # Decoder Cross-Attention for class
        for name in ("ca_qcontent_proj_c", "ca_qpos_proj_c", "ca_kcontent_proj_c",
                     "ca_kpos_proj_c", "ca_v_proj_c", "ca_qpos_sine_proj_c"):
            setattr(self, name, nn.Linear(d_model, d_model))
        self.cross_attn_c = MultiheadAttention(d_model * 2, nhead, dropout=dropout, vdim=d_model)

        self.nhead = nhead
        self.rm_self_attn_decoder = rm_self_attn_decoder

        # Implementation of Feedforward model
        self.linear1 = nn.ModuleList([nn.Linear(d_model, dim_feedforward) for _ in range(2)])
        self.dropout = nn.ModuleList([nn.Dropout(dropout) for _ in range(2)])
        self.linear2 = nn.ModuleList([nn.Linear(dim_feedforward, d_model) for _ in range(2)])

        self.norm2 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(2)])
        self.norm3 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(2)])
        self.dropout2 = nn.ModuleList([nn.Dropout(dropout) for _ in range(2)])
        self.dropout3 = nn.ModuleList([nn.Dropout(dropout) for _ in range(2)])

        self.linear1_c = nn.Linear(d_model, dim_feedforward)
        self.dropout_c = nn.Dropout(dropout)
        self.linear2_c = nn.Linear(dim_feedforward, d_model)

        self.norm2_c = nn.LayerNorm(d_model)
        self.norm3_c = nn.LayerNorm(d_model)
        self.dropout2_c = nn.Dropout(dropout)
        self.dropout3_c = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.keep_query_pos = keep_query_pos


        # Lightweight fusion of the two boundary branches after each update
        self.span_lr_fuse = nn.Sequential(
            nn.Linear(2 * d_model, 2 * d_model),
            nn.ReLU(inplace=True),
            nn.Linear(2 * d_model, 2 * d_model),
        )
        self.span_lr_fuse_norm = nn.LayerNorm(2 * d_model)

        mlp_hd = 16
        self.sa_competition_mlp = nn.Sequential(
            nn.Linear(1, mlp_hd),
            nn.ReLU(),
            nn.Linear(mlp_hd, 1)
        )

        self.boundary_film_left = _film_block(d_model)
        self.boundary_film_right = _film_block(d_model)
        # Unused unless the saliency prior is active, but the newer released
        # checkpoint stores its weights, so it is instantiated by default. Omitting it
        # reproduces the parameter layout of the older (+ BAS) checkpoint.
        self.saliency_cls_film = _film_block(d_model) if sal_priori_film else None
        self.saliency_film_scale = 0.1

        # init last layer to zero => film starts as identity (stable)
        for film in (self.boundary_film_left, self.boundary_film_right, self.saliency_cls_film):
            if film is not None:
                nn.init.constant_(film[-1].weight, 0.0)
                nn.init.constant_(film[-1].bias, 0.0)


    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def _cross_attn_branch(self, tgt_branch, memory_src, pos, query_pos, query_sine_embed,
                           qcontent_proj, qpos_proj, kcontent_proj, kpos_proj, v_proj,
                           qpos_sine_proj, cross_attn, use_qpos, attn_mask,
                           memory_key_padding_mask, film=None):
        """One query branch of the cross-attention stage (span-left, span-right, class).

        `qpos_proj` is None on every layer but the first (`TransformerDecoder` clears
        `ca_qpos_proj` when `keep_query_pos` is off), so it is read only under `use_qpos`.
        `film` is an optional (projection, boundary_feature) pair; when given, the
        boundary prior modulates the values before attention.
        """
        q_content = qcontent_proj(tgt_branch)     # (nq,bs,d)
        k_content = kcontent_proj(memory_src)     # (hw,bs,d)
        v = v_proj(memory_src)                    # (hw,bs,d)
        k_pos = kpos_proj(pos)                    # (hw,bs,d)

        if film is not None:
            film_proj, boundary_feat = film
            b = boundary_feat.permute(1, 0, 2).to(v.dtype)  # (V_L,B,d)
            gb = film_proj(b)                               # (V_L,B,2d)
            gamma, beta = gb.chunk(2, dim=-1)
            gamma = torch.tanh(gamma) * 0.1  # keep small
            v = v * (1.0 + gamma) + beta

        num_queries, bs, n_model = q_content.shape           # n_model = d
        hw, _, _ = k_content.shape

        if use_qpos:
            q_pos = qpos_proj(query_pos)  # (nq,bs,d)
            q = q_content + q_pos
            k = k_content + k_pos
        else:
            q = q_content
            k = k_content

        q = q.view(num_queries, bs, self.nhead, n_model // self.nhead)
        q_sine = qpos_sine_proj(query_sine_embed)  # (nq,bs,d)
        q_sine = q_sine.view(num_queries, bs, self.nhead, n_model // self.nhead)
        q = torch.cat([q, q_sine], dim=3).view(num_queries, bs, n_model * 2)  # (nq,bs,2d)

        k = k.view(hw, bs, self.nhead, n_model // self.nhead)
        k_pos = k_pos.view(hw, bs, self.nhead, n_model // self.nhead)
        k = torch.cat([k, k_pos], dim=3).view(hw, bs, n_model * 2)  # (hw,bs,2d)

        return cross_attn(
            query=q, key=k, value=v,
            attn_mask=attn_mask, key_padding_mask=memory_key_padding_mask
        )[0]

    def _ffn_residual(self, tgt_branch, tgt2, linear1, dropout, linear2,
                      dropout2, dropout3, norm2, norm3):
        """Post-norm FFN block, shared by the three query branches."""
        tgt_branch = tgt_branch + dropout2(tgt2)
        tgt_branch = norm2(tgt_branch)
        tgt2_ffn = linear2(dropout(self.activation(linear1(tgt_branch))))
        tgt_branch = tgt_branch + dropout3(tgt2_ffn)
        tgt_branch = norm3(tgt_branch)
        return tgt_branch

    def forward(self, tgt, memory, memory_local,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                query_sine_embed=None,
                is_first=False,
                competition_matrix=None,
                span_attn_bias_left: Optional[Tensor] = None,
                span_attn_bias_right: Optional[Tensor] = None,
                boundary_feat_s: Optional[Tensor] = None,  # (B,V_L,d)
                boundary_feat_e: Optional[Tensor] = None,  # (B,V_L,d)
                saliency_evidence: Optional[Tensor] = None,  # (B,V_L,d)
                ):

        # ========== Begin of Self-Attention =============
        if not self.rm_self_attn_decoder:
            # Apply projections here
            # shape: num_queries x batch_size x 256
            q_content = self.sa_qcontent_proj(tgt)  # target is the input of the first decoder layer. zero by default.
            q_pos = self.sa_qpos_proj(query_pos)
            k_content = self.sa_kcontent_proj(tgt)
            k_pos = self.sa_kpos_proj(query_pos)
            v = self.sa_v_proj(tgt)

            num_queries, bs, n_model = q_content.shape
            hw, _, _ = k_content.shape

            q = q_content + q_pos
            k = k_content + k_pos

            if competition_matrix is not None:
                sa_decay = torch.sigmoid(self.sa_competition_mlp(competition_matrix.unsqueeze(-1))).squeeze(-1) # nq, nq, bs
                sa_decay = sa_decay.permute(2, 0, 1).repeat(self.nhead, 1, 1)
            else:
                sa_decay = None

            tgt2 = self.self_attn(q, k, value=v, attn_mask=tgt_mask,
                                  key_padding_mask=tgt_key_padding_mask,
                                  sa_decay=sa_decay)[0]
            # ========== End of Self-Attention =============

            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)

        # ========== Begin of Cross-Attention =============
        tgt_span_left = tgt[..., :self.d_model]
        tgt_span_right = tgt[..., self.d_model:self.d_model * 2]
        tgt_class = tgt[..., self.d_model * 2:]

        if saliency_evidence is not None and self.saliency_cls_film is not None:
            gb = self.saliency_cls_film(saliency_evidence.to(tgt_class.dtype))  # (B,2d)
            gamma, beta = gb.chunk(2, dim=-1)                                   # (B,d), (B,d)
            gamma = torch.tanh(gamma) * self.saliency_film_scale
            tgt_class = tgt_class * (1.0 + gamma.unsqueeze(0)) + beta.unsqueeze(0)  # (nq,B,d)
        
        # The three branches share one implementation and differ only in the target
        # slice, the memory they read, the projection set and the boundary prior.
        # The span branches' `ca_qpos_proj` is None on every layer but the first, so
        # it may only be indexed when the branch that consumes it will run.
        use_qpos = is_first or self.keep_query_pos

        tgt2_span_l = self._cross_attn_branch(
            tgt_span_left, memory, pos,
            query_pos[..., :self.d_model],
            query_sine_embed[..., :self.d_model],
            self.ca_qcontent_proj[0],
            self.ca_qpos_proj[0] if use_qpos else None,
            self.ca_kcontent_proj[0],
            self.ca_kpos_proj[0], self.ca_v_proj[0], self.ca_qpos_sine_proj[0],
            self.cross_attn[0], use_qpos,
            attn_mask=span_attn_bias_left if span_attn_bias_left is not None else memory_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            film=(self.boundary_film_left, boundary_feat_s) if boundary_feat_s is not None else None,
        )
        tgt2_span_r = self._cross_attn_branch(
            tgt_span_right, memory, pos,
            query_pos[..., self.d_model:self.d_model * 2],
            query_sine_embed[..., self.d_model:self.d_model * 2],
            self.ca_qcontent_proj[1],
            self.ca_qpos_proj[1] if use_qpos else None,
            self.ca_kcontent_proj[1],
            self.ca_kpos_proj[1], self.ca_v_proj[1], self.ca_qpos_sine_proj[1],
            self.cross_attn[1], use_qpos,
            attn_mask=span_attn_bias_right if span_attn_bias_right is not None else memory_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            film=(self.boundary_film_right, boundary_feat_e) if boundary_feat_e is not None else None,
        )
        # The class branch reads memory_local, takes no boundary prior, and its own
        # qpos projection is never cleared.
        tgt2_c = self._cross_attn_branch(
            tgt_class, memory_local, pos,
            query_pos[..., self.d_model * 2:],
            query_sine_embed[..., self.d_model * 2:],
            self.ca_qcontent_proj_c, self.ca_qpos_proj_c, self.ca_kcontent_proj_c,
            self.ca_kpos_proj_c, self.ca_v_proj_c, self.ca_qpos_sine_proj_c,
            self.cross_attn_c, use_qpos,
            attn_mask=memory_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        tgt_class = self._ffn_residual(
            tgt_class, tgt2_c, self.linear1_c, self.dropout_c, self.linear2_c,
            self.dropout2_c, self.dropout3_c, self.norm2_c, self.norm3_c)
        tgt_span_left = self._ffn_residual(
            tgt_span_left, tgt2_span_l, self.linear1[0], self.dropout[0], self.linear2[0],
            self.dropout2[0], self.dropout3[0], self.norm2[0], self.norm3[0])
        tgt_span_right = self._ffn_residual(
            tgt_span_right, tgt2_span_r, self.linear1[1], self.dropout[1], self.linear2[1],
            self.dropout2[1], self.dropout3[1], self.norm2[1], self.norm3[1])

        # ========== End of Cross-Attention =============

        # Lightweight information exchange between the two boundary branches
        lr = torch.cat([tgt_span_left, tgt_span_right], dim=-1)          # (nq,bs,2d)
        lr = lr + self.span_lr_fuse(lr)
        lr = self.span_lr_fuse_norm(lr)
        tgt_span_left, tgt_span_right = lr[..., :self.d_model], lr[..., self.d_model:]

        tgt = torch.cat([tgt_span_left, tgt_span_right, tgt_class], dim=-1)
        return tgt

class TransformerDecoderLayerThin(nn.Module):
    """removed intermediate layer"""
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt2 = self.linear1(tgt2)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        activation='prelu',
        use_bnd_priori=args.use_bnd_priori,
        use_sal_priori=args.use_sal_priori,
        sal_priori_film=args.sal_priori_film,
        saliency_in_transformer=args.saliency_in_transformer,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "prelu":
        return nn.PReLU()
    if activation == "selu":
        return F.selu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
