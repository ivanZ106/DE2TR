import torch
from torch.utils.data import Dataset, BatchSampler, Sampler, ConcatDataset
import bisect
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import random
import logging
from os.path import join, exists
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.basic_utils import load_jsonl, l2_normalize_np_array
from utils.tensor_utils import pad_sequences_1d
from de2tr.span_utils import span_xx_to_cxw

logger = logging.getLogger(__name__)

class StartEndDataset(Dataset):
    Q_FEAT_TYPES = ["pooler_output", "last_hidden_state"]
    """One line in data loaded from data_path."
    {
      "qid": 7803,
      "query": "Man in gray top walks from outside to inside.",
      "duration": 150,
      "vid": "RoripwjYFp8_360.0_510.0",
      "relevant_clip_ids": [13, 14, 15, 16, 17],
      "relevant_windows": [[26, 36]]
    }
    """

    def __init__(self, dset_name, data_path, v_feat_dirs, q_feat_dir,
                 q_feat_type="last_hidden_state",
                 max_q_l=32, max_v_l=75, data_ratio=1.0, ctx_mode="video",
                 normalize_v=True, normalize_t=True, load_labels=True,
                 clip_len=2, max_windows=5, span_loss_type="l1", txt_drop_ratio=0,
                 dset_domain=None):
        self.dset_name = dset_name
        self.data_path = data_path
        self.data_ratio = data_ratio
        self.v_feat_dirs = v_feat_dirs \
            if isinstance(v_feat_dirs, list) else [v_feat_dirs]
        self.q_feat_dir = q_feat_dir
        self.q_feat_type = q_feat_type
        self.max_q_l = max_q_l
        self.max_v_l = max_v_l
        self.ctx_mode = ctx_mode
        self.use_tef = "tef" in ctx_mode
        self.use_video = "video" in ctx_mode
        self.normalize_t = normalize_t
        self.normalize_v = normalize_v
        self.load_labels = load_labels
        self.clip_len = clip_len
        self.max_windows = max_windows  # maximum number of windows to use as labels
        self.span_loss_type = span_loss_type
        self.txt_drop_ratio = txt_drop_ratio
        if "val" in data_path or "test" in data_path:
            assert txt_drop_ratio == 0
        # checks
        assert q_feat_type in self.Q_FEAT_TYPES

        # data
        self.data = self.load_data()
        
        # load specific domain data for tvsum dataset
        if self.dset_name == 'tvsum':
            target_domain = dset_domain
            assert target_domain in ["BK", "BT", "DS", "FM", "GA", "MS", "PK", "PR", "VT", "VU"]

            new_data = []
            for d in self.data:
                if target_domain == d['domain']:
                    new_data.append(d)
            self.data = new_data
        

    def load_data(self):
        datalist = load_jsonl(self.data_path)
        if self.data_ratio != 1:
            n_examples = int(len(datalist) * self.data_ratio)
            datalist = datalist[:n_examples]
            logger.info("Using {}% of the data: {} examples"
                        .format(self.data_ratio * 100, n_examples))
        return datalist

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        meta = self.data[index]

        model_inputs = dict()
        model_inputs["query_feat"] = self._get_query_feat_by_qid(meta["qid"])  # (Dq, ) or (Lq, Dq)

        if self.use_video:
            model_inputs["video_feat"] = self._get_video_feat_by_vid(meta["vid"])  # (Lv, Dv)
            ctx_l = len(model_inputs["video_feat"])
        else:
            ctx_l = self.max_v_l

        if self.use_tef:
            tef_st = torch.arange(0, ctx_l, 1.0) / ctx_l
            tef_ed = tef_st + 1.0 / ctx_l
            tef = torch.stack([tef_st, tef_ed], dim=1)  # (Lv, 2)

            if self.use_video:
                model_inputs["video_feat"] = torch.cat(
                    [model_inputs["video_feat"], tef], dim=1)  # (Lv, Dv+2)
            else:
                model_inputs["video_feat"] = tef

        if self.load_labels:
            if self.dset_name == 'tvsum': 

                max_l = ctx_l//2 

                meta_label = meta['label']
                agg_scores = np.sum(meta_label - np.ones_like(meta_label), axis=-1)[:ctx_l] # start from 1, so minus 1
                sort_indices = np.argsort(agg_scores)  # increasing
                pos_idx = torch.tensor(sort_indices[max_l:])
                
                mask = torch.zeros_like(torch.ones(ctx_l))

                if pos_idx.max() >= len(mask):
                    new_mask = torch.zeros_like(torch.ones(pos_idx.max()+1 ))
                    new_mask[pos_idx] = 1
                    new_mask[:len(mask)] = mask
                    mask = new_mask
                else:
                    mask[pos_idx] = 1

                model_inputs["pos_mask"] = mask 
                
                
                neg_idx = torch.tensor(list(set(range(ctx_l)) - set(pos_idx)))
                

                pad_tensor = torch.ones(ctx_l) * -2
                pad_tensor[:len(pos_idx)] = pos_idx
                model_inputs["pos_idx"] = pad_tensor

                pad_tensor = torch.ones(ctx_l) * -2
                pad_tensor[:len(neg_idx)] = neg_idx
                model_inputs["neg_idx"] = pad_tensor

                model_inputs["span_labels"] = torch.tensor([[0., 0.]])
                meta_label = meta['label']
                model_inputs["saliency_pos_labels"], model_inputs["saliency_neg_labels"], model_inputs["saliency_all_labels"] = \
                            self.get_saliency_labels_all_tvsum(meta_label, ctx_l)
            else:

                pos_idx = torch.tensor(meta['relevant_clip_ids'])
                mask = torch.zeros_like(torch.ones(ctx_l))

                if pos_idx.max() >= len(mask):
                    new_mask = torch.zeros_like(torch.ones(pos_idx.max()+1 ))
                    new_mask[pos_idx] = 1
                    new_mask[:len(mask)] = mask
                    mask = new_mask
                else:
                    mask[pos_idx] = 1

                model_inputs["pos_mask"] = mask 


                model_inputs["span_labels"], model_inputs['mask_labels'], y_s, y_e = self.get_span_mask_labels(
                    meta["relevant_windows"],
                    ctx_l,
                    sigma=1.5,
                    video_feat=model_inputs["video_feat"],
                    dynamic_sigma=False, #True,
                    sigma_min=0.8,
                    sigma_max=2.0,
                    r_ratio=0.25,
                    r_min=1,
                    r_max=8,
                )
                model_inputs["boundary_start_labels"] = y_s
                model_inputs["boundary_end_labels"] = y_e
                model_inputs["boundary_start_labels_all"] = y_s.max(axis=0).astype(np.float32)  # (L,)
                model_inputs["boundary_end_labels_all"] = y_e.max(axis=0).astype(np.float32)    # (L,)
                # ...existing code...
                if "subs_train" not in self.data_path:
                    model_inputs["saliency_pos_labels"], model_inputs["saliency_neg_labels"], model_inputs["saliency_all_labels"] = \
                        self.get_saliency_labels_all(meta["relevant_clip_ids"], meta["saliency_scores"], ctx_l)
                else:
                    model_inputs["saliency_pos_labels"], model_inputs["saliency_neg_labels"], model_inputs["saliency_all_labels"] = \
                        self.get_saliency_labels_sub_as_query(meta["relevant_windows"][0], ctx_l)  # only one gt

        return dict(meta=meta, model_inputs=model_inputs)

    def _gaussian_1d(self, L, center, sigma):
        t = np.arange(L, dtype=np.float32)
        return np.exp(-0.5 * ((t - float(center)) / float(sigma)) ** 2)


    def get_saliency_labels_sub_as_query(self, gt_window, ctx_l, max_n=2):
        gt_st = int(gt_window[0] / self.clip_len)
        gt_ed = max(0, min(int(gt_window[1] / self.clip_len), ctx_l) - 1)
        if gt_st > gt_ed:
            gt_st = gt_ed

        if gt_st != gt_ed:
            pos_clip_indices = random.sample(range(gt_st, gt_ed+1), k=max_n)
        else:
            pos_clip_indices = [gt_st, gt_st]

        neg_pool = list(range(0, gt_st)) + list(range(gt_ed+1, ctx_l))
        neg_clip_indices = random.sample(neg_pool, k=max_n)
        
        score_array = np.zeros(ctx_l)
        score_array[gt_st:gt_ed+1] = 1

        return pos_clip_indices, neg_clip_indices, score_array
        

    def get_saliency_labels(self, rel_clip_ids, scores, ctx_l, max_n=1, add_easy_negative=True):
        """Sum the scores from the three annotations, then take the two clips with the
        maximum scores as positive, and two with the minimum scores as negative.
        Args:
            rel_clip_ids: list(int), list of relevant clip ids
            scores: list([anno1_score, anno2_score, anno3_score]),
            ctx_l: int
            max_n: int, #clips to use as positive and negative, for easy and hard negative, respectively.
            add_easy_negative: bool, if True, sample eay negative outside the relevant_clip_ids.
        """
        # indices inside rel_clip_ids
        scores = np.array(scores)  # (#rel_clips, 3)
        agg_scores = np.sum(scores, 1)  # (#rel_clips, )
        sort_indices = np.argsort(agg_scores)  # increasing

        # indices in the whole video
        # the min(_, ctx_l-1) here is incorrect, but should not cause
        # much troubles since this should be rarely used.
        hard_pos_clip_indices = [min(rel_clip_ids[idx], ctx_l-1) for idx in sort_indices[-max_n:]]
        hard_neg_clip_indices = [min(rel_clip_ids[idx], ctx_l-1) for idx in sort_indices[:max_n]]
        easy_pos_clip_indices = []
        easy_neg_clip_indices = []
        if add_easy_negative:
            easy_neg_pool = list(set(range(ctx_l)) - set(rel_clip_ids))
            if len(easy_neg_pool) >= max_n:
                easy_pos_clip_indices = random.sample(rel_clip_ids, k=max_n)
                easy_neg_clip_indices = random.sample(easy_neg_pool, k=max_n)
            else:  # copy the hard ones
                easy_pos_clip_indices = hard_pos_clip_indices
                easy_neg_clip_indices = hard_neg_clip_indices

        pos_clip_indices = hard_pos_clip_indices + easy_pos_clip_indices
        neg_clip_indices = hard_neg_clip_indices + easy_neg_clip_indices
        return pos_clip_indices, neg_clip_indices

    def get_saliency_labels_all(self, rel_clip_ids, scores, ctx_l, max_n=1, add_easy_negative=True):
        """Sum the scores from the three annotations, then take the two clips with the
        maximum scores as positive, and two with the minimum scores as negative.
        Args:
            rel_clip_ids: list(int), list of relevant clip ids
            scores: list([anno1_score, anno2_score, anno3_score]),
            ctx_l: int
            max_n: int, #clips to use as positive and negative, for easy and hard negative, respectively.
            add_easy_negative: bool, if True, sample eay negative outside the relevant_clip_ids.
        """
        # indices inside rel_clip_ids
        scores = np.array(scores)  # (#rel_clips, 3)
        agg_scores = np.sum(scores, 1)  # (#rel_clips, )
        sort_indices = np.argsort(agg_scores)  # increasing

        score_array = np.zeros(ctx_l)
        for idx in range(len(rel_clip_ids)):
            if rel_clip_ids[idx] >= ctx_l:
                score_array_new = np.zeros(ctx_l + 1)
                score_array_new[:ctx_l] = score_array
                score_array = score_array_new
            score_array[rel_clip_ids[idx]] = agg_scores[idx]

        # indices in the whole video
        # the min(_, ctx_l-1) here is incorrect, but should not cause
        # much troubles since this should be rarely used.
        hard_pos_clip_indices = [min(rel_clip_ids[idx], ctx_l-1) for idx in sort_indices[-max_n:]]
        hard_neg_clip_indices = [min(rel_clip_ids[idx], ctx_l-1) for idx in sort_indices[:max_n]]
        easy_pos_clip_indices = []
        easy_neg_clip_indices = []
        if add_easy_negative:
            easy_neg_pool = list(set(range(ctx_l)) - set(rel_clip_ids))
            if len(easy_neg_pool) >= max_n:
                easy_pos_clip_indices = random.sample(rel_clip_ids, k=max_n)
                easy_neg_clip_indices = random.sample(easy_neg_pool, k=max_n)
            else:  # copy the hard ones
                easy_pos_clip_indices = hard_pos_clip_indices
                easy_neg_clip_indices = hard_neg_clip_indices

        pos_clip_indices = hard_pos_clip_indices + easy_pos_clip_indices
        neg_clip_indices = hard_neg_clip_indices + easy_neg_clip_indices
        return pos_clip_indices, neg_clip_indices, score_array

    def get_saliency_labels_all_tvsum(self, labels, ctx_l, max_n=1, add_easy_negative=False):
        
        agg_scores = np.sum(labels - np.ones_like(labels), axis=-1)[:ctx_l] # start from 1, so minus 1
        score_array = agg_scores / 80 * 12
        sort_indices = np.argsort(agg_scores)  # increasing

        hard_pos_clip_indices = [min(idx, ctx_l-1) for idx in sort_indices[-max_n:]]
        hard_neg_clip_indices = [min(idx, ctx_l-1) for idx in sort_indices[:max_n]]
        easy_pos_clip_indices = []
        easy_neg_clip_indices = []
        if add_easy_negative:
            easy_neg_pool = list(set(range(ctx_l)))
            if len(easy_neg_pool) >= max_n:
                easy_pos_clip_indices = random.sample(rel_clip_ids, k=max_n)
                easy_neg_clip_indices = random.sample(easy_neg_pool, k=max_n)
            else:  # copy the hard ones
                easy_pos_clip_indices = hard_pos_clip_indices
                easy_neg_clip_indices = hard_neg_clip_indices

        pos_clip_indices = hard_pos_clip_indices + easy_pos_clip_indices
        neg_clip_indices = hard_neg_clip_indices + easy_neg_clip_indices

        return pos_clip_indices, neg_clip_indices, score_array

    def _boundary_sigma_from_contrast(
        self,
        video_feat: torch.Tensor,
        st: int,
        ed: int,
        side: str,
        r_ratio: float = 0.25,
        r_min: int = 1,
        r_max: int = 8,
        sigma_min: float = 0.8,
        sigma_max: float = 3.0,
        eps: float = 1e-8,
    ) -> float:
        """
        Compute a dynamic sigma for boundary soft labels based on in/out contrast near boundary.

        video_feat: (L, D) torch tensor on CPU
        st, ed: inclusive clip indices
        side: 'start' or 'end'
        """
        assert side in ["start", "end"]
        L = int(video_feat.shape[0])
        if L <= 1:
            return float((sigma_min + sigma_max) * 0.5)

        # exclude TEF dims if used (last 2 dims are normalized time features)
        feat = video_feat
        if getattr(self, "use_tef", False) and feat.shape[1] > 2:
            feat = feat[:, :-2]

        # guard
        st = int(max(0, min(st, L - 1)))
        ed = int(max(0, min(ed, L - 1)))
        if st > ed:
            st = ed

        span_len = ed - st + 1
        r = int(round(r_ratio * span_len))
        r = max(r_min, min(r_max, r))

        # define inside/outside neighborhoods
        if side == "start":
            in_l, in_r = st, min(ed, st + r)
            out_l, out_r = max(0, st - r), st - 1
            # fallback if no left outside: use right outside after end
            if out_r < out_l:
                out_l, out_r = ed + 1, min(L - 1, ed + r)
        else:  # end
            in_l, in_r = max(st, ed - r), ed
            out_l, out_r = ed + 1, min(L - 1, ed + r)
            # fallback if no right outside: use left outside before start
            if out_r < out_l:
                out_l, out_r = max(0, st - r), st - 1

        # if still invalid, fall back to mid sigma
        if in_r < in_l or out_r < out_l:
            return float((sigma_min + sigma_max) * 0.5)

        inside = feat[in_l:in_r + 1].mean(dim=0)
        outside = feat[out_l:out_r + 1].mean(dim=0)

        # hardness in [0,1], larger means more separable boundary -> sharper label
        cos = F.cosine_similarity(inside, outside, dim=0, eps=eps).clamp(-1.0, 1.0)
        hardness = ((1.0 - cos) / 2.0).clamp(0.0, 1.0)

        sigma = sigma_max - float(hardness.item()) * (sigma_max - sigma_min)
        return float(max(sigma_min, min(sigma_max, sigma)))

    def get_span_mask_labels(
        self,
        windows,
        ctx_l,
        sigma=1.5,
        video_feat: torch.Tensor = None,
        dynamic_sigma: bool = False,
        sigma_min: float = 0.8,
        sigma_max: float = 3.0,
        r_ratio: float = 0.25,
        r_min: int = 1,
        r_max: int = 8,
    ):
        """
        windows: list([st, ed]) in seconds. E.g. [[26, 36]], corresponding st_ed clip_indices [[13, 17]] (inclusive)
        returns:
          windows: Tensor (#windows, 2) in cxw (normalized) or ce labels
          masks: Tensor (#windows, ctx_l)
          y_s, y_e: np.ndarray (#windows, ctx_l) soft boundary labels
        """
        if len(windows) > self.max_windows:
            random.shuffle(windows)
            windows = windows[:self.max_windows]

        num_windows = len(windows)
        masks = torch.zeros(num_windows, ctx_l) if num_windows < self.max_windows else torch.zeros(self.max_windows, ctx_l)

        y_s = np.zeros((num_windows, ctx_l), dtype=np.float32)
        y_e = np.zeros((num_windows, ctx_l), dtype=np.float32)

        for w_idx, w in enumerate(windows):
            if w_idx >= self.max_windows:
                break

            st = int(w[0] / self.clip_len)
            ed = max(0, min(int(w[1] / self.clip_len), ctx_l) - 1)  # inclusive
            if st > ed:
                st = ed

            # mask labels (inclusive)
            masks[w_idx, st:ed + 1] = 1

            # dynamic sigma per boundary (optional)
            if dynamic_sigma and (video_feat is not None):
                sig_s = self._boundary_sigma_from_contrast(
                    video_feat=video_feat, st=st, ed=ed, side="start",
                    r_ratio=r_ratio, r_min=r_min, r_max=r_max,
                    sigma_min=sigma_min, sigma_max=sigma_max
                )
                sig_e = self._boundary_sigma_from_contrast(
                    video_feat=video_feat, st=st, ed=ed, side="end",
                    r_ratio=r_ratio, r_min=r_min, r_max=r_max,
                    sigma_min=sigma_min, sigma_max=sigma_max
                )
            else:
                sig_s = float(sigma)
                sig_e = float(sigma)

            y_s[w_idx] = self._gaussian_1d(ctx_l, st, sig_s)
            y_e[w_idx] = self._gaussian_1d(ctx_l, ed, sig_e)

        if self.span_loss_type == "l1":
            windows = torch.Tensor(windows) / (ctx_l * self.clip_len)  # normalized windows in xx
            windows = span_xx_to_cxw(windows)  # normalized windows in cxw
        elif self.span_loss_type == "ce":
            windows = torch.Tensor([
                [int(w[0] / self.clip_len), min(int(w[1] / self.clip_len), ctx_l) - 1]
                for w in windows]).long()  # inclusive
        else:
            raise NotImplementedError

        return windows, masks, y_s, y_e


    def _get_query_feat_by_qid(self, qid):
        if self.dset_name == 'tvsum':
            q_feat = np.load(join(self.q_feat_dir, "{}.npz".format(qid))) # 'token', 'text'
            return torch.from_numpy(q_feat['token'])
        else:
            q_feat_path = join(self.q_feat_dir, f"{qid}.npy")
            q_feat = np.load(q_feat_path).astype(np.float32)
            if self.q_feat_type == "last_hidden_state":
                q_feat = q_feat[:self.max_q_l]
            if self.normalize_t:
                q_feat = l2_normalize_np_array(q_feat)
            if self.txt_drop_ratio > 0:
                q_feat = self.random_drop_rows(q_feat)
            l = len(q_feat)
            q_feat = q_feat.reshape(l, -1)
        return torch.from_numpy(q_feat)  # (D, ) or (Lq, D)

    def random_drop_rows(self, embeddings):
        """randomly mask num_drop rows in embeddings to be zero.
        Args:
            embeddings: np.ndarray (L, D)
        """
        num_drop_rows = round(len(embeddings) * self.txt_drop_ratio)
        if num_drop_rows > 0:
            row_indices = np.random.choice(
                len(embeddings), size=num_drop_rows, replace=False)
            embeddings[row_indices] = 0
        return embeddings


    def _get_video_feat_by_vid(self, vid):
        if self.dset_name == 'tvsum':
            v_feat_list = []
            for _feat_dir in self.v_feat_dirs:
                _feat_path = join(_feat_dir, f"{vid}_rgb.npy")
                _feat_rgb = np.load(_feat_path)[:self.max_v_l].astype(np.float32)

                _feat_path = join(_feat_dir, f"{vid}_opt.npy")
                _feat_opt = np.load(_feat_path)[:self.max_v_l].astype(np.float32)
                
                _feat = np.concatenate([_feat_rgb, _feat_opt], axis=-1)
                if self.normalize_v:
                    _feat = l2_normalize_np_array(_feat)
                v_feat_list.append(_feat)
            # some features are slightly longer than the others
            min_len = min([len(e) for e in v_feat_list])
            v_feat_list = [e[:min_len] for e in v_feat_list]
            v_feat = np.concatenate(v_feat_list, axis=1)

        else:
            v_feat_list = []
            for _feat_dir in self.v_feat_dirs:
                if 'slowfast' in _feat_dir:
                    _feat_path = _feat_dir+"/"+f"{vid}.npz"
                    _feat = np.load(_feat_path)["features"][:self.max_v_l].astype(np.float32)
                    if self.normalize_v:
                        _feat = l2_normalize_np_array(_feat)
                    v_feat_list.append(_feat)
                if 'clip' in _feat_dir:
                    _feat_path = _feat_dir+"/"+f"{vid}.npy"
                    _feat = np.load(_feat_path).astype(np.float32) # L, K, D
                    if self.normalize_v:
                        _feat = l2_normalize_np_array(_feat)
                    l = len(_feat)
                    _feat = _feat.reshape(l, -1)
                    v_feat_list.append(_feat)
            # some features are slightly longer than the others
            min_len = min([len(e) for e in v_feat_list])
            v_feat_list = [e[:min_len] for e in v_feat_list]
            v_feat = np.concatenate(v_feat_list, axis=1)
        return torch.from_numpy(v_feat) # (Lv, D)


class SyntheticDataset(StartEndDataset):
    """
    Synthetic dataset for validation:
    - Keep GT window positions and labels unchanged.
    - Replace non-GT context features with those from another (unrelated) video at feature level.
    - Video length stays the same as the original sample.
    """

    def __init__(
        self,
        *args,
        synth_prob: float = 1.0,
        ensure_diff_vid: bool = True,
        pad_mode: str = "repeat",  # "repeat" or "zero"
        seed: int = 2018,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.synth_prob = float(synth_prob)
        self.ensure_diff_vid = bool(ensure_diff_vid)
        self.pad_mode = str(pad_mode)
        self._rng = random.Random(int(seed))

    def _resize_video_feat(self, feat: torch.Tensor, target_len: int) -> torch.Tensor:
        """feat: (L,D) -> (target_len,D) by trunc/pad."""
        L, D = feat.shape
        if L == target_len:
            return feat
        if L > target_len:
            return feat[:target_len]
        # pad
        pad_len = target_len - L
        if self.pad_mode == "zero":
            pad = torch.zeros(pad_len, D, dtype=feat.dtype)
        else:  # repeat last
            last = feat[-1:].clone()
            pad = last.repeat(pad_len, 1)
        return torch.cat([feat, pad], dim=0)

    def _add_tef_if_needed(self, video_feat: torch.Tensor) -> torch.Tensor:
        """Match StartEndDataset.__getitem__ TEF behavior."""
        if not self.use_tef:
            return video_feat
        ctx_l = int(video_feat.shape[0])
        tef_st = torch.arange(0, ctx_l, 1.0) / ctx_l
        tef_ed = tef_st + 1.0 / ctx_l
        tef = torch.stack([tef_st, tef_ed], dim=1)  # (L,2)
        return torch.cat([video_feat, tef], dim=1)

    def _gt_clip_ranges(self, meta, ctx_l: int):
        """Return list of (st, ed) inclusive clip index ranges from meta['relevant_windows']."""
        if "relevant_windows" not in meta or meta["relevant_windows"] is None:
            return []
        ranges = []
        for (st_sec, ed_sec) in meta["relevant_windows"]:
            st = int(st_sec / self.clip_len)
            ed = max(0, min(int(ed_sec / self.clip_len), ctx_l) - 1)  # inclusive
            if st > ed:
                st = ed
            st = max(0, min(st, ctx_l - 1))
            ed = max(0, min(ed, ctx_l - 1))
            ranges.append((st, ed))
        return ranges

    def _sample_donor_index(self, index: int, cur_vid: str) -> int:
        """Sample a different index (prefer different vid)."""
        if len(self.data) <= 1:
            return index
        for _ in range(20):
            j = self._rng.randrange(0, len(self.data))
            if j == index:
                continue
            if (not self.ensure_diff_vid) or (self.data[j].get("vid", None) != cur_vid):
                return j
        # fallback
        return (index + 1) % len(self.data)

    def __getitem__(self, index):
        # base sample (keeps labels as-is)
        base = super().__getitem__(index)

        # only synthesize when video is used
        if (not self.use_video) or ("video_feat" not in base["model_inputs"]):
            return base

        if self.synth_prob < 1.0 and self._rng.random() > self.synth_prob:
            return base

        meta = base["meta"]
        cur_vid = meta.get("vid", None)

        # original feature (already has TEF if enabled by base __getitem__)
        feat_a = base["model_inputs"]["video_feat"]  # (La, D)
        ctx_l = int(feat_a.shape[0])

        gt_ranges = self._gt_clip_ranges(meta, ctx_l)
        if len(gt_ranges) == 0:
            return base  # no GT windows => skip

        # donor sample/video
        donor_idx = self._sample_donor_index(index, cur_vid)
        donor_meta = self.data[donor_idx]
        donor_vid = donor_meta.get("vid", None)

        # load donor raw video features (no TEF yet), then resize to ctx_l, then add TEF if needed
        donor_raw = self._get_video_feat_by_vid(donor_vid)  # (Lb, Dv)
        donor_raw = donor_raw[: self.max_v_l]  # safety
        donor_raw = self._resize_video_feat(donor_raw, ctx_l)
        feat_b = self._add_tef_if_needed(donor_raw)  # (ctx_l, D)

        # align dims
        if feat_b.shape[1] != feat_a.shape[1]:
            raise ValueError(f"SyntheticDataset: feature dim mismatch A={feat_a.shape}, B={feat_b.shape}")

        # splice: keep donor context, but overwrite GT segments with original GT features
        D = feat_a.shape[1]
        if self.use_tef:
            visual_dim = D - 2
            # keep TEF consistent with original timeline
            feat_b[:, visual_dim:] = feat_a[:, visual_dim:]
        else:
            visual_dim = D

        for (st, ed) in gt_ranges:
            feat_b[st:ed + 1, :visual_dim] = feat_a[st:ed + 1, :visual_dim]

        # write back synthesized feature; labels remain unchanged
        base["model_inputs"]["video_feat"] = feat_b
        # optional debug info
        base["meta"]["synthetic_from_vid"] = donor_vid
        return base

        
def start_end_collate(batch):
    batch_meta = [e["meta"] for e in batch]  # seems no need to collate ?

    model_inputs_keys = batch[0]["model_inputs"].keys()
    batched_data = dict()

    max_video_len = 0
    if "video_feat" in batch[0]["model_inputs"]:
        max_video_len = max([e["model_inputs"]["video_feat"].shape[0] for e in batch])

    for k in model_inputs_keys:
        if k == "span_labels":
            batched_data[k] = [dict(spans=e["model_inputs"]["span_labels"]) for e in batch]
            continue
        if k == "mask_labels":
            padded_masks = []
            for e in batch:
                mask = e["model_inputs"]["mask_labels"] # shape: (num_windows, cur_v_l)
                cur_v_l = mask.shape[1]
                
                # pad the mask on the right if it is shorter than the longest in the batch
                if max_video_len > 0 and cur_v_l < max_video_len:
                    mask = F.pad(mask, (0, max_video_len - cur_v_l), "constant", 0)
                
                padded_masks.append(mask)
            batched_data[k] = padded_masks
            continue  

        if k in ["boundary_start_labels", "boundary_end_labels"]:
            padded_bnd = []
            for e in batch:
                bnd = torch.tensor(e["model_inputs"][k], dtype=torch.float32) \
                    if not torch.is_tensor(e["model_inputs"][k]) else e["model_inputs"][k].float()
                cur_v_l = bnd.shape[1]
                if max_video_len > 0 and cur_v_l < max_video_len:
                    bnd = F.pad(bnd, (0, max_video_len - cur_v_l), "constant", 0)
                padded_bnd.append(bnd)
            batched_data[k] = padded_bnd
            continue

        if k in ["saliency_pos_labels", "saliency_neg_labels"]:
            batched_data[k] = torch.LongTensor([e["model_inputs"][k] for e in batch])
            continue
        if k == "saliency_all_labels":
            pad_data, mask_data = pad_sequences_1d([e["model_inputs"][k] for e in batch], dtype=np.float32, fixed_length=None)
            batched_data[k] = torch.tensor(pad_data, dtype=torch.float32)
            continue
        if k in ["boundary_start_labels_all", "boundary_end_labels_all"]:
            pad_data, _ = pad_sequences_1d([e["model_inputs"][k] for e in batch], dtype=np.float32, fixed_length=None)
            batched_data[k] = torch.tensor(pad_data, dtype=torch.float32)
            continue

        batched_data[k] = pad_sequences_1d(
            [e["model_inputs"][k] for e in batch], dtype=torch.float32, fixed_length=None)
    return batch_meta, batched_data


def prepare_batch_inputs(batched_model_inputs, device, non_blocking=False):
    model_inputs = dict(
        src_txt=batched_model_inputs["query_feat"][0].to(device, non_blocking=non_blocking),
        src_txt_mask=batched_model_inputs["query_feat"][1].to(device, non_blocking=non_blocking),
        src_vid=batched_model_inputs["video_feat"][0].to(device, non_blocking=non_blocking),
        src_vid_mask=batched_model_inputs["video_feat"][1].to(device, non_blocking=non_blocking),
    )
    targets = {}
    if "span_labels" in batched_model_inputs:
        targets["span_labels"] = [
            dict(spans=e["spans"].to(device, non_blocking=non_blocking))
            for e in batched_model_inputs["span_labels"]
        ]
    if "mask_labels" in batched_model_inputs:
        targets["mask_labels"] = [
            m.to(device, non_blocking=non_blocking) 
            for m in batched_model_inputs["mask_labels"]
        ]
    if "saliency_pos_labels" in batched_model_inputs:
        for name in ["saliency_pos_labels", "saliency_neg_labels"]:
            targets[name] = batched_model_inputs[name].to(device, non_blocking=non_blocking)

    if "saliency_all_labels" in batched_model_inputs:
        targets["saliency_all_labels"] = batched_model_inputs["saliency_all_labels"].to(device, non_blocking=non_blocking)

    if "boundary_start_labels_all" in batched_model_inputs:
        targets["boundary_start_labels_all"] = batched_model_inputs["boundary_start_labels_all"].to(device, non_blocking=non_blocking)
        targets["boundary_end_labels_all"] = batched_model_inputs["boundary_end_labels_all"].to(device, non_blocking=non_blocking)
        

    if "boundary_start_labels" in batched_model_inputs:
        targets["boundary_start_labels"] = [
            m.to(device, non_blocking=non_blocking)
            for m in batched_model_inputs["boundary_start_labels"]
        ]
        targets["boundary_end_labels"] = [
            m.to(device, non_blocking=non_blocking)
            for m in batched_model_inputs["boundary_end_labels"]
        ]

    if "pos_mask" in batched_model_inputs:
        targets['src_pos_mask']=batched_model_inputs["pos_mask"][0].to(device, non_blocking=non_blocking)
    targets = None if len(targets) == 0 else targets
    return model_inputs, targets
