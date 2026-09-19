dset_name=hl
ctx_mode=video_tef
v_feat_types=slowfast_clip
t_feat_type=clip 
results_root=results
exp_id=test_data

######## data paths
train_path=data/highlight_train_release.jsonl # highlight_train_release_query_clip_sim_all.jsonl
eval_path=data/highlight_val_release.jsonl
eval_split_name=val

######## setup video+text features
feat_root=../features/qvhighlight

# video features
v_feat_dim=0
v_feat_dirs=()
if [[ ${v_feat_types} == *"slowfast"* ]]; then
  v_feat_dirs+=(${feat_root}/slowfast_features)
  (( v_feat_dim += 2304 ))  # double brackets for arithmetic op, no need to use ${v_feat_dim}
fi
if [[ ${v_feat_types} == *"clip"* ]]; then
  v_feat_dirs+=(${feat_root}/clip_b32_vid_k4) # clip_features
  (( v_feat_dim += 3072 ))  # 512
fi

# text features
if [[ ${t_feat_type} == "clip" ]]; then
  t_feat_dir=${feat_root}/clip_b32_txt_k4/  # clip_text_features
  t_feat_dim=2048 # 512
else
  echo "Wrong arg for t_feat_type."
  exit 1
fi

#### training
bsz=32
num_workers=8
lr_drop=80
lr=0.0001
n_epoch=200
lw_saliency=1.0
seed=2018
VTC_loss_coef=0.3
CTC_loss_coef=0.5
# use_txt_pos=True
set_cost_span=1
set_cost_giou=10
set_cost_class=4
span_loss_coef=1
giou_loss_coef=10
label_loss_coef=4


PYTHONPATH=$PYTHONPATH:. python de2tr/train.py \
--seed $seed \
--set_cost_span $set_cost_span \
--set_cost_giou $set_cost_giou \
--set_cost_class $set_cost_class \
--span_loss_coef $span_loss_coef \
--giou_loss_coef $giou_loss_coef \
--label_loss_coef $label_loss_coef \
--VTC_loss_coef $VTC_loss_coef \
--CTC_loss_coef $CTC_loss_coef \
--dset_name ${dset_name} \
--ctx_mode ${ctx_mode} \
--train_path ${train_path} \
--eval_path ${eval_path} \
--eval_split_name ${eval_split_name} \
--v_feat_dirs ${v_feat_dirs[@]} \
--v_feat_dim ${v_feat_dim} \
--t_feat_dir ${t_feat_dir} \
--t_feat_dim ${t_feat_dim} \
--bsz ${bsz} \
--results_root ${results_root} \
--exp_id ${exp_id} \
--lr ${lr} \
--n_epoch ${n_epoch} \
--lw_saliency ${lw_saliency} \
--enc_layers 4 \
--dec_layers 4 \
--lr_drop ${lr_drop} \
--num_workers ${num_workers} \
${@:1}
