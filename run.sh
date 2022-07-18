export CUDA_VISIBLE_DEVICES=0,1,2,3


LR=5e-5
EPOCH=25
BS=16
POOL=last-avg
CKPT_DIR="ckpt_ernie_gram_bs${BS}_lr${LR}_epoch${EPOCH}"



python -m  torch.distributed.launch --nproc_per_node=4 train.py \
	--batch_size=${BS} \
	--save_dir=${CKPT_DIR} \
	--lr=${LR} \
	--epochs=${EPOCH} \
	--pooling=${POOL} \
	> log_ernie_gram_bs${BS}_lr${LR}_epoch${EPOCH}_pool${POOL} 2>&1
