export CUDA_VISIBLE_DEVICES=0,1,2,3


LR=5e-6
EPOCH=20
BS=32
CKPT_DIR="ckpt_bs${BS}_lr${LR}_epoch${EPOCH}"



python -m  torch.distributed.launch --nproc_per_node=4 train.py \
	--batch_size=${BS} \
	--save_dir=${CKPT_DIR} \
	--lr=${LR} \
	> log_bs${BS}_lr${LR}_epoch${EPOCH 2>&1
