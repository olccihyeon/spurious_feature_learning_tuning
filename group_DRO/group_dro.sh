WATERBIRDS_DIR='../cub/data/waterbird_complete95_forest2water2'

CUDA_VISIBLE_DEVICES=0 python3 run_expt.py -s confounder -d CUB --model imagenet_resnet50_pretrained \
	-t target -c confounder \
	--root_dir $WATERBIRDS_DIR --robust --save_best --save_last --save_step 200 \
	--batch_size 100 --n_epochs 20 --gamma 0.1 --augment_data --lr 0.001 \
	--weight_decay 0.001 --generalization_adjustment 0 --seed 1 \
	--log_dir 'logs/gdro' --dfr_data --dfr_model