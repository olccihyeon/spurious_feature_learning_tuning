WATERBIRDS_DIR='cub/data/waterbird_complete95_forest2water2'
fgWATERBIRDS_DIR='cub/data/waterbirds_birds_places_generate/birds'
bgWATERBIRDS_DIR='cub/data/waterbirds_birds_places_generate/places'


CUDA_VISIBLE_DEVICES=0 python3 train_supervised.py --output_dir 'logs/waterbirds/models' \
	--num_epochs 100 --eval_freq 1 --save_freq 100 --seed 0 \
	--weight_decay 1e-4 --batch_size 32 --init_lr 3e-3 \
	--scheduler cosine_lr_scheduler --data_dir $WATERBIRDS_DIR \
	--data_transform AugWaterbirdsCelebATransform \
	--dataset SpuriousCorrelationDataset --model imagenet_resnet18_pretrained &


CUDA_VISIBLE_DEVICES=1 python3 train_supervised.py --output_dir 'logs/waterbirds/models' \
	--num_epochs 100 --eval_freq 1 --save_freq 100 --seed 1 \
	--weight_decay 1e-4 --batch_size 32 --init_lr 3e-3 \
	--scheduler cosine_lr_scheduler --data_dir $WATERBIRDS_DIR \
	--data_transform AugWaterbirdsCelebATransform \
	--dataset SpuriousCorrelationDataset --model imagenet_resnet18_pretrained &


CUDA_VISIBLE_DEVICES=2 python3 train_supervised.py --output_dir 'logs/waterbirds/models' \
	--num_epochs 100 --eval_freq 1 --save_freq 100 --seed 2 \
	--weight_decay 1e-4 --batch_size 32 --init_lr 3e-3 \
	--scheduler cosine_lr_scheduler --data_dir $WATERBIRDS_DIR \
	--data_transform AugWaterbirdsCelebATransform \
	--dataset SpuriousCorrelationDataset --model imagenet_resnet18_pretrained &


CUDA_VISIBLE_DEVICES=3 python3 train_supervised.py --output_dir 'logs/waterbirds/models' \
	--num_epochs 100 --eval_freq 1 --save_freq 100 --seed 3 \
	--weight_decay 1e-4 --batch_size 32 --init_lr 3e-3 \
	--scheduler cosine_lr_scheduler --data_dir $WATERBIRDS_DIR \
	--data_transform AugWaterbirdsCelebATransform \
	--dataset SpuriousCorrelationDataset --model imagenet_resnet18_pretrained
wait

# CUDA_VISIBLE_DEVICES=0  python3 train_supervised_DFR.py --output_dir 'logs/waterbirds/erm' \
# 	--num_epochs 1 --eval_freq 1 --save_freq 100 --seed 1 --tuning 'retraining' \
# 	--weight_decay 1e-4 --batch_size=32 --init_lr 3e-3  --DFR_data 'base' \
# 	--scheduler 'cosine_lr_scheduler' --data_dir 'cub/data/waterbird_complete95_forest2water2' \
# 	--data_transform 'AugWaterbirdsCelebATransform' --resume 'outputs/waterbirds/0_final_checkpoint.pt' \
# 	--dataset 'SpuriousCorrelationDataset' --model 'imagenet_resnet50_pretrained' --DFRdata_dir 'cub/data/waterbirds_birds_places_generate/places'&

# CUDA_VISIBLE_DEVICES=1  python3 train_supervised_DFR.py --output_dir 'logs/waterbirds/erm' \
# 	--num_epochs 1 --eval_freq 1 --save_freq 100 --seed 1 --tuning 'retraining' \
# 	--weight_decay 1e-4 --batch_size=32 --init_lr 3e-3  --DFR_data 'fg_direct' \
# 	--scheduler 'cosine_lr_scheduler' --data_dir 'cub/data/waterbird_complete95_forest2water2' \
# 	--data_transform 'AugWaterbirdsCelebATransform' --resume 'outputs/waterbirds/0_final_checkpoint.pt' \
# 	--dataset 'SpuriousCorrelationDataset' --model 'imagenet_resnet50_pretrained' --DFRdata_dir 'cub/data/waterbirds_birds_places_generate/birds'&

# CUDA_VISIBLE_DEVICES=2  python3 train_supervised_DFR.py --output_dir 'logs/waterbirds/erm' \
# 	--num_epochs 1 --eval_freq 1 --save_freq 100 --seed 1 --tuning 'retraining' \
# 	--weight_decay 1e-4 --batch_size=32 --init_lr 3e-3  --DFR_data 'bg' \
# 	--scheduler 'cosine_lr_scheduler' --data_dir 'cub/data/waterbird_complete95_forest2water2' \
# 	--data_transform 'AugWaterbirdsCelebATransform' --resume 'outputs/waterbirds/0_final_checkpoint.pt' \
# 	--dataset 'SpuriousCorrelationDataset' --model 'imagenet_resnet50_pretrained' --DFRdata_dir 'cub/data/waterbirds_birds_places_generate/places'&

# CUDA_VISIBLE_DEVICES=3  python3 train_supervised_DFR.py --output_dir 'logs/waterbirds/erm' \
# 	--num_epochs 1 --eval_freq 1 --save_freq 100 --seed 1 --tuning 'retraining' \
# 	--weight_decay 1e-4 --batch_size=32 --init_lr 3e-3  --DFR_data 'fg_indirect' \
# 	--scheduler 'cosine_lr_scheduler' --data_dir 'cub/data/waterbird_complete95_forest2water2' \
# 	--data_transform 'AugWaterbirdsCelebATransform' --resume 'outputs/waterbirds/0_final_checkpoint.pt' \
# 	--dataset 'SpuriousCorrelationDataset' --model 'imagenet_resnet50_pretrained' --DFRdata_dir 'cub/data/waterbirds_birds_places_generate/places'
# wait



# CUDA_VISIBLE_DEVICES=0 python3 dfr_evaluate_spurious.py --data_dir $WATERBIRDS_DIR --data_transform 'AugWaterbirdsCelebATransform' \
#  --dataset 'SpuriousCorrelationDataset' --model 'imagenet_resnet50_pretrained' --ckpt_path 'outputs/waterbirds/0_final_checkpoint.pt' \
#  --fgdata_dir $fgWATERBIRDS_DIR --bgdata_dir $bgWATERBIRDS_DIR --DFR_data 'base'&

# CUDA_VISIBLE_DEVICES=1 python3 dfr_evaluate_spurious.py --data_dir $WATERBIRDS_DIR --data_transform 'AugWaterbirdsCelebATransform' \
#  --dataset 'SpuriousCorrelationDataset' --model 'imagenet_resnet50_pretrained' --ckpt_path 'outputs/waterbirds/1_final_checkpoint.pt' \
#  --fgdata_dir $fgWATERBIRDS_DIR --bgdata_dir $bgWATERBIRDS_DIR --DFR_data 'base'&

# CUDA_VISIBLE_DEVICES=2 python3 dfr_evaluate_spurious.py --data_dir $WATERBIRDS_DIR --data_transform 'AugWaterbirdsCelebATransform' \
#  --dataset 'SpuriousCorrelationDataset' --model 'imagenet_resnet50_pretrained' --ckpt_path 'outputs/waterbirds/2_final_checkpoint.pt' \
#  --fgdata_dir $fgWATERBIRDS_DIR --bgdata_dir $bgWATERBIRDS_DIR --DFR_data 'base'&

# CUDA_VISIBLE_DEVICES=3 python3 dfr_evaluate_spurious.py --data_dir $WATERBIRDS_DIR --data_transform 'AugWaterbirdsCelebATransform' \
#  --dataset 'SpuriousCorrelationDataset' --model 'imagenet_resnet50_pretrained' --ckpt_path 'outputs/waterbirds/3_final_checkpoint.pt' \
#  --fgdata_dir $fgWATERBIRDS_DIR --bgdata_dir $bgWATERBIRDS_DIR --DFR_data 'base'&
# wait


# CUDA_VISIBLE_DEVICES=0 python3 dfr_evaluate_spurious.py --data_dir $WATERBIRDS_DIR --data_transform 'AugWaterbirdsCelebATransform' \
#  --dataset 'SpuriousCorrelationDataset' --model 'imagenet_resnet50_pretrained' --ckpt_path 'outputs/waterbirds/0_final_checkpoint.pt' \
#  --fgdata_dir $fgWATERBIRDS_DIR --bgdata_dir $bgWATERBIRDS_DIR --DFR_data 'direct_fg'&

# CUDA_VISIBLE_DEVICES=1 python3 dfr_evaluate_spurious.py --data_dir $WATERBIRDS_DIR --data_transform 'AugWaterbirdsCelebATransform' \
#  --dataset 'SpuriousCorrelationDataset' --model 'imagenet_resnet50_pretrained' --ckpt_path 'outputs/waterbirds/1_final_checkpoint.pt' \
#  --fgdata_dir $fgWATERBIRDS_DIR --bgdata_dir $bgWATERBIRDS_DIR --DFR_data 'direct_fg'&

# CUDA_VISIBLE_DEVICES=2 python3 dfr_evaluate_spurious.py --data_dir $WATERBIRDS_DIR --data_transform 'AugWaterbirdsCelebATransform' \
#  --dataset 'SpuriousCorrelationDataset' --model 'imagenet_resnet50_pretrained' --ckpt_path 'outputs/waterbirds/2_final_checkpoint.pt' \
#  --fgdata_dir $fgWATERBIRDS_DIR --bgdata_dir $bgWATERBIRDS_DIR --DFR_data 'direct_fg'&

# CUDA_VISIBLE_DEVICES=3 python3 dfr_evaluate_spurious.py --data_dir $WATERBIRDS_DIR --data_transform 'AugWaterbirdsCelebATransform' \
#  --dataset 'SpuriousCorrelationDataset' --model 'imagenet_resnet50_pretrained' --ckpt_path 'outputs/waterbirds/3_final_checkpoint.pt' \
#  --fgdata_dir $fgWATERBIRDS_DIR --bgdata_dir $bgWATERBIRDS_DIR --DFR_data 'direct_fg'&
#  wait


# CUDA_VISIBLE_DEVICES=0 python3 dfr_evaluate_spurious.py --data_dir $WATERBIRDS_DIR --data_transform 'AugWaterbirdsCelebATransform' \
#  --dataset 'SpuriousCorrelationDataset' --model 'imagenet_resnet50_pretrained' --ckpt_path 'outputs/waterbirds/0_final_checkpoint.pt' \
#  --fgdata_dir $fgWATERBIRDS_DIR --bgdata_dir $bgWATERBIRDS_DIR --DFR_data 'indirect_fg'&

# CUDA_VISIBLE_DEVICES=1 python3 dfr_evaluate_spurious.py --data_dir $WATERBIRDS_DIR --data_transform 'AugWaterbirdsCelebATransform' \
#  --dataset 'SpuriousCorrelationDataset' --model 'imagenet_resnet50_pretrained' --ckpt_path 'outputs/waterbirds/1_final_checkpoint.pt' \
#  --fgdata_dir $fgWATERBIRDS_DIR --bgdata_dir $bgWATERBIRDS_DIR --DFR_data 'indirect_fg'&

# CUDA_VISIBLE_DEVICES=2 python3 dfr_evaluate_spurious.py --data_dir $WATERBIRDS_DIR --data_transform 'AugWaterbirdsCelebATransform' \
#  --dataset 'SpuriousCorrelationDataset' --model 'imagenet_resnet50_pretrained' --ckpt_path 'outputs/waterbirds/2_final_checkpoint.pt' \
#  --fgdata_dir $fgWATERBIRDS_DIR --bgdata_dir $bgWATERBIRDS_DIR --DFR_data 'indirect_fg'&

# CUDA_VISIBLE_DEVICES=3 python3 dfr_evaluate_spurious.py --data_dir $WATERBIRDS_DIR --data_transform 'AugWaterbirdsCelebATransform' \
#  --dataset 'SpuriousCorrelationDataset' --model 'imagenet_resnet50_pretrained' --ckpt_path 'outputs/waterbirds/3_final_checkpoint.pt' \
#  --fgdata_dir $fgWATERBIRDS_DIR --bgdata_dir $bgWATERBIRDS_DIR --DFR_data 'indirect_fg'&
#  wait


#  CUDA_VISIBLE_DEVICES=0 python3 dfr_evaluate_spurious.py --data_dir $WATERBIRDS_DIR --data_transform 'AugWaterbirdsCelebATransform' \
#  --dataset 'SpuriousCorrelationDataset' --model 'imagenet_resnet50_pretrained' --ckpt_path 'outputs/waterbirds/0_final_checkpoint.pt' \
#  --fgdata_dir $fgWATERBIRDS_DIR --bgdata_dir $bgWATERBIRDS_DIR --DFR_data 'indirect_fg' --normalize_before &

# CUDA_VISIBLE_DEVICES=1 python3 dfr_evaluate_spurious.py --data_dir $WATERBIRDS_DIR --data_transform 'AugWaterbirdsCelebATransform' \
#  --dataset 'SpuriousCorrelationDataset' --model 'imagenet_resnet50_pretrained' --ckpt_path 'outputs/waterbirds/1_final_checkpoint.pt' \
#  --fgdata_dir $fgWATERBIRDS_DIR --bgdata_dir $bgWATERBIRDS_DIR --DFR_data 'indirect_fg' --normalize_before &

# CUDA_VISIBLE_DEVICES=2 python3 dfr_evaluate_spurious.py --data_dir $WATERBIRDS_DIR --data_transform 'AugWaterbirdsCelebATransform' \
#  --dataset 'SpuriousCorrelationDataset' --model 'imagenet_resnet50_pretrained' --ckpt_path 'outputs/waterbirds/2_final_checkpoint.pt' \
#  --fgdata_dir $fgWATERBIRDS_DIR --bgdata_dir $bgWATERBIRDS_DIR --DFR_data 'indirect_fg' --normalize_before &

# CUDA_VISIBLE_DEVICES=3 python3 dfr_evaluate_spurious.py --data_dir $WATERBIRDS_DIR --data_transform 'AugWaterbirdsCelebATransform' \
#  --dataset 'SpuriousCorrelationDataset' --model 'imagenet_resnet50_pretrained' --ckpt_path 'outputs/waterbirds/3_final_checkpoint.pt' \
#  --fgdata_dir $fgWATERBIRDS_DIR --bgdata_dir $bgWATERBIRDS_DIR --DFR_data 'indirect_fg' --normalize_before &
#  wait


# CUDA_VISIBLE_DEVICES=1 python3 dfr_evaluate_spurious_fg.py --data_dir $WATERBIRDS_DIR --data_transform 'AugWaterbirdsCelebATransform' \
#  --dataset 'SpuriousCorrelationDataset' --model 'imagenet_resnet50_pretrained' --ckpt_path 'outputs/waterbirds/1_final_checkpoint.pt' \
#  --fgdata_dir $WATERBIRDS_DIR --isfg 'base'&
 
# CUDA_VISIBLE_DEVICES=2 python3 dfr_evaluate_spurious_fg.py --data_dir $WATERBIRDS_DIR --data_transform 'AugWaterbirdsCelebATransform' \
#  --dataset 'SpuriousCorrelationDataset' --model 'imagenet_resnet50_pretrained' --ckpt_path 'outputs/waterbirds/2_final_checkpoint.pt' \
#  --fgdata_dir $WATERBIRDS_DIR --isfg 'base'&

# CUDA_VISIBLE_DEVICES=3 python3 dfr_evaluate_spurious_fg.py --data_dir $WATERBIRDS_DIR --data_transform 'AugWaterbirdsCelebATransform' \
#  --dataset 'SpuriousCorrelationDataset' --model 'imagenet_resnet50_pretrained' --ckpt_path 'outputs/waterbirds/3_final_checkpoint.pt' \
#  --fgdata_dir $WATERBIRDS_DIR --isfg 'base'

# wait


# CUDA_VISIBLE_DEVICES=0 python3 dfr_evaluate_spurious_fg.py --data_dir $WATERBIRDS_DIR --data_transform 'AugWaterbirdsCelebATransform' \
#  --dataset 'SpuriousCorrelationDataset' --model 'imagenet_resnet50_pretrained' --ckpt_path 'outputs/waterbirds/0_final_checkpoint.pt' \
#  --fgdata_dir $fgWATERBIRDS_DIR1 --isfg 'fg'&

# CUDA_VISIBLE_DEVICES=1 python3 dfr_evaluate_spurious_fg.py --data_dir $WATERBIRDS_DIR --data_transform 'AugWaterbirdsCelebATransform' \
#  --dataset 'SpuriousCorrelationDataset' --model 'imagenet_resnet50_pretrained' --ckpt_path 'outputs/waterbirds/1_final_checkpoint.pt' \
#  --fgdata_dir $fgWATERBIRDS_DIR1 --isfg 'fg'&
 
# CUDA_VISIBLE_DEVICES=2 python3 dfr_evaluate_spurious_fg.py --data_dir $WATERBIRDS_DIR --data_transform 'AugWaterbirdsCelebATransform' \
#  --dataset 'SpuriousCorrelationDataset' --model 'imagenet_resnet50_pretrained' --ckpt_path 'outputs/waterbirds/2_final_checkpoint.pt' \
#  --fgdata_dir $fgWATERBIRDS_DIR1 --isfg 'fg'

# CUDA_VISIBLE_DEVICES=3 python3 dfr_evaluate_spurious_fg.py --data_dir $WATERBIRDS_DIR --data_transform 'AugWaterbirdsCelebATransform' \
#  --dataset 'SpuriousCorrelationDataset' --model 'imagenet_resnet50_pretrained' --ckpt_path 'outputs/waterbirds/3_final_checkpoint.pt' \
#  --fgdata_dir $fgWATERBIRDS_DIR1 --isfg 'fg'
#  wait


#  CUDA_VISIBLE_DEVICES=0 python3 dfr_evaluate_spurious_fg.py --data_dir $WATERBIRDS_DIR --data_transform 'AugWaterbirdsCelebATransform' \
#  --dataset 'SpuriousCorrelationDataset' --model 'imagenet_resnet50_pretrained' --ckpt_path 'outputs/waterbirds/0_final_checkpoint.pt' \
#  --fgdata_dir $fgWATERBIRDS_DIR2 --isfg 'bg'&

# CUDA_VISIBLE_DEVICES=1 python3 dfr_evaluate_spurious_fg.py --data_dir $WATERBIRDS_DIR --data_transform 'AugWaterbirdsCelebATransform' \
#  --dataset 'SpuriousCorrelationDataset' --model 'imagenet_resnet50_pretrained' --ckpt_path 'outputs/waterbirds/1_final_checkpoint.pt' \
#  --fgdata_dir $fgWATERBIRDS_DIR2 --isfg 'bg'&
 
# CUDA_VISIBLE_DEVICES=2 python3 dfr_evaluate_spurious_fg.py --data_dir $WATERBIRDS_DIR --data_transform 'AugWaterbirdsCelebATransform' \
#  --dataset 'SpuriousCorrelationDataset' --model 'imagenet_resnet50_pretrained' --ckpt_path 'outputs/waterbirds/2_final_checkpoint.pt' \
#  --fgdata_dir $fgWATERBIRDS_DIR2 --isfg 'bg'

# CUDA_VISIBLE_DEVICES=3 python3 dfr_evaluate_spurious_fg.py --data_dir $WATERBIRDS_DIR --data_transform 'AugWaterbirdsCelebATransform' \
#  --dataset 'SpuriousCorrelationDataset' --model 'imagenet_resnet50_pretrained' --ckpt_path 'outputs/waterbirds/3_final_checkpoint.pt' \
#  --fgdata_dir $fgWATERBIRDS_DIR2 --isfg 'bg'


#  --predict_spurious

