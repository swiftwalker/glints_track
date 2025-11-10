python train_glint_unet.py \
--h5 dataset/glints_dataset.h5 \
--output_dir runs/exp8 \
--gpu 3 \
--epochs 50 \
--batch 16 \
--lr 2e-3 \
--loss hybrid \
--lam_focal 1.0 \
--lam_bce 0.1 \
--lam_dice 0.0 \
--div_weight 0.0 \
--div_mode overlap \
--lam_agg 0.0 \
--save_freq 10 \
--keep_last_n 1 \
--shared_memory \
--no_save_optimizer

