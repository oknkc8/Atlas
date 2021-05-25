python train_pytorch.py \
--config-file ./configs/base.yaml \
TRAINER.NAME atlas \
TRAINER.VERSION default \
TRAINER.GPUS 4,5,6,7 \
DATA.NUM_FRAMES_TRAIN 5 \
DATA.NUM_FRAMES_VAL 5 \
DATA.NUM_WORKERS 16 \
