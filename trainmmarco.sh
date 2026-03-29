#!/bin/bash
BASE_MODEL_DIR="bert-base-uncased"
TRAIN_DATA_LIST="mmarco"
POS_DIR="outputs"
NEG_DIR="outputs"
DATA_DIR="dataset"
INBATCH_PKL_PATH_DIR="outputs/inbatch/mmarco_train_inbatch.pkl"
FEATURE_PKL_PATH_DIR="outputs/features/mmarco_train_features.pkl"
BATCH_SIZE=16
GRAD_ACCUM=6
NEG_K=8
NUM_HEADS=4
HIDDEN_DIM=256
OUTPUT_DIM=1
LN="True"
NORM="False"
PADDING_SIDE="right"
NUM_EPOCHS=25
MAX_SEQ_LENGTH=256
LR=8e-6
ALPHA=0.3
BETA=0.6
GAMMA=0.1
ETA=0.01
TEMPERATURE_IN_BATCH=0.5
TEMPERATURE_HARDNEG=0.5
TEMPERATURE_TEACHER_HARDNEG=0.5
SCALE_PARAM=3.0
LOG_INTERVAL=10
EVAL_INTERVAL=200
TB_DIR="PATH_TO_TENSORBOARD_PATH"
PATIENCE=4
NUM_CKPT=6
TRAINING_LOG="PATH_TO_TRAINING_LOG"
OUTPUT_DIR="PATH_TO_OUTPUT_MODEL/mmarco"
WEIGHT_DECAY=1.0
MAX_GRAD_NORM=0.5
INBATCH_MARGIN=0.15
USE_ADAPTIVE_TEMP="True"
DEBUG="True"
HIDDEN_DROPOUT_PROB=0.3
ATTENTION_DROPOUT_PROB=0.2
CLASSIFIER_DROPOUT_PROB=0.3

# New parameters for advanced regularization
LABEL_SMOOTHING=0.1
USE_GRADIENT_CLIPPING="True"
USE_LAYER_WISE_LR_DECAY="False"
WARMUP_RATIO=0.1
MIN_LR_RATIO=0.001


WORLD_SIZE=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-12346}
gpus=1

python -m torch.distributed.run --nproc_per_node=$gpus --nnode=$WORLD_SIZE --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
                                train.py --base_model_dir $BASE_MODEL_DIR \
                                --train_data_list $TRAIN_DATA_LIST \
                                --pos_dir $POS_DIR \
                                --neg_dir $NEG_DIR \
                                --data_dir $DATA_DIR \
                                --inbatch_pkl_path_dir $INBATCH_PKL_PATH_DIR \
                                --feature_pkl_path_dir $FEATURE_PKL_PATH_DIR \
                                --batch_size $BATCH_SIZE \
                                --gradient_accumulation_steps $GRAD_ACCUM \
                                --neg_K $NEG_K \
                                --num_heads $NUM_HEADS \
                                --hidden_dim $HIDDEN_DIM \
                                --output_dim $OUTPUT_DIM \
                                --ln $LN \
                                --norm $NORM \
                                --num_epochs $NUM_EPOCHS \
                                --padding_side $PADDING_SIDE \
                                --max_seq_length $MAX_SEQ_LENGTH \
                                --lr $LR \
                                --alpha $ALPHA \
                                --beta $BETA \
                                --gamma $GAMMA \
                                --eta $ETA \
                                --temperature_in_batch $TEMPERATURE_IN_BATCH \
                                --temperature_hardneg $TEMPERATURE_HARDNEG \
                                --temperature_teacher_hardneg $TEMPERATURE_TEACHER_HARDNEG \
                                --scale_param $SCALE_PARAM \
                                --log_interval $LOG_INTERVAL \
                                --eval_interval $EVAL_INTERVAL \
                                --tb_dir $TB_DIR \
                                --patience $PATIENCE \
                                --num_ckpt $NUM_CKPT \
                                --training_log $TRAINING_LOG \
                                --output_dir $OUTPUT_DIR \
                                --weight_decay $WEIGHT_DECAY \
                                --max_grad_norm $MAX_GRAD_NORM \
                                --hidden_dropout_prob $HIDDEN_DROPOUT_PROB \
                                --attention_dropout_prob $ATTENTION_DROPOUT_PROB \
                                --classifier_dropout_prob $CLASSIFIER_DROPOUT_PROB \
                                --inbatch_margin $INBATCH_MARGIN \
                                --use_adaptive_temp $USE_ADAPTIVE_TEMP \
                                --debug $DEBUG \
                                --label_smoothing $LABEL_SMOOTHING \
                                --use_gradient_clipping $USE_GRADIENT_CLIPPING \
                                --use_layer_wise_lr_decay $USE_LAYER_WISE_LR_DECAY \
                                --warmup_ratio $WARMUP_RATIO \
                                --min_lr_ratio $MIN_LR_RATIO
