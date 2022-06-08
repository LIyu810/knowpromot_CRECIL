# 先处理数据，
CUDA_VISIBLE_DEVICES=0 python main.py --max_epochs=20  --num_workers=8 \
    --model_name_or_path  ../plm/chinese-roberta-wwm-ext \
    --accumulate_grad_batches 3 \
    --batch_size 6 \
    --data_dir dataset/chinese_dialogue \
    --check_val_every_n_epoch 1 \
    --data_class CHINESE \
    --max_seq_length 512 \
    --model_class BertForMaskedLM \
    --litmodel_class DialogueLitModel \
    --task_name normal \
    --lr 5e-5
