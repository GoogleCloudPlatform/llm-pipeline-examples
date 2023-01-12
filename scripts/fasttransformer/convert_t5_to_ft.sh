cd ~/FasterTransformer/build
./bin/t5_gemm 8 4 32 512 8 64 2048 512 8 64 2048 32128 0 2 1

python ../examples/tensorflow/t5/translate_example.py \
    --batch_size 32 \
    --beam_width 4 \
    --max_seq_len 128 \
    --data_type fp32 \
    --test_time 13 \
    --sampling_topk 4 \
    --model t5-small