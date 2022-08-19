export CUDA_VISIBLE_DEVICES=0
# python -m torch.distributed.launch --nproc_per_node=1 tools/test.py
python tools/test.py --output_dir output_rssan_w21 --model RSSAN --model_name RSSAN --log_dir log_rssan_w21 --model_file output_rssan_w21/hsicity/hsicity2/final_state.pth --window_size 21 --test_row_size 12