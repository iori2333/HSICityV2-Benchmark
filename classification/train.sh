# Set visible gpus here
export CUDA_VISIBLE_DEVICES=0,1,2

# config --nproc_per_node and --world_size to gpu numbers you want.
python -m torch.distributed.launch --nproc_per_node=3 tools/train.py --output_dir output_rssan --model RSSAN --model_name RSSAN --window_size 17 --world_size 3