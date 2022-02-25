python -m torch.distributed.launch --nproc_per_node 1 --master_port=12345 main_test.py --cfg configs/LVOT/efficientnetb5_window7_224_1ktoLVOT_finetune_3xlr.yaml --batch-size 128 --use-checkpoint --output=/media/data_models/saved_models/
python -m torch.distributed.launch --nproc_per_node 1 --master_port=12345 main_test.py --cfg configs/LVOT/efficientnetb4_window7_224_1ktoLVOT_finetune_3xlr.yaml --batch-size 128 --use-checkpoint --output=/media/data_models/saved_models/
python -m torch.distributed.launch --nproc_per_node 1 --master_port=12345 main_test.py --cfg configs/LVOT/efficientnetb3_window7_224_1ktoLVOT_finetune_3xlr.yaml --batch-size 128 --use-checkpoint --output=/media/data_models/saved_models/
python -m torch.distributed.launch --nproc_per_node 1 --master_port=12345 main_test.py --cfg configs/LVOT/efficientnetb2_window7_224_1ktoLVOT_finetune_3xlr.yaml --batch-size 128 --use-checkpoint --output=/media/data_models/saved_models/
python -m torch.distributed.launch --nproc_per_node 1 --master_port=12345 main_test.py --cfg configs/LVOT/efficientnetb1_window7_224_1ktoLVOT_finetune_3xlr.yaml --batch-size 128 --use-checkpoint --output=/media/data_models/saved_models/
python -m torch.distributed.launch --nproc_per_node 1 --master_port=12345 main_test.py --cfg configs/LVOT/efficientnetb0_window7_224_1ktoLVOT_finetune_3xlr.yaml --batch-size 128 --use-checkpoint --output=/media/data_models/saved_models/
python -m torch.distributed.launch --nproc_per_node 1 --master_port=12345 main_test.py --cfg configs/LVOT/resnet34_window7_224_1ktoLVOT_finetune_3xlr.yaml --batch-size 128 --use-checkpoint --output=/media/data_models/saved_models/
python -m torch.distributed.launch --nproc_per_node 1 --master_port=12345 main_test.py --cfg configs/LVOT/resnet50_window7_224_1ktoLVOT_finetune_3xlr.yaml --batch-size 128 --use-checkpoint --output=/media/data_models/saved_models/
python -m torch.distributed.launch --nproc_per_node 1 --master_port=12345 main_test.py --cfg configs/LVOT/resnet101_window7_224_1ktoLVOT_finetune_3xlr.yaml --batch-size 128  --use-checkpoint --output=/media/data_models/saved_models/
python -m torch.distributed.launch --nproc_per_node 1 --master_port=12345 main_test.py --cfg configs/LVOT/resnet152_window7_224_1ktoLVOT_finetune_3xlr.yaml --batch-size 128  --use-checkpoint --output=/media/data_models/saved_models/
python -m torch.distributed.launch --nproc_per_node 1 --master_port=12345 main_test.py --cfg configs/LVOT/swin_base_patch4_window7_224_1ktoLVOT_finetune_3xlr.yaml --batch-size 128 --pretrained save/swin_base_patch4_window7_224.pth --use-checkpoint --output=/media/data_models/saved_models/
python -m torch.distributed.launch --nproc_per_node 1 --master_port=12345 main_test.py --cfg configs/LVOT/swin_small_patch4_window7_224_1ktoLVOT_finetune_3xlr.yaml --batch-size 128 --pretrained save/swin_small_patch4_window7_224.pth --use-checkpoint --output=/media/data_models/saved_models/
python -m torch.distributed.launch --nproc_per_node 1 --master_port=12345 main_test.py --cfg configs/LVOT/swin_tiny_patch4_window7_224_1ktoLVOT_finetune_3xlr.yaml --batch-size 128 --pretrained save/swin_tiny_patch4_window7_224.pth --use-checkpoint --output=/media/data_models/saved_models/
python -m torch.distributed.launch --nproc_per_node 1 --master_port=12345 main_test.py --cfg configs/LVOT/swin_base_patch4_window7_224_LVOT.yaml --batch-size 128 --use-checkpoint --output=/media/data_models/saved_models/
python -m torch.distributed.launch --nproc_per_node 1 --master_port=12345 main_test.py --cfg configs/LVOT/swin_small_patch4_window7_224_LVOT.yaml --batch-size 128 --use-checkpoint --output=/media/data_models/saved_models/
python -m torch.distributed.launch --nproc_per_node 1 --master_port=12345 main_test.py --cfg configs/LVOT/swin_tiny_patch4_window7_224_LVOT.yaml --batch-size 128 --use-checkpoint --output=/media/data_models/saved_models/
