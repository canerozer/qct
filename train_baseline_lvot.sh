<<<<<<< HEAD
# python -m torch.distributed.launch --nproc_per_node 1 --master_port=12345 main.py --cfg configs/LVOT/efficientnetb5_window7_224_1ktoLVOT_finetune_3xlr.yaml --batch-size 64 --accumulation-steps 2 --use-checkpoint --output=/media/ilkay/data/caner/saved_models/
# python -m torch.distributed.launch --nproc_per_node 1 --master_port=12345 main.py --cfg configs/LVOT/efficientnetb4_window7_224_1ktoLVOT_finetune_3xlr.yaml --batch-size 64 --accumulation-steps 2 --use-checkpoint --output=/media/ilkay/data/caner/saved_models/
# python -m torch.distributed.launch --nproc_per_node 1 --master_port=12345 main.py --cfg configs/LVOT/efficientnetb3_window7_224_1ktoLVOT_finetune_3xlr.yaml --batch-size 64 --accumulation-steps 2 --use-checkpoint --output=/media/ilkay/data/caner/saved_models/
# python -m torch.distributed.launch --nproc_per_node 1 --master_port=12345 main.py --cfg configs/LVOT/efficientnetb2_window7_224_1ktoLVOT_finetune_3xlr.yaml --batch-size 64 --accumulation-steps 2 --use-checkpoint --output=/media/ilkay/data/caner/saved_models/
# python -m torch.distributed.launch --nproc_per_node 1 --master_port=12345 main.py --cfg configs/LVOT/efficientnetb1_window7_224_1ktoLVOT_finetune_3xlr.yaml --batch-size 64 --accumulation-steps 2 --use-checkpoint --output=/media/ilkay/data/caner/saved_models/
# python -m torch.distributed.launch --nproc_per_node 1 --master_port=12345 main.py --cfg configs/LVOT/efficientnetb0_window7_224_1ktoLVOT_finetune_3xlr.yaml --batch-size 64 --accumulation-steps 2 --use-checkpoint --output=/media/ilkay/data/caner/saved_models/
# python -m torch.distributed.launch --nproc_per_node 1 --master_port=12345 main.py --cfg configs/LVOT/resnet34_window7_224_1ktoLVOT_finetune_3xlr.yaml --batch-size 16 --test-batch-size 256 --accumulation-steps 8 --use-checkpoint --output=/media/ilkay/data/caner/saved_models/
# python -m torch.distributed.launch --nproc_per_node 1 --master_port=12345 main.py --cfg configs/LVOT/resnet50_window7_224_1ktoLVOT_finetune_3xlr.yaml --batch-size 16 --test-batch-size 256 --accumulation-steps 8 --use-checkpoint --output=/media/ilkay/data/caner/saved_models/
# python -m torch.distributed.launch --nproc_per_node 1 --master_port=12345 main.py --cfg configs/LVOT/resnet101_window7_224_1ktoLVOT_finetune_3xlr.yaml --batch-size 64  --accumulation-steps 2 --test-batch-size 256 --use-checkpoint --output=/media/ilkay/data/caner/saved_models/
# python -m torch.distributed.launch --nproc_per_node 1 --master_port=12345 main.py --cfg configs/LVOT/resnet152_window7_224_1ktoLVOT_finetune_3xlr.yaml --batch-size 64  --accumulation-steps 2 --test-batch-size 256 --use-checkpoint --output=/media/ilkay/data/caner/saved_models/
=======
# python -m torch.distributed.launch --nproc_per_node 1 --master_port=12345 main.py --cfg configs/LVOT/efficientnetb5_window7_224_1ktoLVOT_finetune_3xlr.yaml --batch-size 64 --accumulation-steps 2 --use-checkpoint --output=/media/data_models/saved_models/
# python -m torch.distributed.launch --nproc_per_node 1 --master_port=12345 main.py --cfg configs/LVOT/efficientnetb4_window7_224_1ktoLVOT_finetune_3xlr.yaml --batch-size 64 --accumulation-steps 2 --use-checkpoint --output=/media/data_models/saved_models/
# python -m torch.distributed.launch --nproc_per_node 1 --master_port=12345 main.py --cfg configs/LVOT/efficientnetb3_window7_224_1ktoLVOT_finetune_3xlr.yaml --batch-size 64 --accumulation-steps 2 --use-checkpoint --output=/media/data_models/saved_models/
# python -m torch.distributed.launch --nproc_per_node 1 --master_port=12345 main.py --cfg configs/LVOT/efficientnetb2_window7_224_1ktoLVOT_finetune_3xlr.yaml --batch-size 64 --accumulation-steps 2 --use-checkpoint --output=/media/data_models/saved_models/
# python -m torch.distributed.launch --nproc_per_node 1 --master_port=12345 main.py --cfg configs/LVOT/efficientnetb1_window7_224_1ktoLVOT_finetune_3xlr.yaml --batch-size 64 --accumulation-steps 2 --use-checkpoint --output=/media/data_models/saved_models/
# python -m torch.distributed.launch --nproc_per_node 1 --master_port=12345 main.py --cfg configs/LVOT/efficientnetb0_window7_224_1ktoLVOT_finetune_3xlr.yaml --batch-size 64 --accumulation-steps 2 --use-checkpoint --output=/media/data_models/saved_models/
# python -m torch.distributed.launch --nproc_per_node 1 --master_port=12345 main.py --cfg configs/LVOT/resnet34_window7_224_1ktoLVOT_finetune_3xlr.yaml --batch-size 64 --test-batch-size 256 --accumulation-steps 2 --use-checkpoint --output=/media/data_models/saved_models/
# python -m torch.distributed.launch --nproc_per_node 1 --master_port=12345 main.py --cfg configs/LVOT/resnet50_window7_224_1ktoLVOT_finetune_3xlr.yaml --batch-size 64 --test-batch-size 256 --accumulation-steps 2 --use-checkpoint --output=/media/data_models/saved_models/
# python -m torch.distributed.launch --nproc_per_node 1 --master_port=12345 main.py --cfg configs/LVOT/resnet101_window7_224_1ktoLVOT_finetune_3xlr.yaml --batch-size 64  --accumulation-steps 2 --test-batch-size 256 --use-checkpoint --output=/media/data_models/saved_models/
# python -m torch.distributed.launch --nproc_per_node 1 --master_port=12345 main.py --cfg configs/LVOT/resnet152_window7_224_1ktoLVOT_finetune_3xlr.yaml --batch-size 64  --accumulation-steps 2 --test-batch-size 256 --use-checkpoint --output=/media/data_models/saved_models/
>>>>>>> c366a4dc0d223f030d1a724ed41ebf0e8210e7bc

# Models with 3x Learning Rates by using pre-trained models
# python -m torch.distributed.launch --nproc_per_node 1 --master_port=12345 main.py --cfg configs/LVOT/swin_base_patch4_window7_224_1ktoLVOT_finetune_3xlr.yaml --batch-size 64 --pretrained save/swin_base_patch4_window7_224.pth --accumulation-steps 2 --test-batch-size 256 --use-checkpoint --output=/media/ilkay/data/caner/saved_models/
# python -m torch.distributed.launch --nproc_per_node 1 --master_port=12345 main.py --cfg configs/LVOT/swin_small_patch4_window7_224_1ktoLVOT_finetune_3xlr.yaml --batch-size 64 --pretrained save/swin_small_patch4_window7_224.pth --accumulation-steps 2 --test-batch-size 256 --use-checkpoint --output=/media/ilkay/data/caner/saved_models/
# python -m torch.distributed.launch --nproc_per_node 1 --master_port=12345 main.py --cfg configs/LVOT/swin_tiny_patch4_window7_224_1ktoLVOT_finetune_3xlr.yaml --batch-size 64 --pretrained save/swin_tiny_patch4_window7_224.pth --accumulation-steps 2 --test-batch-size 256 --use-checkpoint --output=/media/ilkay/data/caner/saved_models/

# Training models from scratch
# python -m torch.distributed.launch --nproc_per_node 1 --master_port=12345 main.py --cfg configs/LVOT/swin_base_patch4_window7_224_LVOT.yaml --batch-size 64 --accumulation-steps 2 --test-batch-size 256 --use-checkpoint --output=/media/ilkay/data/caner/saved_models/
# python -m torch.distributed.launch --nproc_per_node 1 --master_port=12345 main.py --cfg configs/LVOT/swin_small_patch4_window7_224_LVOT.yaml --batch-size 64 --accumulation-steps 2 --test-batch-size 256 --use-checkpoint --output=/media/ilkay/data/caner/saved_models/
# python -m torch.distributed.launch --nproc_per_node 1 --master_port=12345 main.py --cfg configs/LVOT/swin_tiny_patch4_window7_224_LVOT.yaml --batch-size 64 --accumulation-steps 2 --test-batch-size 256 --use-checkpoint --output=/media/ilkay/data/caner/saved_models/

# Models with Regular Learning Rates by using pre-trained models
# python -m torch.distributed.launch --nproc_per_node 1 --master_port=12345 main.py --cfg configs/LVOT/swin_base_patch4_window7_224_1ktoLVOT_finetune.yaml --batch-size 64 --pretrained save/swin_base_patch4_window7_224.pth --accumulation-steps 2 --test-batch-size 256 --use-checkpoint --output=/media/ilkay/data/caner/saved_models/
python -m torch.distributed.launch --nproc_per_node 1 --master_port=12345 main.py --cfg configs/LVOT/swin_small_patch4_window7_224_1ktoLVOT_finetune.yaml --batch-size 64 --pretrained save/swin_small_patch4_window7_224.pth --accumulation-steps 2 --test-batch-size 256 --use-checkpoint --output=/media/ilkay/data/caner/saved_models/
python -m torch.distributed.launch --nproc_per_node 1 --master_port=12345 main.py --cfg configs/LVOT/swin_tiny_patch4_window7_224_1ktoLVOT_finetune.yaml --batch-size 64 --pretrained save/swin_tiny_patch4_window7_224.pth --accumulation-steps 2 --test-batch-size 256 --use-checkpoint --output=/media/ilkay/data/caner/saved_models/
