#python -m torch.distributed.launch --nproc_per_node 1 --master_port=12345 main.py --cfg configs/OCXR/efficientnetb0_window7_1024_22ktoOCXR_finetune_3xlr.yaml --batch-size 16 --accumulation-steps 8 --use-checkpoint --output=/media/ilkay/9d5efc93-0dbc-4fa3-8400-8fa3282615fa/saved_models/
#python -m torch.distributed.launch --nproc_per_node 1 --master_port=12345 main.py --cfg configs/OCXR/efficientnetb1_window7_1024_22ktoOCXR_finetune_3xlr.yaml --batch-size 16 --accumulation-steps 8 --use-checkpoint --output=/media/ilkay/9d5efc93-0dbc-4fa3-8400-8fa3282615fa/saved_models/
#python -m torch.distributed.launch --nproc_per_node 1 --master_port=12345 main.py --cfg configs/OCXR/efficientnetb2_window7_1024_22ktoOCXR_finetune_3xlr.yaml --batch-size 16 --accumulation-steps 8 --use-checkpoint --output=/media/ilkay/9d5efc93-0dbc-4fa3-8400-8fa3282615fa/saved_models/
# python -m torch.distributed.launch --nproc_per_node 1 --master_port=12345 main.py --cfg configs/OCXR/resnet34_window7_1024_22ktoOCXR_finetune_3xlr.yaml --batch-size 16 --accumulation-steps 8 --use-checkpoint --output=/media/ilkay/9d5efc93-0dbc-4fa3-8400-8fa3282615fa/saved_models/
# python -m torch.distributed.launch --nproc_per_node 1 --master_port=12345 main.py --cfg configs/OCXR/resnet50_window7_1024_22ktoOCXR_finetune_3xlr.yaml --batch-size 16 --accumulation-steps 8 --use-checkpoint --output=/media/ilkay/9d5efc93-0dbc-4fa3-8400-8fa3282615fa/saved_models/
# python -m torch.distributed.launch --nproc_per_node 1 --master_port=12345 main.py --cfg configs/OCXR/resnet101_window7_1024_22ktoOCXR_finetune_3xlr.yaml --batch-size 16 --accumulation-steps 8 --use-checkpoint --output=/media/ilkay/9d5efc93-0dbc-4fa3-8400-8fa3282615fa/saved_models/
# python -m torch.distributed.launch --nproc_per_node 1 --master_port=12345 main.py --cfg configs/OCXR/resnet152_window7_1024_22ktoOCXR_finetune_3xlr.yaml --batch-size 16 --accumulation-steps 8 --use-checkpoint --output=/media/ilkay/9d5efc93-0dbc-4fa3-8400-8fa3282615fa/saved_models/
python -m torch.distributed.launch --nproc_per_node 1 --master_port=12345 main.py --pretrained save/swin_base_patch4_window12_384_22k.pth --cfg configs/OCXR/swin_base_patch4_window8_1024_22ktoOCXR_finetune_3xlr.yaml --batch-size 16 --accumulation-steps 8 --use-checkpoint --output=/media/ilkay/9d5efc93-0dbc-4fa3-8400-8fa3282615fa/saved_models/