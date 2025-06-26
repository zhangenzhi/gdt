import os
import sys
sys.path.append("./")
    
import torch
from torch import nn
import torch.utils.data as data  # For custom dataset (optional)
import torchvision.transforms as transforms

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import logging

# from model.vit import create_vit_model
from dataset.imagenet import imagenet_distribute, imagenet_subset
from model.gdt_vit import create_gdt_cls

# Configure logging
def log(args):
    logging.basicConfig(
        filename=args.logname,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device_id):
    model.train()  # Set model to training mode
    total_step = len(train_loader)
    best_val_acc = 0.0
    logging.info("Training the ViT model for %d epochs...", num_epochs)

    for epoch in range(num_epochs):
        logging.info("Epoch %d/%d", epoch + 1, num_epochs)
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device_id, non_blocking=True)
            labels = labels.to(device_id, non_blocking=True)
            optimizer.zero_grad()

            # Forward pass, calculate loss
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Log training progress
            running_loss += loss.item()
            if i % 100 == 99 and device_id == 0:  # Log every 100 mini-batches
                logging.info('[%d, %5d] loss: %.3f', epoch + 1, i + 1, running_loss / 100)
                running_loss = 0.0

        # Validate after each epoch
        model.eval()
        val_acc = evaluate_model(model, val_loader, device_id)
        logging.info("Epoch: %d, Validation Accuracy: %.4f", epoch + 1, val_acc)

        # Save the best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_vit_model.pth")

        logging.info('Finished Training Step %d', epoch + 1)

    logging.info('Finished Training. Best Validation Accuracy: %.4f', best_val_acc)

def evaluate_model(model, val_loader, device_id):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device_id, non_blocking=True)
            labels = labels.to(device_id, non_blocking=True)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

def gdt_imagenet_train(args):
    local_rank = int(os.environ['SLURM_LOCALID'])
    os.environ['MASTER_ADDR'] = str(os.environ['HOSTNAME']) #str(os.environ['HOSTNAME'])
    os.environ['MASTER_PORT'] = "29500"
    os.environ['WORLD_SIZE'] = os.environ['SLURM_NTASKS']
    os.environ['RANK'] = os.environ['SLURM_PROCID']
    print("MASTER_ADDR:{}, MASTER_PORT:{}, WORLD_SIZE:{}, WORLD_RANK:{}, local_rank:{}".format(os.environ['MASTER_ADDR'], 
                                                    os.environ['MASTER_PORT'], 
                                                    os.environ['WORLD_SIZE'], 
                                                    os.environ['RANK'],
                                                    local_rank))
    dist.init_process_group(                                   
    	backend='nccl',                                         
   		init_method='env://',                                   
    	world_size=args.world_size,                              
    	rank=int(os.environ['RANK'])                                               
    )
    print("SLURM_LOCALID/lcoal_rank:{}, dist_rank:{}".format(local_rank, dist.get_rank()))

    print(f"Start running basic DDP example on rank {local_rank}.")
    device_id = local_rank % torch.cuda.device_count()
    
    # Create DataLoader for training and validation
    dataloaders = imagenet_distribute(args=args)

    # Create ViT model
    IMG_SIZE = args.img_size
    stages_config = [
        {"depth": 4, "patch_size_in": 32, "patch_size_out": 16, "k_selected_ratio": 0.25, "max_seq_len": (IMG_SIZE//32)**2},
        {"depth": 4, "patch_size_in": 16, "patch_size_out": 8, "k_selected_ratio": 0.25, "max_seq_len": (IMG_SIZE//32)**2},
        {"depth": 4, "patch_size_in": 8, "patch_size_out": 4, "k_selected_ratio": 0.25, "max_seq_len": (IMG_SIZE//32)**2},
        {"depth": 4, "patch_size_in": 4, "patch_size_out": 2, "k_selected_ratio": 0.25, "max_seq_len": (IMG_SIZE//32)**2},
    ]
    model = create_gdt_cls(img_size=IMG_SIZE, stages_config=stages_config, target_leaf_size=16, encoder_embed_dim=786, classifier_embed_dim=786, num_classes=1000, in_channels=3)
    model.to(device_id)
    model = DDP(model, device_ids=[device_id])

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Train the model
    train_model(model, dataloaders['train'], dataloaders['val'], criterion, optimizer, args.num_epochs, device_id=device_id)
    dist.destroy_process_group()

def gdt_vit_ddp(args):
    log(args=args)
    args.world_size = int(os.environ['SLURM_NTASKS'])
    gdt_imagenet_train(args=args)


def gdt_imagenet_train_local(args):

    device_id = 0
    # Create DataLoader for training and validation
    dataloaders = imagenet_subset(args=args)

    # Create ViT model
    IMG_SIZE = args.img_size
    stages_config = [
        {"depth": 4, "patch_size_in": 32, "patch_size_out": 16, "k_selected_ratio": 0.25, "max_seq_len": (IMG_SIZE//32)**2},
        {"depth": 4, "patch_size_in": 16, "patch_size_out": 8, "k_selected_ratio": 0.25, "max_seq_len": (IMG_SIZE//32)**2},
        {"depth": 4, "patch_size_in": 8, "patch_size_out": 4, "k_selected_ratio": 0.25, "max_seq_len": (IMG_SIZE//32)**2},
        {"depth": 4, "patch_size_in": 4, "patch_size_out": 2, "k_selected_ratio": 0.25, "max_seq_len": (IMG_SIZE//32)**2},
    ]
    model = create_gdt_cls(img_size=IMG_SIZE, stages_config=stages_config, target_leaf_size=16, encoder_embed_dim=786, classifier_embed_dim=786, num_classes=1000, in_channels=3)
    model.to(device_id)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Train the model
    train_model(model, dataloaders['train'], dataloaders['val'], criterion, optimizer, args.num_epochs, device_id=device_id)
    
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="GDT-ViT Training Script")
    
    # Arguments for local execution
    parser.add_argument('--task', type=str, default='imagenet', help='Type of task')
    parser.add_argument('--logname', type=str, default='train.log', help='logging of task.')
    parser.add_argument('--output', type=str, default='./output', help='output dir')
    parser.add_argument('--savefile', type=str, default='vit-imagenet', help='output dir')
    parser.add_argument('--gpus', type=int, default=8, help='Epochs for iteration')
    parser.add_argument('--nodes', type=int, default=1, help='Epochs for iteration')
    parser.add_argument('--data_dir', type=str, default='/lustre/orion/nro108/world-shared/enzhi/dataset/imagenet', help='Path to the ImageNet dataset directory')
    parser.add_argument('--seq_length', type=int, default=196, help='Epochs for iteration')
    parser.add_argument('--num_epochs', type=int, default=3, help='Epochs for iteration')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for DataLoader')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for DataLoader')
    parser.add_argument('--pretrained', type=bool, default=False, help='Use pretrained weights')
    parser.add_argument('--reload', type=bool, default=True, help='Reuse previous weights')
    
    args = parser.parse_args()
    args.output = os.path.join(args.output, args.task)
    os.makedirs(args.output, exist_ok=True)
    
    # Call the local training function
    gdt_imagenet_train_local(args)