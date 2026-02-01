import torch
import os
import logging

def save_checkpoint(model, optimizer, scheduler, scaler, epoch, config, path):
    """
    Saves a comprehensive checkpoint for the HMAE training process.
    """
    # Check if we are using DDP and get the underlying module
    model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'config': config
    }
    
    torch.save(checkpoint, path)
    logging.info(f"Checkpoint saved at epoch {epoch+1}: {path}")

def load_checkpoint(model, optimizer, scheduler, scaler, path, device):
    """
    Loads a checkpoint and restores the state of the model, optimizer, scheduler, and scaler.
    """
    if not os.path.exists(path):
        logging.warning(f"No checkpoint found at {path}. Starting from scratch.")
        return 0

    checkpoint = torch.load(path, map_location=device)
    
    # Load model state
    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
        
    # Load training states
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1
    logging.info(f"Successfully loaded checkpoint: {path} (resuming from epoch {start_epoch})")
    
    return start_epoch