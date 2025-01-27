import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset_definitions import create_dataloaders
from model_definitions import build_resnet_multioutput
from torchvision import transforms
import numpy as np

# CONFIG
CSV_PATH = "data/train_classic.csv"
SAVE_MODEL_PATH = "saved_models/best_model.pth"
BATCH_SIZE = 32  # Increased batch size
VAL_RATIO = 0.2
NUM_EPOCHS = 50  # More epochs with early stopping
LEARNING_RATE = 1e-4
NUM_WORKERS = 4  # Increased workers for better loading
PATIENCE = 5  # For early stopping
WEIGHT_DECAY = 1e-4  # For AdamW optimizer

# Enhanced transforms with augmentation
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def train_model(model, train_loader, val_loader, num_epochs, lr, device="cpu"):
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    
    best_val_loss = np.inf
    no_improve = 0
    
    # For tracking per-category metrics
    score_names = ["Arms", "Chest", "Abs", "Vascularity", 
                  "Proportions", "Potential", "Legs", "Back"]
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mae': [[] for _ in score_names]
    }

    model.to(device)

    for epoch in range(num_epochs):
        # --- Training Phase ---
        model.train()
        train_loss_accum = 0.0
        
        for images, scores in train_loader:
            images, scores = images.to(device), scores.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, scores)
            loss.backward()
            optimizer.step()
            
            train_loss_accum += loss.item() * images.size(0)

        # --- Validation Phase ---
        model.eval()
        val_loss_accum = 0.0
        mae_accum = [0.0] * len(score_names)
        
        with torch.no_grad():
            for images, scores in val_loader:
                images, scores = images.to(device), scores.to(device)
                outputs = model(images)
                
                val_loss = criterion(outputs, scores)
                val_loss_accum += val_loss.item() * images.size(0)
                
                # Calculate MAE for each category
                mae = torch.abs(outputs - scores).mean(dim=0)
                for i in range(len(score_names)):
                    mae_accum[i] += mae[i].item() * images.size(0)

        # --- Statistics ---
        train_loss = train_loss_accum / len(train_loader.dataset)
        val_loss = val_loss_accum / len(val_loader.dataset)
        val_mae = [mae / len(val_loader.dataset) for mae in mae_accum]
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # --- Early Stopping Check ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), SAVE_MODEL_PATH)
            no_improve = 0
        else:
            no_improve += 1
            
        # --- Store History ---
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        for i, mae in enumerate(val_mae):
            history['val_mae'][i].append(mae)

        # --- Progress Reporting ---
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print("Validation MAE:")
        for name, mae in zip(score_names, val_mae):
            print(f"- {name}: {mae:.4f}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        if no_improve >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    # Load best model weights
    model.load_state_dict(torch.load(SAVE_MODEL_PATH))
    return model, history

if __name__ == "__main__":
    # Create DataLoaders with separate transforms
    train_loader, val_loader = create_dataloaders(
        csv_path=CSV_PATH,
        val_ratio=VAL_RATIO,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        train_transform=train_transform,
        val_transform=val_transform
    )

    # Build model with correct output size
    model = build_resnet_multioutput(num_outputs=8)  # 8 output scores

    # Training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nTraining on {device.upper()} with {len(train_loader.dataset)} samples")
    
    model, history = train_model(
        model, 
        train_loader, 
        val_loader,
        num_epochs=NUM_EPOCHS,
        lr=LEARNING_RATE,
        device=device
    )

    print(f"\nBest model saved to {SAVE_MODEL_PATH}")