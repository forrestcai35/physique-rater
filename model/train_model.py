import torch
import torch.nn as nn
from dataset_definitions import create_dataloaders
from model_definitions import build_resnet_multioutput
from torchvision import transforms

# CONFIG
CSV_PATH = "data/dataset.csv"
SAVE_MODEL_PATH = "saved_models/base.pth" #CHANGE WHEN YOU WANT TO MAKE NEW MODEL
BATCH_SIZE = 16
VAL_RATIO = 0.2
NUM_EPOCHS = 5
LEARNING_RATE = 1e-4
NUM_WORKERS = 2

def train_model(model, train_loader, val_loader, num_epochs, lr, device="cpu"):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.to(device)

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss_accum = 0.0
        for images, scores in train_loader:
            images, scores = images.to(device), scores.to(device)

            optimizer.zero_grad()
            preds = model(images)  # shape [batch_size, 7]
            loss = criterion(preds, scores)
            loss.backward()
            optimizer.step()

            train_loss_accum += loss.item() * images.size(0)

        epoch_train_loss = train_loss_accum / len(train_loader.dataset)

        # Validation phase
        model.eval()
        val_loss_accum = 0.0
        with torch.no_grad():
            for images, scores in val_loader:
                images, scores = images.to(device), scores.to(device)

                preds = model(images)
                val_loss = criterion(preds, scores)
                val_loss_accum += val_loss.item() * images.size(0)

        epoch_val_loss = val_loss_accum / len(val_loader.dataset)

        print(f"[Epoch {epoch+1}/{num_epochs}] "
              f"Train Loss: {epoch_train_loss:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f}")

    return model

if __name__ == "__main__":
    # Define your image transforms
    base_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # Optionally normalize for ImageNet models:
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225]),
    ])

    # Create DataLoaders
    train_loader, val_loader = create_dataloaders(
        csv_path=CSV_PATH,
        val_ratio=VAL_RATIO,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        transform=base_transform
    )

    # Build model
    model = build_resnet_multioutput(num_outputs=7)

    # Training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = train_model(model, train_loader, val_loader, NUM_EPOCHS, LEARNING_RATE, device=device)

    # Save model
    torch.save(model.state_dict(), SAVE_MODEL_PATH)
    print(f"Model saved to {SAVE_MODEL_PATH}")
