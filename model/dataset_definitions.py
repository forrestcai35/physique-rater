import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

class PhysiqueMultiOutputDataset(Dataset):
    """
    Expects a DataFrame with columns:
      image_path, arms_score, chest_score, abs_score, vascularity_score,
      proportions_score, potential_score, legs_score
    """
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1) Load image
        image_path = row["image_path"]
        image = Image.open(image_path).convert("RGB")

        # 2) Apply transforms
        if self.transform:
            image = self.transform(image)

        # 3) Collect the 7 scores
        scores = [
            float(row["arms_score"]),
            float(row["chest_score"]),
            float(row["abs_score"]),
            float(row["vascularity_score"]),
            float(row["proportions_score"]),
            float(row["potential_score"]),
            float(row["legs_score"]),
        ]
        scores_tensor = torch.tensor(scores, dtype=torch.float32)

        return image, scores_tensor

def create_dataloaders(csv_path,
                       val_ratio=0.2,
                       batch_size=16,
                       num_workers=2,
                       transform=None):
    """
    Reads the CSV, creates a dataset, splits it into train/val,
    and returns DataLoaders for both.
    """
    # Read CSV
    df = pd.read_csv(csv_path)

    # Create full dataset
    full_dataset = PhysiqueMultiOutputDataset(df, transform=transform)

    # Split into train/val
    total_len = len(full_dataset)
    val_len = int(total_len * val_ratio)
    train_len = total_len - val_len

    train_dataset, val_dataset = random_split(full_dataset, [train_len, val_len])

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers)

    return train_loader, val_loader
