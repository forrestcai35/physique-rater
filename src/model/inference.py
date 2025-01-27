import torch
from PIL import Image
from torchvision import transforms

from model_definitions import build_resnet_multioutput

# Where the model weights are stored
MODEL_PATH = "saved_models/multi_output_physique.pth"

def predict_scores(model, image_path, transform, device="cpu"):
    """
    Predicts the scores for a single image.
    Returns a list or tensor.
    """
    model.eval()
    model.to(device)

    # Load and transform the image
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        preds = model(img_tensor)  

    return preds.squeeze().cpu().numpy()  

if __name__ == "__main__":
    # Example usage

    # 1) Define same transforms used in training (minus random augmentations)
    base_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),

        # NORMALIZATION FOR TRAINING 
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # 2) Rebuild model architecture
    model = build_resnet_multioutput(num_outputs=7)

    # 3) Load trained weights
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)

    # 4) Inference on a test image
    test_image = "some_test_image.jpg"  # Provide a valid path
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scores = predict_scores(model, test_image, base_transform, device=device)

    # 5) Print results
    print("Predicted multi-output scores:")
    print(f"Arms: {scores[0]:.2f}")
    print(f"Chest: {scores[1]:.2f}")
    print(f"Shoulders: {scores[2]:.2f}")
    print(f"Abs: {scores[3]:.2f}")
    print(f"Legs: {scores[4]:.2f}")
    print(f"Back: {scores[5]:.2f}")
    print(f"Definition: {scores[5]:.2f}")
    print(f"Proportions: {scores[6]:.2f}")
    print(f"Potential: {scores[7]:.2f}")
    print(f"Size: {scores[8]:.2f}")
    print(f"Vascularity: {scores[9]:.2f}")