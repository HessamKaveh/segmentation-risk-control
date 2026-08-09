import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from dataset import get_dataloaders, NUM_CLASSES
from model import SimpleFCN

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
MODEL_PATH = os.path.join(RESULTS_DIR, "best_fcn.pt")


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_loader = get_dataloaders(batch_size=4)

    model = SimpleFCN(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    images, masks = next(iter(val_loader))
    images = images.to(device)

    with torch.no_grad():
        outputs = model(images)
        outputs = torch.nn.functional.interpolate(outputs, size=masks.shape[-2:], mode='bilinear', align_corners=False)
        preds = outputs.argmax(dim=1).cpu()

    n_samples = min(3, images.size(0))
    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 3 * n_samples))

    for i in range(n_samples):
        # Denormalize image
        img = images[i].cpu().permute(1, 2, 0).numpy()
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)

        axes[i, 0].imshow(img); axes[i, 0].set_title("Image"); axes[i, 0].axis("off")
        axes[i, 1].imshow(masks[i], cmap="tab20"); axes[i, 1].set_title("GT Mask"); axes[i, 1].axis("off")
        axes[i, 2].imshow(preds[i], cmap="tab20"); axes[i, 2].set_title("Pred Mask"); axes[i, 2].axis("off")

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "segmentation_results.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved segmentation results to {out_path}")


if __name__ == "__main__":
    evaluate()
