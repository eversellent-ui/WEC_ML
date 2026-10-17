"""
EcoSort Challenge — Model Training & Export Template
=====================================================
Usage:
    python train_template.py --epochs 10 --batch_size 32 --lr 0.001
"""

import argparse
import os
import torch
import torch.nn as nn
from Template_Content.starter_kit.model_A import Material_Identifier
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_dataloaders(data_dir: str, batch_size: int = 32):
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        raise FileNotFoundError(
            f"Dataset not found at '{data_dir}'. Run 'python download_data.py' first!"
        )

        # we did some data augmentation to improve the training data set by increasing variations
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(8), # they are in degrees not radians (also this produces a rotation b/w range of -8.0 to 8.0
        transforms.Resize((96, 96)),
        transforms.ToTensor(),  # converts pil image to float tensor in range 0,1.0
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_loader = DataLoader(
        datasets.ImageFolder(train_dir, transform=train_transform),
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        datasets.ImageFolder(val_dir, transform=val_transform),
        batch_size=batch_size,
        shuffle=False
    )
    return train_loader, val_loader


def export_to_onnx(model: nn.Module, output_path: str = "model.onnx"):
    model.eval()
    dummy_input = torch.randn(1, 3, 96, 96)
    
    print(f"Exporting model to '{output_path}'...")
    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            dynamo=False
        )
    except TypeError:
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
        )
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Exported successfully! File size: {size_mb:.2f} MB (Limit: <= 1.50 MB)")


def main():
    parser = argparse.ArgumentParser(description="WEB CLUB FINAL MODEL")
    parser.add_argument("--data_dir", type=str, default="data/public", help="Path to public dataset")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--output", type=str, default="model.onnx", help="Output ONNX filename")
    args = parser.parse_args()  # parse_args is used to capture terminal commands

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = get_dataloaders(args.data_dir, batch_size=args.batch_size)
    print(f"Loaded {len(train_loader.dataset)} training and {len(val_loader.dataset)} validation images.")

    model = Material_Identifier(num_classes=len(CLASSES)).to(device)
    total_params = count_parameters(model)
    print(f"Total Parameters: {total_params:,} / 350,000 max allowed")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,weight_decay=1e-4)

    scheduler=CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=1e-6)
    max_val_loss=float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss=0.0
        correct=0
        total = 0
        val_loss=0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()  # optimizer being converted back to original form [ after being differentiated]
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_acc = (correct / total) * 100.0

        model.eval()
        v_correct, v_total = 0, 0

        with torch.no_grad():  # no gradient tracking needed here as its for testing
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss=criterion(outputs,labels)  # calculates average loss of the sample
                v_correct += (outputs.argmax(1) == labels).sum().item()
                v_total += labels.size(0)
                val_loss+= loss.item()*images.size(0)

        scheduler.step()
        val_acc = (v_correct / v_total) * 100.0
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}%")

        if val_loss < max_val_loss:
            max_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")





    model.load_state_dict(torch.load("best_model.pth" ,map_location=device, weights_only=True))
    export_to_onnx(model.cpu(), output_path=args.output)
    print(f"\nNext step: Verify your submission before uploading:")
    print(f"  python verify_submission.py --model {args.output}")


if __name__ == "__main__":
    main()
