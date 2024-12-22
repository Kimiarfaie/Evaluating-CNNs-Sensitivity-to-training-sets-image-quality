import argparse
import os
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from tqdm import tqdm
import json
import pandas as pd

# Define transformation for test images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to model input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_test_images(folder):
    """Load test images and labels from a folder organized in synset subfolders."""
    images = []
    labels = []
    synsets = sorted(os.listdir(folder))  # Ensure consistent order
    for synset in synsets:
        synset_folder = os.path.join(folder, synset)
        if not os.path.isdir(synset_folder):
            continue
        for img_file in os.listdir(synset_folder):
            img_path = os.path.join(synset_folder, img_file)
            images.append(img_path)
            labels.append(synset)
    return images, labels, synsets

def load_synset_mapping(mapping_file):
    """Load synset-to-class mapping from a file."""
    mapping = {}
    with open(mapping_file, "r") as f:
        for line in f:
            parts = line.strip().split(", ")
            synset = parts[0]
            class_name = parts[1].split(",")[0]  # Take the first class name
            mapping[synset] = class_name
    return mapping

def evaluate_model(model, images, labels, synsets, class_mapping, device):
    """Evaluate the model and calculate Top-1 and Top-5 accuracy."""
    acc1_count = 0
    acc5_count = 0
    results = []
    class_accuracies = {synset: {"top1_correct": 0, "top5_correct": 0, "total": 0} for synset in synsets}

    for img_path, label in tqdm(zip(images, labels), total=len(images), desc="Evaluating"):
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(img_tensor)
            probs, preds = torch.topk(output, k=5, dim=1)
            probs = torch.softmax(output, dim=1)  # Convert to probabilities
            probs = probs[0].cpu().tolist()  # Extract confidence values
            preds = preds[0].cpu().tolist()  # Extract predictions
        
        top1_pred = preds[0]
        top1_correct = int(top1_pred == synsets.index(label))
        top5_correct = int(synsets.index(label) in preds)
        
        acc1_count += top1_correct
        acc5_count += top5_correct

        # Track per-class accuracy
        class_accuracies[label]["top1_correct"] += top1_correct
        class_accuracies[label]["top5_correct"] += top5_correct
        class_accuracies[label]["total"] += 1

        # Append to results for further analysis
        results.append({
            "image_path": img_path,
            "ground_truth": class_mapping.get(label, label),
            "top1_pred": class_mapping.get(synsets[top1_pred], synsets[top1_pred]),
            "top1_conf": probs[top1_pred],
            "top5_preds": [class_mapping.get(synsets[p], synsets[p]) for p in preds],
            "top5_confs": [probs[p] for p in preds],
            "top1_correct": top1_correct,
            "top5_correct": top5_correct,
        })
    
    # Calculate class-wise average accuracies
    class_avg_accuracies = []
    for synset, data in class_accuracies.items():
        total = data["total"]
        if total > 0:
            avg_top1 = data["top1_correct"] / total
            avg_top5 = data["top5_correct"] / total
        else:
            avg_top1 = avg_top5 = 0.0
        class_avg_accuracies.append({
            "class": class_mapping.get(synset, synset),
            "avg_top1": avg_top1,
            "avg_top5": avg_top5,
        })

    return results, acc1_count / len(images), acc5_count / len(images), class_avg_accuracies

def main():
    parser = argparse.ArgumentParser(description="Evaluate a pretrained model on a test dataset")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the pretrained model file (e.g., model_best.pth.tar)")
    parser.add_argument("--test-folder", type=str, required=True, help="Path to the test dataset folder organized in synset subfolders")
    parser.add_argument("--synset-mapping", type=str, required=True, help="Path to the synset-to-class mapping file")
    parser.add_argument("--id", type=str, required=True, help="Experiment ID to create a folder for saving results")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use (default: 0)")
    parser.add_argument("--arch", type=str, default="alexnet", help="Model architecture (default: alexnet)")
    args = parser.parse_args()

    # Setup output directory
    output_dir = os.path.join("./results", args.id)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "results.xlsx")

    # Load the model
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model = getattr(models, args.arch)(pretrained=False)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    # Load test images and mapping
    images, labels, synsets = load_test_images(args.test_folder)
    class_mapping = load_synset_mapping(args.synset_mapping)

    # Evaluate the model
    results, mean_acc1, mean_acc5, class_avg_accuracies = evaluate_model(model, images, labels, synsets, class_mapping, device)

    # Save results to Excel
    df_results = pd.DataFrame(results)
    df_classes = pd.DataFrame(class_avg_accuracies)
    best_class = df_classes.loc[df_classes["avg_top1"].idxmax()]
    worst_class = df_classes.loc[df_classes["avg_top1"].idxmin()]

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        df_results.to_excel(writer, index=False, sheet_name="Image Results")
        df_classes.to_excel(writer, index=False, sheet_name="Class Averages")
        worksheet = writer.sheets["Class Averages"]
        worksheet.cell(row=len(df_classes) + 2, column=1, value="Best Class:")
        worksheet.cell(row=len(df_classes) + 3, column=1, value=f"{best_class['class']} (Top-1 Avg: {best_class['avg_top1']:.2%})")
        worksheet.cell(row=len(df_classes) + 5, column=1, value="Worst Class:")
        worksheet.cell(row=len(df_classes) + 6, column=1, value=f"{worst_class['class']} (Top-1 Avg: {worst_class['avg_top1']:.2%})")

    # Save summary to JSON
    summary_file = os.path.join(output_dir, "summary.json")
    with open(summary_file, "w") as f:
        json.dump({
            "Top-1 Accuracy": mean_acc1,
            "Top-5 Accuracy": mean_acc5,
            "Total Images": len(images),
            "Best Class": best_class["class"],
            "Best Class Top-1 Avg": best_class["avg_top1"],
            "Worst Class": worst_class["class"],
            "Worst Class Top-1 Avg": worst_class["avg_top1"],
        }, f, indent=4)

    print(f"Top-1 Accuracy: {mean_acc1:.2%}")
    print(f"Top-5 Accuracy: {mean_acc5:.2%}")
    print(f"Results saved in {output_file}")
    print(f"Summary saved in {summary_file}")

if __name__ == "__main__":
    main()
