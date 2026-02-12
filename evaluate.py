import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import hydra
from hydra import compose
from hydra.core.global_hydra import GlobalHydra
import gc
from utils import process_input, process_output, slice_nms
import json
from PIL import Image
import matplotlib.pyplot as plt

def load_case(file_path):
    data = np.load(file_path, allow_pickle=True)
    image = data["imgs"]
    text_prompts = data["text_prompts"].item()
    gt = data["gts"] if "gts" in data else None
    return image, text_prompts, gt


def merge_multiclass_masks(masks, ids):
    bg_mask = 0.5 * torch.ones_like(masks[0:1])
    keep_masks = torch.cat([bg_mask, masks], dim=0)
    class_mask = keep_masks.argmax(dim=0)

    id_map = {j + 1: int(ids[j]) for j in range(len(ids)) if j + 1 != int(ids[j])}
    if len(id_map) > 0:
        orig_mask = class_mask.clone()
        for j in id_map:
            class_mask[orig_mask == j] = id_map[j]

    return class_mask


def postprocess(model_outputs, object_existence, threshold=0.5, do_nms=True):
    if do_nms and model_outputs.shape[0] > 1:
        # do non-max suppression for each slice
        return slice_nms(model_outputs.sigmoid(), object_existence.sigmoid(), 
                                        iou_threshold=0.5, score_threshold=threshold)
    mask = (model_outputs.sigmoid()) * (
        object_existence.sigmoid() > threshold
    ).int().unsqueeze(-1).unsqueeze(-1)
    return mask


def compute_dice_coefficient(mask_gt, mask_pred):
    volume_sum = mask_gt.sum() + mask_pred.sum()
    if volume_sum == 0:
        return 1.0  # Both masks are empty, perfect agreement on healthy
    volume_intersect = (mask_gt & mask_pred).sum()
    if volume_sum == 0:
        return 1.0
    return 2 * volume_intersect / volume_sum


def print_memory_info(stage=""):
    print(
        f"[{stage}] GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB"
    )
    print(
        f"[{stage}] GPU memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB"
    )


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    GlobalHydra.instance().clear()
    hydra.initialize(config_path="configs", job_name="example_prediction")
    cfg = compose(config_name="finetune_biomedparse")
    model = hydra.utils.instantiate(cfg.model, _convert_="object")
    
    # Load the checkpoint
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    
    # Strip "model." prefix from keys
    state_dict = checkpoint['state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('model.'):
            new_state_dict[k[6:]] = v
    
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)
    
    data_dir = os.path.dirname(args.test_json)

    with open(args.test_json, 'r') as f:
        test_data = json.load(f)

    dice_scores = []
    correct_classifications = 0
    total_images = 0

    for item in test_data:
        image_path = os.path.join(data_dir, item['image'])
        mask_path = os.path.join(data_dir, item['label'])
        text = item['text']

        print(f"Processing: {image_path}")

        # Load image and mask
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)
        
        gt_mask = Image.open(mask_path).convert("L")
        gt_mask = gt_mask.resize((image.shape[1], image.shape[0]), Image.NEAREST)
        gt_mask = np.array(gt_mask) > 0

        # Determine ground truth class
        is_gt_healthy = gt_mask.sum() == 0
        gt_class = "Healthy" if is_gt_healthy else text

        imgs, pad_width, padded_size, test_axis = process_input(image, 512)
        imgs = imgs.to(device).int()

        input_tensor = {
            "image": imgs.unsqueeze(0),  # Add batch dimension
            "text": [text],
        }

        with torch.no_grad():
            output = model(input_tensor)

        # Extract confidence and predicted class
        confidence_score = torch.sigmoid(output["predictions"]["object_existence"]).item()
        # Assuming 'pred_labels' is part of the output, may need adjustment
        predicted_class = output["predictions"].get("pred_labels", [text])[0] 
        
        pred_mask = np.zeros_like(gt_mask)
        
        # Logic update based on confidence score
        if confidence_score >= 0.5:
            mask_preds = output["predictions"]["pred_gmasks"]
            mask_preds = F.interpolate(
                mask_preds,
                size=(512, 512),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
            mask_preds = postprocess(mask_preds, output["predictions"]["object_existence"])
            mask_preds = merge_multiclass_masks(mask_preds, [1]) # Assuming single class
            mask_preds = process_output(mask_preds, pad_width, padded_size, test_axis)
            pred_mask = (mask_preds > 0).squeeze()
        else:
            predicted_class = "Healthy"

        # Update stats
        dice_score = compute_dice_coefficient(gt_mask, pred_mask)
        dice_scores.append(dice_score)

        if gt_class == predicted_class:
            correct_classifications += 1
        total_images += 1
        
        print(f"GT Class: {gt_class}, Predicted Class: {predicted_class}, Confidence: {confidence_score:.2f}, Dice: {dice_score:.4f}")

        # Visualization
        fig, ax = plt.subplots(1, 3, figsize=(18, 6))
        ax[0].imshow(image)
        ax[0].set_title("Input Image")
        ax[0].axis("off")

        ax[1].imshow(gt_mask, cmap='gray')
        ax[1].set_title(f"Ground Truth: {gt_class}")
        ax[1].axis("off")

        ax[2].imshow(image)
        ax[2].imshow(pred_mask, cmap='jet', alpha=0.5)
        title = f"Predicted: {predicted_class} (Conf: {confidence_score:.2f})\nDice: {dice_score:.4f}"
        ax[2].set_title(title)
        ax[2].axis("off")
        
        output_filename = os.path.join(args.output_dir, os.path.basename(item['image']))
        plt.savefig(output_filename)
        plt.close()

        # Cleanup
        del imgs, input_tensor, output
        gc.collect()
        torch.cuda.empty_cache()

    # Final summary
    mean_dice = np.mean(dice_scores) if dice_scores else 0
    classification_accuracy = (correct_classifications / total_images) * 100 if total_images > 0 else 0
    
    print("\n--- Evaluation Summary ---")
    print(f"Processed {total_images} images.")
    print(f"Mean Dice Score: {mean_dice:.4f}")
    print(f"Classification Accuracy: {classification_accuracy:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_json", required=True, help="Path to the test JSON file.")
    parser.add_argument("--checkpoint_path", required=True, help="Path to the model checkpoint.")
    parser.add_argument("--output_dir", required=True, help="Directory to save the output images.")
    args = parser.parse_args()
    main(args)
