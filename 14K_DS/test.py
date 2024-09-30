import os
from tqdm import tqdm

def iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) between two bounding boxes.
    Box format: [x_center, y_center, width, height] (normalized).
    """
    box1_x1 = box1[0] - box1[2] / 2
    box1_y1 = box1[1] - box1[3] / 2
    box1_x2 = box1[0] + box1[2] / 2
    box1_y2 = box1[1] + box1[3] / 2
    
    box2_x1 = box2[0] - box2[2] / 2
    box2_y1 = box2[1] - box2[3] / 2
    box2_x2 = box2[0] + box2[2] / 2
    box2_y2 = box2[1] + box2[3] / 2
    
    x1_inter = max(box1_x1, box2_x1)
    y1_inter = max(box1_y1, box2_y1)
    x2_inter = min(box1_x2, box2_x2)
    y2_inter = min(box1_y2, box2_y2)
    
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area



def process_label_file(file):
    """ Process a label file and return a list of bounding boxes. """
    boxes = []
    for line in file:
        # Assuming each line contains: class x_center y_center width height
        parts = line.strip().split()
        if len(parts) == 5:
            class_id = int(parts[0])  # Class ID (if needed)
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            boxes.append([x_center, y_center, width, height])
    return boxes


def load_labels(folder):
    labels = []
    try:
        label_files = [f for f in os.listdir(folder) if f.endswith('.txt')]
        for label_file in tqdm(label_files, desc="Loading Labels", unit="file"):
            with open(os.path.join(folder, label_file), 'r') as f:
                labels.extend(process_label_file(f))
    except PermissionError as e:
        print(f"Permission error accessing {folder}: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
    return labels

def compare_predictions(validation_folder, predicted_folder):
    tp, fp, fn, tn = 0, 0, 0, 0
    total_val_boxes, total_pred_boxes = 0, 0

    try:
        val_labels = load_labels(validation_folder)
        pred_labels = load_labels(predicted_folder)

        total_val_boxes = len(val_labels)
        total_pred_boxes = len(pred_labels)

        if total_val_boxes == 0 or total_pred_boxes == 0:
            print("No bounding boxes found in validation or predicted labels.")
            return 0, 0, 0, 0, total_val_boxes, total_pred_boxes

        matched_pred = [False] * total_pred_boxes

        # Calculate IoU for predictions and count TP, FP, FN
        for gt_box in tqdm(val_labels, desc="Calculating IoU", unit="box"):
            found_match = False
            for i, pred_box in enumerate(pred_labels):
                if iou(gt_box, pred_box) >= 0.5:
                    tp += 1
                    matched_pred[i] = True
                    found_match = True
                    break
            if not found_match:
                fn += 1

        for matched in matched_pred:
            if not matched:
                fp += 1
        
        tn = (total_val_boxes + total_pred_boxes) - (tp + fp + fn)

    except Exception as e:
        print(f"An error occurred: {e}")
        return 0, 0, 0, 0, total_val_boxes, total_pred_boxes

    accuracy = (tp + tn) / (total_val_boxes + total_pred_boxes) if (total_val_boxes + total_pred_boxes) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

    return tp, fp, fn, tn, accuracy, precision, recall, f1_score, total_val_boxes, total_pred_boxes

def save_metrics_to_file(tp, fp, fn, tn, accuracy, precision, recall, f1_score, total_val_boxes, total_pred_boxes, output_file):
    metrics = (
        f"Total Bounding Boxes in Validation Labels: {total_val_boxes}\n"
        f"Total Bounding Boxes in Predicted Labels: {total_pred_boxes}\n"
        f"True Positives (TP): {tp}\n"
        f"False Positives (FP): {fp}\n"
        f"False Negatives (FN): {fn}\n"
        f"True Negatives (TN): {tn}\n"
        f"Accuracy: {accuracy:.2f}\n"
        f"Precision: {precision:.2f}\n"
        f"Recall: {recall:.2f}\n"
        f"F1 Score: {f1_score:.2f}\n"
    )

    # Save metrics to a text file
    with open(output_file, 'w') as f:
        f.write(metrics)

def main():
    # Paths to your folders
    validation_folder = 'C:/Users/Neha KB/Desktop/14K_DS/valid_s/labels'
    predicted_folder = 'C:/Users/Neha KB/Desktop/14K_DS/predicted_s/labels'
    output_metrics_file = 'C:/Users/Neha KB/Desktop/14K_DS/metrics.txt'

    # Compare the predictions and get metrics
    tp, fp, fn, tn, accuracy, precision, recall, f1_score, total_val_boxes, total_pred_boxes = compare_predictions(validation_folder, predicted_folder)

    # Save the metrics to a text file
    save_metrics_to_file(tp, fp, fn, tn, accuracy, precision, recall, f1_score, total_val_boxes, total_pred_boxes, output_metrics_file)

    # Print results for debugging/confirmation
    print(f"Total Bounding Boxes in Validation Labels: {total_val_boxes}")
    print(f"Total Bounding Boxes in Predicted Labels: {total_pred_boxes}")
    print(f"Metrics saved to {output_metrics_file}")

if __name__ == "__main__":
    main()
