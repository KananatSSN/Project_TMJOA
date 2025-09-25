from ultralytics import YOLO
import os
import pprint

def calculate_metrics(predictions, classes_name = ['0', '1']):
    """
    Calculate accuracy, precision, recall, and F1 score for binary classification.
    
    Args:
        predictions: List of (real_value, predicted_value) tuples where values are 0 or 1
    
    Returns:
        dict: Dictionary containing accuracy, precision, recall, and f1_score
    """
    
    # Calculate confusion matrix components
    tp = sum(1 for real, pred in predictions if real == classes_name[1] and pred == classes_name[1])  # True Positives
    fp = sum(1 for real, pred in predictions if real == classes_name[0] and pred == classes_name[1])  # False Positives
    fn = sum(1 for real, pred in predictions if real == classes_name[1] and pred == classes_name[0])  # False Negatives
    tn = sum(1 for real, pred in predictions if real == classes_name[0] and pred == classes_name[0])  # True Negatives
    
    # Total predictions
    total = len(predictions)
    
    # Calculate metrics
    accuracy = (tp + tn) / total
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'confusion_matrix': {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}
    }

def predict_image_folder(model, image_folder_path, conf_threshold = 0.7):
    images_name = os.listdir(image_folder_path)

    result_from_images = []
    for image_name in images_name:

        image_path = os.path.join(image_folder_path, image_name)
        if os.path.exists(image_path):
            
            results = model(image_path, verbose = False)
            result = results[0].probs
            if result :
                
                (predicted_class, conf) = (result.top1, result.top1conf.item())
                result_from_images.append((predicted_class, conf))
        
    # print(result_from_images)

    predicted = []
    for predicted_class, conf in result_from_images:
        # print(predicted_class, conf)
        if conf >= conf_threshold:
            predicted.append(predicted_class)

    predicted_negative = predicted.count(0)
    predicted_positive = predicted.count(1)

    if predicted_positive >= predicted_negative:
        return '1'
    else:
        return '0'

def predict_by_ID(model, test_folder):

    classes_name = os.listdir(test_folder)

    results = []
    for class_name in classes_name:
        print(f"Processing class {class_name}")

        class_folder = os.path.join(test_folder, class_name)
        patients_image_ID = os.listdir(class_folder)
        patient_number = len(patients_image_ID)

        print(f"Found {patient_number} patients in class {class_name}")

        progress = 0
        for patient_image_ID in patients_image_ID:
            patient_image_folder = os.path.join(class_folder, patient_image_ID)

            if os.path.isdir(patient_image_folder):
                progress += 1
                print(f"Processing {progress}/{patient_number}: {patient_image_ID}")

                predicted = predict_image_folder(model, patient_image_folder)
                results.append((class_name, predicted))
                
    metrics = calculate_metrics(results)
    return metrics

if __name__ == "__main__":
    test_folder = r"C:\Users\acer\Desktop\Project_TMJOA\Data\test_grouped-20250924T193516Z-1-001\test_grouped"
    model_path = r"C:\Users\acer\Desktop\Project_TMJOA\model\OA_multiview\runs\classify\MultiView_OA3\weights\best.pt"
    save_result_path = rf"{test_folder}\result.txt"

    model = YOLO(model_path)
    metrics = predict_by_ID(model, test_folder)
    cm = metrics['confusion_matrix']
    pprint.pprint(metrics)

    with open(save_result_path, 'w') as file:
        
        file.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        file.write(f"Precision: {metrics['precision']:.4f}\n")
        file.write(f"Recall: {metrics['recall']:.4f}\n")
        file.write(f"F1 score: {metrics['f1_score']:.4f}\n")
        file.write("           Predicted\n")
        file.write("           0       1\n")
        file.write(f"Actual 0  {cm['tn']:2d}      {cm['fp']:2d}\n")
        file.write(f"       1  {cm['fn']:2d}      {cm['tp']:2d}\n")

    print(f"Results saved to: {save_result_path}")