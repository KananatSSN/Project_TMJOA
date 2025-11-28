from ultralytics import YOLO
import os
import pprint
import csv

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
    
def get_prediction_results(model, testset_folder_path):

    result_from_images = []

    classes_name = os.listdir(testset_folder_path)
    for class_name in classes_name:
       
        class_folder_path = os.path.join(testset_folder_path, class_name)
        if not os.path.isdir(class_folder_path):
            continue

        print(f"Processing class {class_name}")

        patients_image_folder = os.listdir(class_folder_path)
        real = class_name
        for patient_image_folder in patients_image_folder:
            print(f"Processing patient {patient_image_folder}")
            patient_image_folder_path = os.path.join(class_folder_path, patient_image_folder)
            patient_id = patient_image_folder

            images_name = os.listdir(patient_image_folder_path)
            for image_name in images_name:

                image_path = os.path.join(patient_image_folder_path, image_name)

                if not os.path.exists(image_path):
                    print(f"Image path does not exist: {image_path}")
                    continue
                    
                results = model(image_path, verbose = False)
                result = results[0].probs

                if result :
                    (predicted_class, conf) = (result.top1, result.top1conf.item())
                    result_from_images.append((patient_id ,real, predicted_class, conf))

    save_path = rf"{testset_folder_path}\predictions.txt"
    with open(save_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write data
        for patient_id ,real, predicted_class, conf in result_from_images:
            writer.writerow([patient_id ,real, predicted_class, conf])

def predict_by_ID(model, test_folder, conf_threshold = 0.7):

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

                predicted = predict_image_folder(model, patient_image_folder, conf_threshold)
                results.append((class_name, predicted))
                
    metrics = calculate_metrics(results)
    return metrics

if __name__ == "__main__":

    split = "test"

    erosion_model_path = r"C:\Users\kanan\Desktop\Project_TMJOA\model\erosion_multiview\content\runs\classify\MultiView_erosion\weights\best.pt"
    erosion_data_path = rf"d:\Kananat\Data\training_dataset_2D\Multiview\erosion_multiview\{split}_grouped"

    flattenning_model_path = r"C:\Users\kanan\Desktop\Project_TMJOA\model\flattening_multiview\content\runs\classify\MultiView_flattening\weights\best.pt"
    flattenning_data_path = rf"d:\Kananat\Data\training_dataset_2D\Multiview\flattening_multiview\{split}_grouped"

    genSclerosis_model_path = r"C:\Users\kanan\Desktop\Project_TMJOA\model\genSclerosis_cont_multiview\content\runs\classify\genSclerosis_cont\weights\best.pt"
    genSclerosis_data_path = rf"d:\Kananat\Data\training_dataset_2D\Multiview\genSclerosis_multiview\{split}_grouped"

    OA_model_path = r"C:\Users\kanan\Desktop\Project_TMJOA\model\OA_multiview\runs\classify\MultiView_OA3\weights\best.pt"
    OA_data_path = rf"d:\Kananat\Data\training_dataset_2D\Multiview\OA_multiview\{split}_grouped"

    osteophyte_model_path = r"C:\Users\kanan\Desktop\Project_TMJOA\model\osteophyte_multiview\content\runs\classify\osteophyte\weights\best.pt"
    osteophyte_data_path = rf"d:\Kananat\Data\training_dataset_2D\Multiview\osteophyte_multiview\{split}_grouped"

    subCyst_model_path = r"C:\Users\kanan\Desktop\Project_TMJOA\model\subCyst_multiview\content\runs\classify\subCyst\weights\best.pt"
    subCyst_data_path = rf"d:\Kananat\Data\training_dataset_2D\Multiview\subCyst_multiview\{split}_grouped"

    erosion_path = (erosion_model_path, erosion_data_path)
    flattenning_path = (flattenning_model_path, flattenning_data_path)
    genSclerosis_path = (genSclerosis_model_path, genSclerosis_data_path)
    OA_path = (OA_model_path, OA_data_path)
    osteophyte_path = (osteophyte_model_path, osteophyte_data_path)
    subCyst_path = (subCyst_model_path, subCyst_data_path)

    best_conf = {
        'erosion_multiview': 0.84,
        'flattening_multiview': 0.7,
        'genSclerosis_multiview': 0.99,
        'OA_multiview': 0.79,
        'osteophyte_multiview': 0.86,
        'subCyst_multiview': 0.94
    }

    all_path = [erosion_path, flattenning_path, genSclerosis_path, OA_path, osteophyte_path, subCyst_path]

    # for model_path, data_path in all_path:
    #     print(f"Processing model: {data_path}")
    #     model = YOLO(model_path)
    #     get_prediction_results(model, data_path)

    for model_path, test_folder in all_path:
        save_result_path = rf"{test_folder}\result_best_conf.txt"

        dataset_name = os.path.basename(os.path.dirname(test_folder))
        conf_threshold = best_conf.get(dataset_name, 0.7)  # Default to 0.7 if not found

        model = YOLO(model_path)
        metrics = predict_by_ID(model, test_folder, conf_threshold=conf_threshold)
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