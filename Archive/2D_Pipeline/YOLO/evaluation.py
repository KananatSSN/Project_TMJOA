import os
import glob
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO

def evaluate_patient_dataset(model, dataset_path, threshold=0.5, top_n=None):
    """
    Evaluate model on patient dataset with hierarchical structure.
    
    Args:
        model: Your trained model
        dataset_path (str): Path to dataset folder containing class folders (0, 1)
        threshold (float): Threshold for converting probabilities to predictions
        top_n (int): If specified, only use top n most confident predictions for aggregation.
                    If None, use all predictions.
    
    Returns:
        dict: Comprehensive evaluation results
    """
    dataset_path = Path(dataset_path)
    
    # Store results
    patient_results = []
    true_labels = []
    predicted_labels = []
    
    # Process each class folder
    for class_folder in ['0', '1']:
        class_path = dataset_path / class_folder
        
        if not class_path.exists():
            print(f"Warning: Class folder {class_folder} not found")
            continue
        
        true_class = int(class_folder)
        
        # Process each patient folder within the class
        for patient_folder in class_path.iterdir():
            if not patient_folder.is_dir():
                continue
                
            patient_name = patient_folder.name
            
            # Get all PNG images for this patient
            image_paths = list(patient_folder.glob("*.png"))
            
            if not image_paths:
                print(f"Warning: No PNG images found for patient {patient_name}")
                continue
            
            # Convert to string paths
            image_paths_str = [str(path) for path in image_paths]
            
            # Get predictions for all images of this patient
            results = model(image_paths_str, verbose=False)
            
            # Collect predictions with confidence scores
            predictions_with_confidence = []
            all_probs = []
            
            for i, result in enumerate(results):
                probs = result.probs.data.cpu().numpy()
                all_probs.append(probs)
                
                # Individual image prediction
                pred_class = 1 if probs[1] > threshold else 0
                
                # Confidence is the max probability (most confident prediction)
                confidence = max(probs[0], probs[1])
                
                predictions_with_confidence.append({
                    'image_index': i,
                    'image_path': image_paths_str[i],
                    'prediction': pred_class,
                    'confidence': confidence,
                    'probabilities': probs
                })
            
            # Sort by confidence (descending) and take top n
            predictions_with_confidence.sort(key=lambda x: x['confidence'], reverse=True)
            
            if top_n is not None and top_n > 0:
                selected_predictions = predictions_with_confidence[:min(top_n, len(predictions_with_confidence))]
                print(f"Patient {patient_name}: Using top {len(selected_predictions)} out of {len(predictions_with_confidence)} images")
            else:
                selected_predictions = predictions_with_confidence
            
            # Aggregate selected predictions for this patient
            class_predictions = [pred['prediction'] for pred in selected_predictions]
            selected_probs = [pred['probabilities'] for pred in selected_predictions]
            
            # Majority vote for patient (from selected predictions only)
            class_1_votes = sum(1 for pred in class_predictions if pred == 1)
            class_0_votes = len(class_predictions) - class_1_votes
            
            patient_prediction = 1 if class_1_votes > class_0_votes else 0
            
            # Calculate patient-level confidence based on selected predictions
            winning_votes = max(class_0_votes, class_1_votes)
            patient_confidence = winning_votes / len(class_predictions)
            
            # Average probabilities across selected images only
            avg_probs_selected = np.mean(selected_probs, axis=0)
            
            # Average probabilities across all images (for comparison)
            avg_probs_all = np.mean(all_probs, axis=0)
            
            # Store results
            patient_info = {
                'patient_name': patient_name,
                'true_class': true_class,
                'predicted_class': patient_prediction,
                'confidence': patient_confidence,
                'num_images_total': len(image_paths),
                'num_images_used': len(selected_predictions),
                'class_0_votes': class_0_votes,
                'class_1_votes': class_1_votes,
                'avg_prob_class_0_selected': avg_probs_selected[0],
                'avg_prob_class_1_selected': avg_probs_selected[1],
                'avg_prob_class_0_all': avg_probs_all[0],
                'avg_prob_class_1_all': avg_probs_all[1],
                'selected_predictions': selected_predictions,
                'all_predictions': predictions_with_confidence
            }
            
            patient_results.append(patient_info)
            true_labels.append(true_class)
            predicted_labels.append(patient_prediction)
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='binary')
    recall = recall_score(true_labels, predicted_labels, average='binary')
    f1 = f1_score(true_labels, predicted_labels, average='binary')
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    
    # Create results summary
    results_summary = {
        'patient_results': patient_results,
        'metrics': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'num_patients': len(patient_results)
        },
        'confusion_matrix': cm,
        'true_labels': true_labels,
        'predicted_labels': predicted_labels
    }
    
    return results_summary

def print_evaluation_results(results):
    """
    Print formatted evaluation results.
    """
    print("=" * 60)
    print("PATIENT DATASET EVALUATION RESULTS")
    print("=" * 60)
    
    metrics = results['metrics']
    print(f"Number of patients: {metrics['num_patients']}")
    print(f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']:.2%})")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    
    print("\nConfusion Matrix:")
    cm = results['confusion_matrix']
    print(f"                Predicted")
    print(f"                0    1")
    print(f"Actual   0    {cm[0,0]:3d}  {cm[0,1]:3d}")
    print(f"         1    {cm[1,0]:3d}  {cm[1,1]:3d}")
    
    # Show some patient details
    print("\nPatient Results Summary:")
    correct_predictions = 0
    for patient in results['patient_results']:
        status = "✓" if patient['true_class'] == patient['predicted_class'] else "✗"
        if patient['true_class'] == patient['predicted_class']:
            correct_predictions += 1
        
        print(f"{status} {patient['patient_name']:15} | True: {patient['true_class']} | "
              f"Pred: {patient['predicted_class']} | Conf: {patient['confidence']:.2f} | ")

def plot_confusion_matrix(results, save_path=None):
    """
    Plot confusion matrix with better visualization.
    """
    cm = results['confusion_matrix']
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Class 0', 'Class 1'],
                yticklabels=['Class 0', 'Class 1'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def create_results_dataframe(results):
    """
    Create a pandas DataFrame with patient results for further analysis.
    """
    df = pd.DataFrame(results['patient_results'])
    df['correct_prediction'] = df['true_class'] == df['predicted_class']
    return df

def analyze_top_n_effect(model, dataset_path, n_values, threshold=0.5):
    """
    Analyze the effect of using different top_n values for aggregation.
    
    Args:
        model: Your trained model
        dataset_path (str): Path to dataset folder
        n_values (list): List of top_n values to test (e.g., [1, 3, 5, 10, None])
        threshold (float): Threshold for predictions
    
    Returns:
        dict: Results for each top_n value
    """
    results_comparison = {}
    
    for n in n_values:
        print(f"\n{'='*50}")
        print(f"Evaluating with top_n = {n if n is not None else 'All'}")
        print(f"{'='*50}")
        
        results = evaluate_patient_dataset(model, dataset_path, threshold=threshold, top_n=n)
        results_comparison[n] = results
        
        # Print summary
        metrics = results['metrics']
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
    
    return results_comparison

def plot_top_n_comparison(results_comparison):
    """
    Plot comparison of different top_n values.
    """
    n_values = []
    accuracies = []
    f1_scores = []
    precisions = []
    recalls = []
    
    for n, results in results_comparison.items():
        n_label = str(n) if n is not None else 'All'
        n_values.append(n_label)
        metrics = results['metrics']
        accuracies.append(metrics['accuracy'])
        f1_scores.append(metrics['f1_score'])
        precisions.append(metrics['precision'])
        recalls.append(metrics['recall'])
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    ax1.bar(n_values, accuracies, color='skyblue')
    ax1.set_title('Accuracy by Top N')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1)
    
    ax2.bar(n_values, f1_scores, color='lightgreen')
    ax2.set_title('F1-Score by Top N')
    ax2.set_ylabel('F1-Score')
    ax2.set_ylim(0, 1)
    
    ax3.bar(n_values, precisions, color='lightcoral')
    ax3.set_title('Precision by Top N')
    ax3.set_ylabel('Precision')
    ax3.set_ylim(0, 1)
    
    ax4.bar(n_values, recalls, color='lightyellow')
    ax4.set_title('Recall by Top N')
    ax4.set_ylabel('Recall')
    ax4.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.show()

# Example usage:
if __name__ == "__main__":
    # Assuming you have your model loaded
    model = YOLO(r'C:\Users\kanan\Desktop\Project_TMJOA\2D_Pipeline\YOLO\runs\classify\OA_model_train\weights\best.pt')  # Load the classification model
    test_set = r"D:\Kananat\Data\raw_Data_and_extra\Open access data\Baseline\Baseline_dataset\all_2D_patientwise"

    # Single evaluation with top 3 most confident predictions
    results = evaluate_patient_dataset(model, test_set, top_n=10)
    print_evaluation_results(results)
    
    # Compare different top_n values
    # n_values = [1, 3, 5, 10, None]  # None means use all predictions
    # comparison = analyze_top_n_effect(model, 'path/to/your/dataset', n_values)
    # plot_top_n_comparison(comparison)
    
    pass