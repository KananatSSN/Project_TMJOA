import csv
import numpy as np
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import os
# Read as list of lists

def calculate_final_prediction(data, ID, conf_threshold = 0.7):
    predictions = [(x[2], x[3]) for x in data if x[0] == ID]

    filtered_by_conf = [(predicted_class, confidence) for predicted_class, confidence in predictions if float(confidence) >= conf_threshold]
    predicted_0 = sum(1 for predicted_class, confidence in filtered_by_conf if predicted_class=='0')
    predicted_1 = sum(1 for predicted_class, confidence in filtered_by_conf if predicted_class=='1')

    if predicted_1 >= predicted_0:
        return '1'
    else:
        return '0'
    
def calculate_TPR_FPR(result_list):

    TP, FP, TN, FN = 0, 0, 0, 0
    for real, predicted in result_list:
        if real == '1' and predicted =='1':
            TP += 1
        elif real == '0' and predicted == '1':
            FP += 1
        elif real == '0' and predicted == '0':
            TN += 1
        elif real == '1' and predicted == '0':
            FN += 1

    TPR = TP/(TP+FN) if (TP+FN) > 0 else 1.0
    FPR = FP/(FP+TN) if (FP+TN) > 0 else 0.0
    return TPR, FPR

def calculate_metrics_varying_conf(data_path, step_size = 0.01):

    if not os.path.isfile(data_path):
        print(f"File {data_path} does not exist.")
        return None
    
    save_path = os.path.join(os.path.dirname(data_path), 'metrics_varying_conf.txt')

    with open(data_path, 'r') as file:
        csv_reader = csv.reader(file)
        data = list(csv_reader)

    ID_list = list(set([(sublist[0], sublist[1]) for sublist in data]))
    conf_list = np.arange(0.5, 1+step_size, step_size).tolist()
    
    result_list = []
    for conf in conf_list:

        realANDpredict = []
        for ID, real in ID_list:
            predicted = calculate_final_prediction(data, ID, conf_threshold = conf)
            realANDpredict.append([real, predicted])

        TP, FP, TN, FN = 0, 0, 0, 0
        for real, predicted in realANDpredict:
            if real == '1' and predicted =='1':
                TP += 1
            elif real == '0' and predicted == '1':
                FP += 1
            elif real == '0' and predicted == '0':
                TN += 1
            elif real == '1' and predicted == '0':
                FN += 1

        accuracy, precision, recall, f1_score = calculate_binary_metrics(TP, FP, TN, FN)
        result_list.append([conf, accuracy, precision, recall, f1_score, TP, FP, TN, FN])
    
    with open(save_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(result_list)
    print(f"Metrics saved to {save_path}")

def calculate_metrics(data_path, conf):

    with open(data_path, 'r') as file:
        csv_reader = csv.reader(file)
        data = list(csv_reader)

    save_path = os.path.join(os.path.dirname(data_path), f'metrics_at_conf_{str(conf).replace(".", "_")}.txt')

    ID_list = list(set([(sublist[0], sublist[1]) for sublist in data]))

    realANDpredict = []
    for ID, real in ID_list:
        predicted = calculate_final_prediction(data, ID, conf_threshold=conf)
        realANDpredict.append([real, predicted])

    TP, FP, TN, FN = 0, 0, 0, 0
    for real, predicted in realANDpredict:
        if real == '1' and predicted =='1':
            TP += 1
        elif real == '0' and predicted == '1':
            FP += 1
        elif real == '0' and predicted == '0':
            TN += 1
        elif real == '1' and predicted == '0':
            FN += 1

    accuracy, precision, recall, f1_score = calculate_binary_metrics(TP, FP, TN, FN)

    with open(save_path, 'w') as file:
        file.write(f"Accuracy: {accuracy:.4f}\n")
        file.write(f"Precision: {precision:.4f}\n")
        file.write(f"Recall: {recall:.4f}\n")
        file.write(f"F1 score: {f1_score:.4f}\n")
        file.write("           Predicted\n")
        file.write("           0       1\n")
        file.write(f"Actual 0  {TN:2d}      {FP:2d}\n")
        file.write(f"       1  {FN:2d}      {TP:2d}\n")


def calculate_binary_metrics(tp, fp, tn, fn):
    """
    Calculate accuracy, precision, recall, and F1-score for binary classification.
    
    Args:
        tp (int): True Positives
        fp (int): False Positives  
        tn (int): True Negatives
        fn (int): False Negatives
        
    Returns:
        dict: Dictionary containing accuracy, precision, recall, and f1_score
    """
    
    # Calculate accuracy
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    
    # Calculate precision (handle division by zero)
    if tp + fp == 0:
        precision = 0.0
    else:
        precision = tp / (tp + fp)
    
    # Calculate recall (handle division by zero)
    if tp + fn == 0:
        recall = 0.0
    else:
        recall = tp / (tp + fn)
    
    # Calculate F1-score (handle division by zero)
    if precision + recall == 0:
        f1_score = 0.0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)
    
    return accuracy, precision, recall, f1_score

def calculate_TPR_AND_FPR(data_path, step_size = 0.01):
    
    with open(data_path, 'r') as file:
        csv_reader = csv.reader(file)
        data = list(csv_reader)

    ID_list = list(set([(sublist[0], sublist[1]) for sublist in data]))

    conf_list = np.arange(0, 1+step_size, step_size).tolist()
    
    TPR_list = []
    FPR_list = []

    for conf in conf_list:

        realANDpredict = []
        for ID, real in ID_list:
            predicted = calculate_final_prediction(data, ID, conf_threshold = conf)
            realANDpredict.append([real, predicted])

        TPR, FPR = calculate_TPR_FPR(realANDpredict)
        TPR_list.append(TPR)
        FPR_list.append(FPR)

    return TPR_list, FPR_list

def calculate_auc_and_plot(tpr_list, fpr_list, title="ROC Curve"):
    """
    Calculate AUC from TPR and FPR lists and plot the ROC curve.
    
    Parameters:
    tpr_list (list): List of True Positive Rate values
    fpr_list (list): List of False Positive Rate values  
    title (str): Title for the plot
    
    Returns:
    float: AUC value
    """
    
    # Convert to numpy arrays for easier manipulation
    fpr = np.array(fpr_list)
    tpr = np.array(tpr_list)
    
    # Sort by FPR to ensure proper order for AUC calculation
    sorted_indices = np.argsort(fpr)
    fpr_sorted = fpr[sorted_indices]
    tpr_sorted = tpr[sorted_indices]
    
    # Calculate AUC using trapezoidal rule
    auc_value = auc(fpr_sorted, tpr_sorted)
    
    # # Create the plot
    # plt.figure(figsize=(8, 6))
    # plt.plot(fpr_sorted, tpr_sorted, 'b-', linewidth=2, label=f'ROC Curve (AUC = {auc_value:.3f})')
    # plt.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random Classifier')
    
    # # Fill area under curve
    # plt.fill_between(fpr_sorted, tpr_sorted, alpha=0.2)
    
    # # Formatting
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate (FPR)')
    # plt.ylabel('True Positive Rate (TPR)')
    # plt.title(title)
    # plt.legend(loc="lower right")
    # plt.grid(True, alpha=0.3)
    # plt.show()
    
    return auc_value
    
if __name__ == "__main__":

    split = "test"

    erosion_data_path = rf"d:\Kananat\Data\training_dataset_2D\Multiview\erosion_multiview\{split}_grouped\predictions.txt"
    flattening_data_path = rf"d:\Kananat\Data\training_dataset_2D\Multiview\flattening_multiview\{split}_grouped\predictions.txt"
    genSclerosis_data_path = rf"d:\Kananat\Data\training_dataset_2D\Multiview\genSclerosis_multiview\{split}_grouped\predictions.txt"
    OA_data_path = rf"d:\Kananat\Data\training_dataset_2D\Multiview\OA_multiview\{split}_grouped\predictions.txt"
    osteophyte_data_path = rf"d:\Kananat\Data\training_dataset_2D\Multiview\osteophyte_multiview\{split}_grouped\predictions.txt"
    subCyst_data_path = rf"d:\Kananat\Data\training_dataset_2D\Multiview\subCyst_multiview\{split}_grouped\predictions.txt"

    best_conf = {
        'erosion_multiview': 0.84,
        'flattening_multiview': 0.7,
        'genSclerosis_multiview': 0.99,
        'OA_multiview': 0.79,
        'osteophyte_multiview': 0.86,
        'subCyst_multiview': 0.94
    }


    data_path_list = [erosion_data_path, flattening_data_path, genSclerosis_data_path, OA_data_path, osteophyte_data_path, subCyst_data_path]

    for data_path in data_path_list:

        if not os.path.isfile(data_path):
            print(f"File {data_path} does not exist. Skipping.")
            continue

        print(f"Processing file: {data_path}")
        work_dir = os.path.dirname(data_path)
        dataset_name = os.path.basename(os.path.dirname(work_dir))
        conf_threshold = best_conf.get(dataset_name, 0.7)  # Default to 0.7 if not found

        calculate_metrics(data_path, conf_threshold)




###########################################################################

        # # Calculate metrics at varying confidence thresholds
        # print(f"Processing file: {data_path}")
        # calculate_metrics_varying_conf(data_path, step_size=0.01)

###########################################################################

        # # AUC calculation and plotting
        # work_dir = os.path.dirname(data_path)
        # save_path = os.path.join(work_dir, 'result.txt')

        # TPR_list, FPR_list = calculate_TPR_AND_FPR(data_path, step_size=0.001)
        # AUC = calculate_auc_and_plot(TPR_list, FPR_list, title="ROC Curve")
        # print(AUC)

        # with open(save_path, 'a') as file:
        #     file.write(f'AUC: {AUC}\n')
    