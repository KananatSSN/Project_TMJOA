import csv
import numpy as np
from sklearn.metrics import auc
import matplotlib.pyplot as plt
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
    
    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_sorted, tpr_sorted, 'b-', linewidth=2, label=f'ROC Curve (AUC = {auc_value:.3f})')
    plt.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random Classifier')
    
    # Fill area under curve
    plt.fill_between(fpr_sorted, tpr_sorted, alpha=0.2)
    
    # Formatting
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return auc_value
    
if __name__ == "__main__":
    data_path = r"C:\Users\acer\Desktop\Project_TMJOA\predictions.txt"
    TPR_list, FPR_list = calculate_TPR_AND_FPR(data_path, step_size=0.001)
    AUC = calculate_auc_and_plot(TPR_list, FPR_list, title="ROC Curve")
    print(AUC)