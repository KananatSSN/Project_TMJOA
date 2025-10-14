import pandas as pd
import numpy as np

# Read the CSV file
df = pd.read_csv(r'C:\Users\acer\Desktop\Project_TMJOA\Data\Labels.csv')

# Extract patient ID from the id column
# Assumes format: 'patient_id_L/R_Year' or 'patient_id_L/R'
df['patient_id'] = df['ID'].str.split('_').str[0]

# Get unique patient IDs
unique_patients = df['patient_id'].unique()

# Shuffle patient IDs for random assignment
np.random.seed(42)  # Set seed for reproducibility
shuffled_patients = np.random.permutation(unique_patients)

# Calculate split indices
n_patients = len(shuffled_patients)
train_end = int(0.7 * n_patients)
val_end = int(0.9 * n_patients)

# Assign patients to splits
train_patients = set(shuffled_patients[:train_end])
val_patients = set(shuffled_patients[train_end:val_end])
test_patients = set(shuffled_patients[val_end:])

# Create split column based on patient ID
df['split'] = df['patient_id'].apply(
    lambda pid: 'train' if pid in train_patients 
    else ('val' if pid in val_patients else 'test')
)

# Optional: Remove the temporary patient_id column if you don't need it
df = df.drop('patient_id', axis=1)

# Save the result
df.to_csv('Labels_with_split.csv', index=False)

# Print split statistics
print("Split distribution:")

print("\nPatient distribution:")
print(f"Train patients: {len(train_patients)}")
print(f"Val patients: {len(val_patients)}")
print(f"Test patients: {len(test_patients)}")