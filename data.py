import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ECG_DATA_FOLDER = "ReducedECGDataDenoised"
LABELS_FILE = "3Class.csv"

scaler = StandardScaler()

labels_df = pd.read_csv(LABELS_FILE)
file_names = labels_df['FileName'].values

subjects_data = []
labels = []

# Map string labels to integers
label_mapping = {'SR': 0, 'AFIB': 1, 'AF': 2}  

for file_name in file_names:
    subject_data_list = []
    for electrode in range(12):
        file_path = os.path.join(ECG_DATA_FOLDER, file_name + '.csv')
        subject_data = pd.read_csv(file_path, header=None)
        electrode_data = subject_data.iloc[:, electrode].values
        
        # Scale the electrode data using StandardScaler
        scaled_electrode_data = scaler.fit_transform(electrode_data.reshape(-1, 1)).flatten()
        subject_data_list.append(scaled_electrode_data)
    
    # Stack the electrode data for the current subject to shape [samples, electrodes]
    subject_data_array = np.stack(subject_data_list, axis=-1)
    subjects_data.append(subject_data_array)
    
    # Assuming one label per file, get the label for the current subject
    label = labels_df[labels_df['FileName'] == file_name]['Rhythm'].values[0]
    mapped_label = label_mapping[label]
    labels.append(mapped_label)

# Convert lists to arrays
subjects_data_array = np.array(subjects_data)  # This will have shape [subjects, samples, electrodes]
labels_array = np.array(labels)  # This will have shape [subjects]

# Splitting the data into training and testing sets
subjects = len(file_names)
tt = int(subjects * 0.8)  # 80% for training

training_data = subjects_data_array[:tt]
training_labels = labels_array[:tt]
test_data = subjects_data_array[tt:]
test_labels = labels_array[tt:]

# Save the arrays
np.save("training_data_3class", training_data)
np.save("training_labels_3class", training_labels)
np.save("test_data_3class", test_data)
np.save("test_labels_3class", test_labels)

print('[info] Training and testing datasets are ready!')

# Load the training data and labels

training_data = np.load("/Users/soumilhooda/Desktop/ECG-Chapman/lstm-rcnn/training_data_3class.npy")
training_labels = np.load("/Users/soumilhooda/Desktop/ECG-Chapman/lstm-rcnn/training_labels_3class.npy")

# Load the test data and labels
test_data = np.load("/Users/soumilhooda/Desktop/ECG-Chapman/lstm-rcnn/test_data_3class.npy")
test_labels = np.load("/Users/soumilhooda/Desktop/ECG-Chapman/lstm-rcnn/test_labels_3class.npy")

# Get statistics for training and test data
def get_statistics(data, labels, data_type):
    num_samples = data.shape[0]
    num_features = data.shape[1]
    num_classes = len(np.unique(labels))
    class_counts = {label: np.sum(labels == label) for label in np.unique(labels)}
    mean_features = np.mean(data, axis=0)
    std_features = np.std(data, axis=0)
    
    print(f"Statistics for {data_type} data:")
    print(f"Number of samples: {num_samples}")
    print(f"Number of features: {num_features}")
    print(f"Number of classes: {num_classes}")
    print("Class distribution:")
    for label, count in class_counts.items():
        print(f"- Class {label}: {count} samples")
    print(f"Mean of features: {mean_features}")
    print(f"Standard deviation of features: {std_features}")

# Generate statistics for training data
get_statistics(training_data, training_labels, "training")

# Generate statistics for test data
get_statistics(test_data, test_labels, "test")


