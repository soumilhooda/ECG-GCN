import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.io import loadmat
import csv

ECG_DATA_FOLDER = "ReducedECGDataDenoised"
LABELS_FILE = "SelectedRows_Modified.csv"

# Read the labels file to get the list of file names
labels_df = pd.read_csv(LABELS_FILE)
file_names = labels_df['FileName'].values

# Initialize an empty list to store data for each electrode
electrode_data = []

# Loop through each electrode (assuming 12 electrodes)
for electrode in range(12):
    # Initialize an empty list to store data for this electrode from all subjects
    all_subjects_data = []

    # Loop through each subject
    for file_name in file_names:
        # Construct the file path and read the CSV
        file_path = os.path.join(ECG_DATA_FOLDER, file_name + '.csv')
        subject_data = pd.read_csv(file_path, header=None)

        # Append the data for the current electrode
        all_subjects_data.append(subject_data.iloc[:, electrode].values)

    # Concatenate data for all subjects for this electrode
    concatenated_data = np.concatenate(all_subjects_data)
    electrode_data.append(concatenated_data)

# Stack the electrode data to create the final array
final_data = np.stack(electrode_data, axis=0)

#%%Normalize
xx= np.array(final_data)
NormalizedAll = xx - xx.min();
NormalizedAll = NormalizedAll / xx.max();
NormalizedAll=NormalizedAll.reshape(12,20255000)
print('[info]  Normalised')

#%%Covariance
print('[info] Calculating Covariance matrix')
covariance_matrix = np.cov(NormalizedAll);
print('[info] covariance of Normalized/Standardize data is calculated')
np.savetxt("foo.csv", covariance_matrix)
plt.style.use('seaborn-poster')
plt.imshow(covariance_matrix,extent=[0,12, 0, 12],cmap='viridis')
#%% Pearson matrix and its ABS
print('[info] Calculating Pearson matrix')
Pearson_matrix= np.corrcoef(NormalizedAll)
np.savetxt("Pearson_matrix.csv", Pearson_matrix)
print('[info] Pearson matrix of Normalized/Standardize data is calculated')
print('[info] Calculating Absolute Pearson matrix')
Absolute_Pearson_matrix = abs(Pearson_matrix);
np.savetxt("Absolute_Pearson_matrix.csv", Absolute_Pearson_matrix)
print('[info] Absolute Pearson matrix Calculated')
plt.figure()
plt.imshow(Pearson_matrix,extent=[0,12, 0, 12],cmap='viridis')
plt.figure()
plt.imshow(Absolute_Pearson_matrix,extent=[0,12, 0, 12],cmap='viridis')
#%% Adjacency Matrix
print('[info] Calculating Adjacency Matrix')
Eye_Matrix = np.eye(12, 12);
Adjacency_Matrix = Absolute_Pearson_matrix - Eye_Matrix;
np.save("Adjacency_Matrix", Adjacency_Matrix)
print('[info] Adjacency Matrix is Calculated')
plt.figure()
plt.imshow(Adjacency_Matrix,extent=[0,12,0, 12],cmap='viridis')
#%% Degree Matrix
print('[info] Calculating Degree Matrix')
diagonal_vector = np.sum(Adjacency_Matrix,axis=1)
Degree_Matrix = np.diag(diagonal_vector)
np.savetxt("Degree_Matrix.csv", Degree_Matrix)
print('[info] Degree Matrix Calculated')
plt.figure()
plt.imshow(Degree_Matrix,extent=[0,12,0, 12],cmap='viridis')
#%% Laplacian Matrix
print('[info] Calculating Laplacian Matrix')
Laplacian_Matrix = Degree_Matrix - Adjacency_Matrix;
np.savetxt("Laplacian_Matrix.csv", Laplacian_Matrix)
print('[info] Laplacian Matrix Calculated ')
plt.figure()
plt.imshow(Laplacian_Matrix,extent=[0,12,0, 12],cmap='viridis')
#%%Labels
print('[info] Creating Label matrix ')
labels_df = pd.read_csv("SelectedRows_Modified.csv")
rhythms = labels_df['Rhythm'].values
# Map string labels to integers
label_mapping = {'AR': 0, 'SR': 1}  
mapped_labels = np.array([label_mapping[rhythm] for rhythm in rhythms])
# Repeat each label 5000 times for each sample
repeated_labels = np.repeat(mapped_labels, 5000)
# Reshape the labels to the desired shape
final_labels = repeated_labels.reshape(-1, 1)
print('[info] Labels are prepared ')
#%% 
# data: 20255000X12
# labels: 20255000x1
# Check and reshape NormalizedAll if needed
if NormalizedAll.shape != (20255000, 12):
    NormalizedAll = NormalizedAll.reshape(20255000, 12)
# Check and reshape final_labels if needed
if final_labels.shape != (20255000, 1):
    final_labels = final_labels.reshape(20255000, 1)
print('[info] Train/Test split and ready to run! ')
DATA_ALL = np.append(NormalizedAll,final_labels,axis=1)
rowrank =np.random.permutation(20255000)
All_of_Dataset = DATA_ALL[rowrank, :]
row=20255000
#%%
tt=int(np.fix(row/10*8))
training_set   = All_of_Dataset[0:tt , 0:12];
training_label = All_of_Dataset[0:tt , 12];
test_set       = All_of_Dataset[tt:, 0:12];
test_label     = All_of_Dataset[tt:, 12];
np.save("training_set", training_set)
np.save("training_label", training_label)
np.save("test_set", test_set)
np.save("test_label", test_label)

print('[info] Everything is ready now! ')