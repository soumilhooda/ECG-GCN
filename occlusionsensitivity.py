import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd

def occlusion_sensitivity(model, input_data, target_class, window_size, stride):
    # Get the original prediction
    original_prediction = model.predict(input_data)[0, 0]

    # Initialize an empty list to store the occlusion sensitivity values
    occlusion_sensitivity_values = []

    # Iterate through the input data with the specified stride
    for i in range(0, input_data.shape[1] - window_size + 1, stride):
        # Create a copy of the input data
        input_data_copy = input_data.copy()
        # Mask out a small window of the input data
        input_data_copy[:, i:i+window_size, :] = 0  # Replace the window with zeros
        # Compute the prediction after occluding the window
        occluded_prediction = model.predict(input_data_copy)[0, 0]
        # Compute the occlusion sensitivity value for this window
        occlusion_sensitivity_values.append(original_prediction - occluded_prediction)

    return np.array(occlusion_sensitivity_values)

def preprocess_data(data_path):
    try:
        df = pd.read_csv(data_path, header=None)
        data = df.to_numpy().astype(np.float32)

        # Scale the electrode data separately for each electrode
        scaler = StandardScaler()
        scaled_data = np.zeros_like(data)
        for i in range(data.shape[1]):
            scaled_data[:, i] = scaler.fit_transform(data[:, i].reshape(-1, 1)).flatten()

        scaled_data = scaled_data.reshape(1, 5000, 12)
        return scaled_data
    except Exception as e:
        print("Error loading data:", e)
        return None

def visualize_occlusion_heatmap(model, sample_data, target_class, window_size, stride, num_ecg_samples=None):
    num_leads = sample_data.shape[2]  # Get the number of leads (12 in your case)

    # # Normalize ECG data between -1 and 1
    # sample_data = 2 * (sample_data - np.min(sample_data)) / (np.max(sample_data) - np.min(sample_data)) - 1

    # Calculate occlusion sensitivity for the target class
    occlusion_sensitivity_values = occlusion_sensitivity(model, sample_data, target_class, window_size, stride)

    # Standardize occlusion sensitivity values
    scaler = StandardScaler()
    occlusion_sensitivity_values = scaler.fit_transform(occlusion_sensitivity_values.reshape(-1, 1)).flatten()

    # Limit ECG samples if num_ecg_samples is provided
    if num_ecg_samples is not None:
        sample_data = sample_data[:, :num_ecg_samples, :]

    # Create figure and subplots for each lead
    fig, axes = plt.subplots(num_leads, 1, figsize=(10, 8), sharex=True)

    # Set colormap based on target class
    cmap = 'plasma' if target_class == 0 else 'plasma'

# Plot each lead with occlusion sensitivity heatmap
    for i in range(num_leads):
        # Create color-mapped ECG line
        ecg_colors = plt.get_cmap(cmap)(occlusion_sensitivity_values)

        # Interpolate occlusion sensitivity values to match ECG signal length
        interp_occlusion_sensitivity = np.interp(np.linspace(0, 1, sample_data.shape[1]), 
                                                 np.linspace(0, 1, len(occlusion_sensitivity_values)), 
                                                 occlusion_sensitivity_values)
        ecg_colors = plt.get_cmap(cmap)(interp_occlusion_sensitivity)

        im = axes[i].scatter(range(sample_data.shape[1]), sample_data[0, :, i], c=ecg_colors, marker='.', s=2)  # Adjust marker size as needed

    # Add colorbar to the right of the plots
    cax = fig.add_axes([1.05, 0.1, 0.02, 0.8])  # Adjust position and size as needed
    cbar = fig.colorbar(im, cax=cax, orientation='vertical', label=f'Occlusion Sensitivity (Class {target_class})')


    # Set x-axis label
    axes[-1].set_xlabel('Time (ECG Samples)')

    plt.tight_layout()
    plt.show()

class_to_visualize = 0  
num_ecg_samples_to_plot = 5000  

# Path to the CSV file
csv_file_path = "/Users/soumilhooda/Desktop/ECG-Chapman/V2/ONN_Models/ECGDataDenoised/MUSE_20180209_172046_21000.csv"

# Preprocess the data
sample_data = preprocess_data(csv_file_path)
visualize_occlusion_heatmap(model, sample_data, target_class=class_to_visualize, window_size=25, stride=5, num_ecg_samples=num_ecg_samples_to_plot)