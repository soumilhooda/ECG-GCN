import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
import heartpy as hp
from umap import UMAP  # Import UMAP library



sampling_rate = 500

# Load model
model_path = '/Users/soumilhooda/Desktop/ECG-Chapman/SERA-AwGOP/SE_RA_ONN_model_SR-AFIB_savedmodel'
try:
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    print("Error loading the model:", e)

# Define a function to preprocess data
def preprocess_data(data_path):
    try:
        df = pd.read_csv(data_path, header=None)
        data = df.to_numpy().astype(np.float32)

        # Scale the electrode data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data.reshape(-1, 1)).flatten()
        data = scaled_data.reshape(1, 5000, 12)
        return data
    except Exception as e:
        print("Error loading data:", e)
        return None

def get_hrv_metrics(peaks):
    """
    Calculates basic Heart Rate Variability (HRV) metrics from R-peak indices.

    Args:
        peaks (np.array): A NumPy array containing the indices of the detected R-peaks.

    Returns:
        dict: A dictionary containing the calculated HRV metrics.
    """

    # Calculate RR intervals
    rr_intervals = np.diff(peaks)  # Time between successive R-peaks
    rr_intervals_ms = rr_intervals * 1000 / sampling_rate  # Convert to milliseconds (assuming a sampling rate)

    # Time-domain metrics
    hrv_metrics = {
        "mean_rr": np.mean(rr_intervals_ms),  # Mean RR interval
        "sdnn": np.std(rr_intervals_ms),  # Standard deviation of RR intervals
        "rmssd": np.sqrt(np.mean(np.square(np.diff(rr_intervals_ms)))),  # Root mean square of successive differences
        "nn50": nn50(rr_intervals_ms),  # Number of pairs of successive RR intervals differing by more than 50ms
        "pnn50": pnn50(rr_intervals_ms)  # Proportion of NN50 divided by total number of RR intervals
    }

    # Frequency-domain metrics (using Lomb-Scargle periodogram)
    try:
        from astropy.timeseries import LombScargle
        frequency, power = LombScargle(peaks, np.ones_like(peaks)).autopower(minimum_frequency=0.04,
                                                                            maximum_frequency=0.5)
        lf_band = (0.04, 0.15)  # Low-frequency band
        hf_band = (0.15, 0.4)  # High-frequency band
        lf_power = np.trapz(power[(frequency >= lf_band[0]) & (frequency <= lf_band[1])], frequency[(frequency >= lf_band[0]) & (frequency <= lf_band[1])])
        hf_power = np.trapz(power[(frequency >= hf_band[0]) & (frequency <= hf_band[1])], frequency[(frequency >= hf_band[0]) & (frequency <= hf_band[1])])
        hrv_metrics.update({
            "lf": lf_power,  # Power in the low-frequency band
            "hf": hf_power,  # Power in the high-frequency band
            "lf_hf_ratio": lf_power / hf_power  # Ratio of LF to HF power
        })
    except ImportError:
        print("Warning: 'astropy' library not found, frequency-domain HRV metrics not calculated.")

    return hrv_metrics

def nn50(rr_intervals):
    """
    Calculates the number of pairs of successive RR intervals that differ by more than 50ms.
    """
    differences = np.abs(np.diff(rr_intervals))
    return np.sum(differences > 50)

def pnn50(rr_intervals):
    """
    Calculates the proportion of NN50 (number of pairs of successive RR intervals differing by more than 50ms) 
    divided by the total number of RR intervals.
    """
    return nn50(rr_intervals) / len(rr_intervals)

# Set page title and description
st.set_page_config(page_title="ECG Classification")

# Title
st.title("ECG Classification: SR vs AFIB")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")


# Calculate HRV metrics and display results
if uploaded_file is not None:
    try:
        # Read CSV data
        df = pd.read_csv(uploaded_file, header=None)
        data = df.to_numpy().astype(np.float32)

        # Scale the electrode data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data.reshape(-1, 1)).flatten()
        data = scaled_data.reshape(1, 5000, 12)

        # Display progress bar during prediction
        with st.spinner("Classifying ECG..."):
            # Run prediction
            predictions = model(data)
            predictions = predictions.numpy()

            # Apply threshold and convert to integers
            predictions = (predictions > 0.5).astype(int)

            # Map predictions to labels
            prediction_label = "Atrial Fibrillation (AFIB)" if predictions[0][0] == 1 else "Sinus Rhythm (SR)"

        # Display prediction with styling
        st.markdown(f"<h3 style='color: green;'>Prediction: {prediction_label}</h3>", unsafe_allow_html=True)

        # Get R-peak indices
        rpeaks = hp.process(hp.scale_data(df.iloc[:, 0]), sample_rate=500)[0]['peaklist']

        # Calculate HRV metrics
        hrv_metrics = get_hrv_metrics(rpeaks)

        # Display HRV metrics
        st.markdown("---")
        st.subheader("HRV Metrics")
        st.write(f"Mean RR Interval: {hrv_metrics['mean_rr']:.2f} ms")
        st.write(f"Standard deviation of RR intervals (SDNN): {hrv_metrics['sdnn']:.2f} ms")
        st.write(f"Root mean square of successive differences (RMSSD): {hrv_metrics['rmssd']:.2f} ms")
        st.write(f"Number of pairs of successive RR intervals differing by more than 50ms (NN50): {hrv_metrics['nn50']}")
        st.write(f"Proportion of NN50 divided by total number of RR intervals (pNN50): {hrv_metrics['pnn50']:.2f}")
        st.write(f"Power in the low-frequency band (LF): {hrv_metrics['lf']:.2f}")
        st.write(f"Power in the high-frequency band (HF): {hrv_metrics['hf']:.2f}")
        st.write(f"LF/HF Ratio: {hrv_metrics['lf_hf_ratio']:.2f}")
        st.markdown("---")

        # Create subplots for ECG plot and prediction
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot ECG signals
        for i in range(12):
            ax1.plot(df.iloc[:, i], label=f"Channel {i+1}", linewidth=0.8)
        ax1.set_title("ECG Signals")
        ax1.set_xlabel("Sample")
        ax1.set_ylabel("Amplitude")
        ax1.legend(fontsize=8)

        # Display prediction as text in the second subplot
        ax2.text(0.5, 0.5, prediction_label, ha='center', va='center', size=20)
        ax2.axis('off')

        # Display the plot in Streamlit
        st.pyplot(fig)


    except Exception as e:
        st.error(f"Error processing data: {e}")
else:
    st.write("Please upload a CSV file containing ECG data.")
