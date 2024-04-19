import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.utils import plot_model
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tf_keras_vis.gradcam import Gradcam
from sklearn.decomposition import PCA
import umap

np.random.seed(42)
tf.random.set_seed(42)

def save_model_architecture(model, model_name):
    plot_model(model, to_file=f"{model_name}_architecture.png", show_shapes=True)

early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

datasets = [

    {"name": "SR-AFIB", "train_data": "data/3-class_0vs1_train_data.npy", "train_labels": "data/3-class_0vs1_train_labels.npy",
     "test_data": "data/3-class_0vs1_test_data.npy", "test_labels": "data/3-class_0vs1_test_labels.npy", "num_classes": 2, "label_classes": [0, 1]},

    # {"name": "SR-AF", "train_data": "data/3-class_0vs2_train_data.npy", "train_labels": "data/3-class_0vs2_train_labels.npy",
    #  "test_data": "data/3-class_0vs2_test_data.npy", "test_labels": "data/3-class_0vs2_test_labels.npy", "num_classes": 2, "label_classes": [0, 1]},

    # {"name": "AFIB-AF", "train_data": "data/3-class_1vs2_train_data.npy", "train_labels": "data/3-class_1vs2_train_labels.npy",
    #  "test_data": "data/3-class_1vs2_test_data.npy", "test_labels": "data/3-class_1vs2_test_labels.npy", "num_classes": 2, "label_classes": [0, 1]},

    # {"name": "2-class-mixed", "train_data": "data/training_data_2class.npy", "train_labels": "data/training_labels_2class.npy",
    #  "test_data": "data/test_data_2class.npy", "test_labels": "data/test_labels_2class.npy", "num_classes": 2, "label_classes": [0, 1]},
     
    # {"name": "3-class-mixed", "train_data": "data/training_data_3class.npy", "train_labels": "data/training_labels_3class.npy",
    #  "test_data": "data/test_data_3class.npy", "test_labels": "data/test_labels_3class.npy", "num_classes": 3, "label_classes": [0, 1, 2]}
]

filter_combinations = [
    [32, 64, 128],
]


def calculate_and_plot_umap(data, labels, name, n_neighbors=15, min_dist=0.1):
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=0)
    embedding = reducer.fit_transform(data.reshape(data.shape[0], -1))

    plt.figure(figsize=(8, 6))
    for label_class in np.unique(labels):
        indices = np.where(labels == label_class)
        plt.scatter(embedding[indices, 0], embedding[indices, 1], label=label_class)
    plt.legend()
    plt.title(f"UMAP Visualization - {name}")
    plt.savefig(f"umap_{name}.png")
    plt.close()

def calculate_and_plot_pca(data, labels, name):
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data.reshape(data.shape[0], -1))

    plt.figure(figsize=(8, 6))
    for label_class in np.unique(labels):
        indices = np.where(labels == label_class)
        plt.scatter(data_pca[indices, 0], data_pca[indices, 1], label=label_class)
    plt.legend()
    plt.title(f"PCA Visualization - {name}")
    plt.savefig(f"pca_{name}.png")
    plt.close()

def plot_weight_histograms(model):
    """
    Plots weight histograms for all layers in the model.
    """
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv1D) or isinstance(layer, tf.keras.layers.Dense):
            weights = layer.get_weights()[0]
            plt.figure()
            plt.hist(weights.flatten(), bins=50)
            plt.xlabel("Weight Value")
            plt.ylabel("Frequency")
            plt.title(f"Weight Histogram - {layer.name}")
            plt.savefig(f"weights_histogram_{layer.name}.png")
            plt.close()


def calculate_and_plot_tsne(data, labels, name, perplexity=30):
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=0)
    data_embedded = tsne.fit_transform(data.reshape(data.shape[0], -1))

    plt.figure(figsize=(8, 6))
    for label_class in np.unique(labels):
        indices = np.where(labels == label_class)
        plt.scatter(data_embedded[indices, 0], data_embedded[indices, 1], label=label_class)
    plt.legend()
    plt.title(f"t-SNE Visualization - {name}")
    plt.savefig(f"tsne_{name}.png") 
    plt.close()  



def visualize_filters(model):
    """
    Visualizes the filters for all convolutional layers in the model.
    """
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv1D):
            filters, biases = layer.get_weights()

            # Normalize filter values to 0-1
            f_min, f_max = filters.min(), filters.max()
            filters = (filters - f_min) / (f_max - f_min)

            # Plot the filters
            n_filters, ix = filters.shape[-1], 1
            plt.figure(figsize=(10, 5))
            for i in range(n_filters):
                # Get the filter
                f = filters[:, :, i]
                # Plot each channel separately
                for j in range(filters.shape[1]):
                    ax = plt.subplot(n_filters, filters.shape[1], ix)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    plt.imshow(f[:, j], cmap='gray')
                    ix += 1
            plt.suptitle(f"Filters of Layer: {layer.name}")
            plt.savefig(f"filters_{layer.name}.png")
            plt.close()



def downsample_block(x, filters):
    x = layers.Conv1D(filters // 2, 1, strides=1, padding='same')(x)  # Use filters // 2
    x = advanced_pool_operator(x)
    return x

def advanced_nodal_operator(x, filters, kernel_size=5, activation='relu'):
    # Branch 1: Convolution with dilation
    y1 = layers.Conv1D(filters // 2, kernel_size, dilation_rate=2, padding='same')(x)
    y1 = layers.BatchNormalization()(y1)
    y1 = layers.Activation(activation)(y1)

    # Branch 2: Depthwise separable convolution
    y2 = layers.SeparableConv1D(filters // 2, kernel_size, padding='same')(x)
    y2 = layers.BatchNormalization()(y2)
    y2 = layers.Activation(activation)(y2)

    # Concatenate outputs
    y = layers.Concatenate()([y1, y2])
    return y

def advanced_pool_operator(x, pool_size=2, strides=1):
    # Mixed Pooling: Average and Max
    y1 = layers.AveragePooling1D(pool_size, strides, padding='same')(x)
    y2 = layers.MaxPooling1D(pool_size, strides, padding='same')(x)
    y = layers.Concatenate()([y1, y2])
    return y

def squeeze_and_excitation_block(x, ratio=16):
    num_channels = x.shape[-1]
    squeeze = layers.GlobalAveragePooling1D()(x)
    excitation = layers.Dense(num_channels // ratio, activation='relu')(squeeze)
    excitation = layers.Dense(num_channels, activation='sigmoid')(excitation)
    excitation = layers.Reshape((1, num_channels))(excitation)
    scale = layers.Multiply()([x, excitation])
    return scale

def residual_block_SE_RA_ONN(x, filters, kernel_size=5, downsample=False):
    y = advanced_nodal_operator(x, filters, kernel_size)
    y = advanced_nodal_operator(y, filters, kernel_size)

    if downsample:
        x = downsample_block(x, filters)  # Downsample x only once

    y = squeeze_and_excitation_block(y)  # Apply SE block to the nodal output

    # Advanced GOP: Attention-based weighting
    attention_weights = layers.Dense(1, activation='sigmoid')(x)
    gop_out = layers.Multiply()([attention_weights, y])  # Weighted output
    gop_out = layers.Add()([gop_out, x])  # Residual connection
    gop_out = layers.Activation('relu')(gop_out)
    return gop_out

def create_SE_RA_ONN(input_shape, num_classes, filters):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(filters[0], 5, strides=2, padding='same', activation='relu')(inputs)
    x = layers.MaxPooling1D(pool_size=3, strides=2, padding='same')(x)

    for f in filters[1:]: 
        x = residual_block_SE_RA_ONN(x, f, downsample=True)  
        x = residual_block_SE_RA_ONN(x, f)  
        x = residual_block_SE_RA_ONN(x, f)
        x = residual_block_SE_RA_ONN(x, f)

    x = layers.GlobalAveragePooling1D()(x)
    if(num_classes==2):
        outputs = layers.Dense(1, activation='sigmoid')(x)
    else:
        outputs = layers.Dense(3, activation='softmax')(x)
    model = models.Model(inputs, outputs)
    return model

from sklearn.metrics import confusion_matrix

def calculate_metrics(test_labels, test_predictions):
    cm = confusion_matrix(test_labels, test_predictions)
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    sensitivity = recall
    f1 = 2 * (precision * recall) / (precision + recall)
    return precision, recall, specificity, sensitivity, f1

def train_and_evaluate_model_SE_RA_ONN(model_fn, input_shape, num_classes, filters, training_data, training_labels, test_data, test_labels, label_classes, name):

    calculate_and_plot_tsne(training_data, training_labels, f"{model_name}_training", perplexity=30) 
    calculate_and_plot_tsne(test_data, test_labels, f"{model_name}_test", perplexity=30)
    calculate_and_plot_umap(training_data, training_labels, f"{model_name}_training")
    calculate_and_plot_umap(test_data, test_labels, f"{model_name}_test")
    calculate_and_plot_pca(training_data, training_labels, f"{model_name}_training")
    calculate_and_plot_pca(test_data, test_labels, f"{model_name}_test")

    model = model_fn(input_shape, num_classes, filters)
    if num_classes == 2:
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    else:
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print("Model Summary:")
    model.summary()
    history = model.fit(training_data, training_labels, epochs=5, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
    test_predictions = model.predict(test_data)
    test_loss, test_accuracy = model.evaluate(test_data, test_labels)
    print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

    # Get the loss function
    loss_fn = model.loss  

    # Grad-CAM visualization
    gradcam = Gradcam(model, model_modifier=None, clone=True)

    # Select the output class you want to visualize (adjust the index as needed)
    class_idx = 1  

    # Compute the loss for the chosen class
    loss = loss_fn(test_labels[0:1], test_predictions[0:1])[:, class_idx]

    cam = gradcam(loss, test_data[0:1])  # Visualize for first test sample
    cam = np.uint8(cam[0] * 255)
    plt.imshow(cam, alpha=0.5, cmap='jet')
    plt.imshow(test_data[0].reshape(5000, 1), alpha=0.5, cmap='gray')
    plt.title(f"Grad-CAM Visualization - {model_name}")
    plt.savefig(f"gradcam_{model_name}.png")
    plt.close()
    
    test_predictions_binary = (test_predictions > 0.5).astype(int)
    # Save confusion matrix
    cm = confusion_matrix(test_labels, test_predictions_binary)
    print("Confusion Matrix:")
    print(cm)
    np.save(f'confusion_matrix_{name}.npy', cm)

    # Calculate additional scores
    precision, recall, specificity, sensitivity, f1 = calculate_metrics(test_labels, test_predictions_binary)
    print(f'Precision: {precision}, Recall: {recall}, Specificity: {specificity}, Sensitivity: {sensitivity}, F1 Score: {f1}')

    model.save(f'SE_RA_ONN_model_{name}_savedmodel', save_format='tf')

    visualize_filters(model)
    plot_weight_histograms(model)
    
    return test_accuracy

results = {}
for dataset in datasets:
    training_data = np.load(dataset["train_data"])
    training_labels = np.load(dataset["train_labels"])
    test_data = np.load(dataset["test_data"])
    test_labels = np.load(dataset["test_labels"])
    
    for filters in filter_combinations:
        model_name = f"{dataset['name']}_{'_'.join(map(str, filters))}_filters"
        accuracy = train_and_evaluate_model_SE_RA_ONN(create_SE_RA_ONN, (5000, 12), dataset["num_classes"], filters, training_data, training_labels, test_data, test_labels, dataset["label_classes"],dataset["name"])
        results[model_name] = accuracy

with open("SE_RA_ONN_model_accuracies_12_Class.txt", "w") as f:
    for model_name, accuracy in results.items():
        f.write(f"{model_name}: {accuracy}\n")
