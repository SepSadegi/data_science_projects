import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import backend as K

# Configure matplotlib
mpl.rc('xtick', labelsize=16)
mpl.rc('ytick', labelsize=16)
mpl.rcParams.update({'font.size': 15})
plt.ioff()

# Define the path to the dataset
data_directory = '.../chest_X-ray_deep_learning'
image_dir = data_directory + '/images-small/'

# Flags for plotting sections
plot_image = False

# Load training,validation, and test datasets
train_df = pd.read_csv(os.path.join(data_directory, 'train-small.csv')) # 875 images used for training
valid_df = pd.read_csv(os.path.join(data_directory, 'valid-small.csv')) # 109 images used for validation
test_df = pd.read_csv(os.path.join(data_directory, 'test.csv')) # 420 images used for testing

# We are interested in 5 out of 14 pathologies
labels = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema', 'Fibrosis',
          'Hernia', 'Infiltration', 'Mass', 'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']

# Preventing Data Leakage between the train, validation, and test datasets.
def check_for_leakage(df1, df2, patient_col):
    """
    Return True if there any patients are in both df1 and df2.

    Args:
    :param df1 (dataframe): dataframe describing first dataset
    :param df2 (dataframe): dataframe describing second dataset
    :param patient_col (str): string name of column with patient IDs

    :return: leakage (bool): True if there is leakage, otherwise False
    """
    # Extract patients IDs and convert to sets
    df1_patients_unique = set(df1[patient_col])
    df2_patients_unique = set(df2[patient_col])

    # Find intersection of patients IDs
    patients_in_both_groups = df1_patients_unique.intersection(df2_patients_unique)

    # Return True if there is any overlap
    leakage = len(patients_in_both_groups) > 0

    return leakage

print("leakage between train and valid: {}".format(check_for_leakage(train_df, valid_df, 'PatientId')))
print("leakage between train and test: {}".format(check_for_leakage(train_df, test_df, 'PatientId')))
print("leakage between valid and test: {}".format(check_for_leakage(valid_df, test_df, 'PatientId')))

def get_train_generator(df,
                        image_dir,
                        x_col,
                        y_cols,
                        shuffle=True,
                        batch_size=8,
                        seed=1,
                        target_w=320,
                        target_h=320):
    """
        Return generator for training set, normalizing using batch
        statistics.

        Args:
          df (dataframe): dataframe specifying training data.
          image_dir (str): directory where image files are held.
          x_col (str): name of column in df that holds filenames.
          y_cols (list): list of strings that hold y labels for images.
          batch_size (int): images per batch to be fed into model during training.
          seed (int): random seed.
          target_w (int): final width of input images.
          target_h (int): final height of input images.

        Returns:
            train_generator (DataFrameIterator): iterator over training set
        """

    print("getting train generator ...")
    # Normalize images
    image_generator = ImageDataGenerator(
        samplewise_center=True,
        samplewise_std_normalization=True
    )

    # Flow from directory with specified batch size and target image size
    generator = image_generator.flow_from_dataframe(
        dataframe=df,
        directory=image_dir,
        x_col=x_col,
        y_col=y_cols,
        class_mode="raw",
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        target_size=(target_w, target_h)
    )

    return generator

# Build a separate generator for valid and test sets
# A separate generator is needed because we cannot use batch statistic on validation and test data
def get_test_and_valid_generator(valid_df,
                                 test_df,
                                 train_df,
                                 image_dir,
                                 x_col,
                                 y_cols,
                                 sample_size=100,
                                 batch_size=8,
                                 seed=1,
                                 target_w=320,
                                 target_h=320):
    """
        Return generators for validation set and test set using
        normalization statistics from training set.

        Args:
          valid_df (dataframe): dataframe specifying validation data.
          test_df (dataframe): dataframe specifying test data.
          train_df (dataframe): dataframe specifying training data.
          image_dir (str): directory where image files are held.
          x_col (str): name of column in df that holds filenames.
          y_cols (list): list of strings that hold y labels for images.
          sample_size (int): size of sample to use for normalization statistics.
          batch_size (int): images per batch to be fed into model during training.
          seed (int): random seed.
          target_w (int): final width of input images.
          target_h (int): final height of input images.

        Returns:
            test_generator (DataFrameIterator) and valid_generator: iterators over test set and validation set respectively
    """
    print("getting train and valid generators ...")

    # Get generator to sample dataset
    raw_train_generator = ImageDataGenerator().flow_from_dataframe(
        dataframe=train_df,
        directory=image_dir,
        x_col=x_col, # "Image"
        y_col=y_cols, # labels
        class_mode="raw",
        batch_size=sample_size,
        shuffle=True,
        target_size=(target_w, target_h)
    )

    # Get a sample batch to fit normalization statistics
    try:
        batch = next(iter(raw_train_generator))
    except StopIteration:
        raise ValueError("Generator yielded no data")

    data_sample = batch[0]  # Get the first batch of images

    # Use sample to fit mean and std for test set generator
    image_generator = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True
    )

    # Fit generator to sample from training data
    image_generator.fit(data_sample)

    # Get test generator
    valid_generator = image_generator.flow_from_dataframe(
        dataframe=valid_df,
        directory=image_dir,
        x_col=x_col,
        y_col=y_cols,
        class_mode="raw",
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        target_size=(target_w, target_h)
    )

    test_generator = image_generator.flow_from_dataframe(
        dataframe=test_df,
        directory=image_dir,
        x_col=x_col,
        y_col=y_cols,
        class_mode="raw",
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        target_size=(target_w, target_h)
    )

    return valid_generator, test_generator

train_generator = get_train_generator(train_df, image_dir, "Image", labels)
valid_generator, test_generator = get_test_and_valid_generator(valid_df, test_df, train_df, image_dir, "Image", labels)

# x, y = train_generator.__getitem__(0)
generated_image, _ = train_generator[0]

if plot_image:
    plt.figure(figsize=(6, 6))
    plt.imshow(generated_image[0], cmap='gray')
    plt.show()

# Model Development
# Moving onto model training and development, class imbalance must be taken in consideration
# before neural network training.
# One of the challenges with working with medical diagnostic datasets is the large class imbalance present in such datasets.

if plot_image:
    plt.figure(figsize=(10, 8))
    plt.xticks(rotation=90)
    plt.bar(x=labels, height=np.mean(train_generator.labels, axis=0))
    plt.title("Frequency of Each Class")
    plt.tight_layout()
    plt.show()

# Compute Class Frequencies
def compute_class_freqs(labels):
    # Total number of patients (rows)
    N = np.size(labels, axis=0)

    positive_frequencies = np.sum(labels == 1, axis=0) / N
    negative_frequencies = np.sum(labels == 0, axis=0) / N
    return positive_frequencies, negative_frequencies

#################################
# For testing the function
label_matrix = np.array([
    [1, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 1],
    [1, 0, 1]])

x, y = compute_class_freqs(label_matrix)
print("Testing the Class Frequency Function: ", x, y)
#################################

freq_pos, freq_neg = compute_class_freqs(train_generator.labels)
print("Positive Frequencies:", freq_pos)
print("Negative Frequencies:", freq_neg)

if plot_image:
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.bar(x - width / 2, freq_pos, width, label="Positive", color="royalblue")
    ax.bar(x + width / 2, freq_neg, width, label="Negative", color="orange")

    # Labels and title
    ax.set_xlabel("Class")
    ax.set_ylabel("Value")
    ax.set_title('Positive and Negative Frequencies by Class')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90)
    ax.legend()
    plt.tight_layout()
    # plt.savefig('positive_negative_frequencies_class.jpg', format='jpg', dpi=300)
    plt.show()

# Addressing Class Imbalance
pos_weights = freq_neg
neg_weights = freq_pos
pos_contribution = freq_pos * pos_weights
neg_contribution = freq_neg * neg_weights

if plot_image:
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.bar(x - width / 2, pos_contribution, width, label="Positive", color="yellowgreen")
    ax.bar(x + width / 2, neg_contribution, width, label="Negative", color="crimson")

    # Labels and title
    ax.set_xlabel("Class")
    ax.set_ylabel("Value")
    ax.set_title('Positive and Negative Frequencies after Applying Weightings ')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90)
    ax.legend()
    plt.tight_layout()
    # plt.savefig('positive_negative_frequencies_weighting.jpg', format='jpg', dpi=300)
    plt.show()

# Define the loss function
# For the multi-class loss, we add up the average loss for each individual class.
# Note that we also want to add a small value, ϵ, to the predicted values before taking their logs.
# This is simply to avoid a numerical error that would otherwise occur if the predicted value happens to be zero.

def get_weighted_loss(pos_weights, neg_weights, epsilon=1e-7):
    """
    Return weighted loss function given negative weights and positive weights.

    Args:
      pos_weights (np.array): array of positive weights for each class, size (num_classes)
      neg_weights (np.array): array of negative weights for each class, size (num_classes)

    Returns:
      weighted_loss (function): weighted loss function
    """

    def weighted_loss(y_true, y_pred):
        """
        Return weighted loss value.

        Args:
            y_true (Tensor): Tensor of true labels, size is (num_examples, num_classes)
            y_pred (Tensor): Tensor of predicted labels, size is (num_examples, num_classes)
        Returns:
            loss (float): overall scalar loss summed across all classes
        """
        # initialize loss to zero
        loss = 0.0

        for i in range(len(pos_weights)):
            # for each class, add average weighted loss for that class
            loss += -(K.mean(pos_weights[i] * y_true[:, i] * K.log(y_pred[:, i] + epsilon) +\
                             neg_weights[i] * (1 - y_true[:, i]) * K.log(1 - y_pred[:, i] + epsilon)))
        return loss

    return weighted_loss

# Class Imbalance (weighted) Unit Test
print("Test example:\n")

y_true = tf.constant(np.array(
    [[1, 1, 1],
     [1, 1, 0],
     [0, 1, 0],
     [1, 0, 1]]
), dtype=tf.float32)

print("y_true:\n")
print(y_true.numpy())

w_p = np.array([0.25, 0.25, 0.5])
w_n = np.array([0.75, 0.75, 0.5])
print("\nw_p:\n")
print(w_p)

print("\nw_n:\n")
print(w_n)

y_pred_1 = tf.constant(0.7 * np.ones(y_true.shape), dtype=tf.float32)
print("\ny_pred_1:\n")
print(y_pred_1.numpy())

y_pred_2 = tf.constant(0.3 * np.ones(y_true.shape), dtype=tf.float32)
print("\ny_pred_2:\n")
print(y_pred_2.numpy())

# Test with a large epsilon in order to catch errors
L = get_weighted_loss(w_p, w_n, epsilon=1)

print("\nIf we weighted them correctly, we expect the two losses to be the same.")
L1 = L(y_true, y_pred_1).numpy()
L2 = L(y_true, y_pred_2).numpy()
print(f"\nL(y_pred_1)= {L1:.4f}, L(y_pred_2)= {L2:.4f}")
print(f"Difference is L1 - L2 = {L1 - L2:.4f}")

# DenseNet121
# We will use a pre-trained DenseNet121 model which we can load directly from Keras and then add two layers on top of it:
# 1. A "GlobalAveragePooling2D" layer to get the average of the last convolution layers from DenseNet121.
# 2. A "Dense" layer with sigmoid activation to get the prediction logits for each of our classes.

# create the base pre-trained model
base_model = DenseNet121(weights=os.path.join(data_directory, 'densenet.hdf5'), include_top=False)

x = base_model.output

# add a global spatial average pooling layer
x = GlobalAveragePooling2D()(x)

# and a logistic layer
predictions = Dense(len(labels), activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam', loss=get_weighted_loss(pos_weights, neg_weights))

# Training
# Using the `mode.fit()` function in Keras to train model
# Had it train on a small subset of the dataset (future model building will incorporate more data).

history = model.fit(train_generator,
                              validation_data=valid_generator,
                              steps_per_epoch=100,
                              validation_steps=25,
                              epochs=3)

if plot_image:
    plt.figure(figsize=(10, 8))
    plt.plot(history.history['loss'])
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.title("Training Loss Curve")
    plt.show()

# Training on the larger dataset
# Given that the original dataset is 40GB+ 40GB+ in size and the training process on the full dataset takes a few hours
# (and I don't have a proper GPU to tackle the data), I use pre-trained weights from a provided model in the ChestX-ray
# downloaded content.

model.load_weights(os.path.join(data_directory, "pretrained_model.h5"))

# Prediction and Evaluation
# Evaluating the model using the test set
predicted_vals = model.predict(test_generator, steps = len(test_generator))

# ROC Curve and AUROC
# Use 'AUC' (Area Under the Curve) from the ROC ([Receiver Operating Characteristic]
# (https://en.wikipedia.org/wiki/Receiver_operating_characteristic)) curve.
# - referred to as the AUROC value
# - Larger 'AUC', better predictions (more or less)

def get_roc_curve(labels, predicted_vals, generator, save_path=os.path.join(data_directory,"roc_curve.jpg")):
    auc_roc_vals = []
    plt.figure(1, figsize=(12, 12))

    # Generate color for each label using colormap
    colors = cm.get_cmap('tab20', len(labels))

    for i in range(len(labels)):
        try:
            gt = generator.labels[:, i]
            pred = predicted_vals[:, i]
            auc_roc = roc_auc_score(gt, pred)
            auc_roc_vals.append(auc_roc)
            fpr_rf, tpr_rf, _ = roc_curve(gt, pred)

            plt.plot(fpr_rf, tpr_rf,
                     label=labels[i] + " (" + str(round(auc_roc, 3)) + ")",
                     color=colors(i), linewidth=2)

        except Exception as e:
            print(f"Error in generating ROC curve for {labels[i]: {str(e)}}. "
                  f"Dataset lacks enough examples.")

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Save the plot
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

    return auc_roc_vals

auc_rocs = get_roc_curve(labels, predicted_vals, test_generator)

# Visualizing Learning with GradCAM