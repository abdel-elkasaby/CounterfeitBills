import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report

# Path to the dataset
image_dir = Path('C:/Users/abdel/OneDrive/Desktop/TF Work/Bills/ES_images/')

# Get list of all image file paths
filepaths = list(image_dir.glob(r'**/*.jpg'))

# Check if filepaths is empty
if not filepaths:
    raise ValueError("No image files found in the specified directory.")

# Initialize labels with random 0s and 1s to avoid single-class issue
labels = np.random.randint(2, size=len(filepaths))

# Create DataFrame
data = {'Filepath': [str(path) for path in filepaths], 'Label': labels}
labels_df = pd.DataFrame(data)

# Save to CSV
labels_df.to_csv('C:/Users/abdel/OneDrive/Desktop/TF Work/Bills/labels.csv', index=False)

# Read the CSV file
labels_df = pd.read_csv('C:/Users/abdel/OneDrive/Desktop/TF Work/Bills/labels.csv')

# Ensure the Filepath column contains full paths
labels_df['Filepath'] = labels_df['Filepath'].apply(lambda x: str(Path(x).resolve()))

# Check if the DataFrame is empty after processing paths
if labels_df.empty:
    raise ValueError("The DataFrame is empty after processing file paths. Please check the file paths in the CSV file.")

# Convert labels to strings
labels_df['Label'] = labels_df['Label'].astype(str)

# Display the first few rows of the DataFrame for debugging
print(labels_df.head())

# Split dataset into training and testing sets
train_df, test_df = train_test_split(labels_df, train_size=0.7, shuffle=True, random_state=1)

# Check if the training and test sets are populated
if train_df.empty or test_df.empty:
    raise ValueError("The training or test DataFrame is empty after the train-test split.")

# Create ImageDataGenerators
train_generator = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    validation_split=0.2
)

val_generator = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    validation_split=0.2
)

test_generator = ImageDataGenerator(
    rescale=1./255
)

# Generate batches of tensor image data
train_images = train_generator.flow_from_dataframe(
    dataframe=train_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(240, 240),  # Change to 240x240
    color_mode='rgb',
    class_mode='binary',
    batch_size=32,
    shuffle=True,
    seed=42,
    subset='training'
)

val_images = val_generator.flow_from_dataframe(
    dataframe=train_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(240, 240),  # Change to 240x240
    color_mode='rgb',
    class_mode='binary',
    batch_size=32,
    shuffle=True,
    seed=42,
    subset='validation'
)

test_images = test_generator.flow_from_dataframe(
    dataframe=test_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(240, 240),  # Change to 240x240
    color_mode='rgb',
    class_mode='binary',
    batch_size=32,
    shuffle=False
)

# Build the model
inputs = tf.keras.Input(shape=(240, 240, 3))  # Change to 240x240
x = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu')(inputs)
x = tf.keras.layers.MaxPool2D()(x)
x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(x)
x = tf.keras.layers.MaxPool2D()(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    train_images,
    validation_data=val_images,
    epochs=5,  # Set to 5 epochs
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=2,  # Early stopping patience set to 2 epochs
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            patience=2  # Learning rate reduction patience set to 2 epochs
        )
    ]
)

# Evaluate the model on the test set
results = model.evaluate(test_images, verbose=0)
print("Test Loss: {:.5f}".format(results[0]))
print("Test Accuracy: {:.2f}%".format(results[1] * 100))

# Make predictions with the model
predictions = (model.predict(test_images) >= 0.5).astype(int)

# Update labels in the original DataFrame with predictions
labels_df.loc[test_df.index, 'Label'] = predictions

# Save the updated DataFrame with predictions to a new CSV file
labels_df.to_csv('C:/Users/abdel/OneDrive/Desktop/TF Work/Bills/predicted_labels.csv', index=False)

# Convert test labels to a NumPy array
test_labels = np.array(test_images.labels).astype(int)

# Confusion matrix and classification report
cm = confusion_matrix(test_labels, predictions, labels=[0, 1])
clr = classification_report(test_labels, predictions, labels=[0, 1], target_names=["Counterfeit", "Real"])

plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='g', vmin=0, cmap='Blues', cbar=False)
plt.xticks(ticks=[0.5, 1.5], labels=["Counterfeit", "Real"])
plt.yticks(ticks=[0.5, 1.5], labels=["Counterfeit", "Real"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

print("Classification Report:\n----------------------\n", clr)
