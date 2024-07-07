import os
import cv2
import glob
import time
import pickle
import warnings
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from keras.preprocessing.image import ImageDataGenerator

warnings.filterwarnings("ignore")
train_time = time.time()
# Define the path to the main folder containing numbered folders
data_folder = 'Data\\Product Classification'

# Initialize an empty list to store image paths
training_data = []
training_labels = []
val_data = []
val_labels = []
# Iterate through each numbered folder
for folder_name in range(1, 21):
    folder_name = str(folder_name)
    folder_path = os.path.join(data_folder, folder_name)

    # Check if the folder is a directory
    if os.path.isdir(folder_path):
        train_folder = os.path.join(folder_path, 'Train')
        validation_folder = os.path.join(folder_path, 'Validation')
        # Check if the 'Train' folder exists within the numbered folder
        if os.path.exists(train_folder) and os.path.isdir(train_folder):
            # Use glob to collect image files (e.g., assuming they are JPEG files)
            image_files = glob.glob(os.path.join(train_folder, '*.png'))
            for image_file in image_files:
                # Read the image and append it to the training data list
                image = cv2.imread(image_file)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = cv2.resize(image, (224, 224))
                image = np.expand_dims(image, axis=-1)
                training_data.append(image)
                training_labels.append(folder_name)

            val_image_files = glob.glob(os.path.join(validation_folder, '*.png'))
            for Vimage_file in val_image_files:
                # Read the image and append it to the training data list
                image1 = cv2.imread(Vimage_file)
                image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
                image1 = cv2.resize(image1, (224, 224))
                image1 = np.expand_dims(image1, axis=-1)
                val_data.append(image1)
                val_labels.append(folder_name)


# ------------------------------------------------------Data Augmentation
# Create an ImageDataGenerator for data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
random = 50
datagen.fit(training_data, seed=random)
# Augment and append augmented images to training data
augmented_training_data = []
augmented_training_labels = []
for img, label in zip(training_data, training_labels):
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    label = np.expand_dims(label, axis=0)  # Add batch dimension

    # Generate a single augmented image
    batch = next(datagen.flow(img, label, batch_size=1))
    augmented_img, augmented_label = batch
    augmented_img = np.squeeze(augmented_img, axis=0)  # Remove batch dimension
    augmented_label = np.squeeze(augmented_label, axis=0)  # Remove batch dimension

    augmented_training_data.append(augmented_img)
    augmented_training_labels.append(augmented_label)

augmented_training_data = np.array(augmented_training_data)
augmented_training_labels = np.array(augmented_training_labels)


# Concatenate augmented data with original training data
augmented_training_data = np.concatenate((training_data, augmented_training_data), axis=0)
augmented_training_labels = np.concatenate((training_labels, augmented_training_labels), axis=0)

augmented_training_data = np.squeeze(augmented_training_data, axis=-1)


# Ensure that the images in augmented_training_data have the correct data type
augmented_training_data = augmented_training_data.astype(np.uint8)


sift = cv2.SIFT_create()
descriptor_list = []
for img in augmented_training_data:
    kp, des = sift.detectAndCompute(img, None)
    descriptor_list.append(des)

all_descriptors = np.vstack(descriptor_list)
num_of_clusters = 90
pickle.dump(num_of_clusters, open("num_of_clusters.pickle", 'wb'))

kmeans_obj = KMeans(n_clusters=num_of_clusters, n_init=25, max_iter=500, random_state=random)
predicted_labels = kmeans_obj.fit_predict(all_descriptors)

# Create histogram
mega_histogram = np.array([np.zeros(num_of_clusters) for i in range(len(augmented_training_data))])
old_count = 0
for i in range(len(augmented_training_data)):
    l = len(descriptor_list[i])
    for j in range(l):
        idx = predicted_labels[old_count + j]
        mega_histogram[i][idx] += 1
    old_count += l

y_scalar = np.array([abs(np.sum(mega_histogram[:, h], dtype=np.int32)) for h in range(num_of_clusters)])
plt.bar(np.arange(num_of_clusters), y_scalar)
plt.xlabel("Visual Word Index")
plt.ylabel("Frequency")
plt.title("Complete training Vocabulary Generated")
plt.show()

scaler = StandardScaler()
scaled_mega_histogram = scaler.fit_transform(mega_histogram)

svm = SVC()
svm.fit(scaled_mega_histogram, augmented_training_labels)
train_acc = svm.score(scaled_mega_histogram, augmented_training_labels)
print(f"SVM training Accuracy:{train_acc*100:.2f}")

log = LogisticRegression(random_state=random)
log.fit(scaled_mega_histogram, augmented_training_labels)
train_log_acc = log.predict(scaled_mega_histogram)
print(f"Logistic training Accuracy:{accuracy_score(augmented_training_labels, train_log_acc)*100:.2f}")

print(f"Total training time: {int(time.time()-train_time)} sec ")

# -------------------------------------------------------------validation
val_time = time.time()
val_descriptor_list = []
for Vimg in val_data:
    kp, des = sift.detectAndCompute(Vimg, None)
    val_descriptor_list.append(des)

all_val_descriptors = np.vstack(val_descriptor_list)
val_predicted_labels = kmeans_obj.predict(all_val_descriptors)
pickle.dump(kmeans_obj, open("kmeans.pickle", 'wb'))
val_mega_histogram = np.array([np.zeros(num_of_clusters) for i in range(len(val_data))])
old_count = 0
for i in range(len(val_data)):
    l = len(val_descriptor_list[i])
    for j in range(l):
        idx = val_predicted_labels[old_count + j]
        val_mega_histogram[i][idx] += 1
    old_count += l

vy_scalar = np.array([abs(np.sum(val_mega_histogram[:, h], dtype=np.int32)) for h in range(num_of_clusters)])
plt.bar(np.arange(num_of_clusters), vy_scalar)
plt.xlabel("Visual Word Index")
plt.ylabel("Frequency")
plt.title("Complete validation Vocabulary Generated")
plt.show()

val_scaled_mega_histogram = scaler.transform(val_mega_histogram)
pickle.dump(scaler, open("scaler.pickle", 'wb'))
svm_pred = svm.predict(val_scaled_mega_histogram)
acc = svm.score(val_scaled_mega_histogram, val_labels)
pickle.dump(svm, open("svm.pickle", 'wb'))
print(f"SVM Accuracy:{acc*100:.2f}")


pickle.dump(log, open("log.pickle", 'wb'))
Lpre = log.predict(val_scaled_mega_histogram)
print(f"Logistic Accuracy:{accuracy_score(val_labels, Lpre)*100:.2f}")

print(f"Total validation time: {int(time.time()-val_time)} sec ")

# for i, Vimg in enumerate(val_data):
#     plt.imshow(Vimg, cmap='gray')
#     plt.title(
#         f"Image {i + 1}\nTrue Label: {val_labels[i]}\nSVM Predicted Label: {svm_pred[i]}\nLogistic Predicted Label: {Lpre[i]}")
#     plt.show()

correctly_classified = []

for i, Vimg in enumerate(val_data):
    is_correct_svm = svm_pred[i] == val_labels[i]
    is_correct_log = Lpre[i] == val_labels[i]

    correctly_classified.append(is_correct_svm and is_correct_log)

# Plotting the graph
plt.bar(range(1, len(val_data) + 1), [1 if correct else 0 for correct in correctly_classified], color=['green' if correct else 'red' for correct in correctly_classified])
plt.xlabel("Image Index")
plt.ylabel("Correctly Classified (1) / Incorrectly Classified (0)")
plt.title("Correct Classification of Images in Validation Set")
plt.show()