# Image Verification and Recognition Project

## Overview

This project focuses on image verification and recognition using machine learning techniques and deep learning models. It involves two main approaches:

1. **Bag-of-Visual-Words (BoVW) with SVM and Logistic Regression**
2. **Siamese Neural Network for Image Triplet Classification**

### 1. Bag-of-Visual-Words (BoVW) Approach

#### Project Structure

The project is structured as follows:
- **Data Preparation**: Images are loaded from structured directories (`Train` and `Validation` folders) and preprocessed.
- **Data Augmentation**: Augmentation techniques such as rotation, shift, and flip are applied using `ImageDataGenerator`.
- **Feature Extraction**: SIFT (Scale-Invariant Feature Transform) is used to extract keypoint descriptors from images.
- **Clustering**: K-means clustering is applied on descriptors to generate a vocabulary of visual words.
- **Histogram Representation**: Each image is represented by a histogram of visual word occurrences.
- **Model Training**: Support Vector Machine (SVM) and Logistic Regression models are trained on histograms.
- **Evaluation**: Performance metrics like accuracy are calculated on a validation set.

#### Code Files
- `bovw_svm_logistic.py`: Implements the BoVW approach using SVM and Logistic Regression.
- `siamese.py`: Implements the Siamese Neural Network approach for image triplet classification.

### 2. Siamese Neural Network Approach

#### Project Structure

The Siamese Neural Network approach involves:
- **Triplet Generation**: Triplets of images (anchor, positive, negative) are generated from training data.
- **Model Architecture**: Utilizes a base encoder (Xception pretrained on ImageNet) to extract image embeddings.
- **Triplet Loss**: Computes the triplet loss function to learn embeddings where similar images are closer in space.
- **Training**: The model is trained on generated triplets, optimizing to minimize triplet loss.
- **Validation**: Evaluates model accuracy on a separate validation set using learned embeddings.

#### Code Files
- `siamese.py`: Implements the Siamese Neural Network for image triplet classification.

## Usage

### Prerequisites
- Python 3.x
- TensorFlow, OpenCV, scikit-learn, matplotlib, NumPy

### Instructions
1. **BoVW Approach**:
   - Run `bovw_svm_logistic.py` to train SVM and Logistic Regression models on BoVW features.
   - Evaluate model performance on validation data.

2. **Siamese Neural Network Approach**:
   - Run `siamese.py` to train the Siamese Neural Network model on generated triplets.
   - Evaluate model accuracy and visualize performance metrics.

## Results

### Performance Metrics
- SVM and Logistic Regression accuracies using BoVW approach.
- Accuracy and confusion matrix for Siamese Neural Network on validation data.

## Conclusion

This project demonstrates two distinct approaches for image verification and recognition, showcasing the effectiveness of traditional machine learning with BoVW and modern deep learning with Siamese Networks.

For detailed implementation and results, refer to the respective Python scripts (`bovw_svm_logistic.py` and `siamese.py`).


