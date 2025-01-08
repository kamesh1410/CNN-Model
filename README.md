# 🐶🐱 Dog vs Cat Prediction Using CNN

A Convolutional Neural Network (CNN) based model to classify images of dogs and cats. This project involves data preparation, model implementation, evaluation, and performance optimization to predict whether an image is of a dog or a cat.

---

## 📁 Dataset Preparation

- **Dataset Source**: Kaggle's "Dogs vs. Cats" dataset.
- **Data Exploration**:
  - Loaded images from the dataset.
  - Split the dataset into training and validation sets.
  
- **Data Preprocessing**:
  - **Resizing**: Images were resized to a consistent shape.
  - **Normalization**: Pixel values were normalized to the range [0, 1].
  - **Data Augmentation**: Applied transformations like rotation, zoom, and flips to improve model generalization.

---

## 🔍 Feature Engineering

- **Image Preprocessing**:
  - Resized all images to a fixed size of 256x256 pixels.
  - Scaled pixel values to [0, 1] for easier convergence in the CNN model.

- **Data Augmentation**: 
  - Applied transformations such as:
    - Random rotations
    - Horizontal flipping
    - Zooming to increase the diversity of the training data.

---

## 🛠️ Model Implementation

### Model Architecture:
- Built a Convolutional Neural Network (CNN) with the following layers:
  1. **Convolutional Layers**: For extracting features.
  2. **MaxPooling Layers**: To down-sample the feature maps.
  3. **Flatten Layer**: To convert the 2D feature map into a 1D feature vector.
  4. **Dense Layers**: To classify the extracted features into 'Dog' or 'Cat'.
  
### CNN Layers:
- Convolution layers with 32, 64, and 128 filters.
- MaxPooling layers to reduce spatial dimensions.
- Dropout layers to prevent overfitting.
- Final output layer with softmax activation for classification.

---

## ⚙️ Hyperparameter Tuning

- Used **Grid Search** for optimizing the learning rate, batch size, and number of epochs.
- Implemented early stopping to prevent overfitting and to stop training when performance plateaus.

---

## 📊 Model Evaluation

- **Accuracy**: Measured the classification accuracy of the model on the test set.
- **Loss**: Monitored the loss value throughout the training process.

---

## 🏆 Results and Insights

- Achieved an accuracy of over 90% on the test set.
- Model showed strong generalization despite the variations in the images due to augmentation.
- Performance can be further improved with more advanced CNN architectures (e.g., VGG, ResNet).

---

## 📄 Documentation

This project includes:
- A detailed README explaining the steps for building, training, and evaluating the CNN model.
- Code implemented in Jupyter Notebooks for reproducibility.

---

## 🙌 Acknowledgments

- **Kaggle** for the "Dogs vs Cats" dataset.
- **TensorFlow/Keras** for providing tools for deep learning model development.
- **Matplotlib** and **Seaborn** for data visualization.

---

## 📬 Contact

For any questions or suggestions, feel free to reach out:

- **Name**: G. Kamesh
- **Email**: [kamesh743243@gmail.com](mailto:kamesh743243@gmail.com)
