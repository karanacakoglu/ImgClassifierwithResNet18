# 🫁 Chest X-Ray Medical Image Classification (Pneumonia)

### Project Overview
Pneumonia is a critical respiratory infection that requires rapid and accurate diagnosis. This project implements a **Deep Learning** model based on the **ResNet18** architecture to automatically detect signs of pneumonia from Chest X-Ray (CXR) images.

---

### 📊 Dataset Information
* **Source:** [Chest X-Ray Images (Pneumonia) - Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
* **Content:** The dataset is organized into 3 folders (train, test, val) and contains subfolders for each image category (Pneumonia/Normal).
* **Composition:** 5,856 X-Ray images (JPEG) and 2 categories (Pneumonia/Normal).
* **Characteristics:** All chest X-ray imaging was performed as part of routine clinical care.

---

### 🛠️ Technical Implementation
* **Model Backbone:** **ResNet18** (Residual Networks) was chosen for its efficient feature extraction and ability to train deep networks without degradation.
* **Preprocessing:** - Resizing images to $224 \times 224$ pixels.
    - Normalization based on ImageNet mean and standard deviation.
    - Data Augmentation (Random Rotation, Zoom) to prevent overfitting on clinical samples.
* **Transfer Learning:** Fine-tuned a pre-trained model to adapt from general object recognition to specific medical pattern recognition.

---

### 🚀 Results & Performance
The model focuses on high **Recall (Sensitivity)** to ensure that potential pneumonia cases are not missed, which is critical in a medical context.

---

### 🔗 Quick Links
- **Colab Notebook:** [View Medical Imaging Implementation](https://github.com/karanacakoglu)
- **Frameworks:** Python, PyTorch/TensorFlow, OpenCV, Matplotlib.
