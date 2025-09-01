Handwriiten Digit Recognizer

## 📄 Improved README (README.md)

This project implements a **Convolutional Neural Network (CNN)** to classify handwritten digits from the **MNIST dataset** (0–9). It demonstrates how deep learning can be applied to image recognition with high accuracy.  

---

## 🚀 Features
- Loads and preprocesses the MNIST dataset (normalization, reshaping, one-hot encoding).  
- Builds a CNN with **Conv2D, MaxPooling, Flatten, and Dense layers**.  
- Trains the model with **Adam optimizer** and categorical cross-entropy loss.  
- Achieves **~98% test accuracy** on digit classification.  
- Visualizes predictions for sample handwritten digits.  

---

## 🛠️ Tech Stack
- **Python 3**  
- **TensorFlow / Keras** – Deep Learning framework  
- **NumPy** – Data manipulation  
- **Matplotlib** – Visualization  

---

## ⚙️ Installation & Usage
1. Clone this repository or download the files.  
2. Install required dependencies:  
   ```bash
   pip install tensorflow matplotlib numpy
````

3. Run the script:

   ```bash
   python digit_recognizer.py
   ```
4. The program will:

   * Train a CNN model
   * Evaluate it on the test set
   * Display test accuracy
   * Show a sample prediction with the digit image

---

## 📊 Example Output

```
Epoch 1/5
1688/1688 [==============================] - 30s 17ms/step - loss: 0.2001 - accuracy: 0.9405 - val_loss: 0.0667 - val_accuracy: 0.9802
...
Test Accuracy: 98.35%
```

Sample prediction visualization:
*(Model predicts the digit on the displayed image)*

---

## 📌 Future Improvements

* Add GUI/web app for real-time digit recognition.
* Deploy model with Flask/Django or Streamlit.
* Experiment with deeper CNNs or pretrained models.
* Integrate with real-world handwriting input (touchpad or image upload).

---

## 👨‍💻 Author

Developed as a demonstration of **deep learning for image classification** using CNNs.

```

---

## 📑 Brief Project Report  

**Title:** Handwritten Digit Recognition using Convolutional Neural Networks  

**Objective:**  
To build a deep learning model capable of classifying handwritten digits (0–9) with high accuracy, using the MNIST dataset.  

**Methodology:**  
1. Loaded the MNIST dataset (60,000 training + 10,000 test images).  
2. Preprocessed data by normalizing pixel values and one-hot encoding labels.  
3. Designed a CNN with convolution, pooling, and dense layers.  
4. Trained the model for 5 epochs with the Adam optimizer.  
5. Evaluated performance on the test dataset and visualized predictions.  

**Results:**  
- The CNN achieved **~98% test accuracy**.  
- Successfully predicted digits on unseen test images.  
- Demonstrated robust performance on digit classification tasks.  

**Conclusion:**  
This project highlights the effectiveness of **CNNs in computer vision tasks**. Even with a simple architecture, the model reaches near state-of-the-art performance on MNIST. It can be extended to real-world handwritten digit applications and deployed in interactive systems.  

---

