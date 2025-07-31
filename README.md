# 12.-Face-Mask-Detection
🛠️ Project: Face Mask Detection using CNN

# Try the Web App  [click to try](https://facemaskdetectionbyzaidnaeem.streamlit.app/)

🧠 Today's Focus:
I took a deep dive into computer vision today and built a Convolutional Neural Network (CNN) to detect whether a person is wearing a face mask or not. This model can play a vital role in public health compliance systems—especially in high-risk environments.

Despite training on a relatively small dataset of 7,550 images, the model achieved 94.5% accuracy and 94.5% recall, making it both precise and reliable in real-world scenarios. ⚡

⏱️ Model Training:
Training took over 2 hours, emphasizing the importance of optimized architectures and good preprocessing—even with a moderately sized dataset.

📊 Model Summary:
Classifies images into:
😷 Face Mask
😐 No Face Mask

📈 Model Performance:
✅ Accuracy: 94.5%
✅ Recall: 94.5%
✅ Works well under different lighting and angles
✅ Strong generalization on validation/test images

📌 Model Highlights:
✅ Built a custom CNN from scratch (no transfer learning!)
✅ Applied image preprocessing: resizing, normalization, augmentation
✅ Trained on a balanced dataset with real-world variation
✅ Early stopping and dropout used to avoid overfitting
✅ Binary classification with softmax activation

🔍 Techniques Applied:
✅ Convolutional Neural Networks
✅ Image Augmentation (Flip, Zoom, Shift)
✅ Evaluation Metrics: Accuracy, Precision, Recall
✅ Model Optimization: Dropout, Adam Optimizer, Learning Rate Scheduling

🛠️ Tech Stack:
Python | TensorFlow/Keras | OpenCV | NumPy | Matplotlib

🌐 Use Case Impact:
Perfect for implementation in security systems, public venues, retail stores, and transport hubs where real-time face mask compliance is essential.
