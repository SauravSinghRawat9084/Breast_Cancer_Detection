#  Breast Cancer Detection using Histopathology Images

This project focuses on detecting breast cancer using histopathology images with deep learning models. We used a dataset from Kaggle and implemented classification using pre-trained CNN architectures (VGG16, ResNet50) and hyperparameter tuning with Keras Tuner.

---

##  Dataset

The dataset is available on Kaggle:  
**[Breast Cancer Histopathological Images Dataset](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images)**

>  **Note:** Ensure you have the appropriate permissions to use this dataset as per Kaggle's terms.

---

##  Model Architectures

###  Keras Tuner
Keras Tuner helps with automatic hyperparameter optimization. It supports techniques like:
- Random Search
- Bayesian Optimization
- Hyperband

###  Pretrained Models

####  VGG16
- Known for its uniform and simple 3x3 convolution architecture.
- Performs well on a wide range of image classification tasks.

#### ResNet-50
- Introduces *residual learning* which improves training for deeper networks.
- Achieves strong performance on medical image datasets.

---

##  Dependencies

- Python (version 3.10.12)
- TensorFlow (version 2.13.0)
- Keras (version 2.13.1)
- Keras Tuner (version 1.3.5)
- Other required libraries

# Breast Cancer Classification Results
## ResNet50 Model:
-  Precision: 0.8569
-  Accuracy: 0.8848
-  Recall: 0.9196
-  F1 Score: 0.8871

## VGG16 Model:
- Precision: 0.8892
- Accuracy: 0.8673
- Recall: 0.8343
- F1 Score: 0.8609

## Keras Tuner Model:
- Precision: 0.8784
- Accuracy: 0.8760
- Recall: 0.8684
- F1 Score: 0.8734

  ---
  
### Conclusion:
The ResNet50 model demonstrated high recall, indicating its effectiveness in correctly identifying positive cases.
VGG16 achieved a good balance between precision and recall, leading to a reliable classification performance.
The Keras Tuner model, with its tuned hyperparameters, showcased competitive precision and accuracy.
These results provide insights into the performance of different models for breast cancer classification. Consideration of specific requirements and trade-offs between precision and recall can guide the selection of the most suitable model for your application.

### Future Work
-Research Paper: Document the methodology, results, and insights gained from this project to contribute to the existing body of knowledge in the field of breast cancer detection using histopathology images.

-Website Hosting: Develop and host a website where users can interact with the model, submit images for analysis, and access information about breast cancer detection.

### Contributors
Saurav Singh Rawat

### Acknowledgments
Special thanks to Kaggle for providing the breast cancer histopathological images dataset and the open-source community for their valuable contributions to the field of deep learning.

Feel free to contribute, report issues, or suggest improvements. Happy coding
