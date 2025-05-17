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

Make sure the following libraries are installed:

```bash
Python==3.10.12
TensorFlow==2.13.0
Keras==2.13.1
Keras-Tuner==1.3.5
numpy
matplotlib
sklearn
Install them with:

bash
Copy
Edit
pip install -r requirements.txt
 Classification Results
Model	Accuracy	Precision	Recall	F1 Score
ResNet50	88.48%	85.69%	91.96%	88.71%
VGG16	86.73%	88.92%	83.43%	86.09%
Keras Tuner	87.60%	87.84%	86.84%	87.34%

Insights:

ResNet50 excels in identifying true positive cases (high recall).

VGG16 maintains a solid balance between precision and recall.

Keras Tuner finds a middle ground with good overall performance through hyperparameter tuning.

Future Work
Research Paper: Document and publish findings for academic contribution.

Website Hosting: Build an interactive web app where users can upload images for classification and learn more about breast cancer detection.

Contributor
Saurav Singh Rawat

Acknowledgments
Thanks to Kaggle for the dataset.

Gratitude to the open-source deep learning community for tools like TensorFlow, Keras, and Keras Tuner.

Contribute
Feel free to:

Star this repo

 Report bugs

Suggest improvements

Happy coding!
