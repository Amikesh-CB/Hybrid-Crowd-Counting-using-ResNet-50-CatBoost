# Hybrid Crowd Counting using ResNet-50 & CatBoost

A hybrid deep learning and machine learning framework for crowd counting that combines **ResNet-50** for visual feature extraction with **CatBoost** for crowd count regression. The framework leverages transfer learning to extract robust semantic features from surveillance images and employs gradient boosting to model complex non-linear relationships, resulting in improved counting accuracy on the **MALL Dataset**.



## Overview

Accurate crowd counting is essential for applications such as public safety, smart surveillance, event management, and urban analytics. This project investigates a hybrid approach that integrates deep feature extraction with ensemble learning, demonstrating how pretrained convolutional neural networks and gradient boosting can complement each other for regression tasks.



## Key Features

* Extracted high-level visual representations using an **ImageNet-pretrained ResNet-50**.
* Implemented two crowd counting pipelines:

  * **End-to-End ResNet-50 Regression**
  * **Hybrid ResNet-50 + CatBoost Regression**
* Applied transfer learning to reduce training time while improving feature quality.
* Performed image preprocessing and normalization for consistent model performance.
* Optimized CatBoost hyperparameters to improve regression accuracy.
* Evaluated models using **MAE**, **MSE**, **RMSE**, and **R² Score**.
* Compared deep learning and hybrid regression approaches through quantitative analysis and visualizations.
* Designed a modular pipeline that can be extended to other crowd counting datasets.



## Dataset

**MALL Dataset**

* 2,000 surveillance images
* Resolution: **640 × 480**
* Ground-truth annotations stored in **mall_gt.mat**
* Crowd count range: **13–53 people per frame**
* Real-world challenges:

  * Perspective distortion
  * Occlusion
  * Illumination variation
  * Scale variation
  * Background complexity


## Methodology

### 1. Data Preprocessing

* Resized images to **224 × 224**
* Normalized images using ImageNet statistics
* Applied optional data augmentation:

  * Horizontal Flip
  * Random Crop
  * Color Jitter



### 2. Feature Extraction

A pretrained **ResNet-50** model was used to learn discriminative visual features from crowd images.

Two modeling strategies were explored:

#### A. End-to-End Deep Regression

* Replaced the final fully connected layer with a single regression neuron.
* Trained using **Mean Squared Error (MSE)** loss.
* Optimized with the **Adam** optimizer.

#### B. Hybrid Regression

* Extracted feature embeddings from ResNet-50.
* Trained a **CatBoost Regressor** on the extracted embeddings.
* Leveraged gradient boosting to model non-linear relationships between image features and crowd counts.

The hybrid approach demonstrated improved predictive performance compared to direct CNN regression.



## Results

| Metric   | ResNet-50 |  CatBoost |
| -------- | --------: | --------: |
| MAE      |     2.519 | **2.320** 
| MSE      |     9.927 | **8.854** |
| R² Score |     0.805 | **0.826** |

**Key Insight:**
The CatBoost model consistently outperformed the end-to-end regression model by learning complex relationships from deep feature embeddings while reducing prediction error.



## Visualizations

The repository includes:

* Training and Validation Loss Curves
* Predicted vs Actual Crowd Count Scatter Plot
* Correlation Heatmap
* Model Performance Comparison

These visualizations provide insights into convergence behavior, prediction quality, and overall model performance.



## Evaluation Metrics

The models were evaluated using standard regression metrics:

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* R² Score

These metrics provide a comprehensive assessment of prediction accuracy and model generalization.



## Project Structure

```text
📦 Hybrid-Crowd-Counting
├── dataset/
├── src/
│   ├── resnet_model.py
│   ├── catboost_model.py
│   ├── train_resnet.py
│   ├── extract_embeddings.py
│   ├── train_catboost.py
│   └── utils.py
├── results/
├── README.md
└── requirements.txt
```



## Installation

```bash
pip install -r requirements.txt
```


## Usage

### Train the ResNet-50 Regression Model

```bash
python train_resnet.py
```

### Extract Deep Feature Embeddings

```bash
python extract_embeddings.py
```

### Train the CatBoost Regressor

```bash
python train_catboost.py
```


## Technology Stack

* Python
* PyTorch
* ResNet-50
* CatBoost
* OpenCV
* NumPy
* Pandas
* Matplotlib
* Scikit-learn



## Future Enhancements

* Implement density-map estimation models (CSRNet, DM-Count, BL, etc.).
* Integrate perspective-aware crowd counting.
* Explore lightweight backbones such as MobileNetV3 and EfficientNet for edge deployment.
* Extend the framework to support video-based temporal crowd counting.
* Deploy the model as a web application or REST API for real-time inference.



## Repository Highlights

* Hybrid Deep Learning + Gradient Boosting architecture
* Transfer Learning with ResNet-50
* Comparative performance analysis
* Modular and reproducible project structure
* Suitable foundation for research and real-world surveillance applications



## Author

**Amikesh Kesharwani**

B.Tech (Information Technology) | Machine Learning & Computer Vision Enthusiast

Feel free to open an issue or contribute improvements to the project.
