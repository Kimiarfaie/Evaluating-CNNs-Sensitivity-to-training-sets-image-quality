# Evaluating-CNNs-Sensitivity-to-training-sets-image-quality

This repository contains the code for coursework of IMT4895 Specialisation in Colour Imaging course in Computer Science department at NTNU, investigating the impact of training data quality, particularly color distortions, on the robustness and generalization of deep learning models.

# Project Overview
Deep learning models, such as convolutional neural networks (CNNs), have achieved state-of-the-art performance in computer vision tasks. However, real-world images often suffer from distortions caused by environmental factors, acquisition devices, or post-processing operations. This can significantly degrade model performance. While robustness to distortions like noise and blur has been studied extensively, color distortions—such as changes in hue and saturation—remain underexplored.

A dynamic distortion framework was implemented to create controlled datasets with regards to their image quality, and trained popular architectures such as AlexNet, ResNet18, and ResNet50 on both clean and distorted data. The models were then evaluated on datasets with unseen distortions to analyze their robustness and generalization abilities.



