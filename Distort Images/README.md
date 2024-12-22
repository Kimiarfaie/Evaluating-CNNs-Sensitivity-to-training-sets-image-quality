# Distorting the dataset

In this section the ImageNet subset is distorted in a way that the image quality is maintained within a specific perceptual range. The distortions are applied dynamically to ensure consistent quality across the dataset while introducing controlled changes in color.

## Distortion Framework
The distortion framework utilizes the **S-CIELAB image quality metric** to control the perceptual quality of distorted images. This approach ensures that the distortions are introduced uniformly across the dataset while maintaining perceptual quality.

1. Starting with an initial distortion level, the saturation of the current image is modified.
2. The S-CIELAB metric is calculated between the distorted image and the original image to measure perceptual quality.
3. If the quality falls within a defined perceptual range, the distorted image is saved.
4. If the quality is outside the range, the distortion level is adjusted dynamically, and the process is repeated until the desired quality is achieved.

This iterative approach ensures that all distorted images meet the defined quality standards.

Use two_dynamic_distortimages.m to distort the dataset, this will create a distorted version of the subset ImageNet used. You can change the distortion type by changing **distortion_function** variable. 
## Distortion types

**Saturation Change**: Applied to the training set to create a distorted training dataset.

**Hue Change and Saturation Change**: For the test sets, the same framework can be used to generate test datasets with different types of distortions. 
