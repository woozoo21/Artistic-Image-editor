# Artistic Image Editor - Neural Style Transfer

## Overview
Neural Style Transfer (NST) is a powerful machine learning technique that takes two images—one representing the content and the other representing the style—and blends them to create a new image that combines the content of the first with the artistic style of the second. This method is based on deep learning and neural networks, specifically utilizing convolutional neural networks (CNNs) to process the image content and style.

## What Does It Do?
The tool in this repository allows you to apply Neural Style Transfer to images with various options:

Content Image: The image you wish to transform.
Style Image: The artistic style that will be applied to your content image.
You can start with pencil sketch effects and gradually tweak the intensity of the pencil effect for a finer touch. Additionally, the tool supports artistic transformations, including:

Oil Painting Effect: Convert your image into a style reminiscent of classic oil paintings.
Famous Artworks Style: Apply iconic styles, like Van Gogh’s Starry Night, to your content image.
The tool allows you to adjust the level of intensity for different effects to achieve a highly customizable outcome. You can then save your transformed artwork or exit if you're not satisfied with the result.

## Features

### 1. Neural Style Transfer Effects:
Pencil Sketch: Create stunning pencil-style sketches from your images.
Oil Painting: Transform your image into an oil painting effect, giving it a textured and vivid appearance.
Artistic Styles: Apply famous artwork styles like Starry Night by Van Gogh to your image for an artistic flair.

### 2. Interactive User Interface (UI/UX):
Live Previews: The tool lets you see a live preview of the transformed image as you adjust the settings. The image automatically updates based on your changes to the slider intensity.
Image Loading: The app smoothly loads images, even when changes are made to the style or effect settings, ensuring a seamless user experience.
Save or Exit: Once you're happy with the result, you can save the image to your device. Alternatively, if you don’t like the result, simply exit without saving.

## Dependencies

OpenCV: For image processing and applying effects.
TensorFlow: For deep learning models that perform Neural Style Transfer.
Numpy: For handling numerical operations.
Matplotlib: For visualizing images.

### 1. To install the required dependencies, run:
```
pip install -r requirements.txt
```
### 2. Clone the Repository:
```
git clone https://github.com/woozoo21/Artistic-Image-editor.git
cd Artistic-Image-editor
```
### 3. Run the Application:
```
python app.py
```

## Credits
https://www.youtube.com/watch?v=yn3bWvQZIIo
https://youtu.be/bgeZv_8j7ug?si=HPUrZggC15TFwtiE
