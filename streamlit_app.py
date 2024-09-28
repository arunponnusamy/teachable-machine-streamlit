import cv2
import streamlit as st
from PIL import Image, ImageOps
import numpy as np
from io import BytesIO
from keras.models import load_model  # TensorFlow is required for Keras to work

st.header('Teachable Machine Demo')
image = st.file_uploader('Upload an image', type=['png', 'jpg', 'jpeg'])


def convert_image(img):
  buf = BytesIO()
  img.save(buf, format="PNG")
  byte_im = buf.getvalue()
  return byte_im

if image is not None:
  img_pil = Image.open(image)
  st.write('Original Image')
  st.image(img_pil)

  img_np = np.array(img_pil)
  img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

  # Disable scientific notation for clarity
  np.set_printoptions(suppress=True)
  
  # Load the model
  model = load_model("keras_model.h5", compile=False)
  
  # Load the labels
  class_names = open("labels.txt", "r").readlines()
  
  # Create the array of the right shape to feed into the keras model
  # The 'length' or number of images you can put into the array is
  # determined by the first position in the shape tuple, in this case 1
  data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
  
  # Replace this with the path to your image
  image = Image.open("thumbs_up_test.jpeg").convert("RGB")
  
  # resizing the image to be at least 224x224 and then cropping from the center
  size = (224, 224)
  image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
  
  # turn the image into a numpy array
  image_array = np.asarray(image)
  
  # Normalize the image
  normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
  
  # Load the image into the array
  data[0] = normalized_image_array
  
  # Predicts the model
  prediction = model.predict(data)
  index = np.argmax(prediction)
  class_name = class_names[index]
  confidence_score = prediction[0][index]
  
  # Print prediction and confidence score
  print("Class:", class_name[2:], end="")
  print("Confidence Score:", confidence_score)

  st.write("Class: " + str(class_name[2:]))
  st.write("Confidence Score: " + str(confidence_score))
  
  
