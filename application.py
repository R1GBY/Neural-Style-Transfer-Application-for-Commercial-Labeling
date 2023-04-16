import tensorflow as tf
import numpy as np
import os
from flask import Flask, render_template, request, redirect, url_for
import PIL
from PIL import Image

import tensorflow_hub as hub

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load the TensorFlow Hub model
model_url = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
model = hub.load(model_url)

def increment_filename(filename):
  base, ext = os.path.splitext(filename)
  i = 1
  while os.path.exists(filename):
    filename = f"{base}_{i}{ext}"
    i += 1
  return filename

##################################################################################
#Converting the format that the model (Neural Network) uses to process images to a format that we can visualize
def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return PIL.Image.fromarray(tensor)


#This function is giving the user the ability to load images

def load_img(path_to_img):
  # max dimension of the images that we are importing
  max_dim = 512


#Here we are loading our file, decoding it into a 3 dimensional tensor and converting it into a file based on format
  img = tf.io.read_file(path_to_img)
  img = tf.image.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)

  shape = tf.cast(tf.shape(img)[:-1], tf.float32)
  long_dim = max(shape)
  scale = max_dim / long_dim

  new_shape = tf.cast(shape * scale, tf.int32)

  img = tf.image.resize(img, new_shape)
  img = img[tf.newaxis, :]
  return img
##################################################################################
@app.route('/')
def index():
    images = [filename for filename in os.listdir('static/uploads')
              if filename.endswith('.jpg') or filename.endswith('.png')]
    return render_template('index.html', images=images)

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    filename = file.filename
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    label = request.form['label-design']

    style = request.form['style-image']


    max_dim = 512

    content_path = file_path
    style_path = 'static/uploads/style_0.jpg'

    # Loading in our content and style image
    content_image = load_img(content_path)
    style_image = load_img(style_path)

    # Run the model on the image and style image
    stylized_image = model(tf.constant(content_image), tf.constant(style_image))[0]
    tensor_to_image(stylized_image)

    # save the image
    #stylized_filename = 'stylized_image.jpg'

    stylized_filename = increment_filename("stylized_image.jpg")

    saving_path = 'static/uploads/'
    oldpwd = os.getcwd()
    os.chdir(saving_path)
    tensor_to_image(stylized_image).save(stylized_filename)

    #merging label and background stylized image

    image1 = Image.open(stylized_filename)
    image2 = Image.open(label)

    width, height = image2.size
    image1 = image1.resize((width, height))
    image2 = image2.resize((width, height))

    merged_image = Image.new('RGB', (width, height), (0, 0, 0))
    merged_image.paste(image1, (0, 0))
    merged_image.paste(image2, (0, 0), image2)
    merged_image.save(stylized_filename)

    #turning back to old folder
    os.chdir(oldpwd)

    return render_template('result.html', filename=filename, stylized_filename=stylized_filename)

if __name__ == '__main__':
    app.run(debug=True, port=8080)