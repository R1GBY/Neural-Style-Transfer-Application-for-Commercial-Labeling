from flask import Flask, render_template, request, session
import os
from werkzeug.utils import secure_filename
import tensorflow as tf
import tensorflow_hub as hub
# content image path & style image path
import cv2
# Allows the user to import compressed versions of any tensorflow hub model that will be used in the code
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'

# The following libraries below allow the user to display the images as they are being modified
import IPython.display as display
import matplotlib.pyplot as plt
import matplotlib as mpl

# Establishing the formatting of image outputs
mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False

import numpy as np
import PIL.Image
import time
import functools


# *** Backend operation

#https://thinkinfi.com/upload-and-display-image-in-flask-python/

# WSGI Application
# Defining upload folder path
UPLOAD_FOLDER = os.path.join('staticFiles', 'uploads')
# # Define allowed files
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

# Provide template folder name
# The default folder name should be "templates" else need to mention custom folder name for template path
# The default folder name for static files should be "static" else need to mention custom folder for static path
app = Flask(__name__, template_folder='./templates', static_folder='./staticFiles')
# Configure upload folder for Flask application
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define secret key to enable session
app.secret_key = 'This is your secret key to utilize session in Flask'


@app.route('/')
def index():
    return render_template('index_upload_and_display_image.html')


@app.route('/', methods=("POST", "GET"))
def uploadFile():
    if request.method == 'POST':
        # Upload file flask
        uploaded_img = request.files['uploaded-file']
        # Extracting uploaded data file name
        img_filename = secure_filename(uploaded_img.filename)
        # Upload file to database (defined uploaded folder in static path)
        uploaded_img.save(os.path.join(app.config['UPLOAD_FOLDER'], img_filename))
        # Storing uploaded file path in flask session
        session['uploaded_img_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], img_filename)

        return render_template('index_upload_and_display_image_page2.html')




@app.route('/show_image')
def displayImage():
    # Retrieving uploaded file path from session
    img_file_path = session.get('uploaded_img_file_path', "./staticFiles/uploads/")
    print(img_file_path)
    # Display image in Flask application web page
    return render_template('show_image.html', user_image=img_file_path)



if __name__ == '__main__':
    app.run(debug=True)