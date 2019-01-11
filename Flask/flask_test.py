"""
Harley Hou

This program currently works on Taobao.com only.
On clicking the box under thumbnail photo, a popup window should be displayed containing
the photo with all of its human face swapped.
If there is no face in the photo (or no detectable face), original photo should be displayed
The popup window will close on clicking the image itself

You will need: (in windows environment)
The code itself of course
python(3)
openCV and Dlib for face processing
Cmake for Dlib
Visual studio for Cmake (Do not untick the boxes when installing unless u know what u r doing)
numpy
flask
Install provided userscript (script.user.js) to ur browser (with Chrome: tampermonkey or Firefox: greasemonkey)
I'm using tampermonkey, just paste the content and save works for me.
Should work on firefox but never tried

To run the code: simply type >python flask_test.py in terminal
To use another face: put another jpg photo containing one face into the face folder and restart the program

How it works:
Pretty simple, face image is processed on startup.
Get url with the userscript and POST it to local server. Download the image.
Swap the faces, GET the popup page that displays the image. Popup with the userscript.


"""

import os
import base64
import importlib
import random
import urllib.request as req
from flask import Flask, flash, request, redirect, url_for, send_from_directory, render_template
from time import sleep
from faceswap import func_read_face, func_swap_with_face, func_output_img  # for some reason this shows red in my
# -pycharm but it works still


app = Flask(__name__)

random.seed()


@app.route('/')  # not really used now, created for debugging purposes
def home():
    return send_from_directory('test_img', 'F.jpg')


@app.route('/image/')  # there should be better ways to display an image but this works fine for me
def send_img():
    return send_from_directory('processed_photo', 'output.jpg')


@app.route('/up/', methods=['GET', 'POST'])  # the userscript should POST the link to original photo and GET the popup
# with modified photo
def my_form_post():
    if request.method == 'POST':
        text = request.data.decode("ascii")
        # print("got txt: " + text)
        req.urlretrieve(text, "raw_photo/temp_raw_img.jpg")
        return "good"  # for debugging purposes
    else:
        return "http://localhost:5000/popup/"


@app.route('/popup/')
def popup():  # when the popup is called, modified photo is generated from raw background and pre calculated face image
    # folders passed into these functions allowing swapping all images under same folder. Currently redundant.
    func_output_img(func_swap_with_face(proc_face, "raw_photo"), "processed_photo/output.jpg")
    sleep(0.1)  # sometimes the image would be NULL, I'm guessing it's the page returned before image file generated.
    # adding some delay seems to help this problem
    # a random string added after the link so the browser will not cache the image
    return render_template('popup.html', modded_image=('http://localhost:5000/image/'+"?" + str(random.randrange(99999))))


# the face image is processed on the startup to get the landmarks and save calculation time when swapping
proc_face = func_read_face("face")
# print("face_read")


if __name__ == "__main__":
    app.run(debug=True)




