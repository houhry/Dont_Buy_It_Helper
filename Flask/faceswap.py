#!/usr/bin/python

# Copyright (c) 2015 Matthew Earl
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
#     The above copyright notice and this permission notice shall be included
#     in all copies or substantial portions of the Software.
# 
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#     OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#     MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
#     NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#     DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#     OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
#     USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
This code is modified by Harley Hou and changed its functionality quite a bit.
This code should be used by flask_test. Run flask_test if u just want to run this program.
The projects that I have looked into are: deepgaze faceswap facedetection

Functions that should be used: func_read_face, func_swap_with_face, func_output_img
(list of processed FsImage class)func_read_face(string: folder containing face image)
this function should take one image containing one face only, folder passed for the initial intended function:
matching face image with the closest head image. But the matching algorithm never works as the photo focal length is
unknown and the photos are usually modified so that focal length cannot be calculated neither. This leads to very
inaccurate face facing(if that makes any sense) and hilarious output image. That's why this function is removed.
If multiple file or faces exist, the first file first face should be used.

(list of output CV2 image)func_swap_with_face(FsImage class face, string; folder containing head image)
This function swaps all faces from all files in the folder with the provided FsImage face

func_output_img(list of CV2 image for output, string: Path to output file ie. /outputpath/output.jpg)
This function outputs the CV2 image to a given path.

How it works:
input face image, find the face with dlib face detector (where the face is in the picture),
compute the face landmarks (where everything is on that face)
input background image, do the same thing above
make a mask from the face image and swap it with the head image aligning the landmarks.
(untouched form the original code)
output to file

Things I tried:(but not used)
from deepgaze: I tried to use face photos taken from different angles to match the head photos.
a face facing vector is calculated along with landmarks for both face and head images
when swapping every head, looping through all face vector and find the minimum angle between two vectors
The problem is the vector obtained is usually too far off that this function never really worked.
probably due to focal length being unknown

from facedetection: I tried another face detection method which detects face from a larger angle.
It does work but as I am swapping all faces with frontal face photo, it is pointless to detect face that is not
facing front. This function can be turned on by defining DETECTOR_TYPE as DNN instead of DLIB, you will need the
trained model which you can find in the facedetection project on github.

"""
"""
This is the code behind the Switching Eds blog post:

    http://matthewearl.github.io/2015/07/28/switching-eds-with-python/

See the above for an explanation of the code below.

To run the script you'll need to install dlib (http://dlib.net) including its
Python bindings, and OpenCV. You'll also need to obtain the trained model from
sourceforge:

    http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2

Unzip with `bunzip2` and change `PREDICTOR_PATH` to refer to this file. The
script is run like so:

    ./faceswap.py <head image> <face image>

If successful, a file `output.jpg` will be produced with the facial features
from `<head image>` replaced with the facial features from `<face image>`.

"""
# deepgaze
# faceswap
# facedetection
import cv2
import dlib
import numpy
import argparse
import sys
import pickle
import os
import glob
import math


PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"  # for DLIB
PROTOTXT = "deploy.prototxt.txt"  # for DNN
DNN_MODEL = "res10_300x300_ssd_iter_140000.caffemodel"  # for DNN
SCALE_FACTOR = 1 
FEATHER_AMOUNT = 11
DETECTOR_TYPE = "DLIB"  # DLIB DNN
DNN_NOFACE_THRESH = 0.5

MAX_IMAGE_SIZE = 500.0  # pix, higher for better quality, may crash if set too high

# For face facing detection, not used
FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))

P3D_RIGHT_SIDE = numpy.float32([-100.0, -77.5, -5.0]) #0
P3D_GONION_RIGHT = numpy.float32([-110.0, -77.5, -85.0]) #4
P3D_MENTON = numpy.float32([0.0, 0.0, -122.7]) #8
P3D_GONION_LEFT = numpy.float32([-110.0, 77.5, -85.0]) #12
P3D_LEFT_SIDE = numpy.float32([-100.0, 77.5, -5.0]) #16
P3D_FRONTAL_BREADTH_RIGHT = numpy.float32([-20.0, -56.1, 10.0]) #17
P3D_FRONTAL_BREADTH_LEFT = numpy.float32([-20.0, 56.1, 10.0]) #26
P3D_SELLION = numpy.float32([0.0, 0.0, 0.0]) #27
P3D_NOSE = numpy.float32([21.1, 0.0, -48.0]) #30
P3D_SUB_NOSE = numpy.float32([5.0, 0.0, -52.0]) #33
P3D_RIGHT_EYE = numpy.float32([-20.0, -65.5,-5.0]) #36
P3D_RIGHT_TEAR = numpy.float32([-10.0, -40.5,-5.0]) #39
P3D_LEFT_TEAR = numpy.float32([-10.0, 40.5,-5.0]) #42
P3D_LEFT_EYE = numpy.float32([-20.0, 65.5,-5.0]) #45
#P3D_LIP_RIGHT = numpy.float32([-20.0, 65.5,-5.0]) #48
#P3D_LIP_LEFT = numpy.float32([-20.0, 65.5,-5.0]) #54
P3D_STOMION = numpy.float32([10.0, 0.0, -75.0]) #62

TRACKED_POINTS = [0, 4, 8, 12, 16, 17, 26, 27, 30, 33, 36, 39, 42, 45, 62]
camera_distortion = numpy.float32([0.0, 0.0, 0.0, 0.0, 0.0])

landmarks_3D = numpy.float32([P3D_RIGHT_SIDE,
                              P3D_GONION_RIGHT,
                              P3D_MENTON,
                              P3D_GONION_LEFT,
                              P3D_LEFT_SIDE,
                              P3D_FRONTAL_BREADTH_RIGHT,
                              P3D_FRONTAL_BREADTH_LEFT,
                              P3D_SELLION,
                              P3D_NOSE,
                              P3D_SUB_NOSE,
                              P3D_RIGHT_EYE,
                              P3D_RIGHT_TEAR,
                              P3D_LEFT_TEAR,
                              P3D_LEFT_EYE,
                              P3D_STOMION])
# End of face facing detection part

# Points used to line up the images.
ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
                RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS + JAW_POINTS)

# Points from the second image to overlay on the first. The convex hull of each
# element will be overlaid.
OVERLAY_POINTS = [
    LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
    NOSE_POINTS + MOUTH_POINTS,
]

# Amount of blur to use during colour correction, as a fraction of the
# pupillary distance.
COLOUR_CORRECT_BLUR_FRAC = 0.6

if DETECTOR_TYPE == "DLIB":
    detector = dlib.get_frontal_face_detector()
elif DETECTOR_TYPE == "DNN":
    net = cv2.dnn.readNetFromCaffe(PROTOTXT, DNN_MODEL)  # setup DNN

predictor = dlib.shape_predictor(PREDICTOR_PATH)


# not used
class TooManyFaces(Exception):
    pass


# not used
class NoFaces(Exception):
    pass


class FsImage:
    # This is a data class storing one image. containing path and filename, image itself, landmarks and number of faces
    def __init__(self):
        self.path_n_name = "EMPTY"
        self.im_content = None
        self.landmarks = None
        self.n_faces = 0

    def load_from_file(self, img_path):
        self.path_n_name = img_path
        while True:
            with open(img_path, 'rb') as f:
                check_chars = f.read()[-2:]
            if check_chars == b'\xff\xd9':
                break

        self.im_content = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if self.im_content.shape[1] > MAX_IMAGE_SIZE:
            factor = MAX_IMAGE_SIZE / self.im_content.shape[1]
            self.im_content = cv2.resize(self.im_content, (int(self.im_content.shape[1] * factor),
                                                           int(self.im_content.shape[0] * factor)))
        self.landmarks = get_landmarks(self.im_content)
        self.n_faces = len(self.landmarks)

    def load_from_CV2(self, img, img_path):
        self.path_n_name = img_path
        self.im_content = img
        self.landmarks = get_landmarks(self.im_content)
        self.n_faces = len(self.landmarks)

    def get_landmark(self, num):
        return self.landmarks[num]


def dotproduct(v1, v2):
    return sum((a*b) for a, b in zip(v1, v2))


def length(v):
    return math.sqrt(dotproduct(v, v))


def takeSecond(elem):
    return elem[1]


def get_landmarks(im):

    # unused code for face facing calculation
    width, height, _unused = im.shape
    c_x = width / 2
    c_y = height / 2
    f_x = c_x / numpy.tan(60 / 2 * numpy.pi / 180)
    f_y = f_x
    camera_matrix = numpy.float32([[f_x, 0.0, c_x],
                                   [0.0, f_y, c_y],
                                   [0.0, 0.0, 1.0]])
    lm_list = []
    if DETECTOR_TYPE == "DNN":
        rects = dlib.rectangles(0)
        rects_list = []

        num_of_face = 0
        (h, w) = im.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(im, (300, 300)), 1.0, (300, 300), (103.93, 116.77, 123.68))
        net.setInput(blob)
        detections = net.forward()

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]
            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > DNN_NOFACE_THRESH:  # find the face with highest conf
                # compute the (x, y)-coordinates of the bounding box for the
                # object
                box = detections[0, 0, i, 3:7] * numpy.array([w, h, w, h])
                [x1, y1, x2, y2] = box.astype("int")
                rects_list.append([[x1, y1, x2, y2], confidence])
                num_of_face += 1
        rects_list.sort(key=takeSecond, reverse=True)  # rank the faces with confidence, highest on top

        for i in range(0, num_of_face):
            rects.append(dlib.rectangle(*(rects_list[i][0])))  # push the faces into rects container

    elif DETECTOR_TYPE == "DLIB":
        rects = detector(im, 1)  # draw a box on detected face
        print("detected: ", len(rects))
    # if len(rects) == 0:
    #    raise NoFaces
    for i in range(0, len(rects)):  # calculate all landmarks
        _t_mat = [[p.x, p.y] for p in predictor(im, rects[i]).parts()]

        lm_list.append(numpy.matrix(_t_mat))
    print("lm_passed")
    return lm_list  # returns a list of landmarks


def calc_angle(v1, v2):  # not used, calculate angle between vectors
    t_ang = dotproduct(v1, v2) / (length(v1) * length(v2))
    t_ang = t_ang if abs(t_ang) < 1 else 1
    return math.acos(t_ang)


def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im


def draw_convex_hull(im, points, color):
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(im, points, color=color)


def get_face_mask(im, landmarks):
    im = numpy.zeros(im.shape[:2], dtype=numpy.float64)

    for group in OVERLAY_POINTS:
        draw_convex_hull(im,
                         landmarks[group],
                         color=1)

    im = numpy.array([im, im, im]).transpose((1, 2, 0))

    im = (cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
    im = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)

    return im


def transformation_from_points(points1, points2):
    """
    Return an affine transformation [s * R | T] such that:

        sum ||s*R*p1,i + T - p2,i||^2

    is minimized.

    """
    # Solve the procrustes problem by subtracting centroids, scaling by the
    # standard deviation, and then using the SVD to calculate the rotation. See
    # the following for more details:
    #   https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem

    points1 = points1.astype(numpy.float64)
    points2 = points2.astype(numpy.float64)

    c1 = numpy.mean(points1, axis=0)
    c2 = numpy.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    s1 = numpy.std(points1)
    s2 = numpy.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = numpy.linalg.svd(points1.T * points2)

    # The R we seek is in fact the transpose of the one given by U * Vt. This
    # is because the above formulation assumes the matrix goes on the right
    # (with row vectors) where as our solution requires the matrix to be on the
    # left (with column vectors).
    R = (U * Vt).T

    return numpy.vstack([numpy.hstack(((s2 / s1) * R,
                                       c2.T - (s2 / s1) * R * c1.T)),
                         numpy.matrix([0., 0., 1.])])


def read_im_and_landmarks(fname):
    print("start_reading_file")


    im = cv2.imread(fname, cv2.IMREAD_COLOR)
    print("Finished_reading_file")
    if im.im_content.shape[1] > MAX_IMAGE_SIZE:
        factor = MAX_IMAGE_SIZE / im.im_content.shape[1]
        im.im_content = cv2.resize(im.im_content, (int(im.im_content.shape[1] * factor),
                                                   int(im.im_content.shape[0] * factor)))

    s = get_landmarks(im)

    return im, s, fname


def warp_im(im, M, dshape):
    output_im = numpy.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return output_im


def correct_colours(im1, im2, landmarks1):
    blur_amount = COLOUR_CORRECT_BLUR_FRAC * numpy.linalg.norm(
                              numpy.mean(landmarks1[LEFT_EYE_POINTS], axis=0) -
                              numpy.mean(landmarks1[RIGHT_EYE_POINTS], axis=0))
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1
    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)

    # Avoid divide-by-zero errors.
    im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)

    return (im2.astype(numpy.float64) * im1_blur.astype(numpy.float64) /
                                                im2_blur.astype(numpy.float64))


def swap_face(ims1, ims2):
    tmp_set = []
    for back_img in ims1:  # get every image and landmark (every image)
        print('face0')
        for back_lm_lmk in back_img.landmarks:  # every back landmark and vector (every face), swap every face

            matrix = transformation_from_points(back_lm_lmk[ALIGN_POINTS],
                                                ims2[0].get_landmark(0)[ALIGN_POINTS])
            mask = get_face_mask(ims2[0].im_content, ims2[0].get_landmark(0))
            warped_mask = warp_im(mask, matrix, back_img.im_content.shape)
            combined_mask = numpy.max([get_face_mask(back_img.im_content, back_lm_lmk), warped_mask],
                                      axis=0)

            warped_im2 = warp_im(ims2[0].im_content, matrix, back_img.im_content.shape)
            warped_corrected_im2 = correct_colours(back_img.im_content, warped_im2, back_lm_lmk)
            back_img.im_content = back_img.im_content * (1.0 - combined_mask) + warped_corrected_im2 * combined_mask

        tmp_set.append(back_img)  # save that image

    return tmp_set


def pic_output(im_set, out_path):
    for im in im_set:
        if im.im_content.shape[1] > MAX_IMAGE_SIZE:
            factor = MAX_IMAGE_SIZE / im.im_content.shape[1]
            im.im_content = cv2.resize(im.im_content, (int(im.im_content.shape[1] * factor),
                            int(im.im_content.shape[0] * factor)))
        cv2.imwrite(out_path, im.im_content)


def save_face(im, lm):  # unused
    with open("face_file",'wb') as f:
        pickle.dump([im, lm], f)


def load_face():  # unused
    with open("face_file", 'rb') as f:
        return pickle.load(f)


def read_all_image(path):
    image_set = []
    for filename in glob.glob(os.path.join(path, '*.jpg')):
        im = FsImage()
        im.load_from_file(filename)
        image_set.append(im)
    return image_set


def func_read_face(raw_img_face_path):
    proc_img_face = read_all_image(raw_img_face_path)  # face
    return proc_img_face


def func_swap_with_face(proc_img_face, raw_img_back_path):
    proc_img_back = read_all_image(raw_img_back_path)
    output_img = swap_face(proc_img_back, proc_img_face)
    return output_img


def func_output_img(output_img_set, out_path):

    pic_output(output_img_set, out_path)
    return


if False:  # just commented out, for backend only developing
    state = "00"
    n_state = "00"
    while True:
        state = n_state
        if state == "00":
            print("1.gen and swap 2. generate face 3. swap with face")
            print("followed by file names (background_pic) (face_pic)")
            _cmd = input("Input command: (eg.: 1 a.jpg b.jpg)")
            cmd=_cmd.split()
            print(cmd)
            if cmd[0] == "1":
                if len(cmd) == 3:
                    im_back = read_all_image(cmd[1])  # head & background
                    im_face = read_all_image(cmd[2])  # face
                    output_im = swap_face(im_back, im_face)
                    pic_output(output_im)
                    print("done")
                    n_state = "00"
                else:
                    print("cmd error 1")
                    n_state = "00"
            elif cmd[0] == "2":
                if len(cmd) == 2:
                    im_face = read_all_image(cmd[1])  # face
                    save_face(im_face)
                    print("done")
                    n_state = "00"
                else:
                    print("cmd error 2")
                    n_state = "00"
            elif cmd[0] == "3":
                if len(cmd) == 2:
                    im_back = read_all_image(cmd[1])  # head & background
                    [im2, landmarks2] = load_face()
                    output_im = swap_face(im_back, im_face)
                    pic_output(output_im)
                    print("done")
                    n_state = "00"
                else:
                    print("cmd error 3")
            else:
                print("cmd error 0")
        else:
            print("error")




