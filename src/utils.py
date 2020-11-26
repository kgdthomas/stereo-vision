#!/bin/python3
import cv2 as cv
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import numpy as np

#Get all available cameras by trying to open them
def list_cameras(max_tests = 4):
	available_cameras = []

	for i in range(max_tests):
		v = cv.VideoCapture(i)
		s = v.open(i)

		if not s: continue

		available_cameras.append(i)
		v.release()

	return available_cameras

#Convert img in BGR (default in imread) to a pixmap for Qt representation
#https://gist.github.com/kashik0i/2119b4a937b67e2bbdacdfe95dfb644f
def img2pixmap(img, display_w, display_h, conversion = cv.COLOR_BGR2RGB, aspectRatio = Qt.IgnoreAspectRatio):
	"""Convert from an opencv image to QPixmap"""
	rgb_image = cv.cvtColor(img, conversion)
	h, w, ch = rgb_image.shape
	bytes_per_line = ch * w
	convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
	p = convert_to_Qt_format.scaled(display_w, display_h, aspectRatio)
	return QPixmap.fromImage(p)

#List cameras and add them as options in all the given dropdowns
def add_available_cameras(dropdown_dict):
	cameras = ['Off'] + [f'Camera {c}' for c in list_cameras()]
	for d in dropdown_dict.values(): d.addItems(cameras)

#Find chessboard corners in the img and draw them over the given image. This modifies the input img
def find_and_draw_chessboard(img, pattern_size):
	img_g = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	ret, corners = cv.findChessboardCorners(img_g, pattern_size, None)
	if not ret: return False

	refination_c = (cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS, 30, 0.01)
	cv.cornerSubPix(img_g, corners, (11, 11), (-1, -1), refination_c)
	cv.drawChessboardCorners(img, pattern_size, corners, ret)
	return True

#Create a QLabel for img representation with a darkgray background
def create_display(w, h):
	ql = QLabel()
	qp = QPixmap(w, h)
	qp.fill(QColor('darkGray'))
	ql.setPixmap(qp)
	return ql

#Create QSlider with the given config
def create_slider(min_value, max_value, tick_interval = 5, default_value = None, step = 1):
	slider = QSlider(Qt.Horizontal)
	slider.setMinimum(min_value)
	slider.setMaximum(max_value)
	slider.setTickPosition(QSlider.TicksBelow)
	slider.setTickInterval(tick_interval)
	slider.setSingleStep(step)
	if default_value is not None: slider.setValue(default_value)
	return slider

#Set a img in a given display. If img is not numpy.array, set the background
def update_display(w, h, display, img):
	if isinstance(img, np.ndarray): qp = img2pixmap(img, w, h)
	else:
		qp = QPixmap(w, h)
		qp.fill(QColor('darkGray'))

	display.setPixmap(qp)

#Create a QGroupBox with the given objects and a title for each one
def create_object_group(group_title, objects, col_span = 1):
	group = QGroupBox(group_title)
	groupLayout = QGridLayout()

	for i, (name, obj) in enumerate(objects.items()):
		groupLayout.addWidget(QLabel(name), 2*i, 0, 1, col_span, alignment=Qt.AlignTop)
		groupLayout.addWidget(obj, 2*i + 1, 0, 1, col_span, alignment=Qt.AlignTop)

	group.setLayout(groupLayout)
	return group

#Normalize disparity map to allow visualization
def disp2img(disp, minDisparity, numDisparities):
	return ((disp.astype(np.float32) / 16.0) - minDisparity) / numDisparities
