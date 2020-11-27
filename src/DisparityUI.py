#!/bin/python3
import cv2 as cv
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import open3d as o3d
from datetime import datetime as dt
import sys
import os
import pickle

from utils import *

#UI tab to perform disparity map calculation
class DisparityUI(QWidget):
	def __init__(self, parent, **kargs):
		super(DisparityUI, self).__init__()
		self.parentWindow = parent
		self.display_sizes = {'w' : 300, 'h' : 200}
		self.video_input = {'l' : None, 'r' : None}		#Aqui se guardan los objetos cv.VideoCapture o None si la camara no esta activa
		self.frames = {'l' : None, 'r' : None}			#Aqui se guardan los ultimos frames capturados para cada camara
		self.params = None					#Aqui se guardan los parametros configurados actualmente
		self.busy_camera = False				#Flag para indicar si la representacion de frames esta teniendo lugar. Para evitar encolamiento
		self.busy_disparity = False				#Parecido pero para el calculo de disparidades
		self.verbose = kargs['verbose']

		self.timer = QTimer()					#Timer para ejecutar el refresco de frames
		self.timer.timeout.connect(lambda: self._update_images())

		#Create layout
		grid = QGridLayout()

		#Create elements
		list_cameras_button = QPushButton("List cameras")					#List cameras
		self.camera_dropdown = {'l': QComboBox(), 'r': QComboBox()}				#Select left and right camera
		self.camera_display = {'l': self._create_display(), 'r': self._create_display()}	#Left and right camera displays

		self.matcher_sliders = {								#Opencv StereoSGBM_create params
			'minDisparity' : create_slider(2, 256, default_value = 46),			#https://docs.opencv.org/3.4/d2/d85/classcv_1_1StereoSGBM.html
			'numDisparities' : create_slider(2, 512, default_value = 97),
			'blockSize' : create_slider(2, 256, default_value = 23),
			'P1' : create_slider(8, 1024, default_value = 335),
			'P2' : create_slider(8, 2048, default_value = 1024),
			'disp12MaxDiff' : create_slider(1, 640, default_value = 140),
			'uniquenessRatio' : create_slider(0, 10, default_value = 1, step = 0.05),
			'speckleWindowSize' : create_slider(10, 30, default_value = 14),
			'speckleRange' : create_slider(0, 200, default_value = 27)
		}
		self.online_disparity_checkbox = QCheckBox("Real-time dispartiy")			#Enable/disable "realtime" disparity map

		self.filter_sliders = {											#Opencv DisparityWLSFilter params
			'lambda' : create_slider(0, 16000, default_value = 8000, tick_interval = 500, step=100),	#https://docs.opencv.org/3.1.0/d3/d14/tutorial_ximgproc_disparity_filtering.html#gsc.tab=0
			'sigma' : create_slider(0, 40, default_value = 1.5, step = 0.1)					#https://docs.opencv.org/3.1.0/d9/d51/classcv_1_1ximgproc_1_1DisparityWLSFilter.html
		}
		self.use_filter_checkbox = QCheckBox("Use filter")					#Enable/disable the usage of the filter

		self.crop_sliders = {									#Manual image cropping params
			'x1' : create_slider(0, 640, default_value=0),
			'x2' : create_slider(0, 640, default_value=640),
			'y1' : create_slider(0, 640, default_value=0),
			'y2' : create_slider(0, 640, default_value=640)
		}
		self.apply_crop_checkbox = QCheckBox('Apply crop')					#Enable/disable image cropping

		manual_disparity_button = QPushButton("Show disparity map")				#Manually show disparity map
		reconstruct_3d = QPushButton("Reconstruir 3D")						#3D reconstruction and rendering
		save_current_button = QPushButton("Save current")					#Save current images and params

		self.save_current_path = QLineEdit(kargs['default_test_path'])				#Save path input field
													#Images are saved with the timestamp in the name to avoid overwritting

		#Add callbacks and props
		list_cameras_button.clicked.connect(lambda: add_available_cameras(self.camera_dropdown))
		self.camera_dropdown['l'].currentIndexChanged.connect(lambda i: self._on_selected_camera_changed('l', i))
		self.camera_dropdown['r'].currentIndexChanged.connect(lambda i: self._on_selected_camera_changed('r', i))
		for s in self.matcher_sliders.values(): s.valueChanged.connect(lambda: self._param_changed())
		for s in self.filter_sliders.values(): s.valueChanged.connect(lambda: self._param_changed())
		for s in self.crop_sliders.values(): s.valueChanged.connect(lambda: self._param_changed())
		self.online_disparity_checkbox.stateChanged.connect(lambda: self._param_changed())
		self.use_filter_checkbox.clicked.connect(lambda: self._param_changed())
		self.apply_crop_checkbox.stateChanged.connect(lambda: self._param_changed())

		manual_disparity_button.clicked.connect(lambda: self.update_disparity())
		reconstruct_3d.clicked.connect(lambda: self.update_disparity(True))
		save_current_button.clicked.connect(lambda: self._save_current())

		#Populate layout
		grid.addWidget(list_cameras_button, 0, 0, 1, 6, alignment=Qt.AlignTop)
		grid.addWidget(self.camera_dropdown['l'], 1, 0, 1, 3, alignment=Qt.AlignTop)
		grid.addWidget(self.camera_dropdown['r'], 1, 3, 1, 3, alignment=Qt.AlignTop)

		grid.addWidget(self.camera_display['l'], 2, 0, 1, 3, alignment=Qt.AlignTop)
		grid.addWidget(self.camera_display['r'], 2, 3, 1, 3, alignment=Qt.AlignTop)

		grid.addWidget(self._create_slider_group(), 3, 0, 7, 3, alignment=Qt.AlignTop)
		grid.addWidget(self._create_filter_group(), 3, 3, 1, 3, alignment=Qt.AlignTop)
		grid.addWidget(self._create_crop_group(), 4, 3, 1, 3, alignment=Qt.AlignTop)
		grid.addWidget(self.online_disparity_checkbox, 5, 3, 1, 1, alignment=Qt.AlignTop)
		grid.addWidget(self.use_filter_checkbox, 5, 4, 1, 1, alignment=Qt.AlignTop)
		grid.addWidget(self.apply_crop_checkbox, 5, 5, 1, 1, alignment=Qt.AlignTop)

		grid.addWidget(QLabel("Save current path"), 7, 3, 1, 3, alignment=Qt.AlignBottom)
		grid.addWidget(self.save_current_path, 8, 3, 1, 3, alignment=Qt.AlignBottom)

		grid.addWidget(manual_disparity_button, 10, 0, 1, 2, alignment=Qt.AlignBottom)
		grid.addWidget(reconstruct_3d, 10, 2, 1, 2, alignment=Qt.AlignBottom)
		grid.addWidget(save_current_button, 10, 4, 1, 2, alignment=Qt.AlignBottom)

		#Add layout
		self.setLayout(grid)

		#Run timer
		self.timer.start(1000)

	#Update image displays and disparity map if online disparity checkbox is checked
	def _update_images(self):
		#Verify tab and that images/disparity map are not being calculated
		if self.parentWindow.tabs.currentIndex() != 1: return
		if self.busy_camera or self.busy_disparity: return
		self.busy_camera = True

		#Update displays for all active video captures
		for side, vinput in self.video_input.items():
			if vinput is None: continue

			ret, frame = vinput.read()
			self._update_display(side, frame)
			if frame is not None: self.frames[side] = frame

		#Update disparity if requested
		if self.online_disparity_checkbox.isChecked(): self.update_disparity()
		self.busy_camera = False

	#When camera dropdown is changed
	def _on_selected_camera_changed(self, side, selected_index):
		#Release the camera if in use before changing
		if self.video_input[side] is not None: self.video_input[side].release()

		#Check if new camera is Off
		current_camera = self.camera_dropdown[side].currentText()
		if current_camera == 'Off':
			self.video_input[side] = None
			self._update_display(side, None)
			return

		#Check if new camera is already in use
		if current_camera in list(d.currentText() for s,d in self.camera_dropdown.items() if s != side):
			if self.verbose: print('Camera already in use')
			self.camera_dropdown[side].setCurrentIndex(0)
			return

		#Try get and open video capture
		try: camera_id = int(current_camera.split(" ")[-1])
		except: return
		self.video_input[side] = cv.VideoCapture(camera_id)

	#Wrappers
	def _create_display(self): return create_display(self.display_sizes['w'], self.display_sizes['h'])
	def _update_display(self, side, img = None): update_display(self.display_sizes['w'], self.display_sizes['h'], self.camera_display[side], img)
	def _create_slider_group(self): return create_object_group("Matcher parameters", self.matcher_sliders)
	def _create_filter_group(self): return create_object_group("Filter parameters", self.filter_sliders)
	def _create_crop_group(self): return create_object_group("Crop parameters", self.crop_sliders)

	#Update stored params whenever a parameter changed
	def _param_changed(self):
		self.params = {
			'matcher' : {name : s.value() for name,s in self.matcher_sliders.items()},
			'filter' : {name : s.value() for name,s in self.filter_sliders.items()},
			'crop' : {name : s.value() for name,s in self.crop_sliders.items()}
		}
		if self.verbose: print(params)
		#Let the camera tick update disparity
		#if self.online_disparity_checkbox.isChecked(): self.update_disparity()

	#Crop image
	def _crop_img(self, img):
		l = {k : v for k,v in self.params['crop'].items()}
		return img[l['y1']:l['y2'], l['x1']:l['x2']]

	def update_disparity(self, plot3d = False):
		#Verify disparity map is not being calculated
		if self.busy_disparity: return
		self.busy_disparity = True

		#Ensure both frames are valid
		if any(f is None for f in self.frames.values()):
			self.busy_disparity = False
			return

		#Fetch and rectify both frames
		img_l, img_r = self.frames['l'], self.frames['r']
		rect_l, rect_r = self.parentWindow.calibration.cw.rectify_image_pair(img_l, img_r)

		#Create matcher and compute disparity map
		stereo = cv.StereoSGBM_create(**self.params['matcher'])
		disp = stereo.compute(rect_l, rect_r)

		#If filter usage is not requested
		if not self.use_filter_checkbox.isChecked():
			#Normalize disparity to show as image
			disp_norm = disp2img(disp, self.params['matcher']['minDisparity'], self.params['matcher']['numDisparities'])
			#Crop if requested
			if self.apply_crop_checkbox.isChecked(): disp_norm = self._crop_img(disp_norm)
			#Show
			cv.imshow("Disparity map", disp_norm)

		#If filter usage is requested
		else:
			#Calculate right camera disparity
			right_disp = cv.ximgproc.createRightMatcher(stereo).compute(rect_l, rect_r)
			#Create filter and adjust parameters
			wls_filter = cv.ximgproc.createDisparityWLSFilter(stereo)
			wls_filter.setLambda(self.params['filter']['lambda'])
			wls_filter.setSigmaColor(self.params['filter']['sigma'])
			#Compute filtered disparity
			disp = wls_filter.filter(disp, rect_l, disparity_map_right=right_disp)
			#Normalize, crop and show
			disp_norm = disp2img(disp, self.params['matcher']['minDisparity'], self.params['matcher']['numDisparities'])
			if self.apply_crop_checkbox.isChecked(): disp_norm = self._crop_img(disp_norm)
			cv.imshow("Disparity map", disp_norm)

		#If 3dplot is requested
		if plot3d:
			#Fetch perspective transformation matrix from the calibration process
			Q = self.parentWindow.calibration.cw.rectification_results['disparity2depth_mappingMatrix']

			#Crop both disparity and image if requested
			if self.apply_crop_checkbox.isChecked():
				disp = self._crop_img(disp)
				rect_l = self._crop_img(rect_l)

			#Reproyect to 3d and extract points and colors
			points = cv.reprojectImageTo3D(disp, Q)
			colors = cv.cvtColor(rect_l, cv.COLOR_BGR2RGB)
			#Filter out those with min disparity (black)
			mask = disp > disp.min()
			p, c = points[mask], colors[mask]

			#Create point cloud
			pcd = o3d.geometry.PointCloud()
			pcd.points = o3d.utility.Vector3dVector(p)
			pcd.colors = o3d.utility.Vector3dVector(c / 255)
			o3d.visualization.draw_geometries([pcd])

		self.busy_disparity = False

	def _save_current(self):
		#Fetch both frames and ensure they are valid
		img_l, img_r = self.frames['l'], self.frames['r']
		if img_l is None or img_r is None:
			if self.verbose: print('Refused to save None frame')
			return

		#Create a new folder with the current timestamp
		path = '{}/{}'.format(self.save_current_path.text(), int(dt.now().timestamp()))
		os.mkdir(path)
		if self.verbose: print('Created folder at {}'.format(path))

		#Save both images and all the current parameters
		cv.imwrite('{}/cameraL.png'.format(path), img_l)
		cv.imwrite('{}/cameraR.png'.format(path), img_r)
		with open('{}/params.pickle'.format(path), 'wb') as f: pickle.dump(self.params, f)
		if self.verbose: print('Saved cameraL.png, cameraR.png, params.pickle at {}'.format(path))
