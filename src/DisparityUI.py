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

		self.blur_filters_mapping = [                   #Blur filters options mapping
			lambda src, k: cv.blur(src, (k, k)),
			lambda src, k: cv.GaussianBlur(src, (k, k), 0),
			lambda src, k: cv.medianBlur(src, k),
			lambda src, d, sigma, border: cv.bilateralFilter(src, d, sigma, sigma, border)
                ]
		self.preprocessing_options = [
			lambda src: self._apply_blur(src, 0, self.params['preprocessing']['values']['NBF']['kernelSize']),
			lambda src: self._apply_blur(src, 1, self.params['preprocessing']['values']['Gaussian filter']['kernelSize']),
			lambda src: self._apply_blur(src, 2, self.params['preprocessing']['values']['Median filter']['kernelSize']),
			lambda src: self.blur_filters_mapping[3](src,
				self.params['preprocessing']['values']['Bilateral filter']['diameter'],
				self.params['preprocessing']['values']['Bilateral filter']['sigma'],
				self.border_types[self.params['preprocessing']['values']['Bilateral filter']['border']]
			)
		]
		self.blur_filter_options = [
			lambda src: self._apply_blur(src, 0, self.params['filter']['blur_filter']['values']['NBF']['kernelSize']),
			lambda src: self._apply_blur(src, 1, self.params['filter']['blur_filter']['values']['Gaussian filter']['kernelSize']),
			lambda src: self._apply_blur(src, 2, self.params['filter']['blur_filter']['values']['Median filter']['kernelSize']),
			lambda src: self.blur_filters_mapping[3](src,
				self.params['filter']['blur_filter']['values']['Bilateral filter']['diameter'],
				self.params['filter']['blur_filter']['values']['Bilateral filter']['sigma'],
				self.border_types[self.params['filter']['blur_filter']['values']['Bilateral filter']['border']]
			)
		]
		self.plot3d_options = [                         #3D reconstruction options mapping
			lambda img, disp, Q: show_point_cloud(img, disp, Q),
			lambda img, disp, Q: show_bpa_mesh(img, disp, Q,
				self.params['3dconf']['values']['3D BPA Mesh']['radius'],
				self.params['3dconf']['values']['3D BPA Mesh']['num_triangles'],
				self.params['matcher']['minDisparity'],
				self.params['matcher']['numDisparities'],
				verbose = self.verbose
			),
			lambda img, disp, Q: show_poisson_mesh(img, disp, Q,
				self.params['3dconf']['values']['3D Poisson Mesh']['depth'],
				self.params['3dconf']['values']['3D Poisson Mesh']['scale'],
				self.params['3dconf']['values']['3D Poisson Mesh']['linear_fit'],
				self.params['matcher']['minDisparity'],
				self.params['matcher']['numDisparities'],
				verbose = self.verbose
			)
		]
		self.border_types = [cv.BORDER_DEFAULT, cv.BORDER_CONSTANT, cv.BORDER_REPLICATE, cv.BORDER_WRAP, cv.BORDER_REFLECT, cv.BORDER_TRANSPARENT, cv.BORDER_ISOLATED]

		self.timer = QTimer()					#Timer para ejecutar el refresco de frames
		self.timer.timeout.connect(lambda: self._update_images())

		#Create layout
		grid = QGridLayout()

		#Create elements
		list_cameras_button = QPushButton("List cameras")					#List cameras
		self.camera_dropdown = {'l': QComboBox(), 'r': QComboBox()}				#Select left and right camera
		self.camera_display = {'l': self._create_display(), 'r': self._create_display()}	#Left and right camera displays

		self.online_disparity_checkbox = QCheckBox("Real-time disparity")			#Realtime disparity (recalculate every frame)

		self.preprocessing_filters_objects = {          #Filtros de preprocesado de imagen
			'NBF' : {'objects' : {'kernelSize' : create_slider(1, 200)}},
			'Gaussian filter' : {'objects' : {'kernelSize' : create_slider(1, 200)}},
			'Median filter' : {'objects' : {'kernelSize' : create_slider(1, 200)}},
			'Bilateral filter' : {
				'objects' : {
					'diameter' : create_slider(1, 10, default_value = 5),
					'sigma' : create_slider(1, 300, tick_interval = 10, default_value = 30),
					'border' : create_combobox(['DEFAULT', 'CONSTANT', 'REPLICATE', 'WRAP', 'REFLECT', 'TRANSPARENT', 'ISOLATED'])
				},
				'cols' : 3
			}
		}
		self.preprocessing_filters_config = create_tabs_group(self.preprocessing_filters_objects)
		self.use_preprocessing_checkbox = QCheckBox("Image preprocessing")

		self.matcher_sliders = {								#Opencv StereoSGBM_create params
			'minDisparity' : create_slider(1, 256, default_value = 46),			#https://docs.opencv.org/3.4/d2/d85/classcv_1_1StereoSGBM.html
			'numDisparities' : create_slider(1, 512, default_value = 97),
			'blockSize' : create_slider(0, 256, default_value = 23),
			'P1' : create_slider(8, 1024, default_value = 335),
			'P2' : create_slider(8, 2048, default_value = 1024),
			'disp12MaxDiff' : create_slider(1, 640, default_value = 140),
			'preFilterCap' : create_slider(0, 1000),
			'uniquenessRatio' : create_slider(0, 15, default_value = 1),
			'speckleWindowSize' : create_slider(10, 30, default_value = 14),
			'speckleRange' : create_slider(0, 200, default_value = 27),
			'mode' : create_combobox(["MODE_SGBM", "MODE_HH", "MODE_SGBM_3WAY", "MODE_HH4"])
		}

		self.blur_filter_objects = {		#Filtros de blur al disparity map
			'NBF' : {'objects' : {'kernelSize' : create_slider(1, 200)}},
			'Gaussian filter' : {'objects' : {'kernelSize' : create_slider(1, 200)}},
			'Median filter' : {'objects' : {'kernelSize' : create_slider(1, 200)}},
			'Bilateral filter' : {
				'objects' : {
					'diameter' : create_slider(1, 10, default_value = 5),
					'sigma' : create_slider(1, 300, tick_interval = 10, default_value = 30),
					'border' : create_combobox(['DEFAULT', 'CONSTANT', 'REPLICATE', 'WRAP', 'REFLECT', 'TRANSPARENT', 'ISOLATED'])
				},
				'cols' : 3
			}
		}
		self.filter_sliders = {			#Parametros de los filtros al disparity map
			'pre_maxSpeckleSize' : create_slider(0, 1000, tick_interval=500),				#Speckle filter
			'pre_maxDiff' : create_slider(0, 50),
			'blur_filter' : create_tabs_group(self.blur_filter_objects),					#Blur filters: https://www.docs.opencv.org/master/dc/dd3/tutorial_gausian_median_blur_bilateral_filter.html
			'lambda' : create_slider(0, 16000, default_value = 8000, tick_interval = 500, step=100),	#Opencv DisparityWLSFilter: https://docs.opencv.org/3.1.0/d3/d14/tutorial_ximgproc_disparity_filtering.html#gsc.tab=0
			'sigma' : create_slider(0, 40, default_value = 1.5, step = 0.1),				#https://docs.opencv.org/3.1.0/d9/d51/classcv_1_1ximgproc_1_1DisparityWLSFilter.html
			'post_maxSpeckleSize' : create_slider(0, 1000, tick_interval=500),
			'post_maxDiff' : create_slider(0, 50)
		}
		self.use_filters = {                    #Checkbox para activa cada uno de los filtros
			'pre-speckle' : QCheckBox('Use pre speckle filter'), 'blur' : QCheckBox('Use post blur filter'),
			'wls' : QCheckBox('Use wls filter'), 'post-speckle' : QCheckBox('Use post speckle filter')
		}

		self.crop_sliders = {									#Manual image cropping params
			'x1' : create_slider(0, 640), 'x2' : create_slider(0, 640, default_value=640),
			'y1' : create_slider(0, 640), 'y2' : create_slider(0, 640, default_value=640)
		}
		self.apply_crop_checkbox = QCheckBox('Apply crop')					#Enable/disable image cropping

		self.plot3d_objects = {                 #3D plot config
			'3D Point cloud' : {'objects' : { 'label' : QLabel('No parameters are required') }},
			'3D BPA Mesh' : {
				'objects' : {
					'radius' : create_slider(0, 100, tick_interval = 25, default_value=0, scale=0.01, type=float),
					'num_triangles' : create_slider(0, 500000, tick_interval=10000, default_value=100000, step=5000)
				},
				'cols' : 2
			},
			'3D Poisson Mesh' : {
				'objects' : {
					'depth' : create_slider(1, 16, default_value = 8),
					'scale' : create_slider(10, 30, default_value = 11, tick_interval=10, scale=0.1, type=float),
					'linear_fit' : QCheckBox('Linear fit')
				},
				'cols' : 3
			}
		}
		self.plot3d_config = create_tabs_group(self.plot3d_objects)

		self.every_checkbox = {'onlineDisp' : self.online_disparity_checkbox, 'preprocessing' : self.use_preprocessing_checkbox, **self.use_filters, 'crop' : self.apply_crop_checkbox}
		manual_disparity_button = QPushButton("Show disparity map")				#Manually show disparity map
		depth_map = QPushButton("Show depth map")						#Mapa de profundidad
		reconstruct_3d = QPushButton("Reconstruir 3D")						#3D reconstruction and rendering
		save_current_button = QPushButton("Save current at")					#Save current images and params

		self.save_current_path = QLineEdit(kargs['default_test_path'])				#Save path input field
													#A folder is created as the timestamp to avoid overwritting

		#Add callbacks and props
		list_cameras_button.clicked.connect(lambda: add_available_cameras(self.camera_dropdown))
		self.camera_dropdown['l'].currentIndexChanged.connect(lambda i: self._on_selected_camera_changed('l', i))
		self.camera_dropdown['r'].currentIndexChanged.connect(lambda i: self._on_selected_camera_changed('r', i))

		for conf in self.preprocessing_filters_objects.values():                #Callback in the pipeline parameters
			for obj in conf['objects'].values(): set_object_callback(obj, self._param_changed)
		set_object_callback(self.preprocessing_filters_config, self._param_changed)
		for s in self.matcher_sliders.values(): set_object_callback(s, self._param_changed)
		for s in self.filter_sliders.values(): set_object_callback(s, self._param_changed)
		for conf in self.blur_filter_objects.values():
			for obj in conf['objects'].values(): set_object_callback(obj, self._param_changed)
		for s in self.crop_sliders.values(): set_object_callback(s, self._param_changed)
		for c in self.every_checkbox.values(): set_object_callback(c, self._param_changed)
		for conf in self.plot3d_objects.values():
			for obj in conf['objects'].values(): set_object_callback(obj, self._param_changed)
		set_object_callback(self.plot3d_config, self._param_changed)

		set_object_callback(manual_disparity_button, self.update_disparity)
		set_object_callback(depth_map, lambda: self.update_disparity(depth_map = True))
		set_object_callback(reconstruct_3d, lambda: self.update_disparity(plot3d = True))
		set_object_callback(save_current_button, self._save_current)

		#Populate layout
		grid.addWidget(list_cameras_button, 0, 0, 1, 24, alignment=Qt.AlignTop)
		grid.addWidget(self.camera_dropdown['l'], 1, 0, 1, 12, alignment=Qt.AlignTop)
		grid.addWidget(self.camera_dropdown['r'], 1, 12, 1, 12, alignment=Qt.AlignTop)

		grid.addWidget(self.camera_display['l'], 2, 0, 1, 12, alignment=Qt.AlignTop)
		grid.addWidget(self.camera_display['r'], 2, 12, 1, 12, alignment=Qt.AlignTop)

		grid.addWidget(self.preprocessing_filters_config, 3, 0, 6, 12, alignment=Qt.AlignTop)
		grid.addWidget(self._create_matcher_group(), 7, 0, 12, 12, alignment=Qt.AlignTop)
		grid.addWidget(self._create_filter_group(), 3, 12, 10, 12, alignment=Qt.AlignTop)
		grid.addWidget(self._create_crop_group(), 13, 12, 4, 12, alignment=Qt.AlignTop)
		grid.addWidget(self.plot3d_config, 16, 12, 4, 12, alignment=Qt.AlignTop)

		grid.addWidget(self._create_checkbox_group(), 19, 0, 1, 24, alignment=Qt.AlignTop)
		grid.addWidget(manual_disparity_button, 20, 0, 1, 5, alignment=Qt.AlignBottom)
		grid.addWidget(depth_map, 20, 5, 1, 5, alignment=Qt.AlignBottom)
		grid.addWidget(reconstruct_3d, 20, 10, 1, 5, alignment=Qt.AlignBottom)
		grid.addWidget(save_current_button, 20, 15, 1, 5, alignment=Qt.AlignBottom)
		grid.addWidget(self.save_current_path, 20, 20, 1, 4, alignment=Qt.AlignBottom)

#		grid.addWidget(QLabel("Save current path"), 7, 3, 1, 3, alignment=Qt.AlignBottom)
#		grid.addWidget(self.save_current_path, 8, 3, 1, 3, alignment=Qt.AlignBottom)

		#Add layout
		self.setLayout(grid)

		#Run timer
		self._param_changed()
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
		if self.params['checkbox']['onlineDisp']: self.update_disparity()
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
	def _create_matcher_group(self): return create_object_group("Matcher parameters", self.matcher_sliders, cols = 2, fullwidth = [0,1,2])
	def _create_filter_group(self): return create_object_group("Disparity filtering", self.filter_sliders, cols = 2, fullwidth = [2])
	def _create_crop_group(self): return create_object_group("Crop parameters", self.crop_sliders, cols = 4)
	def _create_checkbox_group(self): return create_object_group(None, self.every_checkbox, cols = 7)

#	def _create_slider_group(self): return create_object_group("Matcher parameters", self.matcher_sliders)
#	def _create_filter_group(self): return create_object_group("Filter parameters", self.filter_sliders)
#	def _create_crop_group(self): return create_object_group("Crop parameters", self.crop_sliders)

	#Update stored params whenever a parameter changed
	def _param_changed(self):
		#Update the params that are stored based on slider values
		self.params = {
			'preprocessing' : {'currentTab' : get_object_value(self.preprocessing_filters_config), 'values' : {tab_name : {
				name : get_object_value(obj) for name, obj in conf['objects'].items()
			} for tab_name, conf in self.preprocessing_filters_objects.items()}},
			'matcher' : {name : get_object_value(s) for name,s in self.matcher_sliders.items()},
			'filter' : {name : (get_object_value(s) if not isinstance(s, QTabWidget) else {'currentTab' : get_object_value(s), 'values' : {tab_name : {
				name : get_object_value(obj) for name, obj in conf['objects'].items()
			} for tab_name, conf in self.blur_filter_objects.items()}}) for name,s in self.filter_sliders.items()},
			'crop' : {name : get_object_value(s) for name,s in self.crop_sliders.items()},
			'checkbox' : {name : get_object_value(c) for name,c in self.every_checkbox.items()},
			'3dconf' : {'currentTab' : get_object_value(self.plot3d_config), 'values' : {tab_name : {
				name : get_object_value(obj) for name,obj in conf['objects'].items()
			} for tab_name, conf in self.plot3d_objects.items()}}
		}
		if self.verbose: print(self.params)
		#Let the camera tick update disparity
		#if self.online_disparity_checkbox.isChecked(): self.update_disparity()

	#Crop image
	def _crop_img(self, img):
		l = {k : v for k,v in self.params['crop'].items()}
		return img[l['y1']:l['y2'], l['x1']:l['x2']]

	def _apply_blur(self, img, filterType, kernelSize):
		if filterType != 0 and kernelSize % 2 == 0: kernelSize += 1
		return self.blur_filters_mapping[filterType](img, kernelSize)

	def update_disparity(self, depth_map = False, plot3d = False):
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

		#Preprocessing
		if self.params['checkbox']['preprocessing']:
			rect_l = self.preprocessing_options[self.params['preprocessing']['currentTab']](rect_l)
			rect_r = self.preprocessing_options[self.params['preprocessing']['currentTab']](rect_r)

		#Create matcher and compute disparity map
		stereo = cv.StereoSGBM_create(**self.params['matcher'])
		disp = stereo.compute(rect_l, rect_r)

		#Pre-speckle
		if self.params['checkbox']['pre-speckle']:
			cv.filterSpeckles(disp, 0, self.params['filter']['pre_maxSpeckleSize'], self.params['filter']['pre_maxDiff'])

		#Blur filter
		if self.params['checkbox']['blur']:
			#Disparity to image
			_img = disp2img(disp, self.params['matcher']['minDisparity'], self.params['matcher']['numDisparities'])
			#Convert from float32 to uint8
			_img = cv.cvtColor(cv.normalize(_img, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U), cv.COLOR_GRAY2BGR)

			#Filter image
			_img = self.blur_filter_options[self.params['filter']['blur_filter']['currentTab']](_img)

			#Convert to uint8 to float32
			_img = (cv.cvtColor(_img, cv.COLOR_BGR2GRAY) / 255).astype("float32")
			#Image to disparity
			disp = img2disp(_img, self.params['matcher']['minDisparity'], self.params['matcher']['numDisparities'])

		#WLS
		if self.params['checkbox']['wls']:
			right_disp = cv.ximgproc.createRightMatcher(stereo).compute(rect_r, rect_l)
			wls_filter = cv.ximgproc.createDisparityWLSFilter(stereo)
			wls_filter.setLambda(self.params['filter']['lambda'])
			wls_filter.setSigmaColor(self.params['filter']['sigma'])
			disp = wls_filter.filter(disp, rect_l, disparity_map_right=right_disp)

		#Post-speckle
		if self.params['checkbox']['post-speckle']:
			cv.filterSpeckles(disp, 0, self.params['filter']['post_maxSpeckleSize'], self.params['filter']['post_maxDiff'])

		#Normalize disparity to show as image
		disp_norm = disp2img(disp, self.params['matcher']['minDisparity'], self.params['matcher']['numDisparities'])
		#Crop if requested
		if self.params['checkbox']['crop']: disp_norm = self._crop_img(disp_norm)
		#Show
		cv.imshow("Disparity map", disp_norm)

		#Depth map
		if depth_map:
			Q = self.parentWindow.calibration.cw.rectification_results['disparity2depth_mappingMatrix']
			if self.params['checkbox']['crop']: disp = self._crop_img(disp)
			show_depth_map(disp, Q)

		#If 3dplot is requested
		if plot3d:
			#Fetch perspective transformation matrix from the calibration process
			Q = self.parentWindow.calibration.cw.rectification_results['disparity2depth_mappingMatrix']

			#Crop both disparity and image if requested
			if self.params['checkbox']['crop']:
				disp = self._crop_img(disp)
				rect_l = self._crop_img(rect_l)

			self.plot3d_options[self.params['3dconf']['currentTab']](rect_l, disp, Q)

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
