#!/bin/python3
import cv2 as cv
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import open3d as o3d
import pickle
import os
from datetime import datetime as dt

from utils import *

#UI tab to perform offline disparity map calculations
class OfflineUI(QWidget):
	def __init__(self, parent, **kargs):
		super(OfflineUI, self).__init__()
		self.parentWindow = parent
		self.display_size = {'w' : 300, 'h' : 200}
		self.frames = {'l' : None, 'r' : None}		#Aqui se guardan las imagenes actuales
		self.params = {}				#Aqui se guardan los parametros actuales
		self.update_params = True			#Flag para que los parametros se actualizen segun cambian los sliders
		self.verbose = kargs['verbose']
		self.blur_filters = [
			lambda src, k: cv.blur(src, (k, k)),
			lambda src, k: cv.GaussianBlur(src, (k, k), 0),
			lambda src, k: cv.medianBlur(src, k)
		]

		#Create layout
		grid = QGridLayout()

		#Create elements
		self.load_config = {		#Formulario para seleccionar y cargar imagenes
			'path' : QLineEdit(kargs['default_test_path']), 'search' : QPushButton("Buscar carpetas"),
			'folder' : QComboBox(), 'load' : QPushButton("Cargar ficheros")
		}
		self.image_display = {		#Displays para las imagenes
			'l' : self._create_display(), 'l_r' : self._create_display(),
			'r' : self._create_display(), 'r_r' : self._create_display()
		}

		self.preblur_filter = {
			'filterType' : create_combobox(['Normalized block filter', 'Gaussian filter', 'Median filter']),
			'kernelSize' : create_slider(1, 200)
		}
		self.use_preblur_checkbox = QCheckBox("Blur original image")

		self.matcher_sliders = {	#Parametros del matcher
			'minDisparity' : create_slider(1, 256),
			'numDisparities' : create_slider(1, 512),
			'blockSize' : create_slider(0, 256),
			'P1' : create_slider(8, 1024),
			'P2' : create_slider(8, 2048),
			'disp12MaxDiff' : create_slider(1, 640),
			'preFilterCap' : create_slider(0, 1000),
			'uniquenessRatio' : create_slider(0, 10, tick_interval = 0.05, step = 0.05),
			'speckleWindowSize' : create_slider(10, 30),
			'speckleRange' : create_slider(0, 200),
			'mode' : create_combobox(["MODE_SGBM", "MODE_HH", "MODE_SGBM_3WAY", "MODE_HH4"])
		}
		self.online_disparity_checkbox = QCheckBox("Real-time disparity")

		self.filter_sliders = {		#Parametros de los filtros
			'pre_maxSpeckleSize' : create_slider(0, 200),
			'pre_maxDiff' : create_slider(0, 100),
#			'blur_filterType' : create_combobox(['Normalized block filter', 'Gaussian filter', 'Median filter']),
#			'blur_kernelSize' : create_slider(1, 200),
			'lambda' : create_slider(0, 24000, default_value = 8000, tick_interval=500, step=100),
			'sigma' : create_slider(0, 40, default_value = 1.5, step = 0.1),
			'post_maxSpeckleSize' : create_slider(0, 200),
			'post_maxDiff' : create_slider(0, 100)
		}
		self.use_filters = {
			'pre-speckle' : QCheckBox('Use pre speckle filter'),
#			'blur' : QCheckBox('Use post blur filter'),
			'wls' : QCheckBox('Use wls filter'),
			'post-speckle' : QCheckBox('Use post speckle filter')
		}

		self.crop_sliders = {		#Limites de recortado
			'x1' : create_slider(0, 640),
			'x2' : create_slider(0, 640, default_value = 640),
			'y1' : create_slider(0, 640, default_value = 0),
			'y2' : create_slider(0, 640, default_value = 640)
		}
		self.apply_crop_checkbox = QCheckBox("Apply crop")

		manual_disparity_button = QPushButton("Show disparity map") 	#Acciones manuales
		reconstruct_3d = QPushButton("Reconstruir 3D")
		save_current_btn = QPushButton("Save current")

		#Callbacks and props
		self.load_config['search'].clicked.connect(lambda: self._search_folders())
		self.load_config['load'].clicked.connect(lambda: self._load_files())

		for s in self.preblur_filter.values(): set_object_callback(s, self._param_changed)
		for s in self.matcher_sliders.values(): set_object_callback(s, self._param_changed)
		for s in self.filter_sliders.values(): set_object_callback(s, self._param_changed)
		for s in self.crop_sliders.values(): s.valueChanged.connect(lambda: self._param_changed())
		self.online_disparity_checkbox.stateChanged.connect(lambda: self._param_changed())
		self.use_preblur_checkbox.stateChanged.connect(lambda: self._param_changed())
		for f in self.use_filters.values(): f.clicked.connect(lambda : self._param_changed())
		self.apply_crop_checkbox.stateChanged.connect(lambda: self._param_changed())

		manual_disparity_button.clicked.connect(lambda: self.update_disparity())
		reconstruct_3d.clicked.connect(lambda: self.update_disparity(True))
		save_current_btn.clicked.connect(lambda: self._save_current())

		#Populate layout
		grid.addWidget(self.load_config['path'], 0, 0, 1, 6, alignment=Qt.AlignTop)
		grid.addWidget(self.load_config['search'], 1, 0, 1, 6, alignment=Qt.AlignTop)
		grid.addWidget(self.load_config['folder'], 0, 6, 1, 6, alignment=Qt.AlignTop)
		grid.addWidget(self.load_config['load'], 1, 6, 1, 6, alignment=Qt.AlignTop)

		grid.addWidget(self.image_display['l'], 2, 0, 1, 3, alignment=Qt.AlignTop)
		grid.addWidget(self.image_display['l_r'], 2, 3, 1, 3, alignment=Qt.AlignTop)
		grid.addWidget(self.image_display['r'], 2, 6, 1, 3, alignment=Qt.AlignTop)
		grid.addWidget(self.image_display['r_r'], 2, 9, 1, 3, alignment=Qt.AlignTop)

		grid.addWidget(self._create_preblur_group(), 3, 0, 4, 6, alignment=Qt.AlignTop)
		grid.addWidget(self._create_matcher_group(), 7, 0, 22, 6, alignment=Qt.AlignTop)
		grid.addWidget(self._create_filter_group(), 3, 6, 8, 6, alignment=Qt.AlignTop)
		grid.addWidget(self._create_crop_group(), 11, 6, 8, 6, alignment=Qt.AlignTop)
		grid.addWidget(self._create_checkbox_group(), 19, 6, 8, 6, alignment=Qt.AlignTop)

		grid.addWidget(manual_disparity_button, 30, 0, 1, 4, alignment=Qt.AlignBottom)
		grid.addWidget(reconstruct_3d, 30, 4, 1, 4, alignment=Qt.AlignBottom)
		grid.addWidget(save_current_btn, 30, 8, 1, 4, alignment=Qt.AlignBottom)

		#Add layout
		self.setLayout(grid)

	def _search_folders(self):
		#Fetch the path
		path = self.load_config['path'].text()
		#Clear dropdown and refresh items
		self.load_config['folder'].clear()
		self.load_config['folder'].addItems(os.listdir(path))

	def _load_files(self):
		#Fetch the path and the folder
		path = '{}/{}'.format(self.load_config['path'].text(), self.load_config['folder'].currentText())

		#Verify both images are present
		if not os.path.isfile(f'{path}/cameraL.png'):
			if self.verbose: print(f'File cameraL.png not found in {path}')
			return
		if not os.path.isfile(f'{path}/cameraR.png'):
			if self.verbose: print(f'File cameraR.png not found in {path}')
			return

		#Load, rectify and store the images
		img_l, img_r = cv.imread(f'{path}/cameraL.png'), cv.imread(f'{path}/cameraR.png')
		rect_l, rect_r = self.parentWindow.calibration.cw.rectify_image_pair(img_l, img_r)
		self.frames = {'l' : img_l, 'l_r' : rect_l, 'r' : img_r, 'r_r' : rect_r}

		#Update displays
		for key, frame in self.frames.items():
			update_display(self.display_size['w'], self.display_size['h'], self.image_display[key], frame)

		#Load params
		if os.path.isfile(f'{path}/params.pickle'):
			with open(f'{path}/params.pickle', 'rb') as f: self.params = pickle.load(f)
			self._update_sliders()

	def _update_sliders(self):
		#Es necesario cambiar este flag antes y despues xq si no en el momento que cambiamos
		#los parametros de un slider, se llama la funcion _param_changed y sobreescribe los valores
		#del resto de sliders que se tengan que cambiar
		#Así, como ya tenemos self.params actualizado, evitamos que se llame
		self.update_params = False

		#Update sliders with the stored parameters
		#for name,v in self.params['preprocessing'].items(): ... #TODO
		for name,v in self.params['matcher'].items(): update_object_value(self.matcher_sliders[name], v)
		for name,v in self.params['filter'].items(): update_object_value(self.filter_sliders[name], v)
		for name,v in self.params['crop'].items(): self.crop_sliders[name].setValue(v)

		if self.verbose: print(self.params)
		self.update_params = True

	def _param_changed(self):
		if not self.update_params: return

		#Update the params that are stored based on slider values
		self.params = {
			'prefilter' : {name : get_object_value(s) for name,s in self.preblur_filter.items()},
			'matcher' : {name : get_object_value(s) for name,s in self.matcher_sliders.items()},
			'filter' : {name : get_object_value(s) for name,s in self.filter_sliders.items()},
			'crop' : {name : s.value() for name,s in self.crop_sliders.items()}
		}
		if self.verbose: print(self.params)

		#Update disparity if online_disparity_checkbox.isChecked()
		if self.online_disparity_checkbox.isChecked(): self.update_disparity()

	#Wrappers
	def _create_display(self): return create_display(self.display_size['w'], self.display_size['h'])
	def _create_preblur_group(self): return create_object_group("Image preprocessing", self.preblur_filter, cols = 2)
	def _create_matcher_group(self): return create_object_group("Matcher parameters", self.matcher_sliders, cols = 2, fullwidth = [0,1,2])
	def _create_filter_group(self): return create_object_group("Disparity filtering", self.filter_sliders, cols = 2)
	def _create_crop_group(self): return create_object_group("Crop parameters", self.crop_sliders, cols = 2)
	def _create_checkbox_group(self): return create_object_group("Available options", {
		'odc' : self.online_disparity_checkbox, 'pc' : self.use_preblur_checkbox, **self.use_filters, 'acc' : self.apply_crop_checkbox
	}, cols = 3)

	def _apply_blur(self, img, filterType, kernelSize):
		if filterType != 0 and kernelSize % 2 == 0: kernelSize += 1
		return self.blur_filters[filterType](img, kernelSize)

	def _crop_img(self, img):
		#Get crop limits from the sliders and return the cropped image
		l = {k : v for k,v in self.params['crop'].items()}
		return img[l['y1']:l['y2'], l['x1']:l['x2']]

	#Hay una funcion equivalente en src/DisparityUI.py.
	#Esta mejor comentada alli.
	def update_disparity(self, plot3d = False):
		#Verify both frames are valid
		if any(f is None for f in self.frames.values()): return

		#Seleccionamos los frames rectificados
		rect_l, rect_r = self.frames['l_r'], self.frames['r_r']

		#Aplicamos preprocesados
		if self.use_preblur_checkbox.isChecked():
			rect_l = self._apply_blur(rect_l, self.params['prefilter']['filterType'], self.params['prefilter']['kernelSize'])
			rect_r = self._apply_blur(rect_r, self.params['prefilter']['filterType'], self.params['prefilter']['kernelSize'])

		#Calculamos el mapa de disparidad
		stereo = cv.StereoSGBM_create(**self.params['matcher'])
		disp = stereo.compute(rect_l, rect_r)

		#Calculamos la disparidad de la imagen derecha. (Es utilizada por el wls filter, pero hay que aplicarle tambien
		#los otros filtrados si se usan en conjunto)
		right_disp = cv.ximgproc.createRightMatcher(stereo).compute(rect_r, rect_l)

		#Pre-speckle
		if self.use_filters['pre-speckle'].isChecked():
			cv.filterSpeckles(disp, 0, self.params['filter']['pre_maxSpeckleSize'], self.params['filter']['pre_maxDiff'])
			cv.filterSpeckles(right_disp, 0, self.params['filter']['pre_maxSpeckleSize'], self.params['filter']['pre_maxDiff'])

		#FIXME: blur
		#if self.use_filters['blur'].isChecked():
		if False:
			#Disparity to image
			_img = disp2img(disp, self.params['matcher']['minDisparity'], self.params['matcher']['numDisparities'])
			_right_img = disp2img(right_disp, self.params['matcher']['minDisparity'], self.params['matcher']['numDisparities'])

			#Filter image
			_img = self._apply_blur(_img, self.params['filter']['blur_filterType'], self.params['filter']['blur_kernelSize'])
			_right_img = self._apply_blur(_right_img, self.params['filter']['blur_filterType'], self.params['filter']['blur_kernelSize'])

			#Image to disparity
			disp = img2disp(_img, self.params['matcher']['minDisparity'], self.params['matcher']['numDisparities'])
			right_disp = img2disp(_right_img, self.params['matcher']['minDisparity'], self.params['matcher']['numDisparities'])

		#WLS
		if self.use_filters['wls'].isChecked():
			wls_filter = cv.ximgproc.createDisparityWLSFilter(stereo)
			wls_filter.setLambda(self.params['filter']['lambda'])
			wls_filter.setSigmaColor(self.params['filter']['sigma'])
			disp = wls_filter.filter(disp, rect_l, disparity_map_right=right_disp)

		#Post-speckle
		if self.use_filters['post-speckle'].isChecked():
			cv.filterSpeckles(disp, 0, self.params['filter']['post_maxSpeckleSize'], self.params['filter']['post_maxDiff'])
			cv.filterSpeckles(right_disp, 0, self.params['filter']['pre_maxSpeckleSize'], self.params['filter']['pre_maxDiff'])

		#Show disp map
		disp_norm = disp2img(disp, self.params['matcher']['minDisparity'], self.params['matcher']['numDisparities'])
		if self.apply_crop_checkbox.isChecked(): disp_norm = self._crop_img(disp_norm)
		cv.imshow("Disparity map", disp_norm)

		#3dplot the image
		if plot3d:
			Q = self.parentWindow.calibration.cw.rectification_results['disparity2depth_mappingMatrix']

			if self.apply_crop_checkbox.isChecked():
				disp = self._crop_img(disp)
				rect_l = self._crop_img(rect_l)

			show_point_cloud(rect_l, disp, Q)

	def _save_current(self):
		#Get path and create a new folder
		path = '{}/{}'.format(self.load_config['path'].text(), int(dt.now().timestamp()))
		os.mkdir(path)
		if self.verbose: print('Created folder at {}'.format(path))

		#Save images and params
		cv.imwrite('{}/cameraL.png'.format(path), self.frames['l'])
		cv.imwrite('{}/cameraR.png'.format(path), self.frames['r'])
		with open('{}/params.pickle'.format(path), 'wb') as f: pickle.dump(self.params, f)
		if self.verbose: print('Saved cameraL.png, cameraR.png, params.pickle at {}'.format(path))
