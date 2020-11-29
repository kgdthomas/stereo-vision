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

		self.blur_filters_mapping = [			#Blur filters options mapping
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

		self.plot3d_options = [				#3D reconstruction options mapping
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

		#Create layout
		grid = QGridLayout()

		#Create elements
		self.load_config = {				#Formulario para seleccionar y cargar imagenes
			'path' : QLineEdit(kargs['default_test_path']), 'search' : QPushButton("Buscar carpetas"),
			'folder' : QComboBox(), 'load' : QPushButton("Cargar ficheros")
		}
		self.image_display = {				#Displays para las imagenes
			'l' : self._create_display(), 'l_r' : self._create_display(),
			'r' : self._create_display(), 'r_r' : self._create_display()
		}

		self.online_disparity_checkbox = QCheckBox("Real-time disparity")	#Realtime disparity (recalculate when any param change)

		self.preprocessing_filters_objects = {		#Filtros de preprocesado de imagen
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

		self.matcher_sliders = {	#Parametros del matcher
			'minDisparity' : create_slider(1, 256),
			'numDisparities' : create_slider(1, 512),
			'blockSize' : create_slider(0, 256),
			'P1' : create_slider(8, 1024),
			'P2' : create_slider(8, 2048),
			'disp12MaxDiff' : create_slider(1, 640),
			'preFilterCap' : create_slider(0, 1000),
			'uniquenessRatio' : create_slider(0, 15),
			'speckleWindowSize' : create_slider(10, 30),
			'speckleRange' : create_slider(0, 200),
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
			'pre_maxSpeckleSize' : create_slider(0, 1000, tick_interval=500),
			'pre_maxDiff' : create_slider(0, 50),
			'blur_filter' : create_tabs_group(self.blur_filter_objects),
			'lambda' : create_slider(0, 24000, default_value = 8000, tick_interval=500, step=100),
			'sigma' : create_slider(0, 400, tick_interval=10, default_value = 15, scale=0.1, type=float),
			'post_maxSpeckleSize' : create_slider(0, 1000, tick_interval=500),
			'post_maxDiff' : create_slider(0, 50)
		}
		self.use_filters = {			#Checkbox para activa cada uno de los filtros
			'pre-speckle' : QCheckBox('Use pre speckle filter'), 'blur' : QCheckBox('Use post blur filter'),
			'wls' : QCheckBox('Use wls filter'), 'post-speckle' : QCheckBox('Use post speckle filter')
		}

		self.crop_sliders = {			#Limites de recortado
			'x1' : create_slider(0, 640), 'x2' : create_slider(0, 640, default_value = 640),
			'y1' : create_slider(0, 640), 'y2' : create_slider(0, 640, default_value = 640)
		}
		self.apply_crop_checkbox = QCheckBox("Apply crop")

		self.plot3d_objects = {			#3D plot config
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
		manual_disparity_button = QPushButton("Show disparity map") 	#Acciones manuales
		reconstruct_3d = QPushButton("Reconstruir 3D")
		save_current_btn = QPushButton("Save current")

		#Callbacks and props
		set_object_callback(self.load_config['search'], self._search_folders)	#Load image callbacks
		set_object_callback(self.load_config['load'], self._load_files)

		for conf in self.preprocessing_filters_objects.values():		#Callback in the pipeline parameters
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

		set_object_callback(manual_disparity_button, self.update_disparity)	#Callback in the buttons
		set_object_callback(reconstruct_3d, lambda: self.update_disparity(True))
		set_object_callback(save_current_btn, self._save_current)

		#Populate layout
		grid.addWidget(self.load_config['path'], 0, 0, 1, 6, alignment=Qt.AlignTop)
		grid.addWidget(self.load_config['search'], 1, 0, 1, 6, alignment=Qt.AlignTop)
		grid.addWidget(self.load_config['folder'], 0, 6, 1, 6, alignment=Qt.AlignTop)
		grid.addWidget(self.load_config['load'], 1, 6, 1, 6, alignment=Qt.AlignTop)

		grid.addWidget(self.image_display['l'], 2, 0, 1, 3, alignment=Qt.AlignTop)
		grid.addWidget(self.image_display['l_r'], 2, 3, 1, 3, alignment=Qt.AlignTop)
		grid.addWidget(self.image_display['r'], 2, 6, 1, 3, alignment=Qt.AlignTop)
		grid.addWidget(self.image_display['r_r'], 2, 9, 1, 3, alignment=Qt.AlignTop)

		grid.addWidget(self.preprocessing_filters_config, 3, 0, 6, 6, alignment=Qt.AlignTop)
		grid.addWidget(self._create_matcher_group(), 7, 0, 12, 6, alignment=Qt.AlignTop)
		grid.addWidget(self._create_filter_group(), 3, 6, 10, 6, alignment=Qt.AlignTop)
		grid.addWidget(self._create_crop_group(), 13, 6, 4, 6, alignment=Qt.AlignTop)
		grid.addWidget(self.plot3d_config, 16, 6, 4, 6, alignment=Qt.AlignTop)

		grid.addWidget(self._create_checkbox_group(), 19, 0, 1, 12, alignment=Qt.AlignTop)
		grid.addWidget(manual_disparity_button, 20, 0, 1, 4, alignment=Qt.AlignBottom)
		grid.addWidget(reconstruct_3d, 20, 4, 1, 4, alignment=Qt.AlignBottom)
		grid.addWidget(save_current_btn, 20, 8, 1, 4, alignment=Qt.AlignBottom)

		#Add layout
		self.setLayout(grid)

	def _search_folders(self):
		#Fetch the path
		path = self.load_config['path'].text()
		#Clear dropdown and refresh items
		self.load_config['folder'].clear()
		self.load_config['folder'].addItems([d for d in os.listdir(path) if os.path.isdir(f'{path}/{d}')])

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
		categories = {
#			'preprocessing' : self.preblur_filter,
			'matcher' : self.matcher_sliders,
			'filter' : self.filter_sliders,
			'crop' : self.crop_sliders,
			'checkbox' : self.every_checkbox
		}

		#Es necesario cambiar este flag antes y despues xq si no en el momento que cambiamos
		#los parametros de un slider, se llama la funcion _param_changed y sobreescribe los valores
		#del resto de sliders que se tengan que cambiar
		#Así, como ya tenemos self.params actualizado, evitamos que se llame
		self.update_params = False

		#Update sliders with the stored parameters
		try:
			if 'preprocessing' in self.params:
				update_object_value(self.preprocessing_filters_config, self.params['preprocessing']['currentTab'])
				for tab_name, conf in self.params['preprocessing']['values'].items():
					for obj_name, v in conf.items():
						update_object_value(self.preprocessing_filters_objects[tab_name]['objects'][obj_name], v)
		except: pass

		try:
			if 'matcher' in self.params:
				for name,v in self.params['matcher'].items():
					update_object_value(self.matcher_sliders[name], v)
		except: pass

		try:
			if 'filter' in self.params:
				for name, v in self.params['filter'].items():
					if name != 'blur_filter': update_object_value(self.filter_sliders[name], v)
					else:
						update_object_value(self.filter_sliders[name], v['currentTab'])
						for tab_name, conf in v['values'].items():
							for obj_name, val in conf.items():
								update_object_value(self.blur_filter_objects[tab_name]['objects'][obj_name], val)
		except: pass

		try:
			if 'crop' in self.params:
				for name,v in self.params['crop'].items():
					update_object_value(self.crop_sliders[name], v)
		except: pass

		try:
			if 'checkbox' in self.params:
				for name,v in self.params['checkbox'].items():
					update_object_value(self.every_checkbox[name], v)
		except: pass

		try:
			if '3dconf' in self.params:
				update_object_value(self.plot3d_config, self.params['3dconf']['currentTab'])
				for tab_name, conf in self.params['3dconf']['values'].items():
					for obj_name, v in conf.items():
						update_object_value(self.plot3d_objects[tab_name]['objects'][obj_name], v)
		except: pass

		self.update_params = True
		#Llamamos manualmente a _param_changed para que actualize el mapa de disparidad si asi
		#esta configurado en los parametros que acabamos de cargar
		self._param_changed()

	def _param_changed(self):
		if not self.update_params: return

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

		#Update disparity if online disparity
		if self.params['checkbox']['onlineDisp']: self.update_disparity()

	#Wrappers
	def _create_display(self): return create_display(self.display_size['w'], self.display_size['h'])
	def _create_matcher_group(self): return create_object_group("Matcher parameters", self.matcher_sliders, cols = 2, fullwidth = [0,1,2])
	def _create_filter_group(self): return create_object_group("Disparity filtering", self.filter_sliders, cols = 2, fullwidth = [2])
	def _create_crop_group(self): return create_object_group("Crop parameters", self.crop_sliders, cols = 4)
	def _create_checkbox_group(self): return create_object_group(None, self.every_checkbox, cols = 7)

	def _apply_blur(self, img, filterType, kernelSize):
		if filterType != 0 and kernelSize % 2 == 0: kernelSize += 1
		return self.blur_filters_mapping[filterType](img, kernelSize)

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
		if self.params['checkbox']['preprocessing']:
			rect_l = self.preprocessing_options[self.params['preprocessing']['currentTab']](rect_l)
			rect_r = self.preprocessing_options[self.params['preprocessing']['currentTab']](rect_r)

		#Calculamos el mapa de disparidad
		stereo = cv.StereoSGBM_create(**self.params['matcher'])
		disp = stereo.compute(rect_l, rect_r)

		#Pre-speckle
		if self.params['checkbox']['pre-speckle']:
			cv.filterSpeckles(disp, 0, self.params['filter']['pre_maxSpeckleSize'], self.params['filter']['pre_maxDiff'])

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

		#Show disp map
		disp_norm = disp2img(disp, self.params['matcher']['minDisparity'], self.params['matcher']['numDisparities'])
		if self.params['checkbox']['crop']: disp_norm = self._crop_img(disp_norm)
		cv.imshow("Disparity map", disp_norm)

		#3dplot the image
		if plot3d:
			Q = self.parentWindow.calibration.cw.rectification_results['disparity2depth_mappingMatrix']

			if self.params['checkbox']['crop']:
				disp = self._crop_img(disp)
				rect_l = self._crop_img(rect_l)

			self.plot3d_options[self.params['3dconf']['currentTab']](rect_l, disp, Q)

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
