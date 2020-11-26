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

		#Create layout
		grid = QGridLayout()

		#Create elements
		self.load_config = {		#Formulario para seleccionar y cargar imagenes
			'path' : QLineEdit("imgs/test"), 'search' : QPushButton("Buscar carpetas"),
			'folder' : QComboBox(), 'load' : QPushButton("Cargar ficheros")
		}
		self.image_display = {		#Displays para las imagenes
			'l' : self._create_display(), 'l_r' : self._create_display(),
			'r' : self._create_display(), 'r_r' : self._create_display()
		}
		self.matcher_sliders = {	#Parametros del matcher
			'minDisparity' : create_slider(2, 256),
			'numDisparities' : create_slider(2, 512),
			'blockSize' : create_slider(2, 256),
			'P1' : create_slider(8, 1024),
			'P2' : create_slider(8, 2048),
			'disp12MaxDiff' : create_slider(1, 640),
			'uniquenessRatio' : create_slider(0, 10, tick_interval = 0.05, step = 0.05),
			'speckleWindowSize' : create_slider(10, 30),
			'speckleRange' : create_slider(0, 200)
		}
		self.online_disparity_checkbox = QCheckBox("Real-time disparity")

		self.filter_sliders = {		#Parametros del filtro
			'lambda' : create_slider(0, 24000, default_value = 8000, tick_interval=500, step=100),
			'sigma' : create_slider(0, 40, default_value = 1.5, step = 0.1)
		}
		self.use_filter_checkbox = QCheckBox('Use filter')

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

		for s in self.matcher_sliders.values(): s.valueChanged.connect(lambda: self._param_changed())
		for s in self.filter_sliders.values(): s.valueChanged.connect(lambda: self._param_changed())
		for s in self.crop_sliders.values(): s.valueChanged.connect(lambda: self._param_changed())
		self.online_disparity_checkbox.stateChanged.connect(lambda: self._param_changed())
		self.use_filter_checkbox.clicked.connect(lambda: self._param_changed())
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

		grid.addWidget(self._create_matcher_group(), 3, 0, 7, 6, alignment=Qt.AlignTop)
		grid.addWidget(self._create_filter_group(), 3, 6, 1, 6, alignment=Qt.AlignTop)
		grid.addWidget(self._create_crop_group(), 4, 6, 1, 6, alignment=Qt.AlignTop)
		grid.addWidget(self.online_disparity_checkbox, 5, 6, 1, 6, alignment=Qt.AlignTop)
		grid.addWidget(self.use_filter_checkbox, 6, 6, 1, 6, alignment=Qt.AlignTop)
		grid.addWidget(self.apply_crop_checkbox, 7, 6, 1, 6, alignment=Qt.AlignTop)

		grid.addWidget(manual_disparity_button, 10, 0, 1, 4, alignment=Qt.AlignTop)
		grid.addWidget(reconstruct_3d, 10, 4, 1, 4, alignment=Qt.AlignTop)
		grid.addWidget(save_current_btn, 10, 8, 1, 4, alignment=Qt.AlignBottom)

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
		for name,v in self.params['matcher'].items(): self.matcher_sliders[name].setValue(v)
		for name,v in self.params['filter'].items(): self.filter_sliders[name].setValue(v)
		for name,v in self.params['crop'].items(): self.crop_sliders[name].setValue(v)

		if self.verbose: print(self.params)
		self.update_params = True

	def _param_changed(self):
		if not self.update_params: return

		#Update the params that are stored based on slider values
		self.params = {
			'matcher' : {name : s.value() for name,s in self.matcher_sliders.items()},
			'filter' : {name : s.value() for name,s in self.filter_sliders.items()},
			'crop' : {name : s.value() for name,s in self.crop_sliders.items()}
		}
		if self.verbose: print(self.params)

		#Update disparity if online_disparity_checkbox.isChecked()
		if self.online_disparity_checkbox.isChecked(): self.update_disparity()

	#Wrappers
	def _create_display(self): return create_display(self.display_size['w'], self.display_size['h'])
	def _create_matcher_group(self): return create_object_group("Matcher parameters", self.matcher_sliders)
	def _create_filter_group(self): return create_object_group("Filter parameters", self.filter_sliders)
	def _create_crop_group(self): return create_object_group("Crop parameters", self.crop_sliders)

	def _crop_img(self, img):
		#Get crop limits from the sliders and return the cropped image
		l = {k : v for k,v in self.params['crop'].items()}
		return img[l['y1']:l['y2'], l['x1']:l['x2']]

	#Hay una funcion equivalente en src/DisparityUI.py.
	#Esta mejor comentada alli.
	def update_disparity(self, plot3d = False):
		#Verify both frames are valid
		if any(f is None for f in self.frames.values()): return

		#Seleccionamos los frames rectificados y calculamos el mapa de disparidad
		rect_l, rect_r = self.frames['l_r'], self.frames['r_r']
		stereo = cv.StereoSGBM_create(**self.params['matcher'])
		disp = stereo.compute(rect_l, rect_r)

		#Show the disparity as it is if no filter is requested
		if not self.use_filter_checkbox.isChecked():
			disp_norm = disp2img(disp, self.params['matcher']['minDisparity'], self.params['matcher']['numDisparities'])
			if self.apply_crop_checkbox.isChecked(): disp_norm = self._crop_img(disp_norm)
			cv.imshow("Disparity map", disp_norm)

		#Apply filter to the disparity and show
		else:
			right_disp = cv.ximgproc.createRightMatcher(stereo).compute(rect_r, rect_l)
			wls_filter = cv.ximgproc.createDisparityWLSFilter(stereo)
			wls_filter.setLambda(self.params['filter']['lambda'])
			wls_filter.setSigmaColor(self.params['filter']['sigma'])
			disp = wls_filter.filter(disp, rect_l, disparity_map_right=right_disp)
			disp_norm = disp2img(disp, self.params['matcher']['minDisparity'], self.params['matcher']['numDisparities'])
			if self.apply_crop_checkbox.isChecked(): disp_norm = self._crop_img(disp_norm)
			cv.imshow("Disparity map", disp_norm)

		#3dplot the image
		if plot3d:
			Q = self.parentWindow.calibration.cw.rectification_results['disparity2depth_mappingMatrix']

			if self.apply_crop_checkbox.isChecked():
				disp = self._crop_img(disp)
				rect_l = self._crop_img(rect_l)

			points = cv.reprojectImageTo3D(disp, Q)
			colors = cv.cvtColor(rect_l, cv.COLOR_BGR2RGB)
			mask = disp > disp.min()
			p, c = points[mask], colors[mask]

			pcd = o3d.geometry.PointCloud()
			pcd.points = o3d.utility.Vector3dVector(p)
			pcd.colors = o3d.utility.Vector3dVector(c / 255)
			o3d.visualization.draw_geometries([pcd])

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
