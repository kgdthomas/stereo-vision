#!/bin/python3
import cv2 as cv
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import numpy as np
from datetime import datetime as dt
import glob
import sys
import os

from CalibrationWrapper import CalibrationWrapper
from utils import *

#UI tab to perform camera calibration
class CalibrationUI(QWidget):
	def __init__(self, parent, **kargs):
		super(CalibrationUI, self).__init__()
		self.parentWindow = parent
		self.chessboard_pattern = (kargs['chessboard_rows'], kargs['chessboard_columns']) 	#Numero de filas y columnas interiores
		self.chessboard_square_size = kargs['chessboard_square_size']				#Tamaño del cuadrado del tablero. Se puede dejar a 1.
													#La calibración se obtendrá en estas unidades
		self.video_input = {'l': None, 'r': None} 		#Aqui se guardan los objectos cv.VideoCapture o None si no esta activa la camara
		self.display_sizes = {'w' : 600, 'h' : 400}		#En px
		self.store_images = False				#Flag para indicar si hay captura de imagenes activada
		self.capture_timeout = 0				#Contador para incrementar el tiempo desde la ultima captura
		self.last_frame = dt.now()				#Fecha en la que se tomo el frame anterior, para incrementar el contador anterior adecuadamente
		self.cw = None						#Calibration wrapper object
		self.verbose = kargs['verbose']

		self.timer = QTimer()					#QTimer para ejecutar el refresco de frames
		self.timer.timeout.connect(lambda: self._update_images())

		#Create layout
		grid = QGridLayout()

		#Create elements
		list_cameras_button = QPushButton("List cameras")								#List available cameras
		self.camera_dropdown = {'l': QComboBox(), 'r': QComboBox()}							#Select left and right cameras
		self.camera_display = {'l': self._create_display(), 'r': self._create_display()}				#Show left and right video inputs
		self.checkboxes = {'l':  QCheckBox("Show chessboard corners"), 'r': QCheckBox("Show chessboard corners")}	#Find and draw chessboard corners in every frame
		self.line_edit = {												#Inputs for image capture configuration
			'path' : QLineEdit(kargs['default_calibration_path']),
			'file_template' : QLineEdit("camera{}_{}.png"),
			'current_index' : QLineEdit("1"),
			'capture_timeout' : QLineEdit("3")
		}
		self.toggle_capture_button = QPushButton("Iniciar capturas")							#Start/Stop automatic image capture
		self.capture_timeout_label = QLabel("")										#Show image capture timer & other errors
		calibrate_button  = QPushButton("Calibrar camaras")								#Start camera calibration

		#Add callbacks and props
		list_cameras_button.clicked.connect(lambda: add_available_cameras(self.camera_dropdown))
		self.camera_dropdown['l'].currentIndexChanged.connect(lambda i: self._on_selected_camera_changed('l', i))
		self.camera_dropdown['r'].currentIndexChanged.connect(lambda i: self._on_selected_camera_changed('r', i))
		#self.line_edit['current_index'].setReadOnly(True)
		self.toggle_capture_button.clicked.connect(lambda: self._toggle_capture())
		calibrate_button.clicked.connect(lambda: self._start_calibration())

		#Populate layout
		grid.addWidget(list_cameras_button, 0, 0, 1, 6, alignment=Qt.AlignTop)
		grid.addWidget(self.camera_dropdown['l'], 1, 0, 1, 3, alignment=Qt.AlignTop)
		grid.addWidget(self.camera_dropdown['r'], 1, 3, 1, 3, alignment=Qt.AlignTop)

		grid.addWidget(self.checkboxes['l'], 2, 0, 1, 3, alignment=Qt.AlignTop)
		grid.addWidget(self.checkboxes['r'], 2, 3, 1, 3, alignment=Qt.AlignTop)
		grid.addWidget(self.camera_display['l'], 3, 0, 1, 3, alignment=Qt.AlignTop)
		grid.addWidget(self.camera_display['r'], 3, 3, 1, 3, alignment=Qt.AlignTop)

		grid.addWidget(QLabel('Store image path'), 4, 0, 1, 1, alignment=Qt.AlignTop)
		grid.addWidget(self.line_edit['path'], 5, 0, 1, 1, alignment=Qt.AlignTop)
		grid.addWidget(QLabel('Image name template'), 6, 0, 1, 1, alignment=Qt.AlignTop)
		grid.addWidget(self.line_edit['file_template'], 7, 0, 1, 1, alignment=Qt.AlignTop)

		grid.addWidget(QLabel('Image current index'), 4, 2, 1, 1, alignment=Qt.AlignTop)
		grid.addWidget(self.line_edit['current_index'], 5, 2, 1, 1, alignment=Qt.AlignTop)
		grid.addWidget(QLabel('Capture image timeout'), 6, 2, 1, 1, alignment=Qt.AlignTop)
		grid.addWidget(self.line_edit['capture_timeout'], 7, 2, 1, 1, alignment=Qt.AlignTop)

		grid.addWidget(self.toggle_capture_button, 8, 0, 1, 3, alignment=Qt.AlignTop)
		grid.addWidget(self.capture_timeout_label, 9, 0, 1, 3, alignment=Qt.AlignTop)
		grid.addWidget(calibrate_button, 10, 0, 6, 3, alignment=Qt.AlignBottom)

		#Add layout
		self.setLayout(grid)

		#Run timer
		self.timer.start(1000 // kargs['max_frame_rate'])

	def _update_images(self):
		#Si no esta activo este tab evitamos consumir cpu
		if self.parentWindow.tabs.currentIndex() != 0: return

		#Vemos si toca o no guardar los frames actuales
		try:
			if not self.store_images: timeout_value = None
			else: timeout_value = float(self.line_edit['capture_timeout'].text())
		except:
			timeout_value = None
			if self.verbose: print('No se ha podido parsear el tiempo entre capturas')

		#No toca guardar los frames
		if timeout_value is None or self.capture_timeout < timeout_value:
			#Actualizamos los videos activos
			for side, vinput in self.video_input.items():
				if vinput is None: continue

				ret, frame = vinput.read()
				#Detectamos los cuadrados del tablero si esta el checkbox activo
				if self.checkboxes[side].isChecked(): find_and_draw_chessboard(frame, self.chessboard_pattern)

				#Actualizamos los displays
				self._update_display(side, frame)

			#Actualizamos el timer para sacar capturas y el contador visual
			if self.store_images: self.capture_timeout += (dt.now() - self.last_frame).total_seconds()
			self.last_frame = dt.now()
			if timeout_value is not None: self.capture_timeout_label.setText("Captura en : {:.3f}".format(timeout_value - self.capture_timeout))

		#Toca guardar imagenes
		else:
			if self.verbose: print('Procediendo a guardar los frames actuales')

			#Obtenemos los parametros configurados por el usuario
			path = self.line_edit["path"].text()
			image_template = self.line_edit["file_template"].text()
			current_index = int(self.line_edit["current_index"].text())

			#Asegurar que en las imagenes se obtienen todas las coordenadas del tablero,
			#y que por tanto, seran images validas para la calibracion
			can_use_to_calibrate = all(vinput is not None for vinput in self.video_input.values())
			frames = {}
			for side, vinput in self.video_input.items():
				ret, frame = vinput.read()

				#Obtenemos el frame
				can_use_to_calibrate = (can_use_to_calibrate and ret)
				frames[side] = frame

				#Buscamos las esquinas
				frame_cpy = frame.copy()
				success = find_and_draw_chessboard(frame_cpy, self.chessboard_pattern)
				can_use_to_calibrate = (can_use_to_calibrate and success)

				#Actualizamos el display
				if self.checkboxes[side].isChecked(): self._update_display(side, frame_cpy)
				else: self._update_display(side, frame)

			#Guardamos las imagenes si ambas era validas
			if can_use_to_calibrate:
				for side, frame in frames.items():
					current_file = '{}/{}'.format(path, image_template.format(side.upper(), current_index))
					cv.imwrite(current_file, frame)
					if self.verbose: print('Imagen guardada en {}'.format(current_file))

				#Reseteamos los contadores, incrementamos el indice de captura y actualizamos el display del timeout de captura
				self.line_edit["current_index"].setText(str(current_index + 1))
				self.capture_timeout_label.setText("Captura en : {:.3f}".format(timeout_value))
				self.capture_timeout = 0
			else:
				if self.verbose: print('No se encuentran el patron completo en alguno de los frames')


	#Cuando se modifica la seleccion de una camara
	def _on_selected_camera_changed(self, side, selected_index):
		#Release video before changing
		if self.video_input[side] is not None: self.video_input[side].release()

		#Check if new camera is Off
		current_camera = self.camera_dropdown[side].currentText()
		if current_camera == 'Off':
			self.video_input[side] = None
			self._update_display(side, None)
			return

		#Check if the camera is already in use
		if current_camera in list(d.currentText() for s,d in self.camera_dropdown.items() if s != side):
			if self.verbose: print('Camera already in use')
			self.camera_dropdown[side].setCurrentIndex(0)
			return

		#Try to get the camera and start the capture
		try: camera_id = int(current_camera.split(" ")[-1])
		except: return
		self.video_input[side] = cv.VideoCapture(camera_id)


	#Wrappers
	def _create_display(self): return create_display(self.display_sizes['w'], self.display_sizes['h'])
	def _update_display(self, side, img = None): update_display(self.display_sizes['w'], self.display_sizes['h'], self.camera_display[side], img)

	#Start/Stop automatic image capture.
	def _toggle_capture(self):
		#Set in flag the oposite to the actual status (started/stoped)
		flag = not self.store_images

		#If true = we are requested to start the capture
		if flag:
			#Verify both cameras are active
			if any(v is None for v in self.video_input.values()):
				self.capture_timeout_label.setText("Ambas camaras deben estar activas")
				return

			#Get the camera timeout
			try: timeout_value = float(self.line_edit['capture_timeout'].text())
			except:
				self.capture_timeout_label.setText("Valor de timeout no valido")
				return

		#If false = we are requested to stop the capture
		else:
			#Clear the timeout label
			self.capture_timeout_label.setText("")

		#SetReadOnly in the input fields if the capture is about to start, enable them otherwise
		#Swap the text in the button acordingly
		#Restart the capture timeout and save the actual capture status
		self.line_edit['path'].setReadOnly(flag)
		self.line_edit['file_template'].setReadOnly(flag)
		self.line_edit['capture_timeout'].setReadOnly(flag)
		self.toggle_capture_button.setText("Detener capturas" if flag else "Iniciar capturas")
		self.capture_timeout = 0
		self.store_images = flag

	#Calibrate cameras
	def _start_calibration(self):
		#Read all the images that match the pattern in the input field
		#and are located in the path that is configured
		imgs = {side : [cv.cvtColor(cv.imread(fname), cv.COLOR_BGR2GRAY) for fname in glob.glob('{}/{}'.format(
			self.line_edit['path'].text(),
			self.line_edit['file_template'].text().format(side.upper(), '*')
		))] for side in self.video_input.keys()}

		if any(len(side_imgs) == 0 for side_imgs in imgs.values()):
			if self.verbose: print('No se han encontrado todas las fotos necesarias')
			return

		#Create all the object points. like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0) * chessboard_square_size
		object_points = np.zeros((np.prod(self.chessboard_pattern), 3), np.float32)
		object_points[:, :2] = np.indices(self.chessboard_pattern).T.reshape(-1, 2)
		object_points *= self.chessboard_square_size

		#Count num images and prepare object to store the found object and image points
		num_images = len(imgs['l'])
		obj_points = []
		img_points = {'l' : [], 'r' : []}
		for i in range(num_images):
			if self.verbose: print('Processing image {}'.format(i))
			for side, images in imgs.items():
				#Find chessboard corners
				ret, corners = cv.findChessboardCorners(images[i], self.chessboard_pattern, None)
				cv.cornerSubPix(images[i], corners, (11, 11), (-1, -1), (cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS, 30, 0.01))
				#Store img points
				img_points[side].append(corners)

			#Store object points. This should only be done if the chessboard corners are correctly found
			#This is ensured by the capture process
			obj_points.append(object_points)

		if self.verbose: print('Iniciando calibracion. Puede llevar un tiempo si hay muchas imagenes')
		#Start calibration
		self.cw = CalibrationWrapper(imageSize = imgs['l'][0].shape[:2])
		self.cw.calibrate(obj_points, img_points)

		#Enable Disparity and offline window tabs
		self.parentWindow.tabs.setTabEnabled(1, True)
		self.parentWindow.tabs.setTabEnabled(2, True)

		if self.verbose: print('Calibracion completada')
