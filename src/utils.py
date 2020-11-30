#!/bin/python3
import cv2 as cv
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import numpy as np
import open3d as o3d
from plot_heatmap import plot_heatmap
import matplotlib.pyplot as plt

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
def create_slider(min_value, max_value, tick_interval = 5, default_value = None, step = 1, scale = 1, type=int):
	#QSlider class configuration
	slider = QSlider(Qt.Horizontal)
	slider.setMinimum(min_value)
	slider.setMaximum(max_value)
	slider.setTickPosition(QSlider.TicksBelow)
	slider.setTickInterval(tick_interval)
	slider.setSingleStep(step)
	if default_value is not None: slider.setValue(default_value)

	#Custom configuration
	slider.custom_scale = scale
	slider.custom_type = type

	return slider

def create_combobox(options = []):
	cbox = QComboBox()
	cbox.addItems(options)
	return cbox

#Set a img in a given display. If img is not numpy.array, set the background
def update_display(w, h, display, img):
	if isinstance(img, np.ndarray): qp = img2pixmap(img, w, h)
	else:
		qp = QPixmap(w, h)
		qp.fill(QColor('darkGray'))

	display.setPixmap(qp)

#Create a QGroupBox with the given objects and a title for each one
def create_object_group(group_title, objects, cols = 1, fullwidth = []):
	group = QGroupBox(group_title) if group_title else QWidget()
	groupLayout = QGridLayout()

	offset = 0
	for i, (name, obj) in enumerate(objects.items()):
		idx = i + offset
		if i not in fullwidth:
			if not (isinstance(obj, QCheckBox) or isinstance(obj, QLabel) or isinstance(obj, QTabWidget)):
				groupLayout.addWidget(QLabel(name), 2*(idx//cols), idx%cols, 1, 1, alignment=Qt.AlignTop)
			groupLayout.addWidget(obj, 2*(idx//cols) + 1, idx%cols, 1, 1, alignment=Qt.AlignTop)
		else:
			if not (isinstance(obj, QCheckBox) or isinstance(obj, QLabel) or isinstance(obj, QTabWidget)):
				groupLayout.addWidget(QLabel(name), 2*(idx//cols), 0, 1, cols, alignment=Qt.AlignTop)
			groupLayout.addWidget(obj, 2*(idx//cols) + 1, 0, 1, cols, alignment=Qt.AlignTop)
			offset += (cols - 1)

	group.setLayout(groupLayout)
	return group

#Create some object_groups in diferent tabs
def create_tabs_group(tabs_dict):
	tabs = QTabWidget()
	for name, t in tabs_dict.items(): tabs.addTab(create_object_group(
		t['name'] if ('name' in t) else None,
		t['objects'] if ('objects' in t) else {},
		t['cols'] if ('cols' in t) else 1,
		t['fullWidth'] if ('fullWidth' in t) else []
	), name)
	return tabs

#Normalize disparity map to allow visualization
def disp2img(disp, minDisparity, numDisparities):
	return ((disp.astype(np.float32) / 16.0) - minDisparity) / numDisparities

#Undo above operation
def img2disp(img, minDisparity, numDisparities):
	return (((img * numDisparities) + minDisparity) * 16.0).astype(np.int16)


#Get slider value/combobox index
def get_object_value(obj):
	if isinstance(obj, QSlider): return obj.custom_type(obj.value() * obj.custom_scale)
	if isinstance(obj, QComboBox): return obj.currentIndex()
	if isinstance(obj, QCheckBox): return obj.isChecked()
	if isinstance(obj, QTabWidget): return obj.currentIndex()

#Set slider/combobox value
def update_object_value(obj, value):
	if isinstance(obj, QSlider): obj.setValue(obj.custom_type(value / obj.custom_scale))
	if isinstance(obj, QComboBox): obj.setCurrentIndex(value)
	if isinstance(obj, QCheckBox): obj.setChecked(value)
	if isinstance(obj, QTabWidget): obj.setCurrentIndex(value)

#Set slider/combobox callbacks
def set_object_callback(obj, callback):
	if isinstance(obj, QSlider): obj.valueChanged.connect(lambda: callback())
	if isinstance(obj, QComboBox): obj.currentIndexChanged.connect(lambda: callback())
	if isinstance(obj, QCheckBox): obj.stateChanged.connect(lambda: callback())
	if isinstance(obj, QTabWidget): obj.currentChanged.connect(lambda: callback())
	if isinstance(obj, QPushButton): obj.clicked.connect(lambda: callback())

#Depth map
def show_depth_map(disp, Q):
	points = cv.reprojectImageTo3D(disp, Q)
	ylen, xlen = points.shape[:2]
	x, y = [i for i in range(xlen)], [ylen - j for j in range(ylen)]
	z = [pix[2] for column in points for pix in column]
	plot_heatmap(x,y,z, x_label = "Pixel x", y_label = "Pixel y")

#3D point cloud
def compute_point_cloud(img, disp, Q, normals = None):
	points = cv.reprojectImageTo3D(disp, Q)
	colors = cv.cvtColor(img, cv.COLOR_BGR2RGB)
	mask = disp > disp.min()

	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(points[mask])
	pcd.colors = o3d.utility.Vector3dVector(colors[mask] / 255)
	if isinstance(normals, np.ndarray): pcd.normals = o3d.utility.Vector3dVector(normals[mask])

	return pcd

def show_point_cloud(img, disp, Q):
	pcd = compute_point_cloud(img, disp, Q)
	o3d.visualization.draw_geometries([pcd])

#Compute normals. https://stackoverflow.com/questions/53350391/surface-normal-calculation-from-depth-map-in-python
def compute_normals(disp, minDisp, numDisp):
	disp_norm = cv.cvtColor(disp2img(disp, minDisp, numDisp), cv.COLOR_GRAY2BGR).astype("float64")
	normals = np.array(disp_norm, dtype="float32")
	h,w,d = disp_norm.shape
	for i in range(1, w-1):
		for j in range(1, h-1):
			t = np.array([i,j-1,disp_norm[j-1,i,0]],dtype="float64")
			f = np.array([i-1,j,disp_norm[j,i-1,0]],dtype="float64")
			c = np.array([i,j,disp_norm[j,i,0]] , dtype = "float64")
			d = np.cross(f-c,t-c)
			normals[j,i,:] = d / np.sqrt((np.sum(d**2)))

	return -normals

#BPA Mesh
def compute_bpa_mesh(img, disp, Q, radius, minDisp, numDisp, verbose = False):
	if verbose: print('Computing normals...')
	normals = compute_normals(disp, minDisp, numDisp)

	if verbose: print('Computing point cloud')
	pcd = compute_point_cloud(img, disp, Q, normals = normals * 255)

	if radius == 0:
		distances = pcd.compute_nearest_neighbor_distance()
		radius = 3 * np.mean(distances)
	if verbose: print(f'Using radius = {radius}')

	if verbose: print('Computing mesh...')
	radii = o3d.utility.DoubleVector([radius, radius * 2])
	return o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, radii)

def show_bpa_mesh(img, disp, Q, radius, max_triang, minDisp, numDisp, verbose = False):
	bpa_mesh = compute_bpa_mesh(img, disp, Q, radius, minDisp, numDisp, verbose)

	if verbose: print('Mesh decimation')
	dec_mesh = bpa_mesh.simplify_quadric_decimation(max_triang)
	dec_mesh.remove_degenerate_triangles()
	dec_mesh.remove_duplicated_triangles()
	dec_mesh.remove_duplicated_vertices()
	dec_mesh.remove_non_manifold_edges()

	if verbose: print('Showing...')
	o3d.visualization.draw_geometries([dec_mesh])

#Poisson mesh
def compute_poisson_mesh(img, disp, Q, depth, scale, fit, minDisp, numDisp, verbose = False):
	if verbose: print('Computing normals...')
	normals = compute_normals(disp, minDisp, numDisp)

	if verbose: print('Computing point cloud')
	pcd = compute_point_cloud(img, disp, Q, normals = normals * 255)

	if verbose: print('Computing mesh...')
	poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth, scale=scale, linear_fit=fit)[0]

	if verbose: print('Cropping mesh')
	return poisson_mesh.crop(pcd.get_axis_aligned_bounding_box())

def show_poisson_mesh(img, disp, Q, depth, scale, fit, minDisp, numDisp, verbose = False):
	poisson_mesh = compute_poisson_mesh(img, disp, Q, depth, scale, fit, minDisp, numDisp, verbose)

	if verbose: print('Showing...')
	o3d.visualization.draw_geometries([poisson_mesh])
