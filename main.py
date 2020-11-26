#!/bin/python3
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import argparse
import sys
import os

#Import UI
sys.path.insert(0, '{}/src'.format(os.path.dirname(os.path.abspath(__file__))))
from CalibrationUI import CalibrationUI
from DisparityUI import DisparityUI
from OfflineUI import OfflineUI

class App(QMainWindow):
	def __init__(self, **kargs):
		super(App, self).__init__()

		#Create the window
		self.setWindowTitle("Stereo Vision - Victor Labayen")
		self.setGeometry(0, 0, kargs['screen_width'], kargs['screen_height'])

		#Create the tab screen
		tabs_parent = QWidget()
		tabs_parent_layout = QVBoxLayout()
		self.tabs = QTabWidget()
		self.tabs.resize(kargs['screen_width'], kargs['screen_height'])

		#Add tabs
		self.calibration = CalibrationUI(self, **kargs)
		self.disparity = DisparityUI(self, **kargs)
		self.offline = OfflineUI(self, **kargs)
		self.tabs.addTab(self.calibration, "Camera calibration")
		self.tabs.addTab(self.disparity, "Disparity map")
		self.tabs.addTab(self.offline, "Offline tests")

		#Add the layouts
		self.tabs.setTabEnabled(1, False)	#Disable disparity & offline tabs util calibration is made
		self.tabs.setTabEnabled(2, False)
		tabs_parent_layout.addWidget(self.tabs)
		tabs_parent.setLayout(tabs_parent_layout)
		self.setCentralWidget(tabs_parent)

		#On tab changed
		self.tabs.currentChanged.connect(lambda idx: self.on_tab_changed(idx))

	def on_tab_changed(self, index):
		#Release all video captures that are not being shown
		if index != 0:
			for vinput in self.calibration.video_input.values():
				if vinput is not None: vinput.release()
		elif index != 1:
			for vinput in self.disparity.video_input.values():
				if vinput is not None: vinput.release()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--screen_width', default=1200, type=int, help="Pixel width of the window")
	parser.add_argument('--screen_height', default=800, type=int, help="Pixel height of the window")
	parser.add_argument('--chessboard-rows', default=6, type=int, help="Number of inside row in the chessboard")
	parser.add_argument('--chessboard-columns', default=9, type=int, help="Number of inside columns in the chessboard")
	parser.add_argument('--chessboard-square-size', default=1.0, type=float, help="Size of the chessboard squares in the units we want the camera matrix")
	parser.add_argument('--max-frame-rate', default=60, type=int, help="Max frame rate in capture mode. Disparity mode is fixed at 1fps")
	parser.add_argument('-v', '--verbose', default=False, action="store_true")
	args = parser.parse_args()

	app = QApplication([])
	myapp = App(**vars(args))
	myapp.show()
	app.exec_()
