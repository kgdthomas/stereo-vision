import cv2 as cv
import pickle

#Class to encapsulate stereo calibration
class CalibrationWrapper:
	def __init__(self, imageSize = None,
		calibration_criteria = (cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS, 100, 1e-5),
		calibration_flags = (cv.CALIB_FIX_ASPECT_RATIO + cv.CALIB_ZERO_TANGENT_DIST + cv.CALIB_SAME_FOCAL_LENGTH),
		rectification_flags = 0
	):
		self.imageSize = imageSize
		self.calibration_criteria = calibration_criteria
		self.calibration_flags = calibration_flags
		self.rectification_flags = rectification_flags
		self.calibration_results = None
		self.rectification_results = None
		self.undistort_rectify_map = None

	#Camera calibration
	def _stereoCalibrate(self, object_points, image_points):
		if self.imageSize is None: raise ValueError('Image size must be defined')

		r = cv.stereoCalibrate(
			object_points, image_points['l'], image_points['r'],
			None, None, None, None,
			imageSize = self.imageSize,
			criteria = self.calibration_criteria,
			flags = self.calibration_flags
		)

		return {
			'ret' : r[0],
			'cameraMatrix_l' : r[1], 'distortionCoeffs_l' : r[2],
			'cameraMatrix_r' : r[3], 'distortionCoeffs_r' : r[4],
			'rotationMatrix' : r[5], 'translationVector' : r[6],
			'essentialMatrix' : r[7], 'fundamentalMatrix' : r[8]
		}

	#Img rectification
	def _stereoRectify(self):
		if self.calibration_results is None: raise ValueError('Calibration is required to perform rectification')

		r = cv.stereoRectify(
			self.calibration_results['cameraMatrix_l'], self.calibration_results['distortionCoeffs_l'],
			self.calibration_results['cameraMatrix_r'], self.calibration_results['distortionCoeffs_r'],
			imageSize = self.imageSize,
			R = self.calibration_results['rotationMatrix'],
			T = self.calibration_results['translationVector'],
			flags = self.rectification_flags
		)

		return {
			'rectificationTransform_l' : r[0], 'projectionMatrix_l' : r[2],
			'rectificationTransform_r' : r[1], 'projectionMatrix_r' : r[3],
			'disparity2depth_mappingMatrix' : r[4],
			'validBoxes_l' : r[5], 'validBoxes_r' : r[6]
		}

	#Img undistort and rectify
	def _initUndistortRectifyMap(self, side):
		if self.rectification_results is None: raise ValueError('Rectification is required to perform undistortion & rectification')

		cam_matrix = self.calibration_results['cameraMatrix_{}'.format(side)]
		dist_coef = self.calibration_results['distortionCoeffs_{}'.format(side)]
		rect_transf = self.rectification_results['rectificationTransform_{}'.format(side)]
		proj_matrix = self.rectification_results['projectionMatrix_{}'.format(side)]

		r = cv.initUndistortRectifyMap(cam_matrix, dist_coef, rect_transf, proj_matrix, self.imageSize, cv.CV_32FC1)
		return {
			'undistortion_map' : r[0],
			'rectification_map' : r[1]
		}

	#All the above steps
	def calibrate(self, object_points, image_points):
		self.calibration_results = self._stereoCalibrate(object_points, image_points)
		self.rectification_results = self._stereoRectify()
		urm_l = self._initUndistortRectifyMap('l')
		urm_r = self._initUndistortRectifyMap('r')
		self.undistort_rectify_map = {'l' : urm_l, 'r' : urm_r}

	#Dump object internal data to file
	def save(self, file):
		with open(file, 'wb') as f: pickle.dump({
			'imageSize' : self.imageSize,
			'calibration_results' : self.calibration_results,
			'rectification_results' : self.rectification_results,
			'undistort_rectify_map' : self.undistort_rectify_map
		}, f)

	#Load object internal data from file
	def load(self, file):
		with open(file, 'rb') as f: data = pickle.load(f)
		self.imageSize = data['imageSize']
		self.calibration_results = data['calibration_results']
		self.rectification_results = data['rectification_results']
		self.undistort_rectify_map = data['undistort_rectify_map']
		return self

	#Rectify and return left and right images
	def rectify_image_pair(self, img_l, img_r, interpolation = cv.INTER_NEAREST):
		if self.undistort_rectify_map is None: raise ValueError('undistortion & rectification are required to rectify an image pair')

		return (
			cv.remap(img_l, self.undistort_rectify_map['l']['undistortion_map'], self.undistort_rectify_map['l']['rectification_map'], interpolation),
			cv.remap(img_r, self.undistort_rectify_map['r']['undistortion_map'], self.undistort_rectify_map['r']['rectification_map'], interpolation)
		)

	#Crop image: Obsolete
	def crop_image(self, img, minDisparity, numDisparities, blockSize):
		x,y,w,h = cv.getValidDisparityROI(
			self.rectification_results['validBoxes_l'],
			self.rectification_results['validBoxes_r'],
			minDisparity, numDisparities, blockSize
		)
		return img[x:x+w, y:y+h]


#	def crop_image_pair(self, img_l, img_r, equal_size = True):
#		x_l, y_l, w_l, h_l = self.rectification_results['validBoxes_l']
#		x_r, y_r, w_r, h_r = self.rectification_results['validBoxes_r']

#		cropped_l = img_l[y_l:y_l+h_l, x_l:x_l+w_l]
#		cropped_r = img_r[y_r:y_r+h_r, x_r:x_r+w_r]

#		if not equal_size: return (cropped_l, cropped_r)

#		y_l, x_l = cropped_l.shape
#		y_r, x_r = cropped_r.shape

#		min_y = min(y_l, y_r)
#		min_x = min(x_l, x_r)

#		cropped_l = cropped_l[0:min_y, 0:min_x]
#		cropped_r = cropped_r[0:min_y, 0:min_x]

#		return (cropped_l, cropped_r)

#	def rectify_and_crop_image_pair(self, img_l, img_r, interpolation = cv.INTER_NEAREST, equal_size = True):
#		return self.crop_image_pair(*self.rectify_image_pair(img_l, img_r, interpolation), equal_size)
