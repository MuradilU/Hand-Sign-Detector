import cv2
import numpy as np
import os
from keras.models import load_model

roi_x=0.5
roi_y=0.8
contour = False

class HandSignDetector:

	def __init__(self):
		# Dimensions for roi
		self.roi_x = 0.5
		self.roi_y = 0.8
		# Flag for finding contours
		self.contour = False
		# Threshold value for image thresholding
		self.thresholdVal = 70

		self.datasetCategory = "train"
		self.currentClass = "zero"
		self.currentClassCount = 0
		self.buildDataset = False

		self.model = load_model("C:/Computer Vision/model_3.h5")

	def drawROI(self, frame):
		# Draws rectangular roi in the frame
		return cv2.rectangle(frame, (int(roi_x * frame.shape[1]), 0),
				(frame.shape[1], int(roi_y * frame.shape[0])), (255, 0, 0), 2)

	def cutROI(self, frame, thresh):
		# Cut the roi out of the frame
		return thresh[0:int(roi_y * frame.shape[0]),int(roi_x * frame.shape[1]):frame.shape[1]]

	def threshold(self, frame):
		# 1. Convert to grayscale
		# 2. Blur the gray frame for smoother thresholding
		# 3. Apply Otsu's thresholding
		# 4. Perform morphological open operation
		gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		blur = cv2.GaussianBlur(gray_frame, (5, 5), 0)
		ret, thresh = cv2.threshold(blur, self.thresholdVal, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
		kernel = np.ones((5, 5), np.uint8)
		thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
		return thresh

	def showPrediction(self, frame, prediction):
		## Function to place prediction text on main video

		predictionString = str(prediction)

		# Convert prediction class to corresponding string
		if predictionString == "6":
			predictionString = "ok"
		elif predictionString == "7":
			predictionString = "up"
		elif predictionString == "8":
			predictionString = "down"

		predictionString = "Prediction: " + predictionString
		cv2.putText(frame, predictionString, (10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0), thickness=2)
		cv2.imshow("frame", frame)

	def selectClass(self, currentClass):
		# Save png training or testing data in specified folder

		self.currentClass = currentClass
		path = "C:/Computer Vision/dataset/%s/%s" % (self.datasetCategory, self.currentClass)
		if not os.path.exists(path):
			os.mkdir(path)
		self.currentClassCount = len(os.listdir(path))
		print("Current class is " + self.currentClass)

	def startHandDetector(self):
		# Open camera
		capture = cv2.VideoCapture(0)

		while capture.isOpened():
			ret, frame = capture.read()				# Retrieve a frame
			frame = cv2.flip(frame, 1)				# Flip frame horizontally

			frame = self.drawROI(frame)				# Draw region of interest
			thresh = self.threshold(frame)			# Apply thresholding to get binary frame
			thresh = self.cutROI(frame, thresh)		# Cut out the region of interest from the frame

			cv2.imshow('frame', frame)				# Show frame
			cv2.imshow('thresh', thresh)			# Show cut out of the region of interest

			if self.contour:
				self.findContours(frame, thresh)

			cnnInput = cv2.resize(thresh, (28, 28))	# Resizing ROI to 28*28 pixels to feed into neural network

			# If in dataset mode, start saving frames in specified folder
			# Else predict 
			if self.buildDataset:
				path = "C:/Computer Vision/dataset/{0}/{1}/{1}_{2}.png".format(self.datasetCategory, self.currentClass, self.currentClassCount)
				cv2.imwrite(path, cnnInput)
				self.currentClassCount = self.currentClassCount + 1
				print("Current image count for " + self.currentClass + "class: " + str(self.currentClassCount))
			else:
				cnnInput = np.expand_dims(cnnInput, axis=-1)
				cnnInput = np.expand_dims(cnnInput, axis=0)
				prediction = self.model.predict(cnnInput)
				prediction = np.argmax(prediction[0])
				frame = self.showPrediction(frame, prediction)
				print(prediction)

			# Keyboard shortcuts
			key = cv2.waitKey(30) & 0xff
			if key == 27:
				capture.release()
				cv2.closeAllWindows()
			elif key == ord('c'):
				self.contour = True
			elif key == ord('0'):
				self.selectClass("zero")
			elif key == ord('1'):
				self.selectClass("one")
			elif key == ord('2'):
				self.selectClass("two")
			elif key == ord('3'):
				self.selectClass("three")
			elif key == ord('4'):
				self.selectClass("four")
			elif key == ord('5'):
				self.selectClass("five")
			elif key == ord('k'):
				self.selectClass("ok")
			elif key == ord('g'):
				self.selectClass("up")
			elif key == ord('b'):
				self.selectClass("down")
			elif key == ord('r'):
				self.datasetCategory = "train"
				print("Current dataset category is " + self.datasetCategory)
			elif key == ord('t'):
				self.datasetCategory = "test"
				print("Current dataset category is " + self.datasetCategory)
			elif key == ord('d'):
				self.buildDataset = not self.buildDataset

if __name__ == "__main__":
	handDetector = HandSignDetector()
	handDetector.startHandDetector()