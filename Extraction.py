"""
	Neural Network
	by Vincent Jeanselme
	vincent.jeanselme@gmail.com
"""

import os
import struct
import urllib.request
import io
import gzip
import numpy as np

def extractImagesLabels(labelsFileName, imagesFileName):
	"""
	Functions which creates list of images and labels by extracting the content
	of the given files
	"""
	imagesFile = open(imagesFileName,"rb")
	imagesHeader = struct.unpack(">4L", imagesFile.read(struct.calcsize(">4L")))

	labelsFile = open(labelsFileName,"rb")
	labelsHeader = struct.unpack(">2L", labelsFile.read(struct.calcsize(">2L")))

	magic, nimages, height, width = imagesHeader
	magic, nlabels = labelsHeader

	imagesRes = []
	labelsRes = []

	for n in range(nimages):
		label = np.zeros((10, 1))
		label[int.from_bytes(labelsFile.read(1),byteorder='big')] = 1.0
		img = np.array([(int.from_bytes(imagesFile.read(1),byteorder='big'))/128 - 1
			for i in range(height * width)])
		imagesRes.append(img.reshape((784,1)))
		labelsRes.append(label)

	return np.array(labelsRes), np.array(imagesRes)

def downloadDecompress(url, fileName, saveDirectory):
	"""
	Downloads and extract the content of the given fileName
	"""
	if not(os.path.exists(saveDirectory + fileName)):
		response = urllib.request.urlopen(url + fileName + ".gz")
		compressedFile = io.BytesIO(response.read())

		with open(saveDirectory + fileName, 'wb') as out:
		    out.write(compressedFile.read())
