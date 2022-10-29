#!/usr/bin/env python3

"""
RBE/CS549 Spring 2022: Computer Vision
Homework 0: Alohomora: Phase 1 Starter Code

Colab file can be found at:
	https://colab.research.google.com/drive/1FUByhYCYAfpl8J9VxMQ1DcfITpY8qgsF

Author(s): 
Prof. Nitin J. Sanket (nsanket@wpi.edu), Lening Li (lli4@wpi.edu), Gejji, Vaishnavi Vivek (vgejji@wpi.edu)
Robotics Engineering Department,
Worcester Polytechnic Institute

Code adapted from CMSC733 at the University of Maryland, College Park.
"""

# Code starts here:

from cgitb import text
import numpy as np
import cv2
from matplotlib import image
import skimage.transform as tf
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


#To convolve an image and filter
def convolve2d(image, kernel, padding=0, stride =1):
	kernel = np.flipud(np.fliplr(kernel))
	kernel_x = kernel.shape[0]
	kernel_y = kernel.shape[1]
	image_x = image.shape[0]
	image_y = image.shape[1]
	padding = int((kernel_x - 1)/2)
	x = int((image_x + 2*padding - kernel_x)/stride + 1)
	y = int((image_y + 2*padding - kernel_y)/stride + 1)

	filtered_image = np.zeros((x,y))

	padded_image = cv2.copyMakeBorder(image, padding, padding, padding, padding,cv2.BORDER_DEFAULT)
	
	for y in range(image.shape[1]):
		if y>image.shape[-1] - kernel_y:
			break
		if y%stride == 0:
			for x in range (image.shape[0]):
				if x>image.shape[0] - kernel_x:
					break
				try:
					if x%stride == 0:
						filtered_image[x,y] = (kernel * padded_image[x:x+kernel_x, y:y+kernel_y]).sum()
				except:
					break
	
	return filtered_image

#gaussian 1-d (specifying the degree of derivative of gaussian)
def Gaussian1d(sigma, x, d_order):
	gauss = 1 / np.sqrt(2 * np.pi * sigma**2) * np.exp(-(x**2) / (2 * sigma**2))

	if d_order == 0:
		return gauss
	elif d_order == 1:        #1st order
		return -gauss*(x/sigma**2)
	elif d_order == 2:        #2nd order
		return gauss*((x**2-sigma**2)/sigma**4)
	else:
		return gauss

# make 2d gaussian using 1d gaussians (use when sigma different for both x and y)
def filter_gauss2d(sigma, kernel_size, d_order_x, d_order_y):
	interval = kernel_size/2.5
	[x, y] = np.meshgrid(np.linspace(-interval, interval, kernel_size),
						np.linspace(-interval,interval,kernel_size))
	grid = np.array([x.flatten(), y.flatten()])
	gauss_x = Gaussian1d(3*sigma, grid[0,...], d_order_x)
	gauss_y = Gaussian1d(sigma, grid[1,...], d_order_y)
	gauss = gauss_x * gauss_y
	filter = np.reshape(gauss, (kernel_size, kernel_size))
	return filter

# make2d Gaussian filter when sigma same for a x and y
def GaussianFilter2d(sigma, kernel_size):
	interval = kernel_size/2.5
	x, y = np.meshgrid(np.linspace(-interval, interval, kernel_size),
						np.linspace(-interval,interval,kernel_size))
	gauss = 1 / (2 * np.pi * sigma**2) * np.exp(-(x**2 + y**2) / (2 * sigma**2))
	return gauss

# make Laplacian of gaussian filter
def Laplacian(sigma, kernel_size):
	laplacian_filter = np.array([[0, 1, 0],
								[1, -4, 1],
								[0, 1, 0]])
	gauss = GaussianFilter2d(sigma, kernel_size)
	log = convolve2d(gauss, laplacian_filter)
	return log

#make DoG filters
def DOG_filters(sigma, kernel_size, no_of_orient, sobel_vertical):
		dog_filters = []
		orientations = np.linspace(0,360,no_of_orient)                #define orientations
		for x in sigma:
			gauss_kernel = GaussianFilter2d(x, kernel_size)
			sobel_convolve = convolve2d(gauss_kernel, sobel_vertical)
			for i in range(0, no_of_orient):
				filter = tf.rotate(sobel_convolve, orientations[i])
				dog_filters.append(filter)
		return dog_filters

#make LM filters
def LMfilters(sigma, kernel_size):
	filters = []
	orientations = np.linspace(0,180,6)
	# first and second order derivatives of gaussian filter
	for i in range(0,len(sigma)-1):
		gauss_kernel = filter_gauss2d(sigma[i], kernel_size, 0, 1)     #first order derivative gaussian filter
		for j in range(0,len(orientations)):
			filter = tf.rotate(gauss_kernel, orientations[j])
			filters.append(filter)
		gauss_kernel = filter_gauss2d(sigma[i], kernel_size, 0, 2)     #second order derivative gaussian filter
		for j in range(0,len(orientations)):
			filter = tf.rotate(gauss_kernel, orientations[j])
			filters.append(filter)

	# laplacian filter at normal scale
	for i in range(0,len(sigma)):
		filters.append(Laplacian(sigma[i], kernel_size))              #laplacian filters
	# laplacian filter at three times the normal scale
	for i in range(0,len(sigma)):
		filters.append(Laplacian(3*sigma[i], kernel_size))
	# normal gaussian filter
	for i in range(0,len(sigma)):
		filters.append(GaussianFilter2d(sigma[i], kernel_size))       #normal gaussian filters
	return filters

#make gabor filters
#gabor filter code is inpired by the code available on wikipedia

def Gabor(sigma, kernel_size, theta, Lambda, psi, gamma):
	sigma_x = sigma
	sigma_y = float(sigma)/gamma
	nstds = 3 
	xmax = max(abs(nstds * sigma_x * np.cos(theta)), abs(nstds * sigma_y * np.sin(theta)))
	xmax = np.ceil(max(1, xmax))
	ymax = max(abs(nstds * sigma_x * np.sin(theta)), abs(nstds * sigma_y * np.cos(theta)))
	ymax = np.ceil(max(1, ymax))
	xmin = -xmax
	ymin = -ymax
	(y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))
	x_theta = x * np.cos(theta) + y * np.sin(theta)
	y_theta = -x * np.sin(theta) + y * np.cos(theta)

	gb = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * np.cos(2 * np.pi / Lambda * x_theta + psi)
	return gb


def gabor_filters(sigma, kernel_size, theta, Lambda, psi, gamma, number):
	filters = []
	orientations = np.linspace(90,270,number)
	for i in range(0,len(sigma)):
		gabor_kernel = Gabor(sigma[i],kernel_size, theta, Lambda, psi, gamma)
		for j in range(0, number):
			filter = tf.rotate(gabor_kernel, orientations[j])
			filters.append(filter)
	return filters

# to plot a given set of images in grid
def plot_images(fig_size, filters, x_len, y_len, name):
	fig = plt.figure(figsize = fig_size)
	length = len(filters)
	for idx in np.arange(length):
		ax = fig.add_subplot(y_len, x_len, idx+1, xticks = [], yticks = [])
		plt.imshow(filters[idx], cmap = 'gray')
	plt.savefig(name)
	plt.close()

#convolve filter banks with image (to be used for texton maps)
def texton(image, filter_bank):
	t_map = np.array(image)
	for i in range(0, len(filter_bank)):
		filter = np.array(filter_bank[i])
		filter_map = convolve2d(image, filter)
		t_map = np.dstack((t_map, filter_map))
	return t_map

#create texton maps
def Texton_map(image, dog_F, lm_F, gabor_F, clusters):
	size = image.shape
	t_map_dog = texton(image, dog_F)       #convolve image with DoG filter bank
	t_map_lm = texton(image, lm_F)         #convolve image with LM filter bank
	t_map_gabor = texton(image, gabor_F)   #convolve image with Gabor filter bank
	t_map = np.dstack((t_map_dog, t_map_lm, t_map_gabor))          #filter outputs stacked in form of image channels
	total_filters = t_map.shape[2]
	l = size[0]
	b = size[1]
	tex = np.reshape(t_map, ((l*b),total_filters))    
	kmeans = KMeans(n_clusters=clusters, random_state=0).fit(tex)       #Kmeans applied to cluster each image pixel between 1 to 64.
	pred = kmeans.predict(tex)
	pred_ = np.reshape(pred, (l,b))
	
	return pred_

#to create brightness maps
def brightness_map(image, clusters):        #grayscale input image is used
	image = np.array(image)
	size = image.shape
	image_ = np.reshape(image, ((size[0]*size[1]),1))
	kmeans = KMeans(n_clusters=clusters, random_state=0).fit(image_)
	pred = kmeans.predict(image_)                                  #Kmeans applied to cluster each image pixel between 1 to 16
	pred_ = np.reshape(pred, (size[0],size[1]))
	
	return pred_

def color_map(image, clusters):       #color image input is used
	image = np.array(image)
	size = image.shape
	image_ = np.reshape(image, ((size[0]*size[1]),size[2]))
	kmeans = KMeans(n_clusters=clusters, random_state=0).fit(image_)
	pred = kmeans.predict(image_)                                 #Kmeans applied to cluster each image pixel between 1 to 16
	pred_ = np.reshape(pred, (size[0],size[1]))
	
	return pred_

def half_disc_masks(scales):
	half_discs = []
	angles = [0, 180, 30, 210, 45, 225, 60, 240, 90, 270, 120, 300, 135, 315, 150, 330]           #rotation angles (not equally spaced)
	no_of_disc = len(angles)
	for radius in scales:
		kernel_size = 2*radius + 1
		cc = radius
		kernel = np.zeros([kernel_size, kernel_size])
		for i in range(radius):
			for j in range(kernel_size):
				a = (i-cc)**2 + (j-cc)**2                                     #to create one disc
				if a <= radius**2:
					kernel[i,j] = 1
		
		for i in range(0, no_of_disc):                                       #rotate to make other discs
			mask = tf.rotate(kernel, angles[i])
			mask[mask<=0.5] = 0
			mask[mask>0.5] = 1
			half_discs.append(mask)
	return half_discs

#general function to calculate gradients
def Gradients(map, bins, filters):
	a, b = map.shape
	grad = np.array(map)
	# count = 0
	# print(len(filters))
	i = 0
	while i < len(filters)-1:
		# print(count)
		# count+=1
		chi_sqr_dist = chi_square_distance(map, bins, filters[i], filters[i+1])      #chi-square distance calculate using two opposite half disc masks
		grad = np.dstack((grad, chi_sqr_dist))                                       #stack all chi-square distances 
		i += 2
	gradient = np.mean(grad, axis = 2)                                              #take mean over all the channels to get a single gradient value
	# print(gradient)
	return gradient

def chi_square_distance(map, bins, mask, inv_mask):
	chi_sqr_dist = map*0
	for i in range(0, bins):
		tmp = np.zeros_like(map)
		tmp[map == i] = 1
		# g_i = convolve2d(tmp, mask)
		# h_i = convolve2d(tmp, inv_mask)
		g_i = cv2.filter2D(tmp, -1, mask)
		h_i = cv2.filter2D(tmp, -1, inv_mask)
		chi_sqr_dist = chi_sqr_dist + ((g_i - h_i)**2)/(g_i + h_i + 0.01)   #chi-square distance formula (0.001 added so that value does not become Nan in case denominator becomes 0)
	chi_sqr_dist = chi_sqr_dist/2
	return chi_sqr_dist

def rgb2gray(rgb):
    	return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def main():

	"""
	Generate Difference of Gaussian Filter Bank: (DoG)
	Display all the filters in this filter bank and save image as DoG.png,
	use command "cv2.imwrite(...)"
	"""
	sobel_vertical = np.array([[-1, 0, 1],
								[-4, 0, 4],
								[-1, 0, 1]])
	sigma1 = [3,5]
	kernel_size = 49
	orientations1 = 16
	dog_filters = DOG_filters(sigma1, kernel_size, orientations1, sobel_vertical)
	plot_images((20,2), dog_filters, x_len = 16, y_len = 2, name = 'filters/DoG_filters.png')

	"""
	Generate Leung-Malik Filter Bank: (LM)
	Display all the filters in this filter bank and save image as LM.png,
	use command "cv2.imwrite(...)"
	"""
	sigma2_small = [1, np.sqrt(2), 2, 2*np.sqrt(2)]
	lm_small_filters = LMfilters(sigma2_small, kernel_size)
	plot_images((12,4), lm_small_filters, x_len = 12, y_len = 4, name = 'filters/LM.png')

	sigma2_large = [np.sqrt(2), 2, 2*np.sqrt(2), 4]
	# lm_large_filters = LMfilters(sigma2_large, kernel_size)
	# plot_images((20,4), lm_large_filters, x_len = 12, y_len = 4, name = 'LM_large_filters.png')

	"""
	Generate Gabor Filter Bank: (Gabor)
	Display all the filters in this filter bank and save image as Gabor.png,
	use command "cv2.imwrite(...)"
	"""
	sigma3 = [3,5,7,9,12]
	gabor = gabor_filters(sigma3,kernel_size, theta = 0.25, Lambda = 1, psi = 1, gamma = 1, number = 8)
	plot_images((8,5), gabor, x_len = 8, y_len = 5, name = 'filters/Gabor.png')

	"""
	Generate Half-disk masks
	Display all the Half-disk masks and save image as HDMasks.png,
	use command "cv2.imwrite(...)"
	"""
	half_disc = half_disc_masks(scales = [5, 10, 15])
	plot_images((8,6), half_disc, x_len = 8, y_len = 6, name = 'filters/HDMasks.png')

	for i in range(1, 11):
		print(i)
		img = image.imread('../BSDS500/Images/' + str(i) + '.jpg')
		gray_img = rgb2gray(img)
		maps = []
		grads = []
		comparison = []
		"""
		Generate texture ID's using K-means clustering
		Display texton map and save image as TextonMap_ImageName.png,
		use command "cv2.imwrite('...)"
		"""
		texton_m = Texton_map(img, dog_filters, lm_small_filters, gabor, 64)
		pred_t = 3*texton_m
		cm = plt.get_cmap('gist_rainbow')
		color_pred_t = cm(pred_t)
		maps.append(color_pred_t)
		plt.imshow(color_pred_t)
		plt.savefig('texton_maps/TextonMap_' + str(i) + '.png')
		plt.close()

		"""
		Generate Brightness Map
		Perform brightness binning 
		"""
		bright_m = brightness_map(gray_img, 16)
		# print(bright_m.shape, bright_m)
		maps.append(bright_m)
		plt.imshow(bright_m, cmap = 'gray')
		plt.savefig('brightness_maps/BrightnessMap_' + str(i) +'.png')
		plt.close()

		"""
		Generate Color Map
		Perform color binning or clustering
		"""
		color_m = color_map(img, 16)
		pred_c = 30*color_m
		color_pred_c = cm(pred_c)
		maps.append(color_pred_c)
		plt.imshow(color_pred_c)
		plt.savefig('color_maps/ColorMap_' + str(i) +'.png')
		plt.close()

		#print all maps
		plot_images((12,6), maps, x_len = 3, y_len = 1, name = 'maps/' + str(i) + '.png')

		"""
		Generate Texton Gradient (Tg)
		Perform Chi-square calculation on Texton Map
		Display Tg and save image as Tg_ImageName.png,
		use command "cv2.imwrite(...)"
		"""		
		texton_gradient = Gradients(texton_m, 64, half_disc)
		grads.append(texton_gradient)
		plt.imshow(texton_gradient)
		plt.savefig('texton_gradients/Tg_' + str(i) + '.png')
		plt.close()

		"""
		Generate Brightness Gradient (Bg)
		Perform Chi-square calculation on Brightness Map
		Display Bg and save image as Bg_ImageName.png,
		use command "cv2.imwrite(...)"
		"""
		brightness_gradient = Gradients(bright_m, 16, half_disc)
		grads.append(brightness_gradient)
		plt.imshow(brightness_gradient)
		plt.savefig('brightness_gradients/Bg_' + str(i) + '.png')
		plt.close()

		"""
		Generate Color Gradient (Cg)
		Perform Chi-square calculation on Color Map
		Display Cg and save image as Cg_ImageName.png,
		use command "cv2.imwrite(...)"
		"""
		color_gradient = Gradients(color_m, 16, half_disc)
		grads.append(color_gradient)
		plt.imshow(color_gradient)
		plt.savefig('color_gradients/Cg' + str(i) + '.png')
		plt.close()

		#plot all gradients
		plot_images((12,6), grads, x_len = 3, y_len = 1, name = 'gradients/' + str(i) + '.png')

		"""
		Read Canny Baseline
		use command "cv2.imread(...)"
		"""
		img_canny = image.imread('../BSDS500/CannyBaseline/' + str(i) + '.png')
		comparison.append(img_canny)

		"""
		Read Sobel Baseline
		use command "cv2.imread(...)"
		"""
		img_sobel = image.imread('../BSDS500/SobelBaseline/' + str(i) + '.png')
		comparison.append(img_sobel)

		"""
		Combine responses to get pb-lite output
		Display PbLite and save image as PbLite_ImageName.png
		use command "cv2.imwrite(...)"
		"""
		pb_lite = (1/3)*(texton_gradient + brightness_gradient + color_gradient) * (0.5*img_canny + 0.5*img_sobel)
		comparison.append(pb_lite)
		plt.imshow(pb_lite, cmap = 'gray')
		plt.savefig('pb-lite_outputs/PbLite_' + str(i) + '.png')
		plt.close()

		#plot comparisons between canny, sobel and pb_lite
		plot_images((12, 6), comparison, x_len = 3, y_len = 1, name = 'comparison/' + str(i) + '.png')

	   
if __name__ == '__main__':
    main()
 


