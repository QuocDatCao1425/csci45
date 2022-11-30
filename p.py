import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def EdgeImage(grayscale_image): #Sobel method
	height=grayscale_image.shape[0]
	width =grayscale_image.shape[1]
	img2=np.zeros((height,width), dtype=np.uint8)
	img=grayscale_image
	for i in range(1,height-1):
		for j in range(1,width-1):
			#horizontal kernal
			x=img[i-1][j-1]*(-1) +img[i][j-1]*(-2) +img[i+1][j-1]*(-1)    +img[i-1][j+1]*(1) +img[i][j+1]*(2) +img[i+1][j+1]*(1)
			#vertical kernel
			y=img[i-1][j-1]*(1) +img[i-1][j]*(2) +img[i-1][j+1]*(1)    +img[i+1][j-1]*(-1) +img[i+1][j]*(-2) +img[i+1][j+1]*(-1)
			v=math.sqrt(x*x+y*y)
			if v>255:
				v=255
			img2[i][j]=v

	return img2

def BinaryImage(edge_image, threshold):
	height=edge_image.shape[0]
	width =edge_image.shape[1]
	b_w_image=edge_image
	for i in range(height):
		for j in range(width):
			if edge_image[i][j]<threshold:
				b_w_image[i][j]=0
			if edge_image[i][j]>=threshold:
				b_w_image[i][j]=255
	return b_w_image

def HoughGraph(b_w_image):
	#longest distance sqrt(height^2 + width^2)
	height=b_w_image.shape[0]
	width =b_w_image.shape[1]

	longest_distance=round(math.sqrt(height*height+width*width))	
	Hough_Graph=np.zeros((longest_distance,361), dtype=np.uint8)
	Graph=np.zeros((longest_distance,361), dtype=np.uint8)
	#Graph=np.zeros((longest_distance,181), dtype=np.uint8)
	
	count =0
	for y in range(height):
		for x in range(width):
			if(b_w_image[y][x]==255):
				for a in range(361):
					dist=x*math.cos(math.radians(a))+y*math.sin(math.radians(a))
					dist=round(dist)
					if Hough_Graph[dist][a]<255:
						Hough_Graph[dist][a]+=1
					else:
						Hough_Graph[dist][a]=255


	
	return Hough_Graph



#---------------------------------------------------------------------------------------------------------------------------


def drawGraph(hough_graph, original_image, threshold):
	
	arr=[]
	height=hough_graph.shape[0]
	width =hough_graph.shape[1]
	for r in range(height):
		for angle in range(width):
			if hough_graph[r][angle]>=threshold:
				#convert to radian
				x1=r*math.cos((angle)*math.pi/180)
				y1=r*math.sin((angle)*math.pi/180)
				x2 = int(x1 - 10000*(-math.sin((angle)*math.pi/180)))
				y2 = int(y1 - 10000*( math.cos((angle)*math.pi/180)))
				arr.append([x1,y1,x2,y2])
				
				
	for i in arr:
		plt.axline((i[0], i[1]), (i[2], i[3]), linewidth=1, color='r')
	plt.imshow(original_image)
	plt.show()
				

#-----------------------------------------------------------------

grayscale_image= cv2.imread("tower.jpg",cv2.IMREAD_GRAYSCALE)
cv2.imshow('Grayscale image',grayscale_image)

edge_detected_image=EdgeImage(grayscale_image)
cv2.imshow('edge image',edge_detected_image)

x1=np.amax(edge_detected_image)
print('max grayscale value of the edge detected image is: ', x1)
threshold1=int(input('Choose threshold for binary image filter: value1 = '))

binary_image=BinaryImage(edge_detected_image,threshold1)
cv2.imshow('Black White Image',binary_image)

hg=HoughGraph(binary_image)
cv2.imshow('Hough Graph',hg)

x2=np.amax(hg)
print('-----------------------')
print('max grayscale value in hough space is: ', x2)
threshold2=int(input('Choose threshold for output line detection image: value2 = '))

drawGraph(hg, grayscale_image,threshold2)

cv2.waitKey(0)
cv2.destroyAllWindows()


