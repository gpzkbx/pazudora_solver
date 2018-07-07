#!usr/bin/python

from pazudora_improve_test import *
import numpy as np
import cv2
import copy
import time
import multiprocessing
import socket

x0 = 0
x1 = 600
y0 = 225
y1 = 575

'''
red_l = 178
red_h = 10
blue_l = 86
blue_h = 117
green_l = 56
green_h = 78
yellow_l = 20
yellow_h = 35
purple_l = 130
purple_h = 150
pink_l = 155
pink_h = 170
'''

red_range = range(178, 180) + range(0, 20)
blue_range = range(86, 117)
green_range = range(56, 78)
yellow_range = range(25, 35)
purple_range = range(130, 140)
pink_range = range(150, 170)

red = [0, 0, 255]
blue = [255, 0, 0]
green = [0, 128, 0]
yellow = [0, 255, 255]
purple = [128, 0,128]
pink = [238, 130, 238]


def getaverage(focus, index):
	tmp = []
	for x in range(0, 10):
		for y in range(0, 10):
			tmp.append(focus[24+x][24+x][index])
	tmp = np.array(tmp)
	return np.average(tmp)

def getmost(focus, index):
	tmp = []
	for x in range(0, 10):
		for y in range(0, 10):
			tmp.append(focus[24+x][24+x][index])
	return max(set(tmp), key = tmp.count)

def update(color_list, result):
	tmp = copy.deepcopy(color_list)
	for x in range(0, 5):
		for y in range(0, 6):
			if result[x][y] != 6:
				tmp[x][y] = result[x][y]
	return tmp

def draw_circles(arrange_img, color_list):
	for x in range(0, 5):
		c_x = x * 58
		for y in range(0, 6):
			c_y = y * 58
			if color_list[x][y] == 0:
				cv2.circle(arrange_img, (c_y+29, c_x+29), 29, red, 2)
			elif color_list[x][y] == 1:
				cv2.circle(arrange_img, (c_y+29, c_x+29), 29, blue, 2)
			elif color_list[x][y] == 2:
				cv2.circle(arrange_img, (c_y+29, c_x+29), 29, green, 2)
			elif color_list[x][y] == 3:
				cv2.circle(arrange_img, (c_y+29, c_x+29), 29, yellow, 2)
			elif color_list[x][y] == 4:
				cv2.circle(arrange_img, (c_y+29, c_x+29), 29, purple, 2)
			elif color_list[x][y] == 5:
				cv2.circle(arrange_img, (c_y+29, c_x+29), 29, pink, 2)

def color_judge(arrange_img):
	color_list = np.zeros((5,6))
	#img = np.copy(arrange_img)
	for x in range(0, 5):
		c_x = x * 58
		for y in range(0, 6):
			c_y = y * 58
			focus = arrange_img[c_x: c_x + 57, c_y: c_y + 57]
			focus = cv2.cvtColor(focus, cv2.COLOR_BGR2HSV)
			center = getmost(focus, 0)
			if center in red_range:
				color_list[x][y] = 0
			elif center in blue_range:
				color_list[x][y] = 1
			elif center in green_range:
				color_list[x][y] = 2
			elif center in yellow_range:
				color_list[x][y] = 3
			elif center in purple_range:
				color_list[x][y] = 4
			elif center in pink_range:
				color_list[x][y] = 5
			else:
				color_list[x][y] = 6
	return color_list

def bright_judge(arrange_img):
	bright = 0
	for x in range(0, 5):
		c_x = x * 58
		for y in range(0, 6):
			c_y = y * 58
			focus = arrange_img[c_x: c_x + 57, c_y: c_y + 57]
			focus = cv2.cvtColor(focus, cv2.COLOR_BGR2HSV)
			bright += getmost(focus, 2)
	bright /= 30
	return bright

def communicate(bmovement):
	server_address = ('192.168.11.27', 6789)
	max_size = 1000
	client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	client.connect(server_address)
	
	client.sendall(bmovement)
	response = client.recv(max_size)
	client.close()

if __name__ == '__main__':

	mode = 0
	ok_to_cal_flag = 1
	ok_to_com_flag = 1
	cal_over_flag = 0
	move_over_flag = 0
	start = 0
	queue = multiprocessing.Queue()
	movement = 0

	cap = cv2.VideoCapture(0)
	cv2.namedWindow('origin', cv2.WINDOW_NORMAL)
	#cv2.namedWindow('screen', cv2.WINDOW_NORMAL)
	cv2.namedWindow('arrangement', cv2.WINDOW_NORMAL)

	color_list = np.zeros((5, 6))
	color_list_old = np.ones((5, 6)) * 6

	while(cap.isOpened()):
		ret, img = cap.read()
		if ret == True:
			height, width = img.shape[:2]
			img = cv2.resize(img, (int(width * 600 / height), 600))
			cv2.rectangle(img, (y0, x0), (y1, x1), (0, 0, 255), 10)
			#print (width, height)
			#screen = img[x0 + 6: x1 - 5, y0 + 6: y1 - 5]
			arrangement = img[x1 - 290: x1 - 10, y0 + 10: y1 - 10]
			arrangement = cv2.resize(arrangement, (348, 290))
			arrangement_copy = np.copy(arrangement)

			if mode == 0:  #waitmode
				print mode
				if start == 0:
					start = time.time()
				color_list = update(color_list, color_judge(arrangement))
				print bright_judge(arrangement)
				if (time.time() - start >= 0.5):
					if (len(color_list[color_list == color_list_old]) == 30) and bright_judge(arrangement) > 200:
						mode = 1
						color_list_old = np.ones((5,6)) * 6
						ok_to_cal_flag = 1
					color_list_old = np.copy(color_list)
					start = 0
				cv2.imshow('origin', img)
				cv2.imshow('arrangement', arrangement)
				if cv2.waitKey(1) == 27:
					break
			elif mode == 1:  #drawing circle, calculating
				print mode
				color_list = update(color_list, color_judge(arrangement))
				if ok_to_cal_flag:
					p = multiprocessing.Process(target = pazudora_beamsearch, args = (color_list.tolist(), queue))
					p.start()
					ok_to_cal_flag = 0
				if not p.is_alive(): # 'c'
					cal_over_flag = 1
				draw_circles(arrangement, color_list)
				print color_list
				if cal_over_flag:
					cal_over_flag = 0
					mode = 2
					ok_to_com_flag = 1
					arrangement_copy = np.copy(arrangement)
					movement = queue.get()
				cv2.imshow('origin', img)
				cv2.imshow('arrangement', arrangement)
				if cv2.waitKey(1) == 27:
					break
			elif mode == 2:  #moving touchpen 
				print mode
				print 'moving'
				print movement
				if ok_to_com_flag:
					p_x, p_y = movement.pop(0)
					movement.extend([p_x, p_y])
					bmovement = ''.join(chr(i) for i in movement)
						
					p = multiprocessing.Process(target = communicate, args = (bmovement,))
					p.start()
					ok_to_com_flag = 0
				if not p.is_alive():
					move_over_flag = 1
				if move_over_flag:
					mode = 0
					move_over_flag = 0
				cv2.imshow('origin', img)
				cv2.imshow('arrangement', arrangement_copy)
				if cv2.waitKey(1) == 27:
					break
		else:
			break
	cap.release()
	cv2.destroyAllWindows()







