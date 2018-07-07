#!usr/bin/python
import heapq
import copy
import cv2
import numpy as np
import random
import time
import multiprocessing as mp

'''
#puzzle variable
outer_layer = {(0,0), (0,1), (0,2), (0,3), (0,4), (0,5),
			   (1,0),							  (1,5),
			   (2,0),							  (2,5),
			   (3,0),							  (3,5),
			   (4,0), (4,1), (4,2), (4,3), (4,4), (4,5)}

left_layer = {(1,0),
			  (2,0),
			  (3,0)}

right_layer = {(1,5),
			   (2,5),
			   (3,5)}

top_layer = {(0,1), (0,2), (0,3), (0,4)}

bottom_layer = {(4,1), (4,2), (4,3), (4,4)}

lt_corner = {(0,0)}

rt_corner = {(0,5)}

lb_corner = {(4,0)}

rb_corner = {(4,5)}

Initial = [[0, 1, 2, 3, 4, 5],
		   [1, 2, 3, 4, 5, 0],
		   [2, 3, 4, 5, 0, 1],
		   [3, 4, 5, 0, 1, 2],
		   [4, 5, 0, 1, 2, 3]]
'''

#interface variable
cols = 600
rows = 500

red = [0, 0, 255]
blue = [255, 0, 0]
green = [0, 128, 0]
yellow = [0, 255, 255]
purple = [128, 0,128]
pink = [238, 130, 238]

col_div = 6  #-
row_div = 5  #|
w = cols / col_div
h = rows / row_div

proc = 4

#class state-------------------------------------------
class State:
	
	def __init__(self, arrangement, hold=None, pre_action=None):
		self.arrangement = arrangement
		self.hold = hold
		self.pre_action = pre_action

#pazudora problem ----------------------------------------
def actions(A):
	move = {0, 1, 2, 3}
	if A.hold[0] == 0:
		move -= {0}
	if A.hold[0] == 4:
		move -= {2}
	if A.hold[1] == 0:
		move -= {1}
	if A.hold[1] == 5:
		move -= {3}
	if A.pre_action != None:
		return move - {(A.pre_action + 2) % 4}
	else:
		return move


def result(state, action):
	x = state.hold[0]
	y = state.hold[1]
	arrangement = copy.deepcopy(state.arrangement)
	if action == 0: #up
		arrangement[x][y], arrangement[x - 1][y] = arrangement[x - 1][y], arrangement[x][y]
		x -= 1
	elif action == 1: #left
		arrangement[x][y], arrangement[x][y - 1] = arrangement[x][y - 1], arrangement[x][y]
		y -= 1
	elif action == 2: #down
		arrangement[x][y], arrangement[x + 1][y] = arrangement[x + 1][y], arrangement[x][y]
		x += 1
	elif action == 3: #right
		arrangement[x][y], arrangement[x][y + 1] = arrangement[x][y + 1], arrangement[x][y]
		y += 1
	return State(arrangement, hold = (x, y), pre_action = action)

def value(state, node_depth, depth):
	count, combos = combo_count(state, node_depth, depth)
	state2 = combo_clear(state, combos)
	return count + combo_count(state2, node_depth, depth)[0]

#class node--------------------------------------------
class Node:
	
	def __init__(self, state, parent=None, action=None):
		self.state = state
		self.parent = parent
		self.action = action
		self.depth = 0
		if parent:
			self.depth = parent.depth + 1

	def __repr__(self):
		return "<Node %s>" % (self.state,)

	def __lt__(self, node):
		return self.state < node.state

	def expand(self):
		return [self.child_node(action) for action in actions(self.state)]

	def child_node(self, action):
		next = result(self.state, action)
		return Node(next, self, action)

	def solution(self):
		path = self.path()
		return [path[0]] + [node.action for node in path[2:]]

	def path(self):
		node, path_back = self, []
		while node:
			path_back.append(node)
			if node.parent == None:
				path_back.append(node.state.hold)
				break
			node = node.parent
		return list(reversed(path_back))

	def __eq__(self, other):
		return isinstance(other, Node) and self.state == other.state

	def __hash__(self):
		return hash(self.state)

#combo-------------------------------------------------

class Combo():

	def __init__(self):
		self.combo_dict = {'0':[], '1':[], '2':[], '3':[], '4':[], '5':[]}

def surroundings(combo):
	drops = []
	for drop in combo:
		drops.extend([drop, (drop[0]-1, drop[1]), (drop[0]+1, drop[1]), (drop[0], drop[1]-1), (drop[0], drop[1]+1)])
	return set(drops)

def combo_combine(combo_list):
	if combo_list == []:
		return []
	head = combo_list.pop(0)
	combo_new = [[head, surroundings(head)]]
	while combo_list:
		combo = combo_list.pop(0)
		for index in range(0, len(combo_new)):
			if combo & combo_new[index][1] != set():
				combo_new[index][0] |= combo
				combo_new[index][1] |= surroundings(combo)
				break
		else:
			combo_new.append([combo, surroundings(combo)])
	return [group[0] for group in combo_new]


def combo_count(state, node_depth, depth):
	combos = Combo()
	count = 0
	arrangement = copy.deepcopy(state.arrangement)
	if (depth - node_depth) > 4 and state.hold != None:
		arrangement[state.hold[0]][state.hold[1]] = 6
	for i in range(0, row_div):
		for j in range(0, col_div - 2):
			focus = arrangement[i][j]
			focus_ = arrangement[i][j+1]
			focus__ = arrangement[i][j+2]
			average_ = (focus + focus_ + focus__) / 3.0
			if (focus - average_) ** 2 + (focus_ - average_) ** 2 + (focus__ - average_) ** 2 == 0 and average_ != 6:
				combos.combo_dict[str(int(average_))].append({(i, j), (i, j+1), (i, j+2)})
	for i in range(0, row_div - 2):
		for j in range(0, col_div):
			focus = arrangement[i][j]
			focusl = arrangement[i+1][j]
			focusll = arrangement[i+2][j]
			averagel = (focus + focusl + focusll) / 3.0
			if (focus - averagel) ** 2 + (focusl - averagel) ** 2 + (focusll - averagel) ** 2 == 0 and averagel != 6:
				combos.combo_dict[str(int(averagel))].append({(i, j), (i+1, j), (i+2, j)})
	for color in combos.combo_dict.keys():
		combos.combo_dict[color] = combo_combine(combos.combo_dict[color])
		count += len(combos.combo_dict[color])
		#print color,
		#print ':',
		#print combos.combo_dict[color]

	return (count, combos)

def rearrange(arrangement, drops):
	heapq.heapify(drops)
	while drops:
		drop = heapq.heappop(drops)
		for i in range(0, drop[0]):
			arrangement[drop[0] - i][drop[1]] = arrangement[drop[0] - i - 1][drop[1]]
		arrangement[0][drop[1]] = 6

def combo_clear(state, combos):
	arrangement = copy.deepcopy(state.arrangement)
	Combos = copy.deepcopy(combos)
	drops = []
	for color in Combos.combo_dict.keys():
		while Combos.combo_dict[color]:
			combo = list(Combos.combo_dict[color].pop())
			while combo:
				drops.append(combo.pop())
	rearrange(arrangement, drops)
	return State(arrangement)

def combo_max(arrangement):
	drop_list = [0, 0, 0, 0, 0, 0]
	count = 0
	for i in range(0, row_div):
		for j in range(0, col_div):
			drop_list[arrangement[i][j]] += 1
	while drop_list:
		drop = drop_list.pop()
		count += drop // 3
	return count

#beam_search-------------------------------------------------
def pazudora_beamsearch(initial, queue = None, width = 90, depth = 24, frequency = 3):
	candidates = [Node(State(initial, (index/6, index%6))) for index in range(0, 30)]
	while candidates[0].depth < depth:
		for times in range(0, frequency):
			length = len(candidates)
			for index in range(0, length):
				node = candidates.pop(0)
				candidates.extend(node.expand())

		can_heap = []
		for candidate in candidates:
			heapq.heappush(can_heap, (-1 * value(candidate.state, candidate.depth, depth), candidate))
		candidates = [heapq.heappop(can_heap)[1] for i in range(0, width)]

	if queue != None:
		queue.put(candidates[0].solution())
	else:
		return candidates[0].path()


def pazudora_beamsearch2(initial, queue = None, width = 120, depth = 24, frequency = 4):
	potential = combo_max(initial)
	candidates = [Node(State(initial, (index/6, index%6))) for index in range(0, 30)]
	good_list = []
	best = (0,None)

	while candidates[0].depth < depth:
		for times in range(0, frequency):
			length = len(candidates)
			for index in range(0, length):
				node = candidates.pop(0)
				candidates.extend(node.expand())

		can_heap = []
		goo_heap = []
		for candidate in candidates:
			heapq.heappush(can_heap, (-1 * value(candidate.state, candidate.depth, depth), candidate))
			heapq.heappush(goo_heap, (-1 * value(candidate.state, depth, depth), candidate))
		candidates = [heapq.heappop(can_heap)[1] for i in range(0, int(width))]
		good_list.append(heapq.heappop(goo_heap))

		if candidates[0].depth == 18:
			#width //= 1.5
			#frequency -= 1
			print '18'
		elif candidates[0].depth == 12:
			#width //= 2
			frequency -= 1
			print '12'

	while good_list:
		can = good_list.pop(0)
		if can[0] < best[0]:
			best = can

	if queue != None:
		queue.put(best[1].solution())
	else:
		return best[1].path()

def pazudora_beamsearch3(initial, queue = None, width = 120, depth = 24, frequency = 4):
	potential = combo_max(initial)
	candidates = [Node(State(initial, (index/6, index%6))) for index in range(0, 30)]
	good_list = []
	best = (0,None)

	while candidates[0].depth < depth:
		for times in range(0, frequency):
			length = len(candidates)
			for index in range(0, length):
				node = candidates.pop(0)
				candidates.extend(node.expand())

		can_heap = make_can_heap(candidates)
		goo_heap = make_goo_heap(candidates)
		heapq.heapify(can_heap)
		heapq.heapify(goo_heap)
		candidates = [heapq.heappop(can_heap)[1] for i in range(0, int(width))]
		good_list.append(heapq.heappop(goo_heap))

		if candidates[0].depth == 18:
			#width //= 1.5
			#frequency -= 1
			print '18'
		elif candidates[0].depth == 12:
			#width //= 2
			frequency -= 1
			print '12'

	while good_list:
		can = good_list.pop(0)
		if can[0] < best[0]:
			best = can

	if queue != None:
		queue.put(best[1].solution())
	else:
		return best[1].path()

def make_can_heap(candidates):
	que = mp.Queue()
	length = len(candidates) // 4
	ps = [
		mp.Process(target=sub_make_can_heap, args=(que, 0, length, candidates)), 
		mp.Process(target=sub_make_can_heap, args=(que, 1, length, candidates)), 
		mp.Process(target=sub_make_can_heap, args=(que, 2, length, candidates)), 
		mp.Process(target=sub_make_can_heap, args=(que, 3, length, candidates))
		]

	for p in ps:
		p.start()

	can_heap = []
	for i in range(4):
		can_heap.extend(que.get())

	return can_heap

def sub_make_can_heap(que, index, length, candidates):
	sub_heap = []

	ini = index * length
	fin = (index + 1) * length

	for i in range(ini, fin):
		candidate = candidates[i]
		sub_heap.append((-1 * value(candidate.state, candidate.depth, 24), candidate))

	que.put(sub_heap)

def make_goo_heap(candidates):
	que = mp.Queue()
	length = len(candidates) // 4

	ps = [
		mp.Process(target=sub_make_goo_heap, args=(que, 0, length, candidates)), 
		mp.Process(target=sub_make_goo_heap, args=(que, 1, length, candidates)),
		mp.Process(target=sub_make_goo_heap, args=(que, 2, length, candidates)),
		mp.Process(target=sub_make_goo_heap, args=(que, 3, length, candidates))
		]
	for p in ps:
		p.start()

	goo_heap = []
	for i in range(4):
		goo_heap.extend(que.get())

	return goo_heap

def sub_make_goo_heap(que, index, length, candidates):
	sub_heap = []

	ini = index * length
	fin = (index + 1) * length

	for i in range(ini, fin):
		candidate = candidates[i]
		sub_heap.append((-1 * value(candidate.state, 24, 24), candidate))

	que.put(sub_heap)



#interface---------------------------------------------------

def random_generate():
	arrangement = np.random.randint(0, 6, 30)
	arrangement = arrangement.reshape((5, 6))
	return arrangement.tolist()

def display(arrangement, window_name, hold=None):
	arran_image = np.zeros((rows, cols, 3), np.uint8)

	for x in range(0, row_div):
		x1 = x * h
		for y in range(0, col_div):
			y1 = y * w
			if arrangement[x][y] == 0:
				cv2.circle(arran_image, (y1+w/2, x1+h/2), w/3, red, -1)
			elif arrangement[x][y] == 1:
				cv2.circle(arran_image, (y1+w/2, x1+h/2), w/3, blue, -1)
			elif arrangement[x][y] == 2:
				cv2.circle(arran_image, (y1+w/2, x1+h/2), w/3, green, -1)
			elif arrangement[x][y] == 3:
				cv2.circle(arran_image, (y1+w/2, x1+h/2), w/3, yellow, -1)
			elif arrangement[x][y] == 4:
				cv2.circle(arran_image, (y1+w/2, x1+h/2), w/3, purple, -1)
			elif arrangement[x][y] == 5:
				cv2.circle(arran_image, (y1+w/2, x1+h/2), w/3, pink, -1)

	if hold:
		cv2.circle(arran_image, (hold[1] * w + w/2, hold[0] * h + h/2), 10, [255, 255, 255], -1)
	#return arran_image
	cv2.imshow(window_name, arran_image)

if __name__ == '__main__':

	cv2.namedWindow('puzzle_arrangement', cv2.WINDOW_NORMAL)
	#cv2.namedWindow('puzzle_clear', cv2.WINDOW_NORMAL)
	'''
	while True:
		arrangement = random_generate()
		state = State(arrangement)
		solution = combo_count(state, 2, 10)
		print solution[0]
		#print solution[1]
		display(state.arrangement, 'puzzle_arrangement')
		#print combo_clear(state, solution[1]).arrangement
		display(combo_clear(state, solution[1]).arrangement, 'puzzle_clear')
		k = cv2.waitKey(0)
		if k == 48:
			continue
		else:
			break
	cv2.destroyAllWindows()
	'''


	'''
	while True:
		arrangement = random_generate()
		display(arrangement, 'puzzle_arrangement', (3,4))
		print combo_count(State(arrangement, (3,4)),3, 10)[0]
		k = cv2.waitKey(0)
		if k == 48:
			continue
		elif k == 27:
			break
		else:
			break
	cv2.destroyAllWindows()
	'''
	
	arrangement = random_generate()
	solution = pazudora_beamsearch3(arrangement, width = 120, depth = 24, frequency = 3)
	hold = solution.pop(0)
	print 'route :', len(solution) - 1 
	for i in solution:
		display(i.state.arrangement, 'puzzle_arrangement', i.state.hold)
		print i.state.hold
		cv2.waitKey(500)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	



	



