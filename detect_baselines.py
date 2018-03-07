#!/usr/bin/python

import os
import sys
import numpy as np
import caffe
import cv2
import math
import scipy.ndimage as nd


DEBUG = True
USE_COL_DETECTION = False

# acceptable image suffixes
IMAGE_SUFFIXES = ('.jpg', '.jpeg', '.tif', '.tiff', '.png', '.bmp', '.ppm', '.pgm')

NET_FILE = os.path.join(os.path.dirname(__file__), "model.prototxt")
WEIGHTS_FILE = os.path.join(os.path.dirname(__file__), "simple_weights.caffemodel")

TILE_SIZE = 256
PADDING_SIZE = 50

# number of subwindows processed by a network in a batch
# Higher numbers speed up processing (only marginally once BATCH_SIZE > 16)
# The larger the batch size, the more memory is consumed (both CPU and GPU)
BATCH_SIZE=3

LEFT_EDGE = -2
TOP_EDGE = -1
MIDDLE = 0
RIGHT_EDGE = 1
BOTTOM_EDGE = 2

def setup_network():
	network = caffe.Net(NET_FILE, WEIGHTS_FILE, caffe.TEST)
	print "Using Weights in", WEIGHTS_FILE
	return network


def fprop(network, ims, batchsize=BATCH_SIZE):
	# batch up all transforms at once
	idx = 0
	responses = list()
	while idx < len(ims):
		sub_ims = ims[idx:idx+batchsize]

		network.blobs["data"].reshape(len(sub_ims), ims[0].shape[2], ims[0].shape[1], ims[0].shape[0])

		for x in range(len(sub_ims)):
			transposed = np.transpose(sub_ims[x], [2,0,1])
			transposed = transposed[np.newaxis, :, :, :]
			network.blobs["data"].data[x,:,:,:] = transposed

		idx += batchsize

		# propagate on batch
		network.forward()
		output = np.copy(network.blobs["prob"].data)
		responses.append(output)
		print "Progress %d%%" % int(100 * idx / float(len(ims)))
	return np.concatenate(responses, axis=0)


def predict(network, ims):
	all_outputs = fprop(network, ims)
	predictions = np.squeeze(all_outputs)
	return predictions


def get_subwindows(im):
	height, width, = TILE_SIZE, TILE_SIZE
	y_stride, x_stride, = TILE_SIZE - (2 * PADDING_SIZE), TILE_SIZE - (2 * PADDING_SIZE)
	if (height > im.shape[0]) or (width > im.shape[1]):
		print "Invalid crop: crop dims larger than image (%r with %r)" % (im.shape, tokens)
		exit(1)
	ims = list()
	bin_ims = list()
	locations = list()
	y = 0
	y_done = False
	while y  <= im.shape[0] and not y_done:
		x = 0
		if y + height > im.shape[0]:
			y = im.shape[0] - height
			y_done = True
		x_done = False
		while x <= im.shape[1] and not x_done:
			if x + width > im.shape[1]:
				x = im.shape[1] - width
				x_done = True
			locations.append( ((y, x, y + height, x + width),
					(y + PADDING_SIZE, x + PADDING_SIZE, y + y_stride, x + x_stride),
					 TOP_EDGE if y == 0 else (BOTTOM_EDGE if y == (im.shape[0] - height) else MIDDLE),
					  LEFT_EDGE if x == 0 else (RIGHT_EDGE if x == (im.shape[1] - width) else MIDDLE)
			) )
			ims.append(im[y:y+height,x:x+width,:])
			x += x_stride
		y += y_stride

	return locations, ims


def stich_together(locations, subwindows, size, dtype=np.uint8):
	output = np.zeros(size, dtype=dtype)
	for location, subwindow in zip(locations, subwindows):
		outer_bounding_box, inner_bounding_box, y_type, x_type = location
		y_paste, x_paste, y_cut, x_cut, height_paste, width_paste = -1, -1, -1, -1, -1, -1
		#print outer_bounding_box, inner_bounding_box, y_type, x_type

		if y_type == TOP_EDGE:
			y_cut = 0
			y_paste = 0
			height_paste = TILE_SIZE - PADDING_SIZE
		elif y_type == MIDDLE:
			y_cut = PADDING_SIZE
			y_paste = inner_bounding_box[0]
			height_paste = TILE_SIZE - 2 * PADDING_SIZE
		elif y_type == BOTTOM_EDGE:
			y_cut = PADDING_SIZE
			y_paste = inner_bounding_box[0]
			height_paste = TILE_SIZE - PADDING_SIZE

		if x_type == LEFT_EDGE:
			x_cut = 0
			x_paste = 0
			width_paste = TILE_SIZE - PADDING_SIZE
		elif x_type == MIDDLE:
			x_cut = PADDING_SIZE
			x_paste = inner_bounding_box[1]
			width_paste = TILE_SIZE - 2 * PADDING_SIZE
		elif x_type == RIGHT_EDGE:
			x_cut = PADDING_SIZE
			x_paste = inner_bounding_box[1]
			width_paste = TILE_SIZE - PADDING_SIZE

		#print (y_paste, x_paste), (height_paste, width_paste), (y_cut, x_cut)

		output[y_paste:y_paste+height_paste, x_paste:x_paste+width_paste] = subwindow[y_cut:y_cut+height_paste, x_cut:x_cut+width_paste]

	return output

def linePreprocess(pred, orig, ccRes, simple=True):
	rhoRes = 1.0
	thetaRes = math.pi/180
	threshold = 200
	minLineLength = 30
	maxLenGap = 200

	if pred.sum() == 0:
		return pred
	newPred, boundaries = removeSidePredictions(pred,orig, ccRes)
	if boundaries is None: #meaning there were no baselines predicted
		return pred
	if USE_COL_DETECTION:
		splits = [get_split_col(orig,pred)]
	else:
		splits = None
	if not simple:
		if orig.shape[2]>1:
			gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
		else:
			gray=orig
		splits = get_vert_lines(gray)
		for split in splits:
			splitBaselines(pred,ccRes,split)
	else:
		connectLines(newPred,boundaries,splits,ccRes[1], rhoRes, thetaRes, threshold, minLineLength, maxLenGap)
	return pred


def get_split_col(im,pred):
	proj = np.mean(im, axis=0, dtype=np.float32)
	proj = np.squeeze(cv2.bilateralFilter(proj[np.newaxis,:], 9, 12, 12))

	l = proj.shape[0]

	trunc_proj = proj[l / 4:-l / 4]
	l2 = trunc_proj.shape[0]

	max_idx = np.argmax(trunc_proj)
	min_idx = np.argmin(trunc_proj)
	_min = np.min(trunc_proj)
	_max = np.max(trunc_proj)

	max_dist = abs(max_idx - (l2 / 2)) / float(l2)
	min_dist = abs(min_idx - (l2 / 2)) / float(l2)

	copy = np.copy(trunc_proj)
	copy[max(0, min_idx-50): min(trunc_proj.shape[0] - 1, min_idx + 50)] = _max
	next_min = np.min(copy)

	inner_max = np.max(trunc_proj[int(.4 * l2): int(.6 * l2)])

	ret = None
	if max_dist < 0.1:
			ret = max_idx + l / 4
	elif min_dist < 0.1 and (next_min - _min) > 10:
			ret = min_idx + l / 4
	elif max_dist < 0.2 and (_max - inner_max) < 10:
			ret = max_idx + l / 4

	#double check we aren't splicing a lot of predictions
	if ret is not None:
		colSum=0
		for y in range(1,pred.shape[0]):
			if pred[y-1,ret]==0 and pred[y,ret]>0:
				colSum+=1
		if colSum>6:
			ret=None

	return ret

def get_vert_lines(im):

	edges = cv2.Sobel(im, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)
	edges[edges < 0] = 0
	edges = (255 * (edges / np.max(edges))).astype(np.uint8)
	edges[edges < 50] = 0
	edges[edges != 0] = 255

	structure = np.ones((11,1))
	edges = nd.binary_closing(edges, structure=structure)
	structure = np.ones((41,1))
	edges = nd.binary_opening(edges, structure=structure)

	edges = (255 * edges).astype(np.uint8)

	proj = np.mean(edges, axis=0, dtype=np.float32)
	proj = np.squeeze(cv2.bilateralFilter(proj[np.newaxis,:], 9, 20, 20))

	vert_lines = list()

	while True:
		idx = np.argmax(proj)
		if proj[idx] < 15:
			break
		vert_lines.append((idx, proj[idx]))
		proj[max(0, idx - 50):min(idx + 50, proj.shape[0])] = 0

	out = list()
	for idx, val in vert_lines:
		if len(vert_lines) > 3 or val > 25:
			out.append(idx)

	return out

def cropBlack(img,gt):
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	median = np.median(gray)
	thresh = median*0.8

	cutTop=0
	while np.median(gray[cutTop,:]) < thresh:
		cutTop+=1
	
	cutBot=-1
	while np.median(gray[cutBot,:]) < thresh:
		cutBot-=1

	cutLeft=0
	while np.median(gray[:,cutLeft]) < thresh:
		cutLeft+=1
	
	cutRight=-1
	while np.median(gray[:,cutRight]) < thresh:
		cutRight-=1

	return cutTop, -1*(cutBot+1), cutLeft, -1*(cutRight+1)


def removeCC(ccId, ccs, stats, removeFrom):
	for y in range(stats[ccId,cv2.CC_STAT_TOP],stats[ccId,cv2.CC_STAT_HEIGHT]+stats[ccId,cv2.CC_STAT_TOP]):
		for x in range(stats[ccId,cv2.CC_STAT_LEFT],stats[ccId,cv2.CC_STAT_WIDTH]+stats[ccId,cv2.CC_STAT_LEFT]):
			if ccs[y,x]==ccId:
				removeFrom[y,x]=0


def getLen(line):
	return math.sqrt( (line['x1']-line['x2'])**2 + (line['y1']-line['y2'])**2 )

def convertToLineSegments(pred, ccRes):
	ret=[]
	numLabels, labels, stats, cent = ccRes #cv2.connectedComponentsWithStats(pred, 4, cv2.CV_32S)
	for l in range(1,numLabels):
		if stats[l,cv2.CC_STAT_WIDTH]>30:
			topLeft=-1
			topRight=-1
			for y in range(stats[l,cv2.CC_STAT_TOP],stats[l,cv2.CC_STAT_HEIGHT]+stats[l,cv2.CC_STAT_TOP]):
				if topLeft == -1 and labels[y,stats[l,cv2.CC_STAT_LEFT]]==l:
					topLeft=y
					if topRight != -1:
						break
				if topRight == -1 and labels[y,stats[l,cv2.CC_STAT_LEFT]+stats[l,cv2.CC_STAT_WIDTH]-1]==l:
					topRight=y
					if topLeft != -1:
						break

			botLeft=-1
			botRight=-1
			for y in range(stats[l,cv2.CC_STAT_HEIGHT]+stats[l,cv2.CC_STAT_TOP]-1,stats[l,cv2.CC_STAT_TOP],-1):
				if botLeft == -1 and labels[y,stats[l,cv2.CC_STAT_LEFT]]==l:
					botLeft=y
					if botRight != -1:
						break
				if botRight == -1 and labels[y,stats[l,cv2.CC_STAT_LEFT]+stats[l,cv2.CC_STAT_WIDTH]-1]==l:
					botRight=y
					if botLeft != -1:
						break
			ret.append({'x1':  stats[l,cv2.CC_STAT_LEFT],
						'y1':  (topLeft+botLeft)/2,
						'x2':  stats[l,cv2.CC_STAT_WIDTH]+stats[l,cv2.CC_STAT_LEFT]-1,
						'y2':  (topRight+botRight)/2,
						'cc':  l
						})
		else:
			removeCC(l,labels,stats,pred)

	return ret, labels, stats



#assumes binary prediction
def removeSidePredictions(pred,orig,ccRes):
	cropTop, cropBot, cropLeft, cropRight = cropBlack(orig,pred)

	#clear pred on black areas
	if cropTop>0:
		pred[:cropTop,:]=0
	if cropBot>0:
		pred[-cropBot:,:]=0
	if cropLeft>0:
		pred[:,:cropLeft]=0
	if cropRight>0:
		pred[:,-cropRight:]=0


	lines, ccs, ccStats = convertToLineSegments(pred, ccRes)

	if len(lines) == 0:
		return pred, None

	meanLen=0
	for line in lines:
		meanLen += getLen(line)
	meanLen/=len(lines)

	lineIm = np.zeros(pred.shape)
	for line in lines:
		if line is not None:
			cv2.line(lineIm, (line['x1'],line['y1']), (line['x2'],line['y2']), 1, 1)
	hist = np.sum(lineIm, axis=0)
	if cropLeft<4 or cropRight<4: #we can skip if we found black on both ends

		#vert hist of lines

		#construct linear filter based on mean line length
		kValues = [0.0]*int(meanLen*0.75)
		lenh=int(meanLen*0.75)/2
		for i in range(lenh):
			kValues[i] = -1.0*(lenh-i)
			kValues[-i] = (lenh-i)
		kernelLeftEdge = np.array(kValues)/lenh
		leftEdges = cv2.filter2D(hist,-1,kernelLeftEdge,None, (-1,-1), 0, cv2.BORDER_REPLICATE)

		maxV = np.amax(leftEdges)
		minV = np.amin(leftEdges)

		threshLeft = minV+(maxV-minV)*0.5
		threshRight = minV+(maxV-minV)*0.5

		leftPeaks = []
		hitLeft=False
		leftV=0
		rightPeaks = []
		hitRight=True
		rightV=-9999999
		for x in range(1,leftEdges.shape[0]-1):
			if leftEdges[x]>threshLeft and leftEdges[x]>leftEdges[x-1] and leftEdges[x]>leftEdges[x+1]:
				if hitRight:
					hitRight=False
					rightV=0
				if hitLeft:
					if leftEdges[x]>leftV:
						leftV=leftEdges[x]
						leftPeaks[-1]=x
				else:
					leftPeaks.append(x)
					hitLeft=True
					leftV=leftEdges[x]
			if leftEdges[x]<threshRight and leftEdges[x]<leftEdges[x-1] and leftEdges[x]<leftEdges[x+1]:
				if hitLeft:
					hitLeft=False
					leftV=0
				if hitRight:
					if leftEdges[x]<rightV:
						rightV=leftEdges[x]
						rightPeaks[-1]=x
				else:
					rightPeaks.append(x)
					hitRight=True
					rightV=leftEdges[x]

		#prune peaks, assuming max left mataches min right and so on
		newLeftPeaks=[]
		newRightPeaks=[]
		while len(leftPeaks)>0 and len(rightPeaks)>0:
			maxLeft=leftPeaks[0]
			maxLeftV=leftEdges[maxLeft]
			for l in leftPeaks[1:]:
				if leftEdges[l] > maxLeftV:
					maxLeft=l
					maxLeftV=leftEdges[maxLeft]

			i=0
			while i < len(rightPeaks) and rightPeaks[i]<maxLeft:
				i+=1
			if i == len(rightPeaks):
				#then maxLeft has no matching peak
				newLeftPeaks.append(maxLeft)
				leftPeaks.remove(maxLeft)
				continue
			minRight=rightPeaks[i]
			minRightV=leftEdges[minRight]
			for r in rightPeaks[i:]:
				if leftEdges[r] < minRightV:
					minRight=r
					minRightV=leftEdges[minRight]

			if maxLeft>=minRight:
				print 'Error in peak pruning: '+predFile
				break

			newLeftPeaks.append(maxLeft)
			newRightPeaks.append(minRight)
			i=0
			while i < len(leftPeaks):
				if leftPeaks[i]>=maxLeft and leftPeaks[i]<=minRight:
					del leftPeaks[i]
				else:
					i+=1
			i=0
			while i < len(rightPeaks):
				if rightPeaks[i]>=maxLeft and rightPeaks[i]<=minRight:
					del rightPeaks[i]
				else:
					i+=1

		#pickup spare right peak
		if len(rightPeaks)>0:
			minRight=rightPeaks[0]
			minRightV=leftEdges[minRight]
			for r in rightPeaks[0:]:
				if leftEdges[r] < minRightV:
					minRight=r
					minRightV=leftEdges[minRight]
			newRightPeaks.append(minRight)
			keepRight = rightPeaks[-1]
		else:
			keepRight = pred.shape[1]-1

		if len(leftPeaks)>0:
			minLeft=leftPeaks[0]
			minLeftV=leftEdges[minLeft]
			for r in leftPeaks[0:]:
				if leftEdges[r] < minLeftV:
					minLeft=r
					minLeftV=leftEdges[minLeft]
			newLeftPeaks.append(minLeft)
			keepLeft=leftPeaks[0]
		else:
			keepLeft=0

		leftPeaks=sorted(newLeftPeaks)
		rightPeaks=sorted(newRightPeaks)

		#check if up agains edge
		if cropLeft<4:  #Left side
			prune=-1
			if len(rightPeaks)>1:
				if rightPeaks[0] < leftPeaks[0]:
					if rightPeaks[0] < rightPeaks[1]-leftPeaks[0]:
						prune= rightPeaks[0]
						keepLeft = leftPeaks[0]
				else:
					if leftPeaks[0]<meanLen*0.4 and rightPeaks[0]-leftPeaks[0] < rightPeaks[1]-leftPeaks[1]:
						prune= rightPeaks[0]
						keepLeft=leftPeaks[1]

			for i in range(len(lines)):
				line=lines[i]
				if (line['x1']<=meanLen/5 and getLen(line)<meanLen*0.75 and line['x2']<keepLeft) or (prune!=-1 and prune-line['x1']>line['x2']-prune):
					removeCC(line['cc'],ccs,ccStats,pred)
					lines[i]=None

		if cropRight<4: #Right side
			width = orig.shape[1]
			prune=-1
			if len(leftPeaks)>1:
				print leftPeaks
				print rightPeaks
				if rightPeaks[-1] < leftPeaks[-1]:
					if width-leftPeaks[-1] < rightPeaks[-1]-leftPeaks[-2]:
						prune= leftPeaks[-1]
						keepRight = rightPeaks[-1]
				else:
					if rightPeaks[-1]-leftPeaks[-1] < rightPeaks[-2]-leftPeaks[-2]:
						prune= leftPeaks[-1]
						keepRight = rightPeaks[-2]

			for i in range(len(lines)):
				line=lines[i]
				
				if line is not None and ((line['x2']>=pred.shape[1]-(1+meanLen/5) and getLen(line)<meanLen*0.75 and line['x1']>keepRight) or (prune!=-1 and prune-line['x1']<line['x2']-prune)):
					removeCC(line['cc'],ccs,ccStats,pred)
					lines[i]=None


	trans01=[]
	trans10=[]
	for x in range(1,hist.shape[0]):
		if hist[x-1]<=3 and hist[x]>3:
			trans01.append(x)
		if hist[x-1]>3 and hist[x]<=3:
			trans10.append(x-1)

	boundariesRet=[]
	if len(trans01)!=0 and len(trans10)!=0:
		leftBs = trans01
		rightBs = trans10
		lastLeft=leftBs[0]
		lastRight=rightBs[0]
		leftI=0
		rightI=0
		while leftI<len(leftBs) and rightI<len(rightBs):
			while rightI<len(rightBs) and rightBs[rightI]<leftBs[leftI]:
				rightI+=1
			lastLeft=leftBs[leftI]
			rightB_ = pred.shape[1]
			if rightI<len(rightBs):
				rightB_=rightBs[rightI]
			while leftI<len(leftBs) and leftBs[leftI]<rightB_:
				leftI+=1
			while rightI<len(rightBs) and (leftI>=len(leftBs) or rightBs[rightI]<leftBs[leftI]):
				rightI+=1

			boundariesRet.append((lastLeft,rightBs[rightI-1]))

	return pred, boundariesRet


def getClusterLine(cluster,bb):
	leftY=0
	leftCount=0
	rightY=0
	rightCount=0
	for line in cluster:
		if abs(line[0]-bb[0])<5:
			leftY+=line[1]
			leftCount+=1
		if abs(line[2]-bb[2])<5:
			rightY+=line[3]
			rightCount+=1

	return (bb[0], leftY/leftCount, bb[2], rightY/rightCount)


def clusterPrune(lines,pred, ccLabels):

	ccMap={}
	cluster={}
	for line in lines:
		y=line[1]
		step=1
		if line[2]<line[0]:
			step=-1
		if line[2] == line[0]:
			continue
		slope = float(line[3]-line[1])/float(line[2]-line[0])
		i=0
		ccFirst=None

		for x in range(line[0],line[2],step):
			y = int(line[1] + i*slope)
			if y>max(line[1],line[3]) or y<min(line[1],line[3]):
				print (x,y,line,slope)
				assert False
			cc = ccLabels[y,x]
			if cc==0 or pred[y,x]==0:
				continue
			while cc in ccMap and ccMap[cc] is not None:
				cc=ccMap[cc]
			if ccFirst is None:
				ccFirst=cc
				if cc not in cluster:
					cluster[cc]=[]
			elif ccFirst != cc:
				ccMap[cc]=ccFirst
				if cc in cluster:
					cluster[ccFirst] += cluster[cc]
					cluster[cc]=None
			cluster[ccFirst].append(line)


			i+=1

	ret = []
	for cc,cLines in cluster.items():
		if cLines is not None:
			maxDist=0
			maxLine=None
			for line in cLines:
				dist = math.sqrt( ((line[0]-line[2])**2) + ((line[1]-line[3])**2) )
				if dist>maxDist:
					maxDist=dist
					maxLine=line
			ret.append(maxLine)
	return ret


def lineEq(line):
	m = float(line[3]-line[1])/float(line[2]-line[0])
	b = line[1]-m*line[0]
	return m,b


def goodIntersection(line1, line2):
	m1, b1 = lineEq(line1)
	m2, b2 = lineEq(line2)
	if m1==m2:
		return False
	xIntersection = (b1-b2)/(m2-m1)
	return xIntersection>min(line1[0],line2[0]) and xIntersection<max(line1[0],line2[0]) and \
			xIntersection>min(line1[2],line2[2]) and xIntersection<max(line1[2],line2[2])
		   

def connectLines(pred,boundaries,splits, ccRes, rhoRes, thetaRes, threshold, minLineLength, maxLenGap):
	img=pred.copy()
	element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
	predEroded = cv2.erode(pred,element)
	lines = cv2.HoughLinesP(predEroded, rhoRes, thetaRes, threshold, None, minLineLength, maxLenGap)


	angles=[]
	angleMean=0
	if lines is None:
	   return
	for line in lines:
		x1=line[0,0]
		y1=line[0,1]
		x2=line[0,2]
		y2=line[0,3]
		angle = math.atan2(y2-y1,x2-x1)
		angleMean += angle
		angles.append(angle)

	angleMean /= len(angles)
	angleStd=0
	for angle in angles:
		angleStd += (angleMean-angle)**2
	angleStd = math.sqrt(angleStd/len(angles))


	prunedLines=[]

	#potting.idprune by angle
	if angleStd!=0:
		for line in lines:
			x1=line[0,0]
			y1=line[0,1]
			x2=line[0,2]
			y2=line[0,3]
			angle = math.atan2(y2-y1,x2-x1)
			if abs((angle-angleMean)/angleStd)<2.5:
				prunedLines.append((x1,y1,x2,y2))

	prunedLines2 = clusterPrune(prunedLines, pred, ccRes)

	prunedLines3 = prunedLines2[:]
	for i in range(len(prunedLines2)):
		for j in range(i,len(prunedLines2)):
			if goodIntersection(prunedLines2[i],prunedLines2[j]):
				if prunedLines2[i][0]>prunedLines2[j][2]:
					prunedLines3.append( (prunedLines2[i][0],prunedLines2[i][1],prunedLines2[j][2],prunedLines2[j][3]) )
				elif prunedLines2[j][0]>prunedLines2[i][2]:
					prunedLines3.append( (prunedLines2[j][0],prunedLines2[j][1],prunedLines2[i][2],prunedLines2[i][3]) )
	#prune by boundaries
	#print boundaries
	for i in range(1,len(boundaries)):
		dontCross = (boundaries[i-1][1]+boundaries[i][0])/2
		for l in range(len(prunedLines3)):
			line = prunedLines3[l]
			if line is not None and min(line[0],line[2])<dontCross and max(line[0],line[2])>dontCross:
				prunedLines3[l]=None
	if splits is not None:
		for dontCross in splits:
			for l in range(len(prunedLines3)):
				line = prunedLines3[l]
				if line is not None and min(line[0],line[2])<dontCross and max(line[0],line[2])>dontCross:
					prunedLines3[l]=None

	for line in prunedLines3:
		if line is not None:
			x1,y1,x2,y2 = line
			cv2.line(pred, (x1,y1), (x2,y2), 255, 7)


def splitBaselines(pred,ccRes,split):
	CUT_THRESH=60
	if split<CUT_THRESH-2 or pred.shape[1]-split<CUT_THRESH-2:
		return
	okCCs=[]
	dontCCs=[]
	ccs=ccRes[1]
	ccStats=ccRes[2]
	for y in range(pred.shape[0]):
		if pred[y,split]>0:
			cc = ccs[y,split]
			if cc in okCCs:
				pred[y,split-1:split+2]=0
			elif cc not in dontCCs:
				left = split-ccStats[cc][cv2.CC_STAT_LEFT]
				right = (ccStats[cc][cv2.CC_STAT_LEFT]+ccStats[cc][cv2.CC_STAT_WIDTH]-1)-split
				if left>CUT_THRESH and right>CUT_THRESH:
					okCCs.append(cc)
					pred[y,split-1:split+2]=0
				else:
					dontCCs.append(cc)


def apply_post_processing(binary, im, simple):
	ccRes = cv2.connectedComponentsWithStats(binary, 4, cv2.CV_32S)
	finalPred = linePreprocess(binary, im, ccRes, simple)
	return finalPred

def pred_to_pts(pred, simple):
	global_threshold = 127
	slice_size = 25
	if simple:
		small_threshold = 100
	else:
		small_threshold = 50

	connectivity = 4
	ret, binary = cv2.threshold(pred,global_threshold,255,cv2.THRESH_BINARY)
	output= cv2.connectedComponentsWithStats(binary, connectivity, cv2.CV_32S)
	print output[0]
	baselines = []
	#skip background
	for label_id in xrange(1, output[0]):
		min_x = output[2][label_id][0]
		min_y = output[2][label_id][1]
		max_x = output[2][label_id][2] + min_x
		max_y = output[2][label_id][3] + min_y
		cnt = output[2][label_id][4]

		if cnt < small_threshold:
			continue

		baseline = output[1][min_y:max_y, min_x:max_x]

		pts = []
		x_all, y_all = np.where(baseline == label_id)
		first_idx = y_all.argmin()
		first = (y_all[first_idx]+min_x, x_all[first_idx]+min_y)

		pts.append(first)
		for i in xrange(0, baseline.shape[1], slice_size):
			next_i = i+slice_size
			baseline_slice = baseline[:, i:next_i]

			x, y = np.where(baseline_slice == label_id)
			x_avg = x.mean()
			y_avg = y.mean()
			pts.append((int(y_avg+i+min_x), int(x_avg+min_y)))

		last_idx = y_all.argmax()
		last = (y_all[last_idx]+min_x, x_all[last_idx]+min_y)
		pts.append(last)

		if len(pts) <= 1:
			continue

		baselines.append(pts)

	return baselines

def write_baseline_pts(baselines, filename, scale=4):
	with open(filename, 'w') as f:
		for baseline in baselines:
			baseline_txt = []
			for pt in baseline:
				pt_txt = "{},{}".format(pt[0] * scale, pt[1] * scale)
				baseline_txt.append(pt_txt)
			f.write(";".join(baseline_txt)+"\n")


def write_results(binary, out_txt, simple):
	baselines = pred_to_pts(binary, simple)
	print len(baselines)
	write_baseline_pts(baselines, out_txt)
	

def main(in_image, out_txt, simple):
	print "Loading Image"
	im = cv2.imread(in_image, cv2.IMREAD_COLOR)

	print "Resizing Image"
	im = cv2.resize(im, (im.shape[1] / 4, im.shape[0] / 4) )

	print "Preprocessing"
	data = 0.003921568 * (im - 127.)

	print "Loading network"
	network = setup_network()

	print "Tiling input"
	locations, subwindows = get_subwindows(data)
	print "Number of tiles: %d" % len(subwindows)

	print "Starting Predictions"
	raw_subwindows = predict(network, subwindows)

	print "Reconstructing whole image from tiles"
	result = (255 * stich_together(locations, raw_subwindows, tuple(im.shape[0:2]), np.float32)).astype(np.uint8)

	if DEBUG:
		out_file = out_txt[:-4] + ".png"
		cv2.imwrite(out_file, result)

	print "Applying Post Processing"
	post_processed = apply_post_processing(result, im, simple)

	if DEBUG:
		out_file = out_txt[:-4] + "_post.png"
		cv2.imwrite(out_file, post_processed)

	print "Writing Final Result"
	write_results(post_processed, out_txt, simple)

	print "Done"
	print "Exiting"


if __name__ == "__main__":
	if len(sys.argv) < 3:
		print "USAGE: python detect_baselines.py in_image out_txt [simple|complex] [gpu#] [weights]"
		print "\tin_image is the input image to be labeled"
		print "\tout_txt is the resulting baseline file"
		print "\tgpu is an integer device ID to run networks on the specified GPU.  If omitted, CPU mode is used"
		exit(1)
	in_image = sys.argv[1]
	out_txt = sys.argv[2]

	if not os.path.exists(in_image):
		raise Exception("in_image %s does not exist" % in_image)

	try:
		simple = sys.argv[3] == 'simple'
		if not simple:
			WEIGHTS_FILE = os.path.join(os.path.dirname(__file__), "complex_weights.caffemodel")
	except:
		simple = True

	# use gpu if specified
	try:
		gpu = int(sys.argv[4])
		if gpu >= 0:
			caffe.set_mode_gpu()
			caffe.set_device(gpu)
	except:
		caffe.set_mode_cpu()

	try:
		WEIGHTS_FILE = sys.argv[5]
	except:
		pass

	main(in_image, out_txt, simple)

