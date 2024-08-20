import random
import sys
import pygame
from pygame.locals import *
import math
import timeit
import ast
import time

pygame.init()
width, height = 1024, 1024
screen = pygame.display.set_mode((width, height))

def normalize(vector2D):
    xsquared = vector2D[0]*vector2D[0]
    ysquared = vector2D[1]*vector2D[1]
    magnitude = math.sqrt(xsquared+ysquared)
    return [vector2D[0]/magnitude, vector2D[1]/magnitude]

rows, cols = 256, 256
totalVoxelsPossible = rows*cols
levelsOfDepth = int(math.log2(rows))
maxPossibleNodes = (4**(levelsOfDepth+1) - 1)
rayOrigin = [1024, 200]
rayDirection = normalize([-1, 1])
wholeBoxStack = []


with open('Quadtree.txt', 'r') as file:
    # Read the entire file content
    content = file.read()
    # Use ast.literal_eval to convert the string back to a Python object
    new_artificialMemory = ast.literal_eval(content)
    artificalMemory = new_artificialMemory


def calculateXPlane(planeX, rayDirection, rayOrigin):
    rayDir = rayDirection
    if rayDir[0] == 0:
        rayDir[0] = 0.0001
    #y=mx+b
    coefficient1 = 1 / rayDir[0]
    coefficient2 = (-1*rayOrigin[0])/rayDir[0]
    tx = coefficient1*planeX + coefficient2
    return tx


def calculateYPlane(planeY, rayDirection, rayOrigin):
    rayDir = rayDirection
    if rayDir[1] == 0:
        rayDir[1] = 0.0001

    # y=mx+b
    coefficient1 = 1 / rayDir[1]
    coefficient2 = (-1 * rayOrigin[1]) / rayDir[1]
    ty = coefficient1 * planeY + coefficient2
    return ty


def calculateQuadrantBasedOfEdgePos(rayEdgePos):
    top = True
    left = True
    tolerance = 1e-9
    if rayEdgePos[0] > 0.5:
        left = False
    if rayEdgePos[1] > 0.5:
        top = False
    if top == True and left == True:
        return [1, 0, 0, 0]
    if top == True and left == False:
        return [0, 1, 0, 0]
    if top == False and left == True:
        return [0, 0, 1, 0]
    if top == False and left == False:
        return[0, 0, 0, 1]


def render_quadtree(address, x, y, pixelsWidth, lod, steps = 0):
    if steps > lod:
        return
    steps = steps+1
    thisLevelBoxWidth = pixelsWidth/2
    node = artificalMemory[address]
    #print(node)
    if node != 0:
        validMask = node[1][:4]
        leafMask =  node[1][4:]
        firstChildAddress = binary_to_decimal(node[0])+address
        childCount = 0

        for i in range(0, 4):
            quadrant = [0, 0]
            if i == 0:
                quadrant = [0, 0]
            if i == 1:
                quadrant = [1, 0]
            if i == 2:
                quadrant = [0, 1]
            if i == 3:
                quadrant = [1, 1]

            pos = [x+(quadrant[0]*thisLevelBoxWidth), y+(quadrant[1]*thisLevelBoxWidth)]
            if validMask[i] != 0:
                thisAddress = firstChildAddress + childCount
                childCount = childCount + 1
                pygame.draw.rect(screen, (10, 10, 10), Rect(pos[0], pos[1], thisLevelBoxWidth, thisLevelBoxWidth), 1)
                if leafMask[i] == 0:
                    #print(node)
                    render_quadtree(thisAddress, pos[0], pos[1], thisLevelBoxWidth, lod, steps)
                if leafMask[i] != 0:
                    pygame.draw.rect(screen, (0, 255, 0), Rect(pos[0], pos[1], thisLevelBoxWidth, thisLevelBoxWidth), 1)

def render_quadtree_box(address, x, y, pixelsWidth, lod, steps = 0):
    if steps > lod:
        return
    steps = steps+1
    thisLevelBoxWidth = pixelsWidth/2
    node = artificalMemory[address]
    #print(node)
    if node != 0:
        validMask = node[1][:4]
        leafMask =  node[1][4:]
        firstChildAddress = binary_to_decimal(node[0])+address
        childCount = 0

        for i in range(0, 4):
            quadrant = [0, 0]
            if i == 0:
                quadrant = [0, 0]
            if i == 1:
                quadrant = [1, 0]
            if i == 2:
                quadrant = [0, 1]
            if i == 3:
                quadrant = [1, 1]

            pos = [x+(quadrant[0]*thisLevelBoxWidth), y+(quadrant[1]*thisLevelBoxWidth)]
            if validMask[i] != 0:
                thisAddress = firstChildAddress + childCount
                childCount = childCount + 1
                pygame.draw.rect(screen, (10, 10, 10), Rect(pos[0], pos[1], thisLevelBoxWidth, thisLevelBoxWidth), 1)
                if leafMask[i] == 0:
                    #print(node)
                    render_quadtree_box(thisAddress, pos[0], pos[1], thisLevelBoxWidth, lod, steps)
                if leafMask[i] != 0:
                    pygame.draw.rect(screen, (0, 255, 0), Rect(pos[0], pos[1], thisLevelBoxWidth, thisLevelBoxWidth), 1)
                    wholeBoxStack.append([pos[0], pos[1], thisLevelBoxWidth])

def binary_to_decimal(binary):
    binary = ''.join(str(bit) for bit in binary)
    decimal = int(binary, 2)
    return decimal


def childrenBefore(node, checkQuad):
    childrensBefore = 0
    childCount = 0
    validMask = node[1][:4]
    for i in range(0, 4):
        if checkQuad[i] != 0:
            childrensBefore = childCount
        if validMask[i] != 0:
            childCount = childCount+1
    return childrensBefore


def rayBoxQuadrant(rayStuff, boxStack):
    boxStuff = boxStack[-1]
    rayDir = rayStuff[0]
    rayOri = rayStuff[1]
    size = boxStuff[2]
    x1 = boxStuff[0]
    x2 = boxStuff[0] + size
    y1 = boxStuff[1]
    y2 = boxStuff[1]+size
    intersectData = calculateBoxIntersect(x1, x2, y1, y2, rayStuff[0], rayStuff[1])
    xIntersectPoint = rayOri[0] + rayDir[0] * intersectData[0]
    yIntersectPoint = rayOri[1] + rayDir[1] * intersectData[0]
    #pygame.draw.rect(screen, (0, 255, 0), Rect(xIntersectPoint-5, yIntersectPoint-5, 10, 10))
    xIntersectNorm = (xIntersectPoint-x1)/size
    yIntersectNorm = (yIntersectPoint-y1)/size
    quadrant = calculateQuadrantBasedOfEdgePos([xIntersectNorm, yIntersectNorm])
    return quadrant


#OKAY. We're going to do this all again, using alot of the old functions, but completely redoing push, pop, advance, and the whole ray casting loop. Wish me luck! Btw the end of the rewritten section will be "The End!" (you can tell I'm trying now lol)

#  Describes the beginning, end, and middle data of each stack.
#  @ = root
#  # = normal
#  ! = checking for possible node
#  below examples are for 4 total boxes.

#boxStack: stores all boxes, as well as the current quadrant we are checking but not pushed into. @##!, len=4
#quadStack: stores all quadrants, as well as the current quadrant we are checking but not pushed into, and does not store the quadrant for the root node since it has no parent. ##!, len=3
#nodeStack: stores the addresses in memory for all of the nodes. @##, len=3
def newPush(boxStack, nodeStack, quadStack, rayStuff):
    #Adds one item onto every list. Pushes into the checkQuadrant if there are any voxels inside it (validMask == 1 & checkQuadrant == 1).
    checkQuadrant = quadStack[-1]
    lastNodeAddress = nodeStack[-1]
    lastNode = artificalMemory[lastNodeAddress]
    lastNodeValid = lastNode[1][:4]
    lastNodeLeaf = lastNode[1][4:]
    for i in range(0, 4):
        if checkQuadrant[i] == 1 and lastNodeValid[i] == 1:
            if lastNodeLeaf[i] == 1:
                return [boxStack, nodeStack, quadStack, True, False]
            lastNodeChildPointer = binary_to_decimal(lastNode[0])
            newNodeAddress = childrenBefore(lastNode, checkQuadrant) + lastNodeChildPointer + lastNodeAddress
            nodeStack.append(newNodeAddress)
            newQuadrant = rayBoxQuadrant(rayStuff, boxStack)
            quadStack.append(newQuadrant)
            newBox = newBoxStuff(boxStack[-1], quadStack[-1])
            boxStack.append(newBox)
            return [boxStack, nodeStack, quadStack, True, True]
    return [boxStack, nodeStack, quadStack, False, True]

def newAdvance(boxStack, nodeStack, quadStack, rayStuff):
    rayDir = rayStuff[0]
    rayOri = rayStuff[1]
    tempBox = boxStack[-1]
    tempQuad = quadStack[-1]
    parentBox = boxStack[-2]
    parentNodeAddress = nodeStack[-1]
    #print(parentNodeAddress)
    parentNode = artificalMemory[parentNodeAddress]
    parentValidMask = parentNode[1][:4]
    parentLeafMask = parentNode[1][4:]
    leave = False
    found = False
    for i in range(0, 3):
        intersectDat=calculateInvertedBoxIntersect(tempBox[0], tempBox[0]+tempBox[2], tempBox[1], tempBox[1]+tempBox[2], rayDir, rayOri)
        tmax = intersectDat[0]
        xIntersect = intersectDat[1]

        moveDat = moveQuadrant(tempQuad, xIntersect)
        tempQuad = moveDat
        tempBox = newBoxStuff(parentBox, tempQuad)
        xIntersectPoint = rayOri[0] + rayDir[0] * tmax
        yIntersectPoint = rayOri[1] + rayDir[1] * tmax

        dx1 = abs(parentBox[0] - xIntersectPoint)
        dx2 = abs(parentBox[0]+parentBox[2] - xIntersectPoint)
        dy1 = abs(parentBox[1] - yIntersectPoint)
        dy2 = abs(parentBox[1]+parentBox[2] - yIntersectPoint)

        if dx1 > 0.0001 and dx2 < 0.0001 or dx1 < 0.0001 and dx2 > 0.0001 or dy1 > 0.0001 and dy2 < 0.0001 or dy1 < 0.0001 and dy2 > 0.0001:
            leave = True

        boxStack.pop()
        boxStack.append(tempBox)
        quadStack.pop()
        quadStack.append(tempQuad)
        if leave == False and found == False:
            for j in range(0, 4):
                if parentValidMask[j] == 1 and tempQuad[j] == 1:
                    if parentLeafMask[j] == 1:
                        return [boxStack, nodeStack, quadStack, leave, False]
                    return [boxStack, nodeStack, quadStack, leave, True]
    return [boxStack, nodeStack, quadStack, leave, True]

def actualCastRay(rayStuff, box, uv):
    start = time.time()
    boxStack = []
    nodeStack = []
    quadStack = []

    checkQuad = calculateQuadrantBasedOfEdgePos(uv)
    boxStack.append(box)
    quadCorrespondingBox = newBoxStuff(boxStack[-1], checkQuad)
    boxStack.append(quadCorrespondingBox)
    nodeStack.append(0)
    quadStack.append(checkQuad)
    keepGoing = True

    for steps in range(0, 100):
        #pygame.draw.rect(screen, (0, 200, 200), Rect(boxStack[-1][0], boxStack[-1][1], boxStack[-1][2], boxStack[-1][2]), 1)
        if keepGoing == False:
            break
        pushDat = newPush(boxStack, nodeStack, quadStack, rayStuff)
        keepGoing = pushDat[4]
        pushed = pushDat[3]

        if pushed == False:
            advanceDat = newAdvance(boxStack, nodeStack, quadStack, rayStuff)
            while advanceDat[3] == True:
                if True == True:
                    if len(quadStack) < 2:
                        return 10000
                    boxStack.pop()
                    nodeStack.pop()
                    quadStack.pop()
                    advanceDat = newAdvance(boxStack, nodeStack, quadStack, rayStuff)
                    keepGoing = advanceDat[4]

    lastBox = boxStack[-1]
    foundBoxDat = calculateBoxIntersect(lastBox[0], lastBox[0]+lastBox[2], lastBox[1], lastBox[1]+lastBox[2], rayStuff[0], rayStuff[1])
    t = foundBoxDat[0]
    return t

#The End!
def rotatePoint(rotationMatrix, point, rotateAround):
    #pygame.draw.rect(screen, (0, 255, 0), Rect(rotateAround[0], rotateAround[1], 5, 5), 1)
    newPoint = [point[0]-rotateAround[0], point[1]-rotateAround[1]]
    #pygame.draw.rect(screen, (255, 0, 255), Rect(newPoint[0]+rotateAround[0], newPoint[1]+rotateAround[1], 5, 5), 1)
    rotatedPoint = [rotationMatrix[0] * newPoint[0] + rotationMatrix[1] * newPoint[1], rotationMatrix[2] * newPoint[0] + rotationMatrix[3] * newPoint[1]]
    rotatedPoint = [rotatedPoint[0]+rotateAround[0], rotatedPoint[1]+rotateAround[1]]
    #pygame.draw.rect(screen, (0, 255, 255), Rect(rotatedPoint[0], rotatedPoint[1], 5, 5), 1)
    return rotatedPoint

def newBoxStuff(boxStuff, checkQuadrant):
    size = boxStuff[2]
    x = boxStuff[0]
    y = boxStuff[1]
    newSize = size/2
    newX = x
    newY = y
    top = True
    left = True
    if checkQuadrant == [0, 1, 0, 0] or checkQuadrant == [0, 0, 0, 1]:
        left = False
    if checkQuadrant == [0, 0, 0, 1] or checkQuadrant == [0, 0, 1, 0]:
        top = False

    if top == False:
        newY = y + newSize
    if left == False:
        newX = x + newSize
    return [newX, newY, newSize]

def moveQuadrant(oldQuadrant, xMove):
    newQuadrant = [0]
    if xMove == True: #Move on x
        if oldQuadrant == [0, 0, 0, 1]:
            newQuadrant = [0, 0, 1, 0]
        if oldQuadrant == [0, 0, 1, 0]:
            newQuadrant = [0, 0, 0, 1]
        if oldQuadrant == [0, 1, 0, 0]:
            newQuadrant = [1, 0, 0, 0]
        if oldQuadrant == [1, 0, 0, 0]:
            newQuadrant = [0, 1, 0, 0]
    if xMove != True: #Move on y
        if oldQuadrant == [0, 0, 0, 1]:
            newQuadrant = [0, 1, 0, 0]
        if oldQuadrant == [0, 0, 1, 0]:
            newQuadrant = [1, 0, 0, 0]
        if oldQuadrant == [0, 1, 0, 0]:
            newQuadrant = [0, 0, 0, 1]
        if oldQuadrant == [1, 0, 0, 0]:
            newQuadrant = [0, 0, 1, 0]
    return newQuadrant

def calculateInvertedBoxIntersect(planeX1, planeX2, planeY1, planeY2, rayDir, rayOri):

    tx1 = calculateXPlane(planeX1, rayDir, rayOri)
    tx2 = calculateXPlane(planeX2, rayDir, rayOri)

    ty1 = calculateYPlane(planeY1, rayDir, rayOri)
    ty2 = calculateYPlane(planeY2, rayDir, rayOri)

    tmin = max(min(tx1, tx2), min(ty1, ty2))
    tmax = min(max(tx1, tx2), max(ty1, ty2))

    xIntersect = False
    if max(tx1, tx2) < max(ty1, ty2):
        xIntersect = True

    if tmin > tmax or tmax < 0:
        return [10000, False]  # No intersection

    return [tmax, xIntersect]  # Return the entry point

def calculateBoxIntersect(planeX1, planeX2, planeY1, planeY2, rayDir, rayOri):

    if rayOri[0] < planeX2 and rayOri[0] > planeX1 and rayOri[1] < planeY2 and rayOri[1] > planeY1:
        return [0, False, False]

    tx1 = calculateXPlane(planeX1, rayDir, rayOri)
    tx2 = calculateXPlane(planeX2, rayDir, rayOri)

    ty1 = calculateYPlane(planeY1, rayDir, rayOri)
    ty2 = calculateYPlane(planeY2, rayDir, rayOri)

    tmin = max(min(tx1, tx2), min(ty1, ty2))
    tmax = min(max(tx1, tx2), max(ty1, ty2))

    xIntersect = False
    if min(tx1, tx2) > min(ty1, ty2):
        xIntersect = True

    if tmin > tmax or tmax < 0:
        return [10000, False, True]  # No intersection

    #pygame.draw.rect(screen, (255, 255, 0), Rect(xIntersectPoint - 5, yIntersectPoint - 5, 10, 10), 10)
    if tmin < 0:
        tmin = 0
    return [tmin, xIntersect, False]  # Return the entry point

def rotate_vector(vector, angle):
    cos_theta = math.cos(angle)
    sin_theta = math.sin(angle)
    return (vector[0] * cos_theta - vector[1] * sin_theta, vector[0] * sin_theta + vector[1] * cos_theta)

def castRayIntoMovedBox(rayStuff, boundingBox):
    quadtreeBox = boundingBox
    intersectDat = calculateBoxIntersect(quadtreeBox[0], quadtreeBox[0] + quadtreeBox[2], quadtreeBox[1],
                                         quadtreeBox[1] + quadtreeBox[2], rayStuff[0], rayStuff[1])
    tinto = intersectDat[0]
    t = tinto
    if intersectDat[2] == False:
        point = [rayStuff[1][0] + rayStuff[0][0] * tinto, rayStuff[1][1] + rayStuff[0][1] * tinto]
        uv = [(point[0] - quadtreeBox[0]) / quadtreeBox[2], (point[1] - quadtreeBox[1]) / quadtreeBox[2]]
        t = actualCastRay(rayStuff, quadtreeBox, uv)
        if t > 10000:
            t = 10000
    return t

def nAABBIntersect(nAABB, rayStuff):
    # x1 x2
    # y1 y2
    # x1 x2 y1 y2
    rayOri = [rayStuff[1][0], rayStuff[1][1]]
    rotationMatrix = nAABB[3]
    AABB = [nAABB[0], nAABB[1], nAABB[2]]
    newPoint = rotatePoint(rotationMatrix, rayOri, [AABB[0]+AABB[2]/2, AABB[1]+AABB[2]/2])
    newEnd = rotatePoint(rotationMatrix, [newPoint[0]+rayStuff[0][0]*30, newPoint[1]+rayStuff[0][1]*30], newPoint)
    newDir = normalize([newEnd[0]-newPoint[0], newEnd[1]-newPoint[1]])
    intersectData = calculateBoxIntersect(AABB[0], AABB[0]+AABB[2], AABB[1], AABB[1]+AABB[2], newDir, newPoint)
    t = 10000
    if intersectData[2] == False:
        intersectPointInTransform = [newPoint[0]+newDir[0]*intersectData[0], newPoint[1]+newDir[1]*intersectData[0]]
        uv = [(intersectPointInTransform[0]-AABB[0])/AABB[2], (intersectPointInTransform[1] - AABB[1])/AABB[2]]
        transformRayStuff = [newDir, newPoint]
        t = actualCastRay(transformRayStuff, AABB, uv)
        if t > 10000:
            t = 10000
    return t

def doABunchOfRays(pointDirection, pointOrigin, numRays, fov, mode, timecount, timebeforedivide):
    start = time.time()
    half_fov = fov / 2
    angle_increment = fov / (numRays - 1)
    normalized_direction = normalize(pointDirection)
    for i in range(numRays):
        angle = -half_fov + i * angle_increment
        direction = rotate_vector(normalized_direction, angle)
        rayStuff = [direction, pygame.mouse.get_pos()]

        degrees = math.radians(timecount)
        t = nAABBIntersect([0, 0, 1024, [math.cos(degrees), math.sin(degrees), -math.sin(degrees), math.cos(degrees)]], rayStuff)
        drawRayLine(rayStuff[0], rayStuff[1], t)
        pygame.draw.rect(screen, (255, 0, 0), Rect(rayStuff[1][0]+rayStuff[0][0]*t, rayStuff[1][1]+rayStuff[0][1]*t, 5, 5), 0)
    end = time.time()
    timebeforedivide = (end - start)

    #print(uvmax)
render_quadtree_box(0, 0, 0, width, (levelsOfDepth))
screen.fill((0, 0, 0))

def followMouse(rayStuff):
    rayOrigin = rayStuff[1]
    rayDirection = rayStuff[0]
    mousePos = pygame.mouse.get_pos()
    x=mousePos[0]
    y=mousePos[1]
    if x % 2 == 0:
        x = x + 1
    if y % 2 == 0:
        y = y + 1
    rayOri = [x, y]

    #rayDir = normalize([mousePos[0]-rayOri[0], mousePos[1]-rayOri[1]])
    rayStuff = [normalize([0, 1]), rayOri]
    return rayStuff

def drawRayLine(rayDir, rayOri, t):
    startPoint = rayOri
    endPoint = [0]*2
    endPoint[0] = (rayOri[0] + rayDir[0]*t)
    endPoint[1] = (rayOri[1] + rayDir[1]*t)
    pygame.draw.line(screen, (25, 25, 25), (int(startPoint[0]), int(startPoint[1])), (int(endPoint[0]), int(endPoint[1])))

timecount = 0
timebeforedivide = 0
while True:
    timecount = timecount + 1
    start = time.time()
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
    screen.fill((0, 0, 0))
    rayStuff = followMouse([rayDirection, rayOrigin])

    #rayStuff[0] = normalize([-1, 1])
    if rayDirection[0] == 0:
        rayDirection[0] = 0.000001
    if rayDirection[1] == 0:
        rayDirection[1] = 0.000001
    #render_quadtree(0, 0, 0, width, (levelsOfDepth))
    for i in range(0, len(wholeBoxStack)):
        box = wholeBoxStack[i]
        #pygame.draw.rect(screen, (0, 255, 255), Rect(box[0], box[1], box[2], box[2]), 1)

    doABunchOfRays(rayStuff[0], rayStuff[1], 300, 3, 0, timecount, timebeforedivide)

    #t = actualCastRay(rayStuff, 0)
    #drawRayLine(rayStuff[0], rayStuff[1], t)
    pygame.display.flip()
    end = time.time()
    elapsed = end-start
    #print(elapsed*1000, "ms")