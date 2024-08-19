import sys
import pygame
from pygame.locals import *
import math
import timeit

pygame.init()
width, height = 1024, 1024
screen = pygame.display.set_mode((width, height))


last_pos = None
rows, cols = 256, 256
totalVoxelsPossible = rows*cols
voxels = [[0] * rows for _ in range(cols)]
voxelSize = width/rows
levelsOfDepth = int(math.log2(rows))
maxPossibleNodes = (4**(levelsOfDepth+1) - 1) // 3
Frames = 0
    #Storaged Flipped
    #01
    #01
    #is really
    #00
    #11

bytes = maxPossibleNodes
bitSize = 15
maxPointerSize = 9
artificalMemory = [0] * bytes

def renderRawVoxels():
    i = 0
    j = 0
    while i < rows:
        while j < cols:

            if voxInBounds(i, j, i+1, j+1):
                j = j
                pygame.draw.rect(screen, (120, 155, 155), pygame.Rect(i * voxelSize, j * voxelSize, voxelSize, voxelSize))
            if voxels[i][j] != 0:
                j = j
            pygame.draw.rect(screen, (50, 50, 50), pygame.Rect(i * voxelSize, j * voxelSize, voxelSize, voxelSize), 1)
            j = j+1
        i = i+1
        j = 0

def voxInBounds(startX, startY, endX, endY):
    i = startX
    j = startY
    i2 = endX
    j2 = endY
    while i < i2:
        while j < j2:
            if voxels[i][j] != 0:
                return True
            j = j+1
        i = i + 1
        j = startY
    return False
#Build the masks for this node. each node has 2 masks, each mask is 4 bits.
#0011 0000 means the two bottom quadrants are parents aswell.
#If the second mask was 0011, then they would be leaf nodes and would not have any children.
def buildMasks(topLeftX, topLeftY, voxelsResolution):
    if voxelsResolution == 1:
        print("ERROR: Size input to buildMasks() Went Below Voxel Resolution")
        return False
    #Create empty masks
    ValidMask = [0, 0, 0, 0]
    LeafMask = [0, 0, 0, 0]
    childSize = int(voxelsResolution/2)

    #Define the index positions for each potential child based off of the parent input parameters
    b1X, b1Y = topLeftX, topLeftY
    b2X, b2Y = topLeftX+childSize, topLeftY
    b3X, b3Y = topLeftX, topLeftY+childSize
    b4X, b4Y = topLeftX+childSize, topLeftY+childSize

    if childSize > 1: #If children are not highest resoltion, only set ValidMask
        if voxInBounds(b1X, b1Y, b1X+childSize, b1Y+childSize):
            ValidMask[0] = 1
        if voxInBounds(b2X, b2Y, b2X + childSize, b2Y + childSize):
            ValidMask[1] = 1
        if voxInBounds(b3X, b3Y, b3X + childSize, b3Y + childSize):
            ValidMask[2] = 1
        if voxInBounds(b4X, b4Y, b4X + childSize, b4Y + childSize):
            ValidMask[3] = 1
    if childSize == 1: #If children are highest resolution, set LeafMask
        if voxInBounds(b1X, b1Y, b1X+childSize, b1Y+childSize):
            ValidMask[0] = 1
            LeafMask[0] = 1
        if voxInBounds(b2X, b2Y, b2X + childSize, b2Y + childSize):
            ValidMask[1] = 1
            LeafMask[1] = 1
        if voxInBounds(b3X, b3Y, b3X + childSize, b3Y + childSize):
            ValidMask[2] = 1
            LeafMask[2] = 1
        if voxInBounds(b4X, b4Y, b4X + childSize, b4Y + childSize):
            ValidMask[3] = 1
            LeafMask[3] = 1
    #Combine the masks into one array
    combinedMask = [ValidMask[0], ValidMask[1], ValidMask[2], ValidMask[3], LeafMask[0], LeafMask[1], LeafMask[2], LeafMask[3]]
    return combinedMask

def buildMasksForWholeResolution(level):
    size = 2**level
    thisLevelMasksTemp = [0] * (size * size)
    k = 0

    # Generate all coordinates for the grid
    coordinates = [(i, j) for i in range(size) for j in range(size)]

    # Sort the coordinates by their Z-order index
    coordinates.sort(key=lambda coord: z_order_index(coord[0], coord[1], level))

    for coord in coordinates:
        x, y = coord

        temp = buildMasks(0 + int((rows / size) * x), 0 + int((rows / size) * y), rows / size)
        set = False
        for num in temp:
            if num != 0:
                set = True
        if set:
            thisLevelMasksTemp[k] = temp
            k = k+1
    thisLevelMasks = [0]*(k+1)
    thisLevelMasks[0] = k
    for l in range(k):
        thisLevelMasks[l+1] = thisLevelMasksTemp[l]
    return thisLevelMasks

def z_order_index(x, y, level):
    z = 0
    for i in range(level):
        z |= ((x >> i) & 1) << (2 * i)
        z |= ((y >> i) & 1) << (2 * i + 1)
    return z

def createQuadtree():
    i = 0
    usedMemory = 0
    sliceTotal = 0
    while i < levelsOfDepth:
        j = 0
        slice = buildMasksForWholeResolution(i)
        sliceAmount = slice[0]
        while j < sliceAmount:
            artificalMemory[usedMemory] = slice[j+1]
            usedMemory = usedMemory + 1
            j = j + 1
        sliceTotal = sliceTotal + sliceAmount
        i = i + 1
    #print("slice total", sliceTotal)
    pointers = createPointers(artificalMemory, sliceTotal)
    k = 0
    while k < sliceTotal:
        newMemory = [0]*2
        newMemory[0] = pointers[k]
        newMemory[1] = artificalMemory[k]
        artificalMemory[k] = newMemory
        k = k+1

def createPointers(allMasks, maskCount):
    pointers = [3] * maskCount
    childCounter = 0
    address = 0
    j = 0
    k = 0
    parentCount = 0
    while address < maskCount:
        if allMasks[address] != 0:
            parent = False
            j = 0
            k = 0
            while k < 4:
                if allMasks[address][k] != 0 and allMasks[address][k+4] == 0:
                    parent = True
                k = k + 1
            if parent:
                parentCount = parentCount+1
                childPointerDec = childCounter-address+1
            else:
                childPointerDec = 0
            while j < 4:
                if allMasks[address][j] != 0 and allMasks[address][j+4] == 0:
                    childCounter = childCounter+1
                j = j + 1
            pointers[address] = decimal_to_binary(childPointerDec, maxPointerSize)
        address = address + 1
    return pointers

def decimal_to_binary(n, num_digits):
    if n == 0:
        return [0]*num_digits
    binary = []
    while n > 0:
        binary.insert(0, n % 2)
        n = n // 2
    while len(binary) < num_digits:
        binary.insert(0, 0)
    return binary

def binary_to_decimal(binary):
    binary = ''.join(str(bit) for bit in binary)
    decimal = int(binary, 2)
    return decimal

def calculate_quadrant_to_voxel(x, y, resolution):
    normalizedX = x/resolution
    normalizedY = y/resolution
    bottom = False
    right = False
    if normalizedX >= 0.5:
        right = True
    if normalizedY >= 0.5:
        bottom = True
    if bottom and right: return [0, 0, 0, 1]
    if bottom and right == False: return [0, 0, 1, 0]
    if bottom == False and right: return [0, 1, 0, 0]
    if bottom == False and right == False: return [1, 0, 0, 0]

def calculate_path_to_voxel(x, y, startResolution):
    resolution = startResolution
    relativeX = x
    relativeY = y
    path = [0]*levelsOfDepth
    address = 0
    while resolution > 1:
        tempPath = calculate_quadrant_to_voxel(relativeX, relativeY, resolution)
        if tempPath == [0, 0, 0, 1]:
            relativeX = relativeX - resolution/2
            relativeY = relativeY - resolution/2
        if tempPath == [0, 0, 1, 0]:
            relativeY = relativeY + resolution/2
        if tempPath == [0, 1, 0, 0]:
            relativeX = relativeX - resolution/2
        path[address] = tempPath
        resolution = resolution / 2
        address = address+1
    return path

def insertNode(node, address):
    replacementNode = node
    while address+1 < maxPossibleNodes:
        temp = artificalMemory[address]
        artificalMemory[address] = replacementNode
        replacementNode = temp
        address = address+1

def recalculatePointersForWholeMemory():
    artMemory2 = [0] * maxPossibleNodes
    seperateMaskAddress = 0
    while seperateMaskAddress < maxPossibleNodes:
        if artificalMemory[seperateMaskAddress] != 0:
            artMemory2[seperateMaskAddress] = artificalMemory[seperateMaskAddress][1]
        seperateMaskAddress = seperateMaskAddress + 1
    newPointers = createPointers(artMemory2, maxPossibleNodes)
    insertPointers = 0
    while insertPointers < maxPossibleNodes:
        if artificalMemory[insertPointers] != 0:
            artificalMemory[insertPointers][0] = newPointers[insertPointers]
        insertPointers = insertPointers + 1

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
                pygame.draw.rect(screen, (255, 255, 255), Rect(pos[0], pos[1], thisLevelBoxWidth, thisLevelBoxWidth), 1)
                if leafMask[i] == 0:
                    #print(node)
                    render_quadtree(thisAddress, pos[0], pos[1], thisLevelBoxWidth, lod, steps)
                if leafMask[i] != 0:
                    pygame.draw.rect(screen, (0, 255, 0), Rect(pos[0], pos[1], thisLevelBoxWidth, thisLevelBoxWidth), 1)

def updateVoxelsFromMouse():
    global last_pos  # store the last position
    mouseX = pygame.mouse.get_pos()[0]/width
    mouseY = pygame.mouse.get_pos()[1]/width
    pressed = pygame.mouse.get_pressed(5)

    voxMouseX = int(mouseX*rows)
    voxMouseY = int(mouseY*cols)

    if pressed[0]:
        # If last_pos exists, interpolate between last_pos and current position
        if last_pos is not None:
            dx = voxMouseX - last_pos[0]
            dy = voxMouseY - last_pos[1]
            distance = max(abs(dx), abs(dy))
            if distance > 0:  # Check to avoid division by zero
                for i in range(distance + 1):
                    interpolateX = last_pos[0] + i / distance * dx
                    interpolateY = last_pos[1] + i / distance * dy
                    voxels[int(interpolateX)][int(interpolateY)] = 1

    last_pos = (voxMouseX, voxMouseY)  # update last_pos for next frame

def clearVoxels():
    for i in range(len(voxels)):
        for j in range(len(voxels[i])):
            voxels[i][i] = 1
            voxels[(cols-1)-i][i] = 1
            voxels[int((cols)/2)][j] = 1
            voxels[j][int((cols+1)/2)] = 1
    for i in range(len(voxels)):
        for j in range(len(voxels[i])):
            if (cols/2-i)*(cols/2-i)+(cols/2-j)*(cols/2-j) < 150:
                voxels[i][j] = 0

insertNode([[0, 0, 0, 0, 0, 0, 0, 0, 1],[0, 0, 0, 0, 0, 0, 0, 0]], 0)

programStartTime = timeit.default_timer()

k = 0
i = 0
while True:
    updateVoxelsFromMouse()
    #voxels[0][0] = 1
    #clearVoxels()
    createQuadtree()
    start_time = timeit.default_timer() * 1000
    Frames = Frames + 1
    for event in pygame.event.get():
        if event.type == QUIT:
            with open("Quadtree.txt", "w") as file:
                new_content = str(artificalMemory)
                file.write(new_content)
                file.close()
            pygame.quit()
            sys.exit()
    screen.fill((0, 0, 0))
    render_quadtree(0, 0, 0, width, (levelsOfDepth))
    pygame.display.flip()
      # End the timer
    end_time = timeit.default_timer() * 1000
