import numpy as np
import random as rd
import bpy
import bmesh
import itertools 
import math
import pathlib
import copy


def Random(max):
    randomNumber = rd.uniform(0,max)
    return round(randomNumber, 3)


def DeleteAll():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    
    for material in bpy.data.materials:
        bpy.data.materials.remove(material)
    
    for meshes in bpy.data.meshes:
        bpy.data.meshes.remove(meshes)
        
        
        
##############
#2 The Terrain
##############  
def TerrainFromHeights(A,n,m):
    
    rangeN = np.linspace(0, n, num=n)
    rangeM = np.linspace(0, m, num=m)
    
    terrainPolygon = [(x,y,A[i][j]) for i, x in enumerate(rangeM) for j, y in enumerate(rangeN)]
    ShowMeshFaces(terrainPolygon,n,m)

       
def ShowMeshFaces(A,n,m):
    mesh = bpy.data.meshes.new("Surface")

    bm = bmesh.new()

    vertices = A
    
    print("vertices")
    print(vertices)
    
    #Assign indices to all vertices according to the nxm, so that I can easily 
    #specify with vertex I want.
    numbersN = list(range(0,n))
    numbersM = list(range(0,m))
    indices = [(i,j) for i, x in enumerate(numbersM) for j, y in enumerate(numbersN)] 
    combinations = [val for val in zip(vertices,indices)]
    
    #This is how I get all the different quads
    quads = []
    for x in range(m-1):
        for y in range(n-1):
            quad = [vert for (vert,(i,j)) in combinations if i >= x and i <= x + 1 and j >= y and j <= y+1]
            quads.append(quad)
    
    #rearange the order, because otherwise the face isn't connected correclty.      
    for quad in quads:
        tmp = quad[2]
        quad[2] = quad[3]
        quad[3] = tmp
        
        quad = tuple(bm.verts.new(vert) for vert in quad)
        bm.faces.new(quad)
        
    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new("Surface", mesh)

    scene = bpy.context.scene
    scene.collection.objects.link(obj)
  
    

#################
#2.1 Adding Noise
#################
def ZeroHeights(n,m):
    return np.zeros((m,n))


def AddBump(A,n,m):
    x = Random(m-1)
    y = Random(n-1)
    
    s = Random(((n+m)/10))
    
    for i in range(m-1):
        for j in range(n-1):
            A[i][j] = SmoothFunction(A[i][j],s,i,j,x,y,"add")
    
    return A


def AddBumps(A,n,m,amount):
    for i in range(amount):
        A = AddBump(A,n,m)
    return A


#I use this as the erosion dip
def AddDip(A,n,m,r,x,y):
    #I use the radius as the s in the formula given by the tutorial
    s = r*(n*m)/10000

    #cap the search for neighbours outside of terrain
    maxX = x+r+1
    maxY = y+r+1
    if maxX > m:
        maxX = m-1
    if maxY > n: 
        maxY = n-1
        
    for i in range(x-1,maxX):
        for j in range(y-1,maxY):
            A[i][j] = SmoothFunction(A[i][j],s,i,j,x,y,"sub")
            if A[i][j] < 0:
                A[i][j] = 0
    
    return A

    
def SmoothFunction(element,s,i,j,x,y,op):
    
    vector2D = [i-x,j-y]
    d = np.linalg.norm(vector2D)
    if d < s:
        if op == "add":
            return element + s/math.pi * (1 + math.cos(math.pi*d/s))
        else: 
            #I found this to be a nice curve for the erosion dip
            return element - (s*0.5)/math.pi * (1 + math.cos(math.pi*d/(s)))
    else:
        return element
    
    
    
############   
#2.2 Erosion
############
def ErodeRandomPoint(A,n,m,x,y):
    neighboursAndP = []
    
    #Choose a random point p= (x,y) within the range of the terrain.
    if x == 0 or y == 0 or x == m-1 or y == n-1:
        return A 

    p = A[x][y]
    
    if p <= 0:
        return A
    
    #Look at the 8 neighbours of p, and find the lowest one. Say it is q.
    for i in range(x-1,x+2):
        for j in range(y-1,y+2):
            #I save it like this so that I have the coords of the min point and can recurse easily
            neighboursAndP.append([A[i][j],(i,j)])  
    
    #I check p as well, so that I know if its a local minima
    q = min(neighboursAndP, key = lambda z: z[0]) #https://stackoverflow.com/questions/14802128/tuple-pairs-finding-minimum-using-python
    
    #If q is higher than p, lower q to be equal height top
    if q[0] > p:
        A[q[1][0]][q[1][1]] = p
        return A
        
    #local minimal, go back
    if q[1] == (x,y):
        return A
    
    AddDip(A,n,m,2,i,j)
    return ErodeRandomPoint(A,n,m,q[1][0],q[1][1])


def AddErosions(A,n,m,amount):
    for i in range(amount):
        x = Random(m-1)
        y = Random(n-1)
        A = ErodeRandomPoint(A,n,m,int(x),int(y))
    return A



#################
#2.3 Adding Water
#################
def WaterPlane(n,m):
    height = Random((n+m)/20)  #<---- changed the "/10" to "/20", because my waterheight would be too high.
    return np.full((m,n),height), height



##########################
#3 Adding Vegetation
#3.1 Poisson disk sampling
##########################
def GenerateTree(A,n,m,h,T):
    x = int(Random(m-1))
    y = int(Random(n-1))
    
    #Trees  cannot  grow  above  the  tree  line,  which  we  will  fix  at 1000 meters  for this exercise.
    if A[x][y] >= 20:  #<---- changed this from "1000" to "20", because my bumps aren't going that high.
        return T 
    
    #Trees cannot be under water.
    if A[x][y] <= h:
        return T 
    
    #Trees cannot be closer than 10 meters to each other
    treeDistance = 3  #<---- changed this frfom "10" to "3". Since my trees were a lot bigger the my mountains, I changed their scale
                      #and thus chnaged this distance between them.
    for (i,j) in T:
        vector2D = [i-x,j-y]
        d = np.linalg.norm(vector2D)
        if d < 3:
            return T
            
    #No constraint applies, so set x,y coords in TreeBoolMap to True      
    T.append((x,y))
    return T


def GenerateMushroom(A,n,m,h,T,M):
    x = int(Random((m-1)*100))
    y = int(Random((n-1)*100))
    
    #Mushrooms cannot grow closer than 0.01 meters to each other
    mushroomDistance = 0.01
    for (i,j) in M:
        vector2D = [i-x,j-y]
        d = np.linalg.norm(vector2D)
        if d < mushroomDistance:
            return M
    
    #Mushrooms cannot grow closer than 0.1 meters to the center of a tree trunk       
    #Mushrooms grow in the shadow, and therefore must be within a distance of at most 2 meters from a tree.
    shadowDistance = 2 #2*100
    treeDistance = 0.1
    for (i,j) in T:
        vector2D = [i-(x/100),j-(y/100)]
        d = np.linalg.norm(vector2D)
        if d < shadowDistance and d > treeDistance:
            M.append((x,y)) #<---- "/100" weg
    
    return M
    

def AddVegetation(A,n,m,h):
    TreeList = []
    MushroomList = []
    
    treeAmount = int(Random(n*m/2))
    mushroomAmount = int(Random(n*m*2))
    
    #Try to Place
    for i in range(treeAmount):
        TreeList = GenerateTree(A,n,m,h,TreeList)
        
    for (x,y) in TreeList:
        ShowTree(A,n,m,x,y)
    
    for i in range(mushroomAmount):
        MushroomList = GenerateMushroom(A,n,m,h,TreeList,MushroomList)
    
    for (x,y) in MushroomList:
        ShowMushroom(A,n,m,x,y)


#By using bilinear interpolation, I can approximate the correct mushroom height.
#However, as you might have noticed I am not using this.
#This is because for some reason it doesn't work and I couldn't figure out why...
def ZApprox(A,x,y):
    
    #All x and y points of the square
    xMax = math.ceil(x / 100)
    xMin = int(x/100)
    yMax = math.ceil(y / 100)
    yMin = int(y/100)

    #Calculate z of the first line
    z0 = A[xMin][yMin]
    z1 = A[xMax][yMin]

    t1 = x/(xMax*100)

    z01 = (1-t1)*z0 + t1*z1

    #Calculate z of the second line
    z2 = A[xMin][yMax]
    z3 = A[xMax][yMax]

    z23 = (1-t1)*z2 + t1*z3

    #The final z
    t2 = y/(yMax*100)
    z = (1-t2)*z01 + t2*z23
        
    return z
    

#I changed the scale of my moushrooms and trees, because they weren't in proportion to my mountains
scaleChange = 0.5
treeHeight = 11.9

def ShowTree(A,n,m,x,y):
    filepathTree = str(pathlib.Path().home()/"OneDrive"/"Documenten"/"GitHub"/"3DModelleren"/"3D-modelleren-TUTORIAL-4-PRODECURAL"/"models"/"tree.blend")
    with bpy.data.libraries.load(filepathTree, link=False) as (dataTree, dataTo):
        dataTo.objects = dataTree.objects
    
    #https://www.youtube.com/watch?v=ZrN9w8SMFjo
    scene = bpy.context.scene
    for obj in dataTo.objects:
        obj.location = (x,y,A[x][y]+treeHeight*scaleChange)
        obj.scale *= scaleChange                   
        scene.collection.objects.link(obj)


def ShowMushroom(A,n,m,x,y):
    filepathTree = str(pathlib.Path().home()/"OneDrive"/"Documenten"/"GitHub"/"3DModelleren"/"3D-modelleren-TUTORIAL-4-PRODECURAL"/"models"/"schroom.blend")
    with bpy.data.libraries.load(filepathTree, link=False) as (dataTree, dataTo):
        dataTo.objects = dataTree.objects
    
    #https://www.youtube.com/watch?v=ZrN9w8SMFjo
    scene = bpy.context.scene
    for obj in dataTo.objects:
        obj.location = (x/100,y/100,A[int(x/100)][int(y/100)])
        obj.scale *= scaleChange                  
        scene.collection.objects.link(obj)



###########################
#4.1 Building the quadtree
###########################
#Since this part uses a lot of intermediate described steps, I decided to place these
#descriptions as comments in my code, thinking it would be easier during grading.

#I made a class square so that I can easily specify the origin of each square and put where the points should go to.
class square:
    def __init__(self, size, origin):
        self.size = size
        self.origin = origin
        
    def __str__(self):
        return f'square({self.size},{self.origin})'
    
    
class quadtree_cell:
    def __init__(self, square, points, parent, children):
        self.square = square
        self.points = points
        self.parent = parent
        self.children = children
    
    def __str__(self):
        return f'quadtree_cell({self.square},{self.points},{self.parent},{self.children})'
    
        
def EmptyQuadtree(n):
    return quadtree_cell(square(n*n,(0,0)),[],None,[])


def SelectSquare(C,x,y):
    n = C.square.size/4             #6x6 = 36 -> 3x3 = 9
    subSquareLength = math.sqrt(n)  #sqrt(9)  -> 3 lengte per square
    
    if (x >= C.square.origin[0] and x < C.square.origin[0] + subSquareLength and y >= C.square.origin[1] and y < C.square.origin[1] + subSquareLength): #first square
        return C.children[0]     
    elif (x >= C.square.origin[0] + subSquareLength and x <= C.square.origin[0] + 2*subSquareLength and y >= C.square.origin[1] and y < C.square.origin[1] + subSquareLength): #second square
        return C.children[1]     
    elif (x >= C.square.origin[0] and x < C.square.origin[0] + subSquareLength and y >= C.square.origin[1] + subSquareLength and y <= C.square.origin[1] + 2*subSquareLength): #third square
        return C.children[2]
    else: #fourth sqaure
        return C.children[3]     
    
       
def FindLeafCell(C,x,y):
    # Check if C is already subdivided (if the array of children is non-empty). 
    if C.children != []:
        
        #If so, check which of the child squares contains(x,y), and recurse on that cell.
        square = SelectSquare(C,x,y)
        return FindLeafCell(square,x,y)     
    
    else:
        return C
    
k = 10
def InsertPoint(C,x,y):
    #Check  if C is  a  leaf  cell  (if  the  array  of  children  is  empty).   
    if C.children == []:
        
        #check  if  it  contains  any  points
        if C.points == []:
             #If it is empty, create a new array and insert the point(x,y)
             C.points = [(x,y)]
        #If it is not empty  
        else:
            
            #Check if C contains too many points 
            if len(C.points) >= k: #<------ k = 10
                n = C.square.size/4           #6x6 = 36 -> 3x3 = 9
                qLen = math.sqrt(n)           #sqrt(9)  -> 3 length per square        
                #Create all new child squares
                q1 = quadtree_cell(square(n,(C.square.origin[0],C.square.origin[1])),[],C,[])
                q2 = quadtree_cell(square(n,(C.square.origin[0]+qLen,C.square.origin[1])),[],C,[])
                q3 = quadtree_cell(square(n,(C.square.origin[0],C.square.origin[1]+qLen)),[],C,[])
                q4 = quadtree_cell(square(n,(C.square.origin[0]+qLen,C.square.origin[1]+qLen)),[],C,[])
                C.children = [q1,q2,q3,q4]
                
                #Then, for all points in the  array  of C,  check  in  which  child  cell  they  are,  and  insert  them  into  the corresponding cell
                Leaf = FindLeafCell(C,x,y)
                InsertPoint(Leaf,x,y)            
                
                for i,j in C.points:
                    Leaf = FindLeafCell(C,i,j)
                    InsertPoint(Leaf,i,j)
                    
                #Then remove the array of points from C.
                C.points = []

            else:
                #If it is not empty and points < k, add the point(x,y) to the array.
                C.points.append((x,y))
                
    #If  not,  call FindLeafCell(C,x,y) and insert(x,y) into the returning leaf cell instead.
    else:
        Leaf = FindLeafCell(C,x,y)
        InsertPoint(Leaf,x,y)
        
        
def DrawQuadTree(C):
    #Draw the parent
    qLen = math.sqrt(C.square.size)  #sqrt(9)  -> 3 length per square  
                
    mesh = bpy.data.meshes.new("Square")
    obj = bpy.data.objects.new("Square", mesh)

    scene = bpy.context.scene
    scene.collection.objects.link(obj)

    vertices = [(C.square.origin[0],C.square.origin[1],0),(C.square.origin[0]+qLen,C.square.origin[1],0),(C.square.origin[0],C.square.origin[1]+qLen,0),(C.square.origin[0]+qLen,C.square.origin[1]+qLen,0)]
    edges = [(0,1),(1,3),(2,3),(2,0)]
    
    
    mesh.from_pydata(vertices, edges, [])
    mesh.update()
    
    #Draw the points
    if C.points != []:
        for x,y in C.points:
            DrawPoint(x,y)
    
    #Draw the children
    if C.children != []:
        for child in C.children:
            DrawQuadTree(child)


def DrawPoint(x,y):
    mesh = bpy.data.meshes.new("Point")
    obj = bpy.data.objects.new("Point", mesh)

    vertex = [(x,y,0)]
    
    scene = bpy.context.scene
    scene.collection.objects.link(obj)
    
    mesh.from_pydata(vertex,[],[])
    mesh.update()
 
    
#Used for the PruneAndSearchTest
def DrawSphere(x,y):
    bpy.ops.object.metaball_add(type='BALL', radius=0.30, enter_editmode=False, align='WORLD', location=(x, y, 0), scale=(1,1,1))

    
     
##########################
#4.2 Querying the quadtree
##########################

def Pythagorean(x0,x1,y0,y1):
    return math.sqrt((x1 - x0)**2 + (y1 - y0)**2)


#Returns the distance to the closest corner.
def ClosestCorner(C,point):
    x, y = point
    #"Divide" C in 4 and check in which square the point is, depending on that we know the closest corner.
    n = C.square.size/4             #6x6 = 36 -> 3x3 = 9
    subSquareLength = math.sqrt(n)  #sqrt(9)  -> 3 length per square

    if (x >= C.square.origin[0] and x < C.square.origin[0] + subSquareLength and y >= C.square.origin[1] and y < C.square.origin[1] + subSquareLength): #first square
        return Pythagorean(x,C.square.origin[0],y,C.square.origin[1])
    elif (x >= C.square.origin[0] + subSquareLength and x <= C.square.origin[0] + 2*subSquareLength and y >= C.square.origin[1] and y < C.square.origin[1] + subSquareLength): #second square
        return Pythagorean(x,C.square.origin[0] + subSquareLength,y,C.square.origin[1])   
    elif (x >= C.square.origin[0] and x < C.square.origin[0] + subSquareLength and y >= C.square.origin[1] + subSquareLength and y <= C.square.origin[1] + 2*subSquareLength): #third square
        return Pythagorean(x,C.square.origin[0],y,C.square.origin[1] + subSquareLength)   
    else:
        return Pythagorean(x,C.square.origin[0] + subSquareLength,y,C.square.origin[1] + subSquareLength)  


#I believe we should be checking edges instead of corners, because another point, which is closer to q than p, could be right next to the edge in another square
def ClosestEdge(C,point):
    n = C.square.size/4             #6x6 = 36 -> 3x3 = 9
    subSquareLength = math.sqrt(n)  #sqrt(9)  -> 3 length per square
    
    #distance between the point and every edge of the square
    distances = [
        point[0] - C.square.origin[0],
        point[1] - C.square.origin[1],
        C.square.origin[0] + subSquareLength - point[0],
        C.square.origin[1] + subSquareLength - point[1]
        ]
    minimalDistance = min(distances) 
    
    return minimalDistance
    

def FindPoint(C):
    if C.points != []:
        
        r = Random(len(C.points))
        point = C.points[int(r)]
        return point
    
    else:
        for child in C.children:
            return FindPoint(child)
        
    
def ClosestPoint(C,x,y):
    
    #Call FindLeafCell(C,x,y) to find the leaf cell L that contains(x,y).
    L = FindLeafCell(C,x,y)
    
    #If L has  points  in  its  array,  take  any  point p in  the  array.   
    #This will be our initial guess for the overall closest point.
    if L.points != []:
        r = Random(len(L.points))
        point = L.points[int(r)]
        
    #If L does not have points, and also has no parent, then the quadtree is empty in this case return null.
    elif L.points == [] and L.parent == None:
        return None
    
    #If L does  not  have  points,  but  it  does  have  a  parent,  then L’s parent must have another child with points in it.
    elif L.points == [] and L.parent != None:
        for child in L.parent.children:
            if child.points != []:
                L = child
                break
            
        if len(L.points) > 0:
            r = Random(len(L.points))
            point = L.points[int(r)]
        else:
            #search until we find a point
            for child in L.parent.children:
                if child.children != []:
                    point = ClosestPoint(child,x,y)
                    break
        
    return PruneAndSearch(C,point,(x,y))


def PruneAndSearch(C,p,q):
    
    #Check if the closest corner of the square of C is closer to q than p is.  
    #If not, then none of the points in this branch of the tree can be the closest point, so we can break the recursion and return p.
    if Pythagorean(p[0],q[0],p[1],q[1]) < ClosestCorner(C,q):
        return p
    
    #Otherwise, 
    else:
        #if C is a leaf cell, find the closest point in this cell to q by checking all of them.
        if C.children == []:
            initial = Pythagorean(q[0],p[0],q[1],p[1])
            distances = [Pythagorean(q[0],point[0],q[1],point[1]) for point in C.points]
            
            #If the child has no points then just return p
            if len(distances) == 0:
                return p
                        
            #If  we  do find a point that is closer than p, we return that point instead.
            i = distances.index(min(distances))
            
            #If  none  of  them  is  closer to q than p is,  we return p.  
            if min(distances) > int(initial):
                return p

            else:
                return C.points[i]
            
        #If C is  not  a  leaf  cell,  we  recursively  call PruneAndSearch on C’s children.
        #Then return the closest point to q among the return values of the 4 resursivecalls.  
        else:
            points = [PruneAndSearch(child,p,q) for child in C.children] #if child.points != []]
            distances = [Pythagorean(q[0],point[0],q[1],point[1]) for point in points]
            i = distances.index(min(distances))
            return points[i]
 
 
#######################
#4.3 Using the quadtree
#######################

#Im not using the *100 anymore
def ShowMushroomQuad(A,n,m,x,y):
    filepathTree = str(pathlib.Path().home()/"OneDrive"/"Documenten"/"GitHub"/"3DModelleren"/"3D-modelleren-TUTORIAL-4-PRODECURAL"/"models"/"schroom.blend")
    with bpy.data.libraries.load(filepathTree, link=False) as (dataTree, dataTo):
        dataTo.objects = dataTree.objects
    
    #https://www.youtube.com/watch?v=ZrN9w8SMFjo
    scene = bpy.context.scene
    for obj in dataTo.objects:
        obj.location = (x,y,A[int(x)][int(y)])
        obj.scale *= scaleChange                  
        scene.collection.objects.link(obj)
        
        
        
def AddVegetationUpgraded(A,n,m,h):
    TreeList = []
    MushroomList = []
    
    size = max(n,m)
    trees = EmptyQuadtree(size)
    mushrooms = EmptyQuadtree(size)
        
    treeAmount = int(Random(n*m/2))
    mushroomAmount = int(Random(n*m*2))
    
    #We need to insert an inital dummy point
    TreeList.append((0,0))
    InsertPoint(trees,0,0) 
    InsertPoint(mushrooms,0,0) 
    
    #Try to Place
    for i in range(treeAmount):
        TreeList = GenerateTreeUpgraded(A,n,m,h,TreeList,trees)
        
    for (x,y) in TreeList:
        ShowTree(A,n,m,x,y)
    
    for i in range(mushroomAmount):
        MushroomList = GenerateMushroomUpgraded(A,n,m,h,MushroomList,trees,mushrooms)
    
    for (x,y) in MushroomList:
        ShowMushroomQuad(A,n,m,x,y)
    
    
    
def GenerateTreeUpgraded(A,n,m,h,T,trees):
    x = int(Random(m-1))
    y = int(Random(n-1))
    
    #Trees  cannot  grow  above  the  tree  line,  which  we  will  fix  at 1000 meters  for this exercise.
    if A[x][y] >= 20:  #<---- changed this from "1000" to "20", because my bumps aren't going that high.
        return T 
    
    #Trees cannot be under water.
    if A[x][y] <= h:
        return T 
    
    #test if there is a point that is closer than 3 meters
    testPoint = ClosestPoint(trees,x,y)
    testDistance = Pythagorean(x,testPoint[0],y,testPoint[1])
    
    #Trees cannot be closer than 10 meters to each other
    treeDistance = 10  #<---- changed this frfom "10" to "3". Since my trees were a lot bigger the my mountains, I changed their scale
                      #and thus chnaged this distance between them.
    if testDistance < treeDistance:
        return T
            
    #No constraint applies, so set x,y coords in TreeBoolMap to True  
    #Insert point and add to list
    InsertPoint(trees,x,y)    
    T.append((x,y))
    return T


def GenerateMushroomUpgraded(A,n,m,h,M,trees,mushrooms):
    x = Random(m)
    y = Random(n)
    
    #test if there is a point that is closer than 0.01 meters
    testPointM = ClosestPoint(mushrooms,x,y)
    testDistanceM = Pythagorean(x,testPointM[0],y,testPointM[1])
    
    #Mushrooms cannot grow closer than 0.01 meters to each other
    mushroomDistance = 0.01
    if testDistanceM < mushroomDistance:
        return M
    
    #Mushrooms cannot grow closer than 0.1 meters to the center of a tree trunk       
    #Mushrooms grow in the shadow, and therefore must be within a distance of at most 2 meters from a tree.
    testPointT = ClosestPoint(trees,x,y)
    testDistanceT = Pythagorean(x,testPointT[0],y,testPointT[1])
    
    shadowDistance = 2
    treeDistance = 0.1

    if testDistanceT < shadowDistance and testDistanceT > treeDistance:
        #Insert point in both quads and add to list
        InsertPoint(mushrooms,x,y)
        InsertPoint(trees,x,y)
        M.append((x,y))
    
    return M



##########################[Tests]##############################
#A couple of test which could possible make the grading easier#
###############################################################

def ErosionTest(n,m,Bumps,Erosion):
    DeleteAll()
    
    A = ZeroHeights(n,m)
    AddBumps(A,n,m,Bumps)
    AddErosions(A,n,m,Erosion)
    TerrainFromHeights(A,n,m)

#ErosionTest(100,100,50,300)
    

def PoissonTest(n,m): #Without QuadTree
    DeleteAll()
    
    A = ZeroHeights(n,m)
    AddVegetation(A,n,m,-1)
    TerrainFromHeights(A,n,m)

#PoissonTest(30,30)


#Part 3 overall test
def Part3():
    DeleteAll()
    n = 100
    m = 100
    A = ZeroHeights(n,m)
    AddBumps(A,n,m,70)
    AddErosions(A,n,m,300)
    water, height = WaterPlane(n,m)
    AddVegetation(A,n,m,height)
    TerrainFromHeights(water,n,m)
    TerrainFromHeights(A,n,m)

#Part3()


def QuadTreeTest(n,amount):
    DeleteAll()
    
    A = EmptyQuadtree(n)
    length = math.sqrt(A.square.size)
    
    for i in range(amount):
        x = Random(length)
        y = Random(length)
        InsertPoint(A,x,y)
        
    DrawQuadTree(A)

#The standard k is 10
#k = 1
#QuadTreeTest(6,50)


#There is something wrong with my pruning, it doesnt always show the correct result. 
#Unfortunately I couldn't figure out why...
def PruneAndSearchTest(n,amount):
    DeleteAll()
    
    A = EmptyQuadtree(n)
    length = math.sqrt(A.square.size)
    
    for i in range(amount):
        x = Random(length)
        y = Random(length)
        InsertPoint(A,x,y)
    
    DrawQuadTree(A)
    
    pointCheck = (3.5,5)
    print(ClosestPoint(A,pointCheck[0],pointCheck[1]))
    
    DrawSphere(pointCheck[0], pointCheck[1])

#k = 1
#PruneAndSearchTest(6,10)


#From this test, we can see that my pruning isn't always working correctly working correctly.
k = 10
def PoissonTestQuadTree(n,m): #With QuadTree
    DeleteAll()
    
    A = ZeroHeights(n,m)
    AddVegetationUpgraded(A,n,m,-1)
    TerrainFromHeights(A,n,m)

#PoissonTestQuadTree(30,30)


#Part 4 overall test
#Often times I get a "list index out of range" error which is due to the unfound error in the Pruning/ClosestPoint
#But note that it can sometimes show the result.
def Part4():
    DeleteAll()
    n = 100
    m = 100
    A = ZeroHeights(n,m)
    AddBumps(A,n,m,70)
    AddErosions(A,n,m,300)
    water, height = WaterPlane(n,m)
    AddVegetationUpgraded(A,n,m,height)
    TerrainFromHeights(water,n,m)
    TerrainFromHeights(A,n,m)

#Part4()
