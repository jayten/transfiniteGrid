import numpy as np 
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def DrawMesh(x,y):
    """
    Draw computational mesh

    parameters: 
        x (2d numpy array)
        y (2d numpy array)

    output: 
        Matplotlib plot
    """
    fig = plt.figure()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.plot(x,y, color='black')
    plt.plot(x.T,y.T, color='black')
    plt.show()



class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __str__(self):
        return "(%s, %s)" %(self.x, self.y)

    def __repr__(self):
        return "<Point (%s, %s)>" %(self.x, self.y)

    def __add__(self, other):
        return Point(self.x+other.x, self.y+other.y)

    def __sub__(self, other):
        return Point(self.x-other.x, self.y-other.y)
    
    def __eq__(self, other):
        return  (self.x==other.x and self.y==other.y)


class Connector:
    def __init__(self, size=2):
        self.size = size
        self.x = np.empty((size), dtype=np.float64)
        self.y = np.empty((size), dtype=np.float64)
        self.p1 = Point(0,0)
        self.p2 = Point(0,0)
        self.type = "Linear"

    def __repr__(self):
        return "<Connector of type:%s with startpoint:%s and endpoint:%s>"%(self.type, self.p1, self.p2)

    def __str__(self):
        return "Connector of type:%s with startpoint:%s and endpoint:%s"%(self.type, self.p1, self.p2)

    def set_size(self, size):
        self.size = size
    
    def update_endpoints(self):
        self.p1 = Point(self.x[0], self.y[0])
        self.p2 = Point(self.x[-1], self.y[-1])

    def linear(self, p1, p2):
        self.p1 = p1
        self.p2 = p2
        self.x = np.linspace(p1.x, p2.x ,self.size, endpoint=True)
        self.y = np.linspace(p1.y, p2.y ,self.size, endpoint=True)
    
    def circle(self, p1, p2, p3):
        self.p1 = p1
        self.p2 = p3
        self.type = "Arc"
        d1,d2 = (p1 - p2, p1 - p3)
        b = 0.5*np.array(((p1.x*p1.x - p2.x*p2.x) + (p1.y*p1.y - p2.y*p2.y),
                          (p1.x*p1.x - p3.x*p3.x) + (p1.y*p1.y - p3.y*p3.y)))
        deter = d1.x*d2.y - d2.x*d1.y
        Ainv = np.array(((d2.y, -d1.y), (-d2.x, d1.x)))/deter
        h,k = np.matmul(Ainv,b)
        r = np.sqrt((p1.x-h)**2 + (p1.y-k)**2)
        # angle betweeen (0,2pi)
        # arctan will give angle between (-pi,pi)
        theta1 = (2*np.pi + np.arctan((p1.y-k)/(p1.x-h+EPS)))%(2*np.pi)
        theta3 = (2*np.pi + np.arctan((p3.y-k)/(p3.x-h+EPS)))%(2*np.pi)
        theta = np.linspace(theta1, theta3, self.size, endpoint=True)
        self.x = h + r*np.cos(theta)
        self.y = k + r*np.sin(theta)

    
    def fullcircle(self, c, r, p1):
        self.p1 = p1
        self.p2 = p1
        theta1 = np.arccos((p1.x-c.x)/r)
        t = np.linspace(theta1,theta1+2*np.pi,self.size, endpoint=True)
        self.x = c.x + r*np.cos(t)
        self.y = c.y + r*np.sin(t)

    def curve(self, points):
        self.p1 = Point(points[0,0], points[1,0])
        self.p2 = Point(points[0,-1], points[1,-1])
        tck, u = splprep(points, u=None, s=0.0, per=1)
        u_new = np.linspace(u.min(), u.max(), self.size)
        self.x, self.y = splev(u_new, tck, der=0)


    def join(self, other):
        if self.p2 == other.p1:
            self.x = np.concatenate((self.x, other.x[1:]))
            self.y = np.concatenate((self.y, other.y[1:]))
            self.size += other.size - 1
            self.p2 = other.p2
        else:
            print("p1=", self.p2, "and p2=", other.p1)
            raise Exception("Error: Cannot join two connector")


def transfinite(left, right, bottom, top):
    '''the simplest of the
        ‘Coons’ patches’
    '''
    corner1 = Point( left.x[ 0],  left.y[ 0])
    corner2 = Point(right.x[ 0], right.y[ 0])
    corner3 = Point(  top.x[ 0],   top.y[ 0])
    corner4 = Point(  top.x[-1],   top.y[-1])
    eta     = np.linspace(0,1,bottom.size, endpoint=True)
    nta     = np.linspace(0,1,  left.size, endpoint=True)
    #x,y = mesh(left.x, left.y, right.x, right.y, bottom.x, bottom.y, top.x, top.y, bottom.size, right.size)
    x = ( np.outer(1-eta,   left.x)
        + np.outer(  eta,  right.x)
        + np.outer(1-nta, bottom.x).T
        + np.outer(  nta,    top.x).T
        - np.outer(1-eta,1-nta)*corner1.x
        - np.outer(  eta,1-nta)*corner2.x
        - np.outer(1-eta,  nta)*corner3.x
        - np.outer(  eta,  nta)*corner4.x)
    y = ( np.outer(1-eta,   left.y)
        + np.outer(  eta,  right.y)
        + np.outer(1-nta, bottom.y).T
        + np.outer(  nta,    top.y).T
        - np.outer(1-eta,1-nta)*corner1.y
        - np.outer(  eta,1-nta)*corner2.y
        - np.outer(1-eta,  nta)*corner3.y
        - np.outer(  eta,  nta)*corner4.y)
    return (x,y)


# Now the above function will be used to 
# generate a simple transition finite mesh
#
## 15-degree Ramp
Imax = 37#201#37
Jmax = 25#129#25

# Define the point which defines the extent of the domain
p1, p2, p3, p4, p5, p6= (Point(-1,0), 
                      Point(0,0), 
                      Point(1,0.5), 
                      Point(-1,1), 
                      Point(2,1),
                      Point(2,0.5))


# Generate connector between points
#Connector Bottom
cB  = Connector(size=int((Imax-1)/3) + 1)
cB1 = Connector(size=int((Imax-1)/3) + 1)
cB2 = Connector(size=int((Imax-1)/3) + 1)
#Connector Top
cT = Connector(size=Imax)
#Connector Left
cL = Connector(size=Jmax)
#Connector Right
cR = Connector(size=Jmax)

# Bottom connector between p1 -- p2 --  p3 --  p6
cB.linear(p1, p2)
cB1.linear(p2, p3)
cB2.linear(p3, p6)
# Top connector between p4 and p5
cT.linear(p4, p5)
# Left connector between p1 an p4
cL.linear(p1, p4)
# Left connector between p3 and p5
cR.linear(p3, p5)
# Joinging segmented bottom connector into one
cB.join(cB1)
del(cB1)
cB.join(cB2)
del(cB2)

# Generate a transfinite mesh between left, right, bottom, and top connector
x, y = transfinite(cL, cR, cB, cT)
# Visualize mesh 
DrawMesh(x,y)

#---------------------------------------------
#  Additional simple mesh generation code
#---------------------------------------------

## Lid-Driven cavity
''' A simple square domain for simulation
   of viscous flow in a cavity driven by 
   a moving upper wall
'''
Imax = 17
Jmax = 17
x = np.linspace(0,1,Imax, endpoint=True)
y = np.linspace(0,1,Jmax, endpoint=True)
x,y = np.meshgrid(x,y)
#DrawMesh(x,y)

## Sod Problem
''' Simple one-dimensional flow problem with
    rectangular domain
'''
Imax = 17
Jmax = 2
x = np.linspace(0,1,Imax, endpoint=True)
y = np.linspace(0,0.1,Jmax, endpoint=True)
x,y = np.meshgrid(x,y)
#DrawMesh(x,y)
