"""
Written By Jake Cerwin
Carnegie Mellon
Email JCERWIN@andrew.cmu.edu
12/14/2021


"""
import pygame
import numpy as np
from math import cos, sin, atan, sqrt, acos

# This is the txt file the shape will be generated from, change file location to change shape
filename = 'object.txt'

"""
Overview:
This file utilizes pygame for its i/o

It generates a 2D projection of the 3D object specified 
by the filename above using a perspective projection multiplying 
each point as vector with the rotation matrix as specifed by 
this link:
https://en.wikipedia.org/wiki/3D_projection#:~:text=provide%20additional%20realism.-,Mathematical%20formula,-%5Bedit%5D
"""

# AXIS enumeration used throughout document
X, Y, Z = 0, 1, 2

# Window size used
WIDTH = 600
HEIGHT = 400

# Polygon size
SCALE = 40

# Color presets
WHITE = (255, 255, 255)
BLUE = (0,0,255)

###############################################################
# Classes
###############################################################

# Vertex class for containing id and coordinates of each point
class Vertex:
    def __init__(self, n, x, y, z):
        self.n = n
        self.x = x
        self.y = y
        self.z = z


# Shape Class for containing vertices, edges, x,y,z rotation
# center (cx, cy) position of shape, scale and vertex illustration sizes
class Shape:
    # Initilize shape
    def __init__(self, vertices, edges, faces):
        self.vertices = vertices
        self.edges = edges
        self.faces = faces
        self.rotation = [0,0,0]
        self.cx = WIDTH // 2
        self.cy = HEIGHT // 2
        self.scale = SCALE
        self.rad = self.scale // 10

    # generates rotation matrix based on current x,y,z rotation values.
    # Constructs a rotation matrix. This' dot product with initial points
    # generates a 2d projection of figure at current rotation
    # math information available at
    # https://en.wikipedia.org/wiki/3D_projection#:~:text=provide%20additional%20realism.-,Mathematical%20formula,-%5Bedit%5D
    def rotation_matrix(self):
        x = self.rotation[X]
        y = self.rotation[Y]
        z = self.rotation[Z]

        sX, cX = sin(x), cos(x)
        sY, cY = sin(y), cos(y)
        sZ, cZ = sin(z), cos(z)

        return np.array([
            [cY * cZ, -cY * sZ, sY],
            [cX * sZ + sX * sY * cZ, cX * cZ - sZ * sX * sY, -cY * sX],
            [sZ * sX - cX * sY * cZ, cX * sZ * sY + sX * cZ, cX * cY]
        ])

    # adjusts rotation values in x, y, z directions respectively in radians
    def rotate(self, x, y, z):
        self.rotation[X] += x
        self.rotation[Y] += y
        self.rotation[Z] += z

    # resets polygon to original shape
    def reset(self):
        self.rotation[X] = 0
        self.rotation[Y] = 0
        self.rotation[Z] = 0

    # draws the figure on the
    def project(self):
        def draw_edge(p1, p2):
            pygame.draw.line(
                screen, BLUE,
                (p1.x, p1.y), (p2.x, p2.y)
            )
            return

        def draw_polygon_alpha(points):
            TBLUE = (0, 0, 255, 97)
            lx, ly = zip(*points)
            min_x, min_y, max_x, max_y = min(lx), min(ly), max(lx), max(ly)
            target_rect = pygame.Rect(min_x, min_y, max_x - min_x, max_y - min_y)
            shape_surf = pygame.Surface(target_rect.size, pygame.SRCALPHA)
            pygame.draw.polygon(shape_surf, TBLUE, [(x - min_x, y - min_y) for x, y in points])
            screen.blit(shape_surf, target_rect)

        def draw_shaded_polygon(points):
            def mag(x):
                return sqrt(sum(i ** 2 for i in x))

            # find the equation of the plane
            p1, p2, p3 = points

            # These two vectors are in the plane
            v1 = p3 - p1
            v2 = p2 - p1

            # the cross product is a vector normal to the plane
            cp = np.cross(v1, v2)

            zvector = np.array([0,0,1])
            # find the angle between z vector and the normal vector of the plane
            angle = acos(np.dot(cp, zvector) / mag(cp))

            # find the complementary angle
            distance = abs(angle -(np.pi/2))

            # determine color based on complementary angle
            TBLUE = (0, 0, ( 160 * (distance / (np.pi / 2)) + 95))

            pygame.draw.polygon(screen, TBLUE, [(p[0], p[1]) for p in points])

            return

        points = self.vertices.values()

        tp = dict()

        for vertex in points:
            point = np.array([vertex.x, vertex.y, vertex.z])

            projected = np.dot(self.rotation_matrix(), point.reshape((3, 1)))
            x = int(projected[0][0] * self.scale + self.cx)
            y = int(projected[1][0] * self.scale + self.cy)
            z = int(projected[2][0] * self.scale)

            pygame.draw.circle(screen, BLUE, (x, y), self.rad)
            if vertex.n not in tp:
                # save the new coordinates to avoid recalculation
                tp[vertex.n] = Vertex(vertex.n, x, y, z)

        for edge in self.edges:
            p1 = tp[edge[0]]
            p2 = tp[edge[1]]

            draw_edge(p1, p2)

        # order faces by increasing depth to ensure front facing ones are renderered on top
        faces = []
        for vertices in self.faces:
            points = []
            depth = []
            for point in vertices:
                points.append(np.array([tp[point].x, tp[point].y, tp[point].z]))
                depth.append(tp[point].z)

            depth.sort()

            faces.append((depth,points))

        # we sort faces by their lowest, second lowest, than highest point,
        # this ensures correct ordering during the render
        faces.sort(key=lambda x:(x[0][0], x[0][1], x[0][2]))

        for depth, points in faces:
            draw_shaded_polygon(points)

###############################################################
# File Input
###############################################################

"""
Input file specification

The first line contains 2 integers. the first integer is the number of 
verties that define the 3d, and the second number is the number of faces
that define the 3d object

starting at the second line each line will define one vertex of the 3d 
object and will consist of an integer followed by three real numbers.
The integer is the ID of the vertex and the three real numbers define 
(x,y,z) coordinates of the vertex. The number of lines in this section
will be equal to the first integer in the file.

Following this vertex section will be a section defining the faces
of the 3d object. The number of lines in this section will be equal to
the second integer of the first line
"""


# will cast everything it can to int type
# will leave remaining as its original type
def int_cast(x):
    try:
        return int(x)
    except ValueError:
        return x


# will cast everything it can to float type
# will leave remaining as its original type
def flt_cast(x):
    try:
        return float(x)
    except ValueError:
        return x


with open(filename, 'r') as f:
    lines = f.readlines()
    nV, nF = [int_cast(s) for s in lines[0].split(',')]

    # add vertices
    vertices = {}
    for line in lines[1:nV+1]:
        n, x, y, z = line.split(',')
        x, y, z = [flt_cast(s) for s in [x, y, z]]
        vertices[int(n)] = Vertex(int(n), x,y,z)

    # add faces
    faces = []
    for line in lines[nV+1:]:
        faces.append([int_cast(s) for s in line.split(',')])

    # from faces find edges. This currently only works for 3 edges per face
    edges = []
    edge_set = set()
    for face in faces:
        for i in range(len(face) - 1):
            for j in range(i + 1, len(face)):
                # we store edges in lowerid, higherid order for clarity and organization
                edge = (min([face[i], face[j]]), max([face[i], face[j]]))

                # edge set ensures we aren't redrawing the same edge multiple times
                if edge not in edge_set:
                    edges.append(edge)
                    edge_set.add(edge)

    # generate polygon shape instance
    shape = Shape(vertices, edges, faces)

###############################################################
# GUI
###############################################################

pygame.display.set_caption("NEOCIS PROJECTION Extra Credit- Jake Cerwin")
screen = pygame.display.set_mode((WIDTH, HEIGHT))

projection_matrix = np.matrix([])

background = pygame.Surface(screen.get_size())
ts, w, h, c1, c2 = 50, *screen.get_size(), (160, 160, 160), (192, 192, 192)
tiles = [((x*ts, y*ts, ts, ts), c1 if (x+y) % 2 == 0 else c2) for x in range((w+ts-1)//ts) for y in range((h+ts-1)//ts)]
for rect, color in tiles:
    pygame.draw.rect(background, color, rect)

last_x, last_y = pygame.mouse.get_pos()
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                pygame.quit()
                exit()
            if event.key == pygame.K_r:
                shape.reset()

        if pygame.mouse.get_pressed()[0]:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            delta_x, delta_y = mouse_x - last_x, mouse_y - last_y

            # clean out some noise
            if abs(delta_x) < 10: delta_x = 0
            if abs(delta_y) < 10: delta_y = 0

            shape.rotate(delta_y / 10000, delta_x / 10000, 0)

        else:
            last_x, last_y = pygame.mouse.get_pos()

    # update
    screen.blit(background, (0, 0))
    shape.project()
    pygame.display.update()

