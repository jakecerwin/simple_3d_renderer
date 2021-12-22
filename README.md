# simple_3d_renderer
A simple python 3d geometry renderer 

This was created as part of an interview

This program utilizes pygame for its i/o


It generates a 2D projection of the 3D object specified 
by the filename above using a perspective projection multiplying 
each point as vector with the rotation matrix as specifed by 
this link:

https://en.wikipedia.org/wiki/3D_projection#:~:text=provide%20additional%20realism.-,Mathematical%20formula,-%5Bedit%5D

### Input file specification:

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

