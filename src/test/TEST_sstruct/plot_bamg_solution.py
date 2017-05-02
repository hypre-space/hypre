from scipy import *
from pylab import *

# This file just loads and plots a solution file for a single processor case.  It loads
fname = 'sstruct.out.x.00.00.00000'
fname = 'sysbamg_tv_l=0,k=4.dat.00.00000'
# where the numbers at the end are 'part, var, myid', that is part=0, var=0 and myid=0 here.
# It is also assumed that this is a regular 3D grid, ordered lexicographically.

# Grab the file
file  = open(fname, 'r') 
text = file.readlines()
file.close()

# Grab the lower grid extents 
grid = text[5]
grid_dims = zeros((2,3))
start = grid.find('(', 0)
end = grid.find(',', 0)
grid_dims[0,0] = int(grid[start+1:end])
start = end+1
end = grid.find(',', start)
grid_dims[0,1] = int(grid[start+1:end])
start = end+1
end = grid.find(')', start)
grid_dims[0,2] = int(grid[start+1:end])

# Grab the upper grid extents 
start = grid.find('(', start)
end = grid.find(',', start)
grid_dims[1,0] = int(grid[start+1:end])
start = end+1
end = grid.find(',', start)
grid_dims[1,1] = int(grid[start+1:end])
start = end+1
end = grid.find(')', start)
grid_dims[1,2] = int(grid[start+1:end])

# Now peel off all the non-data lines 
text = text[10:]

# Loop over the lines, grabbing data values
grid_size = array((grid_dims[1,0] - grid_dims[0,0] + 1, 
                   grid_dims[1,1] - grid_dims[0,1] + 1, 
                   grid_dims[1,2] - grid_dims[0,2] + 1)) 
data = zeros((grid_size[0]*grid_size[1]*grid_size[2],))

for index, line in enumerate(text):
    start = line.find(')', 0)
    data[index] = float(line[start+1:])

# This reorders the data in the same way as the file (row-major, the first index varies most often)
data = data.reshape(grid_size)
print "Grid size is:  " + str(grid_size)

# Now visualize!
figure(1)
title('Slice z=0')
imshow(data[:,:,0])
colorbar()
figure(2)
title('Slice z=half')
imshow(data[:,:,int(grid_size[2]/2)])
colorbar()
figure(3)
title('Slice z=last')
imshow(data[:,:,-1])
colorbar()
show()


