% laplacian_write_matrix script
% 
% Test driver for write_matrix function.
% This driver writes out a 2d or 3d laplacian matrix using either a 3pt or 4
% pt stencil in symmetric format.
% 
% Author: Peter Brown, 12-17-97
% 12-18-97  PNB  Added capability to write out vectors
% 12-22-97  PNB  Converted scripts to run under matlab 4.2

% Set up stencil, grid extents, and coefficient information based on dim.
dim = 2;
if (dim == 2),
  symmetric = 1;
  stencil = zeros(3,3);
  stencil(1,1) = 0; stencil(1,2) = 0; stencil(1,3) = 0;
  stencil(2,1) =-1; stencil(2,2) = 0; stencil(2,3) = 0;
  stencil(3,1) = 0; stencil(3,2) =-1; stencil(3,3) = 0;
  del_x = 1.0;
  del_y = 1.0;
  data(1) = 2.0*(1.0/(del_x*del_x) + 1.0/(del_y*del_y));
  data(2) = -1.0/(del_x*del_x);
  data(3) = -1.0/(del_y*del_y);
  imin = 1; jmin = 1; kmin = 1; % Note that imin,jmin,kmin must all equal 1
  imax = 10; jmax = 10; kmax = 1;
else
  symmetric = 1;
  stencil = zeros(4,3);
  stencil(1,1) = 0; stencil(1,2) = 0; stencil(1,3) = 0;
  stencil(2,1) =-1; stencil(2,2) = 0; stencil(2,3) = 0;
  stencil(3,1) = 0; stencil(3,2) =-1; stencil(3,3) = 0;
  stencil(4,1) = 0; stencil(4,2) = 0; stencil(4,3) =-1;
  del_x = 1.0;
  del_y = 1.0;
  del_z = 1.0;
  data(1) = 2.0*(1.0/(del_x*del_x) + 1.0/(del_y*del_y) + 1.0/(del_z*del_z));
  data(2) = -1.0/(del_x*del_x);
  data(3) = -1.0/(del_y*del_y);
  data(4) = -1.0/(del_z*del_z);
  imin = 1; jmin = 1; kmin = 1; % Note that imin,jmin,kmin must all equal 1
  imax = 10; jmax = 10; kmax = 10;
end

% Load grid values
grid = [imin,jmin,kmin,imax,jmax,kmax];
nx = imax - imin + 1;
ny = jmax - jmin + 1;
nz = kmax - kmin + 1;

% Load stencil_size
[n,m]=size(stencil);
stencil_size = n;

% Load matrix data values
A = spalloc(nx*ny*nz,nx*ny*nz, 3*nx*ny*nz);
for kz = 1:nz,
  for jy = 1:ny,
    for ix = 1:nx,
      for j = 1:stencil_size,
	l = ix + (jy-1)*nx + (kz-1)*ny*nx;
	m = ix+stencil(j,1) + (jy+stencil(j,2)-1)*nx + ...
	    (kz+stencil(j,3)-1)*ny*nx;
	if ((ix+stencil(j,1) > 0) & (jy+stencil(j,2) > 0) & ...
	      (kz+stencil(j,3) > 0) & ...
	      (ix+stencil(j,1) <= nx) & (jy+stencil(j,2) <= ny) & ...
	      (kz+stencil(j,3) <= nz)) ...	      
	      A(l,m) = data(j);
	end
      end
    end
  end
end

% Open output file for matrix
fid = fopen('test_2d_matrix.out', 'wt');
if (fid == -1),
  printf('Error: cannot open input file %s\n', filename);
  return;
end

% Write the matrix to the file.
ierr = write_matrix(fid,A,symmetric,dim,grid,stencil);

% Open output file for vector
fid = fopen('test_2d_vector_in.out', 'wt');
if (fid == -1),
  printf('Error: cannot open input file %s\n', filename);
  return;
end

% Load a vector.
v = ones(nx*ny*nz,1);

% Write the vector to the file.
ierr = write_vector(fid,v,dim,grid);

% Full matrix and output vector calculation.
D = spdiags(A,0); B = A + A'; B = spdiags(D,0,B);
w = B*v; % Matrix vector product
% w = B\v; % Matrix solve

% Open output file for vector
fid = fopen('test_2d_vector_out.out', 'wt');
if (fid == -1),
  printf('Error: cannot open input file %s\n', filename);
  return;
end

% Write the vector to the file.
ierr = write_vector(fid,w,dim,grid);

