% laplacian_write_matrix script
% 
% Test driver for write_matrix function.
% This driver writes out a 2d or 3d laplacian matrix using either a 3pt or 4
% pt stencil in symmetric format.
% 
% Author: Peter Brown, 12-17-97

% Set up stencil, grid extents, and coefficient information based on dim.
dim = 3;
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
l = zeros(nx,ny,nz);
for kz = kmin:kmax,
  for jy = jmin:jmax,
    for ix = imin:imax,
      l(ix,jy,kz) = (ix-imin+1) + (jy-jmin)*(imax-imin+1) + ...
	  (kz-kmin)*(jmax-jmin+1)*(imax-imin+1);
    end
  end
end
A = spalloc(nx*ny*nz,nx*ny*nz, 3*nx*ny*nz);
for kz = kmin:kmax,
  for jy = jmin:jmax,
    for ix = imin:imax,
      for j = 1:stencil_size,
	if ((ix+stencil(j,1) > 0) & (jy+stencil(j,2) > 0) & ...
	      (kz+stencil(j,3) > 0) & ...
	      (ix+stencil(j,1) <= nx) & (jy+stencil(j,2) <= ny) & ...
	      (kz+stencil(j,3) <= nz)) ...	      
	      A(l(ix,jy,kz), ...
	      l(ix+stencil(j,1),jy+stencil(j,2),kz+stencil(j,3))) ...
	      = data(j);
	end
      end
    end
  end
end

% Open output file
fid = fopen('test.out', 'wt');
if (fid == -1),
  printf('Error: cannot open input file %s\n', filename);
  return;
end

% Write the matrix to the file.
ierr = write_matrix(fid,A,symmetric,dim,grid,stencil);
