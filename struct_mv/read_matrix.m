% function [A,symmetric,dim,grid,stencil] = read_matrix(fid);
% --------------------------------------------------------------------------
% Driver to create an output matrix file that one_to_many can then read to
% create several input matrix files for the parallel smg code to use.
% --------------------------------------------------------------------------
% Input:
%       fid       -  output file unit number (fid = fopen( ... ))
% Output:
%       A         -  matrix object in full or sparse format
%       symmetric -  flag for output file indicating if matrix is symmetric
%                    =1 if symmetric, =0 if nonsymmetric
%       dim       -  dimension of the underlying grid (2 or 3)
%       grid      -  column or row vector of length 6 containing grid
%                    extents.  grid must be loaded as
%
%                    grid(1) = imin; grid(2) = jmin; grid(3) = kmin;
%                    grid(4) = imax; grid(5) = jmax; grid(6) = kmax;
%
%                    where imin <= i <= imax, jmin <= j <= jmax, and
%                    kmin <= k <= kmax.  Note that imin = jmin = kmin = 1
%                    must always be enforced since matlab matrices must have
%                    positive indices
%       stencil   -  array of size (stencil_size,3) where stencil_size is
%                    the number of elements in the stencil
% 
% Original Version:  12-17-97
% Author: pnb
% --------------------------------------------------------------------------
% 
function [A,symmetric,dim,grid,stencil] = read_matrix(fid);


% --------------------------------------------------------------
% Read in the matrix Symmetric, Grid and Stencil information.
% --------------------------------------------------------------

s = fscanf(fid, '%s',1);

% read symmetric info
s = fscanf(fid,'%s',1);
symmetric = fscanf(fid,'%d',1);

% read grid info
s = fscanf(fid, '%s',1);
dim = fscanf(fid, '%d', 1);
d = fscanf(fid, '%d', 1);
s = fscanf(fid,'%s',1);
s = fscanf(fid,'%1s',1);
imin = fscanf(fid,'%d',1);
s = fscanf(fid,'%1s',1);
jmin = fscanf(fid,'%d',1);
s = fscanf(fid,'%1s',1);
kmin = fscanf(fid,'%d',1);
s = fscanf(fid,'%1s',1);
s = fscanf(fid,'%1s',1);
s = fscanf(fid,'%1s',1);
imax = fscanf(fid,'%d',1);
s = fscanf(fid,'%1s',1);
jmax = fscanf(fid,'%d',1);
s = fscanf(fid,'%1s',1);
kmax = fscanf(fid,'%d',1);
s = fscanf(fid,'%1s',1);

nx = imax - imin + 1;
ny = jmax - jmin + 1;
nz = kmax - kmin + 1;
grid = zeros(6,1);
grid(1,1) = imin; grid(2,1) = jmin; grid(3,1) = kmin;
grid(4,1) = imax; grid(5,1) = jmax; grid(6,1) = kmax;

% read stencil info
s = fscanf(fid,'%s',1);
stencil_size = fscanf(fid,'%d',1);
stencil = zeros(stencil_size,3);
for i = 1:stencil_size,
  s = fscanf(fid,'%s',1);
  stencil(i,1) = fscanf(fid,'%d',1);
  stencil(i,2) = fscanf(fid,'%d',1);
  stencil(i,3) = fscanf(fid,'%d',1);
end
% read matrix data values 
s = fscanf(fid, '%s',1);
A = spalloc(nx*ny*nz,nx*ny*nz, stencil_size*nx*ny*nz);
for kz = 1:nz,
  for jy = 1:ny,
    for ix = 1:nx,
      for j = 1:stencil_size;
	l = ix + (jy-1)*nx + (kz-1)*ny*nx;
	m = ix+stencil(j,1) + (jy+stencil(j,2)-1)*nx + ...
	    (kz+stencil(j,3)-1)*ny*nx;
	s = fscanf(fid,'%s',1);
	s = fscanf(fid,'%s',1);
	s = fscanf(fid,'%s',1);
	s = fscanf(fid,'%s',1);
	s = fscanf(fid,'%s',1);
	data = fscanf(fid,'%e',1);
	if ((ix+stencil(j,1) > 0) & (jy+stencil(j,2) > 0) & ...
	      (kz+stencil(j,3) > 0) & ...
	      (ix+stencil(j,1) <= nx) & (jy+stencil(j,2) <= ny) & ...
	      (kz+stencil(j,3) <= nz)) ...	      
	      A(l,m) = data;
	end
      end
    end
  end
end

if (symmetric == 1)
  D = spdiags(A,0); B = A + A'; B = spdiags(D,0,B);
  A = B;
end

% Close input file
fclose(fid);


