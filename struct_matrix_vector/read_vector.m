% function [v,dim,grid] = read_vector(fid);
% --------------------------------------------------------------------------
% Driver to create an output vector file that one_to_many can then read to
% create several input vector files for the parallel smg code to use.
% --------------------------------------------------------------------------
% Input:
%       fid       -  output file unit number (fid = fopen( ... ))
% output:
%       v         -  vector object in full or sparse format
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
% 
% Original Version:  12-18-97
% Author: pnb
% --------------------------------------------------------------------------
% 
function [v,dim,grid] = read_vector(fid);

% --------------------------------------------------------------
% Read the vector Grid information.
% --------------------------------------------------------------

s = fscanf(fid, '%s',1);

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

% read vector data values 
s = fscanf(fid, '%s',1);
v = zeros(nx*ny*nz,1);
for kz = 1:nz,
  for jy = 1:ny,
    for ix = 1:nx,
      l = ix + (jy-1)*nx + (kz-1)*ny*nx;
      s = fscanf(fid,'%s',1);
      s = fscanf(fid,'%s',1);
      s = fscanf(fid,'%s',1);
      s = fscanf(fid,'%s',1);
      s = fscanf(fid,'%s',1);
      data = fscanf(fid,'%e',1);
      v(l) = data;
    end
  end
end

% Close input file
fclose(fid);


