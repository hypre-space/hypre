function [A,symmetric,dim,grid,stencil] = readstruct(filename);
% --------------------------------------------------------------------------
% [A,symmetric,dim,grid,stencil] = readstruct('filename')
%   Reads from file 'filename' a matrix A in Struct format
% --------------------------------------------------------------------------

fid=fopen(filename,'r');

% --------------------------------------------------------------
% Read in the matrix Symmetric, Grid and Stencil information.
% --------------------------------------------------------------

s = fscanf(fid, '%s',1);

% read symmetric info
s = fscanf(fid,'%s',1);
symmetric = fscanf(fid,'%d',1);

% read constant-coefficient info
s = fscanf(fid,'%s',1);
constcoeff = fscanf(fid,'%d',1);

% read grid info
s = fscanf(fid, '%s',1);
dim = fscanf(fid, '%d', 1);
d = fscanf(fid, '%d', 1);
s = fscanf(fid,'%s',1);
s = fscanf(fid,'%1s',1);
xmin = fscanf(fid,'%d',1);
s = fscanf(fid,'%1s',1);
ymin = fscanf(fid,'%d',1);
s = fscanf(fid,'%1s',1);
zmin = fscanf(fid,'%d',1);
s = fscanf(fid,'%1s',1);
s = fscanf(fid,'%1s',1);
s = fscanf(fid,'%1s',1);
xmax = fscanf(fid,'%d',1);
s = fscanf(fid,'%1s',1);
ymax = fscanf(fid,'%d',1);
s = fscanf(fid,'%1s',1);
zmax = fscanf(fid,'%d',1);
s = fscanf(fid,'%1s',1);

grid = zeros(6,1);
grid(1,1) = xmin; grid(2,1) = ymin; grid(3,1) = zmin;
grid(4,1) = xmax; grid(5,1) = ymax; grid(6,1) = zmax;

nx = xmax - xmin + 1;
ny = ymax - ymin + 1;
nz = zmax - zmin + 1;
n  = nx*ny*nz;

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

% read matrix coefficients
s = fscanf(fid,'%s',1);
[Fdata, count]  = fscanf(fid,'%d: (%d, %d, %d; %d) %e', [6, inf]);
fclose(fid);

Adata = zeros(3,size(Fdata,2));
ii = 1;
for i = 1:size(Fdata,2)
  s = Fdata(5,i) + 1;
  ix = Fdata(2,i);
  iy = Fdata(3,i);
  iz = Fdata(4,i);
  jx = ix + stencil(s,1);
  jy = iy + stencil(s,2);
  jz = iz + stencil(s,3);
  if ( (jx >= xmin) & (jx <= xmax) & ...
       (jy >= ymin) & (jy <= ymax) & ...
       (jz >= zmin) & (jz <= zmax) )
    Adata(1,ii) = ix + iy*nx + iz*ny*nx + 1;
    Adata(2,ii) = jx + jy*nx + jz*ny*nx + 1;
    Adata(3,ii) = Fdata(6,i);
    ii = ii + 1;
  end
end
Adata = Adata(:,1:(ii-1));

A = sparse(Adata(1,:)', Adata(2,:)', Adata(3,:)');

if (symmetric),
  D = diag(diag(A));
  A = A + A' - D;
end