% function [ierr] = write_matrix(fid,A,symmetric,dim,grid,stencil);
% --------------------------------------------------------------------------
% Driver to create an output matrix file that one_to_many can then read to
% create several input matrix files for the parallel smg code to use.
% --------------------------------------------------------------------------
% Input:
%       fid       -  output file unit number (fid = fopen( ... ))
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
function [ierr] = write_matrix(fid,A,symmetric,dim,grid,stencil);

ierr = 0;

[n,m] = size(stencil);  % number of rows in stencil is stencil_size
stencil_size = n;

% --------------------------------------------------------------
% Write in the matrix Symmetric, Grid and Stencil information.
% --------------------------------------------------------------

fprintf(fid, 'StructMatrix\n');
fprintf(fid, '\nSymmetric: %d\n', symmetric);

% write grid info
imin = grid(1); jmin = grid(2); kmin = grid(3);
imax = grid(4); jmax = grid(5); kmax = grid(6);
nx = imax - imin + 1;
ny = jmax - jmin + 1;
nz = kmax - kmin + 1;
fprintf(fid, '\nGrid:\n');
fprintf(fid, '%d\n', dim);
fprintf(fid, '%d\n', 1);
if (dim == 2),
  fprintf(fid, '0:  (0, 0, 0)  x  (%d, %d, 0)\n',nx-1,ny-1); 
else
  fprintf(fid, '0:  (0, 0, 0)  x  (%d, %d, %d)\n',nx-1,ny-1,nz-1); 
end

% write stencil info
fprintf(fid, '\nStencil:\n');
fprintf(fid, '%d\n', stencil_size);
for i = 1:stencil_size,
  fprintf(fid, '%d: %d %d %d\n', i-1, ...
             stencil(i,1),stencil(i,2),stencil(i,3));
end
% write matrix data values 
l = zeros(nx,ny,nz);
for kz = kmin:kmax,
  for jy = jmin:jmax,
    for ix = imin:imax,
      l(ix,jy,kz) = (ix-imin+1) + (jy-jmin)*(imax-imin+1) + ...
	  (kz-kmin)*(jmax-jmin+1)*(imax-imin+1);
    end
  end
end
fprintf(fid, '\nData:\n');
for kz = 1:nz,
  for jy = 1:ny,
    for ix = 1:nx,
      for j = 1:stencil_size,
	if ((ix+stencil(j,1) > 0) & (jy+stencil(j,2) > 0) & ...
	      (kz+stencil(j,3) > 0) & ...
	      (ix+stencil(j,1) <= nx) & (jy+stencil(j,2) <= ny) & ...
	      (kz+stencil(j,3) <= nz)),
	  data = A(l(ix,jy,kz),...
	      l(ix+stencil(j,1),jy+stencil(j,2),kz+stencil(j,3)));
	else
	  data = 0;
	end
	if (dim == 2),
	  fprintf(fid, '0: (%d, %d, %d; %d) %e\n',ix-1,jy-1,0,j-1,data);
	else
	  fprintf(fid, '0: (%d, %d, %d; %d) %e\n',ix-1,jy-1,kz-1,j-1,data);
	end
      end
    end
  end
end

% Close input file
fclose(fid);


