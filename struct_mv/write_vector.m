% function [ierr] = write_vector(fid,v,dim,grid);
% --------------------------------------------------------------------------
% Driver to create an output vector file that one_to_many can then read to
% create several input vector files for the parallel smg code to use.
% --------------------------------------------------------------------------
% Input:
%       fid       -  output file unit number (fid = fopen( ... ))
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
% 12-22-97  pnb  modified to use matlab 4.2
% --------------------------------------------------------------------------
% 
function [ierr] = write_vector(fid,v,dim,grid);

ierr = 0;

% --------------------------------------------------------------
% Write the vector Grid information.
% --------------------------------------------------------------

fprintf(fid, 'StructVector\n');

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

% write vector data values 
fprintf(fid, '\nData:\n');
for kz = 1:nz,
  for jy = 1:ny,
    for ix = 1:nx,
      l = ix + (jy-1)*nx + (kz-1)*ny*nx;
      data = v(l);
      if (dim == 2),
	fprintf(fid, '0: (%d, %d, %d; %d) %e\n',ix-1,jy-1,0,0,data);
      else
	fprintf(fid, '0: (%d, %d, %d; %d) %e\n',ix-1,jy-1,kz-1,0,data);
      end
    end
  end
end

% Close input file
fclose(fid);


