function A = readij(filename)
%-----------------------------------------------------------------------------
% A = readij('filename'):
%   Reads from file 'filename' a matrix A in IJ format.
%-----------------------------------------------------------------------------

fid=fopen(filename,'r');

nrows = fscanf(fid,'%d',1); % number of rows (nrows)
[Adata, count] = fscanf(fid,'%d,%d,%e', [3, inf]);
fclose(fid);

Adata(1,:) = Adata(1,:) + 1;
Adata(2,:) = Adata(2,:) + 1;

A = sparse(Adata(1,:)', Adata(2,:)' ,Adata(3,:)');

