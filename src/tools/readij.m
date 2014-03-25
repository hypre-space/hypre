function A = readij(filename)
%-----------------------------------------------------------------------------
% A = readij('filename'):
%   Reads from file 'filename' a matrix A in IJ format.
%-----------------------------------------------------------------------------

fid=fopen(filename,'r');

[ijlohi, count] = fscanf(fid,'%d %d %d %d', 4);
[Adata, count]  = fscanf(fid,'%d %d %e', [3, inf]);
fclose(fid);

Adata(1,:) = Adata(1,:) + (1-ijlohi(1));
Adata(2,:) = Adata(2,:) + (1-ijlohi(3));

A = sparse(Adata(1,:)', Adata(2,:)' ,Adata(3,:)');

