function writeij(filename, A)
%-----------------------------------------------------------------------------
% writeij('filename', A):
%   Writes to file 'filename' a sparse matrix A in IJ format.
%-----------------------------------------------------------------------------

fid=fopen(filename,'w');

nrows = size(A,1);
fprintf(fid,'%d %d %d %d\n', 0, nrows-1, 0, nrows-1);

% the 'find' function does things in column order, so use A^T
[J,I,V]=find(A');
B(1,:) = I' - 1;
B(2,:) = J' - 1;
B(3,:) = V';

fprintf(fid,'%d %d %.10e\n', B);

fclose(fid);
