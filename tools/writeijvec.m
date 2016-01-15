function writeijvec(filename, x)
%-----------------------------------------------------------------------------
% writeijvec('filename', x):
%   Writes to file 'filename' a vector x in IJ format.
%-----------------------------------------------------------------------------

fid=fopen(filename,'w');

nrows = size(x,1);
fprintf(fid,'%d %d\n', 0, nrows-1);

% the 'find' function does things in column order, so use A^T
i = (0:nrows-1)';
B(1,:) = i';
B(2,:) = x';

fprintf(fid,'%d %.10e\n', B);

fclose(fid);
