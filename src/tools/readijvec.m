function x = readijvec(filename)
%-----------------------------------------------------------------------------
% x = readijvec('filename'):
%   Reads from file 'filename' a vector x in IJ format.
%-----------------------------------------------------------------------------

fid=fopen(filename,'r');

[ijlohi, count] = fscanf(fid,'%d %d', 2);
[xdata, count]  = fscanf(fid,'%d %e', [2, inf]);
fclose(fid);

x = xdata(2,:)';
