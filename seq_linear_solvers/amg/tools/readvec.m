function [v] = readvec(filename)
%-----------------------------------------------------------------------------
% [v] = readvec('filename'):
%   Reads from file 'filename' a vector v.
%
%   Format:
%   First line is 'nv' the number of rows in matrix.  Integer.
%   Next 'nv' lines are the vector coefficients.
%
%-----------------------------------------------------------------------------

fid=fopen(filename,'r');

% read junk line
junk = fscanf(fid,'%d',2);

nv = fscanf(fid,'%d',1); % number of variables (nv)
[v, count] = fscanf(fid,'%d ',nv);
fclose(fid);

