function [xy] = readxy(filename)
%-----------------------------------------------------------------------------
% [xy] = readxy('filename'):
%   Reads from file 'filename' a set of xy pairs.
%
%   Format:
%   First line is a dummy, two integers.
%   Next line is 'nv' the number of rows.  Integer.
%   Next 'nv' lines are the x(i), y(i) pairs.
%
%-----------------------------------------------------------------------------
 
fid=fopen(filename,'r');
 
% read junk line
junk = fscanf(fid,'%d',2);
 
nv = fscanf(fid,'%d',1); % number of variables (nv)
nv2=2*nv;
[xy, count] = fscanf(fid,'%le %le',nv2);
xy=reshape(xy,2,nv)';
fclose(fid);
 
