function [xyz] = readxyz(filename)
%-----------------------------------------------------------------------------
% [xyz] = readxyz('filename'):
%   Reads from file 'filename' a set of xyz triplets.
%
%   Format:
%   First line is 'nv' the number of rows.  Integer.
%   Next 'nv' lines are the x(i), y(i), z(i).
%
%-----------------------------------------------------------------------------
 
fid=fopen(filename,'r');
 
nv = fscanf(fid,'%d',1); % number of variables (nv)
nv3=3*nv;
[xyz, count] = fscanf(fid,'%le %le %le',nv3);
xyz=reshape(xyz,3,nv)';
fclose(fid);
 
