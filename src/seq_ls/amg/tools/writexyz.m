function writexyz(xyz,fname)
%-----------------------------------------------------
% function writexyz(xyz,fname)
%    Writes xyz data to filename
%
%    Format:
%    First line is 'nv', the length of the vector.
%    Next nv lines are the data, x(i), y(i), z(i)
%------------------------------------------------------
[nv,nw]=size(xyz);
fid=fopen(fname,'w');
fprintf(fid,'%d\n',nv); % number of variables (nv)
for j=1:nv,
    fprintf(fid,'%.15e  %.15e  %.15e\n',xyz(j,1:3));
end
 fclose(fid);
