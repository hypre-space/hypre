function writevec(f,fname)
%-----------------------------------------------------
% function writevec(f,fname)
%    Writes vector f to filename
%
%    Format:
%    First line is 'nv', the length of the vector.
%    Next nv lines are the data
%------------------------------------------------------
nv = length(f);
fid=fopen(fname,'w');
fprintf(fid,'%d\n',nv); % number of variables (nv)
fprintf(fid,'%.15e\n',f);
 
fclose(fid);
