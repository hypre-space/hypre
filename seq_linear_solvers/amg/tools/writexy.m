function writexy(xy,fname)
%-----------------------------------------------------
% function writexy(xy,fname)
%    Writes xy data to filename
%
%    Format:
%    First line is the dummy line '1 1'
%    Second line is 'nv', the length of the vector.
%    Next nv lines are the data, x(i), y(i), 
%------------------------------------------------------
[nv,nw]=size(xy);
fid=fopen(fname,'w');
fprintf(fid,'1 1\n'); % dummy row
fprintf(fid,'%d\n',nv); % number of variables (nv)
for j=1:nv,
    fprintf(fid,'%.15e  %.15e\n',xy(j,1:2));
end
 fclose(fid);
