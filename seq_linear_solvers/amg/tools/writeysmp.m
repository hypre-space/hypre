function writeysmp(nv,ia,ja,a,fname)
%  function writeysmp(nv,ia,ja,a,fname)
%
%  writes ysmp file in 'fname' (no
%  qualifiers added, eg. not 'fname.ysmp')
%

outdat = [fname];
fid=fopen(outdat,'w');

fprintf(fid,'1 1\n'); % dummy row
fprintf(fid,'%d\n',nv); % number of variables (nv)
fprintf(fid,'%d\n',ia);
fprintf(fid,'%d\n',ja);
fprintf(fid,'%.15e\n',a);

fclose(fid);
