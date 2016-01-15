function [nv,ia,ja,a] = writeysmp(filename,A,nv,ia,ja,a)
%  function [nv,ia,ja,a] = writeysmp(filename,A,nv,ia,ja,a)
%
%  writes ysmp file in 'filename' (no qualifiers added,
%  eg. not 'filename.ysmp')
%
%  arguments nv, ia, ja, and a are optional.
%

if nargin < 6
   [nv,ia,ja,a] = manalyze(A);
end;

outdat = [filename];
fid=fopen(outdat,'w');

fprintf(fid,'%d\n',nv); % number of variables (nv)
fprintf(fid,'%d\n',ia);
fprintf(fid,'%d\n',ja);
fprintf(fid,'%.15e\n',a);

fclose(fid);
