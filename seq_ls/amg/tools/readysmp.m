function [A] = readysmp(filename)
%-----------------------------------------------------------------------------
% [A] = readysmp('filename'):
%   Reads from file 'filename' a matrix A in YSMP format.
%
%   YSMP format:
%   First line is 'nv' the number of rows in matrix.  Integer.
%   Next 'nv+1' lines are ia(1:nv+1).  ia(j) points to the location
%   in 'a' and 'ja' where the first entry for row 'j' lives.  ia(nv+1)
%   points to the element one greater than length(a). Integers.
%   Next 'ia(nv+1)-1' lines contain 'ja' the list of non-zero columns.
%   The numbers [ja(ia(j)):ja(ia(j+1)-1)] contain the nonzero columns in
%   row 'j', with 'j', the diagonal column, listed first.  That is,
%   ja(ia(j))=j.  Integers.
%   Next 'ia(nv+1)-1' lines contain 'a'  values.
%   The numbers [a(ia(j)):a(ia(j+1)-1)] contain the nonzero values in
%   row 'j', corresponding to columns [ja(ia(j)):ja(ia(j+1)-1)].  Reals.
%
%-----------------------------------------------------------------------------

fid=fopen(filename,'r');

% read junk line
junk = fscanf(fid,'%d',2);

nv = fscanf(fid,'%d',1); % number of variables (nv)
[ia, count] = fscanf(fid,'%d ',nv+1);
[ja, count] = fscanf(fid,'%d ',ia(nv+1)-1); % This is ja
[a, count]=fscanf(fid,'%f ',ia(nv+1)-1); % This is the matrix a
fclose(fid);

kz=zeros(length(ja),1);
p=1;
for j=1:nv,
      plen = ia(j+1)-ia(j);
      kz(p:p+plen-1)=j*ones(plen,1);
      p=p+plen;
end

A = sparse(kz,ja,a);

