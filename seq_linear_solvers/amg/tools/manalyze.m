function [nv,ia,ja,a] = manalyze(A);
%-----------------------------------------
%function [nv,ia,ja,a] = manalyze(A);
%   Analyzes matrix A, yields ysmp format
%
%   A must be a matrix in "full" form
%-----------------------------------------
[nv, cols]=size(A);
ia=zeros(nv+1,1);
ip = zeros(nv,1);
iu = ip;
iv = ia;
u = rand(nv,1);
f = ip;

disp('analyzing matrix')
iaend=1;
a=[];
ja=[];
for k=1:nv,
    if rem(k,100) == 0, disp(['Row ' num2str(k)]);end
    ia(k)=iaend;
    nzind = find(A(k,:)~=0);
    numnz = length(nzind);
    iaend = iaend+numnz;
    ctr = find(nzind==k);
    lft = nzind(1:ctr-1);
    rowind = [nzind(ctr:numnz) lft];
    ja = [ja rowind];
    a = [a A(k,rowind)];
end
ia(nv+1)=iaend;
ja=ja';
a=a';


   

nu = 1;         % there is but one function
np = nv;      % number of points equals number vars
iu = iu+1;      % (all variables are unknown number 1)
ip = [1:nv]'; % each variable is at it's own point
iv = [1:nv+1]'; % each variable is at it's own point
