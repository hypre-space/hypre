function matlab2hypreIJ(matlab_smatrix,num_procs,matrix_filename,varargin)
%MATLAB2HYPREIJ converts MATLAB matrix to HYPRE IJ format.
%
%   This program was tested on Matlab 7.1.
%
%   Description:
%     matlab2hypreIJ(matlab_smatrix,num_procs,matrix_filename)
%     generates hypre input files for sparse matrix
%
%   Examples:
%     Create sparse matrix matlab_smatrix in Matlab, generate files and run on hypre
%     In Matlab:
%     1) create sparse matrix matlab_smatrix
%     2) matlab2hypreIJ(matlab_smatrix,2,'matrix')
%     (generates two files matrix.00000, matrix.00001)
%     Using Hypre:
%     3) mpirun -np 2 ij_es -lobpcg -fromfile matrix
%
%     Change floating point format in output files
%     matlab2hypreIJ(matlab_smatrix,2,'matrix','15.10e')
%     changes format of floating point numbers in file
%
%   See also testIJmatlabhypre.m, matlab2hypreParVectors.m, 
%      hypreIJ2matlab.m, hypreParVectors2matlab.m
%
%   Author: Merico E. Argentati, Dept of Mathematics, University of Colorado,
%      Denver.
%


% check for sparsity
if issparse(matlab_smatrix)==0
   error('The matrix must be sparse.')
end

% check if the matrix is square
if size(matlab_smatrix,2)~=size(matlab_smatrix,1)
    error('The matrix must be a square matrix.')
end

% default print format   
my_format='20.19e';
if (nargin>3)
    my_format=varargin(1);
end
prt_format=strcat('%d %d %',char(my_format),'\n');

% intialization
[n,n]=size(matlab_smatrix);
[hypre_data(:,1),hypre_data(:,2),hypre_data(:,3)]=find(matlab_smatrix);
hypre_data=sortrows(hypre_data);
nrows=size(hypre_data,1);
for i=1:nrows
    hypre_data(i,1)=hypre_data(i,1)-1;
    hypre_data(i,2)=hypre_data(i,2)-1;
end

% generate partitioning
part=generate_part(n,num_procs);

% generate Hypre input files
s1='00000';
index=1;
for i=1:num_procs
    s=int2str(i-1);
    ls=size(s,2);
    if ls<5
        s2=s1(1:(5-ls));    
    end
    filename2=strcat(matrix_filename,'.',s2,s);
     fprintf('Generating file: %s\n',filename2);
    fid = fopen(filename2,'w');
    
    % get first and last row index
    index1=index;
    while (index <= nrows) && (hypre_data(index,1)<part(i+1))
        index=index+1;
    end
    index2=index-1;
    ilower=min(hypre_data(index1:index2,1));
    iupper=max(hypre_data(index1:index2,1));
    jlower=ilower;
    jupper=iupper;
    fprintf(fid,'%d %d %d %d\n',ilower,iupper,jlower,jupper);
    fprintf(fid,prt_format,hypre_data(index1:index2,:,:)');
    fclose(fid);
end
return

function [partitioning]=generate_part(length,num_procs)
% [partitioning]=generate_part(length,num_procs)
% generate partitioning across processes
% See Hypre getpart.c
partitioning=zeros(num_procs+1,1);
size = floor(length/num_procs);
rest= length - size*num_procs;
for i=1:num_procs-1
    partitioning(i+1) = partitioning(i)+size;
    if (i==1)
        partitioning(i+1)=partitioning(i+1)+rest;
    end
end
partitioning(num_procs+1)=length;
return

