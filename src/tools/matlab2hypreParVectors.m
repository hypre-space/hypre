function matlab2hypreParVectors(matlab_mvector,num_procs,vector_filename,varargin)
%MATLAB2HYPREPARVECTORS Converts MATLAB Vectors to HYPRE IJ Format.
%
%   This program was tested on Matlab 7.1.
%
%   Description:
%     matlab2hypreParVectors(matlab_mvector,num_procs) converts MATLAB 
%     formatted vectors to HYPRE formatted ones, and writes the files to 
%     the disk. This function takes as an input an NxM matlab matrix 
%     matlab_mvector, where M represents the number of vectors and N stands
%     for the size of each vector and writes the matrix to the number of 
%     files specified by giving num_procs argument
%
%     matlab2hypreParVectors(matlab_mvector,num_procs,vector_filename)
%     writes the HYPRE formatted vectors to the output file named 
%     vector_filename e.g. if vector_filename is 'vectors' the program 
%     will generate first 4 files and the output files will be:
%       vectors.0.0, vectors.0.1, vectors.0.INFO.0, vectors.0.INFO.1.
%     the Mth output will be, consequently, 
%       vectors.m.0
%       vectors.m.1
%       vectors.m.INFO.0
%       vectors.m.INFO.1.
%
%     matlab2hypreParVectors(matlab_mvector,num_procs,vector_filename,
%       precision) converts from MATLAB to HYPRE with specified precision
%       given by floating point precision.
%
%     Examples:
%       matlab2hypreParVectors(A,2,'vectors')
%       1) Create a matrix A in matlab, consisting of M vectors of size 
%         N to be converted. 
%       2)Set num_procs to 2. (Specifies the number of processors) 
%       3)'vectors' specifies the hypre output filenames:
%         vectors.0.0
%         vectors.0.1
%         vectors.0.INFO.0
%         vectors.0.INFO.1
%
%       matlab2hypreParVectors(A,2,'vectors','15.10e')
%       changes format of floating point numbers in file
%
%   See also matlabIJ2hypre, testIJmatlabhypre.m, 
%      hypreIJ2matlab.m, hypreParVectors2matlab.m
%
%   Author: Diana Zakaryan, Dept of Mathematics, University of Colorado,
%      Denver, 15-Mar-2005.
%

for j=1:size(matlab_mvector,2)
    % create filename
    filename = strcat(vector_filename, '.', num2str(j-1));
    B=matlab_mvector(:,j);

    % call single vector convertor
    matlab2hypreParVector(B,num_procs,filename,varargin);
end

function matlab2hypreParVector(matlab_mvector,num_procs,filename,varargin)

% check 
if isvector(matlab_mvector)~=1
    error('The argument must be a vector.')
end

% default print format
my_format='20.19e';
if (nargin>4)
    my_format=varargin(1);
end
prt_format=strcat('%',char(my_format));

% intialization
n=size(matlab_mvector,1);

% generate partitioning
part=generate_part(n,num_procs);

% generate Hypre input files
for i=1:num_procs
    s=int2str(i-1);
    filename2=strcat(filename,'.',s);
    fprintf('Generating file: %s\n',filename2);
    X=matlab_mvector(part(i)+1:part(i+1));
    nrows=part(i+1)-part(i);
    dlmwrite(filename2, nrows, 'precision', '%d');
    dlmwrite(filename2, X, '-append','precision',prt_format);
    
    % writing INFO file
    filename2=strcat(filename,'.','INFO','.',s);
    fprintf('Generating INFO file: %s\n',filename2);
    Y = [n;0;nrows];
    dlmwrite(filename2, Y, 'precision', '%d');
    clear ('X', 'Y');
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



