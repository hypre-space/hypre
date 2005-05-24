function matlab_smatrix = hypreIJ2matlab(matrix_filename,imax)
%HYPREIJ2MATLAB converts HYPRE IJ matrix to MATLAB format.
%
%   This program was tested on Matlab 7.1.
%
%   Description:
%     matlab_smatrix = hypreIJ2matlab(matrix_filename,imax) converts 
%     HYPRE formatted sparse square matrix to MATLAB formatted one.
%     This function takes as an input a HYPRE matrix specified by  
%     the given argument 'matrix_filename', having its size specified
%     by the argument 'imax' to a MATLAB formatted sparse square matrix
%     Hypre formatted file is represented as:
%     IMIN IMAX IMIN IMAX
%     i j value
%
%   Example:
%     1) create hypre formatted matrix:
%     matrix.00000, matrix.00001
%     specify the size of the matrix 
%     by inputing the argument imax
%     2) hypreIJ2matlab ('matrix', 10) 
%     (creates a sparse 10X10 matrix in matlab 
%     filled with elements from matrix.00000, matrix.00001
%
%   See also matlabIJ2hypre, matlab2hypreParVectors.m, 
%      testIJmatlabhypre.m, hypreParVectors2matlab.m
%
%   Author: Diana Zakaryan, Dept of Mathematics, University of Colorado,
%      Denver, 15-Mar-2005.
%


% read all the attributes of the specified filename
% in the specified directory
[stat,mess]=fileattrib(strcat(matrix_filename,'*')); 

% allocate memory for sparse matrix
matlab_smatrix=spalloc(imax,imax,1);

% fill the matrix
for i=1:size(mess,2)
    [pathstr,name,ext,versn]= fileparts(mess(i).Name);
    filename_temp=strcat(name, ext);
    hypre_data = dlmread(filename_temp,'',1,0);
    size_hypredata = size(hypre_data, 1);
    if size_hypredata==0
        continue;
    end
    matlab_smatrix = matlab_smatrix+...
        sparse(hypre_data(1:size_hypredata,1)+1,hypre_data(1:size_hypredata,2)+1,hypre_data(1:size_hypredata,3),imax,imax,floor(size_hypredata));
    clear ('hypre_data');
end




