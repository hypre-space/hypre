function [A] = testIJmatlabhypre (inputmatlab, matrix_or_vector, filename, varargin)
%TESTIJMATLABHYPRE Test IJ Martix Conversion between MATLAB and HYPRE Formats.
%
%    This program was tested on Matlab 7.1.
%
%   Description:
%     [A] = testIJmatlabhypre (inputmatlab, matrix_or_vector, filename,
%     varargin) calculates the accuracy  of MATLAB to HYPRE and HYPRE to 
%     MATLAB conversion of input matrix or vector.
%
%     To calculate the accuracy of conversion [A], this function converts 
%     matrix (or vector) bidirectionally between MATLAB and HYPRE formats 
%     and calculates the difference between the input matrix and the result
%     of bidirectional conversion of the input matrix.
%
%     The input format can be either MATLAB or HYPRE. If the input format
%     is MATLAB then the intermediate format is HYPRE. If the input format 
%     is HYPRE then the intermediate format is MATLAB.
%
%     The function performs two conversions: forward conversion from input 
%     format to intermediate format and backward conversion from 
%     intermediate format to input format. The result of such bidirectional
%     conversion is a matrix in the same format as the input matrix.
%
%     [A] = testIJmatlabhypre (inputmatlab, matrix_or_vector, filename,
%     numprocs) calculates the accuracy of MATLAB-HYPRE-MATLAB conversion 
%     performed on number of processors given by numprocs.
%
%     [A] = testIJmatlabhypre (inputmatlab, matrix_or_vector, filename, ,
%     imax) calculates the accuracy of HYPRE-MATLAB-HYPRE matrix conversion,
%     where imax is the size of a sparse square HYPRE matrix.
%
%   Inputs:
%     Inputmatlab = 'matlab' if input file is in MATLAB format
%                    'hypre' if input file is in HYPRE format
%     matrix_or_vector = 'matrix' if input file is a matrix
%                      = 'vector' if input file is a vector
%     filename = full file name of the input file
%
%   Example:
%     A = testIJmatlabhypre( 'matlab', 'matrix', 'mymatrix' )
%     will produce the difference [A] between input MATLAB matrix
%     and the result of MATLAB-HYPRE-MATLAB conversion of the input
%     matrix.
%
%     A = testIJmatlabhypre( 'hypre', 'vector', 'mymatrix' )
%     will produce the difference [A] between input HYPRE vector
%     and the result of HYPRE-MATLAB-HYPRE conversion of the input
%     vector.
%
%     A = testIJmatlabhypre( 'matlab', 'matrix', 'mymatrix', 2 )
%     will produce the difference [A] between input MATLAB matrix
%     and the result of MATLAB-HYPRE-MATLAB conversion of the input
%     matrix ran on two processors.
%
%     A = testIJmatlabhypre( 'hypre', 'matrix', 'mymatrix' )
%     will produce the difference [A] between input sparse square HYPRE 
%     matrix of size given by imax and the result of HYPRE-MATLAB-HYPRE
%     conversion of the input matrix.
%
%   See also matlabIJ2hypre, matlab2hypreParVectors.m, 
%      hypreIJ2matlab.m, hypreParVectors2matlab.m
%
%   Author: Diana Zakaryan, Dept of Mathematics, University of Colorado,
%      Denver, 15-Mar-2005.
%

if nargin > 3
    numprocs = varargin{1}
end

if nargin > 4
    imax = varargin{2}
end

if nargin > 5
    matrix_filename = varargin{3};
    vector_filename = varargin{4};
    A=hypreIJ2matlab(matrix_filename,imax);
    B=hypreParVectors2matlab(vector_filename,1);
    if size(A,2)~=size(B,1)
        error('Dimentions must agree')
    else
    size(A*B)
    end
    clear ('A', 'B');
end
    
    
    

% find out if the check is for matlab format to hypre format
if size(inputmatlab,2) == 6
    % find out if the check is for matrix
    if size(matrix_or_vector,2) == 6
        % code for matlab to hypre in matrix
        % todo add code to check if .mat file or ascii file
        load (filename);
        % converts a matlab formatted matrix to 
        % hypre formatted one
        
        matlab2hypreIJ (matlab_smatrix, numprocs, 'temp_m_hypre');
        imax = size(matlab_smatrix, 1);
        
        % converts a hypre formatted matrix to
        % matlab formatted one
        matlab_out_m = hypreIJ2matlab ('temp_m_hypre', imax);

        % compare input and output files;
        A1=matlab_smatrix - matlab_out_m
        fprintf('the min of the difference is: %\n', num2str(min(A1)));
        fprintf('the max of the difference is: %\n', num2str(max(A1)));
        clear('matlab_out_m');
        delete ('../temp_m_hypre.*');
    
    % find out if the check is for vector
    elseif size(matrix_or_vector,2) == 7
        %code for matlab to hypre in vector
        load (filename);
        
        % converts a matlab formatted vector to
        % hypre formatted one
        matlab2hypreParVectors (matlab_vector,numprocs,'temp_v_hypre');
        % converts a hypre formatted vector to
        % matlab formatted one
        matlab_out_v = hypreParVectors2matlab('temp_v_hypre',1);
      
        % compare input and output files;
        A2=matlab_vector - matlab_out_v;
        fprintf('the min of the difference is: %\n', num2str(min(A2)));
        fprintf('the max of the difference is: %\n', num2str(max(A2)));
        clear('matlab_out_v');
        delete ('../temp_v_hypre.*');
    else
        %return error message
    end

% find out if the check is for hypre format to matlab format
elseif size(inputmatlab,2) == 5
    % find out if the check is for matrix
    if size(matrix_or_vector,2) == 6
        % code for hypre to matlab in matrix
        % converts a hypre formatted matrix to 
        % matlab formatted one 
        A=hypreIJ2matlab('matrix',imax);
        % converts a matlab formatted matrix to 
        % hypre formatted one
        matlab2hypreIJ(A,numprocs,'matrix_test') %note that for this particular comparison 
        % numprocs should be initialized as 2.
        for i=0:numprocs
            s=int2str(i);
            hypre_data_m1=dlmread(strcat('matrix','.0000',s));
            hypre_data_m2=dlmread(strcat('matrix_test','.0000',s));
            % compare input and output files;
            B1=hypre_data_m1-hypre_data_m2
        end
        
    % find out if the check is for vector        
    elseif size(matrix_or_vector,2) == 7
        % converts a hypre formatted vector to 
        % matlab formatted one 
        A=hypreParVectors2matlab('vectors',1);
        % converts a matlab formatted vector to 
        % hypre formatted one
        matlab2hypreParVectors(A,numprocs,'vector_test');
        for i=0:(numprocs-1)
            s=int2str(i);
            hypre_data_v1 = dlmread( strcat('vectors','.0.',s));
            hypre_data_v2 = dlmread( strcat('vector_test','.0.',s));
            % compare input and output files;
            B2=hypre_data_v1-hypre_data_v2
        end
        for i=0:(numprocs-1)
            s=int2str(i);
            hypre_data_v1 = dlmread( strcat('vectors','.0','.INFO.',s));
            hypre_data_v2 = dlmread( strcat('vector_test','.0','.INFO.',s));
            % compare 'INFO' input and output files;
            B3=hypre_data_v1-hypre_data_v2
        end
    else
        %return error message
    end    
end    
