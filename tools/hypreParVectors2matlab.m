function matlab_vectors = hypreParVectors2matlab(filename,num_of_vectors)
%HYPREPARVECTORS2MATLAB Converts Hypre IJ Vectors to Matlab.
%
%   This program was tested on Matlab 7.1.
%
%   Description:
%     [matlab_vectors] = hypreParVectors2matlab(filename, num_of_vectors)
%     converts hypre formatted vectors, specified by the argument 'filename'
%     to matlab formatted vectors. Number of vectors to be converted is 
%     given by the argument num_of_vectors
%
%   Inputs:
%     filename = name of files without suffix
%     num_of_vectors = number of files containing vector description
%
%    Example:
%      [matlab_vectors] = hypreParVectors2matlab( 'vectors', 5 ) converts
%      HYPRE vector represented in 5 input HYPRE files to MATLAB format.
%      HYPRE formatted vectors (filename is 'vectors', for 2 CPUs) are 
%        represented as follows:
%          vectors.0.0  vectors.0.1  vectors.INFO.0.0  vectors.INFO.0.1
%          vectors.1.0  vectors.1.1  vectors.INFO.1.0  vectors.INFO.1.1
%          vectors.2.0  vectors.2.1  vectors.INFO.2.0  vectors.INFO.2.1
%          ...
%      The first line in vectors.0.0 and vectors.0.1 represents the 
%        number of elements in that particular file
%        vectors.INFO.0.0 and vectors.INFO.0.1 contain information
%        about the size of the vector and the size of the particular file
%      For these two HYPRE formatted vectors 
%        vectors.0.0  vectors.0.1  vectors.INFO.0.0  vectors.INFO.0.1
%        vectors.1.0  vectors.1.1  vectors.INFO.1.0  vectors.INFO.1.1
%        vectors.2.0  vectors.2.1  vectors.INFO.2.0  vectors.INFO.2.1
%        ...
%        the program will create a (size_of_vector)-by-2 matrices in matlab.
%
%   See also matlabIJ2hypre, matlab2hypreParVectors.m, 
%      hypreIJ2matlab.m, testIJmatlabhypre.m
%
%   Author: Diana Zakaryan, Dept of Mathematics, University of Colorado,
%      Denver, 15-Mar-2005.
%

% empty vector 
matlab_vectors = [];

% calls the actual program num_of_vectors time
% in order to convert all num_of_vectors hypre vectors to matlab format
for j=1:num_of_vectors
    s=int2str(j-1);
    filename2=strcat(filename,'.',s);
    % calls single vector convertor
    matlab_vector = hypreParVector2matlab(filename2);
    matlab_vectors = cat(2,matlab_vectors,matlab_vector);
end

    
function matlab_vector = hypreParVector2matlab(filename2);

%read all the attributes of the specified filename
%in the specified directory
[stat,mess]=fileattrib(strcat(filename2,'.*')); 
%[pathstr,name,ext,versn]= fileparts(mess(1).Name);
%empty vector 
matlab_vector=[];
for i=1:size(mess,2)
    [pathstr,name,extension,versn]= fileparts(mess(i).Name);
    k = findstr('.INFO',name);
    if size(k) == 1
        continue;
    end
    % fills in with data
    filename_temp=strcat(name,extension);
    hypre_data = dlmread(filename_temp,'',1,0);
    if i==1
        matlab_vector = hypre_data;
    else
        matlab_vector=cat(1,matlab_vector,hypre_data);
    end
    clear ('hypre_data');
end


