function [A,symmetric,dim,grid,stencil] = readstruct(filename);
% --------------------------------------------------------------------------
% [A,symmetric,dim,grid,stencil] = readstruct('filename')
%   Reads from file 'filename' a matrix A in Struct format
% --------------------------------------------------------------------------

fid=fopen(filename,'r');

% --------------------------------------------------------------
% Read in the matrix Symmetric, Grid and Stencil information.
% --------------------------------------------------------------

s = fscanf(fid, '%s',1);               % read "StructMatrix"

% read symmetric info
s = fscanf(fid,'%s',1);                % read "Symmetric:"
symmetric = fscanf(fid,'%d',1);        % read symmetric

% read constant-coefficient info
s = fscanf(fid,'%s',1);                % read "ConstantCoefficient:"
constcoeff = fscanf(fid,'%d',1);       % read constcoeff

% read grid info
s = fscanf(fid, '%s',1);               % read "Grid:"
dim = int64(fscanf(fid, '%d', 1));     % read dim
d = fscanf(fid, '%d', 1);              % read num boxes
grid = int64(zeros(2,dim));
s = fscanf(fid,'%s',1);                % read "0:"
s = fscanf(fid,'%1s',1);               % read "("
for d = 1:dim
  grid(1,d) = fscanf(fid,'%d',1);      % read imin index
  s = fscanf(fid,'%1s',1);             % read "," or ")"
end
s = fscanf(fid,'%1s',1);               % read "x"
s = fscanf(fid,'%1s',1);               % read "("
for d = 1:dim
  grid(2,d) = fscanf(fid,'%d',1);      % read imax index
  s = fscanf(fid,'%1s',1);             % read "," or ")"
end

% read periodic info
s = fscanf(fid, '%s',1);               % read "Periodic:"
periodic = int64(zeros(1,dim));
for d = 1:dim
  periodic(d) = fscanf(fid,'%d',1);    % read periodic value
end

% read range and domain strides
ranstride = int64(zeros(1,dim));
domstride = int64(zeros(1,dim));
s = fscanf(fid, '%s',1);               % read "Range"
s = fscanf(fid, '%s',1);               % read "Stride:"
for d = 1:dim
  ranstride(d) = fscanf(fid,'%d',1);   % read ranstride
end
s = fscanf(fid, '%s',1);               % read "Domain"
s = fscanf(fid, '%s',1);               % read "Stride:"
for d = 1:dim
  domstride(d) = fscanf(fid,'%d',1);   % read domstride
end

% read stencil info
s = fscanf(fid,'%s',1);                   % read "Stencil:"
stencil_size = int64(fscanf(fid,'%d',1)); % read stencil_size
stencil = int64(zeros(stencil_size,dim));
for i = 1:stencil_size,
  s = fscanf(fid,'%s',1);              % read "i:"
  for d = 1:dim
    stencil(i,d) = fscanf(fid,'%d',1); % read stencil offset
  end
end

% read matrix coefficients (currently only written for up to 5 dimensions)
s = fscanf(fid,'%s',1);                % read "Constant"
s = fscanf(fid,'%s',1);                % read "Data:"
s = fscanf(fid,'%s',1);                % read "Data:"
if (dim == 1)
  [Fdata, count]  = fscanf(fid,'%d: (%d; %d) %e', [4, inf]);
elseif (dim == 2)
  [Fdata, count]  = fscanf(fid,'%d: (%d, %d; %d) %e', [5, inf]);
elseif (dim == 3)
  [Fdata, count]  = fscanf(fid,'%d: (%d, %d, %d; %d) %e', [6, inf]);
elseif (dim == 4)
  [Fdata, count]  = fscanf(fid,'%d: (%d, %d, %d, %d; %d) %e', [7, inf]);
elseif (dim == 5)
  [Fdata, count]  = fscanf(fid,'%d: (%d, %d, %d, %d, %d; %d) %e', [8, inf]);
else
  error('Currently only support up to 5-dimensional grids');
end
fclose(fid);

% Set up the matrix A

% Compute igrid and jgrid
igrid = int64(zeros(1,dim));
jgrid = int64(zeros(1,dim));
for d = 1:dim
  igrid(1,d) = idivide(grid(1,d),ranstride(d),'ceil');
  igrid(2,d) = idivide(grid(2,d),ranstride(d),'floor');
  jgrid(1,d) = idivide(grid(1,d),domstride(d),'ceil');
  jgrid(2,d) = idivide(grid(2,d),domstride(d),'floor');
end

% Compute strides for mapping from grid points to ranks
istride = int64(ones(1,dim));
jstride = int64(ones(1,dim));
for d = 1:(dim-1)
  istride(d+1) = istride(d) * (igrid(2,d) - igrid(1,d) + 1);
  jstride(d+1) = jstride(d) * (jgrid(2,d) - jgrid(1,d) + 1);
end

ipoint = int64(zeros(1,dim));
jpoint = int64(zeros(1,dim));
Adata = ones(3,size(Fdata,2));
ii = 1;
for i = 1:size(Fdata,2)
  % set s to the stencil entry number
  s = Fdata(dim+2,i) + 1;
  % compute ipoint and jpoint
  for d = 1:dim
    ipoint(d) = Fdata(d+1,i);
    jpoint(d) = ipoint(d) + stencil(s,d);
    % map points and shift to an origin of zero
    ipoint(d) = idivide(ipoint(d),ranstride(d),'floor') - igrid(1,d);
    jpoint(d) = idivide(jpoint(d),domstride(d),'floor') - jgrid(1,d);
    if (periodic(d) > 0)
      jperiodic = idivide(periodic(d),domstride(d),'floor');
      jwrap = mod( (jpoint(d) + jperiodic), jperiodic );
      jpoint(d) = jwrap;
    end
  end

  % check to see if jpoint is in the grid
  ingrid = 1;
  for d = 1:dim
    if ( (jpoint(d) < grid(1,d)) | (jpoint(d) > grid(2,d)) )
       ingrid = 0;
    end
  end

  if (ingrid > 0)
    for d = 1:dim
      Adata(1,ii) = Adata(1,ii) + ipoint(d)*istride(d);
      Adata(2,ii) = Adata(2,ii) + jpoint(d)*jstride(d);
    end
    Adata(3,ii) = Fdata(dim+3,i);
    ii = ii + 1;
  end
end
Adata = Adata(:,1:(ii-1));

A = sparse(Adata(1,:)', Adata(2,:)', Adata(3,:)');

if (symmetric),
  D = diag(diag(A));
  A = A + A' - D;
end