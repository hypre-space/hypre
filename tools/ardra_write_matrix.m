% Driver to write ardra matrices to parallel smg input file
% This driver can be used as a template for people to modify.  Once the
% matrix A is loaded into matlab, and the grid, dim, symmetric and stencil
% information is defined, a call to write_matrix writes the matrix to the
% output file.
% 
% Author: Peter Brown, 12-17-97
% 12-18-97  PNB  Added capability to write out vectors
% 12-30-97  PNB  Changed to write out symmetric or nonsymmetric stencil data

% Set problem
iprob = 1;
if (iprob == 1),
  imin = 1; jmin = 1; kmin = 1;
  imax = 9; jmax = 9; kmax = 9;
elseif (iprob == 2),
  imin =  1; jmin =  1; kmin =  1;
  imax = 15; jmax = 15; kmax = 17;
elseif (iprob == 3),
  imin =  1; jmin =  1; kmin =  1;
  imax = 29; jmax = 29; kmax = 33;
end

% Set up dim, symmetric, and stencil information based.
symmetric = 1;
dim = 3;
if (symmetric == 0),
  stencil = zeros(27,3);
   stencil(1,1) =-1;  stencil(1,2) =-1;  stencil(1,3) = 0;
   stencil(2,1) = 0;  stencil(2,2) =-1;  stencil(2,3) = 0;
   stencil(3,1) = 1;  stencil(3,2) =-1;  stencil(3,3) = 0;
   stencil(4,1) =-1;  stencil(4,2) = 0;  stencil(4,3) = 0;
   stencil(5,1) = 0;  stencil(5,2) = 0;  stencil(5,3) = 0;
   stencil(6,1) = 1;  stencil(6,2) = 0;  stencil(6,3) = 0;
   stencil(7,1) =-1;  stencil(7,2) = 1;  stencil(7,3) = 0;
   stencil(8,1) = 0;  stencil(8,2) = 1;  stencil(8,3) = 0;
   stencil(9,1) = 1;  stencil(9,2) = 1;  stencil(9,3) = 0;
  stencil(10,1) =-1; stencil(10,2) =-1; stencil(10,3) =-1;
  stencil(11,1) = 0; stencil(11,2) =-1; stencil(11,3) =-1;
  stencil(12,1) = 1; stencil(12,2) =-1; stencil(12,3) =-1;
  stencil(13,1) =-1; stencil(13,2) = 0; stencil(13,3) =-1;
  stencil(14,1) = 0; stencil(14,2) = 0; stencil(14,3) =-1;
  stencil(15,1) = 1; stencil(15,2) = 0; stencil(15,3) =-1;
  stencil(16,1) =-1; stencil(16,2) = 1; stencil(16,3) =-1;
  stencil(17,1) = 0; stencil(17,2) = 1; stencil(17,3) =-1;
  stencil(18,1) = 1; stencil(18,2) = 1; stencil(18,3) =-1;
  stencil(19,1) =-1; stencil(19,2) =-1; stencil(19,3) = 1;
  stencil(20,1) = 0; stencil(20,2) =-1; stencil(20,3) = 1;
  stencil(21,1) = 1; stencil(21,2) =-1; stencil(21,3) = 1;
  stencil(22,1) =-1; stencil(22,2) = 0; stencil(22,3) = 1;
  stencil(23,1) = 0; stencil(23,2) = 0; stencil(23,3) = 1;
  stencil(24,1) = 1; stencil(24,2) = 0; stencil(24,3) = 1;
  stencil(25,1) =-1; stencil(25,2) = 1; stencil(25,3) = 1;
  stencil(26,1) = 0; stencil(26,2) = 1; stencil(26,3) = 1;
  stencil(27,1) = 1; stencil(27,2) = 1; stencil(27,3) = 1;
else
  stencil = zeros(14,3);
   stencil(1,1) =-1;  stencil(1,2) =-1;  stencil(1,3) = 0;
   stencil(2,1) = 0;  stencil(2,2) =-1;  stencil(2,3) = 0;
   stencil(3,1) = 1;  stencil(3,2) =-1;  stencil(3,3) = 0;
   stencil(4,1) =-1;  stencil(4,2) = 0;  stencil(4,3) = 0;
   stencil(5,1) = 0;  stencil(5,2) = 0;  stencil(5,3) = 0;
   stencil(6,1) =-1;  stencil(6,2) =-1;  stencil(6,3) =-1;
   stencil(7,1) = 0;  stencil(7,2) =-1;  stencil(7,3) =-1;
   stencil(8,1) = 1;  stencil(8,2) =-1;  stencil(8,3) =-1;
   stencil(9,1) =-1;  stencil(9,2) = 0;  stencil(9,3) =-1;
  stencil(10,1) = 0; stencil(10,2) = 0; stencil(10,3) =-1;
  stencil(11,1) = 1; stencil(11,2) = 0; stencil(11,3) =-1;
  stencil(12,1) =-1; stencil(12,2) = 1; stencil(12,3) =-1;
  stencil(13,1) = 0; stencil(13,2) = 1; stencil(13,3) =-1;
  stencil(14,1) = 1; stencil(14,2) = 1; stencil(14,3) =-1;
end

% Load grid values
grid = [imin,jmin,kmin,imax,jmax,kmax];
nx = imax - imin + 1;
ny = jmax - jmin + 1;
nz = kmax - kmin + 1;

% Load stencil_size
[n,m]=size(stencil);
stencil_size = n;

% Load matrix from file
A = spalloc(nx*ny*nz,nx*ny*nz, stencil_size*nx*ny*nz);
if (iprob == 1),
  load dco_8.dat;
  A = sparse(dco_8(:,1),dco_8(:,2),dco_8(:,3),nx*ny*nz,nx*ny*nz);
elseif (iprob == 2),
  load dco_14.dat;
  A = sparse(dco_14(:,1),dco_14(:,2),dco_14(:,3),nx*ny*nz,nx*ny*nz);
elseif (iprob == 3),
  load dco_28.dat;
  A = sparse(dco_28(:,1),dco_28(:,2),dco_28(:,3),nx*ny*nz,nx*ny*nz);
end

% Modify matrix to make it nonsingular
A = A + 0.001*speye(size(A));

% Open output file for matrix
fid = fopen('ardra_matrix.out', 'wt');
if (fid == -1),
  printf('Error: cannot open input file %s\n', filename);
  return;
end

% Write the matrix to the file.
if (symmetric == 0),
  ierr = write_matrix(fid,A,symmetric,dim,grid,stencil);
else
  % Write only the lower triangular part plus the main diagonal.
  L = tril(A);
  ierr = write_matrix(fid,L,symmetric,dim,grid,stencil);
end

% Open output file for vector
fid = fopen('ardra_vector_in.out', 'wt');
if (fid == -1),
  printf('Error: cannot open input file %s\n', filename);
  return;
end

% Load input vector.
v = ones(nx*ny*nz,1);

% Write the vector to the file.
ierr = write_vector(fid,v,dim,grid);

% Calculate output vector
w = A*v; % Matrix vector product
% w = A\v; % Matrix solve

% Open output file for vector
fid = fopen('ardra_vector_out.out', 'wt');
if (fid == -1),
  printf('Error: cannot open input file %s\n', filename);
  return;
end

% Write the vector to the file.
ierr = write_vector(fid,w,dim,grid);
