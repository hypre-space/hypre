function L = lapmake9(Nx,Ny,stncl)
%function L = lapmake9(Nx,Ny,stncl)
%
% Makes Laplacian matrix based on 9-pt input stencil (3x3 matrix)
% using Nx nodes in x-direction and Ny nodes in y-direction 
%
% Form is    | B C          |
%            | A B C        |
%            |   A B C      |
%            |       . . .  |
%            |         A B C|
%            |           A B|
%
% where A, B, and C are tridiagonal matrices.


B = diag(stncl(2,2)*ones(Nx,1));
B = B + diag(stncl(2,3)*ones(Nx-1,1),1);
B = B + diag(stncl(2,1)*ones(Nx-1,1),-1);

C = diag(stncl(1,2)*ones(Nx,1));
C = C + diag(stncl(1,3)*ones(Nx-1,1),1);
C = C + diag(stncl(1,1)*ones(Nx-1,1),-1);

A = diag(stncl(3,2)*ones(Nx,1));
A = A + diag(stncl(3,3)*ones(Nx-1,1),1);
A = A + diag(stncl(3,1)*ones(Nx-1,1),-1);


L = [B C zeros(Nx,(Ny-2)*Nx)];

for j=2:Ny-1,

    L = [L;[zeros(Nx,(j-2)*Nx) A B C zeros(Nx,(Ny-3-(j-2))*Nx)]];
end

L = [L; zeros(Nx,(Ny-2)*Nx) A B];
