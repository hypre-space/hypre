%function A = lapmake(Nx,Ny)
%
% Makes standard 5-pt Laplacian matrix on grid with with Nx intervals
% in x direction and Ny intervals in y direction

function A = lapmake(Nx,Ny)

dx=1/Nx;
dx2=dx*dx;
dy=1/Ny;
dy2=dy*dy;
dig=2/dx2+2/dy2;
xoff=-1/dx2;
yoff=-1/dy2;
Nx1=Nx-1;


C = toeplitz([dig xoff zeros(1,Nx-3)]);
B = yoff*eye(Nx1);

A = [C B zeros(Nx-1,(Ny-3)*Nx1)];

for j=2:Ny-2,

    A = [A;[zeros(Nx1,(j-2)*Nx1) B C B zeros(Nx1,(Ny-4-(j-2))*Nx1)]];
end

A = [A; zeros(Nx-1,(Ny-3)*Nx1) B C];
