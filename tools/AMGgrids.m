function [] = AMGgrids(file)
%-------------------------------------------------------------------------
% AMGgrids(fileprefix)
%
% 'fileprefix' - string prefix for input file '<fileprefix>.CF.dat'
%------------------------------------------------------------------------- 

% delete all figures
delete(get(0, 'Children'));

%------------------------------------------------------------
% Load CF and grid data
%------------------------------------------------------------

file_CF = strcat(file, '.CF.dat');
CF = load(file_CF);

n = size(CF,1);

% This should also be read from file for generality
m = sqrt(n);
grid = zeros(n,2);
k = 1;
for j = 1:m
  for i = 1:m
    grid(k,:) = [i,j];
    k = k + 1;
  end
end 
h = 1;

figure;
hold on;
axis image;
box on;
xlabel('');
ylabel('');
set(gca, 'Visible', 'off');

% set the colormap
cmap = hot;
colormap(cmap);
caxis([0.0, 1.2]);
% handle = colorbar('vert');
% %set(handle, 'Ylim', [0, 1]);

lmax = max(CF);
for l = 0:lmax
  ind = find(CF == l);
  sz = size(ind, 1);
  x = zeros(4,sz);
  y = zeros(4,sz);
  c = zeros(4,sz);
  ll = l / lmax;
  hh = (l+1)*(0.1)*h;

  % plot the CM values
  for k = 1:sz
    xx = grid(ind(k),1);
    yy = grid(ind(k),2);
    x(:,k) = [xx-hh, xx+hh, xx+hh, xx-hh]';
    y(:,k) = [yy-hh, yy-hh, yy+hh, yy+hh]';
    c(:,k) = [ll, ll, ll, ll]';
  end
  patch(x, y, c);
end
shg;
