function [] = AMGgrids3(file,maxl,minl,numfiles)
%-------------------------------------------------------------------------
% AMGgrids3(file,maxl,minl,numf)
%
% 'file' - prefix string for input file '<file>'
% 'maxl' - the maximal level to plot
% 'minl' - the minimal level to plot
% 'numf' - the number of input files
%
% If numf is specified, the function expects that the files are named
% 'file'.xxxxx, where xxxxx is a five-digit number starting at 0.
%------------------------------------------------------------------------- 

% delete all figures
delete(get(0, 'Children'));

%------------------------------------------------------------
% Load CF and grid data
%------------------------------------------------------------

if (nargin==4)
  data = [];
  xcoord = [];
  ycoord = [];
  processor = [];
  for l = 0:numfiles-1
    filepart = sprintf ('%s.%.5d',file,l);
    datapart = load (filepart);
    data = [ data ; datapart ];
    processorpart = l*ones(size(datapart(:,1),1),1);
    processor = [ processor; processorpart ];
  end
else
  numfiles=1;
  data = load(file);
   processor = ones(size(data(:,1)),1);
end

CF = data(:,4);
n = size(CF,1);

% This should also be read from file for generality
m = n.^(1./3);
h = 1;

figure;
hold on;
%axis image;
%axis square;
%box on;
axis equal;
xlabel('');
ylabel('');
zlabel('');
%set(gca, 'Visible', 'off');
view(3);

% set the colormap
cmap = hot;
colormap(cmap);
caxis([0.0, 1.2]);
% handle = colorbar('vert');
% %set(handle, 'Ylim', [0, 1]);

lmax = max(CF);
if (nargin<2 || maxl<0) maxl=lmax; end
if (nargin<3 || minl<0 ) minl=0; end
maxl = min(maxl,lmax);
minl = max(minl,0);
for l = minl:maxl
  ind = find(CF == l);
  if (l==maxl)
     ind = find (CF >= l);
  else
    ind = find(CF == l);
  end
  sz = size(ind, 1);
  x = zeros(8);
  y = zeros(8);
  z = zeros(8);
  f = zeros (24);
  fvc = zeros (8);
  ll = l / lmax;
  hh = (l+1)*(0.1)*h; 

  % plot the CM values
  for k = 1:sz
    xx  = data(ind(k),1);
    yy  = data(ind(k),2);
    zz  = data(ind(k),3);

    x = [xx-hh, xx+hh, xx+hh, xx-hh, xx-hh, xx+hh, xx+hh, xx-hh];
    y = [yy-hh, yy-hh, yy+hh, yy+hh, yy-hh, yy-hh, yy+hh, yy+hh];
    z = [zz-hh, zz-hh, zz-hh, zz-hh, zz+hh, zz+hh, zz+hh, zz+hh];
    %c(:,k) = [ll, ll, ll, ll]';
    v = [x ; y ; z ]';
    f = [1, 2, 6, 5; 2, 3, 7, 6; 3, 4, 8, 7; 4, 1, 5, 8; 1, 2, 3, 4; 5, 6, 7, 8];

    %fvc = ll;
    fvc =  processor(ind(k)) / numfiles;
    patch('Vertices',v,'Faces',f, 'FaceVertexCData', fvc, 'Facecolor', 'flat');
  end
end
shg;
