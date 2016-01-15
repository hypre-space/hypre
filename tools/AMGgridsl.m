function [] = AMGgridsl(file,maxl,minl,numfiles)
%-------------------------------------------------------------------------
% AMGgridsl(file,maxl,minl,numf)
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

xmin = min (data(:,1));
xmax = max (data(:,1));
ymin = min (data(:,2));
ymax = max (data(:,2));

CF = data(:,3);
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

hh = (lmax)*(0.2)*h;
axis ([xmin-hh,xmax+hh,ymin-hh,ymax+hh]);
axis square;
box on;
xlabel('');
ylabel('');
set(gca, 'Visible', 'off');

for l = minl:maxl
  if (l==maxl)
     ind = find (CF >= l);
  else
    ind = find(CF == l);
  end
  sz = size(ind, 1);
  x = zeros(4,sz);
  y = zeros(4,sz);
  c = zeros(4,sz);
  ll = l / lmax;
  hh = (l+1)*(0.1)*h; 

  % plot the CM values
  for k = 1:sz
    %xx = grid(ind(k),1);
    %yy = grid(ind(k),2);
    xx  = data(ind(k),1);
    yy  = data(ind(k),2);
    pp = processor(ind(k)) / numfiles;

    x(:,k) = [xx-hh, xx+hh, xx+hh, xx-hh]';
    y(:,k) = [yy-hh, yy-hh, yy+hh, yy+hh]';
    %c(:,k) = [ll, ll, ll, ll]';
    c(:,k) = [pp, pp, pp, pp]';
  end
  patch(x, y, c);
end
shg;
