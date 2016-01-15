function generatexy(nx, ny, filename)
%
%

% set up xy data
n = nx*ny;
xy = zeros(n, 2);
k = 1;
for j = 1 : ny
  for i = 1 : nx
    xy(k,:) = [i j];
    k = k + 1;
  end;
end;

% write out xy data
fid=fopen(filename, 'w');
fprintf(fid, '%e %e\n', xy');
fclose(fid);

