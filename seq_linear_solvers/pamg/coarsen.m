function coarsen(xyfilename)
%-----------------------------------------------------------------------------
% This function animates the coarsening process for PAMG
%-----------------------------------------------------------------------------

%---------------------------------------
% read in the xy data
%---------------------------------------

fid=fopen(xyfilename, 'r');
[xy, count] = fscanf(fid, '%e %e', [2 inf]);
xy = xy';
fclose(fid);

iter = 0;
while (1)

  %---------------------------------------
  % read measures
  %---------------------------------------

  filename = sprintf('coarsen.out.measures.%04d', iter);
  fid=fopen(filename, 'r');

  if (fid == -1)
    return;
  end

  [measures, count] = fscanf(fid, '%f ');
  fclose(fid);

  %---------------------------------------
  % set up the strength graph
  %---------------------------------------

  % read in initial strength graph
  filename = sprintf('coarsen.out.strength.%04d', iter);
  S = readysmp(filename);

  % set positive entries to 0
  S = S < 0;
  
  %---------------------------------------
  % set up coarse and fine points
  %---------------------------------------

  % read C/F marker array
  filename = sprintf('coarsen.out.CF.%04d', iter);
  fid=fopen(filename, 'r');
  [CFmarker, count] = fscanf(fid, '%d ');
  fclose(fid);

  % set up coarse and fine arrays
  cindex = 1;
  findex = 1;
  for i = 1 : count
    if (CFmarker(i) == 1)
      coarse(cindex) = i;
      cindex = cindex + 1;
    end
    if (CFmarker(i) == -1)
      fine(findex) = i;
      findex = findex + 1;
    end
  end

  %---------------------------------------
  % plot the current strength graph and
  % the C/F points
  %---------------------------------------

  % plot strength graph
  clf;
  gplot(S, xy);
  hold;

  % plot C-pts
  if (cindex > 1)
    handles = plot(xy(coarse,1), xy(coarse,2), 'rs');
    set(handles, 'MarkerSize', 5);
    set(handles, 'MarkerFaceColor', 'r');
  end

  % plot F-pts
  if (findex > 1)
    handles = plot(xy(fine,1), xy(fine,2), 'ks');
    set(handles, 'MarkerSize', 5);
    set(handles, 'MarkerFaceColor', 'k');
  end

  % plot measures
  strmeasures = num2str(measures, '%3.1f');
  handles = text(xy(:,1), xy(:,2), strmeasures);
  set(handles, 'FontSize', 8);
  set(handles, 'HorizontalAlignment', 'center');

  %---------------------------------------
  % pause until the user hits a key
  %---------------------------------------

  pause
  iter = iter + 1;

end

