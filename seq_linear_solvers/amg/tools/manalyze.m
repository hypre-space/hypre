function [nv,ia,ja,a] = manalyze(A);
%-----------------------------------------
%function [nv,ia,ja,a] = manalyze(A);
%   Analyzes matrix A, yields ysmp format
%-----------------------------------------

% determine row/column indices of nonzero entries
[ja, I, a] = find(A');

nv = size(A, 1);
na = size(a, 1);

% compute ia and put diagonal entry first on each row
ia = zeros(nv+1, 1);
ia(1) = 1;
for i = 1 : nv
   k = ia(i);
   while (k <= na)
      if (I(k) ~= i)
         break;
      end

      % swap diagonal entry with first entry
      if (ja(k) == i)
         tmp = a(ia(i));
         a(ia(i)) = a(k);
         a(k) = tmp;
         tmp = ja(ia(i));
         ja(ia(i)) = ja(k);
         ja(k) = tmp;
      end

      k = k + 1;
   end
   ia(i+1) = k;
end

