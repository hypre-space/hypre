function [nv,ia,ja,a] = manalyze(A);
%-----------------------------------------
%function [nv,ia,ja,a] = manalyze(A);
%   Analyzes matrix A, yields ysmp format
%-----------------------------------------

% determine row/column indices of nonzero entries
% add to diagonal while determining indices to
% ensure that diagonal will be present even if 0

d = diag(A);
mn = min(d);
shf=mn+1;
[ja, I] = find(A'+shf*speye(size(A)));
t = find(A'+shf*speye(size(A)));
AA=A';
a=full(AA(t));
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

