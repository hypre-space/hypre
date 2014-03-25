
function [y] = matvec(x,y,a,ja,ia,nv,alpha,beta)
% function [y] = matvec(x,y,a,ja,ia,nv,alpha,beta)
%
%   Computes matrix-vector product alpha*A*x + beta*y
%
%   A is the ysmp matrix a,ja,ia
%   nv is the number of rows in A
%

if alpha == 0,
   for i=1:nv,
       y(i) = beta*y(i);
   end
else
   temp = beta/alpha;
   if temp ~= 1,
      if temp == 0,
         for i = 1:nv,
             y(i) = 0;
         end
      else 
         for i = 1:nv,
             y(i) = temp*y(i);
         end
      end
   end
   for i = 1:nv,
       for j = ia(i):ia(i+1)-1;
           y(i) = y(i)+a(j)*x(ja(j));
       end
   end
   if alpha ~= 1,
      for i=1:nv,
          y(i) = alpha*y(i);
      end
   end
end
