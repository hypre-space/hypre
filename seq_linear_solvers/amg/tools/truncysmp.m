function [newia,newja,newa]=truncysmp(newrows,newcols,ia,ja,a);
% function [newia,newja,newa]=truncysmp(newrows,newcols,ia,ja,a);
%
%          truncates the matrix given by ia, ja, and a
%          to be newrows x newcols.  Results are stored
%          in newia, newja, newa.  Use writeysmp
%          to put the truncated matrix out to a file.
%

newa=0*a;
newja=0*ja;
newia=ia(1:newrows+1);
iaind=1;
jaind=1;

for j=1:newrows,
    for k=ia(j):ia(j+1)-1;
        if ja(k) <= newcols,
           newa(jaind)=a(k);
           newja(jaind)=ja(k);
           jaind=jaind+1;
        else
           for l=j+1:newrows+1;
               newia(l)=newia(l)-1;
           end
        end
    end
end

newja = newja(1:newia(newrows+1)-1);
newa = newa(1:newia(newrows+1)-1);
