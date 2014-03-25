c=====================================================================
c     vector copy routine.  performs   y <- x 
c=====================================================================

      subroutine vcopy(x,y,n)

      implicit real*8 (a-h,o-z)

      dimension x (*)
      dimension y (*)

      do 1 i = 1,n
         y(i) = x(i)
  1   continue

      return
      end
