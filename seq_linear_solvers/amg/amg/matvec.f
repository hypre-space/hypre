c=====================================================================
c     matvec routine:
c     compute y = alpha*A*x + beta*y
c
c     if jtrans==1, computes y=alpha*A^T*x + beta*y (A transpose)
c=====================================================================

      subroutine matvec(nv, alpha, a, ia, ja, x, beta, y, jtrans)

      implicit real*8 (a-h,o-z)

      dimension x  (*)
      dimension y  (*)
      dimension a  (*)
      dimension ia (*)
      dimension ja (*)


c-----------------------------------------------------------------------
c Do (alpha == 0.0) computation 
c-----------------------------------------------------------------------

      if (alpha .eq. 0.0) then
         do 20 i=1,nv
            y(i) = beta * y(i)
 20      continue
         return
      endif

c-----------------------------------------------------------------------
c      y = (beta/alpha)*y
c-----------------------------------------------------------------------

      temp = beta / alpha 
      if (temp .ne. 1.0) then
         if (temp .eq. 0.0) then
            do 30 i = 1,nv
               y(i) = 0.0
 30         continue
         else
            do 40 i = 1,nv
               y(i) = y(i) * temp
 40         continue
         endif
      endif
 
c-----------------------------------------------------------------------
c      y = y + A*x   or y = y + A^T * x
c-----------------------------------------------------------------------

      if (jtrans .ne. 1) then
         do 60 i = 1,nv
            jlo =  ia(i)
            jhi =  ia(i+1)-1
            if (jlo .gt. jhi) go to 60
            do 50 j = jlo,jhi
               if (ja(j) .lt. 0) go to 50
               y(i) = y(i) + a(j) * x(ja(j))
 50         continue
 60      continue
          else
             do 80 i = 1,nv
                jlo =  ia(i)
                jhi =  ia(i+1)-1
                if (jlo .gt. jhi) go to 80
                do 70 j = jlo,jhi
                      if (ja(j) .lt. 0) go to 70
                   y(ja(j)) = y(ja(j)) + a(j) * x(i)
 70             continue
 80          continue
      endif  

c-----------------------------------------------------------------
c        y = alpha*y
c-----------------------------------------------------------------*/
      if (alpha .ne. 1.0) then
         do 90 i = 1,nv
            y(i) = y(i) * alpha
 90   continue
      endif

      return
      end





         

