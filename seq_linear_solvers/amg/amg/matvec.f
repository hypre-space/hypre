c=====================================================================
c     matvec routine:
c     compute z = alpha*A*x + beta*y
c=====================================================================

      subroutine matvec(nv, z, alpha, a, ia, ja, x, beta, y)

      implicit real*8 (a-h,o-z)

      dimension x  (*)
      dimension y  (*)
      dimension z  (*)
      dimension a  (*)
      dimension ia (*)
      dimension ja (*)

c---------------------------------------------------------------------

      if (alpha .eq. 0.0) then
         do 20 i=1,nv
            z(i) = beta * y(i)
 20      continue
         return
      else
          do 40 i = 1,nv
             z(i)=0.e0
             do 30 j = ia(i),ia(i+1)-1
                z(i) = z(i) + a(j) * x(ja(j))
 30          continue
             z(i) = alpha * z(i)
40       continue
          if (beta .ne. 0.0) then
             do 50 i=1,nv
                z(i) = z(i) + beta * y(i)
 50          continue
          endif
      endif 
      return
      end





         

