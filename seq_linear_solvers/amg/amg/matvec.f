c=====================================================================
c     matvec routine:
c     compute z = alpha*A*x + beta*y
c
c     if jtrans==1, computes z=alpha*A^T*x + beta*y (A transpose)
c=====================================================================

      subroutine matvec(nv, z, alpha, a, ia, ja, x, beta, y, jtrans)

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
          if (jtrans .ne. 1) then
             do 40 i = 1,nv
                z(i)=0.e0
                jlo =  ia(i)
                jhi =  ia(i+1)-1
c                if (jlo .gt. jhi) go to 40
                do 30 j = jlo,jhi
                   z(i) = z(i) + a(j) * x(ja(j))
 30             continue
                z(i) = alpha * z(i)
 40          continue
          else
             do 60 i = 1,nv
                jlo =  ia(i)
                jhi =  ia(i+1)-1
                if (jlo .gt. jhi) go to 60
                do 50 j = jlo,jhi
                   z(ja(j)) = z(ja(j)) + a(j) * x(i)
 50             continue
                z(i) = alpha * z(i)
 60          continue  
          endif
          if (beta .ne. 0.0) then
             do 70 i=1,nv
                z(i) = z(i) + beta * y(i)
 70          continue
          endif
      endif 
      return
      end





         

