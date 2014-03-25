      subroutine gmres_driver (n, nnz, ia, ja, a, rhs, sol, ipar, rpar,
     .     lenpmx, plu, jlu, ju, perm, qperm, rscale, cscale,
     .     rwork, lrw, ier_gmres, ier_input)
c-----------------------------------------------------------------------
      implicit none
      integer n, nnz, lenpmx, lrw, ipar(20), ier_gmres, ier_input
      integer ia(n+1), ja(nnz), ju(n+1), jlu(lenpmx), perm(n),
     .        qperm(n)
      real*8  a(nnz), rhs(n), sol(n), plu(lenpmx), rscale(n), cscale(n),
     .        rpar(20), rwork(lrw)
c     Local variables
      integer ireorder, iscale, iout, i
      real*8  max, condest
c
      ier_input = 0
c-----------------------------------------------------------------------
c     Load values from ipar and rpar.
c-----------------------------------------------------------------------
c     check for negative ipar values.
      do i=1,20
         if (ipar(i) .lt. 0) then
            ier_input = i
            return
         endif
      enddo
c     check for ireorder value.
      if (ipar(1) .gt. 0) then
         ireorder = ipar(1)
      else
         ireorder = 0  ! default is not to perform reordering.
      endif
c     check for iscale value.
      if (ipar(2) .gt. 0) then
         iscale = ipar(2)
      else
         iscale = 0  ! default is not to perform scaling.
      endif
c     check for iout value.
      if (ipar(3) .gt. 0) then
         iout = ipar(3)
      else
         iout = 0 ! default is no informational messages.
      endif
c-----------------------------------------------------------------------
c     check for enough space in rwork
      if (lrw .lt. n) then
         ipar(20) = n
         ier_input = 42
         return
      endif
c-----------------------------------------------------------------------
c     preconditioner solve

c     copy input vector to temporary location.
      do i = 1,n
         rwork(i) = rhs(i)
      enddo
c     code for row scaling.
      if ((iscale .eq. 1) .or. (iscale .eq. 3)) then
         do i = 1,n
            rwork(i) = rscale(i)*rwork(i)
         enddo
      endif
      if (ireorder .ne. 0) then
         call dvperm (n,rwork,perm) 
      endif

      condest = 0.d0
      do i = 1, n
         condest = max(condest,abs(rwork(i)))
      enddo
      write (*,35) condest
  35  format ('inf-norm lower bound before: ', d16.2)
       
      call lusol0 (n,rwork,sol,plu,jlu,ju)

      condest = 0.d0
      do i = 1, n
         condest = max(condest,abs(sol(i)))
      enddo
      write (*,36) condest
  36  format ('inf-norm lower bound after : ', d16.2)

      if (ireorder .ne. 0) then
         call dvperm (n,sol,qperm) 
      endif
c     code for column scaling.
      if ((iscale .eq. 2) .or. (iscale .eq. 3)) then
         do i = 1,n
            sol(i) = cscale(i)*sol(i)
         enddo
      endif

      ier_gmres = 0
      return
c-------------end-of-subroutine-gmres_driver----------------------------
c-----------------------------------------------------------------------
      end



