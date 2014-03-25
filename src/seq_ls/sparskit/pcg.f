      subroutine pcg(n, rhs, sol, ipar, fpar, w)
      implicit none
      integer n, ipar(16)
      real*8 rhs(n), sol(n), fpar(16), w(n,*)
c-----------------------------------------------------------------------
c     This is a implementation of the Preconditioner Conjugate Gradient
c     (PCG) method for solving linear system using 
c     reverse communication
c
c     fpar(7) is used here internally to store <r, z>.
c     w(:,1) -- r, the residual vector
c     w(:,2) -- p, the conjugate direction
c     w(:,3) -- A*p, matrix multiply the conjugate direction
c     w(:,4) -- z, auxilliary vector for PCG
c     w(:,5) -- change in the solution (sol) is stored here until
c               termination of this solver
c-----------------------------------------------------------------------
c     external functions used
c
      real*8 distdot
      logical stopbis, brkdn
      external distdot, stopbis, brkdn, bisinit
c
c     local variables
c
      integer i
      real*8 alpha, beta
      logical lp,rp,pre
      save
c
c     check the status of the call
c
      if (ipar(1).le.0) ipar(10) = 0
      goto (10, 20, 30, 40, 50), ipar(10)
c
c     initialization
c
      call bisinit(ipar,fpar,5*n,1,lp,rp,w)
      pre = (lp) .or. (rp)
      if (ipar(1).lt.0) return
c
c     request for matrix vector multiplication A*x in the initialization
c
      ipar(1) = 1
      ipar(8) = n+1
      ipar(9) = ipar(8) + n
      ipar(10) = 1
      do i = 1, n
         w(i,2) = sol(i)
      enddo
      return
 10   ipar(7) = ipar(7) + 1
      do i = 1, n
         w(i,1) = rhs(i) - w(i,3)
      enddo
      fpar(11) = fpar(11) + n
c
c     calculate preconditioned initial residual.  set z0 = M^{-1} r0.
c
      if (pre) then
         ipar(1) = 2
         ipar(8) = 1
         ipar(9) = 3*n + 1
         ipar(10) = 2
         return
      else
         do i = 1, n
            w(i,4) = w(i,1)
         enddo
      endif
c
 20   continue
c     set p0 = z0
      do i = 1, n
         w(i,2) = w(i,4)
      enddo
c
      fpar(7) = distdot(n,w,1,w(1,4),1)
      fpar(11) = fpar(11) + 2 * n
      fpar(3) = sqrt(distdot(n,w,1,w,1))
      fpar(5) = fpar(3)
      if (abs(ipar(3)).eq.2) then
         fpar(4) = fpar(1) * sqrt(distdot(n,rhs,1,rhs,1)) + fpar(2)
         fpar(11) = fpar(11) + 2 * n
      else if (ipar(3).ne.999) then
         fpar(4) = fpar(1) * fpar(3) + fpar(2)
      endif
c
c     before iteration can continue, we need to compute A * p, which
c     includes the preconditioning operations
c
 30   ipar(1) = 1
      ipar(8) = n + 1
      ipar(9) = 2*n + 1
      ipar(10) = 4
      return
c
c     continuing with the iterations
c
 40   ipar(7) = ipar(7) + 1
      alpha = distdot(n,w(1,2),1,w(1,3),1)
      fpar(11) = fpar(11) + 2*n
      if (brkdn(alpha,ipar)) goto 60
      alpha = fpar(7) / alpha
      do i = 1, n
         w(i,5) = w(i,5) + alpha * w(i,2)
         w(i,1) = w(i,1) - alpha * w(i,3)
      enddo
      fpar(11) = fpar(11) + 4*n
c
c     are we ready to terminate ?
c
      if (stopbis(n,ipar,1,fpar,w,w(1,2),alpha)) goto 60
c
c     calculate preconditioned initial residual
c
      if (pre) then
         ipar(1) = 2
         ipar(8) = 1
         ipar(9) = 3*n + 1
         ipar(10) = 5
         return
      else
         do i = 1, n
            w(i,4) = w(i,1)
         enddo
      endif
         
c
c     continue the iterations
c
 50   continue
      beta = fpar(7)
      fpar(7) = distdot(n,w,1,w(1,4),1)
      beta = fpar(7) / beta
      do i = 1, n
         w(i,2) = w(i,4) + beta * w(i,2)
      enddo
      fpar(11) = fpar(11) + 2*n
      goto 30
c
c     clean up -- necessary to accommodate the right-preconditioning
c
 60   call tidycg(n,ipar,fpar,sol,w(1,5))
c
      return
c-----end-of-pcg
c-----------------------------------------------------------------------
      end
