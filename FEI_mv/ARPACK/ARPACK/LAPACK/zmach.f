      double precision function zmach(job)
      integer job
c
c     double complex floating point arithmetic constants.
c     for the linpack test drivers only.
c     not used by actual linpack subroutines.
c
c     smach computes machine parameters of floating point
c     arithmetic for use in testing only.  not required by
c     linpack proper.
c
c     if trouble with automatic computation of these quantities,
c     they can be set by direct assignment statements.
c     assume the computer has
c
c        b = base of arithmetic
c        t = number of base  b  digits
c        l = smallest possible exponent
c        u = largest possible exponent
c
c     then
c
c        eps = b**(1-t)
c        tiny = 100.0*b**(-l+t)
c        huge = 0.01*b**(u-t)
c
c     dmach same as smach except t, l, u apply to
c     double precision.
c
c     cmach same as smach except if complex division
c     is done by
c
c        1/(x+i*y) = (x-i*y)/(x**2+y**2)
c
c     then
c
c        tiny = sqrt(tiny)
c        huge = sqrt(huge)
c
c
c     job is 1, 2 or 3 for epsilon, tiny and huge, respectively.
c
      double precision eps,tiny,huge,s
c
      eps = 1.0d0
   10 eps = eps/2.0d0
      s = 1.0d0 + eps
      if (s .gt. 1.0d0) go to 10
      eps = 2.0d0*eps
c
      s = 1.0d0
   20 tiny = s
      s = s/16.0d0
      if (s*1.0d0 .ne. 0.0d0) go to 20
      tiny = tiny/eps
      s = (1.0d0,0.0d0)/dcmplx(tiny,0.0d0)
      if (s .ne. 1.0d0/tiny) tiny = dsqrt(tiny)
      huge = 1.0d0/tiny
c
      if (job .eq. 1) zmach = eps
      if (job .eq. 2) zmach = tiny
      if (job .eq. 3) zmach = huge
      return
      end
