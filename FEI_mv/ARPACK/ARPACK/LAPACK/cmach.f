      real function cmach(job)
      integer job
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
c
      real eps,tiny,huge,s
c
      eps = 1.0
   10 eps = eps/2.0
      s = 1.0 + eps
      if (s .gt. 1.0) go to 10
      eps = 2.0*eps
      cmach =eps
      if( job .eq. 1) return
c
      s = 1.0
   20 tiny = s
      s = s/16.0
      if (s*1.0 .ne. 0.0) go to 20
      tiny = (tiny/eps)*100.
      s = real((1.0,0.0)/cmplx(tiny,0.0))
      if (s .ne. 1.0/tiny) tiny = sqrt(tiny)
      huge = 1.0/tiny
      if (job .eq. 1) cmach = eps
      if (job .eq. 2) cmach = tiny
      if (job .eq. 3) cmach = huge
      return
      end
