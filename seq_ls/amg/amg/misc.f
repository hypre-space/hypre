C     C### filename: MS.FOR
c     
c==== FILE MS.FOR ====================================================
c     
c     MISCELLANEOUS ROUTINES FOR AMGS01
c     
c=====================================================================
c     
c=====================================================================
c     
c     routines for function definition
c     
c=====================================================================
c     
      subroutine putf(k,irhs,imin,imax,f,iu,ip,xp,yp)
c     
c---------------------------------------------------------------------
c     
      implicit real*8 (a-h,o-z)
c     
      dimension imin(*),imax(*)
      dimension f  (*)
      dimension iu (*)
      dimension ip (*)
      real*8    xp (*)
      real*8    yp (*)
c     
c---------------------------------------------------------------------
c     
      ilo=imin(k)
      ihi=imax(k)
      if(irhs.lt.0) return
      if(irhs.eq.1) go to 20
      do 10 i=ilo,ihi
         f(i)=0.e0
 10   continue
      return
 20   do 21 i=ilo,ihi
         f(i)=1.e0
 21   continue
      return
      end
c     
      subroutine putu(k,rndu,imin,imax,u,iu,ip,xp,yp)
c     
c---------------------------------------------------------------------
c     
c     sets level k function u to a grid function:
c     
c     - 0.0 .lt. rndu .lt. 1.0: random function with random values
c     influenced by the value of rndu
c     - rndu=0.0:               zero
c     - rndu=1.0:               one
c     
c---------------------------------------------------------------------
c     
      implicit real*8 (a-h,o-z)
c     
      dimension imin(*),imax(*)
      dimension u  (*)
      dimension iu (*)

      dimension ip (*)
      real*8 xp (*)
      real*8 yp (*)
c     
c---------------------------------------------------------------------
c     
      imn=imin(k)
      imx=imax(k)
      if (rndu.lt.0.9999999.and.rndu.gt.0.0) goto 20
      if (rndu.ne.0.0) goto 50
      do 10 i=imn,imx
         u(i)=0.e0
 10   continue
      return
c     
 20   s=rndu
      do 30 i=imn,imx
         u(i)=random(s)
 30   continue
      return
c     
 50   if(rndu.gt.1.) go to 200
      do 100 i=imn,imx
         u(i)=1.0e0
 100  continue
      return
 200  if(rndu.gt.2.) go to 300
      do 210 i=imn,imx
         x=xp(ip(i))
         y=yp(ip(i))
         if(iu(i).eq.1) u(i)=x*(2.*y-1)
         if(iu(i).eq.2) u(i)=-x*x
 210  continue
      return
 300  if(rndu.gt.3.) go to 400
      do 310 i=imn,imx
         x=xp(ip(i))
         y=yp(ip(i))
         if(iu(i).eq.1) u(i)=x*(2.*y-1)
         if(iu(i).eq.2) u(i)=-1.5*x
 310  continue
      return
 400  if(rndu.gt.4.) return
      pi=acos(-1.0)
      do 410 i=imn,imx
         x=xp(ip(i))
         y=yp(ip(i))
         if(iu(i).eq.1) u(i)=sin(pi*x)*sin(pi*y)
         if(iu(i).eq.2) u(i)=cos(2.0*pi*x)*sin(pi*y)
         if(iu(i).eq.3) u(i)=sin(pi*x)*cos(3.0*pi*y)
         if(iu(i).eq.4) u(i)=cos(pi*x)*cos(pi*y)
         if(iu(i).eq.5) u(i)=sin(2.0*pi*x)*sin(3.0*pi*y)
         if(iu(i).eq.6) u(i)=cos(pi*x)*sin(1.0+pi*y)
         if(iu(i).eq.7) u(i)=cos(4.0*pi*x)*sin(2.5*pi*y)
 410  continue
      return
      end
c     
      subroutine putz(k,imin,imax,u)
c     
c---------------------------------------------------------------------
c     
      implicit real*8 (a-h,o-z)
c     
      dimension imin(*),imax(*)
      dimension u (*)
c     
c---------------------------------------------------------------------
c     
      ilo=imin(k)
      ihi=imax(k)
      do 10 i=ilo,ihi
         u(i)=0.e0
 10   continue
      return
      end
c     
      real*8 function random(s)
      implicit real*8 (a-h,o-z)
      random=100.0e0* exp(s)
      random=random-float(int(random))
      s=random
      return
      end
c     
c=====================================================================
c     
c     real/integer parameter decomposition routines
c     
c=====================================================================
c     
      subroutine idec(int0,nnum,ndigit,iarr)
      implicit real*4 (a-h,o-z)
c     
c     decompose non-negative integer int0 into nnum integers
c     
c     input:  int0   - integer (0.le. int0 .le.999999999)
c     nnum   - integer (1.le. nnum .le.9); number of integers
c     to be returned on array iarr (see below)
c     
c     output: ndigit - integer; number of digits of int0
c     iarr   - integer-array with the following contents:
c     iarr(1)        = first      digit of int0,
c     iarr(2)        = second     digit of int0, ....
c     iarr(nnum-1)   = (nnum-1)st digit of int0,
c     iarr(nnum)     = rest of int0
c     if nnum > ndigit, the corresponding components
c     of iarr are put to zero.
c     
      dimension iarr(10)
      data eps /1.00000000001/
c     
      if (int0.ge.10) goto 10
      ndigit=1
      iarr(1)=int0
 1    do 5 i=ndigit+1,nnum
         iarr(i)=0
 5    continue
      return
c     
 10   ndigit=1+int(alog10(eps*float(int0)))
      nrest=int0
      do 20 i=ndigit,1,-1
         iarr(i)=nrest-nrest/10*10
         nrest=nrest/10
 20   continue
      if (ndigit.le.nnum) goto 1
c     
      nrest=iarr(ndigit)
      ie=0
      do 30 i=ndigit-1,nnum,-1
         ie=ie+1
         nrest=nrest+iarr(i)*10**ie
 30   continue
      iarr(nnum)=nrest
      return
      end
c     
c.....................................................................
c     
c     rdec                                               subroutine
c     
c.....................................................................
c     
      subroutine rdec(r0,r1,r2)
      implicit real*8 (a-h,o-z)
c     
c     decompose non-negative real r0 into two reals r1,r2
c     
c     input:  r0 - real number of the form i.j, i and j integers.
c     the number of digits of i is not allowed to exceed
c     the total sum of digits is not allowed to exceed 15
c     
c     output: r1 - real number: r1=0.i
c     r2 - real number: r2=0.j
c     
      if (r0.ge.1.0) goto 10
      r1=0.e0
      r2=r0
      return
c     
 10   r1=float(int(r0))
      r2=r0-r1
      do 20 i=1,15
         r1=r1*0.1
         if (r1.lt.1.0) return
 20   continue
      stop
      end




C     C### filename: GP.FOR
c     
c==== FILE GP.FOR ====================================================
c     
c     GENERAL PURPOSE ROUTINES
c     
c=====================================================================
c     
c     timing & time/date routines
c     
c=====================================================================
c     
      subroutine ctime(nsec)
c     
c=====================================================================
c     
c     returns time elaspsed since midnight (in seconds)
c     
c=====================================================================
c     
c     time in milliseconds :
c     

      integer time, nsec
      nsec = time()

      return
      end
