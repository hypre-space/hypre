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
