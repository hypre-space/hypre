c     
      subroutine dirslv(k,imin,imax,u,f,a,ia,ja)
c     
c---------------------------------------------------------------------
c     
c     Direct solution
c     
c     solve the problem exactly by gauss elimination.
c     
c     new version (11/12/89)
c     
c     this is a "low" storage version.
c     the pointer ic locates the first entry stored in the
c     vector c. jcmn and jcmx contain the first and last
c     column numbers stored.
c     
c     no pivoting in this preliminary version.
c     
c---------------------------------------------------------------------
c     
      implicit real*8 (a-h,o-z)
c     
      dimension imin(25),imax(25)
      dimension u  (*)
      dimension f  (*)
      dimension ia (*)
      dimension a  (*)
      dimension ja (*)
c     
      parameter(ndimmat=100000,ndimrhs=600)
      dimension c(ndimmat),d(ndimrhs)
      dimension ic(ndimrhs),jcmn(ndimrhs),jcmx(ndimrhs),jcmn2(ndimrhs)
c     
c---------------------------------------------------------------------
c     
      ilo=imin(k)
      ihi=imax(k)
      npts=ihi-ilo+1
      ishft=1-ilo
      if(npts.eq.0) return
      if(npts.gt.1) go to 1
      u(ilo)=f(ilo)/a(ia(ilo))
      return

 1    if(npts.gt.600) stop 'drslv4'
c     
c     load the matrix and right hand side
c     
      jmx=1
      kc=1
      do 40 i=ilo,ihi
c     
c     find jmn and jmx
c     
         jmn=npts
         jlo=ia(i)
         jhi=ia(i+1)-1
         do 10 j=jlo,jhi
            jc=ja(j)+ishft
            if(jc.lt.jmn) jmn=jc
            if(jc.gt.jmx) jmx=jc
 10      continue
         ic(i+ishft)=kc
         jshft=kc-jmn+ishft
         jcmn(i+ishft)=jmn
         jcmx(i+ishft)=jmx
         do 20 jc=jmn,jmx
            c(kc)=0.
            kc=kc+1
            if(kc.gt.ndimmat) stop 'drslv4'
 20      continue
         do 30 j=jlo,jhi
            c(ja(j)+jshft)=a(j)
 30      continue
         d(i+ishft)=f(i)
 40   continue
c     print *,'  drslv4 -- storage used =',kc
      ic(npts+1)=kc
c     
c     find icmx
c     
      jmn=npts
      do 50 n1=npts,1,-1
         if(jcmn(n1).lt.jmn) jmn=jcmn(n1)
         jcmn2(n1)=jmn
 50   continue
c     
c     perform foreward elimination
c     
 100  do 200 n1=1,npts-1
         j1shft=ic(n1)-jcmn(n1)
         do 190 n2=n1+1,npts
            if(jcmn2(n2).gt.n1) go to 200
            if(jcmn(n2).gt.n1) go to 190
            j2shft=ic(n2)-jcmn(n2)
            if(c(n1+j2shft).eq.0.e0) go to 190
            g=c(n1+j2shft)/c(n1+j1shft)
            do 180 n3=n1+1,jcmx(n1)
               c(n3+j2shft)=c(n3+j2shft)-g*c(n3+j1shft)
 180        continue
            d(n2)=d(n2)-g*d(n1)
 190     continue
 200  continue
c     
c     perform back-substitution
c     
      do 290 n1=npts,2,-1
         j1shft=ic(n1)-jcmn(n1)
         d(n1)=d(n1)/c(n1+j1shft)
         do 280 n2=n1-1,1,-1
            if(jcmx(n2).lt.n1) go to 290
            j2shft=ic(n2)-jcmn(n2)
            if(c(n1+j2shft).eq.0.e0) go to 280
            d(n2)=d(n2)-d(n1)*c(n1+j2shft)
 280     continue
 290  continue
 295  d(1)=d(1)/c(1)
c     
c     replace the solution
c     
      do 300 n=1,npts
         u(n-ishft)=d(n)
 300  continue
c     write(6,1234) npts,dnorm
c     1234 format(' drslv2 -- npts=',i2,' dnorm=',1p,e9.2)
      return
      end
