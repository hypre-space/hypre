C@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
C
C     TEST DRIVER FOR AMG1R5  (VIA INTERFACE ROUTINE AUX1R5)
C
C     RELEASE 1.1, JUNE 1985
C
C@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
C
C    -------------------------------------------------------------
C    | POISSON EQUATION ON (0,1) X (0,1)                         |
C    | WITH PERIODIC BOUNDARY CONDITIONS:                        |
C    |                                                           |
C    |                                                           |
C    |           U   + U    =   F(X,Y)                           |
C    |            XX    YY                                       |
C    |                                                           |
C    |                                                           |
C    -------------------------------------------------------------
C
C
      IMPLICIT REAL*8 (A-H,O-Z)
cveh     REAL*4 TOLD,TNEW,TSETA,TAMG
      integer TOLD,TNEW,TSETA,TAMG
C
      COMMON /SCONN/  KSTRNG,ISTRNG,NSWEEP
      DIMENSION A(125000),JA(125000),IA(10000),
     *          U(10000),F(10000)
      dimension iu(10000)
      dimension ip(10000)
      dimension iv(10000)
      dimension xp(10000)
      dimension yp(10000)
      dimension ifc(10000)
      common /layerx/  hxl(20),nlx,jlxlo(20),jlxhi(20)
      common /layery/  hyl(20),nly,jlylo(20),jlyhi(20)
      common /bound/   ibcn,ibcs,ibce,ibcw
      common /domain/ nx,ny
      common /mesh/   hx,hy
      CHARACTER*80 DATALN
      CHARACTER*15 FNPRB,FNOUT
cveh  CHARACTER*15 DT,TM
      CHARACTER*24 DT
    
C
C     READ FILE WITH TEST CASES (FILE NAMES). LOOP OVER CASES
C
      fnprb='prob.dat'
      fnout='new.out'
c     PRINT *,'DATA1: ',FNPRB,'  DATA2: ',FNAMG,'  TO: ',FNOUT
      OPEN(8,FILE=FNPRB,STATUS='OLD')
c     REWIND 8
      open(6,file=FNOUT,status='UNKNOWN')
cveh  
cveh  Following  line (date) replaced with the one that follows it
cveh
cveh      CALL DATE(DT)
      CALL FDATE(DT)
cveh      DT='date goes here '
cveh      CALL TIME(TM)
      WRITE(6,1559) FNOUT,DT
1559  FORMAT(' FILE:',A15,10X,'DATE:',A24)
Cveh 1559  FORMAT(' FILE:',A15,10X,'DATE:',A15,5X,'TIME:',A15)
      write(6,1550)
1550  format(/'  NEW VERSION'/)
      write(6,1551) fnprb
1551  format(/'  PROBLEM DATA: FILE=',a15/)
2     READ(8,1111,END=3) DATALN
      WRITE(6,1111) DATALN
1111  format(a80)
      GO TO 2
3     REWIND 8
      write(6,1553)
1553  format(/'  AMG OUTPUT:'/)
C
C===> DEFAULT VALUES (DEFINITION OF PROBLEM)
C
      ibcn=1
      ibcs=1
      ibce=1
      ibcw=1
      hx=1.d0
      hy=1.d0
c     read(8,2000) nx
c     read(8,2000) ny
c     read(8,2000) nlx
      read(8,*) nx
      read(8,*) ny
      read(8,*) nlx
2000  format(bn,10x,2(i9,1x),f10.6)
3000  format(bn,10x,f10.6)
4000  format(bn,10x,4I1)
6000  format('  NX=',i3,'  NY=',i3,'  HX=',d10.3,'  HY=',d10.3)
6001  format('  X LAYER # ',i2,'  JLO=',i3,'  JHI=',i3,'  H=',d10.3)
6101  format('  Y LAYER # ',i2,'  JLO=',i3,'  JHI=',i3,'  H=',d10.3)
6002  format('  BOUNDARY CONDITIONS (N/S/E/W)=',4i2)
6003  format('  STRONG CONNECTIONS - KSTRNG=',I3,
     *       '  ISTRNG=',I2,'  NSWEEP=',I2)
      do 10 n=1,nlx
c     read(8,2000) jlxlo(n),jlxhi(n),hxl(n)
      read(8,*) jlxlo(n),jlxhi(n),hxl(n)
   10 continue
      read(8,*) nly
      do 11 n=1,nly
c     read(8,2000) jlylo(n),jlyhi(n),hyl(n)
      read(8,*) jlylo(n),jlyhi(n),hyl(n)
   11 continue
      read(8,4000) IBCN,IBCS,IBCE,IBCW
      IF(IBCS.EQ.2.OR.IBCN.EQ.2) THEN
        IBCS=2
        IBCN=2
      ENDIF
      IF(IBCW.EQ.2.OR.IBCE.EQ.2) THEN
        IBCW=2
        IBCE=2
      ENDIF
C
C===> DEFINE PROBLEM
C
100   CALL CTIME(TOLD)
      call seta(nx,ny,hx,hy,a,ja,ia,f,igrid,nnu,xp,yp)
c
c     define "system" pointers
c
      nu = 1
      nv = nnu
      np = nnu
      do 20 i=1,nnu
      iu(i)=1
      ip(i)=i
      iv(i)=i
      ifc(i)=0
c     xp(i)=0.0
c     yp(i)=0.0
      u(i) = 1.0
20    continue
      iv(nnu+1)=nnu+1
      isw=1
c     
      CALL CTIME(TNEW)
      TSETA=TNEW-TOLD
C
C===> SOLVE PROBLEM
C
      TOLD=TNEW
c     CALL AUX1R5(A,IA,JA,U,F,IG,
c    +           NDA,NDIA,NDJA,NDU,NDF,NDIG,NNU,MATRIX,
c    +           EPS,IFIRST,ifmg,ISWTCH,IOUT,IPRINT,
c    +           IERR)
      call amgs01(u,f,a,ia,ja,iu,ip,iv,xp,yp,ifc,nu,nv,np,isw)
      CALL CTIME(TNEW)
      TAMG=TNEW-TOLD
      WRITE (6,9000) TSETA,TAMG
      CLOSE(6)
      CLOSE(8)
      CLOSE(9)
8004  FORMAT(8I10) 
8005  FORMAT(5D12.5)
9999  STOP
C
9000  FORMAT(///'***** CPU-TIME (TSETA):',I10,' SEC *****'
     *         /'***** CPU-TIME (TAMG) :',I10,' SEC *****'/)
      END
c
      subroutine seta(nx,ny,hx,hy,a,ja,ia,f,igrid,nnu,xp,yp)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      common /layerx/  hxl(20),nlx,jlxlo(20),jlxhi(20)
      common /layery/  hyl(20),nly,jlylo(20),jlyhi(20)
      common /bound/  ibcn,ibcs,ibce,ibcw
      dimension a(*),ia(*),ja(*),f(*)
      dimension xx(65),yy(65)
      dimension e(4,4),x(4),y(4)
      dimension ipt(4)
      dimension ibc(5000)
      dimension xp(*)
      dimension yp(*)
c
c     set the grid
c
      xx(1)=0.
      do 10 i=1,nx
      h=hx
      do 5 n=1,nlx
      if(i.ge.jlxlo(n).and.i.le.jlxhi(n)) h=hxl(n)
    5 continue
      xx(i+1)=xx(i)+h
   10 continue
      yy(1)=0.
      do 20 i=1,ny
      h=hy
      do 15 n=1,nly
      if(i.ge.jlylo(n).and.i.le.jlyhi(n)) h=hyl(n)
   15 continue
      yy(i+1)=yy(i)+h
   20 continue
c     do 21 i=1,nx+1
c     print *,' i,x=',i,xx(i)
c  21 continue
c     do 22 i=1,ny+1
c     print *,' i,y=',i,yy(i)
c  22 continue
c
c     set form of a
c
      ka=1
      ip=0
      ihi=nx+1
      if(ibcw.eq.2) ihi=nx
      jhi=ny+1
      if(ibcs.eq.2) jhi=ny
      do 40 j=1,jhi
      jjlo=max0(j-1,1)
      jjhi=min0(j+1,ny+1)
      if(ibcs.eq.2) jjlo=j-1
      if(ibcn.eq.2) jjhi=j+1
      do 40 i=1,ihi
      iilo=max0(i-1,1)
      iihi=min0(i+1,nx+1)
      if(ibcw.eq.2) iilo=i-1
      if(ibce.eq.2) iihi=i+1
      ip=ip+1
      xp(ip) = xx(i)
      yp(ip) = yy(j)
      xp(ip) = xx(i)
      ia(ip)=ka
      a(ka)=0.
      ja(ka)=ip
      ka=ka+1
      do 30 jj=jjlo,jjhi
      jjj=jj
      if(jj.lt.1) jjj=jhi
      if(jj.gt.jhi) jjj=1
      do 30 ii=iilo,iihi
      iii=ii
      if(ii.lt.1) iii=ihi
      if(ii.gt.ihi) iii=1
      iip=(jjj-1)*ihi+iii
      if(iip.eq.ip) go to 30
      a(ka)=0.
      ja(ka)=iip
      ka=ka+1
   30 continue
   40 continue
      ia(ip+1)=ka
      nnu=ip
c
c     assemble matrix
c
      do 100 j=1,ny
      jp=j+1
      if(jp.gt.jhi) jp=1
      do 100 i=1,nx
      ip=i+1
      if(ip.gt.ihi) ip=1
      x(1)=xx(i)
      x(2)=xx(i+1)
      x(3)=xx(i+1)
      x(4)=xx(i)
      y(1)=yy(j)
      y(2)=yy(j)
      y(3)=yy(j+1)
      y(4)=yy(j+1)
      call setelt(e,x,y)
      ipt(1)=(j-1)*ihi+i
      ipt(2)=(j-1)*ihi+ip
      ipt(3)=(jp-1)*ihi+ip
      ipt(4)=(jp-1)*ihi+i
      do 90 n1=1,4
      ip1=ipt(n1)
      do 90 n2=1,4
      ip2=ipt(n2)
      jjlo=ia(ip1)
      jjhi=ia(ip1+1)-1
      do 80 jj=jjlo,jjhi
      if(ja(jj).ne.ip2) go to 80
      a(jj)=a(jj)+e(n1,n2)
      go to 90
   80 continue
      print *,'  i,j,ip1,ip2= ',i,j,ip1,ip2
      stop 'fuckup 1'
   90 continue
  100 continue
c
c     set dirichlet boundary conditions
c
      n1=0
      n2=0
      do 200 j=1,jhi
      do 200 i=1,ihi
      n1=n1+1
      ibc(n1)=0
      if(i.eq.1.and.ibcw.eq.1) go to 200
      if(i.eq.nx+1.and.ibce.eq.1) go to 200
      if(j.eq.1.and.ibcs.eq.1) go to 200
      if(j.eq.ny+1.and.ibcn.eq.1) go to 200
      n2=n2+1
      ibc(n1)=n2
  200 continue
      nnu=n2
      if(n2.eq.n1) go to 1000
      ka=1
      do 220 i=1,n1
      ii=ibc(i)
      if(ii.eq.0) go to 220
      jlo=ia(i)
      jhi=ia(i+1)-1
      ia(ii)=ka
      xp(ii)=xp(i)
      yp(ii)=yp(i)
      do 210 j=jlo,jhi
      if(ibc(ja(j)).eq.0) go to 210
      a(ka)=a(j)
      ja(ka)=ibc(ja(j))
      ka=ka+1
  210 continue
  220 continue
      ia(nnu+1)=ka
C
C===> SET RIGHT HAND SIDE TO ZERO
C
1000  DO 1010 I=1,NNU
        F(I) = 0.D0
1010  CONTINUE
      return
      end
C
      SUBROUTINE SETELT(E,XX,YY)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DIMENSION E(*)
      DIMENSION XX(4),YY(4)
C
C     2-D HELMHOLZ ELEMENT (CURRENTLY ONLY IN THE X-Y PLANE)
C
      HX=XX(2)-XX(1)
      HX2=HX*HX
      HY=YY(4)-YY(1)
      HY2=HY*HY
      DD=HX*HY/6.
      E(1)= DD*( 2./HX2+2./HY2)
      E(2)= DD*(-2./HX2+1./HY2)
      E(3)= DD*(-1./HX2-1./HY2)
      E(4)= DD*( 1./HX2-2./HY2)
      E(5)= DD*(-2./HX2+1./HY2)
      E(6)= DD*( 2./HX2+2./HY2)
      E(7)= DD*( 1./HX2-2./HY2)
      E(8)= DD*(-1./HX2-1./HY2)
      E(9)= DD*(-1./HX2-1./HY2)
      E(10)=DD*( 1./HX2-2./HY2)
      E(11)=DD*( 2./HX2+2./HY2)
      E(12)=DD*(-2./HX2+1./HY2)
      E(13)=DD*( 1./HX2-2./HY2)
      E(14)=DD*(-1./HX2-1./HY2)
      E(15)=DD*(-2./HX2+1./HY2)
      E(16)=DD*( 2./HX2+2./HY2)
      RETURN
      END
C
      SUBROUTINE OUTG(M,IMIN,IMAX,ICG)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
C
C     THIS PRINTS OUT THE GRID
C     NO ASSUMPTION IS MADE ON POINT ORDERING
C
      common /domain/ nx,ny
      common /bound/  ibcn,ibcs,ibce,ibcw
      DIMENSION IMIN(*),IMAX(*),ICG(*)
      DIMENSION KPT(80)
C
      ILO=IMIN(1)
      IHI=IMAX(1)
C
      ighi=nx+1
      if(ibce.eq.2) ighi=nx
      jghi=ny+1
      if(ibcn.eq.2) jghi=ny
      n=0
      jlo=1
      if(ibcs.eq.1) jlo=2
      jhi=ny+1
      if(ibcn.ge.1) jhi=ny
      ilo=1
      if(ibcw.eq.1) ilo=2
      ihi=nx+1
      if(ibce.ge.1) ihi=nx
      DO 970 J=1,jghi
      DO 910 I=1,ighi
      KPT(I)=0
  910 CONTINUE
      IF(J.LT.JLO.OR.J.GT.JHI) GO TO 970
      DO 960 I=ILO,IHI
      n=n+1
      KPT(i)=1
      iii=n
      DO 260 KKK=2,m
      III=ICG(III)
      IF(III.LE.0) GO TO 270
      KPT(i)=KKK
  260 CONTINUE
  270 CONTINUE
  960 CONTINUE
  970 WRITE(6,3000) (KPT(I),I=1,ighi)
      RETURN
 1000 FORMAT(/'  AMG GRID '/)
 3000 FORMAT(80I1)
      END
