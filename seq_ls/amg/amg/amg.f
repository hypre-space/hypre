C@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
C
C     File: amgdrv.f
C     TEST DRIVER FOR AMG Problems
C     reads matrix in from file AMG.in.ysmp.         Use link.
C     reads right-hand side from  AMG.in.rhs.        Use link.
C     reads problem specs from AMG.in.prob.          Use link.
C     reads initial approximation from AMG.in.initu. Use link.
C     reads AMG control from AMG.in.solv.            Use link.
C
C@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
C
      IMPLICIT REAL*8 (A-H,O-Z)
      integer TOLD,TNEW,TAMG
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
      CHARACTER*15 FNOUT,FYSMP,FRHS,FPROB,FINITU,FSOLV
      CHARACTER*24 DT
    
C
C     READ INPUT FILES WITH MATRIX, RIGHT-HAND SIDE, PROBLEM SPECS,
C     INITIAL GUESS.
C

      fnout='new.out'
      FYSMP = 'AMG.in.ysmp'
      FRHS = 'AMG.in.rhs'
      FPROB = 'AMG.in.prob'
      FINITU = 'AMG.in.initu'
      FSOLV = 'AMG.in.solv'

      CALL FDATE(DT)

      open(6,file=FNOUT,status='UNKNOWN')
      WRITE(6,1559) FNOUT,DT
1559  FORMAT(' FILE:',A15,10X,'DATE:',A24)
      write(6,1550)
1550  format(/'  NEW VERSION'/)
cveh      write(6,1551) fnprb
1551  format(/'  PROBLEM DATA: FILE=',a15/)
1111  format(a80)
      write(6,1553)
1553  format(/'  AMG OUTPUT:'/)

CVEH
C     READ IN MATRIX FROM AMG.in.ysmp
CVEH
      open (8,FILE=FYSMP,STATUS='OLD')
      read (8,*) junk
      read (8,*) nv
      read (8,*) (ia(j), j=1,nv+1)
      read (8,*) (ja(j), j=1,ia(nv+1)-1)
      read (8,*) (a(j), j=1,ia(nv+1)-1)
      close (8)

CVEH
C     READ IN PROBLEM DETAILS FROM AMG.in.prob
CVEH
      open (8,FILE=FPROB,STATUS='OLD')
      read (8,*) junk
      read (8,*) nv1
      if (nv1 .ne. nv) then 
         write (6,*) 'Problem specs incompatible with matrix'
         write (6,*) 'nv1 is not equal to nv'
         stop
      endif
      read(8,*) nu,np
      read (8,*) (iu(j), j=1,nv)
      read (8,*) (ip(j), j=1,nv)
      read (8,*) (iv(j), j=1,np+1)
      close (8)


CVEH
C     READ IN RHS FROM AMG.in.rhs
CVEH
      open (8,FILE=FRHS,STATUS='OLD')
      read (8,*) junk
      read (8,*) nv1
      if (nv1 .ne. nv) then
         write (6,*) 'Right hand side incompatible with matrix'
         write (6,*) 'nv1 is not equal to nv'
         stop
      endif
      read (8,*) (f(j), j=1,nv)
      close (8)


CVEH
C     READ IN INITIAL GUESS FROM AMG.in.initu
CVEH
      open (8,FILE=FINITU,STATUS='OLD')
      read (8,*) junk
      read (8,*) nv1
      if (nv1 .ne. nv) then 
         write (6,*) 'Right hand side incompatible with matrix'
         write (6,*) 'nv1 is not equal to nv'
         stop
      endif
      read (8,*) (u(j), j=1,nv)
      close (8)



      write (6,*) nu,np,nv,'    nu, np, nv'
      write (6,*) ia(1), ia(2), ia(nv), ia(nv+1), ' ia(1, 2, nv, nv+1)'  
      write(6,*) ja(1), ja(2), ja(ia(nv+1)-1), '  ja(1, 2, ia(nv+1)-1)'
      write (6,*) a(1), a(2), a(ia(nv+1)-1),'  a(1, 2, ia(nv+1)-1)'
      write (6,*) iu(1), iu(nv), '  iu(1), iu(nv)'
      write (6,*) ip(1), ip(2), ip(nv),'  ip(1), ip(2), ip(nv)'
      write (6,*) iv(1), iv(2), iv(np), ' iv(1), iv(2), iv(np)' 
      write (6,*) u(1), u(nv), '  u(1), u(nv)'
      write (6,*) f(1), f(nv),'  f(1), f(nv)'
      write (6,*) ifc(1), ifc(nv), '  ifc(1, nv)'
      write (6,*) xp(1), xp(nv), '  xp(1, nv)'
      write (6,*) yp(1), yp(nv), '  yp(1, nv)'

      write(6,'(1x)')

C
C              Tell AMGS01 to call the setup phase (& read amg.dat)
C
      isw=1
c     
       CALL CTIME(TNEW)
cveh       TSETA=TNEW-TOLD
C
C===> SOLVE PROBLEM
C
      TOLD=TNEW
      write(6,*) 'calling amgs01'
      call amgs01(u,f,a,ia,ja,iu,ip,iv,xp,yp,ifc,nu,nv,np,isw)
      CALL CTIME(TNEW)
      TAMG=TNEW-TOLD
      WRITE (6,9000) TAMG
      CLOSE(6)
      CLOSE(9)
8004  FORMAT(8I10) 
8005  FORMAT(5D12.5)
9999  STOP

9000  FORMAT(///'***** Running TIME (TAMG) :',I10,' SEC *****'/)
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
