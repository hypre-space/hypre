C@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
C
C     Example DRIVER for AMG library
C
C     reads matrix in from file:        AMG2.in.ysmp         
C     reads right-hand side from:       AMG2.in.rhs        
C     reads initial approximation from: AMG2.in.initu.
C
C@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
C
      IMPLICIT REAL*8 (A-H,O-Z)
C

      parameter(ndima=600000,ndimu=250000)
      DIMENSION A(ndima),JA(ndima),IA(ndimu),U(ndimu),F(ndimu)
      CHARACTER*15 FYSMP,FRHS,FINITU,FCONT

      dimension mu(10), iprlx(4), ntrlx(4),ierlx(4),iurlx(4)
      integer data
C
C     READ INPUT FILES WITH MATRIX, RIGHT-HAND SIDE, PROBLEM SPECS,
C     INITIAL GUESS.
C

      FYSMP = 'AMG2.in.ysmp'
      FRHS = 'AMG2.in.rhs'
      FINITU='AMG2.in.initu'
      FCONT = 'AMG2.in.control'


CVEH
C     READ IN CONTROL PARAMETERS FROM AMG2.in.control
CVEH
      open (8,FILE=FCONT,STATUS='OLD')
      read (8,*) numu, nump
      read (8,*) tol
      read (8,*) levmax
      read (8,*) ncg, ecg
      read (8,*) nwt, ewt
      read (8,*) nstr
      read (8,*) ncyc
      read (8,*) (mu(j), j=1,10)
      read (8,*) (ntrlx(j), j=1,4)
      read (8,*) (iprlx(j), j=1,4)
      read (8,*) (ierlx(j), j=1,4)
      read (8,*) (iurlx(j), j=1,4)
      read (8,*) ioutdat
      close (8)


CVEH
C     READ IN MATRIX FROM AMG2.in.ysmp
CVEH
      open (8,FILE=FYSMP,STATUS='OLD')
      read (8,*) junk
      read (8,*) nv
      read (8,*) (ia(j), j=1,nv+1)
      read (8,*) (ja(j), j=1,ia(nv+1)-1)
      read (8,*) (a(j), j=1,ia(nv+1)-1)
      close (8)



CVEH
C     READ IN RHS FROM AMG2.in.rhs
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
C     READ IN INITIAL GUESS FROM AMG2.in.initu
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

      call amg_Initialize(data,0)

      call amg_SetNumUnknowns(numu,data)
      call amg_SetNumPoints(nump,data)

      call amg_SetLogging(ioutdat, "AMG2.runlog", data)
      call amg_SetLevMax(levmax, data)
      call amg_SetEWT(ewt,data)
      call amg_SetNWT(nwt,data)
      call amg_SetNCG(ncg, data)
      call amg_SetECG(ecg, data)
      call amg_SetNWT(nwt, data)
      call amg_SetEWT(ewt, data)
      call amg_SetNSTR(nstr, data)
      call amg_SetNCyc(ncyc, data)

      call amg_Setup(isterr,a,ia,ja,nv,data)
      print *, 'Setup error flag = ', isterr

      call amg_Solve(isverr,u,f,nv,tol,data)
      print *, 'Solve error flag = ', isverr

      call amg_Finalize(data)

      STOP

      END

