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
      CHARACTER*15 FYSMP,FRHS,FINITU

      integer data
C
C     READ INPUT FILES WITH MATRIX, RIGHT-HAND SIDE, PROBLEM SPECS,
C     INITIAL GUESS.
C

      FYSMP = 'AMG.in.ysmp'
      FRHS = 'AMG.in.rhs'
      FINITU='AMG.in.initu'



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

      tol = 1.0e-7

      call amg_Initialize(data,0)


      call amg_SetLogging(3, "AMG.runlog", data)

      call amg_Setup(isterr,a,ia,ja,nv,data)
      print *, 'Setup error flag = ', isterr

      call amg_Solve(isverr,u,f,nv,tol,data)
      print *, 'Solve error flag = ', isverr

      call amg_Finalize(data)

      STOP

      END

