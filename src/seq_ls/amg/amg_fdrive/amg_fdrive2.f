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
      implicit real*8 (a-h,o-z)
C

      parameter(ndima=600000,ndimu=240000)
      dimension a(ndima),ja(ndima),ia(ndimu),u(ndimu),f(ndimu)
      CHARACTER*15 FYSMP,FRHS,FINITU,FCONT

      dimension mu(25), iprlx(4), ntrlx(4)
      integer data
      integer*4 time_ticks, cpu_ticks 
      integer*4 setup_time,setup_cpu,time,cpu,solve_time,solve_cpu
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
      read (8,*) (mu(j), j=1,levmax)
      read (8,*) (ntrlx(j), j=1,4)
      read (8,*) (iprlx(j), j=1,4)
      read (8,*) ioutdat
      close (8)


CVEH
C     READ IN MATRIX FROM AMG2.in.ysmp
CVEH
      open (8,FILE=FYSMP,STATUS='OLD')
      read (8,*) nv
      read (8,*) (ia(j), j=1,nv+1)
      read (8,*) (ja(j), j=1,ia(nv+1)-1)
      read (8,*) (a(j), j=1,ia(nv+1)-1)
      close (8)

      print *, 'Matrix read'

CVEH
C     READ IN RHS FROM AMG2.in.rhs
CVEH
      open (8,FILE=FRHS,STATUS='OLD')
      read (8,*) nv1
      if (nv1 .ne. nv) then
         write (6,*) 'Right hand side incompatible with matrix'
         write (6,*) 'nv1 is not equal to nv'
         stop
      endif
      read (8,*) (f(j), j=1,nv)
      close (8)

      print *, 'RHS read'

CVEH
C     READ IN INITIAL GUESS FROM AMG2.in.initu
CVEH
      open (8,FILE=FINITU,STATUS='OLD')
      read (8,*) nv1
      if (nv1 .ne. nv) then 
         write (6,*) 'Initial guess incompatible with matrix'
         write (6,*) 'nv1 is not equal to nv'
         stop
      endif
      read (8,*) (u(j), j=1,nv)
      close (8)


      print *, 'initial guess read'

      call amg_initialize(data,0)

      call amg_setnumunknowns(numu,data)

      call amg_setnumpoints(nump,data)
      call amg_setlogging(ioutdat, "AMG2.runlog", data)
      call amg_setlevmax(levmax, data)
      call amg_setewt(ewt,data)
      call amg_setnwt(nwt,data)
      call amg_setncg(ncg, data)
      call amg_setecg(ecg, data)
      call amg_setnstr(nstr, data)
      call amg_setncyc(ncyc, data)
 
      call amg_clock_init()
      call amg_clock(time_ticks)
      call amg_cpuclock(cpu_ticks)
      time_old = time_ticks
      cpu_old = cpu_ticks


      call amg_setup(isterr,a,ia,ja,nv,data)
      print *, 'Setup error flag = ', isterr

      call amg_clock(time_ticks)
      call amg_cpuclock(cpu_ticks)

      setup_time = time_ticks - time_old
      setup_cpu = cpu_ticks - cpu_old

      call amg_solve(isverr,u,f,nv,tol,data)
      print *, 'Solve error flag = ', isverr

      call amg_clock(time_ticks)
      call amg_cpuclock(cpu_ticks)
      solve_time = time_ticks - time_old - setup_time
      solve_cpu = cpu_ticks - cpu_old - setup_cpu
      time = time_ticks - time_old
      cpu = cpu_ticks - cpu_old

      print *, ' '
      print *, 'Setup Times: '
c     call print_t(dfloat(setup_time), dfloat(setup_cpu))

      print *, ' '
      print *, 'Solve Times: '
c      call print_t(dfloat(solve_time), dfloat(solve_cpu))
   
      print *, ' '
      print *, 'Total Times: '
c      call print_t(dfloat(time), dfloat(cpu))


      call amg_finalize(data)

      stop

      end

