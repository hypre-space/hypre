c==== FILE DR.FOR ====================================================
c
c     AMGS01 :  main driver routine for AMGS01
c
c=====================================================================
c
      subroutine amgs01(u, f, a, ia, ja,
     *                  nv, nu, np, iu, ip, iv, xp, yp, zp,
     *                  ifc, isw, lfname)
c
c---------------------------------------------------------------------
c
c     Version 1/9/95
c
c     1. Written as "callable" solver, rather than previous format.
c
c     2. This version does not use the common statements used in
c     previous amg system codes. instead, needed arrays are
c     passed in. arrays "local" to amg are dimensioned locally.
c
c     3. Dimensions are defined in PARAMS.INC
c
c        ndima = total matrix size (for all levels)
c        ndimu = total number of variables on all levels
c        ndimp = total number of points on all levels
c        ndimb = total size of interpolation matrices, all levels
c
c     4. AMG setup/solution parameters are read from file AMG.DAT
c
c     5. required input:
c
c        ia,a,ja  - amg matrix (usual format)
c        iu       - variable/unknown correspondence
c        ip       - variable/point correspondence
c        ifc      - forced coarse grid information (pointwise)
c        xp,yp    - point coordinates
c        nv       - number of variables
c        nu       - number of unknowns
c        np       - number of points
c
c     6. Arrays defined locally:
c
c        imin,imax,ipmn,ipmx
c        iv
c        icf,ifg
c        ib,b,jb
c
c---------------------------------------------------------------------
c
      implicit real*8 (a-h,o-z)
      integer TOLD,TNEW,TAMG
      CHARACTER*24 DT
c
c=>   parameter statement
c
      include 'params.amg'
c
c---- INPUT ARRAYS ---------------------------------------------------
c
c=>   amg matrix
c
      dimension ia (*)
      dimension a  (*)
      dimension ja (*)
c
c=>   solution & rhs
c
      dimension u  (*)
      dimension f  (*)
c
c=>   variable/unknown correspondence
c
      dimension iu (*)
c
c=>   variable/point correspondence
c
      dimension ip (*)
c
c=>   point/variable correspondence
c
      dimension iv (*)
c
c=>   point-oriented arrays (2-D)
c
      real*8 xp (*)
      real*8 yp (*)
      real*8 zp (*)
c
c=>   forced coarse grid information
c
      dimension ifc(*)
c
      character*(*)  lfname
c
c---- LOCAL ARRAYS ---------------------------------------------------
c
c=>   level bounds
c
      dimension imin(25),imax(25)
      dimension ipmn(25),ipmx(25)
c
c=>   fine/coarse grid pointers
c
      dimension icg(ndimu)
      dimension ifg(ndimu)
c
c=>   interpolation/restriction weights
c
      dimension ib (ndimu)
      dimension b  (ndimb)
      dimension jb (ndimb)
c
c=>   coarsening dependence array
c
      dimension icdep(10,10)
c
c=>   solution parameters
c
      dimension mu (25)
      dimension ntrlx(4)
      dimension iprlx(4)
      dimension ierlx(4)
      dimension iurlx(4)
c
      SAVE

c---------------------------------------------------------------------
c open the log file and write some initial info
c---------------------------------------------------------------------

      open(6,file=lfname,status='UNKNOWN')

      CALL FDATE(DT)
      WRITE(6,1559) lfname,DT
1559  FORMAT(' FILE:',A15,10X,'DATE:',A24)
      write(6,1550)
1550  format(/'  NEW VERSION'/)
      write(6,1553)
1553  format(/'  AMG OUTPUT:'/)

c---------------------------------------------------------------------
c write some debugging info
c---------------------------------------------------------------------

CVEH
      write (6,*) ('in AMGs01')
      write (6,*) nu,np,nv,'    nu, np, nv'
      write (6,*) ia(1), ia(2), ia(nv), ia(nv+1), ' ia(1, 2, nv, nv+1)'  
      write(6,*) ja(1), ja(2), ja(ia(nv+1)-1), '  ja(1, 2, ia(nv+1)-1)'
      write (6,*) a(1), a(2), a(ia(nv+1)-1),'  a(1, 2, ia(nv+1)-1)'
      write (6,*) iu(1), iu(nv), '  iu(1), iu(nv)'
      write (6,*) ip(1), ip(2), ip(nv),'  ip(1), ip(2), ip(nv)'
      write (6,*) u(1), u(nv), '  u(1), u(nv)'
      write (6,*) f(1), f(nv),'  f(1), f(nv)'
      write (6,*) ifc(1), ifc(nv), '  ifc(1, nv)'
      write (6,*) xp(1), xp(nv), '  xp(1, nv)'
      write (6,*) yp(1), yp(nv), '  yp(1, nv)'
      write(6,'(1x)')
CVEH

c---------------------------------------------------------------------
c start timing
c---------------------------------------------------------------------

      CALL CTIME(TOLD)

c
c---------------------------------------------------------------------
c
c     set fine level point & variable bounds
c
      nun=nu
      ipmn(1)=1
      ipmx(1)=np
      imin(1)=1
      imax(1)=nv
c
c     call AMGS01 setup/solve routines
c
c===> perform setup/solve or solve only
c
c       isw = 1 - read amg data, set fg form, setup, solve
c             2 -                                    solve
c
      go to (100,200),isw
c
c===> read amg setup/solve parameters
c
100   call input(levmax,ecg,ncg,ewt,nwt,nstr,icdep,
     *           ncyc,mu,ntrlx,iprlx,ierlx,iurlx,
     *           ioutdat,ioutgrd,ioutmat,ioutres,ioutsol,
     *           ipltgrd,ipltsol)

      if(ioutdat.ne.0) then
c
c       write info to outfile
c
        write(6,'(1x)')
        write(6,'(''Reading AMG Setup/Solve parameters'')')
c       write(6,'(1x)')
c       write(6,'(''     Filename: '',a)') 'amg.dat'
c       write(6,'(1x)')
c
c===>   echo data file to output
c
        call printf('amg.dat')
c       write(6,'(1x)')
      endif
c
c=>   set parameters if needed
c
      levels=levmax
      if (levels.eq.0) stop 'levels=0'
c
c===> call amg setup
c
      print *,'  calling setup in dr'
c     call outm(imin(1),imax(1),a,ia,ja,iu,ip,ipmn,ipmx,iv,xp,yp)

      call ctime(itold)

      call setup(levels,nstr,ecg,ncg,ewt,nwt,icdep,ioutmat,
     *           nu,imin,imax,u,f,a,ia,ja,iu,ip,icg,ifg,
     *           b,ib,jb,ipmn,ipmx,iv,xp,yp,ifc)

      call ctime(itnew)
      itnew = itnew-itold

      write (6,8888) itnew 
 8888 format (//' Setup phase complete'/
     *        'TIME for setup phase ',I7,' sec'/) 
c
c===> perform desired output
c
c     if(ipltgrd.eq.1) then
c       call plotg(levels,ipmn,ipmx,xp,yp)
c     elseif(ipltgrd.eq.2) then
c       call plotgr(levels,nun,imin,imax,ia,ja,iu,ip,
c    *              icg,ipmn,ipmx,xp,yp)
c     endif
c
c===> set initial approximation & rhs
c
      rndu=.37632
      rndu=.5429
c     rndu=4.0
c
      call putu(1,rndu,imin,imax,u,iu,ip,xp,yp)
      irhs=0
      call putf(1,irhs,imin,imax,f,iu,ip,xp,yp)
c
c===> call amg solve
c
200   call solve(levels,ncyc,mu,ntrlx,iprlx,ierlx,iurlx,ioutres,
     *     nun,imin,imax,u,f,a,ia,ja,iu,icg,b,ib,jb,
     *     ipmn,ipmx,iv,ip,xp,yp)
c
c===> perform desired output
c
c     if(ipltsol.gt.0) then
c       call plotf(nun,u,iu,ip,ipmn,ipmx,iv,xp,yp)
c     endif
c
c     output the solution
c
      if(ioutsol.ne.0) then
c       output routine here
        call splot(imin,imax,u,iu,ip,ipmn,ipmx,iv,xp,yp)
      endif

c---------------------------------------------------------------------
c end timing
c---------------------------------------------------------------------

      CALL CTIME(TNEW)
      TAMG=TNEW-TOLD

c---------------------------------------------------------------------
c write some info and close the log file
c---------------------------------------------------------------------

      WRITE (6,9000) TAMG
9000  FORMAT(///'***** Running TIME (TAMG) :',I10,' SEC *****'/)

      CLOSE(6)

c
      return
      end
