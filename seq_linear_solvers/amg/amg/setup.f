C###  filename: ST.FOR
c     
c==== FILE ST.FOR ====================================================
c     
c     SETUP: setup for systems
c     
c=====================================================================
c     
      subroutine setup(levels,nstr,ecg,ncg,ewt,nwt,icdep,idump,
     *     nun,imin,imax,a,ia,ja,iu,ip,icg,ifg,
     *     b,ib,jb,ipmn,ipmx,iv,xp,yp,
     *     ndimu,ndimp,ndima,ndimb,lfname)
c     
c---------------------------------------------------------------------
c     
c     set up amg components
c     
c---------------------------------------------------------------------
c     
      implicit real*8 (a-h,o-z)
c     
c     include 'params.amg'
c     
      dimension imin(25),imax(25)
      dimension ia (*)
      dimension a  (*)
      dimension ja (*)
      dimension iu (*)
      dimension ip (*)
      dimension icg(*)
      dimension ifg(*)

      dimension ipmn(25),ipmx(25)
      dimension iv (*)
      real*8 xp (*)
      real*8 yp (*)

      dimension ib (*)
      dimension b  (*)
      dimension jb (*)
c     
      dimension icdep(10,10)
c     
      character*(*)  lfname

c---------------------------------------------------------------------
c     open the log file and write some initial info
c---------------------------------------------------------------------

      open(6,file=lfname,access='append')

      write(6,9000)
 9000 format(/'AMG SETUP INFO:'/)

c     
c---------------------------------------------------------------------
c     
c     initialize & set fine level bounds
c     
      ib(1)=1
      k=1
c     
c     =>   set form of fine grid matrix & print statistics
c     
      call trunc(1,imin,imax,a,ia,ja)
c     
      call symm(1,1,imin,imax,a,ia,ja,icg,ifg)
c     
      if(levels.le.1) then
         close(6)
         return
      endif
c     
c===  > coarsen problem
c     
 20   k=k+1
c     
c     =>   choose coarse grid and define interpolation
c     
      call crsgd(k-1,nstr,ecg,ncg,ewt,nwt,levels,icdep,
     *     nun,imin,imax,a,ia,ja,iu,ip,icg,ifg,
     *     b,ib,jb,ipmn,ipmx,iv,xp,yp,
     *     ndimu,ndimp,ndima,ndimb)
c     
c     =>   test for coarsest grid
c     
      if(k.gt.levels) go to 30
c     
c     =>   compute coarse grid matrix
c     
      call opdfn(k,levels,ierr,ndima,
     *     imin,imax,a,ia,ja,icg,ifg,b,ib,jb)
c     
c     =>   set form of coarse grid matrix
c     
      call trunc(k,imin,imax,a,ia,ja)

      call symm(k,isymm,imin,imax,a,ia,ja,icg,ifg)
c     
      go to 20
c     
 30   continue
c     
c     compute & print statistics after coarsening

      call fstats(k-1,levels,idump,
     *     nun,imin,imax,a,ia,ja,iu,ip,icg,ifg,
     *     b,ib,jb,ipmn,ipmx,iv,xp,yp)

      close(6)

      return
      end
