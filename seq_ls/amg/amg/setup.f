C### filename: ST.FOR
c
c==== FILE ST.FOR ====================================================
c
c     SETUP: setup for systems
c
c=====================================================================
c
      subroutine setup(levels,nstr,ecg,ncg,ewt,nwt,icdep,idump,
     *                 nun,imin,imax,u,f,a,ia,ja,iu,ip,icg,ifg,
     *                 b,ib,jb,ipmn,ipmx,iv,xp,yp,ifc,
     *                 ndimu,ndimp,ndima,ndimb,lfname)
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
      dimension u  (*)
      dimension f  (*)
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
c     vector used for forced coarse grid points (formerly icgfix)
c
      dimension ifc(*)
c
      character*(*)  lfname

c---------------------------------------------------------------------
c open the log file and write some initial info
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
c     set array limits
c
      call memlimit('a',ndima)
      call memlimit('ja',ndima)
      call memlimit('ia',ndimu)
      call memlimit('iu',ndimu)
      call memlimit('ip',ndimu)
      call memlimit('icg',ndimu)
      call memlimit('ifg',ndimu)
      call memlimit('ib',ndimu)
      call memlimit('b',ndimb)
      call memlimit('jb',ndimb)
c
c     set initial data usage (temporary)
c
      call memacct('a',ia(imax(1)+1),1)
      call memacct('ja',ia(imax(1)+1),1)
      call memacct('ia',imax(1)+1,1)
      call memacct('iu',imax(1),1)
      call memacct('ip',imax(1),1)
      call memacct('icg',0,1)
      call memacct('ifg',0,1)
      call memacct('ib',0,1)
      call memacct('b',0,1)
      call memacct('jb',0,1)
c
c     load ifg into ifc
c
c      do 10 i=1,imax(1)
c10    ifc(i)=ifg(i)
c
c=>   set form of fine grid matrix & print statistics
c
      call trunc(1,imin,imax,a,ia,ja)
c
      call symm(1,1,imin,imax,a,ia,ja,icg,ifg)
c
c     set fine level data usage (permanent)
c
      call memacct('a',ia(imax(1)+1),1)
      call memacct('ja',ia(imax(1)+1),1)
      call memacct('ia',imax(1)+1,1)
      call memacct('iu',imax(1),1)
      call memacct('ip',imax(1),1)
      call memacct('ib',imax(1)+1,1)
c
      call msample
c
      if(levels.le.1) then
        close(6)
        return
      endif
c
c===> coarsen problem
c
20    k=k+1
c
c=>   choose coarse grid and define interpolation
c
      call crsgd(k-1,nstr,ecg,ncg,ewt,nwt,levels,icdep,
     *           nun,imin,imax,a,ia,ja,iu,ip,icg,ifg,
     *           b,ib,jb,ipmn,ipmx,iv,xp,yp,ifc,
     *           ndimu,ndimp,ndima,ndimb)
c
c=>   test level k-1 matrix
c
      call stats(k-1,levels,idump,
     *           nun,imin,imax,a,ia,ja,iu,ip,icg,ifg,
     *           b,ib,jb,ipmn,ipmx,iv,xp,yp)

c     call memoryx(k-1,imin,imax,ia,ib,ipmn,ipmx)
c
c=>   test for coarsest grid
c
      if(k.gt.levels) go to 30
c
c=>   compute coarse grid matrix
c
      call opdfn(k,levels,ierr,ndima,
     *           imin,imax,a,ia,ja,icg,ifg,b,ib,jb)
c
c=>   set form of coarse grid matrix
c
      call trunc(k,imin,imax,a,ia,ja)

      call symm(k,isymm,imin,imax,a,ia,ja,icg,ifg)
c
c     set final data usage (permanent)
c
      call memacct('a',ia(imax(k)+1),1)
      call memacct('ja',ia(imax(k)+1),1)
      call memacct('ia',imax(k)+1,1)
      call memacct('iu',imax(k),1)
      call memacct('ip',imax(k),1)
      call memacct('icg',imax(k-1),1)
      call memacct('ib',imax(k)+1,1)
      call memacct('b',ib(imax(k-1)+1),1)
      call memacct('jb',ib(imax(k-1)+1),1)
c
      call msample
c
      go to 20
c
30    continue
c
      call msample
c
c     test operators
c
c     do 40 k=2,levels
c     call testa (k,imin,imax,u,f,a,ia,ja,icg,b,ib,jb)
c     call testa2(k,imin,imax,u,f,a,ia,ja,icg,b,ib,jb)
c40   continue
c
c=>   compress unneeded entries from interpolation/restriction
c
c     call crushb(levels,imin,imax,icg,b,ib,jb)
c
      call mreport(1)

      close(6)

      return
      end
