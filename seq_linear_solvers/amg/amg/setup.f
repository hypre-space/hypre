c=====================================================================
c     
c     SETUP: setup for systems
c     
c=====================================================================
c     
      subroutine setup(ierr,levels,nstr,ecg,ncg,ewt,nwt,icdep,ioutdat,
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
     
      dimension imin(*),imax(*)
      dimension ia (*)
      dimension a  (*)
      dimension ja (*)
      dimension iu (*)
      dimension ip (*)
      dimension icg(*)
      dimension ifg(*)

      dimension ipmn(*),ipmx(*)
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
c     initialize the error flag to zero
c---------------------------------------------------------------------

      ierr = 0

c---------------------------------------------------------------------
c     open the log file and write some initial info
c---------------------------------------------------------------------

      if (ioutdat .ge. 2) then
         open(9,file=lfname,access='append')
         write(9,9000)
      endif
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
cveh 
cveh  calls to trunc and symm are removed to preserve original
cveh  matrix as much as possible.
cveh
cveh      call trunc(1,imin,imax,a,ia,ja)
c     
cveh      call symm(1,1,imin,imax,a,ia,ja,icg,ifg)
c     
      if(levels.le.1) then
         if (ioutdat .ne. 0) close(9)
         return
      endif
c     
c===  > coarsen problem
c     
 20   k=k+1
c     
c     =>   choose coarse grid and define interpolation
c     
      call crsgd(ierr,k-1,nstr,ecg,ncg,ewt,nwt,levels,icdep,
     *     nun,imin,imax,a,ia,ja,iu,ip,icg,ifg,
     *     b,ib,jb,ipmn,ipmx,iv,xp,yp,
     *     ndimu,ndimp,ndima,ndimb)

      if (ierr .ne. 0) return
c     
c     =>   test for coarsest grid
c     
      if(k.gt.levels) go to 30
c     
c     =>   compute coarse grid matrix
c     
      call opdfn(ierr,k,levels,ndima,
     *     imin,imax,a,ia,ja,icg,ifg,b,ib,jb)
      if (ierr .ne. 0) return
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
    
      if (ioutdat .ge. 2) then
         call fstats(ierr,k-1,levels,
     *        nun,imin,imax,a,ia,ja,iu,ip,icg,ifg,
     *        b,ib,jb,ipmn,ipmx,iv,xp,yp)

          close(9)
      endif

      return
      end
