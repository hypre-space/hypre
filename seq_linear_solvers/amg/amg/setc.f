c
c=====================================================================
c
c     the following routine sets the vectors for the level k+1
c
c=====================================================================
c
      subroutine setc(k,imin,imax,iu,ip,icg,ifg,ipmn,ipmx,iv,xp,yp,
     *                ndimu,ndimp,ndima,ndimb)
c
c---------------------------------------------------------------------
c
c     set needed pointers for coarse grid
c
c---------------------------------------------------------------------
c
      implicit real*8 (a-h,o-z)
c
c     include 'params.amg'
c
      dimension imin(25),imax(25)
      dimension iu (*)
      dimension ip (*)
      dimension icg(*)
      dimension ifg(*)

      dimension ipmn(25),ipmx(25)
      dimension iv (*)
      real*8 xp (*)
      real*8 yp (*)
c
c---------------------------------------------------------------------
c
c     set icg and ifg for level k+1
c
      ilo=imin(k)
      ihi=imax(k)
      ic=ihi
      ipc=ipmx(k)
      imin(k+1)=ic+1
      ipmn(k+1)=ipc+1
      ipold=0
      do 220 i=ilo,ihi
ccjwr >>>>> test change
c     if(icg(i).le.0) icg(i)=0
c     if(icg(i).eq.0) go to 220
      if(icg(i).le.0) go to 220
ccjwr <<<<<
      ic=ic+1
      ipf=ip(i)
      if(ipf.eq.ipold) go to 200
      ipc=ipc+1
      ipold=ipf
      iv(ipc)=ic
      xp(ipc)=xp(ipf)
      yp(ipc)=yp(ipf)
  200 ip(ic)=ipc
      icg(i)=ic
      ifg(ic)=i
      icg(ic)=0
      iu(ic)=iu(i)
  220 continue
      imax(k+1)=ic
      ipmx(k+1)=ipc
      iv(ipc+1)=ic+1
      if(k.eq.1) return
c
c        reset ifg for level k
c
      ifflo=imin(k-1)
      iffhi=imax(k-1)
      do 510 iff=ifflo,iffhi
      if(icg(iff).gt.0) ifg(icg(iff))=iff
  510 continue
      return
      end
