c
      subroutine ctest(k,imin,imax,ip,icg,ifg,iv,ifc)
c
c---------------------------------------------------------------------
c
c     CTEST forces a predetermined coarse grid
c
c     NOTES:
c
c     1. the vector ifc is defined pointwise, and only on the
c     fine grid. ifc(ip)=k, where k is the coarsest level on
c     which the point appears.
c
c---------------------------------------------------------------------
c
      implicit real*8 (a-h,o-z)
c
      dimension imin(25),imax(25)
      dimension ip (*)
      dimension icg(*)
      dimension ifg(*)

      dimension iv (*)
c
      dimension ifc(*)
c
c---------------------------------------------------------------------
c
      ilo=imin(k)
      ihi=imax(k)
      do 20 i=ilo,ihi
      if(i.eq.iv(ip(i))) then
        if =i
        do 10 l=1,k-1
        if=ifg(if)
10      continue
        ipf=ip(if)
        kc=ifc(ipf)
        iicg=0
        if(kc.gt.k) iicg=1
      endif
      if(icg(i).ne.-1) go to 20
      if(iicg.eq.1) icg(i)=1
20    continue
      return
      end
