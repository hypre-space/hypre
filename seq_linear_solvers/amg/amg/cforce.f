c     
c=====================================================================
c     
c     the following routines are used for setting initial c-points
c     
c=====================================================================
c     
      subroutine cforce(k,ncolor,imin,imax,ip,icg,ifg,iv)
c     
c---------------------------------------------------------------------
c     
c     initialize the grid
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
c---------------------------------------------------------------------
c     
c     set all points initially to be undefined
c     
      if(ncolor.eq.-1) then
         ilo=imin(k)
         ihi=imax(k)
         do 10 i=ilo,ihi
            icg(i)=-1
 10      continue
      endif

      return
      end
