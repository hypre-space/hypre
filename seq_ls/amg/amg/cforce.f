c
c=====================================================================
c
c     the following routines are used for forcing initial c-points
c
c=====================================================================
c
      subroutine cforce(k,ifcg,ncolor,imin,imax,ip,icg,ifg,iv,ifc)
c
c---------------------------------------------------------------------
c
c     initialize the grid and force coarse grid points
c
c     currently, this is only done using predetermined grids
c
c     when a point i is forced to be a c-point, icg(i) is set to 1.
c     In setcw, i will be assigned a maximal weight, which cannot
c     be reduced during coloring.
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
c     set all points initially to be undefined
c
      if(ncolor.eq.-1) then
        ilo=imin(k)
        ihi=imax(k)
        do 10 i=ilo,ihi
        icg(i)=-1
10      continue
      endif

      if(ifcg.ne.0) call ctest(k,imin,imax,ip,icg,ifg,iv,ifc)

      return
      end
