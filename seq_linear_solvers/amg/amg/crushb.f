c
c=====================================================================
c
c     routine to set form of interpolation
c
c=====================================================================
c
      subroutine crushb(levels,imin,imax,icg,b,ib,jb)
c
c---------------------------------------------------------------------
c
c     this converts b (w) to the form required for solution.
c     (i.e., restriction weights are compressed out)
c
c---------------------------------------------------------------------
c
      implicit real*8 (a-h,o-z)
c
      dimension imin(25),imax(25)
      dimension icg(*)

      dimension ib (*)
      dimension b  (*)
      dimension jb (*)
c
c---------------------------------------------------------------------
c
      ilo=imin(1)
      ihi=imax(levels-1)
      kb=1
      do 130 i=ilo,ihi
      if(icg(i).le.0) go to 110
      ib(i)=kb
      b(kb)=1.e0
      jb(kb)=i
      kb=kb+1
      go to 130
  110 icg(i)=0
      jlo=ib(i)
      jhi=ib(i+1)-1
      ib(i)=kb
      if(jlo.gt.jhi) go to 130
      do 120 j=jlo,jhi
      b(kb)=b(j)
      jb(kb)=jb(j)
      kb=kb+1
  120 continue
  130 continue
      ib(ihi+1)=kb
      return
      end
