c
c==============================================================
c
c     Miscellaneous Test/Output routines
c
c==============================================================
c
      subroutine splot(imin,imax,u,iu,ip,ipmn,ipmx,iv,xp,yp)
      implicit real*8(a-h,o-z)
c
c---------------------------------------------------------------
c
c     Symbolic plot routine
c
c---------------------------------------------------------------
c
      dimension u (*)
      dimension iu(*)
      dimension ip(*)
      dimension iv(*)
      dimension xp(*)
      dimension yp(*)
c
      character*80 iline(80)
      dimension nu(10)
      dimension fx(10)
c
      common /mesh/  hx,hy
c
c---------------------------------------------------------------
c
c     determine output bounds & # of unknowns
c
      do 10 i=1,10
      fx(i)=0.0
10    nu(i)=0
c
      xlo = 1.d+99
      xhi =-1.d+99
      ylo = 1.d+99
      yhi =-1.d+99
      do 20 i=imin,imax
      nu(iu(i))=nu(iu(i))+1
      if(xp(ip(i)).lt.xlo) xlo=xp(ip(i))
      if(xp(ip(i)).gt.xhi) xhi=xp(ip(i))
      if(yp(ip(i)).lt.ylo) ylo=yp(ip(i))
      if(yp(ip(i)).gt.yhi) yhi=yp(ip(i))
      if(dabs(u(i)).gt.fx(iu(i))) fx(iu(i))=dabs(u(i))
20    continue
c
c     Define mesh sizes (problem dependent)
c
      ixlo=xlo/hx
      ixhi=xhi/hx
      iylo=ylo/hy
      iyhi=yhi/hy

      do 100 iun=1,10
      if(nu(iun).eq.0) go to 100
      do 30 iy=iylo,iyhi
30    iline(iy)=' '

      do 40 i=imin,imax
      if(iu(i).ne.iun) go to 40
      ix = xp(ip(i))/hx
      iy = yp(ip(i))/hy
      if(dabs(u(i)).lt.0.25*fx(iu(i))) then
        iline(iy)(ix:ix)='.'
      elseif(u(i).lt.-0.75*fx(iu(i))) then
        iline(iy)(ix:ix)='<'
      elseif(u(i).gt. 0.75*fx(iu(i))) then
        iline(iy)(ix:ix)='>'
      elseif(u(i).lt.-0.50*fx(iu(i))) then
        iline(iy)(ix:ix)='='
      elseif(u(i).gt. 0.50*fx(iu(i))) then
        iline(iy)(ix:ix)='#'
      elseif(u(i).lt.-0.25*fx(iu(i))) then
        iline(iy)(ix:ix)='-'
      elseif(u(i).gt. 0.25*fx(iu(i))) then
        iline(iy)(ix:ix)='+'
      endif
40    continue
c
c     write(*,2000) iun,fx(iun)
      write(6,2000) iun,fx(iun)
2000  format(/'  Plot - iu=',i2,'  max=',1p,d9.2/)
      do 50 iy=iyhi,iylo,-1
c     write(*,1000) iline(iy)
      write(6,1000) iline(iy)
1000  format(a80)
50    continue
100   continue
c     write(*,'(1x)')
      write(6,'(1x)')
c
      return
      end
c
c==============================================================
c
      subroutine rplot(k,imin,imax,u,f,a,ia,ja,
     *                 iu,ip,ipmn,ipmx,iv,xp,yp,
     *                 ndimu,ndimp,ndima,ndimb)
      implicit real*8(a-h,o-z)
c
c---------------------------------------------------------------
c
c     Residual plotting routine
c
c---------------------------------------------------------------
c
c     include 'params.amg'
c
      dimension u  (*)
      dimension f  (*)
      dimension a  (*)
      dimension ia (*)
      dimension ja (*)
      dimension iu (*)
      dimension ip (*)
      dimension iv (*)
      dimension xp (*)
      dimension yp (*)
      dimension imin(*),imax(*)
      dimension ipmn(*),ipmx(*)
c
      dimension r(50000)
c
c---------------------------------------------------------------
c
      do 30 i=imin(k),imax(k)
      r(i)=f(i)
      do 20 j=ia(i),ia(i+1)-1
20    r(i)=r(i)-a(j)*u(ja(j))
30    continue
c
      write(*,'(/''  Level '',i2,'' Residuals ''/)') k
      call splot(imin(k),imax(k),r,iu,ip,
     *           ipmn(k),ipmx(k),iv,xp,yp)
      return
      end
c
