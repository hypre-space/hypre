c
c=====================================================================
c
c     routines to test matrices & print statistics
c
c=====================================================================
c
      subroutine testa(k,imin,imax,u,f,a,ia,ja,icg,b,ib,jb)
c
c---------------------------------------------------------------------
c
c     test matrix a (restriction = transpose(interpolation))
c
c---------------------------------------------------------------------
c
      implicit real*8 (a-h,o-z)
c
      dimension imin(25),imax(25)
      dimension u  (*)
      dimension f  (*)
      dimension ia (*)
      dimension a  (*)
      dimension ja (*)
      dimension icg(*)

      dimension ib (*)
      dimension b  (*)
      dimension jb (*)
c
c---------------------------------------------------------------------
c
      iclo=imin(k)
      ichi=imax(k)
c
c     generate random function
c
      seed=.51e0
      unrm=0.e0
      do 10 i=iclo,ichi
      d=10000.e0*exp(seed)
      seed=d-float(int(d))
      u(i)=seed-.5e0
      unrm=unrm+u(i)*u(i)
   10 continue
      unrm=sqrt(unrm)
c
c      interpolate to next finer grid
c
      iflo=imin(k-1)
      ifhi=imax(k-1)
      do 40 i=iflo,ifhi
      if(icg(i).le.0) go to 20
      u(i)=u(icg(i))
      go to 40
   20 jlo=ib(i)
      jhi=ib(i+1)-1
      u(i)=0.e0
      do 30 j=jlo,jhi
      u(i)=u(i)+b(j)*u(icg(jb(j)))
   30 continue
   40 continue
c
c     apply fine grid operator
c
      do 60 i=iflo,ifhi
      f(i)=0.e0
      jlo=ia(i)
      jhi=ia(i+1)-1
      do 50 j=jlo,jhi
      f(i)=f(i)+a(j)*u(ja(j))
   50 continue
   60 continue
c
c     restrict the function
c
      do 70 i=iclo,ichi
      f(i)=0.e0
   70 continue
      do 100 i=iflo,ifhi
      if(icg(i).le.0) go to 80
      f(icg(i))=f(icg(i))+f(i)
      go to 100
   80 jlo=ib(i)
      jhi=ib(i+1)-1
      do 90 j=jlo,jhi
      iic=icg(jb(j))
      f(iic)=f(iic)+b(j)*f(i)
   90 continue
  100 continue
c
c     apply the coarse grid operator and compare
c
      err=0.e0
      do 120 i=iclo,ichi
      er=f(i)
      jlo=ia(i)
      jhi=ia(i+1)-1
      do 110 j=jlo,jhi
      er=er-a(j)*u(ja(j))
  110 continue
      err=err+er*er
  120 continue
      err=sqrt(err)
      write(6,1000) k,unrm,err
 1000 format('  testa  k=',i2,'  unorm=',e10.3,'  err norm=',e10.3)
      return
      end
c
      subroutine testa2(k,imin,imax,u,f,a,ia,ja,icg,b,ib,jb)
c
c---------------------------------------------------------------------
c
c     test matrix a (restriction # transpose(interpolation))
c
c---------------------------------------------------------------------
c
      implicit real*8 (a-h,o-z)
c
      dimension imin(25),imax(25)
      dimension u  (*)
      dimension f  (*)
      dimension ia (*)
      dimension a  (*)
      dimension ja (*)
      dimension icg(*)

      dimension ib (*)
      dimension b  (*)
      dimension jb (*)
c
c---------------------------------------------------------------------
c
      iclo=imin(k)
      ichi=imax(k)
c
c     generate random function
c
      seed=.51e0
      unrm=0.e0
      do 10 i=iclo,ichi
      d=10000.e0*exp(seed)
      seed=d-float(int(d))
      u(i)=seed-.5e0
      unrm=unrm+u(i)*u(i)
   10 continue
      unrm=sqrt(unrm)
c
c      interpolate to next finer grid
c
      iflo=imin(k-1)
      ifhi=imax(k-1)
      do 40 i=iflo,ifhi
      if(icg(i).le.0) go to 20
      u(i)=u(icg(i))
      go to 40
   20 jlo=ib(i)
      jhi=ib(i+1)-1
      u(i)=0.e0
      do 30 j=jlo,jhi
      u(i)=u(i)+b(j)*u(icg(jb(j)))
   30 continue
   40 continue
c
c     apply fine grid operator
c
      do 60 i=iflo,ifhi
      f(i)=0.e0
      jlo=ia(i)
      jhi=ia(i+1)-1
      do 50 j=jlo,jhi
      f(i)=f(i)+a(j)*u(ja(j))
   50 continue
   60 continue
c
c     restrict the function
c
      do 70 i=iclo,ichi
      f(i)=0.e0
   70 continue
      do 100 i=iflo,ifhi
      if(icg(i).le.0) go to 100
      ic=icg(i)
      jlo=ib(i)
      jhi=ib(i+1)-1
      do 90 j=jlo,jhi
      if=jb(j)
      f(ic)=f(ic)+b(j)*f(if)
   90 continue
  100 continue
c
c     apply the coarse grid operator and compare
c
      err=0.e0
      do 120 i=iclo,ichi
      er=f(i)
      jlo=ia(i)
      jhi=ia(i+1)-1
      do 110 j=jlo,jhi
      er=er-a(j)*u(ja(j))
  110 continue
      err=err+er*er
  120 continue
      err=sqrt(err)
      write(6,1000) k,unrm,err
 1000 format('  testa2  k=',i2,'  unorm=',e10.3,'  err norm=',e10.3)
      return
      end
c
      subroutine stats(k,levels,idump,
     *                 nun,imin,imax,a,ia,ja,iu,ip,icg,ifg,
     *                 b,ib,jb,ipmn,ipmx,iv,xp,yp)
c
c---------------------------------------------------------------------
c
c     level k matrix statistics
c
c---------------------------------------------------------------------
c
      implicit real*8 (a-h,o-z)
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
      dimension iarr(10)
c
c---------------------------------------------------------------------
c
c===> decode idump
c
      call idec(idump,3,ndig,iarr)
c
      iout  = iarr(1)
      kdump = iarr(2)
      ndump = iarr(3)

      if(iout.eq.0) return

      if(k.gt.1) go to 10
      write(6,4000)
      write(6,4001)
10    if(k.le.levels) go to 20
      write(6,4001)
      return

20    call testsy(k,nun,imin,imax,a,ia,ja,iu)
      if(k.lt.levels) call testb(k,nun,imin,imax,b,ib,jb,iu,icg)
c
c     print matrix/interpolation info
c
      if(k.eq.levels) iout=min0(iout,3)
c     if(k.ge.kdump.and.k.le.kdump+ndump)
c    *     call outbs(k,imin,imax,icg,b,ib,jb)
      if(k.ge.kdump.and.k.le.kdump+ndump)
     *              call outa(k,iout,
     *                        imin,imax,a,ia,ja,iu,ip,icg,ifg,
     *                        b,ib,jb,ipmn,ipmx,iv,xp,yp)
      return
 4000 format(3x,' matrix    block   sizes   entries per row         ',
     *       '  block row sums'/
     *       3x,' k  i  j   rows entries    min   max   avg         ',
     *       ' min          max')
 4001 format(4x,72('='))
      end
c
      subroutine testsy(k,nun,imin,imax,a,ia,ja,iu)
c
c---------------------------------------------------------------------
c
c     test blocks
c
c---------------------------------------------------------------------
c
      implicit real*8 (a-h,o-z)
c
      dimension imin(25),imax(25)
      dimension ia (*)
      dimension a  (*)
      dimension ja (*)
      dimension iu (*)
c
      dimension rsmn(10,10),rsmx(10,10),npts(10),nnze(10,10)
      dimension ncmn(10,10),ncmx(10,10),rs(10),nc(10)
c
c---------------------------------------------------------------------
c
c     write(6,9876) k,imin(k),imax(k)
 9876 format('  testsy - k,imin,imax=',3i5)
      do 20 n1=1,nun
      npts(n1)=0
      do 10 n2=1,nun
      rsmn(n1,n2)=+1.e38
      rsmx(n1,n2)=-1.e38
      nnze(n1,n2)=0
      ncmn(n1,n2)=16384
      ncmx(n1,n2)=0
   10 continue
   20 continue
      ilo=imin(k)
      ihi=imax(k)
      do 60 i=ilo,ihi
      n1=iu(i)
      if(n1.le.0.or.n1.gt.nun) then
        write(6,'(/''  ERROR IN TESTSY -- IU('',i5,'')='',i5)') i,n1
        stop
      endif
      do 30 n=1,10
      rs(n)=0.e0
      nc(n)=0
   30 continue
      npts(n1)=npts(n1)+1
c     write(6,9871) i,iu(i),npts(n1)
 9871 format('  testsy - i,iu,npts=',3i5)
      jlo=ia(i)
      jhi=ia(i+1)-1
      do 40 j=jlo,jhi
      ii=ja(j)
      if(j.eq.jlo.or.ii.ne.i) go to 39
      write(6,3900) i
 3900 format(/' *** error -- row ',i4,' contains duplicate entries')
   39 n2=iu(ii)
c     if(n2.gt.3) write(6,9872) i,j,ii,n2
 9872 format('  fuck up -- i,j,ii,iu(ii)=',4i4)
      rs(n2)=rs(n2)+a(j)
      nnze(n1,n2)=nnze(n1,n2)+1
      nc(n2)=nc(n2)+1
   40 continue
      do 50 n=1,nun
      if(abs(rs(n)).gt.rsmx(n1,n)) rsmx(n1,n)=abs(rs(n))
      if(abs(rs(n)).lt.rsmn(n1,n)) rsmn(n1,n)=abs(rs(n))
      if(nc(n).gt.ncmx(n1,n)) ncmx(n1,n)=nc(n)
      if(nc(n).lt.ncmn(n1,n)) ncmn(n1,n)=nc(n)
   50 continue
   60 continue
      do 70 n1=1,nun
      if(npts(n1).gt.0) nmax=n1
   70 continue
      do 90 n1=1,nmax
      if(npts(n1).eq.0) go to 90
      do 80 n2=1,nmax
      if(npts(n2).eq.0) go to 90
      rsav=float(nnze(n1,n2))/float(npts(n1))
      write(6,1000) k,n1,n2,npts(n1),nnze(n1,n2),ncmn(n1,n2),
     *              ncmx(n1,n2),rsav,rsmn(n1,n2),rsmx(n1,n2)
   80 continue
   90 continue
      return
 1000 format(3x,i2,2(2x,i1),2(3x,i5),3x,i3,3x,i3,1x,f5.1,2(3x,e10.3))
      end
