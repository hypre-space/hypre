c=====================================================================
c
c     interpolation routines
c
c=====================================================================
c
      subroutine intad(kc,kf,ivstar,nun,imin,imax,
     *                 u,f,a,ia,ja,iu,icg,b,ib,jb)
c
c---------------------------------------------------------------------
c
c     interpolation routine (with V* option)
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
      dimension iu (*)
      dimension icg(*)

      dimension ib (*)
      dimension b  (*)
      dimension jb (*)
c
      dimension d(10,10),s(10)
c
c---------------------------------------------------------------------
c
      if(ivstar.eq.0) go to 70
c
c     perform v* step (minimize energy)
c
      do 20 n1=1,nun
      s(n1)=0.e0
      do 10 n2=1,nun
      d(n1,n2)=0.e0
10    continue
20    continue
      iclo=imin(kc)
      ichi=imax(kc)
      do 40 i=iclo,ichi
      n1=iu(i)
      s(n1)=s(n1)+f(i)*u(i)
      jlo=ia(i)
      jhi=ia(i+1)-1
      do 30 j=jlo,jhi
      ii=ja(j)
      n2=iu(ii)
      if(n2.ge.n1) d(n1,n2)=d(n1,n2)+a(j)*u(ii)*u(i)
30    continue
40    continue
c
      do 50 n1=2,nun
      do 50 n2=1,n1-1
50    d(n1,n2)=d(n2,n1)
      call gselim(d,s,nun)
      do 60 i=iclo,ichi
      n=iu(i)
      u(i)=u(i)*s(n)
60    continue
c
c     perform interpolation
c
70    iflo=imin(kf)
      ifhi=imax(kf)
      do 90 if=iflo,ifhi
      jflo=ib(if)
      jfhi=ib(if+1)-1
      if(icg(if).gt.0) jfhi=jflo
      if(jflo.gt.jfhi) go to 90
      do 80 jf=jflo,jfhi
      if2=jb(jf)
      ic=icg(if2)
      u(if)=u(if)+b(jf)*u(ic)
80    continue
90    continue
      return
      end
c
c=====================================================================
c
c     residual calculation/restriction routines
c
c=====================================================================
c
      subroutine rscali(k,kc,imin,imax,u,f,a,ia,ja,icg,b,ib,jb)
c
c---------------------------------------------------------------------
c
c     compute residual & restrict to coarse grid
c     transpose of interpolation is used for restriction
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
      ilo=imin(k)
      ihi=imax(k)
      iclo=imin(kc)
      ichi=imax(kc)
      do 10 i=iclo,ichi
10    f(i)=0.e0
      do 60 i=ilo,ihi
      r=f(i)
      jlo=ia(i)
      jhi=ia(i+1)-1
      do 20 j=jlo,jhi
      r=r-a(j)*u(ja(j))
20    continue
      jlo=ib(i)
      jhi=ib(i+1)-1
      if(icg(i).gt.0) jhi=jlo
      if(jlo.gt.jhi) go to 60
      do 50 j=jlo,jhi
      ic=icg(jb(j))
      f(ic)=f(ic)+r*b(j)
50    continue
60    continue
      return
      end
c
      subroutine rscalr(k,kc,imin,imax,u,f,a,ia,ja,icg,b,ib,jb)
c
c---------------------------------------------------------------------
c
c     compute residual & restrict to coarse grid
c     a stored restriction operator is used
c
c---------------------------------------------------------------------
c
      implicit real*8 (a-h,o-z)
c
c     include 'params.amg'
      common /params/ndimu,ndimp,ndima,ndimb
c
      dimension imin(25),imax(25)
      dimension u  (*)
      dimension f  (*)
      dimension ia (*)
      dimension a  (*)
      dimension ja (*)
      dimension icg(*)
c
      dimension ib (*)
      dimension b  (*)
      dimension jb (*)
c
      dimension res(5000)
c
c---------------------------------------------------------------------
c
      ilo=imin(k)
      ihi=imax(k)
      if(ihi.gt.5000) stop 'res array too small in rscal'
      iclo=imin(kc)
      ichi=imax(kc)
      do 10 i=iclo,ichi
10    f(i)=0.e0
      do 30 i=ilo,ihi
      res(i)=f(i)
      jlo=ia(i)
      jhi=ia(i+1)-1
      do 20 j=jlo,jhi
      res(i)=res(i)-a(j)*u(ja(j))
20    continue
30    continue
      do 60 i=ilo,ihi
      if(icg(i).le.0) go to 60
      ic=icg(i)
      f(ic)=0.e0
      jlo=ib(i)
      jhi=ib(i+1)-1
c     if(icg(i).gt.0) jhi=jlo
      if(jlo.gt.jhi) go to 60
      do 50 j=jlo,jhi
      f(ic)=f(ic)+b(j)*res(jb(j))
50    continue
60    continue
      return
      end
c





c
c=====================================================================
c
c     residual calculation routines
c
c=====================================================================
c
      subroutine rsdl(k,enrg,res,resv,aip,fu,ru,
     *                uu,iprt,imin,imax,u,f,a,ia,ja,iu)
c
c---------------------------------------------------------------------
c
c     compute (and print) residual
c
c---------------------------------------------------------------------
c
      implicit real*8 (a-h,o-z)
c
c     include 'params.amg'
      common /params/ndimu,ndimp,ndima,ndimb
c
      dimension imin(25),imax(25)
      dimension u  (*)
      dimension f  (*)
      dimension ia (*)
      dimension a  (*)
      dimension ja (*)
      dimension iu (*)
c
      dimension resv(10)
c
c---------------------------------------------------------------------
c
      do 10 i=1,10
      resv(i)=0.e0
10    continue
      resp=res
      enrg=0.e0
cveh  test. compute <Au,u>, <r,u>, <f,u>
      aip=0.e0
      fu=0.e0
      ru=0.e0
      uu=0.e0
cveh
      r2=0.e0
      ilo=imin(k)
      ihi=imax(k)
      do 30 i=ilo,ihi
      s=0.e0
      jlo=ia(i)
      jhi=ia(i+1)-1
      do 20 j=jlo,jhi
20    s=s+a(j)*u(ja(j))
      r=s-f(i)
      r2=r*r
cveh
      aip = aip + s*u(i)
      fu = fu + f(i)*u(i)
      ru = ru + r*u(i)
      uu = uu + u(i)*u(i)
cveh
      enrg=enrg+r*u(i)-u(i)*f(i)
      resv(iu(i))=resv(iu(i))+r2
30    continue
c
      res=0.e0
      do 40 i=1,9
      res=res+resv(i)
40    continue
      res=sqrt(res)
      if(iprt.eq.0) return
      rate=res/resp

      write(6,9997) k,enrg,res,rate
      return
c9997  format('  k :',i2,'  a norm :',1p,e9.2,'  residual :',e9.2,
c     *     '  factor :',e9.2)
9997  format('  k :',i2,'  a norm :',1p,e9.2,'  residual :',e9.2,
     *     '  factor :',e9.2)
      end
