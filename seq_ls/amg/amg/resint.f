
c=====================================================================
c     interpolation routine (with V* option)
c=====================================================================

      subroutine intad(
     *     u, icg, b, ib, jb, n,
     *     uc, fc, ac, iac, jac, iuc, nc,
     *     ivstar, nun)

      implicit real*8 (a-h,o-z)

      dimension u  (*)
      dimension icg(*)

      dimension b  (*)
      dimension ib (*)
      dimension jb (*)

      dimension uc (*)
      dimension fc (*)
      dimension ac (*)
      dimension iac(*)
      dimension jac(*)
      dimension iuc(*)

      dimension d(10,10),s(10)

c---------------------------------------------------------------------

      if(ivstar.eq.0) go to 70

c     perform v* step (minimize energy)

      do 20 n1=1,nun
         s(n1)=0.e0
         do 10 n2=1,nun
            d(n1,n2)=0.e0
 10      continue
 20   continue
      do 40 ic=1,nc
         n1=iuc(ic)
         s(n1)=s(n1)+fc(ic)*uc(ic)
         jclo=iac(ic)
         jchi=iac(ic+1)-1
         do 30 jc=jclo,jchi
            ii=jac(jc)
            n2=iuc(ii)
            if(n2.ge.n1) d(n1,n2)=d(n1,n2)+ac(jc)*uc(ii)*uc(ic)
 30      continue
 40   continue

      do 50 n1=2,nun
         do 55 n2=1,n1-1
            d(n1,n2)=d(n2,n1)
 55      continue
 50   continue
      call gselim(d,s,nun)
      do 60 ic=1,nc
         n2=iuc(ic)
         uc(ic)=uc(ic)*s(n2)
 60   continue

c     perform interpolation

 70   continue
      do 90 i=1,n
         jlo=ib(i)
         jhi=ib(i+1)-1
         if(icg(i).gt.0) jhi=jlo
         if(jlo.gt.jhi) go to 90
         do 80 j=jlo,jhi
            i2=jb(j)
            ic=icg(i2)
            u(i)=u(i)+b(j)*uc(ic)
 80      continue
 90   continue
      return
      end

c=====================================================================
c     restriction routine:
c     compute residual & restrict to coarse grid
c     transpose of interpolation is used for restriction
c=====================================================================

      subroutine rscali(
     *     fc, nc,
     *     u, f, a, ia, ja, icg, b, ib, jb, n)

      implicit real*8 (a-h,o-z)

      dimension fc (*)

      dimension u  (*)
      dimension f  (*)
      dimension a  (*)
      dimension ia (*)
      dimension ja (*)
      dimension icg(*)

      dimension b  (*)
      dimension ib (*)
      dimension jb (*)

c---------------------------------------------------------------------

      do 10 ic=1,nc
         fc(ic)=0.e0
 10   continue
      do 60 i=1,n
         r=f(i)
         jlo=ia(i)
         jhi=ia(i+1)-1
         do 20 j=jlo,jhi
            r=r-a(j)*u(ja(j))
 20      continue
         jlo=ib(i)
         jhi=ib(i+1)-1
         if(icg(i).gt.0) jhi=jlo
         if(jlo.gt.jhi) go to 60
         do 50 j=jlo,jhi
            ic=icg(jb(j))
            fc(ic)=fc(ic)+r*b(j)
 50      continue
 60   continue
      return
      end

c=====================================================================
c     compute (and print) residual
c=====================================================================

      subroutine rsdl(k,enrg,res,resv,aip,fu,ru,uu,
     *     imin,imax,u,f,a,ia,ja,iu)

      implicit real*8 (a-h,o-z)

      dimension u  (*)
      dimension f  (*)
      dimension ia (*)
      dimension a  (*)
      dimension ja (*)
      dimension iu (*)

      dimension imin(25),imax(25)

      dimension resv(10)

c---------------------------------------------------------------------

      do 10 i=1,10
         resv(i)=0.e0
 10   continue
      resp=res
      enrg=0.e0
c     veh  test. compute <Au,u>, <r,u>, <f,u>
      aip=0.e0
      fu=0.e0
      ru=0.e0
      uu=0.e0
c     veh
      r2=0.e0
      ilo=imin(k)
      ihi=imax(k)
      do 30 i=ilo,ihi
         s=0.e0
         jlo=ia(i)
         jhi=ia(i+1)-1
         do 20 j=jlo,jhi
            s=s+a(j)*u(ja(j))
 20      continue
         r=s-f(i)
         r2=r*r
c     veh
         aip = aip + s*u(i)
         fu = fu + f(i)*u(i)
         ru = ru + r*u(i)
         uu = uu + u(i)*u(i)
c     veh
         enrg=enrg+r*u(i)-u(i)*f(i)
         resv(iu(i))=resv(iu(i))+r2
 30   continue

      res=0.e0
      do 40 i=1,9
         res=res+resv(i)
 40   continue
      res=sqrt(res)

      return
      end
