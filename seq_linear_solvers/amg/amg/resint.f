
c=====================================================================
c     interpolation routine (with V* option)
c=====================================================================

      subroutine intad(
     *     u, icg, b, ib, jb, n, vtmp,
     *     uc, fc, ac, iac, jac, iuc, nc,
     *     ivstar, nun)

      implicit real*8 (a-h,o-z)

      dimension vtmp (*)
      dimension u    (*)
      dimension icg  (*)

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

c      call vcopy(u,vtmp,n)
      
c      do 101 i=1,n
c         vtmp(i) = u(i)
c101   continue


      alpha = 1.0
      beta = 1.0
      call matvec(n,alpha,b,ib,jb,uc,beta,u,0)

      return
      end

c=====================================================================
c     restriction routine:
c     compute residual & restrict to coarse grid
c     transpose of interpolation is used for restriction
c=====================================================================

      subroutine rscali(
     *     fc, nc,
     *     u, f, vtmp, a, ia, ja, icg, b, ib, jb, n)

      implicit real*8 (a-h,o-z)

      dimension vtmp(*)
      dimension fc  (*)

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

cveh      compute residual by using matvec
cveh      here vtmp = f-Au


      call vcopy(f,vtmp,n)
 
      alpha = -1.0
      beta = 1.0
      call matvec(n,alpha,a,ia,ja,u,beta,vtmp,0)

cveh      perform restriction using matvec

      alpha = 1.0
      beta = 0.0
      jtrans = 1

      call matvec(n,alpha,b,ib,jb,vtmp,beta,fc,jtrans)

      return
      end

c=====================================================================
c     compute (and print) residual
c=====================================================================

      subroutine rsdl(enrg,res,numu,resv,vtmp,
     *     imin,imax,u,f,a,ia,ja,iu)

      implicit real*8 (a-h,o-z)
  
      dimension vtmp (*)

      dimension u  (*)
      dimension f  (*)
      dimension ia (*)
      dimension a  (*)
      dimension ja (*)
      dimension iu (*)

c      dimension imin(*),imax(*)

      dimension resv(*)

c---------------------------------------------------------------------

      do 10 i=1,numu
         resv(i)=0.e0
 10   continue
      enrg=0.e0

cveh   new residual routine, using matvec
cveh   here vtmp = Au-f

      r2=0.e0
      ilo=imin
      ihi=imax
      nv = ihi-ilo+1

      call vcopy(f,vtmp,nv)

      alpha = 1.0
      beta = -1.0
      call matvec(nv,alpha,a,ia,ja,u,beta,vtmp,0)
     
      do 31 j = 1,nv
         rtmp = vtmp(j)*vtmp(j)
         r2 = r2 + rtmp
         resv(iu(j)) = resv(iu(j)) + rtmp
         enrg=enrg+vtmp(j)*u(j)-u(j)*f(j)
31    continue
      res = sqrt(r2)

      return
      end








