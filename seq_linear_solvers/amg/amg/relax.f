c=====================================================================
c     
c     relaxation routines
c     
c=====================================================================
c     
      subroutine relax(ierr,itrel,iprel,ierel,iurel,
     *     imin,imax,u,f,a,ia,ja,iu,icg,ipmn,ipmx,iv)
c     
c---------------------------------------------------------------------
c     
c     Routine to call relaxation
c     
c     itrel = 1 - Gauss-Seidel
c     itrel = 2 - Kaczmarz
c     itrel = 3 - Point Gauss-Seidel
c     itrel = 4 - Point Kaczmarz (not in effect)
c     itrel = 5 - Collective relaxation (not in effect)
c     itrel = 8 - Normalization
c     itrel = 9 - Direct solver
c     
c     iprel specifies C/F/G variables to relax
c     ierel specifies equation types to relax
c     iurel specifies unknown  types to relax
c     
c---------------------------------------------------------------------
c     
      implicit real*8 (a-h,o-z)
c     
      dimension u  (*)
      dimension f  (*)
      dimension ia (*)
      dimension a  (*)
      dimension ja (*)
      dimension iu (*)
      dimension icg(*)
      dimension iv (*)
c     
c---------------------------------------------------------------------
c     
      go to (100,999,300,999,999,999,999,800,900),itrel
      return
c     
c     Gauss-Seidel relaxation
c     
 100  call relax1(ierr,iprel,ierel,imin,imax,u,f,a,ia,ja,iu,icg)
      return
c     
c     Kaczmarz relaxation (removed)
c     
c     
c     Point Gauss-Seidel relaxation
c     
 300  call relax3(ierr,iprel,u,f,a,ia,ja,iv,ipmn,ipmx,icg)
      return
c     
c     Collective relaxation (removed)
c     
c     
c     Normalization
c     
 800  call norml(ierr,iurel,imin,imax,u,iu)
      return
c     
c     Direct solution (low storage)
c     
 900  call dirslv(ierr,imin,imax,u,f,a,ia,ja)
      return
 999  return
      end
c     
      subroutine relax1(ierr,iprel,ierel,imin,imax,u,f,a,ia,ja,iu,icg)
c     
c---------------------------------------------------------------------
c     
c     Gauss-Seidel relaxation
c     
c     iprel = 1 - relax f-variables only
c     iprel = 2 - relax all variables
c     iprel = 3 - relax c-variables only
c     
c     ierel = n - relax equations of type n
c     ierel = 9 - relax equations of all types
c     
c     iurel = n - relax unknowns of type n
c     iurel = 9 - relax unknowns of all types
c     
c---------------------------------------------------------------------
c     
      implicit real*8 (a-h,o-z)
c     
      dimension u  (*)
      dimension f  (*)
      dimension ia (*)
      dimension a  (*)
      dimension ja (*)
      dimension iu (*)
      dimension icg(*)
c     
c---------------------------------------------------------------------
c     
      ilo=imin
      ihi=imax
      go to (100,200,300) iprel
      ierr = 5
      return
c
c     ierr = 5 indicates user has requested illegal value for iprel 
c     
c     F-variable relaxation
c     
 100  do 120 i=ilo,ihi
         if(icg(i).gt.0) go to 120
         if(ierel.ne.iu(i).and.ierel.ne.9) go to 120
         r=f(i)
         jlo=ia(i)+1
         jhi=ia(i+1)-1
         do 110 j=jlo,jhi
            r=r-a(j)*u(ja(j))
 110     continue
         u(i)=r/a(ia(i))
 120  continue
      return
c     
c     All-variable relaxation
c     
 200  do 220 i=ilo,ihi
         if(ierel.ne.iu(i).and.ierel.ne.9) go to 220
         r=f(i)
         jlo=ia(i)+1
         jhi=ia(i+1)-1
         do 210 j=jlo,jhi
            r=r-a(j)*u(ja(j))
 210     continue
         u(i)=r/a(ia(i))
 220  continue
      return
c     
c     C-variable relaxation
c     
 300  do 320 i=ilo,ihi
         if(icg(i).le.0) go to 320
         if(ierel.ne.iu(i).and.ierel.ne.9) go to 320
         r=f(i)
         jlo=ia(i)+1
         jhi=ia(i+1)-1
         do 310 j=jlo,jhi
            r=r-a(j)*u(ja(j))
 310     continue
         u(i)=r/a(ia(i))
 320  continue
      return
      end
c     
      subroutine relax3(ierr,iprel,u,f,a,ia,ja,iv,ipmn,ipmx,icg)
c     
c---------------------------------------------------------------------
c     
c     Point Gauss-Seidel relaxation
c     
c     iprel = 1 - relax f-variables only
c     iprel = 2 - relax all variables
c     iprel = 3 - relax c-variables only
c     
c     ierel = n - relax equations of type n
c     ierel = 9 - relax equations of all types
c     
c     iurel = n - relax unknowns of type n
c     iurel = 9 - relax unknowns of all types
c     
c---------------------------------------------------------------------
c     
      implicit real*8 (a-h,o-z)
c     
      dimension u  (*)
      dimension f  (*)
      dimension ia (*)
      dimension a  (*)
      dimension ja (*)
      dimension icg(*)
      dimension iv (*)

      dimension d(10,10),s(10)
c     
c---------------------------------------------------------------------
c     
      iplo=ipmn
      iphi=ipmx
      go to (100,200,300,400,500), iprel
      ierr = 6
      return
c     
c       ierr = 6 indicates user has requested illegal value for iprel
c     
c     Relax points with first variable in F
c     
 100  do 180 ipt=iplo,iphi
         ilo=iv(ipt)
         if(icg(ilo).gt.0) go to 180
         ihi=iv(ipt+1)-1
         if(ihi.gt.ilo) go to 120
         r=f(ilo)
         jlo=ia(ilo)+1
         jhi=ia(ilo+1)-1
         do 110 j=jlo,jhi
            r=r-a(j)*u(ja(j))
 110     continue
         u(ilo)=r/a(ia(ilo))
         go to 180
 120     n=0
         ilo1=ilo-1
         nhi=ihi-ilo1
         do 160 i=ilo,ihi
            n=n+1
            do 130 nn=1,nhi
               d(n,nn)=0.e0
 130        continue
            d(n,n)=a(ia(i))
            s(n)=f(i)
            jlo=ia(i)+1
            jhi=ia(i+1)-1
            do 150 j=jlo,jhi
               iii=ja(j)
               if(iii.lt.ilo.or.iii.gt.ihi) go to 140
               nn=iii-ilo1
               d(n,nn)=a(j)
               go to 150
 140           s(n)=s(n)-a(j)*u(iii)
 150        continue
 160     continue
         call gselim(ierr,d,s,nhi)
         if (ierr .ne. 0) return
         n=0
         do 170 i=ilo,ihi
            n=n+1
            u(i)=s(n)
 170     continue
 180  continue
      return
c     
c     Relax all points
c     
 200  do 280 ipt=iplo,iphi
         ilo=iv(ipt)
         ihi=iv(ipt+1)-1
         if(ihi.gt.ilo) go to 220
         r=f(ilo)
         jlo=ia(ilo)+1
         jhi=ia(ilo+1)-1
         do 210 j=jlo,jhi
            r=r-a(j)*u(ja(j))
 210     continue
         u(ilo)=r/a(ia(ilo))
         go to 280
 220     n=0
         ilo1=ilo-1
         nhi=ihi-ilo1
         do 260 i=ilo,ihi
            n=n+1
            do 230 nn=1,nhi
               d(n,nn)=0.e0
 230        continue
            d(n,n)=a(ia(i))
            s(n)=f(i)
            jlo=ia(i)+1
            jhi=ia(i+1)-1
            do 250 j=jlo,jhi
               iii=ja(j)
               if(iii.lt.ilo.or.iii.gt.ihi) go to 240
               nn=iii-ilo1
               d(n,nn)=a(j)
               go to 250
 240           s(n)=s(n)-a(j)*u(iii)
 250        continue
 260     continue
         call gselim(ierr,d,s,nhi)
         if (ierr .ne. 0) return
         n=0
         do 270 i=ilo,ihi
            n=n+1
            u(i)=s(n)
 270     continue
 280  continue
      return
c     
c     Relax points with first variable in C
c     
 300  do 380 ipt=iplo,iphi
         ilo=iv(ipt)
         if(icg(ilo).le.0) go to 380
         ihi=iv(ipt+1)-1
         if(ihi.gt.ilo) go to 320
         r=f(ilo)
         jlo=ia(ilo)+1
         jhi=ia(ilo+1)-1
         do 310 j=jlo,jhi
            r=r-a(j)*u(ja(j))
 310     continue
         u(ilo)=r/a(ia(ilo))
         go to 380
 320     n=0
         ilo1=ilo-1
         nhi=ihi-ilo1
         do 360 i=ilo,ihi
            n=n+1
            do 330 nn=1,nhi
               d(n,nn)=0.e0
 330        continue
            d(n,n)=a(ia(i))
            s(n)=f(i)
            jlo=ia(i)+1
            jhi=ia(i+1)-1
            do 350 j=jlo,jhi
               iii=ja(j)
               if(iii.lt.ilo.or.iii.gt.ihi) go to 340
               nn=iii-ilo1
               d(n,nn)=a(j)
               go to 350
 340           s(n)=s(n)-a(j)*u(iii)
 350        continue
 360     continue
         call gselim(ierr,d,s,nhi)
         if (ierr .ne. 0) return
         n=0
         do 370 i=ilo,ihi
            n=n+1
            u(i)=s(n)
 370     continue
 380  continue
      return
c     
c     Relax points with at least one variable in F
c     
 400  do 480 ipt=iplo,iphi
         ilo=iv(ipt)
         ihi=iv(ipt+1)-1
         if(ihi.gt.ilo) go to 420
         if(icg(ilo).gt.0) go to 480
         r=f(ilo)
         jlo=ia(ilo)+1
         jhi=ia(ilo+1)-1
         do 410 j=jlo,jhi
            r=r-a(j)*u(ja(j))
 410     continue
         u(ilo)=r/a(ia(ilo))
         go to 480
 420     nr=0
         do 422 i=ilo,ihi
            if(icg(i).le.0) nr=nr+1
 422     continue
         if(nr.eq.0) go to 480
         ilo1=ilo-1
         nhi=ihi-ilo1
         n=0
         do 460 i=ilo,ihi
            n=n+1
            do 430 nn=1,nhi
               d(n,nn)=0.e0
 430        continue
            d(n,n)=a(ia(i))
            s(n)=f(i)
            jlo=ia(i)+1
            jhi=ia(i+1)-1
            do 450 j=jlo,jhi
               iii=ja(j)
               if(iii.lt.ilo.or.iii.gt.ihi) go to 440
               nn=iii-ilo1
               d(n,nn)=a(j)
               go to 450
 440           s(n)=s(n)-a(j)*u(iii)
 450        continue
 460     continue
         call gselim(ierr,d,s,n)
         if (ierr .ne. 0) return
         n=0
         do 470 i=ilo,ihi
            n=n+1
            u(i)=s(n)
 470     continue
 480  continue
      return
c     
c     Relax (simultaneously) all C-variables at each point
c     
 500  do 580 ipt=iplo,iphi
         ilo=iv(ipt)
         ihi=iv(ipt+1)-1
         if(ihi.gt.ilo) go to 520
         if(icg(ilo).le.0) go to 580
         r=f(ilo)
         jlo=ia(ilo)+1
         jhi=ia(ilo+1)-1
         do 510 j=jlo,jhi
            r=r-a(j)*u(ja(j))
 510     continue
         u(ilo)=r/a(ia(ilo))
         go to 580
 520     nr=0
         ilo1=ilo-1
         nhi=ihi-ilo1
         nc=0
         do 560 i=ilo,ihi
            nc=nc+1
            if(icg(i).le.0) go to 560
            nr=nr+1
            do 530 nn=1,nhi
               d(nr,nn)=0.e0
 530        continue
            d(nr,nc)=a(ia(i))
            s(nr)=f(i)
            jlo=ia(i)+1
            jhi=ia(i+1)-1
            do 550 j=jlo,jhi
               iii=ja(j)
               if(icg(iii).le.0) go to 540
               if(iii.lt.ilo.or.iii.gt.ihi) go to 540
               nn=iii-ilo1
               d(nr,nn)=a(j)
               go to 550
 540           s(nr)=s(nr)-a(j)*u(iii)
 550        continue
 560     continue
         if(nr.eq.nc) go to 548
         if(nr.eq.0) go to 580
         nc1=0
         nc2=0
         do 547 i=ilo,ihi
            nc1=nc1+1
            if(icg(i).le.0) go to 547
            nc2=nc2+1
            do 546 n=1,nr
               d(n,nc2)=d(n,nc1)
 546        continue
 547     continue
 548     call gselim(ierr,d,s,nr)
         if (ierr .ne. 0) return
         n=0
         do 570 i=ilo,ihi
            if(icg(i).le.0) go to 570
            n=n+1
            u(i)=s(n)
 570     continue
 580  continue
      return
      end
c     
      subroutine gselim(ierr,c,d,npts)
      implicit real*8 (a-h,o-z)
      dimension c(10,10),d(10)
      if(npts.gt.10) then
        ierr = 4
        return
      endif
      if(npts.gt.1) go to 10
      d(1)=d(1)/c(1,1)
      return
c     
c     perform foreward elimination
c     
 10   do 150 n1=1,npts-1
         if(c(n1,n1).eq.0.e0) go to 150
         do 140 n2=n1+1,npts
            if(c(n2,n1).eq.0.e0) go to 140
            g=c(n2,n1)/c(n1,n1)
            do 130 n3=n1+1,npts
               c(n2,n3)=c(n2,n3)-g*c(n1,n3)
 130        continue
            d(n2)=d(n2)-g*d(n1)
 140     continue
 150  continue
c     
c     perform back-substitution
c     
      do 190 n1=npts,2,-1
         d(n1)=d(n1)/c(n1,n1)
         n2hi=n1-1
         do 180 n2=1,n2hi
            if(c(n2,n1).eq.0.e0) go to 180
            d(n2)=d(n2)-d(n1)*c(n2,n1)
 180     continue
 190  continue
      d(1)=d(1)/c(1,1)
      return
      end
c     
      subroutine norml(ierr,iurel,imin,imax,u,iu)
c     
c---------------------------------------------------------------------
c     
c     Normalization (addition of constant)
c     
c     iprel = 1 - relax f-variables only
c     iprel = 2 - relax all variables
c     iprel = 3 - relax c-variables only
c     
c     ierel = n - relax equations of type n
c     ierel = 9 - relax equations of all types
c     
c     iurel = n - relax unknowns of type n
c     iurel = 9 - relax unknowns of all types
c     
c---------------------------------------------------------------------
c     
      implicit real*8 (a-h,o-z)
c     
      dimension u  (*)
      dimension iu (*)
c     
      dimension rs(10),np(10)
c     
c---------------------------------------------------------------------
c     
      ilo=imin
      ihi=imax
      do 10 n=1,10
         rs(n)=0.e0
         np(n)=0
 10   continue
      do 20 i=ilo,ihi
         if(iu(i).ne.iurel.and.iurel.ne.9) go to 20
         rs(iu(i))=rs(iu(i))+u(i)
         np(iu(i))=np(iu(i))+1
 20   continue
      do 30 n=1,10
         if(np(n).eq.0) go to 30
         rs(n)=rs(n)/float(np(n))
 30   continue
      do 40 i=ilo,ihi
         if(iu(i).ne.iurel.and.iurel.ne.9) go to 40
         u(i)=u(i)-rs(iu(i))
 40   continue
      return
      end
