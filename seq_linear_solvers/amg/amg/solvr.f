
c=====================================================================
c     main solver routine
c=====================================================================

      subroutine solve(stoptol,levels,ncyc,mu,ntrlx,iprlx,
     *     ierlx,iurlx,ioutdat,
     *     nun,imin,imax,u,f,vtmp,a,ia,ja,iu,icg,b,ib,jb,
     *     ipmn,ipmx,iv,ip,xp,yp,
     *     ndimu,ndimp,ndima,ndimb,
     *     leva, levb, levv, levp, levi,
     *     numa, numb, numv, nump,
     *     lfname)

      implicit real*8 (a-h,o-z)

      integer told,tnew,ttot



      dimension imin(*),imax(*)
      dimension u   (*)
      dimension f   (*)
      dimension vtmp(*)
      dimension a   (*)
      dimension ia  (*)
      dimension ja  (*)
      dimension iu  (*)
      dimension icg (*)
      dimension iv  (*)

      dimension ip (*)
      dimension xp (*)
      dimension yp (*)

      dimension ipmn(*),ipmx(*)

      dimension ib (*)
      dimension b  (*)
      dimension jb (*)

c     solution parameters (u/d/f/c)

      dimension mu (25)

      dimension ntrlx(4)
      dimension iprlx(4)
      dimension ierlx(4)
      dimension iurlx(4)

c     level arrays

      dimension leva(*)
      dimension levb(*)
      dimension levv(*)
      dimension levp(*)
      dimension levi(*)
      dimension numa(*)
      dimension numb(*)
      dimension numv(*)
      dimension nump(*)

      character*(*)  lfname

c     storage for convergence data

      dimension resv(20)

c     work space

      dimension iarr(10)

c---------------------------------------------------------------------
c     open the log file and write some initial info
c---------------------------------------------------------------------

      if (ioutdat .ne. 0) then
         open(6,file=lfname,access='append')
         write(6,9000)
 9000 format(//'AMG SOLUTION INFO:'/)
      endif
c
c---------------------------------------------------------------------

c===  > decode ncyc

      call idec(ncyc,3,ndig,iarr)
      ivstar=iarr(1)-1
      ifcycl=iarr(2)
      ncycle=iarr(3)
      if(ncycle.eq.0) then
         close(6)
         return
      endif

c===  > find initial residual

      call rsdl(1,enrg,res,resv,vtmp,
     *        imin,imax,u,f,a,ia,ja,iu)
         resi = res
      if((ioutdat .eq.1) .or. (ioutdat .eq. 3)) then
         write(6,1000)
         write(6,1100) res,enrg
      endif

c===  > veh: compute 2-norm of rhs for relative residual

      nv = numv(1)
      fnorm = 0.0
      do 11, i=1,nv
         fnorm = fnorm + f(i)*f(i)
 11   continue
      fnorm = sqrt(fnorm)
      relres = res/fnorm

c===  > cycling

      call ctime(told)
      do 100 ncy=1,ncycle

         icycmp=0

         call cycle(levels,mu,ifcycl,ivstar,
     *        ntrlx,iprlx,ierlx,iurlx,icycmp,
     *        nun,imin,imax,u,f,vtmp,a,ia,ja,iu,icg,
     *        b,ib,jb,ipmn,ipmx,iv,ip,xp,yp,
     *        ndimu,ndimp,ndima,ndimb,
     *        leva, levb, levv, levp, levi,
     *        numa, numb, numv, nump)

            engold = enrg
            resold=res
            call rsdl(1,enrg,res,resv,vtmp,
     *           imin,imax,u,f,a,ia,ja,iu)
            factor=res/resold
            relres = res/fnorm
            delteng=enrg-engold
            if((ioutdat .eq.1) .or. (ioutdat .eq. 3)) then
               write(6,1200) ncy,res,enrg,factor,relres
            endif
            if (relres .lt. stoptol) go to 101
         

 100  continue

 101  if (ncy .gt. ncycle) ncy = ncycle

      afactor=(res/resi)**(1.e0/float(ncy))
      if (ioutdat .ne. 0) then
         write(6,1300) afactor
         call ctime(tnew)
         ttot=ttot+tnew-told
         tcyc=float(ttot)/float(ncy)
         write(6,2000) tcyc,ttot
         ntotv=0
         ntota=0
         do 500 k=1,levels
            ntotv=ntotv+numv(k)
            ntota=ntota+numa(k)
 500     continue
         cmpgr=float(ntotv)/numv(1)
         cmpop=float(ntota)/numa(1)
         cmpcy=float(icycmp)/numa(1)
         write(6,3000) cmpgr,cmpop,cmpcy
      endif
      close(6)

      return

 1000 format(/14x,'                              relative'/,
     *     14x,'residual    energy    factor  residual'
cveh     *     '   <Au,u>   delteng     <f,u>    <r,u>    <u,u>',
     *     /,14x,'--------    ------    ------  --------')
 1100 format(3x,' Initial ',1p,5(1x,e9.2))
 1200 format(3x,' Cycle ',i2,1p,9(1x,e9.2))
 1300 format(/3x,' Average convergence factor   =  ',1p,e9.2)
 2000 format(/5x,'Solution times:'/
     *     10x,'per cycle :',F10.5/
     *     10x,'total     :',I10)

 3000 format(/5x,'Complexity:  grid     = ',f10.5/
     *     5x,'             operator = ',f10.5/
     *     5x,'             cycle    = ',f10.5)

      end

c=====================================================================
c     cycling routine:
c     
c     1. ntrf can have several meanings (nr1,nr2)
c     nr1 defines the first fine grid sweep
c     nr2 defines any subsequent sweeps
c     
c     ntrf = 0   - (0,0)
c     ntrf = 1   - (ntrd,ntru)
c     ntrf > 9   - standard meaning
c     
c     2. mu(k) sets # of cycles to be performed on level k+1
c     
c     2. Cycling is controlled using a level counter nc(k)
c     
c     Each time relaxation is performed on level k, the
c     counter is decremented by 1. If the counter is then
c     negative, we go to the next finer level. If non-
c     negative, we go to the next coarser level. The
c     following actions control cycling:
c     
c     a. nc(1) is initialized to 1.
c     b. nc(k) is initialized to mu(k-1)+ifcycl for k>1.
c     
c     c. During cycling, when going down to level k,
c     nc(k) is set to max0(nc(k),mu(k-1))
c=====================================================================

      subroutine cycle(levels,mu,ifcycl,ivstar,
     *     ntrlx,iprlx,ierlx,iurlx,icomp,
     *     nun,imin,imax,u,f,vtmp,a,ia,ja,iu,icg,
     *     b,ib,jb,ipmn,ipmx,iv,ip,xp,yp,
     *     ndimu,ndimp,ndima,ndimb,
     *     leva, levb, levv, levp, levi,
     *     numa, numb, numv, nump)

      implicit real*8 (a-h,o-z)

      dimension imin(*),imax(*)
      
      dimension vtmp(*)
      dimension u   (*)
      dimension f   (*)
      dimension ia  (*)
      dimension a   (*)
      dimension ja  (*)
      dimension iu  (*)
      dimension icg (*)

      dimension ip (*)
      dimension xp (*)
      dimension yp (*)

      dimension ipmn(*),ipmx(*)
      dimension iv (*)

      dimension ib (*)
      dimension b  (*)
      dimension jb (*)

c     level arrays

      dimension leva(*)
      dimension levb(*)
      dimension levv(*)
      dimension levp(*)
      dimension levi(*)
      dimension numa(*)
      dimension numb(*)
      dimension numv(*)
      dimension nump(*)

c     =>   solution parameters (u/d/f/c)

      dimension mu (25)

      dimension ntrlx(4)
      dimension iprlx(4)
      dimension ierlx(4)
      dimension iurlx(4)

c     storage for convergence/output data

      dimension enrgf(20)
      dimension nc(25),ity(10),ipt(10),ieq(10),iun(10)

      dimension iarr(10)

c---------------------------------------------------------------------

c     initialize level counter for all levels

      m=levels
      nc(1)=1
      do 10 k=2,m
         nc(k)=mu(k-1)+ifcycl
 10   continue

c     set relaxation parameters

      ntrf=ntrlx(1)
      ntrd=ntrlx(2)
      ntru=ntrlx(3)
      ntrc=ntrlx(4)

      iprf=iprlx(1)
      iprd=iprlx(2)
      ipru=iprlx(3)
      iprc=iprlx(4)

      ierf=ierlx(1)
      ierd=ierlx(2)
      ieru=ierlx(3)
      ierc=ierlx(4)

      iurf=iurlx(1)
      iurd=iurlx(2)
      iuru=iurlx(3)
      iurc=iurlx(4)

c     set finer level energy correction to zero

      enrgf(1)=0.e0

c     initialize output quantities

      nun1=min0(nun,4)
      lltop=0


c     set initial cycling parameters

      k=1
      ntrx=ntrf
      iuns=iurf
      ieqs=ierf
      ipts=iprf
      if(ntrf.eq.1.or.ntrf.eq.2) then
         ntrx=ntrd
         iuns=iurd
         ieqs=ierd
         ipts=iprd
      endif

c     decode cycling parameters

 100  if(ntrx.le.9) go to 140

c     decode ntrx (number & type of relaxation sweeps)

      call idec(ntrx,9,ndig,iarr)
      nrelax=iarr(1)
      ii=0
      do 110 i=2,ndig
         if(iarr(i).eq.0) then
            nrelax=nrelax*10
         else
            ii=ii+1
            ity(ii)=iarr(i)
         endif
 110  continue

c===  > decode and test additional relaxation parameters

      call idec(iuns,9,ndig,iun)
      if(ndig.lt.ii) stop 'iuns'

      call idec(ieqs,9,ndig,ieq)
      if(ndig.lt.ii) stop 'ieqs'

      call idec(ipts,9,ndig,ipt)
      if(ndig.lt.ii) stop 'ipts'

c===  > relaxation

      do 130 n=1,nrelax

         icomp=icomp+numa(k)

c     perform partial sweeps

         do 120 i=1,ii
            call relax(1,ity(i),ipt(i),ieq(i),iun(i),
     *           imin(k),imax(k),
     *           u(levv(k)),f(levv(k)),
     *           a(leva(k)),ia(levi(k)),ja(leva(k)),iu(levv(k)),
     *           icg(levv(k)),
     *           ipmn(k),ipmx(k),
     *           iv(levp(k)))
 120     continue

 130  continue

 140  nc(k)=nc(k)-1
      if(nc(k).ge.0.and.k.ne.m) go to 300
      if(k.eq.1) go to 400

c===  > go to next finer grid

 200  kc=k
      kf=k-1
      call intad(
     *     u(levv(kf)), icg(levv(kf)),
     *     b(levb(kf)), ib(levi(kf)), jb(levb(kf)),
     *     numv(kf), vtmp,
     *     u(levv(kc)), f(levv(kc)),
     *     a(leva(kc)), ia(levi(kc)), ja(leva(kc)), iu(levv(kc)),
     *     numv(kc),
     *     ivstar, nun)
      k=k-1

c     set cycling parameters

      ntrx=ntru
      iuns=iuru
      ieqs=ieru
      ipts=ipru
      if(k.eq.1.and.ntrf.gt.9) then
         ntrx=ntrf
         iuns=iurf
         ieqs=ierf
         ipts=iprf
      endif
      go to 100

c===  > go to next coarser grid

 300  kf=k
      kc=k+1
      enrgf(kc)=enrgt
      call putz(1,imin(kc),imax(kc),u(levv(kc)))
      call rscali(
     *     f(levv(kc)),
     *     numv(kc),
     *     u(levv(kf)), f(levv(kf)), vtmp,
     *     a(leva(kf)), ia(levi(kf)), ja(leva(kf)),
     *     icg(levv(kf)),
     *     b(levb(kf)), ib(levi(kf)), jb(levb(kf)),
     *     numv(kf))
      k=k+1

c     reset level counters for coarser level

      nc(k)=max0(nc(k),mu(k-1))

c     set cycling parameters

      ntrx=ntrd
      iuns=iurd
      ieqs=ierd
      ipts=iprd
      if(k.eq.m) then
         ntrx=ntrc
         iuns=iurc
         ieqs=ierc
         ipts=iprc
      endif
      go to 100

 400  continue

      return
      end
