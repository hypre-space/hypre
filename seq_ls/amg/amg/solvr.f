CC### filename: solvr.f
c
c========================================================
c
c     SOLVE: solve for systems
c
c=====================================================================
c
      subroutine solve(levels,ncyc,mu,ntrlx,iprlx,ierlx,iurlx,iprtc,
     *     nun,imin,imax,u,f,a,ia,ja,iu,icg,b,ib,jb,
     *     ipmn,ipmx,iv,ip,xp,yp,lfname)
c
c---------------------------------------------------------------------
c
c     this version uses a predefined restriction operator
c     rather than the transpose of interpolation.
c
c---------------------------------------------------------------------
c
      implicit real*8 (a-h,o-z)
cveh next added
      integer told,tnew,ttot
c
      include 'params.amg'
c
      dimension imin(25),imax(25)
      dimension u  (*)
      dimension f  (*)
      dimension ia (*)
      dimension a  (*)
      dimension ja (*)
      dimension iu (*)
      dimension icg(*)
      dimension iv (*)
c
      dimension ip (*)
      dimension xp (*)
      dimension yp (*)

      dimension ipmn(25),ipmx(25)

      dimension ib (*)
      dimension b  (*)
      dimension jb (*)
c
c=>   solution parameters (u/d/f/c)
c
      dimension mu (25)

      dimension ntrlx(4)
      dimension iprlx(4)
      dimension ierlx(4)
      dimension iurlx(4)
c
      character*(*)  lfname
c
c     storage for convergence data
c
      dimension resv(20)
c
c     work space
c
      dimension iarr(10)

c---------------------------------------------------------------------
c open the log file and write some initial info
c---------------------------------------------------------------------

      open(6,file=lfname,access='append')

      write(6,9000)
 9000  format(/'SOLVE INFO:'/)

c
c---------------------------------------------------------------------
c
c===> find number of unknowns (only used for output simplification)
c
c     ilo=imin(1)
c     ihi=imax(1)
c     nun=0
c     do 10 i=ilo,ihi
c     if(iu(i).gt.nun) nun=iu(i)
c10    continue
c
c===> decode ncyc
c
      call idec(ncyc,3,ndig,iarr)
      ivstar=iarr(1)-1
      ifcycl=iarr(2)
      ncycle=iarr(3)
      if(ncycle.eq.0) then
        close(6)
        return
      endif
c     write(6,7000)
c
c===> find initial residual
c
      if(iprtc.ge.0) then
        call rsdl(1,enrg,res,resv,0,imin,imax,u,f,a,ia,ja,iu)
        resi = res
        write(6,1000)
        write(6,1100) res,enrg
        write(6,*) ' energy = ',enrg
      endif
c
c===> cycling
c
      call ctime(told)
      do 100 ncy=1,ncycle
c
      icycmp=0
c
      call cycle(levels,mu,ifcycl,ivstar,
     *           ntrlx,iprlx,ierlx,iurlx,iprtc,icycmp,
     *           nun,imin,imax,u,f,a,ia,ja,iu,icg,
     *           b,ib,jb,ipmn,ipmx,iv,ip,xp,yp)

      if(iprtc.ge.0) then
        resold=res
        call rsdl(1,enrg,res,resv,0,imin,imax,u,f,a,ia,ja,iu)
        factor=res/resold
        write(6,1200) ncy,res,enrg,factor
      endif

  100 continue

      afactor=(res/resi)**(1.e0/float(ncycle))
      write(6,1300) afactor
      call ctime(tnew)
      ttot=ttot+tnew-told
cveh      tcyc=ttot/ncycle
      tcyc=float(ttot/ncycle)
      write(6,2000) tcyc,ttot
      cmpcy=float(icycmp)/float(ia(imax(1)+1)-1)
      cmpop=float(ia(imax(levels)+1)-1)/float(ia(imax(1)+1)-1)
      cmpgr=float(imax(levels))/float(imax(1))
      write(6,3000) cmpgr,cmpop,cmpcy

      close(6)

      return

1000  format(/'               residual     energy     factor'/,
     *        '               --------     ------     ------')
1100  format(3x,' Initial ',1p,5(2x,e9.2))
1200  format(3x,' Cycle ',i2,1p,5(1x,e9.2))
1300  format(/3x,' Average convergence factor   =  ',1p,e9.2)
cveh        New format statement 2000 replaces following ccccccccccccc
cveh 2000  format(/5x,'Solution times:'/
cveh     *       10x,'per cycle :',f10.5/
cveh     *       10x,'total     :',f10.5)
cveh        New format statement 2000 replaces above ccccccccccccccccc
2000  format(/5x,'Solution times:'/
     *       10x,'per cycle :',F10.5/
     *       10x,'total     :',I10)

3000  format(/5x,'Complexity:  grid     = ',f10.5/
     *        5x,'             operator = ',f10.5/
     *        5x,'             cycle    = ',f10.5)

      end
c
      subroutine cycle(levels,mu,ifcycl,ivstar,
     *                 ntrlx,iprlx,ierlx,iurlx,iprtc,icomp,
     *                 nun,imin,imax,u,f,a,ia,ja,iu,icg,
     *                 b,ib,jb,ipmn,ipmx,iv,ip,xp,yp)
c
c---------------------------------------------------------------------
c
c     cycling routine
c
c     1. ntrf can have several meanings (nr1,nr2)
c        nr1 defines the first fine grid sweep
c        nr2 defines any subsequent sweeps
c
c        ntrf = 0   - (0,0)
c        ntrf = 1   - (ntrd,ntru)
c        ntrf > 9   - standard meaning
c
c     2. mu(k) sets # of cycles to be performed on level k+1
c
c     2. Cycling is controlled using a level counter nc(k)
c
c        Each time relaxation is performed on level k, the
c        counter is decremented by 1. If the counter is then
c        negative, we go to the next finer level. If non-
c        negative, we go to the next coarser level. The
c        following actions control cycling:
c
c        a. nc(1) is initialized to 1.
c        b. nc(k) is initialized to mu(k-1)+ifcycl for k>1.
c
c        c. During cycling, when going down to level k,
c        nc(k) is set to max0(nc(k),mu(k-1))
c
c---------------------------------------------------------------------
c
      implicit real*8 (a-h,o-z)
c
      include 'params.amg'
c
      dimension imin(25),imax(25)
      dimension u  (*)
      dimension f  (*)
      dimension ia (*)
      dimension a  (*)
      dimension ja (*)
      dimension iu (*)
      dimension icg(*)
c
      dimension ip (*)
      dimension xp (*)
      dimension yp (*)

      dimension ipmn(25),ipmx(25)
      dimension iv (*)

      dimension ib (*)
      dimension b  (*)
      dimension jb (*)
c
c=>   solution parameters (u/d/f/c)
c
      dimension mu (25)

      dimension ntrlx(4)
      dimension iprlx(4)
      dimension ierlx(4)
      dimension iurlx(4)
c
c     storage for convergence/output data
c
      dimension resv(20),enrgf(20)
      dimension ll(45),nc(25),ity(10),ipt(10),ieq(10),iun(10)
c
      dimension iarr(10)
c
c---------------------------------------------------------------------
c
c===> set cycling parameters
c
c     initialize level counter for all levels
c
      m=levels
      nc(1)=1
      do 10 k=2,m
      nc(k)=mu(k-1)+ifcycl
   10 continue
c
c     set relaxation parameters
c
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
c
c     set finer level energy correction to zero
c
      enrgf(1)=0.e0
c
c     set level
c
      k=1
c
c     initialize output quantities
c
c     nun1=min0(nun,3)
      nun1=min0(nun,4)
      lltop=0
      if(iprtc.gt.0) write(6,3999)
c
c     set initial cycling parameters
c
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
c
c     decode cycling parameters
c
100   if(ntrx.le.9) go to 140
c
c     decode ntrx (number & type of relaxation sweeps)
c
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
110   continue
c
c===> decode and test additional relaxation parameters
c
      call idec(iuns,9,ndig,iun)
      if(ndig.lt.ii) stop 'iuns'

      call idec(ieqs,9,ndig,ieq)
      if(ndig.lt.ii) stop 'ieqs'

      call idec(ipts,9,ndig,ipt)
      if(ndig.lt.ii) stop 'ipts'
c
c     compute & print residuals
c
      if(iprtc.ge.k) then

        if(lltop.ne.0) then
          write(6,5000) (ll(kk),kk=1,lltop)
          lltop=0
        endif

        call rsdl(k,enrg,res,resv,0,imin,imax,u,f,a,ia,ja,iu)
        enrgt=enrg+enrgf(k)
c       write(6,6001) k,res,enrgt,(resv(i),i=1,nun1)
        write(6,6001) k,res,enrgt,(resv(i),i=1,nun)
c       if(nun.gt.nun1) write(6,6002) (resv(i),i=4,nun)

      endif
c
c===> relaxation
c
      do 130 n=1,nrelax

      icomp=icomp+ia(imax(k)+1)-ia(imin(k))
c
c     perform partial sweeps
c
      do 120 i=1,ii
      call relax(k,ity(i),ipt(i),ieq(i),iun(i),
     *           imin,imax,u,f,a,ia,ja,iu,icg,ipmn,ipmx,iv)
c
c     compute & print residuals
c
      if(iprtc.ge.k) then
        call rsdl(k,enrg,res,resv,0,imin,imax,u,f,a,ia,ja,iu)
        enrgt=enrg+enrgf(k)
c       write(6,6000) k,ity(i),ipt(i),ieq(i),iun(i),res,enrgt,
c    *                (resv(iii),iii=1,nun1)
        write(6,6000) k,ity(i),ipt(i),ieq(i),iun(i),res,enrgt,
     *                (resv(iii),iii=1,nun)
c       if(nun.gt.nun1) write(6,6002) (resv(iii),iii=5,nun)
      endif
c     call rplot(k,imin,imax,u,f,a,ia,ja,
c    *           iu,ip,ipmn,ipmx,iv,xp,yp)

120   continue

130   continue
c
      if(iprtc.gt.0.and.iprtc.lt.k) then
        lltop=lltop+1
        ll(lltop)=k
        if(lltop.ge.25) then
          write(6,5000) (ll(kk),kk=1,25)
          lltop=0
        endif
      endif

140   nc(k)=nc(k)-1
      if(nc(k).ge.0.and.k.ne.m) go to 300
      if(k.eq.1) go to 400
c
c===> go to next finer grid
c
200   k=k-1
      call intad(k+1,k,ivstar,nun,imin,imax,
     *                 u,f,a,ia,ja,iu,icg,b,ib,jb)
c     call rplot(k,imin,imax,u,f,a,ia,ja,
c    *           iu,ip,ipmn,ipmx,iv,xp,yp)
c
c     set cycling parameters
c
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
c
c===> go to next coarser grid
c
300   k=k+1
      enrgf(k)=enrgt
      call putz(k,imin,imax,u)
c     call rscalr(k-1,k,imin,imax,u,f,a,ia,ja,icg,b,ib,jb)
      call rscali(k-1,k,imin,imax,u,f,a,ia,ja,icg,b,ib,jb)
c
c     reset level counters for coarser level
c
      nc(k)=max0(nc(k),mu(k-1))
c
c     set cycling parameters
c
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
c
400   continue

c3999  format(/'    k   tpeu   residual     energy   res 1,2,...')
3999  format(/'    k  tpeu  residual    energy  res 1,2,...')
5000  format(26(1x,i2))
6000  format(3x,i2,2x,4i1,1p,6(1x,e9.2):/31x,4(1x,e9.2))
6001  format(3x,i2,6x,1p,6(1x,e9.2):/31x,4(1x,e9.2))
c6000  format(3x,i2,3x,4i1,1p,5(2x,e9.2))
c6001  format(3x,i2,7x,1p,5(2x,e9.2))
c6002  format(34x,1p,3(2x,e9.2))

      return
      end
c
