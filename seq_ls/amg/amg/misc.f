CC### filename: MS.FOR
c
c==== FILE MS.FOR ====================================================
c
c     MISCELLANEOUS ROUTINES FOR AMGS01
c
c=====================================================================
c
c=====================================================================
c
c     routines for function definition
c
c=====================================================================
c
      subroutine putf(k,irhs,imin,imax,f,iu,ip,xp,yp)
c
c---------------------------------------------------------------------
c
      implicit real*8 (a-h,o-z)
c
      dimension imin(25),imax(25)
      dimension f  (*)
      dimension iu (*)
      dimension ip (*)
      real*8    xp (*)
      real*8    yp (*)
c
c---------------------------------------------------------------------
c
      ilo=imin(k)
      ihi=imax(k)
      if(irhs.lt.0) return
      if(irhs.eq.1) go to 20
      do 10 i=ilo,ihi
   10 f(i)=0.e0
      return
   20 do 21 i=ilo,ihi
      f(i)=1.e0
   21 continue
      return
      end
c
      subroutine putu(k,rndu,imin,imax,u,iu,ip,xp,yp)
c
c---------------------------------------------------------------------
c
c     sets level k function u to a grid function:
c
c     - 0.0 .lt. rndu .lt. 1.0: random function with random values
c                               influenced by the value of rndu
c     - rndu=0.0:               zero
c     - rndu=1.0:               one
c
c---------------------------------------------------------------------
c
      implicit real*8 (a-h,o-z)
c
      dimension imin(25),imax(25)
      dimension u  (*)
      dimension iu (*)

      dimension ip (*)
      real*8 xp (*)
      real*8 yp (*)
c
c---------------------------------------------------------------------
c
      imn=imin(k)
      imx=imax(k)
      if (rndu.lt.0.9999999.and.rndu.gt.0.0) goto 20
      if (rndu.ne.0.0) goto 50
      do 10 i=imn,imx
        u(i)=0.e0
10    continue
      return
c
20    s=rndu
      do 30 i=imn,imx
        u(i)=random(s)
30    continue
      return
c
50    if(rndu.gt.1.) go to 200
      do 100 i=imn,imx
        u(i)=1.0e0
100   continue
      return
200   if(rndu.gt.2.) go to 300
      do 210 i=imn,imx
      x=xp(ip(i))
      y=yp(ip(i))
      if(iu(i).eq.1) u(i)=x*(2.*y-1)
      if(iu(i).eq.2) u(i)=-x*x
210   continue
      return
300   if(rndu.gt.3.) go to 400
      do 310 i=imn,imx
      x=xp(ip(i))
      y=yp(ip(i))
      if(iu(i).eq.1) u(i)=x*(2.*y-1)
      if(iu(i).eq.2) u(i)=-1.5*x
310   continue
      return
400   if(rndu.gt.4.) return
      pi=acos(-1.0)
      do 410 i=imn,imx
      x=xp(ip(i))
      y=yp(ip(i))
      if(iu(i).eq.1) u(i)=sin(pi*x)*sin(pi*y)
      if(iu(i).eq.2) u(i)=cos(2.0*pi*x)*sin(pi*y)
      if(iu(i).eq.3) u(i)=sin(pi*x)*cos(3.0*pi*y)
      if(iu(i).eq.4) u(i)=cos(pi*x)*cos(pi*y)
      if(iu(i).eq.5) u(i)=sin(2.0*pi*x)*sin(3.0*pi*y)
      if(iu(i).eq.6) u(i)=cos(pi*x)*sin(1.0+pi*y)
      if(iu(i).eq.7) u(i)=cos(4.0*pi*x)*sin(2.5*pi*y)
410   continue
      return
      end
c
      subroutine putz(k,imin,imax,u)
c
c---------------------------------------------------------------------
c
      implicit real*8 (a-h,o-z)
c
      dimension imin(25),imax(25)
      dimension u (*)
c
c---------------------------------------------------------------------
c
      ilo=imin(k)
      ihi=imax(k)
      do 10 i=ilo,ihi
   10 u(i)=0.e0
      return
      end
c
      real*8 function random(s)
      implicit real*8 (a-h,o-z)
      random=100.0e0* exp(s)
      random=random-float(int(random))
      s=random
      return
      end
c
c=====================================================================
c
c     real/integer parameter decomposition routines
c
c=====================================================================
c
      subroutine idec(int0,nnum,ndigit,iarr)
      implicit real*4 (a-h,o-z)
c
c     decompose non-negative integer int0 into nnum integers
c
c     input:  int0   - integer (0.le. int0 .le.999999999)
c             nnum   - integer (1.le. nnum .le.9); number of integers
c                      to be returned on array iarr (see below)
c
c     output: ndigit - integer; number of digits of int0
c             iarr   - integer-array with the following contents:
c                      iarr(1)        = first      digit of int0,
c                      iarr(2)        = second     digit of int0, ....
c                      iarr(nnum-1)   = (nnum-1)st digit of int0,
c                      iarr(nnum)     = rest of int0
c                      if nnum > ndigit, the corresponding components
c                      of iarr are put to zero.
c
      dimension iarr(10)
      data eps /1.00000000001/
c
      if (int0.ge.10) goto 10
      ndigit=1
      iarr(1)=int0
1     do 5 i=ndigit+1,nnum
        iarr(i)=0
5     continue
      return
c
   10 ndigit=1+int(alog10(eps*float(int0)))
      nrest=int0
      do 20 i=ndigit,1,-1
         iarr(i)=nrest-nrest/10*10
         nrest=nrest/10
20    continue
      if (ndigit.le.nnum) goto 1
c
      nrest=iarr(ndigit)
      ie=0
      do 30 i=ndigit-1,nnum,-1
         ie=ie+1
         nrest=nrest+iarr(i)*10**ie
30    continue
      iarr(nnum)=nrest
      return
      end
c
c.....................................................................
c
c     rdec                                               subroutine
c
c.....................................................................
c
      subroutine rdec(r0,r1,r2)
      implicit real*8 (a-h,o-z)
c
c     decompose non-negative real r0 into two reals r1,r2
c
c     input:  r0 - real number of the form i.j, i and j integers.
c                  the number of digits of i is not allowed to exceed
c                  the total sum of digits is not allowed to exceed 15
c
c     output: r1 - real number: r1=0.i
c             r2 - real number: r2=0.j
c
      if (r0.ge.1.0) goto 10
      r1=0.e0
      r2=r0
      return
c
   10 r1=float(int(r0))
      r2=r0-r1
      do 20 i=1,15
         r1=r1*0.1
         if (r1.lt.1.0) return
20    continue
      stop
      end
c
c=====================================================================
c
c     output routines
c
c=====================================================================
c
      subroutine printf(prtfile)
c
c---------------------------------------------------------------------
c
c     echo a file to output
c
c---------------------------------------------------------------------
c
      character*(*) prtfile
      character*70  line
c
c---------------------------------------------------------------------
c
      write(6,'(1x)')
      write(6,'(''     Filename: '',a)') prtfile
      write(6,'(1x)')
      open(9,file=prtfile,status='old')
1     read(9,1000,end=99) line
      write(6,2000) line
      go to 1
99    close(9)
      write(6,'(1x)')
      return
1000  format(a70)
2000  format(5x,a70)
      end
c
      subroutine outf(k,iout,imin,imax,f)
c
c---------------------------------------------------------------------
c
      implicit real*8 (a-h,o-z)
c
      dimension imin(25),imax(25)
      dimension f  (*)
c
c---------------------------------------------------------------------
c
      if(iout.lt.6) return
      write(6,1000) k
      ilo=imin(k)
      ihi=imax(k)
      write(6,2000) (f(i),i=ilo,ihi)
      write(6,3000)
      return
1000  format(/' f for k=',i2/)
2000  format(13(1x,d9.3))
3000  format(1x)
      end
c
c---------------------------------------------------------------------
c
      subroutine outa(k,iout,imin,imax,a,ia,ja,iu,ip,icg,ifg,
     *                  b,ib,jb,ipmn,ipmx,iv,xp,yp)
c
c---------------------------------------------------------------------
c
c     print the level k information - 80 col compressed format
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
      dimension ippt(10)
      dimension iarr(20)
      dimension aa(10),jja(10),iiu(10)
c
      character*1 icpt,ifpt,ippt,ispt
      data icpt,ifpt,ispt /'C','F','S'/
c
c---------------------------------------------------------------------
c
      icol = 80
      nr = 6
c
      if(iout.eq.0) return
      ia1=ia(imin(k))
      ia2=ia(imax(k)+1)-1
      ip1=ipmn(k)
      ip2=ipmx(k)
      iv1=iv(ip1)
      iv2=iv(ip2+1)-1
      np=imax(k)-imin(k)+1
      write(6,999) k,np,imin(k),imax(k),ia1,ia2,ip1,ip2,iv1,iv2
      if(iout.le.1) return
      ilo=imin(k)
      ihi=imax(k)
      do 90 i=ilo,ihi
      ipt=ip(i)
      if(iout.gt.6) go to 30
      write(6,1000) i,ia(i),icg(i),ifg(i),iu(i),ipt,iv(ipt)
      if(iout.le.2) go to 80
      nn=(ia(i+1)-ia(i)-1)/nr+1
      jlo=ia(i)-nr
c
      do 20 nrow=1,nn
      jlo=jlo+nr
      jhi=min0(ia(i+1)-1,jlo+nr-1)
c
c     load next row of output arrays
c
      nmx=0
      do 10 j=jlo,jhi
      nmx=nmx+1
      if(icg(ja(j)).eq.0) ippt(nmx)=ispt
      if(icg(ja(j)).lt.0) ippt(nmx)=ifpt
      if(icg(ja(j)).gt.0) ippt(nmx)=icpt
      iiu(nmx) = iu(ja(j))
      jja(nmx) = ja(j)
      aa(nmx)  = a(j)
10    continue
c
      write(6,2010) (jja(n),ippt(n),iiu(n),n=1,nmx)
      write(6,2020) (aa(n),n=1,nmx)
20    continue
c
c     print b
c
30    if(iout.le.4) go to 80
      ibb=ib(i+1)-1
      write(6,3000) i,ib(i),ibb
      if(iout.le.5) go to 80
      nn=(ib(i+1)-ib(i)-1)/nr+1
      jlo=ib(i)-nr
      do 70 nrow=1,nn
      jlo=jlo+nr
      jhi=min0(ib(i+1)-1,jlo+nr-1)
c
c     load next row of output arrays
c
      nmx=0
      do 60 j=jlo,jhi
      nmx=nmx+1
      if(icg(jb(j)).eq.0) ippt(nmx)=ispt
      if(icg(jb(j)).lt.0) ippt(nmx)=ifpt
      if(icg(jb(j)).gt.0) ippt(nmx)=icpt
      iiu(nmx) = iu(jb(j))
      jja(nmx) = jb(j)
      aa(nmx)  = b(j)
60    continue
c
      write(6,3010) (jja(n),ippt(n),iiu(n),n=1,nmx)
      write(6,3020) (aa(n),n=1,nmx)
70    continue
80    write(6,4000)
90    continue
      return
999   format(/2x,'k=',i2,'  np=',i5,'  imin=',i5,'  imax=',i5,
     *         '  ia(imin(k))=',i5,'  ia(imax(k)+1)-1=',i5/
     *         '  ipmn=',i5,'  ipmx=',i5,'  iv(ipmn)=',i5,
     *         '  iv(ipmx+1)-1=',i5/)
1000  format(' i=',i4,'  ia(i)=',i5,'  icg=',i4,
     *       '  ifg=',i4,'  iu=',i2,'  ip=',i5,'  iv(ip)=',i5)
2010  format(/1x,'a ',6(3x,i5,2x,a1,i1))
2020  format(3x,1p,6(1x,d11.4))

3000  format(/' i=',i4,'  ib(i)=',i5,'  ib(i+1)-1=',i5)
3010  format(/1x,'b ',6(3x,i5,2x,a1,i1))
3020  format(3x,1p,6(1x,d11.4))

4000  format(1x)
      end
c
c---------------------------------------------------------------------
c
c     OUTM - New Version (same 80 col compressed format as outa)
c
c---------------------------------------------------------------------
c
      subroutine outm(iout,imin,imax,a,ia,ja,iu,ip,
     *                     ipmn,ipmx,iv,xp,yp,ipb)
c
c---------------------------------------------------------------------
c
c     print the level k information - 80 col compressed format
c
c---------------------------------------------------------------------
c
      implicit real*8 (a-h,o-z)
c
      dimension ia (*)
      dimension a  (*)
      dimension ja (*)
      dimension iu (*)
      dimension ip (*)
      dimension iv (*)
      dimension ipb(*)

      real*8 xp (*)
      real*8 yp (*)
c
      dimension aa(10),jja(10),iiu(10),iip(10)
c
c---------------------------------------------------------------------
c
      if(iout.le.0) return
      icol = 80
      nr = (icol-2)/12
c
      ia1=ia(imin)
      ia2=ia(imax+1)-1
      iv1=iv(ipmn)
      iv2=iv(ipmx+1)-1
      nv=imax-imin+1
      np=ipmx-ipmn+1
      write(6,999) nv,imin,imax,ia1,ia2,np,ipmn,ipmx,iv1,iv2
c
c     loop over points
c
      do 50 ipt=ipmn,ipmx
      ivlo=iv(ipt)
      ivhi=iv(ipt+1)-1
      if(iout.ge.3)
     *  write(6,7000) ipt,ipb(ipt),xp(ipt),yp(ipt),ivlo,ivhi
      do 30 i=ivlo,ivhi
      iipt=ip(i)
      write(6,1000) i,ia(i),iu(i),iipt,iv(iipt),xp(iipt),yp(iipt)
      nn=(ia(i+1)-ia(i)-1)/nr+1
      jlo=ia(i)-nr
c
      do 20 nrow=1,nn
      jlo=jlo+nr
      jhi=min0(ia(i+1)-1,jlo+nr-1)
c
c     load next row of output arrays
c
      nmx=0
      do 10 j=jlo,jhi
      nmx=nmx+1
      iiu(nmx) = iu(ja(j))
      iip(nmx) = ip(ja(j))
      jja(nmx) = ja(j)
      aa(nmx)  = a(j)
10    continue
c
      write(6,2010) (jja(n),iip(n),iiu(n),n=1,nmx)
      write(6,2020) (aa(n),n=1,nmx)
20    continue
30    continue
      write(6,4000)
50    continue
      return
999   format(/'  nv=',i5,'  imin=',i5,'  imax=',i5,
     *  '  ia(imin)=',i5,'  ia(imax+1)-1=',i5/
     *  '  np=',i5,'  ipmn=',i5,'  ipmx=',i5,'  iv(ipmn)=',i5,
     *  '  iv(ipmx+1)-1=',i5/)
1000  format(' i =',i4,'  ia(i)=',i5,
     *  '  iu=',i2,'  ip=',i5,'  iv(ip)=',i5,1p,
     *  '  x=',d9.2,'  y=',d9.2)
2010  format(/1x,'a ',6(2i5,1x,i1))
2020  format(3x,1p,6(1x,d11.4))

4000  format(1x)
7000  format(/' ip=',i4,'  ipb=',i1,'  x,y=',1p,2(1x,d9.2),
     *        '  iv(ip),iv(ip+1)-1=',2i5/)
      end
c
c---------------------------------------------------------------------
c
      subroutine outs(nun,imin,imax,a,ia,ja,iu,ip,
     *                    ipmn,ipmx,iv,xp,yp,ipb)
c
c---------------------------------------------------------------------
c
c     print the problem information as produced by seta
c     in this version, stencils for unknowns are separated.
c
c---------------------------------------------------------------------
c
      implicit real*8 (a-h,o-z)
c
      dimension ia (*)
      dimension a  (*)
      dimension ja (*)
      dimension iu (*)
      dimension ip (*)
      dimension iv (*)
cveh      integer*1 ipb(*)
      integer ipb(*)

      real*8 xp (*)
      real*8 yp (*)
c
      dimension aa(40),jaa(40)
c
c---------------------------------------------------------------------
c
      icol = 80
      nr = (icol-5)/19
c
      ia1=ia(imin)
      ia2=ia(imax+1)-1
      iv1=iv(ipmn)
      iv2=iv(ipmx+1)-1
      nv=imax-imin+1
      np=ipmx-ipmn+1
      write(6,999) nv,imin,imax,ia1,ia2,np,ipmn,ipmx,iv1,iv2
c
c     loop over points
c
      do 70 iip=ipmn,ipmx
      ivlo=iv(iip)
      ivhi=iv(iip+1)-1
      write(6,7000) iip,ipb(iip),xp(iip),yp(iip),ivlo,ivhi

      do 60 i=ivlo,ivhi
      ipt=ip(i)
      write(6,1000) i,ia(i),iu(i),ipt,iv(ipt),xp(ipt),yp(ipt)
c
c     loop over unknowns
c
      do 40 iun=1,nun
      write(6,2000) iu(i),iun
c
c     load unknowns into temporary row
c
      nrow=0
      do 5 j=ia(i),ia(i+1)-1
      if(iu(ja(j)).eq.iun) then
        nrow=nrow+1
        if(nrow.gt.40) stop 'outs problem'
        aa(nrow)=a(j)
        jaa(nrow)=ja(j)
      endif
5     continue

      nn=(nrow-1)/nr+1
      jlo=1-nr
      do 20 n=1,nn
      jlo=jlo+nr
      jhi=min0(nrow,jlo+nr-1)
      nnn=0
      do 10 j=jlo,jhi
      nnn=nnn+1
10    continue
      write(6,2011) n,(jaa(j),aa(j),j=jlo,jhi)
20    continue
40    continue
      write(6,4000)
60    continue
70    continue
      return
999   format(/'  nv=',i5,'  imin=',i5,'  imax=',i5,
     *  '  ia(imin)=',i5,'  ia(imax+1)-1=',i5/
     *  '  np=',i5,'  ipmn=',i5,'  ipmx=',i5,'  iv(ipmn)=',i5,
     *  '  iv(ipmx+1)-1=',i5/)
1000  format(' i=',i4,'  ia(i)=',i5,
     *  '  iu=',i2,'  ip=',i5,'  iv(ip)=',i5,
     *  '  x=',d10.3,'  y=',e10.3)
2000  format(/2x,i1,'-',i1,' stencil')
2011  format('  a',1p,i1,1x,6(1x,i5,3x,d9.2))
3000  format(' i=',i4,'  ib(i)=',i5,'  ib(i+1)-1=',i5)
4000  format(1x)
7000  format(/' ip=',i4,'  ipb=',i1,'  x,y=',1p,2(1x,d9.2),
     *        '  iv(ip),iv(ip+1)-1=',2i5/)
      end
c
c---------------------------------------------------------------------
c
      subroutine outst(nun,imin,imax,a,ia,ja,iu,ip,
     *                     ipmn,ipmx,iv,xp,yp,ipb)
c
c---------------------------------------------------------------------
c
c     print the problem information as produced by seta
c     in this version, stencils for unknowns are separated.
c
c     true stencil form is used
c
c---------------------------------------------------------------------
c
      implicit real*8 (a-h,o-z)
c
      dimension ia (*)
      dimension a  (*)
      dimension ja (*)
      dimension iu (*)
      dimension ip (*)
      dimension iv (*)
cveh      integer*1 ipb(*)
      integer ipb(*)

      real*8 xp (*)
      real*8 yp (*)
c
      dimension st(-1:1,-1:1)
c
c---------------------------------------------------------------------
c
      ia1=ia(imin)
      ia2=ia(imax+1)-1
      iv1=iv(ipmn)
      iv2=iv(ipmx+1)-1
      nv=imax-imin+1
      np=ipmx-ipmn+1
      write(6,999) np,nv
c
c     loop over points
c
      do 70 iip=ipmn,ipmx
      xr=xp(iip)
      yr=yp(iip)
      ivlo=iv(iip)
      ivhi=iv(iip+1)-1
      write(6,7000) iip,xp(iip),yp(iip)

      do 60 i=ivlo,ivhi
      ipt=ip(i)
      iu_row=iu(i)
      write(6,1000) iu_row,ipt,i
c
c     loop over unknowns
c
      do 40 iu_col=1,nun
      st(-1,-1) = 0.0
      st(-1, 0) = 0.0
      st(-1, 1) = 0.0
      st( 0,-1) = 0.0
      st( 0, 0) = 0.0
      st( 0, 1) = 0.0
      st( 1,-1) = 0.0
      st( 1, 0) = 0.0
      st( 1, 1) = 0.0
c
c     load coefficients into array
c
      do 5 j=ia(i),ia(i+1)-1
      if(iu(ja(j)).eq.iu_col) then
        ipt_col=ip(ja(j))
        xc=xp(ipt_col)
        yc=yp(ipt_col)
        if(xc.lt.xr.and.yc.lt.yr) st(-1,-1) = a(j)
        if(xc.lt.xr.and.yc.eq.yr) st(-1, 0) = a(j)
        if(xc.lt.xr.and.yc.gt.yr) st(-1, 1) = a(j)
        if(xc.eq.xr.and.yc.lt.yr) st( 0,-1) = a(j)
        if(xc.eq.xr.and.yc.eq.yr) st( 0, 0) = a(j)
        if(xc.eq.xr.and.yc.gt.yr) st( 0, 1) = a(j)
        if(xc.gt.xr.and.yc.lt.yr) st( 1,-1) = a(j)
        if(xc.gt.xr.and.yc.eq.yr) st( 1, 0) = a(j)
        if(xc.gt.xr.and.yc.gt.yr) st( 1, 1) = a(j)
      endif
5     continue
      write(6,2000) iu_row,iu_col,st(-1, 1),st( 0, 1),st( 1, 1)
      write(6,2001) st(-1, 0),st( 0, 0),st( 1, 0)
      write(6,2001) st(-1,-1),st( 0,-1),st( 1,-1)
      write(6,4000)
20    continue
40    continue
      write(6,4000)
60    continue
70    continue
      return
999   format(/'  # Points =',i5,'  # Variables =',i5/)
1000  format('     Eqn for Un #',i2,'   Pt #',i5,'  Vr #',i5/)
2000  format(5x,2i2,2x,1p,3(d12.5,2x))
2001  format(11x,1p,3(d12.5,2x))
4000  format(1x)
7000  format(/'    Point =',i5,'  x,y=',1p,2(1x,d9.2)/)
      end
c
c---------------------------------------------------------------------
c
      subroutine outms(imin,imax,a,ia,ja,iu)
c
c---------------------------------------------------------------------
c
c     outa - short form
c
c---------------------------------------------------------------------
c
      implicit real*8 (a-h,o-z)
c
      dimension ia (*)
      dimension a  (*)
      dimension ja (*)
      dimension iu (*)
c
c---------------------------------------------------------------------
c
      icol = 80
      nr = (icol-5)/19
c
      ia1=ia(imin)
      ia2=ia(imax+1)-1
      np=imax-imin+1
      write(6,999) np,imin,imax,ia1,ia2
c
c     loop over points
c
      do 60 i=imin,imax
      write(6,1000) i,ia(i),iu(i)
      nn=(ia(i+1)-ia(i)-1)/nr+1
      jlo=ia(i)-nr
      do 20 n=1,nn
      jlo=jlo+nr
      jhi=min0(ia(i+1)-1,jlo+nr-1)
      nnn=0
      do 10 j=jlo,jhi
      nnn=nnn+1
10    continue
      write(6,2011) n,(ja(j),iu(ja(j)),a(j),j=jlo,jhi)
20    continue
      write(6,4000)
60    continue
70    continue
      return
999   format(/2x,'  np=',i5,'  imin=',i5,'  imax=',i5,
     *  '  ia(imin)=',i5,'  ia(imax+1)-1=',i5/)
1000  format(' i=',i4,'  ia(i)=',i5,'  iu=',i2)
2011  format('  a',1p,i1,1x,6(1x,i5,1x,i1,2x,d9.2))
4000  format(1x)
      end
c
c---------------------------------------------------------------------
c
      subroutine outmss(imin,imax,a,ia,ja,iu)
c
c---------------------------------------------------------------------
c
c     outa - short form + shifted a info
c
c---------------------------------------------------------------------
c
      implicit real*8 (a-h,o-z)
c
      dimension ia (*)
      dimension a  (*)
      dimension ja (*)
      dimension iu (*)
c
c---------------------------------------------------------------------
c
      icol = 80
      nr = (icol-5)/19
c
      ia1=ia(imin)
      ia2=ia(imax+1)-1
      np=imax-imin+1
      write(6,999) np,imin,imax,ia1,ia2
c
c     loop over points
c
      do 60 i=imin,imax
      write(6,1000) i,ia(i),iu(i)
      nn=(ia(i+1)-ia(i)-1)/nr+1
      jlo=ia(i)-nr
      do 20 n=1,nn
      jlo=jlo+nr
      jhi=min0(ia(i+1)-1,jlo+nr-1)
      nnn=0
      do 10 j=jlo,jhi
      nnn=nnn+1
10    continue
      write(6,2011) n,(ja(j),iu(ja(j)),a(j),j=jlo,jhi)
20    continue
      write(6,4000)
c>>>>>
c
c     print shifted row info
c
      ishift=imax-imin+2
      ii=i+ishift
      write(6,1001) ii,ia(ii)
      nn=(ia(ii+1)-ia(ii)-1)/nr+1
      jlo=ia(ii)-nr
      do 40 n=1,nn
      jlo=jlo+nr
      jhi=min0(ia(ii+1)-1,jlo+nr-1)
      nnn=0
      do 30 j=jlo,jhi
      nnn=nnn+1
30    continue
      write(6,2011) n,(ja(j),iu(ja(j)),a(j),j=jlo,jhi)
40    continue
      write(6,4000)
1001  format(' shifted  i=',i4,'  ia(ii)=',i5)
c<<<<<
60    continue
70    continue
      return
999   format(/2x,'  np=',i5,'  imin=',i5,'  imax=',i5,
     *  '  ia(imin)=',i5,'  ia(imax+1)-1=',i5/)
1000  format(' i=',i4,'  ia(i)=',i5,'  iu=',i2)
2011  format('  a',i1,1p,1x,6(1x,i5,2x,d10.3))
4000  format(1x)
      end
c
c---------------------------------------------------------------------
c
      subroutine outbs(k,imin,imax,icg,b,ib,jb)
c
c---------------------------------------------------------------------
c
c     outb - short form
c
c---------------------------------------------------------------------
c
      implicit real*8 (a-h,o-z)
c
      dimension imin(25),imax(25)
c
      dimension icg(*)
      dimension ib (*)
      dimension b  (*)
      dimension jb (*)
c
c---------------------------------------------------------------------
c
      icol = 80
      nr = (icol-4)/15
c
      ib1=ib(imin(k))
      ib2=ib(imax(k)+1)-1
      np=imax(k)-imin(k)+1
      write(6,999) np,imin(k),imax(k),ib1,ib2
c
c     loop over points
c
      do 60 i=imin(k),imax(k)
      write(6,1000) i,ib(i),ib(i+1)-1,icg(i)
      nn=(ib(i+1)-ib(i)-1)/nr+1
      jlo=ib(i)-nr
      do 20 n=1,nn
      jlo=jlo+nr
      jhi=min0(ib(i+1)-1,jlo+nr-1)
c     write(6,2011) (jb(j),icg(jb(j)),b(j),j=jlo,jhi)
      write(6,2011) (jb(j),b(j),j=jlo,jhi)
20    continue
      write(6,4000)
60    continue
70    continue
      return
999   format(/2x,'  np=',i5,'  imin=',i5,'  imax=',i5,
     *  '  ib(imin)=',i5,'  ib(imax+1)-1=',i5/)
1000  format(' i=',i5,'  ib(i)=',i5,'  ib(i+1)-1=',i5,'  icg=',i5)
2011  format('  b',1p,1x,6(1x,i4,1x,d9.2))
4000  format(1x)
      end
CC### filename: SL.FOR
c
c==== FILE SL.FOR ====================================================
c
c     SOLVE: solve for systems
c
c=====================================================================
c
      subroutine solve(levels,ncyc,mu,ntrlx,iprlx,ierlx,iurlx,iprtc,
     *     nun,imin,imax,u,f,a,ia,ja,iu,icg,b,ib,jb,
     *     ipmn,ipmx,iv,ip,xp,yp)
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
c     storage for convergence data
c
      dimension resv(20)
c
c     work space
c
      dimension iarr(10)
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
      if(ncycle.eq.0) return
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
      return
c3999  format(/'    k   tpeu   residual     energy   res 1,2,...')
3999  format(/'    k  tpeu  residual    energy  res 1,2,...')
5000  format(26(1x,i2))
6000  format(3x,i2,2x,4i1,1p,6(1x,e9.2):/31x,4(1x,e9.2))
6001  format(3x,i2,6x,1p,6(1x,e9.2):/31x,4(1x,e9.2))
c6000  format(3x,i2,3x,4i1,1p,5(2x,e9.2))
c6001  format(3x,i2,7x,1p,5(2x,e9.2))
c6002  format(34x,1p,3(2x,e9.2))
      end
c
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
      include 'params.amg'
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
c=====================================================================
c
c     relaxation routines
c
c=====================================================================
c
      subroutine relax(k,itrel,iprel,ierel,iurel,
     *                 imin,imax,u,f,a,ia,ja,iu,icg,ipmn,ipmx,iv)
c
c---------------------------------------------------------------------
c
c     Routine to call relaxation
c
c       itrel = 1 - Gauss-Seidel
c       itrel = 2 - Kaczmarz
c       itrel = 3 - Point Gauss-Seidel
c       itrel = 4 - Point Kaczmarz (not in effect)
c       itrel = 5 - Collective relaxation (not in effect)
c       itrel = 8 - Normalization
c       itrel = 9 - Direct solver
c
c       iprel specifies C/F/G variables to relax
c       ierel specifies equation types to relax
c       iurel specifies unknown  types to relax
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

      dimension ipmn(25),ipmx(25)
      dimension iv (*)
c
c---------------------------------------------------------------------
c
      go to (100,999,300,999,999,999,999,800,900),itrel
      return
c
c     Gauss-Seidel relaxation
c
100   call relax1(k,iprel,ierel,imin,imax,u,f,a,ia,ja,iu,icg)
      return
c
c     Kaczmarz relaxation (removed)
c
c
c     Point Gauss-Seidel relaxation
c
300   call relax3(k,iprel,u,f,a,ia,ja,iv,ipmn,ipmx,icg)
      return
c
c     Collective relaxation (removed)
c
c
c     Normalization
c
800   call norml(k,iurel,imin,imax,u,iu)
      return
c
c     Direct solution (low storage)
c
900   call dirslv(k,imin,imax,u,f,a,ia,ja)
      return
999   return
      end
c
      subroutine relax1(k,iprel,ierel,imin,imax,u,f,a,ia,ja,iu,icg)
c
c---------------------------------------------------------------------
c
c     Gauss-Seidel relaxation
c
c       iprel = 1 - relax f-variables only
c       iprel = 2 - relax all variables
c       iprel = 3 - relax c-variables only
c
c       ierel = n - relax equations of type n
c       ierel = 9 - relax equations of all types
c
c       iurel = n - relax unknowns of type n
c       iurel = 9 - relax unknowns of all types
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
c
c---------------------------------------------------------------------
c
      ilo=imin(k)
      ihi=imax(k)
      go to (100,200,300) iprel
      stop 'bad iprel in relax1'
c
c     F-variable relaxation
c
100   do 120 i=ilo,ihi
      if(icg(i).gt.0) go to 120
      if(ierel.ne.iu(i).and.ierel.ne.9) go to 120
      r=f(i)
      jlo=ia(i)+1
      jhi=ia(i+1)-1
      do 110 j=jlo,jhi
      r=r-a(j)*u(ja(j))
110   continue
      u(i)=r/a(ia(i))
120   continue
      return
c
c     All-variable relaxation
c
200   do 220 i=ilo,ihi
      if(ierel.ne.iu(i).and.ierel.ne.9) go to 220
      r=f(i)
      jlo=ia(i)+1
      jhi=ia(i+1)-1
      do 210 j=jlo,jhi
      r=r-a(j)*u(ja(j))
210   continue
      u(i)=r/a(ia(i))
220   continue
      return
c
c     C-variable relaxation
c
300   do 320 i=ilo,ihi
      if(icg(i).le.0) go to 320
      if(ierel.ne.iu(i).and.ierel.ne.9) go to 320
      r=f(i)
      jlo=ia(i)+1
      jhi=ia(i+1)-1
      do 310 j=jlo,jhi
      r=r-a(j)*u(ja(j))
310   continue
      u(i)=r/a(ia(i))
320   continue
      return
      end
c
      subroutine relax3(k,iprel,u,f,a,ia,ja,iv,ipmn,ipmx,icg)
c
c---------------------------------------------------------------------
c
c     Point Gauss-Seidel relaxation
c
c       iprel = 1 - relax f-variables only
c       iprel = 2 - relax all variables
c       iprel = 3 - relax c-variables only
c
c       ierel = n - relax equations of type n
c       ierel = 9 - relax equations of all types
c
c       iurel = n - relax unknowns of type n
c       iurel = 9 - relax unknowns of all types
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

      dimension ipmn(25),ipmx(25)
c
      dimension d(10,10),s(10)
c
c---------------------------------------------------------------------
c
      iplo=ipmn(k)
      iphi=ipmx(k)
      go to (100,200,300,400,500), iprel
c
c     Relax points with first variable in F
c
100   do 180 ipt=iplo,iphi
      ilo=iv(ipt)
      if(icg(ilo).gt.0) go to 180
      ihi=iv(ipt+1)-1
      if(ihi.gt.ilo) go to 120
      r=f(ilo)
      jlo=ia(ilo)+1
      jhi=ia(ilo+1)-1
      do 110 j=jlo,jhi
      r=r-a(j)*u(ja(j))
110   continue
      u(ilo)=r/a(ia(ilo))
      go to 180
120   n=0
      ilo1=ilo-1
      nhi=ihi-ilo1
      do 160 i=ilo,ihi
      n=n+1
      do 130 nn=1,nhi
      d(n,nn)=0.e0
130   continue
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
140   s(n)=s(n)-a(j)*u(iii)
150   continue
160   continue
      call gselim(d,s,nhi)
      n=0
      do 170 i=ilo,ihi
      n=n+1
      u(i)=s(n)
170   continue
180   continue
      return
c
c     Relax all points
c
200   do 280 ipt=iplo,iphi
      ilo=iv(ipt)
      ihi=iv(ipt+1)-1
      if(ihi.gt.ilo) go to 220
      r=f(ilo)
      jlo=ia(ilo)+1
      jhi=ia(ilo+1)-1
      do 210 j=jlo,jhi
      r=r-a(j)*u(ja(j))
210   continue
      u(ilo)=r/a(ia(ilo))
      go to 280
220   n=0
      ilo1=ilo-1
      nhi=ihi-ilo1
      do 260 i=ilo,ihi
      n=n+1
      do 230 nn=1,nhi
      d(n,nn)=0.e0
230   continue
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
240   s(n)=s(n)-a(j)*u(iii)
250   continue
260   continue
      call gselim(d,s,nhi)
      n=0
      do 270 i=ilo,ihi
      n=n+1
      u(i)=s(n)
270   continue
280   continue
      return
c
c     Relax points with first variable in C
c
300   do 380 ipt=iplo,iphi
      ilo=iv(ipt)
      if(icg(ilo).le.0) go to 380
      ihi=iv(ipt+1)-1
      if(ihi.gt.ilo) go to 320
      r=f(ilo)
      jlo=ia(ilo)+1
      jhi=ia(ilo+1)-1
      do 310 j=jlo,jhi
      r=r-a(j)*u(ja(j))
310   continue
      u(ilo)=r/a(ia(ilo))
      go to 380
320   n=0
      ilo1=ilo-1
      nhi=ihi-ilo1
      do 360 i=ilo,ihi
      n=n+1
      do 330 nn=1,nhi
      d(n,nn)=0.e0
330   continue
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
340   s(n)=s(n)-a(j)*u(iii)
350   continue
360   continue
      call gselim(d,s,nhi)
      n=0
      do 370 i=ilo,ihi
      n=n+1
      u(i)=s(n)
370   continue
380   continue
      return
c
c     Relax points with at least one variable in F
c
400   do 480 ipt=iplo,iphi
      ilo=iv(ipt)
      ihi=iv(ipt+1)-1
      if(ihi.gt.ilo) go to 420
      if(icg(ilo).gt.0) go to 480
      r=f(ilo)
      jlo=ia(ilo)+1
      jhi=ia(ilo+1)-1
      do 410 j=jlo,jhi
      r=r-a(j)*u(ja(j))
410   continue
      u(ilo)=r/a(ia(ilo))
      go to 480
420   nr=0
      do 422 i=ilo,ihi
      if(icg(i).le.0) nr=nr+1
422   continue
      if(nr.eq.0) go to 480
      ilo1=ilo-1
      nhi=ihi-ilo1
      n=0
      do 460 i=ilo,ihi
      n=n+1
      do 430 nn=1,nhi
      d(n,nn)=0.e0
430   continue
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
440   s(n)=s(n)-a(j)*u(iii)
450   continue
460   continue
      call gselim(d,s,n)
      n=0
      do 470 i=ilo,ihi
      n=n+1
      u(i)=s(n)
470   continue
480   continue
      return
c
c     Relax (simultaneously) all C-variables at each point
c
500   do 580 ipt=iplo,iphi
      ilo=iv(ipt)
      ihi=iv(ipt+1)-1
      if(ihi.gt.ilo) go to 520
      if(icg(ilo).le.0) go to 580
      r=f(ilo)
      jlo=ia(ilo)+1
      jhi=ia(ilo+1)-1
      do 510 j=jlo,jhi
      r=r-a(j)*u(ja(j))
510   continue
      u(ilo)=r/a(ia(ilo))
      go to 580
520   nr=0
      ilo1=ilo-1
      nhi=ihi-ilo1
      nc=0
      do 560 i=ilo,ihi
      nc=nc+1
      if(icg(i).le.0) go to 560
      nr=nr+1
      do 530 nn=1,nhi
      d(nr,nn)=0.e0
530   continue
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
540   s(nr)=s(nr)-a(j)*u(iii)
550   continue
560   continue
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
546   continue
547   continue
548   call gselim(d,s,nr)
      n=0
      do 570 i=ilo,ihi
      if(icg(i).le.0) go to 570
      n=n+1
      u(i)=s(n)
570   continue
580   continue
      return
      end
c
      subroutine gselim(c,d,npts)
      implicit real*8 (a-h,o-z)
      dimension c(10,10),d(10)
      if(npts.gt.10) stop 'npts too large in gselim'
      if(npts.gt.1) go to 10
      d(1)=d(1)/c(1,1)
      return
c
c     perform foreward elimination
c
   10 do 150 n1=1,npts-1
      if(c(n1,n1).eq.0.e0) go to 150
      do 140 n2=n1+1,npts
      if(c(n2,n1).eq.0.e0) go to 140
      g=c(n2,n1)/c(n1,n1)
      do 130 n3=n1+1,npts
      c(n2,n3)=c(n2,n3)-g*c(n1,n3)
  130 continue
      d(n2)=d(n2)-g*d(n1)
  140 continue
  150 continue
c
c     perform back-substitution
c
      do 190 n1=npts,2,-1
      d(n1)=d(n1)/c(n1,n1)
      n2hi=n1-1
      do 180 n2=1,n2hi
      if(c(n2,n1).eq.0.e0) go to 180
      d(n2)=d(n2)-d(n1)*c(n2,n1)
  180 continue
  190 continue
      d(1)=d(1)/c(1,1)
      return
      end
c
      subroutine norml(k,iurel,imin,imax,u,iu)
c
c---------------------------------------------------------------------
c
c     Normalization (addition of constant)
c
c       iprel = 1 - relax f-variables only
c       iprel = 2 - relax all variables
c       iprel = 3 - relax c-variables only
c
c       ierel = n - relax equations of type n
c       ierel = 9 - relax equations of all types
c
c       iurel = n - relax unknowns of type n
c       iurel = 9 - relax unknowns of all types
c
c---------------------------------------------------------------------
c
      implicit real*8 (a-h,o-z)
c
      dimension imin(25),imax(25)
      dimension u  (*)
      dimension iu (*)
c
      dimension rs(10),np(10)
c
c---------------------------------------------------------------------
c
      ilo=imin(k)
      ihi=imax(k)
      do 10 n=1,10
      rs(n)=0.e0
      np(n)=0
   10 continue
      do 20 i=ilo,ihi
      if(iu(i).ne.iurel.and.iurel.ne.9) go to 20
      rs(iu(i))=rs(iu(i))+u(i)
      np(iu(i))=np(iu(i))+1
   20 continue
      do 30 n=1,10
      if(np(n).eq.0) go to 30
      rs(n)=rs(n)/float(np(n))
   30 continue
      do 40 i=ilo,ihi
      if(iu(i).ne.iurel.and.iurel.ne.9) go to 40
      u(i)=u(i)-rs(iu(i))
   40 continue
      return
      end
c
      subroutine dirslv(k,imin,imax,u,f,a,ia,ja)
c
c---------------------------------------------------------------------
c
c     Direct solution
c
c     solve the problem exactly by gauss elimination.
c
c     new version (11/12/89)
c
c     this is a "low" storage version.
c     the pointer ic locates the first entry stored in the
c     vector c. jcmn and jcmx contain the first and last
c     column numbers stored.
c
c     no pivoting in this preliminary version.
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
c
      parameter(ndimmat=100000,ndimrhs=600)
c     parameter(ndimmat=10000,ndimrhs=600)
      dimension c(ndimmat),d(ndimrhs)
      dimension ic(ndimrhs),jcmn(ndimrhs),jcmx(ndimrhs),jcmn2(ndimrhs)
c
c---------------------------------------------------------------------
c
      ilo=imin(k)
      ihi=imax(k)
      npts=ihi-ilo+1
      ishft=1-ilo
      if(npts.eq.0) return
      if(npts.gt.1) go to 1
      u(ilo)=f(ilo)/a(ia(ilo))
      return

1     if(npts.gt.600) stop 'drslv4'
c
c     load the matrix and right hand side
c
      jmx=1
      kc=1
      do 40 i=ilo,ihi
c
c     find jmn and jmx
c
      jmn=npts
      jlo=ia(i)
      jhi=ia(i+1)-1
      do 10 j=jlo,jhi
      jc=ja(j)+ishft
      if(jc.lt.jmn) jmn=jc
      if(jc.gt.jmx) jmx=jc
10    continue
      ic(i+ishft)=kc
      jshft=kc-jmn+ishft
      jcmn(i+ishft)=jmn
      jcmx(i+ishft)=jmx
      do 20 jc=jmn,jmx
      c(kc)=0.
      kc=kc+1
      if(kc.gt.ndimmat) stop 'drslv4'
20    continue
      do 30 j=jlo,jhi
      c(ja(j)+jshft)=a(j)
30    continue
      d(i+ishft)=f(i)
40    continue
c     print *,'  drslv4 -- storage used =',kc
      ic(npts+1)=kc
c
c     find icmx
c
      jmn=npts
      do 50 n1=npts,1,-1
      if(jcmn(n1).lt.jmn) jmn=jcmn(n1)
      jcmn2(n1)=jmn
50    continue
c
c     perform foreward elimination
c
100   do 200 n1=1,npts-1
      j1shft=ic(n1)-jcmn(n1)
      do 190 n2=n1+1,npts
      if(jcmn2(n2).gt.n1) go to 200
      if(jcmn(n2).gt.n1) go to 190
      j2shft=ic(n2)-jcmn(n2)
      if(c(n1+j2shft).eq.0.e0) go to 190
      g=c(n1+j2shft)/c(n1+j1shft)
      do 180 n3=n1+1,jcmx(n1)
      c(n3+j2shft)=c(n3+j2shft)-g*c(n3+j1shft)
180   continue
      d(n2)=d(n2)-g*d(n1)
190   continue
200   continue
c
c     perform back-substitution
c
      do 290 n1=npts,2,-1
      j1shft=ic(n1)-jcmn(n1)
      d(n1)=d(n1)/c(n1+j1shft)
      do 280 n2=n1-1,1,-1
      if(jcmx(n2).lt.n1) go to 290
      j2shft=ic(n2)-jcmn(n2)
      if(c(n1+j2shft).eq.0.e0) go to 280
      d(n2)=d(n2)-d(n1)*c(n1+j2shft)
280   continue
290   continue
295   d(1)=d(1)/c(1)
c
c     replace the solution
c
      do 300 n=1,npts
      u(n-ishft)=d(n)
300   continue
c     write(6,1234) npts,dnorm
c1234 format(' drslv2 -- npts=',i2,' dnorm=',1p,e9.2)
      return
      end
c
c=====================================================================
c
c     residual calculation routines
c
c=====================================================================
c
      subroutine rsdl(k,enrg,res,resv,iprt,imin,imax,u,f,a,ia,ja,iu)
c
c---------------------------------------------------------------------
c
c     compute (and print) residual
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
CC### filename: GP.FOR
c
c==== FILE GP.FOR ====================================================
c
c     GENERAL PURPOSE ROUTINES
c
c=====================================================================
c
c     timing & time/date routines
c
c=====================================================================
c
cveh      subroutine ctime(time)
      subroutine ctime(nsec)
c
c=====================================================================
c
c     returns time elaspsed since midnight (in seconds)
c
c=====================================================================
c
c time in milliseconds :
c
c       integer mclock
cveh        real*8  time
cveh        integer*4 iticks
cveh begin
cveh        time = mclock()*0.01
cveh        call timer(iticks)
cveh        iticks = 32
cveh end
cveh        time = iticks*0.01

      integer time, nsec
      nsec = time()
cveh
cveh      print *, nsec

      return
      end
c
c=====================================================================
c
c     memory usage accounting routines
c
c=====================================================================
c
      subroutine memacct(array,nhigh,iperm)
c
c=====================================================================
c
c     this routine allows the user to store intermediate
c     memory usage for a number of arrays. This data is
c     stored in a common block, and can be accessed by
c     a sampling routine called by the user at reasonable
c     (often natural) intervals.
c
c     the usage can be designated as temporary or permanent.
c     temporary storage is work storage later discarded,
c     while permanent storage is retained until the run is
c     completed. permanent usage is nondecreasing. Temporary
c     usage is assumed to be higher than permanent usage.
c
c     Input:
c
c       array - character string containing array name
c       nhigh - amount of array used (any integer measure)
c       iperm - 0 = temporary, otherwise permanent
c
c     this routine stores the name for future reference. If
c     some particular order is desired, an initial call
c     with each of the array names (and nhigh=0) will ensure
c     a particular order for storing and reporting.
c
c     currently set for 20 names and 20 sampling times.
c
c---------------------------------------------------------------------
c
      integer nhigh
      character*(*) array
      character*10  name,alias

      parameter(maxname=30,maxsmpl=30)

      common /memory/ name(maxname),alias(maxname),ialias(maxname),
     *                memtemp(maxname),memperm(maxname),
     *                memmax(maxname),memlim(maxname),
     *                nname,nalias
c
c---------------------------------------------------------------------
c
c     initialize name counter
c
      call meminit(0)
c
c     search list for array name
c
      do 10 nn=1,nname
      if(array.eq.name(nn)) go to 30
10    continue
c
c     name not found. check for alias.
c
      do 20 nna=1,nalias
      if(array.eq.alias(nna)) then
        nn=ialias(nna)
        go to 30
      endif
20    continue
c
c     name not found. add to list and initialize usage to zero.
c
      if(nname.ge.maxname) then
        print *,' Too many names in MEMHIGH - name ignored...'
        return
      endif
c
      nname=nname+1
c
      name(nname)=array
      nn=nname
      memtemp(nn)=0
      memperm(nn)=0
      memmax (nn)=0
      memlim (nn)= 999999999
      if(nhigh.eq.0) return
c
c     update usage
c
30    if(iperm.eq.0) then
c
c     temporary use
c
        memtemp(nn)=nhigh
        if(nhigh.gt.memmax(nn)) memmax(nn)=nhigh
        if(nhigh.gt.memlim(nn)) write(6,
     *    '(/'' *** WARNING *** limit exceeded for '',a/)') name(nn)
      else
c
c     permanent use
c
        memperm(nn)=nhigh
        if(nhigh.gt.memtemp(nn)) memtemp(nn)=nhigh
        if(nhigh.gt.memmax(nn)) memmax(nn)=nhigh
        if(nhigh.gt.memlim(nn)) write(6,
     *    '(/'' *** WARNING *** limit exceeded for '',a/)') name(nn)
      endif
c
      return
      end
c
c=====================================================================
c
      subroutine meminit(irestart)
c
c=====================================================================
c
c     initializes name & sample counters
c     done if first call (init=0) or forced (irestart=0)
c
c---------------------------------------------------------------------
c
      character*10  name,alias

      parameter(maxname=30,maxsmpl=30)

      common /memory/ name(maxname),alias(maxname),ialias(maxname),
     *                memtemp(maxname),memperm(maxname),
     *                memmax(maxname),memlim(maxname),
     *                nname,nalias

      common /sample/ memtmps(maxname,maxsmpl),
     *                memprms(maxname,maxsmpl),
     *                memmaxs(maxname,maxsmpl),nsample

      data init /0/
      save
c
c---------------------------------------------------------------------
c
c     initialize name & sample counter
c
      if(init.eq.0.or.irestart.ne.0) then
        nname  =0
        nalias =0
        nsample=0
        init   =1
      endif
      return
      end
c
c
c=====================================================================
c
      subroutine msample
c
c=====================================================================
c
c     store current memory usage & increment sample counter
c
c---------------------------------------------------------------------
c
      character*10  name,alias

      parameter(maxname=30,maxsmpl=30)

      common /memory/ name(maxname),alias(maxname),ialias(maxname),
     *                memtemp(maxname),memperm(maxname),
     *                memmax(maxname),memlim(maxname),
     *                nname,nalias

      common /sample/ memtmps(maxname,maxsmpl),
     *                memprms(maxname,maxsmpl),
     *                memmaxs(maxname,maxsmpl),nsample
c
c---------------------------------------------------------------------
c
      if(nsample.ge.maxsmpl) then
        print *,' Limit reached in MSAMPLE - last data overwritten...'
      else
        nsample=nsample+1
      endif

      do 10 nn=1,nname
      memtmps(nn,nsample)=memtemp(nn)
      memprms(nn,nsample)=memperm(nn)
      memmaxs(nn,nsample)=memmax (nn)
10    continue
      return
      end
c
c=====================================================================
c
      subroutine memalias(newname,oldname)
c
c=====================================================================
c
c     allows accounting for "oldname" based on "newname"
c     in reports, only oldname is used.
c
c---------------------------------------------------------------------
c
      integer nhigh
      character*(*) oldname,newname
      character*10  name,alias

      parameter(maxname=30,maxsmpl=30)

      common /memory/ name(maxname),alias(maxname),ialias(maxname),
     *                memtemp(maxname),memperm(maxname),
     *                memmax(maxname),memlim(maxname),
     *                nname,nalias
c
c---------------------------------------------------------------------
c
c     search alias list for newname - if it exists, return
c
      do 10 nna=1,nalias
      if(newname.eq.alias(nna)) return
10    continue
c
c     search list for oldname
c
      do 20 nn=1,nname
      if(oldname.eq.name(nn)) go to 30
20    continue
c
c     name not found. check for alias of alias.
c
30    do 40 nna=1,nalias
      if(oldname.eq.alias(nna)) then
        nn=ialias(nna)
        go to 50
      endif
40    continue
      stop ' old name does not exist'
c
c     add newname to alias list & reference oldname
c
50    nalias=nalias+1
c
      if(nalias.ge.maxname) then
        print *,' Too many names in MEMALIAS - name ignored...'
        return
      endif
c
      alias(nalias) =newname
      ialias(nalias)=nn
      return
      end
c
c=====================================================================
c
      subroutine mreport(ireport)
c
c=====================================================================
c
c     print report of memory usage
c
c       ireport = 1 - summary (max, permanent)
c       ireport = 2 - history + summary
c
c---------------------------------------------------------------------
c
      character*10  name,alias

      parameter(maxname=30,maxsmpl=30)

      common /memory/ name(maxname),alias(maxname),ialias(maxname),
     *                memtemp(maxname),memperm(maxname),
     *                memmax(maxname),memlim(maxname),
     *                nname,nalias

      common /sample/ memtmps(maxname,maxsmpl),
     *                memprms(maxname,maxsmpl),
     *                memmaxs(maxname,maxsmpl),nsample
c
c---------------------------------------------------------------------
c
      go to (100,200),ireport
      return
c
c     summary report
c
100   write(6,1000)
      do 110 nn=1,nname
      if(memlim(nn).eq.999999999) then
        write(6,1100) name(nn),memmax(nn),memperm(nn)
      else
        write(6,1200) name(nn),memmax(nn),memperm(nn),memlim(nn)
      endif
110   continue
      write(6,'(1x)')
c
1000  format(/'      SUMMARY OF ARRAY USAGE IN AMGS01 '/
     *        '      array        maximum        final      (limit)'/)
1100  format(6x,a7,5x,i8,5x,i8)
1200  format(6x,a7,5x,i8,5x,i8,5x,i8)
c
      return
c
c     history report
c
200   write(6,2000)
      do 250 nn=1,nname
      ns=1
      write(6,2100) name(nn),ns,memtmps(nn,ns),
     *              memprms(nn,ns),memmaxs(nn,ns)
      do 240 ns=2,nsample
      write(6,2200) ns,memtmps(nn,ns),
     *              memprms(nn,ns),memmaxs(nn,ns)
240   continue
      write(6,'(1x)')
250   continue
c
2000  format(/'      HISTORY OF TEMPORARY AND PERMANENT ARRAY USAGE'/
     *        '      array       k   temp use   perm use   max  use'/)
2100  format(6x,a7,4x,i2,3(3x,i8))
2200  format(17x,i2,3(3x,i8))
c
      return
      end
c
c=====================================================================
c
      subroutine memlimit(array,limit)
c
c=====================================================================
c
c     this routine allows the user to load array bounds
c     into the memory usage routines. Any further calls
c     will test against this limit and issue a warning if
c     it is exceeded. This can also be used to add the
c     array to the memory usage data.
c
c---------------------------------------------------------------------
c
      integer limit
      character*(*) array
      character*10  name,alias

      parameter(maxname=30,maxsmpl=30)

      common /memory/ name(maxname),alias(maxname),ialias(maxname),
     *                memtemp(maxname),memperm(maxname),
     *                memmax(maxname),memlim(maxname),
     *                nname,nalias
c
c---------------------------------------------------------------------
c
c     initialize name & sample counter
c
      call meminit(0)
c
c     search list for array name
c
      do 10 nn=1,nname
      if(array.eq.name(nn)) go to 30
10    continue
c
c     name not found. add to list and initialize usage to zero.
c
      if(nname.ge.maxname) then
        print *,' Too many names in MEMHIGH - name ignored...'
        return
      endif
c
      nname=nname+1
c
      name(nname)=array
      nn=nname
      memtemp(nn)=0
      memperm(nn)=0
      memmax (nn)=0
c
c     add limit
c
30    memlim (nn)=limit
c
      return
      end
