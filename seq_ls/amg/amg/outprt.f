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

cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
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
cveh  dimension iarr(20)
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
