c
c=====================================================================
c
c     the following routines are used for setting the form of b
c
c=====================================================================
c
      subroutine binitl(k,imin,imax,ia,ja,ifg,b,ib,jb)
c
c---------------------------------------------------------------------
c
c     *** new version *** 7/24/89
c
c     here, direct connections are loaded first, and on exit,
c     jb(ib(i)) points to the last entry directly strongly
c     connected to i.
c
c     this routine sets the form of b so that row i has enough
c     room for whichever is larger, the set of strong connections
c     or the set of variables strongly connected to i. the set of
c     variables strongly connected to i is loaded into the row,
c     and jb(ib(i)) is set to the last used entry in the row.
c
c     note: a variable name containing a 1 is generally a shifted
c     version of the corresponding variable whose name contains a 0.
c
c---------------------------------------------------------------------
c
      implicit real*8 (a-h,o-z)
c
      include 'params.amg'
c
      dimension imin(25),imax(25)
      dimension ia (*)
      dimension ja (*)
      dimension ifg(*)

      dimension ib (*)
      dimension b  (*)
      dimension jb (*)
c
c---------------------------------------------------------------------
c
      i0lo=imin(k)
      i0hi=imax(k)
      ishift=i0hi-i0lo+2
      i1lo=i0lo+ishift
      i1hi=i0hi+ishift
c
c     count the number of variables strongly connected to each variabl
c
      do 10 i=i0lo,i0hi
      ifg(i)=0
   10 continue
      do 30 i1=i1lo,i1hi
      jlo=ia(i1)+1
      jhi=ia(i1+1)-1
      do 20 j=jlo,jhi
      ii0=ja(j)
      ifg(ii0)=ifg(ii0)+1
   20 continue
   30 continue
c
c     set the form of b
c
      ib(1)=1
      kb=ib(i0lo)
      do 50 i0=i0lo,i0hi
      i1=i0+ishift
c
c     put 1 on diagonal, 0's off
c
      b(kb)=1.e0
      jb(kb)=kb
c
c     determine number of off-diagonals
c
      ns=ia(i1+1)-ia(i1)-1
      nst=ifg(i0)
      ib(i0)=kb
      kb=kb+1
      jjhi=max0(ns,nst)
      do 40 jj=1,jjhi
      b(kb)=0.e0
      jb(kb)=0
      kb=kb+1
   40 continue
   50 continue
      ib(i0hi+1)=kb
c===> check b and jb for overflow
      if(kb.gt.ndimb) go to 9902
c
c     load the transpose for direct connections
c
      do 70 i1=i1lo,i1hi
      i0=i1-ishift
      j1lo=ia(i1)+1
      j1hi=ja(ia(i1))
      do 60 j1=j1lo,j1hi
      ii0=ja(j1)
      jjb=jb(ib(ii0))+1
      jb(jjb)=i0
      jb(ib(ii0))=jjb
   60 continue
   70 continue
c
c     set ifg to last direct entry
c
      do 80 i0=i0lo,i0hi
      ifg(i0)=jb(ib(i0))
   80 continue
c
c     load the transpose for indirect connections
c
      do 100 i1=i1lo,i1hi
      i0=i1-ishift
      j1lo=ja(ia(i1))+1
      j1hi=ia(i1+1)-1
      if(j1lo.gt.j1hi) go to 100
      do 90 j1=j1lo,j1hi
      ii0=ja(j1)
      jjb=ifg(ii0)+1
      jb(jjb)=i0
      ifg(ii0)=jjb
   90 continue
  100 continue
c
c     set temp memory usage
c
      call memacct('ib',i1hi+1,0)
      call memacct('b',kb,0)
      call memacct('jb',kb,0)
c
      return
c
c===> error messages
c
 9902 write(6,9920)
        stop
c
 9920 format(' ### error in binitl: ndimb too small ###')
      end
c
      subroutine bloadf(k,imin,imax,a,ia,ja,icg,b,ib,jb)
c
c---------------------------------------------------------------------
c
c     this routine loads the strong connections of each
c     f-variable i into the row i. this is called after
c     the c/f choice but before interpolation weights
c     have been defined. the f-diagonal entry is removed.
c
c     note: a variable containing a 1 is generally a shifted
c     version of the corresponding variable containing a 0.
c
c---------------------------------------------------------------------
c
      implicit real*8 (a-h,o-z)
c
      include 'params.amg'
c
      dimension imin(25),imax(25)
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
      i0lo=imin(k)
      i0hi=imax(k)
      ishift=i0hi-i0lo+2
      kb=ib(i0lo)
      do 60 i0=i0lo,i0hi
      if(icg(i0).le.0) go to 30
c===> i0 is a c-variable
      jlo=ib(i0)+1
      jhi=jb(ib(i0))
      b(kb)=1.e0
      jb(kb)=i0
      ib(i0)=kb
      kb=kb+1
      if(jlo.gt.jhi) go to 60
      do 10 j=jlo,jhi
      i1=jb(j)
      if(icg(i1).gt.0) go to 10
      b(kb)=0.e0
      jb(kb)=i1
      kb=kb+1
   10 continue
      go to 60
c===> i0 is an f-variable
   30 i1=i0+ishift
      jlo=ia(i1)+1
      jdhi=ja(ia(i1))
      jhi=ia(i1+1)-1
      ib(i0)=kb
      if(jlo.gt.jhi) go to 60
      do 40 j=jlo,jhi
      ii0=ja(j)
      if(icg(ii0).lt.0) go to 40
      b(kb)=0.e0
      if(j.le.jdhi) b(kb)=a(j)
      jb(kb)=ii0
      kb=kb+1
   40 continue
   60 continue
      ib(i0hi+1)=kb
      if(kb.gt.ndimb) go to 9902
c
c     set temp memory usage
c
      call memacct('b',kb,0)
      call memacct('jb',kb,0)
      return
c
c===> error messages
c
 9902 write(6,9920)
        stop
c
 9920 format(' ### error in bloadf: ndimb too small ###')
      end
c
      subroutine bloadc(k,imin,imax,icg,b,ib,jb)
c
c---------------------------------------------------------------------
c
c     this routine loads the transpose of interpolation
c     into the c-rows of b. in addition, unused entries
c     in the f-rows (which will fall at the end and are
c     marked by jb=0 in the first such location) will
c     be taken out.
c
c---------------------------------------------------------------------
c
      implicit real*8 (a-h,o-z)
c
      include 'params.amg'
c
      dimension imin(25),imax(25)
      dimension icg(*)

      dimension ib (*)
      dimension b  (*)
      dimension jb (*)
c
c---------------------------------------------------------------------
c
      i0lo=imin(k)
      i0hi=imax(k)
c
c     this assumes that each f-row contains no diagonal, and that
c     the first unwanted entry in the row is marked with jb=0.
c     zero entries are thrown away.
c
      if(ib(i0hi+1).gt.ndimb) stop 'b overflow in bloadc'
      do 10 i=i0lo,i0hi
      if(icg(i).gt.0) jb(ib(i))=ib(i)
   10 continue
c
c     add c-row entries. jb(ib(i)) points to last used entry.
c
      do 30 i=i0lo,i0hi
      if(icg(i).gt.0) go to 30
      jlo=ib(i)
      jhi=ib(i+1)-1
      if(jlo.gt.jhi) go to 30
      do 20 j=jlo,jhi
      ii=jb(j)
      if(ii.eq.0) go to 30
      if(b(j).eq.0.e0) go to 20
      jjb=jb(ib(ii))+1
      if(jjb.lt.ib(ii+1)) go to 25
      write(6,1000) i,ii
c     call outa(k,5)
      stop
   25 b(jjb)=b(j)
      jb(jjb)=i
      jb(ib(ii))=jjb
   20 continue
   30 continue
c
c     set temp memory usage
c
      call memacct('b',ib(i0hi+1),0)
      call memacct('jb',ib(i0hi+1),0)
c
c     compress the matrix.
c
      kb=ib(i0lo)
      do 70 i=i0lo,i0hi
      if(icg(i).gt.0) go to 50
c===> f-row
      jlo=ib(i)
      jhi=ib(i+1)-1
      ib(i)=kb
      if(jlo.gt.jhi) go to 70
      do 40 j=jlo,jhi
      if(jb(j).eq.0) go to 70
      if(b(j).eq.0.e0) go to 40
      b(kb)=b(j)
      jb(kb)=jb(j)
      kb=kb+1
   40 continue
      go to 70
c===> c-row
   50 jlo=ib(i)+1
      jhi=jb(ib(i))
      ib(i)=kb
      jb(kb)=i
      b(kb)=1.e0
      kb=kb+1
      if(jlo.gt.jhi) go to 70
      do 60 j=jlo,jhi
      b(kb)=b(j)
      jb(kb)=jb(j)
      kb=kb+1
   60 continue
   70 continue
      ib(i0hi+1)=kb
      return
 1000 format('  ##### c-row overflow in bloadc ##### ',2i5)
      end
