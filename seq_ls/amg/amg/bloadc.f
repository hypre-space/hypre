c-------------------------------------------------------------------     
      subroutine bloadc(k,imin,imax,icg,b,ib,jb,
     *     ndimu,ndimp,ndima,ndimb)
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
c     include 'params.amg'
c     
      dimension imin(*),imax(*)
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
 10   continue
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
            stop
 25         b(jjb)=b(j)
            jb(jjb)=i
            jb(ib(ii))=jjb
 20      continue
 30   continue
c     
c     compress the matrix.
c     
      kb=ib(i0lo)
      do 70 i=i0lo,i0hi
         if(icg(i).gt.0) go to 50
c===  > f-row
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
 40      continue
         go to 70
c===  > c-row
 50      jlo=ib(i)+1
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
 60      continue
 70   continue
      ib(i0hi+1)=kb
      return
 1000 format('  ##### c-row overflow in bloadc ##### ',2i5)
      end
