c     
      subroutine trunc(k,imin,imax,a,ia,ja)
c     
c---------------------------------------------------------------------
c     
c     performs a nonsymmetric truncation of all zero off-diagonals
c     
c---------------------------------------------------------------------
c     
      implicit real*8 (a-h,o-z)
c     
      dimension imin(*),imax(*)
      dimension ia (*)
      dimension a  (*)
      dimension ja (*)
c     
c---------------------------------------------------------------------
c     
      etr=1.e-10
c     etr=1.e-4
c     
      ilo=imin(k)
      ihi=imax(k)
      ka=ia(ilo)
      do 20 i=ilo,ihi
         jlo=ia(i)+1
         jhi=ia(i+1)-1
         a(ka)=a(ia(i))
         eps=etr*abs(a(ia(i)))
         ja(ka)=i
         ia(i)=ka
         ka=ka+1
         do 10 j=jlo,jhi
            if(abs(a(j)).le.eps) go to 10
            a(ka)=a(j)
            ja(ka)=ja(j)
            ka=ka+1
 10      continue
 20   continue
      ia(ihi+1)=ka
      return
      end
c     
      subroutine symm(k,isymm,imin,imax,a,ia,ja,icg,ifg)
c     
c---------------------------------------------------------------------
c     
c     symm puts the matrix into symmetric form by adding zeroes
c     
c---------------------------------------------------------------------
c     
      implicit real*8 (a-h,o-z)
c     
      dimension imin(*),imax(*)
      dimension ia (*)
      dimension a  (*)
      dimension ja (*)
      dimension icg(*)
      dimension ifg(*)
c     
c---------------------------------------------------------------------
c     
      if(isymm.eq.0) return
c     
      ilo=imin(k)
      ihi=imax(k)
c     
c     initialize ifg
c     
      do 10 i=ilo,ihi
         ifg(i)=0
 10   continue
c     
c     count the number of additional entries needed
c     per row and store this number in ifg.
c     
      nexe=0
      do 70 i=ilo,ihi
         jlo=ia(i)+1
         jhi=ia(i+1)-1
         if(jlo.gt.jhi) go to 70
         do 60 j=jlo,jhi
            ii=ja(j)
            if(ii.gt.i) go to 30
            if(ii.lt.0) go to 20
            ifg(ii)=ifg(ii)+1
            nexe=nexe+1
            ja(j)=-ja(j)
            go to 60
 20         ja(j)=-ii
            go to 60
 30         jjlo=ia(ii)+1
            jjhi=ia(ii+1)-1
            if(jjlo.gt.jjhi) go to 50
            do 40 jj=jjlo,jjhi
               if(ja(jj).ne.i) go to 40
               ja(jj)=-ja(jj)
               go to 60
 40         continue
 50         ifg(ii)=ifg(ii)+1
            nexe=nexe+1
            ja(j)=-ja(j)
 60      continue
 70   continue
      ka=ia(ihi+1)+nexe
      if(nexe.eq.0) go to 200
c     
c     expand matrix
c     
      iib=ilo+ihi
      do 120 ii=ilo,ihi
         i=iib-ii
         jlo=ia(i)
         jhi=ia(i+1)-1
         ia(i+1)=ka
         if(ifg(i).eq.0) go to 100
         nhi=ifg(i)
         do 90 n=1,nhi
            ka=ka-1
            a(ka)=0.e0
            ja(ka)=0
 90      continue
 100     ifg(i)=ka
         jjb=jlo+jhi
         do 110 jj=jlo,jhi
            j=jjb-jj
            ka=ka-1
            a(ka)=a(j)
            ja(ka)=ja(j)
 110     continue
 120  continue
c     
c     fill in a and ja entries
c     
      do 140 i=ilo,ihi
         jlo=ia(i)+1
         jhi=ia(i+1)-1
         if(jlo.gt.jhi) go to 140
         do 130 j=jlo,jhi
            if(ja(j).eq.0) go to 140
            if(ja(j).gt.0) go to 130
            ja(j)=-ja(j)
            jj=ifg(ja(j))
            ja(jj)=i
            ifg(ja(j))=jj+1
 130     continue
 140  continue
c     
c     reset ifg
c     
 200  if(k.eq.1) return
      iflo=imin(k-1)
      ifhi=imax(k-1)
      do 210 i=iflo,ifhi
         ic=icg(i)
         if(ic.le.0) go to 210
         ifg(ic)=i
 210  continue
      return
      end
