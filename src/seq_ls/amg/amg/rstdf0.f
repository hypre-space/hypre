c     
c=====================================================================
c     
c     the following routines are used for defining restriction
c     
c=====================================================================
c     
      subroutine rstdf0(k,imin,imax,icg,b,ib,jb,
     *     ndimu,ndimp,ndima,ndimb)
c     
c---------------------------------------------------------------------
c     
c     this loads the transpose of interpolation into c-rows of b
c     
c     1. No assumptions are made about previous C-row storage
c     (Except that the C-rows must have had at least 1 entry)
c     
c     2. F-rows are assumed to have no diagonal entry.
c     
c     3. Zero entries in f-rows are removed.
c     
c     4. F-row entries after jb=0 are removed (assumed marked)
c     
c     5. Points with icg = 0 are treated as F-points.
c     
c     **6. F (or special) points are now also compressed from f-rows.
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
c     Compress C-rows and unwanted F-row entries
c     
      ilo=imin(k)
      ihi=imax(k)
      kb=ib(ilo)
      do 30 i=ilo,ihi
c     
c     C-row (NOTE: ICG > 0) - make diagonal entry
c     
         if(icg(i).gt.0) then
            ib(i)=kb
            jb(ib(i))=1
            b(kb)=0.e0
            kb=kb+1
c     
c     F-row - compress out zeroes & anything past jb=0
c     added test: compress out any f-connections.
c     
         else
            ib(i)=kb
            jlo=ib(i)
            jhi=ib(i+1)-1
            do 10 j=jlo,jhi
               if(jb(j).eq.0) go to 20
               if(b(j).eq.0.e0) go to 10
c     
               if(icg(jb(j)).le.0) go to 10
c     
               b(kb)=b(j)
               jb(kb)=jb(j)
               kb=kb+1
 10         continue
 20         continue
         endif
 30   continue
      ib(ihi+1)=kb
c     
c     Count extra entries needed for C-rows (in jb(ib(i)))
c     Increase kb (end of matrix pointer) to allow room.
c     
      do 50 i=ilo,ihi
         if(icg(i).gt.0) go to 50
         do 40 j=ib(i),ib(i+1)-1
            ii=jb(j)
            jb(ib(ii))=jb(ib(ii))+1
            kb=kb+1
 40      continue
 50   continue
c     
c     Expand matrix, starting from top to avoid overwriting data
c     
      do 70 i=ihi,ilo,-1
         if(icg(i).gt.0) then
            ib(i+1)=kb
            kb=kb-jb(ib(i))
            b(kb)=1.e0
            jb(kb)=kb+1
         else
            jlo=ib(i)
            jhi=ib(i+1)-1
            ib(i+1)=kb
            do 60 j=jhi,jlo,-1
               kb=kb-1
               b(kb)=b(j)
               jb(kb)=jb(j)
 60         continue
         endif
 70   continue
c     
c     load c-rows
c     
      do 90 i=ilo,ihi
         if(icg(i).gt.0) go to 90
         jlo=ib(i)
         jhi=ib(i+1)-1
         do 80 j=jlo,jhi
            ii=jb(j)
            kb=jb(ib(ii))
            b(kb)=b(j)
            jb(kb)=i
            jb(ib(ii))=kb+1
 80      continue
 90   continue
c     
c     reset jb(ib(i)) for c-rows
c     
      do 100 i=ilo,ihi
         if(icg(i).gt.0) jb(ib(i))=i
 100  continue
      return
      end
