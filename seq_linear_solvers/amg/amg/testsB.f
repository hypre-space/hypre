c     
c=====================================================================
c     
c     routines to test the f-points from coloring
c     
c=====================================================================
c     
      subroutine ftest(k,itst,ncolor,ewt,nwt,iact,npts,icdep,
     *     imin,imax,a,ia,ja,iu,ip,icg,ifg,ib,iv,
     *     ndimu,ndimp,ndima,ndimb)
c     
c---------------------------------------------------------------------
c     
c     f-point tests
c     
c     itest = 0 - no test
c     itest = 1 - test with sum of matrix connections to Ci
c     itest = 2 - test with number of strong connections to Ci
c     
c---------------------------------------------------------------------
c     
      implicit real*8 (a-h,o-z)
c     
c     include 'params.amg'
c     
      dimension imin(25),imax(25)
      dimension ia (*)
      dimension a  (*)
      dimension ja (*)
      dimension iu (*)
      dimension ip (*)
      dimension icg(*)
      dimension ifg(*)

      dimension ib (*)
      dimension iv (*)

      dimension icdep(10,10)
c     
c---------------------------------------------------------------------
c     
      if(itst.eq.0) return
      go to (10,20),itst
      stop 'ftest does not exist'
 10   call ftest1(k,ncolor,ewt,nwt,iact,npts,icdep,
     *     imin,imax,a,ia,ja,iu,ip,icg,ifg,ib,iv,
     *     ndimu,ndimp,ndima,ndimb)
      return
 20   call ftest2(k,ncolor,iact,npts,icdep,
     *     imin,imax,a,ia,ja,iu,ip,icg,ifg,ib,iv,
     *     ndimu,ndimp,ndima,ndimb)
      return
      end
c     
      subroutine ftest1(k,ncolor,ewt,nwt,iact,npts,icdep,
     *     imin,imax,a,ia,ja,iu,ip,icg,ifg,ib,iv,
     *     ndimu,ndimp,ndima,ndimb)
c     
c---------------------------------------------------------------------
c     
c     f-point tests
c     
c     itest = 0 - no test
c     itest = 1 - test with sum of matrix connections to Ci
c     itest = 2 - test with number of strong connections to Ci
c     
c---------------------------------------------------------------------
c     
      implicit real*8 (a-h,o-z)
c     
c     include 'params.amg'
c     
      dimension imin(25),imax(25)
      dimension ia (*)
      dimension a  (*)
      dimension ja (*)
      dimension iu (*)
      dimension ip (*)
      dimension icg(*)
      dimension ifg(*)

      dimension ib (*)
      dimension iv (*)

      dimension iarr(20)

      dimension icdep(10,10)
c     
c---------------------------------------------------------------------
c     
c     assemble the necessary parameters for setw
c     
      call idec(nwt,5,ndigit,iarr)
      icgd=iarr(3)
      icgr=iarr(4)
      eps2=ewt
c     
      ilo=imin(k)
      ihi=imax(k)
      ishift=ihi-ilo+2
c     
c     set top of stack pointer for recolor stack
c     
      itop=ihi+ishift+1
      ib(itop)=-1
      npts=0
      nnewpt=0
      do 10 i=ilo,ihi
         ifg(i)=0
 10   continue
c     
c     only newly colored f-points are tested
c     
      do 300 i=ilo,ihi
         if(icg(i).ne.ncolor) go to 300
         if(icdep(iu(i),iu(i)).eq.0) go to 300
         i1=i+ishift
         j1lo=ia(i1)+1
c===  > upper bound in row for direct strong connections
         j1dhi=ja(ia(i1))
c===  > upper bound in row for all strong connections
         j1hi=ia(i1+1)-1
c     
c     mark points in the s(i) as follows:
c     
c     if ifg(jj) = 1   jj is a special point
c     = 2   jj is a c-point
c     
c     if i has no neighbors, there are no interpolation weights
c     
         if(j1lo.gt.j1hi) go to 300
         do 20 j1=j1lo,j1hi
            ii=ja(j1)
            if(icg(ii).gt.0) ifg(ii)=2
            if(icg(ii).eq.0) ifg(ii)=1
 20      continue
         nun=0
 100     continue
c     
c     sweep over directly strongly connected f points
c     test to see whether each is covered by a total
c     weight defined by eps
c     
c     if the f-point has no strong connections, define it
c     to be covered, but do not distribute the weight.
c     
         do 200 j1=j1lo,j1dhi
            ii0=ja(j1)
            if(ifg(ii0).gt.0) go to 200
            eps2i=eps2
            if(icgr.eq.2) eps2i=eps2*abs(a(j1)/a(j1lo))
            s=0.e0
            ncn=0
            jjlo=ia(ii0)+1
            jjhi=ia(ii0+1)-1
            if(jjlo.gt.jjhi) go to 120
            do 110 jj=jjlo,jjhi
               if(ifg(ja(jj)).le.1) go to 110
c     
               aij=a(jj)/a(jjlo)
               if(aij.le.0.e0) go to 110
c     
               s=s+a(jj)
               if(aij.gt.eps2) ncn=ncn+1
 110        continue
 120        if(ifg(ii0).le.-999) go to 190
            go to (130,140),icgd
 130        if(ncn.gt.0) go to 190
            go to 150
 140        if(abs(s/a(jjlo)).ge.eps2i) go to 190
c     
c     the point has failed the test
c     
 150        if(iact.eq.2) go to 160
            icg(i)=-1
            npts=npts+1
c     
c     put the point on the stack
c     
            ib(itop)=i1
            ib(i1)=-1
            itop=i1
            go to 210
c     
c     force c-points
c     
 160        nun=nun+1
            if(nun.gt.1) go to 170
            ic=ii0
            nnewpt=nnewpt+1
            icg(ic)=3
            ifg(ic)=2

            go to 100
c     
c     second uncovered neighbor found - i put in C and exit loop
c     
 170        icg(i)=4
            icg(ic)=-2
            ic=i
            go to 210
 190        ifg(ii0)=-999
 200     continue
c     
c     end of test for point i
c     
 210     if(nun.gt.0) then
c     
c     C-variable ic has been forced.
c     Force all dependent/coupled variables.
c     
            iilo=iv(ip(ic))
            iihi=iv(ip(ic)+1)-1
            do 220 ii=iilo,iihi
               if(ii.eq.ic) go to 220
               if(icdep(iu(ii),iu(ic)).ne.0) icg(ii)=5
 220        continue

         endif
c     
c     reset ifg
c     
         do 290 j=j1lo,j1hi
            ifg(ja(j))=0
 290     continue
 300  continue
 1000 format('  ftest1  k=',i2,' # tested points =',i4,
     *     ' # added points =',i4)
      return
      end
c     
      subroutine ftest2(k,ncolor,iact,npts,icdep,
     *     imin,imax,a,ia,ja,iu,ip,icg,ifg,ib,iv,
     *     ndimu,ndimp,ndima,ndimb)
c     
c---------------------------------------------------------------------
c     
c     this is a simplified version which only requires that
c     a strong f-neighbor of i have one strong connection
c     to one of i's interpolation points (or a special point)
c     
c---------------------------------------------------------------------
c     
      implicit real*8 (a-h,o-z)
c     
c     include 'params.amg'
c     
      dimension imin(25),imax(25)
      dimension ia (*)
      dimension ja (*)
      dimension iu (*)
      dimension ip (*)
      dimension icg(*)
      dimension ifg(*)

      dimension ib (*)
      dimension iv (*)
c     
      dimension ntu(10),nfu(10)

      dimension icdep(10,10)
c     
c---------------------------------------------------------------------
c     
c     print *,' ftest2 - k=',k
      ilo=imin(k)
      ihi=imax(k)
      ishift=ihi-ilo+2

      ntu(1)=0
      ntu(2)=0
      ntu(3)=0
      nfu(1)=0
      nfu(2)=0
      nfu(3)=0
c     
c     set top of stack pointer for recolor stack
c     
      itop=ihi+ishift+1
      ib(itop)=-1
      npts=0
      nnewpt=0
      ntest=0
      do 10 i=ilo,ihi
         ifg(i)=0
 10   continue
c     
c     only newly colored f-points are tested
c     
      do 300 i=ilo,ihi
         if(icg(i).ne.ncolor) go to 300
         if(icdep(iu(i),iu(i)).eq.0) go to 300
         ntu(iu(i))=ntu(iu(i))+1
         ntest=ntest+1
         i1=i+ishift
         j1lo=ia(i1)+1
c===  > upper bound in row for direct strong connections
         j1dhi=ja(ia(i1))
c===  > upper bound in row for all strong connections
         j1hi=ia(i1+1)-1
c     
c     mark points in the s(i) as follows:
c     
c     if ifg(jj) = 1   jj is a special point
c     = 2   jj is a c-point
c     
c     if i has no neighbors, there are no interpolation weights
c     
         if(j1lo.gt.j1hi) go to 300
         do 20 j1=j1lo,j1hi
            ii=ja(j1)
            if(icg(ii).gt.0) ifg(ii)=2
            if(icg(ii).eq.0) ifg(ii)=1
 20      continue
         nun=0
 100     continue
c     
c     sweep over directly strongly connected f points
c     test to see whether each is connected to s(i)
c     
c     if the f-point has no strong connections, define it
c     to be covered, but do not distribute the weight.
c     
         do 200 j1=j1lo,j1dhi
            ii0=ja(j1)
            if(ifg(ii0).gt.0) go to 200
            ncn=0
            jj1lo=ia(ii0+ishift)+1
            jj1dhi=ja(ia(ii0+ishift))
            if(jj1lo.gt.jj1dhi) go to 120
            do 110 jj1=jj1lo,jj1dhi
               if(ifg(ja(jj1)).gt.0) ncn=ncn+1
 110        continue
 120        if(ifg(ii0).le.-999) go to 190
            if(ncn.gt.0) go to 190
            go to 150
c     
c     the point has failed the test
c     
 150        if(iact.eq.2) go to 160
c     
c     iact=1 - set for recoloring
c     
            icg(i)=-1
            nfu(iu(i))=nfu(iu(i))+1
            npts=npts+1
c     
c     put the point on the stack
c     
            ib(itop)=i1
            ib(i1)=-1
            itop=i1
            go to 210
c     
c     iact=2 - force c-points
c     
 160        nun=nun+1
            if(nun.gt.1) go to 170
            ic=ii0
            nnewpt=nnewpt+1
            icg(ic)=3
            ifg(ic)=2
            go to 100
 170        icg(i)=3
            icg(ic)=-2
            ic=i
            go to 210
 190        ifg(ii0)=-999
 200     continue
c     
c     end of test for point i
c     
 210     if(nun.gt.0) then
c     
c     C-variable ic has been forced.
c     Force all dependent/coupled variables.
c     
            iilo=iv(ip(ic))
            iihi=iv(ip(ic)+1)-1
            do 220 ii=iilo,iihi
               if(ii.eq.ic) go to 220
               if(icdep(iu(ii),iu(ic)).ne.0) icg(ii)=4
 220        continue

         endif
c     
c     reset ifg
c     
         do 290 j=j1lo,j1hi
            ifg(ja(j))=0
 290     continue
 300  continue
 1000 format('  ftest2  k=',i2,' # tested/failed/added points =',3i5)
 2000 format('    unknowns tested/failed:',3(3x,'iu=',i1,2i5))
      return
      end
c     
c=====================================================================
c     
c     the following routine tests whether the grid is coarse enough
c     
c=====================================================================
c     
      subroutine gtest(k,mmax,imin,imax,icg,ifg,
     *     ndimu,ndimp,ndima,ndimb)
c     
c---------------------------------------------------------------------
c     
c     test for coarsest grid
c     
c     Set mmax = k+1 if # variables <= 15
c     
c     Set mmax = k   if coarsening too slow ( > .9 )
c     Set mmax = k   if coarsening too slow ( > .9 )
c     
c     If mmax = k, set ifg on level k
c     
c---------------------------------------------------------------------
c     
      implicit real*8 (a-h,o-z)
c     
c     include 'params.amg'
c     
      dimension imin(25),imax(25)
      dimension icg(*)
      dimension ifg(*)
c     
c---------------------------------------------------------------------
c     
      ncpts=0
      ilo=imin(k)
      ihi=imax(k)
      nfpts=ihi-ilo+1
      do 10 i=ilo,ihi
         if(icg(i).gt.0) ncpts=ncpts+1
 10   continue
      if(ncpts.le.15)  mmax=k+1
      if(ncpts.eq.nfpts) mmax=k
      if(float(ncpts)/float(nfpts).ge..95) mmax=k
      if(mmax.gt.k) return
      ifflo=imin(k-1)
      iffhi=imax(k-1)
      do 20 iff=ifflo,iffhi
         if(icg(iff).le.0) go to 20
         ifg(icg(iff))=iff
 20   continue
      return
      end
