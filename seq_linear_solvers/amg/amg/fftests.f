c
CCJWR>>>>> TEST ROUTINES
c
c=====================================================================
c
c     routines to test the final f-point choice
c
c=====================================================================
c
      subroutine fftest(k,ewt,icdep,nun,imin,imax,a,
     *                  ia,ja,iu,icg,ifg,ib,
     *                  ndimu,ndimp,ndima,ndimb)
c
c---------------------------------------------------------------------
c
c     f-point tests
c
c       itest = 0 - no test
c       itest = 1 - test with sum of matrix connections to Ci
c       itest = 2 - test with number of strong connections to Ci
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
      dimension icg(*)
      dimension ifg(*)

      dimension ib (*)

      dimension icdep(10,10)

      dimension nc(5),ns(5),nf(5),nff1(5,3),nff2(5)
c
c---------------------------------------------------------------------
c
      call fftest1(k,ewt,icdep,nun,imin,imax,a,ia,ja,iu,icg,ifg,ib,
     *             nc,ns,nf,nff1,ndimu,ndimp,ndima,ndimb)
      call fftest2(k,icdep,nun,imin,imax,a,ia,ja,iu,icg,ifg,ib,
     *             nc,ns,nf,nff2,ndimu,ndimp,ndima,ndimb)
c
      write(6,1000) k
      write(6,2000)
      do 400 iun=1,nun
      write(6,3000) iun,(icdep(iun,j),j=1,nun),
     *       nc(iun),ns(iun),nf(iun),(nff1(iun,l),l=1,3),nff2(iun)
400   continue

1000  format(/'  K =',i2,'  final C/S/F choice with FTEST1 (3 tests)',
     *        ' and FTEST2 (1 test)')
2000  format(/'   iu     icdep      C     S     F',
     *        '    t1.1  t1.2  t1.3  t2.1')
3000  format(4x,i1,4x,3i2,1x,3(1x,i5),2x,4(2x,i4))
      write(6,'(1x)')
      return
      end
c
      subroutine fftest1(k,ewt,icdep,nun,imin,imax,
     *                   a,ia,ja,iu,icg,ifg,ib,nc,ns,nf,nff,
     *                   ndimu,ndimp,ndima,ndimb)
c
c---------------------------------------------------------------------
c
c     usual (ewt) version of f-point test
c
c     this assumes that direct strong connections for i
c     are stored in ia(i+ishift)+1,...,ja(ia(i+ishift))
c     and indirect strong connections are stored in
c     ja(ia(i+ishift)),...,ia(i+ishift+1)-1.
c
c     Let S(i) = C(i) + F(i). Then for j in F(i) we want
c     j to depend on C(i) in a certain way.
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
      dimension icg(*)
      dimension ifg(*)

      dimension ib (*)

      dimension icdep(10,10)

      dimension nc(5),ns(5),nf(5),nff(5,3)
c
c---------------------------------------------------------------------
c
      ilo=imin(k)
      ihi=imax(k)
      ishift=ihi-ilo+2
c
c     initialize ifg(i)=0 for all points
c
      do 10 i=ilo,ihi
      ifg(i)=0
   10 continue

      do 12 iiu=1,3
      nc(iiu)=0
      ns(iiu)=0
      nf(iiu)=0
      do 11 itest=1,3
      nff(iiu,itest)=0
11    continue
12    continue
c
c     (i) test all f-points (icg(i) < 0)
c
      do 300 i=ilo,ihi
c
c     test/count only independent or coupled points
c
c     if(icdep(iu(i),iu(i)).eq.0) go to 300
c
      if(icg(i).gt.0) then
        nc(iu(i))=nc(iu(i))+1
        go to 300
      elseif(icg(i).eq.0) then
        ns(iu(i))=ns(iu(i))+1
        go to 300
      endif
      nf(iu(i))=nf(iu(i))+1

      i1=i+ishift
      j1lo=ia(i1)+1
c===> upper bound in row for direct strong connections
      j1dhi=ja(ia(i1))
c===> upper bound in row for all strong connections
      j1hi=ia(i1+1)-1
c
c     mark points in s(i) as follows:
c
c        if ifg(jj) = 1 - jj is a special point
c                   = 2 - jj is a c-point
c
c     if i has no neighbors, there are no interpolation weights
c
      if(j1lo.gt.j1hi) go to 300

      do 20 j1=j1lo,j1hi
      ii=ja(j1)
      if(icg(ii).eq.0) ifg(ii)=1
      if(icg(ii).gt.0) ifg(ii)=2
   20 continue
  100 continue
c
c     (j) sweep over test points (i.e., ii0 in F(i))
c
      ipass1=1
      ipass2=1
      ipass3=1
      do 200 j1=j1lo,j1dhi
      ii0=ja(j1)
      if(ifg(ii0).gt.0) go to 200
c
c     compute measure for dependence (point is covered if...)
c
      ewtr=ewt*abs(a(j1)/a(j1lo))
c
      s=0.e0
      ncn=0
c
c     (k) sweep over points in N(ii0)
c
c     compute dependence of ii0 on C(i)
c
c     Let Dij = k in C(i) int N(j) with:
c
ccjwr >>>>>  1/24/95 test (don't count connections to S pts)
c       1. k a C or S point, and
ccjwr <<<<<
c       2. ajk is a "positive" connection (i.e., strong sign)
c
c     Let:
c
c     s = sum ajk
c         Dij
c
c     n = |{k in Dij: ajk/ajkmax > ewt}|
c
      jjlo=ia(ii0)+1
      jjhi=ia(ii0+1)-1
      if(jjlo.gt.jjhi) go to 120
      do 110 jj=jjlo,jjhi
c
c     check for k in G-C(i)-S(i)
c
ccjwr >>>>>  1/24/95 test
c     if(ifg(ja(jj)).le.0) go to 110
      if(ifg(ja(jj)).le.1) go to 110
ccjwr <<<<<
c
c     check for k a "negative" connection
c
      if(a(jj)/a(jjlo).le.0.e0) go to 110
c
c     test to see whether each is covered
c     by a total weight defined by eps
c
      s=s+a(jj)
      if(a(jj)/a(jjlo).gt.ewt) ncn=ncn+1
  110 continue
c
c     Now test:
c
c     (1) if icgd = 1 - n > 0 => j covered
c     (2) if icgd = 1 - If icgr = 1 - s/ajkmax > ewt
c     (3)               If icgr = 2 - s/ajkmax > ewt*aij/aijmax
c
c     What does ifg(ii0).le.-999 mean? Forced C-point in ftest1
c
120   continue
c
c120   if(ifg(ii0).le.-999) go to 190
c
c     test 1
c
      if(ncn.eq.0) ipass1=0
c
c     test 2
c
      if(abs(s/a(jjlo)).lt.ewt) ipass2=0
c
c     test 3
c
      if(abs(s/a(jjlo)).lt.ewtr) ipass3=0
c
  200 continue
      if(ipass1.eq.0) nff(iu(i),1)=nff(iu(i),1)+1
      if(ipass2.eq.0) nff(iu(i),2)=nff(iu(i),2)+1
      if(ipass3.eq.0) nff(iu(i),3)=nff(iu(i),3)+1
c
c     reset ifg
c
      do 220 j=j1lo,j1hi
      ifg(ja(j))=0
  220 continue
  300 continue
c
      return
      end
c
      subroutine fftest2(k,icdep,nun,imin,imax,
     *                   a,ia,ja,iu,icg,ifg,ib,nc,ns,nf,nff,
     *                   ndimu,ndimp,ndima,ndimb)
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
      dimension icg(*)
      dimension ifg(*)

      dimension ib (*)
c
      dimension icdep(10,10)

      dimension nc(5),ns(5),nf(5),nff(5)
c
c---------------------------------------------------------------------
c
      ilo=imin(k)
      ihi=imax(k)
      ishift=ihi-ilo+2

      do 12 iiu=1,3
      nc(iiu)=0
      ns(iiu)=0
      nf(iiu)=0
      nff(iiu)=0
12    continue
c
c     set ifg=0
c
      do 10 i=ilo,ihi
      ifg(i)=0
   10 continue
c
      do 300 i=ilo,ihi
c
c     test/count only independent or coupled points
c
c     if(icdep(iu(i),iu(i)).eq.0) go to 300

      if(icg(i).gt.0) then
        nc(iu(i))=nc(iu(i))+1
        go to 300
      elseif(icg(i).eq.0) then
        ns(iu(i))=ns(iu(i))+1
        go to 300
      endif
      nf(iu(i))=nf(iu(i))+1

      i1=i+ishift
      j1lo=ia(i1)+1
c===> upper bound in row for direct strong connections
      j1dhi=ja(ia(i1))
c===> upper bound in row for all strong connections
      j1hi=ia(i1+1)-1
c
c     mark points in the s(i) as follows:
c
c        if ifg(jj) = 1   jj is a special point
c                   = 2   jj is a c-point
c
c     if i has no neighbors, there are no interpolation weights
c
      if(j1lo.gt.j1hi) go to 300
      do 20 j1=j1lo,j1hi
      ii=ja(j1)
      if(icg(ii).gt.0) ifg(ii)=2
      if(icg(ii).eq.0) ifg(ii)=1
   20 continue
  100 continue
c
c     sweep over directly strongly connected f points
c     test to see whether each is connected to s(i)
c
c     if the f-point has no strong connections, define it
c     to be covered, but do not distribute the weight.
c
      ipass=1
      do 200 j1=j1lo,j1dhi
      ii0=ja(j1)
      if(ifg(ii0).gt.0) go to 200
      ncn=0
      jj1lo=ia(ii0+ishift)+1
      jj1dhi=ja(ia(ii0+ishift))
      if(jj1lo.gt.jj1dhi) go to 120
      do 110 jj1=jj1lo,jj1dhi
      if(ifg(ja(jj1)).gt.0) ncn=ncn+1
110   continue
120   continue
c
c     test 1
c
      if(ncn.eq.0) ipass=0

  200 continue

      if(ipass.eq.0) nff(iu(i))=nff(iu(i))+1
c
c     reset ifg
c
      do 220 j=j1lo,j1hi
      ifg(ja(j))=0
  220 continue
  300 continue
c
      return
      end
