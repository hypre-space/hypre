c
c=====================================================================
c
c     the following routines are used to define strong connections
c
c=====================================================================
c
      subroutine strcnc(k,isort,estr,istr,imin,imax,a,ia,ja,iu,
     *                  ndimu,ndimp,ndima,ndimb)
c
c---------------------------------------------------------------------
c
c     define & load strong connections (calls rowsort and strong1)
c
c     NOTES:
c
c     1. These are stored in 'a' beyond level k. the information for
c     the strong connections for variable i is stored in the row
c     i+(imax(k)-imin(k)+1). ia and ja are used in the usual way,
c     with ja giving the shifted variable numbers.
c
c     2. The definition of strong connections is determined by the
c     parameters estr and nstr.
c
c       nstr consists of 3 digits: istr, isort, and npth
c
c         istr  = 1 -- direct strong connections are used. these
c                      are defined by estr and isort.
c               = 2 -- long range connections are used. these are
c                      based on paths along direct strong connections
c                      of length 2. only variables with more than
c                      npth paths are considered strong.
c
c         isort = 0 -- strength of connection  is based on
c                      absolute value of the connection.
c               = 1 -- connections with the same sign as the
c                      diagonal are considered weak.
c
c         npth      -- number of paths of length 2 required
c                      for strong connections when istr=2.
c
c       estr defines strong connections in the usual sense,
c       together with nstr.
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
c
c---------------------------------------------------------------------
c
c     perform row sort
c
      call rowsort(k,isort,estr,imin,imax,a,ia,ja,iu,
     *             ndimu,ndimp,ndima,ndimb)
c
c     determine strong connections & load
c
      if(istr.eq.1) then
        call strong1(k,imin,imax,a,ia,ja,
     *               ndimu,ndimp,ndima,ndimb)
      else
c       call strong2(k,imin,imax,a,ia,ja)
      endif
      return
      end
c
      subroutine rowsort(k,isort,eps,imin,imax,a,ia,ja,iu,
     *                   ndimu,ndimp,ndima,ndimb)
c
c---------------------------------------------------------------------
c
c     Perform partial row sort.
c
c     isort=0 --> sort is by absolute value
c
c     isort=1 --> straight sort is used based on sign of diagonal
c
c     includes test for special points, but over ALL connections.
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
c
c---------------------------------------------------------------------
c
      if(eps.gt.1.e0) return
      ilo=imin(k)
      ihi=imax(k)
      if(isort.ne.0) go to 100
c
c     sort by absolute value
c
      do 60 i=ilo,ihi
      jlo=ia(i)+1
      jhi=ia(i+1)-1
      if(eps.le.0.e0) go to 50
      if(jhi.lt.jlo) go to 50
      amx=-1.e38
c     rs=a(ia(i))
      do 10 j=jlo,jhi
      if(iu(ja(j)).ne.iu(i)) go to 10
      if(abs(a(j)).le.amx) go to 10
      amx=a(j)
      jmx=j
10    continue
      if(amx.gt.0.e0) go to 20
      ja(ia(i))=ia(i)
      go to 60
c
c     define cutoff for strong connections
c
20    ast=eps*amx
      amx=a(jmx)
      imx=ja(jmx)
      a(jmx)=a(jlo)
      ja(jmx)=ja(jlo)
      a(jlo)=amx
      ja(jlo)=imx
      jhi=jhi+1
30    jhi=jhi-1
      if(jlo.ge.jhi) go to 50
      if(abs(a(jhi)).lt.ast.or.iu(ja(jhi)).ne.iu(i)) go to 30
c
c     jhi is a strong connection
c
40    jlo=jlo+1
      if(jlo.ge.jhi) go to 50
      if(abs(a(jlo)).ge.ast.and.iu(ja(jlo)).eq.iu(i)) go to 40
c
c     jlo is a weak connection
c     interchange jhi and jlo
c
      atmp=a(jhi)
      itmp=ja(jhi)
      a(jhi)=a(jlo)
      ja(jhi)=ja(jlo)
      a(jlo)=atmp
      ja(jlo)=itmp
      go to 30
50    ja(ia(i))=jhi
60    continue
      return
c
c     sort by sign of diagonal
c
100   do 190 i=ilo,ihi
      jlo=ia(i)+1
      jhi=ia(i+1)-1
      if(jhi.lt.jlo) go to 180
      if(a(ia(i)).gt.0.e0) go to 140
c
c==>  negative diagonal
c
      ja(ia(i))=ia(i)
c
c     find largest suitable off-diagonal
c     also compute row-sum for "special point" test
c
      rs=0.e0
ccjwr >>>>> test  1/24/95
c     rsmin=.1d0*dabs(a(ia(i)))
      rsmin=0.e0
ccjwr <<<<<
c
      amx=-1.e38
      jmx=0
c
      do 110 j=jlo,jhi
      rs=rs+abs(a(j))
      if(iu(ja(j)).ne.iu(i)) go to 110
      if(a(j).le.amx) go to 110
      amx=a(j)
      jmx=j
110   continue
c
c     test for special points (diagonally dominant)
c
      if(rs.lt.rsmin) go to 190
c
c     test for no strong connections
c
      if(jmx.eq.0.or.amx.le.0.e0) go to 190
c
c     set strong connection cutoff and proceed with sort
c
      ast=eps*amx
      imx=ja(jmx)
      a(jmx)=a(jlo)
      ja(jmx)=ja(jlo)
      a(jlo)=amx
      ja(jlo)=imx
      jhi=jhi+1
120   jhi=jhi-1
      if(jlo.ge.jhi) go to 180
      if(a(jhi).lt.ast.or.iu(ja(jhi)).ne.iu(i)) go to 120
c
c     jhi is a strong connection
c
130   jlo=jlo+1
      if(jlo.ge.jhi) go to 180
      if(a(jlo).ge.ast.and.iu(ja(jlo)).eq.iu(i)) go to 130
c
c     jlo is a weak connection
c     interchange jhi and jlo
c
      atmp=a(jhi)
      itmp=ja(jhi)
      a(jhi)=a(jlo)
      ja(jhi)=ja(jlo)
      a(jlo)=atmp
      ja(jlo)=itmp
      go to 120
c
c==>  positive diagonal
c
140   ja(ia(i))=ia(i)
c
c     find largest suitable off-diagonal
c     also compute row-sum for "special point" test
c
      rs=0.e0
ccjwr >>>>> test  1/24/95
      rsmin=.1e0*abs(a(ia(i)))
c     rsmin=0.d0
ccjwr <<<<<
c
      amx=1.e38
      jmx=0
c
      do 150 j=jlo,jhi
      rs=rs+abs(a(j))
      if(iu(ja(j)).ne.iu(i)) go to 150
      if(a(j).ge.amx) go to 150
      amx=a(j)
      jmx=j
150   continue
c
c     test for special points (diagonally dominant)
c
      if(rs.lt.rsmin) go to 190
c
c     test for no strong connections
c
ccjwr >>>>> test 1/24/95
      if(jmx.eq.0.or.amx.ge.0.e0) go to 190
c     if(amx.ge.0.d0) go to 190
c
c     set strong connection cutoff and proceed with sort
c
      ast=eps*amx
      imx=ja(jmx)
      a(jmx)=a(jlo)
      ja(jmx)=ja(jlo)
      a(jlo)=amx
      ja(jlo)=imx
      jhi=jhi+1
160   jhi=jhi-1
      if(jlo.ge.jhi) go to 180
      if(a(jhi).gt.ast.or.iu(ja(jhi)).ne.iu(i)) go to 160
c
c     jhi is a strong connection
c
170   jlo=jlo+1
      if(jlo.ge.jhi) go to 180
      if(a(jlo).le.ast.and.iu(ja(jlo)).eq.iu(i)) go to 170
c
c     jlo is a weak connection
c     interchange jhi and jlo
c
      atmp=a(jhi)
      itmp=ja(jhi)
      a(jhi)=a(jlo)
      ja(jhi)=ja(jlo)
      a(jlo)=atmp
      ja(jlo)=itmp
      go to 160
180   ja(ia(i))=jhi
190   continue
      return
      end
c
      subroutine strong1(k,imin,imax,a,ia,ja,
     *                   ndimu,ndimp,ndima,ndimb)
c
c---------------------------------------------------------------------
c
c     define direct strong connections.
c
c     loads connections into shifted a rows
c
c     this routine defines only direct strong connections as
c     the strong connections for coloring and interpolation.
c
c     direct strong connections are assumed to be stored in
c     the locations a(ia(i)+1),...,a(ja(ia(i))). the new
c     strong connections for variable i are stored in the row
c     i+(imax(k)-imin(k)+1). ja(ia(i)) is reset to i before
c     returning.
c
c     the unshifted point numbers generally contain a 0,
c     while the shifted numbers contain a 1. ja on the shifted
c     matrix contains the unshifted column numbers. ja of the
c     diagonal points to the last direct strong connection in
c     the row. the entries in a for the direct strong
c     connections are the original matrix entries.
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
c
c---------------------------------------------------------------------
c
c     set up working space.
c
      i0lo=imin(k)
      i0hi=imax(k)
c
      ka=ia(i0hi+1)
      ishift=i0hi-i0lo+2
      i1lo=i0lo+ishift
      i1hi=i0hi+ishift
c===> check ndimu for overflow
      if(i1hi.gt.ndimu) go to 9903
      do 20 i1=i1lo,i1hi
      i0=i1-ishift
c
c     set diagonal entry
c
      a(ka)=0.e0
      ja(ka)=ka
      ia(i1)=ka
      ka=ka+1
c
c     add strong off-diagonals
c
      j0lo=ia(i0)+1
      j0hi=ja(ia(i0))
      if(j0lo.gt.j0hi) go to 20
      do 10 j0=j0lo,j0hi
      ii0=ja(j0)
      a(ka)=a(j0)
      ja(ka)=ii0
      ka=ka+1
10    continue
      ja(ia(i1))=ka-1
20    continue
      ia(i1hi+1)=ka
      if(ka.gt.ndima) go to 9901
c
c     reset ja(ia(i)) to i
c
      do 30 i=i0lo,i0hi
      ja(ia(i))=i
30    continue
c
      return
c
c===> error messages
c
 9901 write(6,9910) ndima
        stop
 9903 write(6,9930) ndimu
        stop
c
 9910 format(' ### error in strong: ndima too small ###',i5)
 9930 format(' ### error in strong: ndimu too small ###',i5)
      end
