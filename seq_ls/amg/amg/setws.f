c
c=====================================================================
c
c     the following routines are used for defining interpolation
c
c=====================================================================
c
      subroutine setw(ierr,k,ewt,nwt,iwts,imin,imax,a,ia,ja,
     *                 iu,icg,ifg,b,ib,jb,
     *                 ipmn,ipmx,ip,iv,xp,yp)
c
c---------------------------------------------------------------------
c
c     call routines to define interpolation weights
c
c---------------------------------------------------------------------
c
      implicit real*8 (a-h,o-z)
c
      dimension imin(*),imax(*)
      dimension ia (*)
      dimension a  (*)
      dimension ja (*)
      dimension iu (*)
      dimension icg(*)
      dimension ifg(*)

      dimension ib (*)
      dimension b  (*)
      dimension jb (*)
c
      dimension ipmn(*),ipmx(*)
      dimension ip (*)
      dimension iv (*)
      dimension xp (*)
      dimension yp (*)
c
c---------------------------------------------------------------------
c
      if(iwts.eq.0) return
      go to (10,20,30,40),iwts
      ierr = 6
      return

10    call setw1(k,imin,imax,icg,b,ib,jb,
     *           ndimu,ndimp,ndima,ndimb)
      return
20    call setw2(k,ewt,nwt,imin,imax,a,ia,ja,iu,icg,ifg,b,ib,jb,
     *           ndimu,ndimp,ndima,ndimb)
      return
30    call setw3(k,ewt,nwt,imin,imax,a,ia,ja,iu,icg,ifg,b,ib,jb,
     *           ipmn,ipmx,ip,iv,xp,yp,
     *           ndimu,ndimp,ndima,ndimb)
      return
40    call setw4(ierr,k,ewt,nwt,imin,imax,a,ia,ja,iu,icg,
     *           ifg,b,ib,jb,
     *           ipmn,ipmx,ip,iv,xp,yp,
     *           ndimu,ndimp,ndima,ndimb)
      return
      end
c
c---------------------------------------------------------------------
c
      subroutine setw1(k,imin,imax,icg,b,ib,jb,
     *                 ndimu,ndimp,ndima,ndimb)
c
c---------------------------------------------------------------------
c
c     define equal interpolation weights (1 row sum)
c
c     1. C-rows are compressed to a diagonal
c
c     2. Interpolation is from all points in original B-row.
c
c---------------------------------------------------------------------
c
      implicit real*8 (a-h,o-z)

      dimension imin(*),imax(*)
      dimension icg(*)

      dimension ib (*)
      dimension jb (*)
      dimension b  (*)
c
c---------------------------------------------------------------------
c
      ilo=imin(k)
      ihi=imax(k)
      kb=ib(ilo)
c
c     equal weights are set
c
      do 20 i=ilo,ihi
c
c     C-point
c
      if(icg(i).gt.0) then
        ib(i)=kb
        b(kb)=1.e0
        jb(kb)=i
        kb=kb+1
c
c     F-point
c
      else
        ib(i)=kb
        jlo=ib(i)
        jhi=ib(i+1)-1
        if(jlo.le.jhi) then
          w=1.e0/float(jhi-jlo+1)
          do 10 j=jlo,jhi
          b(kb)=w
          jb(kb)=jb(j)
          kb=kb+1
10        continue
        endif
      endif
20    continue
      ib(ihi+1)=kb
      return
      end
c
c---------------------------------------------------------------------
c
      subroutine setw2(k,ewt,nwt,imin,imax,a,ia,ja,iu,icg,ifg,b,ib,jb,
     *                 ndimu,ndimp,ndima,ndimb)
c
c---------------------------------------------------------------------
c
c     define standard amg interpolation (no tests are performed)
c
c     1. The REAL part of the operator is used.
c
c     2. Interpolation is only within unknowns.
c
c     3. nwt - 2 digits (this is now the full nwt as defined)
c
c          1st - iwts  (calls this routine)
c          2nd - iddst = 0 - no distribution to diagonal
c                      = 1 - distribution to diagonal
c UNUSED*  3rd - ispt  = 0 - Special points ignored (eliminated)
c                      = 1 - Special points treated as F-points
c                            (for testing & distribution)
c                      = 2 - F-S connection added to diagonal.
c
c     4. No previous form for b is assumed.
c        C-rows get a diagonal entry.
c        F-rows in standard form with the computed weights.
c
c     5. Special points get empty rows here.
c
c---------------------------------------------------------------------
c
      implicit real*8 (a-h,o-z)

      dimension imin(*),imax(*)
      dimension ia (*)
      dimension a  (*)
      dimension ja (*)
      dimension iu (*)
      dimension icg(*)
      dimension ifg(*)

      dimension ib (*)
      dimension b  (*)
      dimension jb (*)
c
      dimension  iarr(10)
c
c---------------------------------------------------------------------
c
c     decompose the parameters
c
      call idec(nwt,3,ndig,iarr)
      iddst=iarr(2)
      ispt =iarr(3)
c
      ilo=imin(k)
      ihi=imax(k)
      ishift=ihi-ilo+2
c
c     set ifg to 0 for use as a marker
c
      do 10 i=ilo,ihi
      ifg(i)=0
10    continue
c
c     Loop over points
c
      kb=ib(ilo)
      do 400 i=ilo,ihi
      ib(i)=kb
c
c=>   i is a c-point. Leave diagonal entry.
c
      if(icg(i).gt.0) then
        b(kb)=1.e0
        jb(kb)=i
        kb=kb+1
        go to 400
      endif
c
c=>   i is a special point. No row.
c
      if(icg(i).eq.0) go to 400
c
c=>   i is an f-point.
c
c     Build row for i from C-points in S(i).
c     Mark such points jj by setting ifg(jj) = kb
c     (where kb is the position of jj in row i).
c
      jslo=ia(i+ishift)+1
      jshi=ia(i+ishift+1)-1
      do 30 j=jslo,jshi
      ii=ja(j)
      if(icg(ii).le.0) go to 30
      b(kb)=0.e0
      jb(kb)=ii
      ifg(ii)=kb
      kb=kb+1
30    continue
c
c     Set last entry for i. If distribution to the diagonal
c     is wanted (iddst=1), set ifg(i)=kb.
c
      b(kb)=a(ia(i))
      if(iddst.ne.0) ifg(i)=kb
c
c     sweep over all direct neighbors.
c
      jlo=ia(i)+1
      jhi=ia(i+1)-1
      do 200 j=jlo,jhi
      ii=ja(j)
      if(iu(ii).ne.iu(i)) go to 200
      if(a(j).eq.0.e0) go to 200
c
c     Marked points (C-points): load the connection directly
c
      if(ifg(ii).gt.0) then
        b(ifg(ii))=b(ifg(ii))+a(j)
        go to 200
      endif
c
c     Special points: no distribution.
c
      if(icg(ii).eq.0) go to 200
c
c     F-point: distribute the connection
c     (along "positive" connections only)
c
      s=0.e0
      jjlo=ia(ii)+1
      jjhi=ia(ii+1)-1
      do 110 jj=jjlo,jjhi
      if(ifg(ja(jj)).le.0) go to 110
c      if(a(jj)/a(jjlo).le.0.e0) go to 110
      s=s+a(jj)
110   continue
c
c     if sum of connections to c(i)=0, distribute to diagonal
c
      if(s.eq.0.e0) then
        b(kb)=b(kb)+a(j)
      else
c
c     distribute weight to interpolation (and diagonal) points
c
        w=a(j)/s
        do 150 jj=jjlo,jjhi
        if(ifg(ja(jj)).le.0) go to 150
c        if(a(jj)/a(jjlo).le.0.e0) go to 150
        b(ifg(ja(jj)))=b(ifg(ja(jj)))+a(jj)*w
150     continue
      endif
200   continue
c
c     get denominator for weight computation
c
      w=-1.e0/b(kb)
c
c     multiply by w, and compress out unwanted entries
c     also reset ifg to zero
c
      jbhi=kb-1
      kb=ib(i)
      ifg(i)=0
      do 210 j=ib(i),jbhi
      b(kb)=b(j)*w
c     if(dabs(b(kb)).le.ewt) go to 210
      ifg(jb(j))=0
      jb(kb)=jb(j)
      kb=kb+1
  210 continue
  400 continue
      ib(ihi+1)=kb
c
      return
      end
c
c---------------------------------------------------------------------
c
      subroutine setw3(k,ewt,nwt,imin,imax,a,ia,ja,iu,icg,ifg,b,ib,jb)
c
c---------------------------------------------------------------------
c
c     define iterative mg interpolation (no tests are performed)
c
c     1. Interpolation is only within unknowns.
c
c     2. nwt - 2 digits (this is now the full nwt as defined)
c
c          1st - iwts  (calls this routine)
c          2nd - nswp  = Number of sweeps to perform
c
c     3. No form for b is assumed.
c        C-rows get a diagonal entry.
c        F-rows in standard form with the computed weights.
c
c     4. Special points get empty rows here.
c
c---------------------------------------------------------------------
c
      implicit real*8 (a-h,o-z)
c
c     include 'params.amg'
c
      dimension imin(*),imax(*)
      dimension ia (*)
      dimension a  (*)
      dimension ja (*)
      dimension iu (*)
      dimension icg(*)
      dimension ifg(*)

      dimension ib (*)
      dimension b  (*)
      dimension jb (*)
c
      dimension  iarr(10)
c
c---------------------------------------------------------------------
c
c
c     decompose the parameters
c
      call idec(nwt,7,ndig,iarr)
      nswp=iarr(2)
      nout=iarr(3)
      ipos=iarr(4)
      icpt=iarr(5)
      if(nswp.lt.1) nswp=1
      ishift=imax(k)-imin(k)+2
c
c     set up proper form of b
c
      kb=ib(imin(k))
      do 20 i=imin(k),imax(k)
        ifg(i)=0
        ib(i) = kb
c
c       Test for C/F points (empty row for S (special) points)
c
c       C-point - load diagonal entry
c
        if(icg(i).gt.0) then
          b(kb) = 1.0
          jb(kb) = i
          kb = kb+1
c
c       F-point - load all strong C-connections (from A+)
c       (Note: initial weights are from STRONGx (matrix entries))
c
        elseif(icg(i).lt.0) then
          i1 = i+ishift
          jlo=ia(i1)+1
          jhi=ia(i1+1)-1
          sum = 0.0
          do 10 j=jlo,jhi
            ii=ja(j)
            if(icg(ii).gt.0) then
              b(kb)=a(j)
              sum = sum+a(j)
              jb(kb)=ii
              kb=kb+1
            endif
10        continue
cveh          if(sum.eq.0.0) then
cveh            print *,'  sum=0 - i=',i
cveh          endif
          do 15 j=ib(i),kb-1
15        b(j) = b(j)/sum
        endif
20    continue
      ib(imax(k)+1)=kb
c
c     Set outer iteration
c
      it = 0
100   it = it+1
c
c     iterate on the weights
c
      do 400 ns=1,nswp
        do 300 i=imin(k),imax(k)
          if(icg(i).ge.0) go to 300
c
c         mark the interpolation points in ifg with location in b
c         (Note: diagonal term is stored in b(kb) for distribution)
c
cc        ifg(i) = kb
cc        b(kb) = a(ia(i))
          ifg(i) = 0
          bdiag  = a(ia(i))
          jblo=ib(i)
          jbhi=ib(i+1)-1
          do 110 j=jblo,jbhi
            ifg(jb(j))=j
            b(j)=0.d0
110       continue
c
c         sweep over direct neighbors
c
          jlo=ia(i)+1
          jhi=ia(i+1)-1
          do 200 j=jlo,jhi
            ii=ja(j)
            if(iu(ii).ne.iu(i)) go to 200
c
c           Strong C-neighbor - add entry to weight
c
            if(ifg(ii).gt.0) then
              b(ifg(ii))=b(ifg(ii))+a(j)
c
c           F connection - perform distribution (using b)
c
ccjwr 11/1/96  elseif(icg(i).ne.0) then
            elseif(icg(ii).lt.0) then
              sum = 0.0
              jjlo=ib(ii)
              jjhi=ib(ii+1)-1
              do 120 jj=jjlo,jjhi
                if(ipos.gt.0.and.b(jj).le.0.0) go to 120
                if(ifg(jb(jj)).gt.0) sum=sum+b(jj)
120           continue
c
              if(sum.eq.0.0) then
                bdiag=bdiag+a(j)
              else
                w=a(j)/sum
                do 130 jj=jjlo,jjhi
                  if(ipos.gt.0.and.b(jj).le.0.0) go to 130
                  if(ifg(jb(jj)).gt.0)
     *              b(ifg(jb(jj)))=b(ifg(jb(jj)))+b(jj)*w
130             continue
              endif
c
c           C connection - perform distribution (using a)
c
            elseif(icg(ii).gt.0) then
              if(icpt.eq.0) then
                bdiag=bdiag+a(j)
              else
                sum = 0.0
                jjlo=ia(ii)+1
                jjhi=ia(ii+1)-1
                do 140 jj=jjlo,jjhi
cveh                  if(a(ia(ii)).eq.0.0) then
cveh                    print *,'  a(ia(i))=0.0 - i=',i
cveh                  endif
                  if(ipos.gt.0.and.a(jj)/a(ia(ii)).ge.0.0) go to 140
                  if(ifg(ja(jj)).gt.0) sum=sum+a(jj)
                  if(icpt.eq.2) then
                    if(ja(jj).eq.i) sum=sum+a(jj)
                  endif
140             continue
c
                if(sum.eq.0.0) then
                  bdiag=bdiag+a(j)
                else
                  w=a(j)/sum
                  do 150 jj=jjlo,jjhi
cveh                    if(a(ia(ii)).eq.0.0) then
cveh                      print *,'  a(ia(i))=0.0 - i=',i
cveh                    endif
                    if(ipos.gt.0.and.a(jj)/a(ia(ii)).ge.0.0) go to 150
                    if(ifg(ja(jj)).gt.0)
     *                b(ifg(ja(jj)))=b(ifg(ja(jj)))+a(jj)*w
                    if(icpt.eq.2) then
                      if(ja(jj).eq.i) bdiag=bdiag+a(jj)*w
                    endif
150               continue
                endif
              endif
            endif
200       continue
c
c         Compute weights for point i (and zero out ifg)
c
cveh          if(bdiag.eq.0.0) then
cveh            print *,' bdiag=0 - i=',i
cveh          endif
          bdiag=-1.d0/bdiag
          do 210 j=jblo,jbhi
            b(j)=b(j)*bdiag
            ifg(jb(j))=0
210       continue
c
300     continue
400   continue
c
c     eliminate negative weights and recompute
c
      if(it.gt.nout) return
c
      kb = ib(imin(k))
      do 420 i=imin(k),imax(k)
        if(icg(i).gt.0) then
          ib(i)=kb
          b(kb)=1.0
          jb(kb)=i
          kb=kb+1
        elseif(icg(i).eq.0) then
          ib(i)=kb
        else
          jlo=ib(i)
          ib(i)=kb
          do 410 j=jlo,ib(i+1)-1
            if(b(j).gt.0.01) then
              b(kb)=b(j)
              jb(kb)=jb(j)
              kb=kb+1
            endif
410       continue
cveh          if(kb.eq.ib(i)) then
cveh            print *,' empty row - i=',i
cveh          endif
        endif
420   continue
      ib(imax(k)+1)=kb
      go to 100
      end
c---------------------------------------------------------------------
c
      subroutine setw4(ierr,k,ewt,nwt,imin,imax,a,ia,ja,iu,icg,
     *                 ifg,b,ib,jb,
     *                 ipmn,ipmx,ip,iv,xp,yp,
     *                 ndimu,ndimp,ndima,ndimb)
c
c---------------------------------------------------------------------
c
c     Define POINT-TYPE interpolation (no tests are performed)
c
c     1. Defines "point" interpolation as much as possible. True
c     point-coarsening is not required (actually, coarsening is
c     arbitrary, although results may be unpredictable).
c
c     2. All F-variables at point ipt interpolate from:
c        a. All C-variables at point ipt.
c        b. All C-variables at points jpt such that j in C_i
c
c     2. Distribution is only within unknowns.
c
c     3. nwt - 2 digits (this is now the full nwt as defined)
c
c          1st - iwts  (calls this routine)
c          2nd - iddst = 0 - no distribution to diagonal
c                      = 1 - distribution to diagonal
c          3rd - ispt  = 0 - Special points treated as F-points
c                            (for testing & distribution)
c                      = 1 - F-S connection added to diagonal.
c
c     4. No previous form for b is assumed.
c        C-rows get a diagonal entry.
c        F-rows in standard form with the computed weights.
c
c     5. Special points get empty rows here.
c
c---------------------------------------------------------------------
c
      implicit real*8 (a-h,o-z)

      dimension imin(*),imax(*)
      dimension ia (*)
      dimension a  (*)
      dimension ja (*)
      dimension iu (*)
      dimension icg(*)
      dimension ifg(*)

      dimension ib (*)
      dimension b  (*)
      dimension jb (*)
c
      dimension ipmn(*),ipmx(*)
      dimension ip (*)
      dimension iv (*)
      dimension xp (*)
      dimension yp (*)
c
c---------------------------------------------------------------------
c
      dimension  iarr(10)
      dimension  bb(10,100),jbb(100)
      dimension  ipp(20)
c
c---------------------------------------------------------------------
c
c     decompose the parameters
c
      call idec(nwt,3,ndig,iarr)
      iddst=iarr(2)
      ispt =iarr(3)
c
      ilo=imin(k)
      ihi=imax(k)
      ishift=ihi-ilo+2
c
c     set ifg to 0 for use as a marker
c
      do 10 i=ilo,ihi
      ifg(i)=0
10    continue
      kb=ib(ilo)
c
c     Loop over points
c
      do 400 ipt=ipmn(k),ipmx(k)
c
c     Check point for F, C and S-variables
c
      ibf=0
      ibc=0
      itop=0
      do 20 i=iv(ipt),iv(ipt+1)-1
      if(icg(i).le.0) then
        ibf=ibf+1
        ifg(i)=ibf
        jbb(ibf)=i
        itop=1
        ipp(1)=ipt
      elseif(icg(i).gt.0) then
        ibc=ibc+1
      endif
20    continue
      npi = iv(ipt+1)-iv(ipt)
c
c     All C or S-variables. Set B-rows
c
      if(ibf.eq.0) then
        do 30 i=iv(ipt),iv(ipt+1)-1
          ib(i)=kb
          if(icg(i).gt.0) then
            b(kb)=1.0
            jb(kb)=i
            kb=kb+1
          endif
30      continue
        go to 400
      endif
c
c     Mixed C & F point (possibly)
c
c     Load C (interpolation)-variables on list
c
      ibc=ibf
c
      do 60 i=iv(ipt),iv(ipt+1)-1
      if(icg(i).gt.0) then
        ibc=ibc+1
        ifg(i)=ibc
        jbb(ibc)=i
      elseif(icg(i).lt.0) then
        irow=ifg(i)
        jlo=ia(i+ishift)+1
        jhi=ia(i+ishift+1)-1
        do 50 j=jlo,jhi
          ii=ja(j)
          if(ifg(ii).ne.0) go to 50
          if(icg(ii).le.0) go to 50
          itop=itop+1
          ipp(itop)=ip(ii)
          iiilo=iv(ip(ii))
          iiihi=iv(ip(ii)+1)-1
          do 40 iii=iiilo,iiihi
            if(icg(iii).le.0) then
              ifg(iii)=-1
              go to 40
            else
              ibc=ibc+1
              ifg(iii)=ibc
              jbb(ibc)=iii
            endif
40        continue
50      continue
      endif
60    continue
c
      do 70 ir=1,ibf
      do 70 ic=1,ibc
70    bb(ir,ic)=0.0
c
c     Set bound for distribution (to center or not)
c
      if(iddst.eq.0) then
        idlo=ibf
      else
        idlo=0
      endif
c
c     Load row entries/perform distribution
c
      do 200 ir=1,ibf
c     i=ifg(ir)
      i=jbb(ir)
      do 190 j=ia(i),ia(i+1)-1
      ii=ja(j)
      if(ifg(ii).gt.0) then
        bb(ir,ifg(ii))=bb(ir,ifg(ii))+a(j)
      else
        s=0.0
        iud=iu(ii)
        do 150 jj=ia(ii)+1,ia(ii+1)-1
        iii=ja(jj)
        if(iu(iii).ne.iud) go to 150
        if(ifg(iii).le.idlo) go to 150
        s=s+a(jj)
150     continue
c
c       Perform distribution
c
        if(s.eq.0.0) then
          bb(ir,ir)=bb(ir,ir)+a(j)
          go to 200
        endif
c
        w = a(j)/s
        do 160 jj=ia(ii)+1,ia(ii+1)-1
        iii=ja(jj)
        if(iu(iii).ne.iud) go to 160
        if(ifg(iii).le.idlo) go to 160
        bb(ir,ifg(iii))=bb(ir,ifg(iii))+a(jj)*w
160     continue

      endif
190   continue
200   continue
c
c     Eliminate small entries
c
c     (to be added)
c
c     invert diagonal block
c
      call iinvert(ierr,bb,ibf,10)
      if (ierr .ne. 0) return
c
c     Load B-rows for variables at point ipt
c
      do 230 i=iv(ipt),iv(ipt+1)-1
      ib(i)=kb
      if(icg(i).gt.0) then
        b(kb)=1.0
        jb(kb)=i
        kb=kb+1
      elseif(icg(i).lt.0) then
        ir=ifg(i)
        do 220 ic=ibf+1,ibc
          w=0.0
          do 210 j=1,ibf
            w=w+bb(ir,j)*bb(j,ic)
210       continue
        if(w.eq.0.0) go to 220
        b(kb)=w
        jb(kb)=jbb(ic)
        kb=kb+1
220     continue
      endif
230   continue
c
c     reset ifg to zero (for all points on list in ipp
c
      do 260 it=1,itop
      do 250 i=iv(ipp(it)),iv(ipp(it)+1)-1
      ifg(i)=0
250   continue
260   continue
c
400   continue

      ib(imax(k)+1)=kb
c
      return
      end
