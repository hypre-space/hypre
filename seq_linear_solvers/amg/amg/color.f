c     
      subroutine color(k,ncolor,jval0,jvalmx,icdep,
     *     imin,imax,ia,ja,iu,ip,icg,ifg,ib,jb,iv,
     *     ndimu,ndimp,ndima,ndimb)
c     
c---------------------------------------------------------------------
c     
c     Perform the variable coloring.
c     
c     NOTES:
c     
c     0. NOTE: 1/25/95 - In an effort to avoid scrambling of the
c     stacks/lists, ifg is being used explicitly as both a value
c     indicator and a flag indicating when a point is in the lists.
c     ifg(i) = 0 IFF i is NOT in the lists.
c     
c     A point is assigned ifg=0 in COLOR if it is put into C or F,
c     and is removed from the lists.
c     
c     Special points coupled to other C or F points should also
c     become C or F points.
c     
c     1. Older version (not the updated version meant to ensure
c     one-pass coloring). Should be more efficient.
c     
c     2. Initial weight is |st(i)|. For undecided points, weight is
c     |st(i)| + |st(i) intersect C|
c     
c     3. The grid is colored. ncolor is -1 on first call to color.
c     On each pass, ncolor is decremented by 1, so that ncolor = -2
c     for the first pass, -3 on the second, etc. On each pass,
c     variables are colored by icg = -ncolor => c, icg = ncolor => f.
c     
c     4. On the first pass, there is a test for "special" variables.
c     These have no strong connections, and are treated as hybrid
c     points. They are not taken to the coarse grid, but are implicitly
c     used in the computation of interpolation weights. For special
c     points, no interpolation weights are defined, and icg=0.
c     
c     5. Parameters isub and itob are set internally.
c     
c     isub = 1 - subtracts 1 from weight for undecided points
c     on which a c-point strongly depends.
c     
c     itob = 0 - next c-point taken from right of list header
c     = 1 - next c-point taken from left of list header
c     
c     6. Includes the vector icdep for flexible coarsening.
c     See notes in SETCW and SETDEP.
c     
c---------------------------------------------------------------------
c     
      implicit real*8 (a-h,o-z)
c     
c     include 'params.amg'
c     
      dimension imin(*),imax(*)
      dimension ia (*)
      dimension ja (*)
      dimension iu (*)
      dimension ip (*)
      dimension icg(*)
      dimension ifg(*)

      dimension iv (*)

      dimension ib (*)
      dimension jb (*)
c     
      dimension icdep(10,10)
c     
      dimension ncu(10),nfu(10),nru(10)
c     
c---------------------------------------------------------------------
c     
c     print *,' color  - k=',k
c     set parameters isub and itob internally
c     
c     cjwr 4/12/96      isub=0
      isub=1
      itob=0
c     
c     initialize
c     
      ncu(1)=0
      ncu(2)=0
      ncu(3)=0
      nfu(1)=0
      nfu(2)=0
      nfu(3)=0
      nru(1)=0
      nru(2)=0
      nru(3)=0

      ilo=imin(k)
      ihi=imax(k)
      ishift=ihi-ilo+2
      nncpts=0
      jcnbhi=0
      jvalxx=jvalmx+1
      nspts=0
c     
      if(ncolor.eq.-1) then
c     
c==   >  set stacks/lists for initial coloring (ncolor=-1)
c     
c     initialize list headers (set lists to empty)
c     
         do 10 j=jval0,jvalxx
            icg(j)=j
            ifg(j)=j
 10      continue
c     
c     loop over all variables
c     
         do 20 i=ihi,ilo,-1
c     
c     check for fully dependent unknown
c     
            iiu=iu(i)
            if(icdep(iiu,iiu).eq.0) go to 20
c     
c     check for special points. set icg=0 & do not put on lists
c     
            i1=i+ishift
            if(ia(i1+1).le.ia(i1)+1) then
               nspts=nspts+1
               icg(i)=0
               ifg(i)=0
               go to 20
            endif
c     
c     initialize recoloring list
c     
            ib(i1)=0
c     
c     put variable on appropriate list
c     
            jv=ifg(i)
            if(jv.lt.jval0.or.jv.gt.jvalxx) stop 'COLOR - out of range'
            icg(i1)=icg(jv)
            ifg(i1)=jv
            icg(jv)=i1
            ifg(icg(i1))=i1
            if(jv.gt.jcnbhi) jcnbhi=jv
 20      continue
      else
c     
c==   >  set lists for recoloring (ncolor < -1).
c     
c     go through list of failed variables
c     zero out ib pointers (initialize for reset stack)
c     
         i1=ihi+ishift+1
 30      i1p=i1
         i1=ib(i1)
         ib(i1p)=0
c     
c     put variable on appropriate list
c     
         if(i1.gt.0) then
            i0=i1-ishift
            jv=ifg(i0)
            if(jv.lt.jval0.or.jv.gt.jvalxx)
     *           stop 'COLOR - jv out of range'
            icg(i1)=icg(jv)
            ifg(i1)=jv
            icg(jv)=i1
            ifg(icg(i1))=i1
            if(jv.gt.jcnbhi) jcnbhi=jv
            go to 30
         endif
      endif
c     
c==   >  decrement ncolor and initialize reset stack
c     
      ncolor=ncolor-1
      itop=-1
c     
c     find highest value nonempty list (jcnbhi)
c     
 100  if(jcnbhi.le.jval0) go to 200
      if(itob.eq.0) iic1=icg(jcnbhi)
      if(itob.ne.0) iic1=ifg(jcnbhi)
c     
c     test for empty list
c     
      if(iic1.gt.jval0) then
         jcnbhi=jcnbhi-1
         go to 100
      endif
c     
c     c-variable found
c     
      iic0=iic1-ishift
      if(ifg(iic0).lt.jval0.or.ifg(iic0).gt.jvalxx) then
         print *,'STOP - ifg out of bounds for chosen C-point ',iic0
         print *,'iic0,ifg,icg  = ',iic0,ifg(iic0),icg(iic0)
         print *,'jval0/mx/xx   = ',jval0,jvalmx,jvalxx
         stop
      endif
      ncu(iu(iic0))=ncu(iu(iic0))+1
c     
c     check for coupling/dependence.
c     no coupling/dependence - single pass loop
c     coupling/dependence    - loop over all variables at iip.
c     
      iu1 =iu(iic0)
      iip =ip(iic0)
      if(icdep(iu1,iu1).le.1) then
         ic0lo=iic0
         ic0hi=iic0
      else
         ic0lo=iv(iip)
         ic0hi=iv(iip+1)-1
      endif
c     
      do 170 ic0=ic0lo,ic0hi
         ic1=ic0+ishift
c     
c     check for ic0 independent of iic0
c     
         iu2 = iu(ic0)
         if(icdep(iu2,iu1).eq.0) go to 170
c     
c     =>   ic0 a dependent/coupled point - put in C (if not there already)
c     
         if(icg(ic0).ge.1) go to 170
c     
         icg(ic0)=-ncolor
         nncpts=nncpts+1
c     
c     remove from lists
c     
         if(ifg(ic0).ne.0) then
            icg(ifg(ic1))=icg(ic1)
            ifg(icg(ic1))=ifg(ic1)
            ifg(ic0)=0
         endif
c     
c     check for full dependence
c     
         if(icdep(iu2,iu2).eq.0) go to 170
c     
c     sweep over st(ic0)
c     
         jlo=ib(ic0)+1
         jhi=jb(ib(ic0))
         if(jlo.gt.jhi) go to 150
         do 140 j=jlo,jhi
            i0=jb(j)
            if(ip(i0).eq.iip) go to 140
            if(icg(i0).ne.-1) go to 140
            i1=i0+ishift
            nfu(iu(i0))=nfu(iu(i0))+1
c     
c     put i0 in F & remove from lists
c     
c     check for i0 on or off lists.
c     if ifg(i0)=0, no points should depend on i0
c     
            if(ifg(i0).eq.0) then
               print *,' point off lists - (old) attempt to remove'
               go to 140
            endif

            icg(i0)=ncolor
            icg(ifg(i1))=icg(i1)
            ifg(icg(i1))=ifg(i1)
            ifg(i0)=0
c     
c     check for fully dependent variables on i0. put in F
c     
            iu3=iu(i0)
            if(icdep(iu3,iu3).eq.3) then
               ii0lo=iv(ip(i0))
               ii0hi=iv(ip(i0)+1)-1
               do 110 ii0=ii0lo,ii0hi
                  iu4=iu(ii0)
                  if(icdep(iu4,iu3).eq.2) icg(ii0)=ncolor
                  if(ifg(ii0).ne.0) print *,' fully dep. var has ifg#0'
 110           continue
            endif
c     
c     sweep over s(i0) [=s(st(ic0))]
c     
            jjlo=ia(i1)+1
            jjhi=ia(i1+1)-1
            do 130 jj=jjlo,jjhi
c     jwr  >>>>> Bug fix 4/24/96
c     ii1=ja(jj)
c     ii0=ii1-ishift
               ii0=ja(jj)
               ii1=ii0+ishift
c     jwr  <<<<<
c     
c     check for:
c     undecided variable
c     variable with max value (forced or semi-forced)
c     no strong connections
c     variable defined at point iip
c     
               if(icg(ii0).ne.-1) go to 130
               if(ifg(ii0).ge.jvalmx) go to 130
               if(ifg(ii0).eq.0) go to 130
               if(ja(ia(ii0)).eq.ia(ii0)) go to 130
               if(ip(ii0).eq.iip) go to 130
c     
c     increment weight for ii0
c     
               ii1=ii0+ishift
               if(ifg(ii0).eq.0) then
                  print *,'attempt to increase ifg when ifg=0'
                  go to 130
               endif
               ifg(ii0)=ifg(ii0)+1
c     
c     put ii0 on reset stack
c     
               if(ib(ii1).eq.0) then
                  ib(ii1)=itop
                  itop=ii1
               endif
 130        continue
 140     continue
c     
c     if desired, decrement weights for s(ic0)
c     
 150     if(isub.eq.0) go to 170
         jlo=ia(ic1)+1
         jhi=ia(ic1+1)-1
         if(jlo.gt.jhi) go to 170
         do 160 j=jlo,jhi
            i0=ja(j)
            i1=i0+ishift
            if(icg(i0).ne.-1) go to 160
            if(ifg(i0).eq.0) go to 160
            if(ifg(i0).gt.jvalmx) go to 160
            ifg(i0)=ifg(i0)-1
            if(ifg(i0).lt.jval0) go to 160
            if(ib(i1).eq.0) then
               ib(i1)=itop
               itop=i1
            endif
 160     continue
 170  continue
c     
c     rearrange the stacks & return to pick new coarse point.
c     
      ii=itop
      itop=-1
 180  if(ii.le.0) go to 100
      i=ii-ishift
c     cjwr >>>>> 1/26/95 - Is this right? indicates special point.
c     should reset only undecided pts (icg=-1)
c     if(icg(i).ne.0) go to 190
      if(icg(i).ne.-1) go to 190
      if(ifg(i).eq.0) stop ' undecided point has ifg=0'
c     cjwr <<<<<
      nru(iu(i))=nru(iu(i))+1
      ifg(icg(ii))=ifg(ii)
      icg(ifg(ii))=icg(ii)
c     jv=max0(ifg(i),jval0)
      jv=min0(ifg(i),jvalmx)
      icg(ii)=icg(jv)
      ifg(ii)=jv
      icg(jv)=ii
      ifg(icg(ii))=ii
      if(ifg(i).gt.jcnbhi) jcnbhi=jv
 190  iip=ii
      ii=ib(ii)
      ib(iip)=0
      go to 180
c     
c     all variables colored. put all points on zero list in f
c     
 200  continue
 210  ii=icg(jval0)
      if(ii.eq.jval0) go to 250
      icg(ifg(ii))=icg(ii)
      ifg(icg(ii))=ifg(ii)
      i=ii-ishift
      icg(i)=ncolor
      ifg(i)=0
c     
c     check for dependent variables and put in F
c     
      if(icdep(iu(i),iu(i)).eq.3) then
         do 220 ii0=iv(ip(i)),iv(ip(i)+1)-1
            if(icdep(iu(ii0),iu(i)).eq.2) icg(ii0)=ncolor
 220     continue
      endif

      go to 210
 250  continue
c     write(6,8000) k,ncolor,nncpts,nspts
c     write(6,8001) (j,ncu(j),nfu(j),nru(j),j=1,3)
c     write(6,8001) (j,ncu(j),nfu(j),nru(j),j=1,1)
c     flush(6)
      return
 8000 format('  crsgd  k=',i2,'  ncolor=',i2,'  # c-points=',i4,
     *     '  sp pts=',i4)
 8001 format('    unknowns forced (c/f/r) :',3(3x,'iu=',i1,3i5))
 9000 format('  crsgd: grid #',i2,' completed')
      end
