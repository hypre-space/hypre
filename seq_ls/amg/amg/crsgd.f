CC### filename: CG.FOR
c==== FILE CG.FOR ====================================================
c
c     CRSGD: coarsening routines for systems
c
c=====================================================================
c
      subroutine crsgd(k,nstr,ecg,ncg,ewt,nwt,mmax,icdep,
     *                 nun,imin,imax,a,ia,ja,iu,ip,icg,ifg,
     *                 b,ib,jb,ipmn,ipmx,iv,xp,yp,ifc,
     *                 ndimu,ndimp,ndima,ndimb)
c
c---------------------------------------------------------------------
c
c     NOTES:
c
c     1. Many earlier (test) options are eliminated in this version.
c
c     2. Only direct strong connections are used.
c
c     3. There are several coarsening options:
c        Separate coarsening.
c        Coupled coarsening.
c        Coarsening driven by one unknown.
c
c     These are controlled by the array icdep, which allow for
c     dependent or coupled coarsening (see setcw & color documentation
c
c     4. Standard interpolation (within each unknown) is used.
c
c---------------------------------------------------------------------
c
c     CRSGD OUTLINE:
c
c     1. Perform rowsort (determine direct strong connections).
c        Determine strong connections & load into "high" storage.
c        Set initial form of b (interpolation/restriction storage).
c
c     2. Force initial C-point choice.
c
c     3. Set coloring weights.
c
c     4. Color the grid.
c
c     5. Test resulting points.
c        If iact=0, no action.
c        If iact=1, recolor the grid (go to 3)
c        If iact=2, no action (left to weight computation)
c
c     6. Test for coarsest grid
c
c     7. Reset form of b (for f-rows)
c
c     8. Set interpolation weights.
c
c     9. Set coarse grid pointers, etc.
c
c     10. Define & store restriction
c
c---------------------------------------------------------------------
c
c     COLORING VERSION --- points are first colored in order
c     to speed up the coarse grid point choice algorithm.
c
c     ncg -- 5 digits
c
c          1st digit -- idep   used to specify dependent coarsening
c
c                              1 - separate coarsening for unknowns
c
c                              2 - coarsen according to one unknown
c                                  (set by second digit, idun).
c
c                              3 - fully coupled point coarsening
c
c                              Note: This determines the definition
c                              of the coarsening indicator array
c                              icdep. Other options can be added in
c                              the routine setdep, or can be passed
c                              in by reading icdep from the data file.
c
c          2nd digit -- idun   driving unknown for dependent coars-
c                              ening. Only used when idep=2.
c
c          3rd digit -- ifcg   sets way to force c-points before
c                              the coloring algorithm is used.
c                              0 - no points are forced
c                              1 - predefined c-points used.
c
c          4th digit -- itst   sets the criterion f-points must meet.
c                              after coloring, resulting f-points are
c                              tested against this criterion.
c                              0 - no test is performed.
c                              1 - standard test (matrix weights)
c                              2 - standard test (number of connection
c
c          5th digit -- iact   defines action taken for points not
c                              passing the f-point test.
c                              0 - no action taken.
c                              1 - the points not passing the test
c                                  are re-colored.
c                              2 - the action is left to the test
c                                  routine
c
c     nstr -- 3 digits
c
c          1st digit -- istr   sets definition of strong connection
c                              1 - direct strong connections used
c                              2 - long range connections used
c                              3 - long range connections used
c                                  counting choice of 2nd order
c
c          2nd digit -- isort  determines row sort.
c                              0 - strong connections determined
c                                  by absolute value.
c                              1 - strong connections determined
c                                  by sign of diagonal entry.
c
c     nwt -- 3+ digits
c
c          1st digit -- iwts   sets the method for defining the
c                              interpolation weights to be used.
c                              0 - no weights are assigned.
c                              1 - equal weights are assigned.
c                              2 - standard method.
c                              3 - iterative weight definition
c                              4 - weights assigned in ftest &
c                                  stored in a+. move into b.
c                              *** (Note: No weights computed
c                                  in ftest1 or ftest2.)
c
c          2nd, 3rd digits     remainder of digits. meaning depends
c                              on the value of iwts
c
c---------------------------------------------------------------------
c
c     WORK/STORAGE AREAS USED
c
c     the following diagram illustrates the storage needs for
c     the crsgd setup phase.
c
c               1                 2 3 4                 5 6        7
c             --------------------------------------------------------
c     ia   ... \///////grid k//////\/\///shifted grid k//\
c             --------------------------------------------------------
c             --------------------------------------------------------
c     ib   ... \///////grid k//////\/\////reset stack////\
c             --------------------------------------------------------
c             --------------------------------------------------------
c     icg  ... \////c/f indicator//\
c             --------------------------------------------------------
c             --------------------------------------------------------
c     ifg  ... \////point values///\
c             --------------------------------------------------------
c             --------------------------------------------------------
c     iu   ... \///////grid k//////\
c             --------------------------------------------------------
c             --------------------------------------------------------
c     ipt  ... \///////grid k//////\
c             --------------------------------------------------------
c
c     the locations marked by 1,2, ...,7 are defined as follows:
c
c       1 -- imin(k)
c       2 -- imax(k)
c       3 -- imax(k)+1
c       4 -- imax(k)+2
c       5 -- imax(k)+(imax(k)-imin(k)+1)+1
c       6 -- imax(k)+(imax(k)-imin(k)+1)+2 = jval0
c       7 -- jvalmx+1
c
c    /// denotes space in use
c
c     the lists used to hold points according to their coloring values
c     are numbered jval0,...,jvalmx,jvalmx+1. the value jvalmx is an
c     upper bound for all possible coloring weights which can naturall
c     occur. (this is the maximal number of strong connections times t
c     maximal number of points strongly connected to a single point.)
c     points which are assigned a weight of jvalmx+1 will always becom
c     c-points, since their value is never decreased during coloring,
c     are the other points. the lists used are doubly linked circular
c     lists. the header pointers are icg(jval) and ifg(jval). if a poi
c     i is on a list, then its associated pointers are icg(i+ishift)
c     and ifg(i+ishift). during the coloring, each point whose value
c     changes is put on a reset list, and the lists are updated after
c     all values have been changed. ib(i+ishift) is the list pointer
c     for point i. when this = 0, i is not to be reset. when it is to
c     be reset, ib points to the next point in the list. itop is the
c     top of list pointer, and ib(i+ishift)=-1 for the point on the
c     bottom.
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

      dimension ifc(*)

      dimension ib (*)
      dimension b  (*)
      dimension jb (*)
c
      dimension iarr(10)
c
      dimension icdep(10,10)
c
c---------------------------------------------------------------------
c
      if(k.eq.0.or.mmax.le.k) return
c
c     assemble the crsgd7 parameters
c
      call idec(nstr,3,ndigit,iarr)
      istr   =iarr(1)
      isort  =iarr(2)

      call idec(ncg,7,ndigit,iarr)
      idep   =iarr(1)
      idun   =iarr(2)
      ifcg   =iarr(3)
      itst   =iarr(4)
      iact   =iarr(5)

      call idec(nwt,5,ndigit,iarr)
      iwts   =iarr(1)
c     write(*,'(1x,i1\)') 0
c
c     print *,'  k,nwt,iwrw,iwts,nwt=',k,nwt,iwrw,iwts,nwt
c     key=ixkey()
c
c===> set array for coupled/dependent coarsening
c
      call setdep(idep,idun,nun,icdep)
c     write(*,'(1x,i1\)') 1
c
c===> define the strong connections
c
      call strcnc(k,isort,ecg,istr,imin,imax,a,ia,ja,iu,
     *            ndimu,ndimp,ndima,ndimb)
c     write(*,'(1x,i1\)') 2
c
c===> set the form of b
c
      call binitl(k,imin,imax,ia,ja,ifg,b,ib,jb,
     *            ndimu,ndimp,ndima,ndimb)
c     write(*,'(1x,i1\)') 3
c
c     set initial color value
c
      ncolor=-1
c
c===> force initial c-points
c
10    call cforce(k,ifcg,ncolor,imin,imax,ip,icg,ifg,iv,ifc)
c     write(*,'(1x,i1\)') 4
c
c===> set the weights for coloring
c
      call setcw(k,ncolor,jval0,jvalmx,icdep,nun,
     *           imin,imax,ia,iu,icg,ifg,ib,jb,
     *           ndimu,ndimp,ndima,ndimb)
c     write(*,'(1x,i1\)') 5
c
c===> perform the coloring
c
      call color(k,ncolor,jval0,jvalmx,icdep,
     *           imin,imax,ia,ja,iu,ip,icg,ifg,ib,jb,iv,
     *           ndimu,ndimp,ndima,ndimb)
c     write(*,'(1x,i1\)') 6
c
c===> test the resulting f-points
c
      call ftest(k,itst,ncolor,ewt,nwt,iact,npts,icdep,
     *           imin,imax,a,ia,ja,iu,ip,icg,ifg,ib,iv,
     *           ndimu,ndimp,ndima,ndimb)
c     write(*,'(1x,i1\)') 7
c
c===> return to recolor points if necessary
c
      if(npts.gt.0.and.iact.eq.1) go to 10
CCJWR >>>>>
c
c     perform tests for verification (run all tests)
c
c     call fftest(k,ewt,icdep,nun,imin,imax,a,ia,ja,iu,icg,ifg,ib,
c    *            ndimu,ndimp,ndima,ndimb)
c     write(*,'(1x,i1\)') 8
c
CCJWR <<<<<
c
c===> test for grid coarse enough
c
      call gtest(k,mmax,imin,imax,icg,ifg,
     *           ndimu,ndimp,ndima,ndimb)
c     write(*,'(1x,i1\)') 9
c
c===> return if coarsest level
c
      if(mmax.le.k) return
c
c===> load f-rows of b with strong c-connections
c
      call bloadf(k,imin,imax,a,ia,ja,icg,b,ib,jb,
     *            ndimu,ndimp,ndima,ndimb)
c     write(*,'(1x,i1\)') 0
c
c===> set the interpolation weights
c
      call setw(k,ewt,nwt,iwts,imin,imax,a,ia,ja,iu,icg,ifg,b,ib,jb,
     *          ipmn,ipmx,ip,iv,xp,yp)
c     write(*,'(1x,i1\)') 1
c
c===> set the pointers for the coarse grid
c
      call setc(k,imin,imax,iu,ip,icg,ifg,ipmn,ipmx,iv,xp,yp)
c     write(*,'(1x,i1\)') 4
c
c===> define restriction (transpose of interpolation only)
c
      call rstdf0(k,imin,imax,icg,b,ib,jb,
     *            ndimu,ndimp,ndima,ndimb)
c     write(*,'(1x,i1\)') 5
c
      return
      end
