c     
c=====================================================================
c     
c     routines for performing coloring
c     
c=====================================================================
c     
      subroutine setcw(k,ncolor,jval0,jvalmx,icdep,nun,
     *     imin,imax,ia,iu,icg,ifg,ib,jb,
     *     ndimu,ndimp,ndima,ndimb)
c     
c---------------------------------------------------------------------
c     
c     Assign initial coloring (and recoloring) weights.
c     
c     NOTES:
c     
c     1. Older version (not the updated version meant to ensure
c     one-pass coloring). Should be more efficient.
c     
c     2. Initial weight is |st(i)|. In COLOR, weight is
c     increased by |st(i) intersect C|
c     
c     3. Allows for forced c-variables (must become c-variables) and
c     semi-forced c-variables (highest priority for c-variables).
c     Semi-forced c-variables CAN become f-variables.
c     
c     4. icg(i) =  -1 for variables to be assigned normal weights
c     icg(i) =   1 for forced c-points
c     icg(i) = 999 for semi-forced c-points.
c     
c     5. On exit, icg=1 for forced c-variables,
c     icg=-1 for variables to be colored (undecided variables).
c     
c     6. Dependent and coupled coarsening can be used. These
c     are defined for unknowns as follows:
c     
c     (Fully) Dependent coarsening (iu1 depends on iu2):
c     ------- --------- ----------
c     
c     color(v(ip,iu1)) = color(v(ip,iu2)) (i.e., same C/F)
c     
c     Coupled coarsening (iu1 is coupled with iu2):
c     ------- ----------
c     
c     v(ip,iu2) in C => v(ip,iu1) in C.
c     v(ip,iu2) in F => no action for v(ip,iu1).
c     
c     Notation: Let i=v(ip,iu1) be a variable.
c     
c     d(i)={j:j=v(ip,iu2) and iu2 fully depends on iu1}
c     
c     c(i)={j:j=v(ip,iu2) and iu2 is coupled with iu1}
c     
c     Dependent & coupled coarsening indicated by array icdep.
c     
c     icdep(iu1,iu2) = 0 - iu1 coarsening independent of iu2
c     icdep(iu1,iu2) = 1 - iu1 coarsening coupled to iu2
c     icdep(iu1,iu2) = 2 - iu1 coarsening fully dependent on iu2
c     
c     In addition, diagonal entries indicate:
c     
c     icdep(iu1,iu1) = 0 - no iu2 depends on iu1
c     icdep(iu1,iu1) = 1 - only iu1 depends on iu1
c     icdep(iu1,iu1) = 2 - some iu2 is coupled to iu1
c     icdep(iu1,iu1) = 3 - some iu2 is fully depends on iu1
c     
c     icdep(iu1,iu1) = 0  =>  unknown iu1 not assigned coloring
c     weights and not put on lists.
c     
c     Thus, for i=v(ip,iu2)
c     
c     c(i) = {j=v(ip,iu1):icdep(iu1,iu2)=1}
c     d(i) = {j=v(ip,iu1):icdep(iu1,iu2)=2}
c     
c     icdep(iu1,iu2) = 1  =>  i= put in c => c(i) put in c
c     
c     icdep(iu1,iu2) = 2  =>  v(ip,iu1) has same c/f assignment
c     as v(ip,iu2).
c     
c     Examples:
c     
c     icdep(iu1,iu2) # 0 iff iu1=iu2 - independent coarsening.
c     icdep(iu1,iu2) # 0 for all iu1, iu2  - full point coarsening.
c     icdep(iu1,iu2) # 0 iff iu1=iuc - coarsen according to iuc.
c     
c     7. number of unknowns (nun) computed here
c     
c     RECOLORING
c     
c     8. For recoloring, failed variables have been marked by
c     icg=-1 and put on a list in shifted ib, with list header
c     ib(i1hi+1).
c     
c     9. Recoloring weight = |st(i) intersect P|
c     
c     10. icdep test is unnecessary for recoloring, since dependent
c     variables are not tested, and not put on the list.
c     
c---------------------------------------------------------------------
c     
      implicit real*8 (a-h,o-z)
c     
c     include 'params.amg'
c     
      dimension imin(25),imax(25)
      dimension ia (*)
      dimension iu (*)
      dimension icg(*)
      dimension ifg(*)

      dimension ib (*)
      dimension jb (*)
c     
      dimension icdep(10,10)
c     
c---------------------------------------------------------------------
c     
c     print *,' setcw  - k=',k
      i0lo=imin(k)
      i0hi=imax(k)
c     
c     set shifted bounds
c     
      ishift=i0hi-i0lo+2
      i1lo=i0lo+ishift
      i1hi=i0hi+ishift
      if(ncolor.lt.-1) go to 100
c     
c==   >  set weights for first coloring pass
c     
c     compute max values of |s(i)| and |st(i)|
c     
      nsmax=0
      nstmax=0
c     
c     nun passed in now. computation not necessary
c     
      do 10 i1=i1lo,i1hi
         i0=i1-ishift
         iiu=iu(i0)
         if(icdep(iiu,iiu).eq.0) go to 10
         ns=ia(i1+1)-ia(i1)-1
         nst=jb(ib(i0))-ib(i0)
         if(ns.gt.nsmax) nsmax=ns
         if(nst.gt.nstmax) nstmax=nst
 10   continue
c     
c     SET:  jval0  - minimum (zero) value
c     jvalmx - maximum possible value
c     (for list setup and semi-forced points)
c     jvalxx - jvalmx+1 (for forced c-points)
c     
      jval0=i0hi+ishift+1
      jvalmx=jval0+nsmax*nstmax+1
      jvalxx=jvalmx+1
      if(jvalxx.gt.ndimu) go to 9903
c     
c     set weights for variables
c     jvalxx for forced c-points      (icg=1)
c     jvalmx for semi-forced c-points (icg=999)
c     
c     NOTE (4/23/96) icg(i) reset to -1 for forced C-variables
c     (Otherwise, not removed from lists in COLOR => inf. loop)
c     
      do 30 i=i0lo,i0hi
         iiu=iu(i)

         if(icdep(iiu,iiu).eq.0) then
            icg(i)=0
            ifg(i)=0
            go to 30
         endif

         iicg=icg(i)
         if(iicg.eq.1) then
            ifg(i)=jvalxx
c     jwr    4/23/96
            icg(i)=-1
         elseif(iicg.eq.999) then
            ifg(i)=jvalmx
            icg(i)=-1
         else
            ifg(i)=jval0+jb(ib(i))-ib(i)
         endif

 30   continue
      return
c     
c==   >  set weights for successive coloring passes
c     
 100  i1=i1hi+1
 110  i1=ib(i1)
      if(i1.lt.0) return
      i0=i1-ishift
c     
c     does this ever occur?
c     
      if(icg(i0).eq.999) then
         ifg(i0)=jvalmx
         icg(i0)=-1
         go to 110
      endif

      ifg(i0)=jval0+1
      jlo=ib(i0)+1
      jhi=jb(ib(i0))

      do 150 j=jlo,jhi
         ii0=jb(j)
         if(icg(ii0).eq.-1) ifg(i0)=ifg(i0)+1
 150  continue

      go to 110
c     
c===  > error messages
c     
 9903 write(6,9930)
      stop
c     
 9930 format(' ### error in setcw: ndimu too small ###')
      end
