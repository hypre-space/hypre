c     
c=====================================================================
c     
c     routine for setting icdep array (coupled/dependent coarsening)
c     
c=====================================================================
c     
      subroutine setdep(idep,idun,nun,icdep)
c     
c---------------------------------------------------------------------
c     
c     1. Dependent and coupled coarsening can be used. These
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
c---------------------------------------------------------------------
c     
      implicit real*8 (a-h,o-z)
c     
      dimension icdep(10,10)
c     
c---------------------------------------------------------------------
c     
      go to (100,200,300),idep
      stop ' idep out of range'
c     
c     separate coarsening for each unknown
c     
 100  do 120 i=1,nun
         do 110 j=1,nun
            icdep(i,j)=0
 110     continue
         icdep(i,i)=1
 120  continue
      return
c     
c     coarsening (pointwise) according to unknown idun
c     
 200  do 220 i=1,nun
         do 210 j=1,nun
            icdep(i,j)=0
 210     continue
         icdep(i,idun)=2
 220  continue
      icdep(idun,idun)=3
      return
c     
c     fully coupled coarsening for all unknowns
c     
 300  do 320 i=1,nun
         do 310 j=1,nun
            icdep(i,j)=1
 310     continue
         icdep(i,i)=2
 320  continue
      return
c     
 900  write(6,'(1x)')
      write(6,'(/''  Coarsening Dependence/Coupling array:''/)')
      do 910 iu1=1,nun
         write(6,'(5x,10i2)') (icdep(iu1,iu2),iu2=1,nun)
 910  continue
      return
      end
