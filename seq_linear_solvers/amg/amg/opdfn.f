c
c=====================================================================
c
c     CG operator computation
c
c=====================================================================
c
      subroutine opdfn(k,levels,ierr,ndima,imin,imax,
     *                 a,ia,ja,icg,ifg,b,ib,jb)
c
c---------------------------------------------------------------------
c
c     set up cg (level k) matrix
c
c     1. The operator is constructed by multiplying the
c     restriction operator, the fine level operator, and the
c     interpolation operator. Terms (of the form c1-f1-f2-c2)
c     are accumulated in the row as it is constructed.
c
c     2. ifg is used as a pointer to previously defined entries
c     in the current row, so that no row searches are necessary.
c
c     3. Each f2-point visited is marked by setting icg = -ic.
c     Previously visited f2-points will not result in new row
c     entries, so no tests are necessary.
c
c---------------------------------------------------------------------
c
      implicit real*8 (a-h,o-z)
c
      dimension imin(25),imax(25)
      dimension ia (*)
      dimension a  (*)
      dimension ja (*)
      dimension icg(*)
      dimension ifg(*)

      dimension ib (*)
      dimension b  (*)
      dimension jb (*)
c
c---------------------------------------------------------------------
c
      if(k.gt.levels) return
      iflo=imin(k-1)
      ifhi=imax(k-1)

      imin(k)=ifhi+1
      iclo   =imin(k)
c     ichi=2*iclo-imin(k-1)
      ichi=iclo+ifhi-iflo
c
c     initialize row pointer ifg
c
      do 10 ic=iclo,ichi
      ifg(ic)=0
10    continue
c
c     sweep over the fine grid to find c-points
c
      ndimaa=ndima+iflo-ifhi
      ka=ia(ifhi+1)
      do 60 if=iflo,ifhi
      ic=icg(if)
      if(ic.le.0) go to 60
c
c     New C-row. Create diagonal entry.
c
      ifg(ic)=ka
      a(ka)=0.e0
      ja(ka)=ic
      jclo=ka
      ka=ka+1
c
c     restriction
c
      do 50 jf1=ib(if),ib(if+1)-1
      if1=jb(jf1)
      wrxx=b(jf1)
c
c     restriction * fg operator
c
      do 40 jf2=ia(if1),ia(if1+1)-1
      if2=ja(jf2)
      rlxx=wrxx*a(jf2)
      ic2=icg(if2)
c
c     restriction * fg operator * interpolation
c
c=>   case 1. if2 is a c-point.
c
      if(ic2.gt.0) then
        if(ifg(ic2).ge.jclo) then
          a(ifg(ic2))=a(ifg(ic2))+rlxx
        else
          ifg(ic2)=ka
          a(ka)=rlxx
          ja(ka)=ic2
          ka=ka+1
        endif
c
c=>   Case 2. if2 is a previously visited f-point
c
      elseif(ic2.eq.-ic) then
        do 20 jf3=ib(if2),ib(if2+1)-1
        if3=jb(jf3)
        ic3=icg(if3)
c>>>>>
c       test ifg
c
c       if(ifg(ic3).lt.jclo.or.ifg(ic3).gt.ka) stop 'opdfn'
        if(ifg(ic3).lt.jclo.or.ifg(ic3).gt.ka) then
          write(6,'('' opdfn error '')')
          write(6,'(''   if , ic  ='',2i5)') if,ic
          write(6,'(''   if2, icg(if2) ='',2i5)') if2,icg(if2)
          write(6,'(''   if3, icg(if3) ='',2i5)') if3,icg(if3)
          write(6,'(''   ic3, ifg(ic3) ='',2i5)') ic3,ifg(ic3)
          stop 'opdfn'
        endif
c<<<<<
        a(ifg(ic3))=a(ifg(ic3))+rlxx*b(jf3)
20      continue
c
c=>   Case 3. if2 is a "new" f-point
c
      else
        icg(if2)=-ic
        do 30 jf3=ib(if2),ib(if2+1)-1
        if3=jb(jf3)
        ic3=icg(if3)
        if(ifg(ic3).ge.jclo) then
          a(ifg(ic3))=a(ifg(ic3))+rlxx*b(jf3)
        else
          ifg(ic3)=ka
          a(ka)=rlxx*b(jf3)
          ja(ka)=ic3
          ka=ka+1
        endif
30      continue
      endif
40    continue
50    continue
      ia(ic+1)=ka
      if(ka.gt.ndimaa) go to 9901
60    continue
c
c     test for matrix out of bounds
c
      if(ka.gt.ndima) go to 9901
c
c     set ifg and determine imax(k)
c
      ic=ifhi
      do 70 if=iflo,ifhi
      if(icg(if).gt.0) then
        ic=ic+1
        ifg(ic)=if
      endif
70    continue
      imax(k)=ic
c     write(6,9000) k
c
c     set temp memory usage
c
      call memacct('a',ka,0)
      call memacct('ja',ka,0)
      call memacct('ifg',ichi,0)

      return
c
c===> error messages
c
 9901 write(6,9910)
      ierr=1
        return
c
 9000 format(' opdfn: grid #',i2,' completed')
 9910 format(' ### error in opdfn: ndima too small ###')
      end
