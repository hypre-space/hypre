c
c=====================================================================
c
c     routine to set form of matrices
c
c=====================================================================
c
      subroutine form(iform,imin,imax,a,ia,ja,iu,ip,icg,iv)
c
c---------------------------------------------------------------------
c
c     reorder unknowns for point or unknown block form
c
c     form rearranges the matrix and renumbers the variables
c     according to the needs of the program.
c
c     if iform = 1, point oriented processing will be performed,
c     and all variables defined at a particular point must be
c     numbered consecutively, and the rows of the matrix will be
c     rearranged correspondingly (as well as the column indices)
c     in order to maintain the variable/equation correspondence.
c     it is assumed that on entry each point has been given
c     a unique point number between 0 and nnu (the total number
c     of variables) and that number has been assigned to all
c     variables i defined at that point by setting ip(i) to that
c     number. on exit, new point numbers 1,2,3,...,np will have
c     been assigned and variables will have been renumbered
c     so that, for variables i and j, i<j implies that
c     ip(i) <= ip(j). in addition, a new vector, iv, is
c     defined so that iv(n), n=1,2,3,...,np, contains the new
c     number of the first variable defined at that point.
c     iv(np+1)=nnu+1, so that the variables defined at point
c     n are numbered iv(n), iv(n)+1, ... iv(n+1)-1.
c
c     if iform = 2, a similar renumbering is performed using
c     variable types rather than point numbers. (this is more
c     for reasons of efficiency than program requirements.)
c     no vector corresponding to iv is defined. 
c
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

      dimension iv (*)
c
      dimension iun(10)
c
c---------------------------------------------------------------------
c
      if(iform.eq.0) return
      if(iform.eq.2) go to 100
      ilo=imin(1)
      ihi=imax(1)
c
c     initialize iv
c
      do 10 i=ilo,ihi
      iv(i)=0
   10 continue
      do 30 i=ilo,ihi
      ipoint=ip(i)
      if(iv(ipoint).ne.0) go to 20
      icg(i)=i
      iv(ipoint)=i
      go to 30
   20 iold=iv(ipoint)
      icg(i)=icg(iold)
      icg(iold)=i
      iv(ipoint)=i
   30 continue
      ka=ia(ihi+1)
      ipp=0
      ii=ihi+1
      inew=1
      do 60 i=ilo,ihi
      if(iv(i).eq.0) go to 60
      ipp=ipp+1
      i1=iv(i)
      iend=i1
      iv(ipp)=inew
   40 i1=icg(i1)
      icg(i1)=inew
      ip(inew)=ipp
      ia(ii)=ka
      jlo=ia(i1)
      jhi=ia(i1+1)-1
      do 50 j=jlo,jhi
      a(ka)=a(j)
      ja(ka)=ja(j)
      ka=ka+1
   50 continue
      if(i1.eq.iend) go to 60
      ii=ii+1
      inew=inew+1
      go to 40
   60 continue
c
c     overwrite old matrix and change ja
c
      iv(ipp+1)=inew
      ia(ii)=ka
      ka=1
      iilo=ihi+1
      iihi=ii
      i=1
      do 80 ii=iilo,iihi
      ia(i)=ka
      jlo=ia(ii)
      jhi=ia(ii+1)-1
      do 70 j=jlo,jhi
      a(ka)=a(j)
      ja(ka)=icg(ja(j))
      ka=ka+1
   70 continue
      i=i+1
   80 continue
      return
c
c     arrange rows by variable type
c
  100 ilo=imin(1)
      ihi=imax(1)
      do 110 i=1,10
      iun(i)=0
  110 continue
      do 130 i=ilo,ihi
      ivtype=iu(i)
      if(iun(ivtype).ne.0) go to 120
      icg(i)=i
      iun(ivtype)=i
      go to 130
  120 iold=iun(ivtype)
      icg(i)=icg(iold)
      icg(iold)=i
      iun(ivtype)=i
  130 continue
      ka=ia(ihi+1)
      ityp=0
      ii=ihi+1
      inew=1
      do 160 i=1,10
      if(iun(i).eq.0) go to 160
      ityp=ityp+1
      i1=iun(i)
      iend=i1
  140 i1=icg(i1)
      icg(i1)=inew
      iu(inew)=ityp
      ia(ii)=ka
      jlo=ia(i1)
      jhi=ia(i1+1)-1
      do 150 j=jlo,jhi
      a(ka)=a(j)
      ja(ka)=ja(j)
      ka=ka+1
  150 continue
      if(i1.eq.iend) go to 160
      ii=ii+1
      inew=inew+1
      go to 140
  160 continue
c
c     overwrite old matrix and change ja
c
      ia(ii)=ka
      ka=1
      iilo=ihi+1
      iihi=ii
      i=1
      do 180 ii=iilo,iihi
      ia(i)=ka
      jlo=ia(ii)
      jhi=ia(ii+1)-1
      do 170 j=jlo,jhi
      a(ka)=a(j)
      ja(ka)=icg(ja(j))
      ka=ka+1
  170 continue
      i=i+1
  180 continue
      return
      end
