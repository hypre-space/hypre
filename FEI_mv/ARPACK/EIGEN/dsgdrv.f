      program dsgdrv 
      implicit none
c
      integer          maxn, maxnev, ldv
      parameter       (maxn=3000, maxnev=10, ldv=maxn )
c
c     %--------------%
c     | Local Arrays |
c     %--------------%
c
      Double precision
     &                v(ldv, maxnev), d(maxnev, 2), ax(maxn),
     &                mx(maxn)
c
c     %---------------%
c     | Local Scalars |
c     %---------------%
c
      integer          n, nev, j
      Double precision      
     &                 sigma
c     %----------------------------------%
c     | Variables used for Harwll-Boeing |
c     | Sparse matrix                    |
c     %----------------------------------%
c
      integer           nzmax
      parameter         (nzmax = 50000)
      integer           nrow, ncol, nnz
      integer           aind(nzmax), aptr(maxn+1), 
     &                  bind(nzmax), bptr(maxn+1)
      double precision  aval(nzmax), bval(nzmax)
      integer           matunit
      character         matin*20 
c
c     %-----------------------------%
c     | BLAS & LAPACK routines used |
c     %-----------------------------%
c
      Double precision           
     &                 dnrm2
      external         dnrm2, daxpy

c     %--------------------%
c     | Intrinsic function |
c     %--------------------%
c
      intrinsic        abs
c
c     %-----------------------%
c     | Executable Statements |
c     %-----------------------%
c
c     ===  read the stiffness matrix ===
c 
      matunit = 10
      matin = 'bcsstk13.rsa'
      open(unit = matunit, file = matin)
      call dreadhb(matunit, nrow, ncol, nnz, aval, aind, aptr)
      close(unit=matunit)
c
      if (nrow .ne. ncol) then
          write(0,*) 'Matrix is not a square matrix!'
          go to 9000
      else if (nrow .ge. maxn) then
          write(0,*) 'Dimension of the matrix exceeds allocated space!'
          go to 9000
      end if
c
c     === read mass matrix ===
c
      matin = 'bcsstm13.rsa'
      open(unit = matunit, file = matin)
      call dreadhb(matunit, nrow, ncol, nnz, bval, bind, bptr)
      close(unit=matunit)
c
c     === specify desired eigenvalues ===
c
      n     = nrow
      sigma = 2.4d3
      nev   =  10 
      if ( n .gt. maxn ) then
         print *, ' ERROR with _SDRV1: N is greater than MAXN '
         go to 9000
      else if ( nev .gt. maxnev ) then
         print *, ' ERROR with _SDRV1: NEV is greater than MAXNEV '
         go to 9000
      end if
c
c     === call sparse eigensolver ===
c
      call dspsgev(n, nev, sigma, aptr, aind, aval, bptr, bind, bval,
     &             d, v,   ldv) 
c
c     === check the accuracy of the results === 
c
      do 20 j=1, nev
         call dsymmv(n, aval, aptr, aind, v(1,j), ax)
         call dsymmv(n, bval, bptr, bind, v(1,j), mx)
         call daxpy(n, -d(j,1), mx, 1, ax, 1)
         d(j,2) = dnrm2(n, ax, 1)
         d(j,2) = d(j,2) / abs(d(j,1))
 20   continue
c
c     %-------------------------------%
c     | Display computed residuals    |
c     %-------------------------------%
c
      call dmout(6, nev, 2, d, maxnev, -6,
     &           'Ritz values and relative residuals')
 9000 continue
      end
