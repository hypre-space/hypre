      program dssdrv 
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
     &                v(ldv, maxnev), d(maxnev, 2), ax(maxn)
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
      parameter         (nzmax = 100000)
      integer           nrow, ncol, nnz, I, IROW
      integer           rowind(nzmax), colptr(maxn+1), IA, JA, NCNT
      double precision  nzvals(nzmax), AA
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
c
c     %--------------------%
c     | Intrinsic function |
c     %--------------------%
c
      intrinsic        abs

      include 'debug.h'
      ndigit = -3
      logfil = 6
      msgets = 0
      msaitr = 0 
      msapps = 0
      msaupd = 0
      msaup2 = 0
      mseigt = 0
      mseupd = 0
c
c     %-----------------------%
c     | Executable Statements |
c     %-----------------------%
c
      matunit = 10
c     matin   = 'bcsstk13.rsa'
      matin   = 'hypreMat'
      open(unit = matunit, file=matin) 
c     call dreadhb(matunit, nrow, ncol, nnz, nzvals, rowind, colptr)
c Charles tong      
      READ ( matunit, 1000 ) nrow, nnz
 1000 FORMAT ( 2I6 )
      colptr(1) = 1
      NCNT = 1
      IROW = 1
      ncol = nrow
      PRINT *, 'nrow = ', nrow, ' ', nnz
      DO I = 1, nnz
         READ ( matunit, 2000 ) IA, JA, AA
         IF ( I .LT. 100 ) THEN
         PRINT *, IA, ' ', JA, ' ', AA
         END IF
         rowind(NCNT) = JA
         nzvals(NCNT) = AA
         IF ( IA .NE. IROW ) THEN
            IROW = IROW + 1
            colptr(IROW) = NCNT
         END IF
         NCNT = NCNT + 1
      END DO          
      colptr(IROW+1) = NCNT
 2000 FORMAT ( 2I6, E25.16 )
      DO I = 1, 10
         DO J = colptr(I), colptr(I+1)-1
            PRINT *, I, ' ', rowind(j), ' ', nzvals(J)
         END DO
      END DO

c Charles tong      

      close(unit = matunit)

      if (nrow .ne. ncol) then
          write(0,*) 'Matrix is not a square matrix!'
          go to 9000
      else if (nrow .ge. maxn) then
          write(0,*) 'Dimension of the matrix exceeds allocated space!'
          go to 9000
      end if

c
c     === dimension, target shift, number of eigenvalues of interest
c     
      n     = nrow
      sigma = 0.0d0
      nev   =  10 
      if ( n .gt. maxn ) then
         print *, ' ERROR with _SDRV1: N is greater than MAXN '
         go to 9000
      else if ( nev .gt. maxnev ) then
         print *, ' ERROR with _SDRV1: NEV is greater than MAXNEV '
         go to 9000
      end if
c
c     === call eigensolver === 
c
      call dspssev(n, nev, sigma, colptr, rowind, nzvals,
     &             d, v,   ldv) 
c
c     === check residual ===
c
      do 20 j=1, nev
         call dsymmv(n, nzvals, colptr, rowind, v(1,j), ax)
         call daxpy(n, -d(j,1), v(1,j), 1, ax, 1)
         d(j,2) = dnrm2(n, ax, 1)
         d(j,2) = d(j,2) / abs(d(j,1))
 20   continue
c
c     %-------------------------------%
c     | Display computed residuals    |
c     %-------------------------------%
c
      print *,'nev = ', nev
      do j=1, 10
      print *,'eigen = ', d(j,1)
      end do
      call dmout(6, nev, 2, d, maxnev, -6,
     &           'Ritz values and relative residuals')
 9000 continue
      end
