      subroutine ilut_driver (n, nnz, ia, ja, a, ipar, rpar,
     .     lenpmx, plu, jlu, ju, perm, qperm, rscale, cscale,
     .     iwork, liw, rwork, lrw, ier_ilut, ier_input)
c-----------------------------------------------------------------------
c This routine calculates an incomplete LU factorization of the matrix
c A defined by the ia, ja, and a arrays, and returns the factorization 
c in the plu, jlu, and ju arrays.  It uses the ILUT code from sparskit
c to calculate the incomplete LU.  Additional output arrays are
c perm, qperm, rscale and cscale.  See below for their definitions.
c
c Specifically, this routine calculates a preconditioner C defined by
c first noting that A can be rewritten as
c
c         A = S_r^{-1} * Q * ( P*(S_r*A*S_c)*Q )* P * S_c^{-1},
c
c where S_r and S_c are diagonal scaling matrices calculated from the
c inverses of the row and column norms of A, respectively, and P and Q
c are permutation matrices where the permutations are obtained via a 
c reverse Cuthill-McKee reordering of the rows and columns of S_r*A*S_c
c with Q = P^{-1} = P^T.  An incomplete factorization of the reordered
c and rescaled matrix P*(S_r*A*S_c)*Q is then performed by calling the
c ilut_factor routine to give
c
c                      L*U \approx P*(S_r*A*S_c)*Q.
c
c The preconditioner C is then defined by
c
c                C = S_r^{-1}* Q * L*U * P * S_c^{-1},
c
c giving
c
c                C^{-1} = S_c * Q * (L*U)^{-1} * P * S_r.
c
c NOTE:  When ireorder.ne.0 and/or iscale.ne.0, the original matrix is
c        returned in reordered and/or rescaled form.
c
c Date:             This is the version of 2-27-97.
c Author:           Peter N. Brown
c Organization:     Center for Applied Scientific Computing, LLNL
c Revision history: 
c  2-27-97  pnb     Original version.
c-----------------------------------------------------------------------
c On input,
c
c   n           -  integer scalar.  n is the number of rows and columns
c                  in the matrix A.
c   nnz         -  integer scalar.  nnz is the number of nonzeros in
c                  the matrix A.
c   ia,ja,a     -  sparse storage arrays for the matrix A in compressed
c                  sparse row format.
c                  ia must be an integer array of length n+1.
c                  ja must be an integer array of length nnz.
c                  ja must be a real array of length nnz.
c   ipar        -  integer array of length 20.  ipar contains
c                  optinal input parameters for ilut_factor.  A positive
c                  ipar(i) value signals a non-default input value.
c
c       ipar(1) =  ireorder, row and column reordering flag
c                  used to determine if reordering of the
c                  rows and columns are desired.  If ireorder .ne. 0,
c                  a reverse Cuthill-McKee reordering of the rows
c                  and columns of the matrix A is done.
c                  Reordering is intended to reduce the amount of
c                  fill-in when performing the incomplete LU
c                  factorization.  Default value is ireorder=0.
c       ipar(2) =  iscale,  flag used to determine if row and column 
c                  scaling of the matrix A  is desired.  If
c                  iscale=0, no scaling of A is performed,
c                  iscale=1, row scaling of A is performed,
c                  iscale=2, column scaling of A is performed,
c                  iscale=3, row and column scaling of A is performed.
c                  The inverse of the row norms of the matrix A
c                  are stored in rscale, while the inverse of the 
c                  column norms of the matrix S_r*A are stored in cscale.
c                  Scaling is performed prior to performing the
c                  incomplete LU factorization.  Incomplete LU
c                  methods (with drop tolerances such as ilut)
c                  can be sensitive to poorly scaled matrices.
c                  This can result in a poor preconditioner, i.e.,
c                  one that is not effective in reducing the
c                  number of gmres iterations for convergence.
c                  Default value is iscale=0.
c       ipar(3) =  iout, logical unit number for writing informational
c                  messages.  Default value is iout=0.
c       ipar(4) -  not used.
c       ipar(5) -  not used.
c       ipar(6) =  lfil_ilut, a fill-in parameter for the ilut
c                  preconditioner.  An additional lfil_ilut values
c                  are allowed per row of L and U in the factorization.
c                  Default value is lfil_ilut=20.
c   rpar        -  real array of length 20.  rpar contains
c                  optinal input parameters for ilut_factor.  A
c                  positive rpar(i) value signals a non-default input
c                  value.
c       rpar(1) =  tol_ilut, a drop tolerance used to determine
c                  when to drop elements in ilut when calculating 
c                  the incomplete LU.  Default value is 
c                  tol_ilut=0.0001.
c   lenpmx      -  integer scalar.  lenpmx is the number of nonzeros
c                  in the preconditioner P.  Must be large enough
c                  to account for fill, lenpmx=nnz+2*lfil_ilut*n+2.
c   ju,jlu,plu  -  sparse storage arrays for the preconditioner P
c                  stored in compressed sparse row format.
c                  iplu must be an integer array of length n+1.
c                  jplu must be an integer array of length lenpmx.
c                  plu must be a real array of length lenpmx.
c   perm        -  integer array of length n for the permutation.
c                  Only applicable if ipar(1) .ne. 0.
c   qperm       -  integer array of length n for the inverse of perm.
c                  Only applicable if ipar(1) .ne. 0.
c   rscale      -  real array of length n for diagonal scaling factors
c                  containing the inverse of the row norms of A.
c                  Only applicable if ipar(2)= .ne. 0.
c   cscale      -  real array of length n for diagonal scaling factors
c                  containing the inverse of the column norms of the
c                  row-scaled A.
c                  Only applicable if ipar(2)= .ne. 0.
c   iwork       -  integer work array of length liw.
c
c   liw         -  integer scalar containing the length of the iwork
c                  array.  liw must be greater than or equal to 3*n.
c   rwork       -  real work array of length lrw.
c   lrw         -  integer scalar containing the length of the rwork
c                  array.  lrw must be greater than or equal to 2*n+1.
c
c On return,
c
c   ju,jlu,plu  -  sparse storage arrays for the preconditioner P
c                  stored in modified sparse row format.
c   perm        -  integer array of length n containing the
c                  permutation. Only applicable if ipar(1) .ne. 0.
c   qperm       -  integer array of length n containing the inverse
c                  permutation. Only applicable if ipar(1) .ne. 0.
c   rscale      -  real array of length n for diagonal scaling factors
c                  containing the inverse of the row norms of A.
c                  Only applicable if ipar(2)= .ne. 0.
c   cscale      -  real array of length n for diagonal scaling factors
c                  containing the inverse of the column norms of the
c                  row-scaled A.
c                  Only applicable if ipar(2)= .ne. 0.
c   ier_ilut    -  integer flag to indicate success of the ilut
c                  routine in calculating the incomplete LU
c                  factorization.
c                  ier_ilut .eq. 0 indicates success.
c                  ier_ilut .ne. 0 indicates failure.
c                  ier_ilut >  0   --> Zero pivot encountered at 
c                                      step number ier_ilut.
c                  ier_ilut = -1   --> Error. input matrix may be 
c                                      wrong.  (The elimination
c                                      process has generated a
c                                      row in L or U with length > n.)
c                  ier_ilut = -2   --> Matrix L overflows.
c                  ier_ilut = -3   --> Matrix U overflows.
c                  ier_ilut = -4   --> Illegal value for lfililut.
c                  ier_ilut = -5   --> Zero row encountered.
c
c                  For ier_ilut = -2 or -3, increase the value of lenpmx
c                  or decrease the value of lfil_ilut if lenpmxc cannot
c                  be increased.
c   ipar(19)    -  minimum needed length of iwork.
c   ipar(20)    -  minimum needed length of rwork.
c   ier_input   -  integer flag to indicate if a value in ipar or rpar
c                  was illegal.  ier_input between 1 and 20 means
c                  ipar(ier_input) was illegal, while ier_input
c                  between 21 and 40 means rpar(ier_input-20) was
c                  illegal.  
c
c                  A value of ier_input=41 means that the declared
c                  length of iwork, liw, was not large enough, and
c                  ipar(19) contains the needed length.
c
c                  A value of ier_input=42 means that the declared
c                  length of rwork, lrw, was not large enough, and
c                  ipar(20) contains the needed length.
c
c-----------------------------------------------------------------------
      implicit none
      integer n, nnz, lenpmx, liw, lrw, ipar(20), ier_ilut, ier_input
      integer ia(n+1), ja(nnz), ju(n+1), jlu(lenpmx), perm(n), qperm(n),
     .        iwork(liw)
      real*8  a(nnz), plu(lenpmx), rscale(n), cscale(n),
     .        rpar(20), rwork(lrw)
c     Local variables
      integer ireorder, iscale, iout, lfil_ilut, i
      integer lmask, llevels, ljr, ljwu, ljwl, lwu, lwl, lenrw, leniw
      real*8  tol_ilut
c
      ier_input = 0
c-----------------------------------------------------------------------
c     Load values from ipar and rpar.
c-----------------------------------------------------------------------
c     check for negative ipar values.
      do i=1,20
         if (ipar(i) .lt. 0) then
            ier_input = i
            return
         endif
      enddo
c     check for ireorder value.
      if (ipar(1) .gt. 0) then
         ireorder = ipar(1)
      else
         ireorder = 0  ! default is not to perform reordering.
      endif
c     check for iscale value.
      if (ipar(2) .gt. 0) then
         iscale = ipar(2)
      else
         iscale = 0  ! default is not to perform scaling.
      endif
c     check for iout value.
      if (ipar(3) .gt. 0) then
         iout = ipar(3)
      else
         iout = 0 ! default is no informational messages.
      endif
c     check for lfil_ilut value.
      if (ipar(6) .gt. 0) then
         lfil_ilut = ipar(6)
      else
         lfil_ilut = 20  ! by default allow up to 20 fill-in elmts.
      endif
c     check for negative rpar values.
      do i=1,20
         if (rpar(i) .lt. 0) then
            ier_input = i + 20
            return
         endif
      enddo
c     check for tol_ilut value.
      if (rpar(1) .gt. 0) then
         tol_ilut = rpar(1)
      else
         tol_ilut = 0.0001  ! default drop tolerance of .0001
      endif
c-----------------------------------------------------------------------
c     Load pointers into work arrays iwork and rwork.
c-----------------------------------------------------------------------
c     iwork pointers
      lmask = 1
      llevels = lmask + n
c     ljr = llevels + n
      ljr = 1
      ljwu = ljr + n
      ljwl = ljwu + n
      leniw = ljwl + n - 1
c     check for enough iwork space
      ipar(19) = leniw
      if (leniw .gt. liw) then
         ier_input = 41
         return
      endif
c     rwork pointers
      lwu = 1
      lwl = lwu + (n + 1)
      lenrw = 2*n + 1
c     check for enough rwork space
      ipar(20) = lenrw
      if (lenrw .gt. lrw) then
         ier_input = 42
         return
      endif
c-----------------------------------------------------------------------
c     Call ilut_factor to solve the linear system.
c-----------------------------------------------------------------------
      call ilut_factor (n, nnz, lfil_ilut, tol_ilut,
     .     ia, ja, a, plu, jlu, ju, lenpmx, perm, qperm, rscale, cscale, 
     .     iwork(lmask), iwork(llevels), iwork(ljr), iwork(ljwu), 
     .     iwork(ljwl), rwork(lwu), rwork(lwl),ireorder, iscale, iout, 
     .     ier_ilut)
      return
c-------------end-of-subroutine-ilut_driver-----------------------------
c-----------------------------------------------------------------------
      end
      subroutine ilut_factor (n, nnz, lfil_ilut, tol_ilut,
     .     ia, ja, a, plu, jlu, ju, lenpmx, perm, qperm, rscale, cscale,
     .     mask, levels, jr, jwu, jwl, wu, wl, ireorder, iscale, iout,
     .     ier_ilut)
c-----------------------------------------------------------------------
c This routine factors the matrix A defined by the ia, ja, a arrays.
c It uses the ILUT code from sparskit to calculate the incomplete LU.
c This routine is normally called by the ilut_driver routine.
c
c NOTE:  When ireorder.ne.0 and/or iscale.ne.0, the original matrix is
c        returned in reordered and/or rescaled form.
c
c
c Date:             This is the version of 2-27-97.
c Author:           Peter N. Brown
c Organization:     Center for Applied Scientific Computing, LLNL
c Revision history:
c  2-27-97  pnb     Original version.
c-----------------------------------------------------------------------
c On input,
c
c   n           -  integer scalar.  n is the number of rows and columns
c                  in the matrix A.
c   nnz         -  integer scalar.  nnz is the number of nonzeros in
c                  the matrix A.
c   ia,ja,a     -  sparse storage arrays for the matrix A in compressed
c                  sparse row format.
c                  ia must be an integer array of length n+1.
c                  ja must be an integer array of length nnz.
c                  ja must be a real array of length nnz.
c   lfil_ilut   -  integer scalar.  lfil_ilut is a fill-in
c                  parameter for ilut preconditioner.  An
c                  additional lfil_ilut values are allowed per
c                  row of L and U in the factorization.
c   tol_ilut    -  real scalar.  tol_ilut is a drop tolerance
c                  for dropping elements in ilut when calculating 
c                  the incomplete LU.
c   lenpmx      -  the maximum allowable size of the jlu and plu
c                  work arrays.  lenpmx must be greater than or
c                  equal to nnz + 2*n*lfil_ilut.
c   plu,jlu,ju  -  work space used for reordering the rows and columns
c                  of the matrix, and to store the incomplete LU
c                  factorization of the matrix A in modified sparse
c                  row format.
c                  ju must be an integer array of length n+1.
c                  jlu must be an integer array of length lenpmx.
c                  plu must be a real array of length lenpmx.
c   perm,qperm  -  integer work arrays of length n used to hold the
c                  row and column permutation matrix (perm), and its
c                  inverse (qperm).  perm and qperm are only used if
c                  ireorder .eq. 0.
c   rscale,cscale
c               -  real work arrays of length n used to hold the
c                  diagonal scaling factors of the matrix.  rscale and
c                  cscale are not used if iscale .eq. 0.
c   mask        -  integer work array of length n. (reordering)
c   levels      -  integer work array of length n. (reordering)
c   jr          -  integer work array of length n. (ilut)
c   jwu         -  integer work array of length n. (ilut)
c   jwl         -  integer work array of length n. (ilut)
c   wu          -  real work array of length n+1.  (ilut)
c   wl          -  real work array of length n.    (ilut)
c   ireorder    -  integer flag to determine if reordering of the
c                  rows and columns are desired.  If ireorder .ne. 0,
c                  a reverse Cuthill-McKee reordering of the rows
c                  and columns of the matrix A is done.
c                  Reordering is intended to reduce the amount of
c                  fill-in when performing the incomplete LU
c                  factorization.
c   iscale      -  integer flag to determine if row scaling of
c                  the matrix A by its row norms is desired.  If
c                  iscale .ne. 0, the matrix A and rhs are
c                  scaled by the row norms of A prior to performing
c                  incomplete LU factorization.  Incomplete LU
c                  methods (with drop tolerances such as ilut)
c                  can be sensitive to poorly scaled matrices.
c                  This can result in a poor preconditioner, i.e.,
c                  one that is not effective in reducing the
c                  number of gmres iterations for convergence.
c   iout        -  logical unit number for writing informational
c                  messages.
c
c On return,
c
c   plu,jlu,ju  -  contains the incomplete LU factorization of the
c                  matrix A in modified sparse row (MSR) format.
c   ier_ilut    -  integer flag to indicate success of the ilut
c                  routine in calculating the incomplete LU
c                  factorization.
c                  ier_ilut .eq. 0 indicates success.
c                  ier_ilut .ne. 0 indicates failure.
c                  ier_ilut >  0   --> Zero pivot encountered at 
c                                      step number ier_ilut.
c                  ier_ilut = -1   --> Error. input matrix may be 
c                                      wrong.  (The elimination
c                                      process has generated a
c                                      row in L or U with length > n.)
c                  ier_ilut = -2   --> Matrix L overflows.
c                  ier_ilut = -3   --> Matrix U overflows.
c                  ier_ilut = -4   --> Illegal value for lfililut.
c                  ier_ilut = -5   --> Zero row encountered.
c                  ier_ilut = -6   --> Zero column encountered.
c
c                  For ier_ilut = -2 or -3, increase the value of lenpmx
c                  or decrease the value of lfil_ilut if lenpmxc cannot
c                  be increased.
c
c-----------------------------------------------------------------------
      implicit none
      integer n, nnz, lenpmx
      real*8  a(nnz)
      integer ia (n+1), ja(nnz)
      real*8  plu(lenpmx)
      integer ju (n+1), jlu(lenpmx)
      integer jr(n), jwu(n), jwl(n)
      real*8 wu(n+1), wl(n), tol_ilut, rscale(n), cscale(n)
      integer perm(n), qperm(n), mask(n), levels(n)
      integer ireorder, iscale, ier_ilut, lfil_ilut, iout
c
      integer nfirst, nlev, i, maskval
c
      ier_ilut = 0
c----------------------------------------------------------------------
c     Scale rows of matrix if desired.
c----------------------------------------------------------------------
      if ((iscale .eq. 1) .or. (iscale .eq. 3)) then
         call roscal (n,0,2,a,ja,ia,rscale,a,ja,ia,ier_ilut)
         if (ier_ilut .ne. 0) then
            ier_ilut = -5
            if (iout .ne. 0) write(iout,*) 
     .           ' Nonzero ier value from ILUT, ier_ilut = ',ier_ilut
            write(iout,9000)
            return
         endif
      endif
c----------------------------------------------------------------------
c     Scale rows of matrix if desired.
c----------------------------------------------------------------------
      if ((iscale .eq. 2) .or. (iscale .eq. 3)) then
         call coscal (n,0,2,a,ja,ia,cscale,a,ja,ia,ier_ilut)
         if (ier_ilut .ne. 0) then
            ier_ilut = -6
            if (iout .ne. 0) write(iout,*) 
     .           ' Nonzero ier value from ILUT, ier_ilut = ',ier_ilut
            write(iout,9000)
            return
         endif
      endif
c----------------------------------------------------------------------
c     Calculate reverse Cuthill-McKee reordering of rows and columns
c     of the matrix if ireorder is nonzero.  Otherwise, use the
c     natural ordering.  Uses the bfs routine from sparskit.
c----------------------------------------------------------------------
      if (ireorder .ne. 0) then
         nfirst = 1
         perm(1)=0
         do i = 1,n
            mask(i) = 1
         enddo
         maskval = 1
         qperm(1) = 1
         call bfs (n,ja,ia,nfirst,perm,
     .        mask,maskval,qperm,levels,nlev)
         call reversp (n,qperm)
c     Calculate inverse of qperm  and put in perm.
         do i = 1,n
            perm(qperm(i))=i
         enddo
c     Permute rows and columns of A using perm. Use ju,jlu,plu as
c     temporary workspace.
         do i=1,n+1
            ju(i)=ia(i)
         enddo
         do i=1,nnz
            jlu(i)=ja(i)
            plu(i)=a(i)
         enddo
         call dperm (n,plu,jlu,ju,a,ja,ia,perm,perm,1)
      endif
c----------------------------------------------------------------------
c     Call Incomplete Factorization Routine ILUT.
c----------------------------------------------------------------------
      call ilut (n,a,ja,ia,lfil_ilut,tol_ilut,plu,jlu,ju,lenpmx,
     .     wu,wl,jr,jwl,jwu,ier_ilut) 
c     Return immediately if ilut failed.
      if (ier_ilut .ne. 0) then
         if (iout .ne. 0) write(iout,*) 
     .        ' Nonzero ier value from ILUT, ier_ilut = ',ier_ilut
         write(iout,9000)
 9000    format(
     ./' ier_ilut >  0   --> Zero pivot encountered at step number'
     ./'                     ier_ilut.'
     ./' ier_ilut = -1   --> Error. input matrix may be wrong.'
     ./'                 (The elimination process has generated a'
     ./'                  row in L or U with length > n.)'
     ./' ier_ilut = -2   --> Matrix L overflows.'
     ./' ier_ilut = -3   --> Matrix U overflows.'
     ./' ier_ilut = -4   --> Illegal value for lfililut.'
     ./' ier_ilut = -5   --> Zero row encountered.'
     ./' ier_ilut = -6   --> Zero column encountered.'
     ./'    '
     ./' For ier_ilut = -2 or -3, increase the value of lenpmx or'
     ./' decrease the value of lfil_ilut if lenpmx cannot be'
     ./' increased.'
     .)
         return
      endif
c
      return
c-------------end-of-subroutine-ilut_factor----------------------------
c----------------------------------------------------------------------
      end


