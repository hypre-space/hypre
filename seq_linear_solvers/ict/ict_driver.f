      subroutine ict_driver (n, nnz, ia, ja, a, ipar, rpar,
     .     lenpmx, plu, jlu, perm, qperm, scale,
     .     iwork, liw, rwork, lrw, ier_ict, ier_input)
c-----------------------------------------------------------------------
c This routine calculates an incomplete LDL^T factorization of the
c matrix A defined by the ia, ja, and a arrays, and returns the
c factorization in the plu and jlu arrays.  It uses the ICT code from
c sparskit to calculate the incomplete LDL^T.  Additional output arrays
c are perm, qperm, and scale.  See below for their definitions.
c
c Specifically, this routine calculates a preconditioner C defined by
c first noting that A can be rewritten as
c
c         A = S^{-1} * Q * ( P*(S*A*S)*Q )* P * S^{-1},
c
c where S is a diagonal scaling matrix calculated from the row norms
c of A, and P and Q are permutation matrices where the permutations
c are obtained via a reverse Cuthill-McKee reordering of
c the rows and columns of S*A*S with Q = P^{-1} = P^T.  An incomplete
c factorization of the reordered and rescaled matrix P*(S*A*S)*Q is
c then performed by calling the ict_factor routine to give
c
c                      L*D*L^T \approx P*(S*A*S)*Q.
c
c The preconditioner C is then defined by
c
c                C = S^{-1}* Q * L*D*L^T * P * S^{-1},
c
c giving
c
c                C^{-1} = S * Q * (L*D*L^T)^{-1} * P * S.
c
c NOTE:  When ireorder.ne.0 and/or iscale.ne.0, the original matrix is
c        returned in reordered and/or rescaled form.
c
c Date:             This is the version of 5-15-97.
c Author:           Peter N. Brown
c Organization:     Center for Applied Scientific Computing, LLNL
c Revision history: 
c  3-21-97  pnb     Original version.
c  5-15-97  Andy Cleary, CASC, LLNL
c                   Workspace requirements reduced by calls to
c                   revised versions of SPARSKIT routines PERPHN and
c                   BFS, and rearrangement of use of preconditioning
c                   matrix for workspace during the reordering.
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
c                  optinal input parameters for ict_factor.  A positive
c                  ipar(i) value signals a non-default input value.
c
c       ipar(1) =  ireorder, row and column reordering flag
c                  used to determine if reordering of the
c                  rows and columns are desired.  If ireorder .ne. 0,
c                  a reverse Cuthill-McKee reordering of the rows
c                  and columns of the matrix A is done.
c                  Reordering is intended to reduce the amount of
c                  fill-in when performing the incomplete LDL^T
c                  factorization.  Default value is ireorder=0.
c       ipar(2) =  iscale,  flag used to determine if row and column 
c                  scaling of the matrix A  is desired.  If
c                  iscale=0, no scaling of A is performed,
c                  iscale=1, row scaling of A is performed,
c                  iscale=2, column scaling of A is performed,
c                  iscale=3, row and column scaling of A is performed.
c                  The inverse of the row norms of the matrix A
c                  are stored in scale.
c                  Scaling is performed prior to performing the
c                  incomplete LDL^T factorization.  Incomplete LDL^T
c                  methods (with drop tolerances such as ict)
c                  can be sensitive to poorly scaled matrices.
c                  This can result in a poor preconditioner, i.e.,
c                  one that is not effective in reducing the
c                  number of pcg iterations for convergence.
c                  Default value is iscale=0.
c       ipar(3) =  iout, logical unit number for writing informational
c                  messages.  Default value is iout=0.
c       ipar(4) -  not used.
c       ipar(5) -  not used.
c       ipar(6) =  lfil_ict, a fill-in parameter for the ict
c                  preconditioner.  An additional lfil_ict values
c                  are allowed per row of L and U in the factorization.
c                  Default value is lfil_ict=20.
c   rpar        -  real array of length 20.  rpar contains
c                  optinal input parameters for ict_factor.  A
c                  positive rpar(i) value signals a non-default input
c                  value.
c       rpar(1) =  tol_ict, a drop tolerance used to determine
c                  when to drop elements in ict when calculating 
c                  the incomplete LDL^T.  Default value is 
c                  tol_ict=0.0001.
c   lenpmx      -  integer scalar.  lenpmx is the number of nonzeros
c                  in the preconditioner P.  Must be large enough
c                  to account for fill, 
c                     lenpmx = max(n + 1 + lfil_ict*n, nnz).
c   jlu,plu     -  sparse storage arrays for the preconditioner P
c                  stored in compressed sparse row format.
c                  jplu must be an integer array of length lenpmx.
c                  plu must be a real array of length lenpmx.
c   perm        -  integer array of length n for the permutation.
c                  Only applicable if ipar(1) .ne. 0.
c   qperm       -  integer array of length n for the inverse of perm.
c                  Only applicable if ipar(1) .ne. 0.
c   scale       -  real array of length n for diagonal scaling factors
c                  containing the inverse of the row norms of A.
c                  Only applicable if ipar(2)= .ne. 0.
c   iwork       -  integer work array of length liw.
c
c   liw         -  integer scalar containing the length of the iwork
c                  array.  liw must be greater than or equal to 
c                           3*n + max(2*n*lfil_ict,2*nnz+n+1)
c   rwork       -  real work array of length lrw.
c   lrw         -  integer scalar containing the length of the rwork
c                  array.  lrw must be greater than or equal to 2*n+1.
c
c On return,
c
c   jlu,plu     -  sparse storage arrays for the preconditioner P
c                  stored in modified sparse row format.
c   perm        -  integer array of length n containing the
c                  permutation. Only applicable if ipar(1) .ne. 0.
c   qperm       -  integer array of length n containing the inverse
c                  permutation. Only applicable if ipar(1) .ne. 0.
c   scale       -  real array of length n for diagonal scaling factors
c                  containing the inverse of the row norms of A.
c                  Only applicable if ipar(2)= .ne. 0.
c   ier_ict     -  integer flag to indicate success of the ict
c                  routine in calculating the incomplete LDL^T
c                  factorization.
c                  ier_ict .eq. 0 indicates success.
c                  ier_ict .ne. 0 indicates failure.
c                  ier_ict >  0   --> Zero pivot encountered at 
c                                      step number ier_ict.
c                  ier_ict = -1   --> Error. input matrix may be 
c                                      wrong.  (The elimination
c                                      process has generated a
c                                      row in L or U with length > n.)
c                  ier_ict = -2   --> Matrix L overflows.
c                  ier_ict = -3   --> Matrix U overflows.
c                  ier_ict = -4   --> Illegal value for lfilict.
c                  ier_ict = -5   --> Zero row encountered.
c
c                  For ier_ict = -2 or -3, increase the value of lenpmx
c                  or decrease the value of lfil_ict if lenpmx cannot
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
      integer n, nnz, lenpmx, liw, lrw, ipar(20), ier_ict, ier_input
      integer ia(n+1), ja(nnz), jlu(lenpmx), perm(n), qperm(n),
     .        iwork(liw)
      real*8  a(nnz), plu(lenpmx), scale(n),
     .        rpar(20), rwork(lrw)
c     Local variables
      integer ireorder, iscale, iout, lfil_ict, i
      integer lmask, llevels, ljr, ljw, ljstrt, lwu, llnk, ljatmp,
     .        liatmp, lenrw, leniw
      real*8  tol_ict
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
c     check for lfil_ict value.
      if (ipar(6) .gt. 0) then
         lfil_ict = ipar(6)
      else
         lfil_ict = 20  ! by default allow up to 20 fill-in elmts.
      endif
c     check for negative rpar values.
      do i=1,20
         if (rpar(i) .lt. 0) then
            ier_input = i + 20
            return
         endif
      enddo
c     check for tol_ict value.
      if (rpar(1) .gt. 0) then
         tol_ict = rpar(1)
      else
         tol_ict = 0.0001  ! default drop tolerance of .0001
      endif
c-----------------------------------------------------------------------
c     Load pointers into work arrays iwork and rwork.
c-----------------------------------------------------------------------
c     iwork pointers
      liatmp = 1
      lmask = liatmp+n+1
      llevels = lmask + n
c     Overlay above memory, which is used in computing permutation, with
C       workspaces for ICT.
      ljw = 1
      ljstrt = ljw + 2*n
      ljr = ljstrt + n
      llnk = ljstrt + lfil_ict*n
      leniw = 3*n+2*lfil_ict*n
c     check for enough iwork space
      ipar(19) = leniw
      if (leniw .gt. liw) then
         ier_input = 41
         return
      endif
c     rwork pointers
      lwu = 1
      lenrw = lwu + n - 1
c     check for enough rwork space
      ipar(20) = lenrw
      if (lenrw .gt. lrw) then
         ier_input = 42
         return
      endif
c-----------------------------------------------------------------------
c     Call ict_factor to solve the linear system.
c-----------------------------------------------------------------------
      call ict_factor (n, nnz, lfil_ict, tol_ict,
     .     ia, ja, a, plu, jlu, iwork(ljstrt), iwork(llnk), lenpmx, 
     .     perm, qperm, scale,
     .     iwork(lmask), iwork(llevels), iwork(ljr), iwork(ljw), 
     .     rwork(lwu), iwork(liatmp),
     .     ireorder, iscale, iout, ier_ict)
      return
c-------------end-of-subroutine-ict_driver------------------------------
c-----------------------------------------------------------------------
      end
      subroutine ict_factor (n, nnz, lfil_ict, tol_ict,
     .     ia, ja, a, plu, jlu, jstart, link, lenpmx, perm, qperm,
     .     scale, mask, levels, jr, jw, wu, iatmp, 
     .     ireorder, iscale, iout, ier_ict)
c-----------------------------------------------------------------------
c This routine factors the matrix A defined by the ia, ja, a arrays.
c It uses the ICT code from sparskit to calculate the incomplete LDL^T.
c This routine is normally called by the ict_driver routine.
c
c NOTE:  When ireorder.ne.0 and/or iscale.ne.0, the original matrix is
c        returned in reordered and/or rescaled form.
c
c
c Date:             This is the version of 5-15-97.
c Author:           Peter N. Brown
c Organization:     Center for Applied Scientific Computing, LLNL
c Revision history:
c  3-21-97  pnb     Original version.
c  5-15-97  Andy Cleary  
c                   Workspace requirements reduced by calls to
c                   revised versions of SPARSKIT routines PERPHN and
c                   BFS, and rearrangement of use of preconditioning
c                   matrix for workspace during the reordering to reduce
c                   data copies.
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
c   lfil_ict   -  integer scalar.  lfil_ict is a fill-in
c                  parameter for ict preconditioner.  An
c                  additional lfil_ict values are allowed per
c                  row of L and U in the factorization.
c   tol_ict    -  real scalar.  tol_ict is a drop tolerance
c                  for dropping elements in ict when calculating 
c                  the incomplete LDL^T.
c   lenpmx      -  the maximum allowable size of the jlu and plu
c                  work arrays.  lenpmx must be greater than or
c                  equal to  max(n + 1 + lfil_ict*n, nnz).
c   plu,jlu     -  work space used for reordering the rows and columns
c                  of the matrix, and to store the incomplete LDL^T
c                  factorization of the matrix A in modified sparse
c                  row format.
c                  jlu must be an integer array of length lenpmx.
c                  plu must be a real array of length lenpmx.
c   perm,qperm  -  integer work arrays of length n used to hold the
c                  row and column permutation matrix (perm), and its
c                  inverse (qperm).  perm and qperm are only used if
c                  ireorder .eq. 0.
c   scale       -  real work array of length n used to hold the
c                  diagonal scaling factors of the matrix.
c                  scale is not used if iscale .eq. 0.
c   iatmp       -  integer work array of length n+1. (reordering)
c   mask        -  integer work array of length n. (reordering)
c   levels      -  integer work array of length n. (reordering)
c   jstart      -  integer work array of length n. (ict)
c   link        -  integer work array of length lfil_ict*n (ict)
c   jr          -  integer work array of length lfil_ict*n. (ict)
c   jw          -  integer work array of length 2*n. (ict)
c                  NOTE: iatmp, mask, and levels may be the same
c                  memory as the other integer work arrays.-AC.
c   wu          -  real work array of length n.  (ict)
c   ireorder    -  integer flag to determine if reordering of the
c                  rows and columns are desired.  If ireorder .ne. 0,
c                  a reverse Cuthill-McKee reordering of the rows
c                  and columns of the matrix A is done.
c                  Reordering is intended to reduce the amount of
c                  fill-in when performing the incomplete LDL^T
c                  factorization.
c   iscale      -  integer flag to determine if row scaling of
c                  the matrix A by its row norms is desired.  If
c                  iscale .ne. 0, the matrix A and rhs are
c                  scaled by the row norms of A prior to performing
c                  incomplete LDL^T factorization.  Incomplete LDL^T
c                  methods (with drop tolerances such as ict)
c                  can be sensitive to poorly scaled matrices.
c                  This can result in a poor preconditioner, i.e.,
c                  one that is not effective in reducing the
c                  number of pcg iterations for convergence.
c   iout        -  logical unit number for writing informational
c                  messages.
c
c On return,
c
c   plu,jlu     -  contains the incomplete LDL^T factorization of the
c                  matrix A in modified sparse row (MSR) format.
c   ier_ict     -  integer flag to indicate success of the ict
c                  routine in calculating the incomplete LDL^T
c                  factorization.
c                  ier_ict .eq. 0 indicates success.
c                  ier_ict .ne. 0 indicates failure.
c                  ier_ict >  0   --> Zero pivot encountered at 
c                                      step number ier_ict.
c                  ier_ict = -1   --> Error. input matrix may be 
c                                      wrong.  (The elimination
c                                      process has generated a
c                                      row in L or U with length > n.)
c                  ier_ict = -2   --> Matrix L overflows.
c                  ier_ict = -3   --> Matrix U overflows.
c                  ier_ict = -4   --> Illegal value for lfilict.
c                  ier_ict = -5   --> Zero row encountered.
c                  ier_ict = -6   --> lenpmx not big enough
c
c                  For ier_ict = -2 or -3, increase the value of lenpmx
c                  or decrease the value of lfil_ict if lenpmxc cannot
c                  be increased.
c
c-----------------------------------------------------------------------
      implicit none
      integer n, nnz, lenpmx
      integer ireorder, iscale, ier_ict, lfil_ict, iout
      real*8  a(nnz)
      integer ia (n+1), ja(nnz)
      real*8  plu(lenpmx)
      integer jlu(lenpmx), jr(lfil_ict*n), link(lfil_ict*n),jw(2*n)
      real*8 wu(n), tol_ict, scale(n)
      integer perm(n), qperm(n), mask(n), levels(n), jstart(n),
     .        jatmp(2*nnz), iatmp(n+1)
c
      real*8 dummy
      integer nfirst, nlev, i, j, maskval, itmp
c
      ier_ict = 0
c----------------------------------------------------------------------
c     Scale rows and columns of matrix if desired.
c----------------------------------------------------------------------
      if (iscale .ne. 0) then
         call rnrms (n,0,a,ja,ia,scale)
         do i = 1,n
            if (scale(i) .eq. 0.0) then
               ier_ict = -5
               if (iout .ne. 0) write(iout,*) 
     .           ' Nonzero ier value from ICT, ier_ict = ',ier_ict
               write(iout,9000)
               return
            else
               scale(i) = 1.0/sqrt(scale(i))
            endif
         enddo
         call diamua(n,0,a,ja,ia,scale,a,ja,ia)
         call amudia(n,0,a,ja,ia,scale,a,ja,ia)
      endif
c----------------------------------------------------------------------
c     Calculate reverse Cuthill-McKee reordering of rows and columns
c     of the matrix if ireorder is nonzero.  Otherwise, use the
c     natural ordering.  Uses the bfs routine from sparskit.
c----------------------------------------------------------------------
      if (ireorder .ne. 0) then
Cbegin removed code-AC 5/15/97
c        first construct the full nonzero structure of the matrix
c         call ssrcsr(0,0,n,a,ja,ia,2*nnz,dummy,jatmp,iatmp,
c     .        jstart,jlu,ier_ict)
Cbegin replacement code
c        construct transpose structure for input to reordering.
c        temporarily store in preconditioner space.
         itmp = 2
         call csrcsrt(n,a, ja, ia, plu, jlu, iatmp, itmp, ier_ict)
Cend
         if (ier_ict .ne. 0) then
            ier_ict = -6
            if (iout .ne. 0) write(iout,*) 
     .           ' Nonzero ier value from ICT, ier_ict = ',ier_ict
            write(iout,9000)
            return
         endif
         nfirst = 1
         perm(1)=0
         do i = 1,n
            mask(i) = 1
         enddo
         maskval = 1
         qperm(1) = 1
Cbegin removed code-AC 5/15/97
c         call bfs (n,ja,ia,nfirst,
c     .        perm,mask,maskval,qperm,levels,nlev)
Cbegin replacement code
         call perphn2 (n,ja,ia,jlu,iatmp,nfirst,
     .        perm,mask,maskval,nlev,qperm,levels)
Cend replacement code
         call reversp (n,qperm)
c     Calculate inverse of qperm  and put in perm.
         do i = 1,n
            perm(qperm(i))=i
         enddo
c     Permute rows and columns of A using perm. 
CBegin removed code-AC 5/15/97
c     Use ju,jlu,plu as
c     temporary workspace.
c         do i=1,n+1
c            jw(i)=ia(i)
c         enddo
c         do i=1,nnz
c            jlu(i)=ja(i)
c            plu(i)=a(i)
c         enddo
c         call dperm (n,plu,jlu,jw,a,ja,ia,perm,perm,1)
c         call csrcoo(n,3,nnz,a,ja,ia,nnz,plu,jatmp,jlu,ier_ict)
c         do i =1,nnz
c           swap row and column indices if col indx > row indx
c            if (jatmp(i) .lt. jlu(i)) then
c               itmp = jatmp(i)
c               jatmp(i) = jlu(i)
c               jlu(i) = itmp
c            endif
c         enddo
c         call coocsr(n,nnz,plu,jatmp,jlu,a,ja,ia)
CBegin replacement code
c     Use ju,jlu,plu as
c     temporary workspace.
         call dperm (n,a,ja,ia,plu,jlu,jw,perm,perm,1)
         call csrcoo(n,1,nnz,plu,jlu,jw,nnz,a,ja,ja,ier_ict)
         do i =1,nnz
c           Copy back into a, ja, and jlu, while
c             swapping row and column indices if col indx > row indx,
c             and swap definitions of ja and jlu so ja will be
c             in-place in call to coicsr
            a( i ) = plu( i )
            if (ja(i) .gt. jlu(i)) then
               itmp = ja(i)
               ja(i) = jlu(i)
               jlu(i) = itmp
            endif
         enddo
c        Form csr from coordinate format in-place.
         call coicsr(n,nnz,1,a,ja,jlu,jw)
c        Copy row pointers from jlu into ia
         do i=1,n+1
            ia(i)=jlu(i)
         enddo
c        Move diagonal elements to the end of each row as expected
c          by the factorization code
         do i=1, n
            do j=ia(i), ia(i+1)-2
               if( ja(j) .eq. i ) then
c                 Swap values
                  dummy = a( j )
                  a( j ) = a( ia(i+1)-1 )
                  a( ia(i+1)-1 ) = dummy
c                 Swap indices
                  ja( j ) = ja( ia(i+1)-1 )
                  ja( ia(i+1)-1 ) = i
               endif
            enddo
         enddo
CEnd replacement code
      endif
c----------------------------------------------------------------------
c     Call Incomplete Factorization Routine ICT.
c----------------------------------------------------------------------
      call ict (n,a,ja,ia,lfil_ict,tol_ict,plu,jlu,jstart,link,jr,
     .   wu,jw,lenpmx,ier_ict)
c     Return immediately if ict failed.
      if (ier_ict .ne. 0) then
         if (iout .ne. 0) write(iout,*) 
     .        ' Nonzero ier value from ICT, ier_ict = ',ier_ict
         write(iout,9000)
 9000    format(
     ./' ier_ict >  0   --> Zero pivot encountered at step number'
     ./'                     ier_ict.'
     ./' ier_ict = -1   --> Error. input matrix may be wrong.'
     ./'                 (The elimination process has generated a'
     ./'                  row in L length > n.)'
     ./' ier_ict = -2   --> Matrix L overflows.'
     ./' ier_ict = -4   --> Illegal value for lfilict.'
     ./' ier_ict = -5   --> Zero row encountered.'
     ./' ier_ict = -6   --> Factorization not positive definite.'
     ./'                    Matrix D has negative entries.'
     ./'                    The PCG iteration may not converge.'
     ./'    '
     ./' For ier_ict = -2, increase the value of lenpmx or'
     ./' decrease the value of lfil_ict if lenpmx cannot be'
     ./' increased.'
     .)
         return
      endif
c     Check to see if any elements of the main diagonal are negative.
c     If so, signal an error and return.
      do i = 1,n
         if (plu(i) .lt. 0.0) then
            ier_ict = -6
            if (iout .ne. 0) write(iout,*) 
     .        ' Nonzero ier value from ICT, ier_ict = ',ier_ict
            write(iout,9000)
            return
         endif
      enddo
c
      return
c-------------end-of-subroutine-ict_factor-----------------------------
c----------------------------------------------------------------------
      end
