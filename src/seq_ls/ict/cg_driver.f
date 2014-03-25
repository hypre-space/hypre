      subroutine cg_driver (n, nnz, ia, ja, a, rhs, sol, ipar, rpar,
     .     lenpmx, plu, jlu, perm, qperm, scale,
     .     rwork, lrw, ier_pcg, ier_input)
c-----------------------------------------------------------------------
c This routine solves the linear system defined by the ia, ja, a and rhs
c arrays, and returns the solution in the array sol.  It uses a
c preconditioned pcg iteration, with preconditioner computed via an
c incomplete LU factorization.  The ICT code from sparskit is assumed
c to have been used to calculate the incomplete LU. 
c
c The routine ict_driver must be called prior to calling this routine,
c since it precalculates the preconditioner specific arrays jlu,
c plu, perm, qperm, and scale.  Specifically, this routine solves
c the linear system 
c
c                      A*C^{-1}*(C*x) = b,
c
c where the preconditioner C is defined by first noting that A can be
c rewritten as
c
c         A = S^{-1} * Q * ( P*(S*A*S)*Q )* P * S^{-1},
c
c where S is a diagonal scaling matrix calculated from the inverses of
c the row norms of A, and P and Q  are permutation matrices where the
c permutations are obtained via a reverse Cuthill-McKee reordering of 
c the rows and columns of S*A*S with Q = P^{-1} = P^T.  An incomplete 
c LDL^T factorization of the reordered  and rescaled matrix P*(S*A*S)*Q
c is then performed (via a prior call to the ict_driver routine) to give
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
c 
c-----------------------------------------------------------------------
c
c Date:             This is the version of 3-21-97.
c Author:           Peter N. Brown
c Organization:     Center for Applied Scientific Computing, LLNL
c Revision history: 
c 03-21-97  pnb     Original version.
c
c-----------------------------------------------------------------------
c On input,
c
c   n           -  integer scalar.  n is the number of rows and columns
c                  in the matrix A.
c   nnz         -  integer scalar.  nnz is the number of nonzeros in
c                  the matrix A.
c   ia,ja,a     -  sparse storage arrays for the matrix A in symmetric
c                  sparse row format.
c                  ia must be an integer array of length n+1.
c                  ja must be an integer array of length nnz.
c                  ja must be a real array of length nnz.
c                  Important: the diagonal element of each row must be
c                    the last entry of that row.-AC.
c   rhs         -  real array of length n.
c                  rhs contains the right hand side of the linear 
c                  system.
c   sol         -  real array of length n.
c                  sol contains an initial guess to the solution of the
c                  linear system.
c   ipar        -  integer array of length 20.  ipar contains
c                  optinal input parameters for ict_driver.  A positive
c                  ipar(i) value signals a non-default input value.
c                  NOTE:  The values of ipar must be consistent with
c                         those used on the call to ict_driver.
c       ipar(1) =  ireorder, row and column reordering flag
c                  used to determine if reordering of the
c                  rows and columns are desired.  If ireorder .ne. 0,
c                  a reverse Cuthill-McKee reordering of the rows
c                  and columns of the matrix A and rhs is done.
c                  Reordering is intended to reduce the amount of
c                  fill-in when performing the incomplete LU
c                  factorization.  Default value is ireorder=0.
c       ipar(2) =  iscale,  flag used to determine if row and column 
c                  scaling of the matrix A  is desired.  If
c                  iscale=0, no scaling of A is performed,
c                  iscale=1, row & column scaling of A is performed,
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
c       ipar(6) -  not used.
c       ipar(7) -  not used.
c       ipar(8) =  maxits, the maximum allowable total number of pcg
c                  iterations.  Default value is maxits=100.
c   rpar        -  real array of length 20.  rpar contains
c                  optinal input parameters for ict_driver.  A
c                  positive rpar(i) value signals a non-default input
c                  value.
c                  NOTE:  The values of rpar must be consistent with
c                         those used on the call to ict_driver.
c       rpar(1) -  not used.
c       rpar(2) =  tol_linsys, the linear system stopping tolerance
c                  for the pcg iteration.  The pcg iteration is
c                  stopped when
c
c                    norm(rhs - A*sol) .le. tol_linsys * norm(rhs),
c
c                  where norm() represents the Euclidean (or l2) norm
c                  of a vector.
c                  Default value is tol_linsys=0.00001.
c   lenpmx      -  integer scalar.  lenpmx is the length of the
c                  preconditioner arrays jlu and plu.
c   jlu,plu     -  sparse storage arrays for the preconditioner P
c                  stored in modified symmetric sparse row format.  
c                  The user must call ict_driver to calculate the
c                  preconditioner before calling this routine, and 
c                  the arrays jlu, plu contain the preconditioner
c                  ict_driver calculates.
c                  jlu must be an integer array of length lenpmx.
c                  plu must be a real array of length lenpmx.
c   perm        -  integer array of length n for the permutation.
c                  Only applicable if ipar(1) .ne. 0.
c   qperm       -  integer array of length n for the inverse of perm.
c                  Only applicable if ipar(1) .ne. 0.
c   scale      -  real array of length n for diagonal scaling factors
c                  containing the inverse of the row norms of A.
c   scale      -  real array of length n for diagonal scaling factors
c                  containing the inverse of the column norms of the
c                  row-scaled A.
c                  Only applicable if ipar(2)= .ne. 0.
c   rwork       -  real work array of length lrw.
c   lrw         -  integer scalar containing the length of the rwork
c                  array.  lrw must be greater than or equal to 6*n.
c
c On return,
c
c   rhs         -  real array of length n.
c                  rhs contains the right hand side of the linear 
c                  system.
c   sol         -  contains the approximate solution of the linear
c                  system, if there were no errors.  Otherwise, sol
c                  is undefined.
c   jlu,plu     -  sparse storage arrays for the preconditioner P
c                  stored in modified sparse row format.
c   perm        -  integer array of length n containing the
c                  permutation. Only applicable if ipar(1) .eq. 0.
c   qperm       -  integer array of length n containing the inverse
c                  permutation. Only applicable if ipar(1) .eq. 0.
c   scale      -  real array of length n for diagonal scaling factors
c                  containing the inverse of the row norms of A.
c   scale      -  real array of length n for diagonal scaling factors
c                  containing the inverse of the column norms of the
c                  row-scaled A.
c                  Only applicable if ipar(2)= .ne. 0.
c   ipar(20)    -  minimum needed length of rwork.
c   ier_pcg     -  integer flag to indicate if the pcg iteration
c                  converged.
c                  ier_pcg .eq. 0 indicates success.
c                  ier_pcg .ne. 0 indicates failure.
c                  ier_pcg =-1 --> convergence not achieved in
c                                    maxits iterations.
c   ier_input   -  integer flag to indicate if a value in ipar or rpar
c                  was illegal.  ier_input between 1 and 20 means
c                  ipar(ier_input) was illegal, while ier_input
c                  between 21 and 40 means rpar(ier_input-20) was
c                  illegal.  
c
c                  A value of ier_input=42 means that the declared
c                  length of rwork, lrw, was not large enough, and
c                  ipar(20) contains the needed length.
c
c-----------------------------------------------------------------------
      implicit none
      integer n, nnz, lenpmx, lrw, ipar(20), ier_pcg, ier_input
      integer ia(n+1), ja(nnz), jlu(lenpmx), perm(n), qperm(n)
      real*8  a(nnz), rhs(n), sol(n), plu(lenpmx), scale(n),
     .        rpar(20), rwork(lrw)
c     Local variables
      integer ireorder, iscale, iout, maxits, i, idx_in, idx_out
      integer lenrw, ipar_pcg(20), ltmp, lencgw, iter
      real*8  tol_linsys, fpar_pcg(20)
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
c     check for maxits value.
      if (ipar(8) .gt. 0) then
         maxits = ipar(8)
      else
         maxits = 100  ! default of 100 maximum pcg iterations.
      endif
c     check for negative rpar values.
      do i=1,20
         if (rpar(i) .lt. 0) then
            ier_input = i + 20
            return
         endif
      enddo
c     check for tol_linsys value.
      if (rpar(2) .gt. 0) then
         tol_linsys = rpar(2)
      else
         tol_linsys = 0.00001  ! default of .00001 for pcg convergence
      endif
c-----------------------------------------------------------------------
c     Load pointers into work array rwork.
c-----------------------------------------------------------------------
      lenrw = 6*n
      lencgw = lenrw - n
c     temporary vector of length n to hold temporary results.
      ltmp = lenrw - n + 1
c     check for enough rwork space
      ipar(20) = lenrw
      if (lenrw .gt. lrw) then
         ier_input = 42
         return
      endif
c----------------------------------------------------------------------
c     Call pcg with reverse communication to solve the preconditioned
c     linear system.
c----------------------------------------------------------------------
      ipar_pcg(1) = 0	       ! always 0 to start the iterative solver
      ipar_pcg(2) = 1	       ! preconditioning if value nonzero
      ipar_pcg(3) = 2          ! use convergence test scheme 2
      ipar_pcg(4) = lencgw     ! length of work vector for PCG
      ipar_pcg(6) = maxits     ! use at most maxits iterations
      fpar_pcg(1) = tol_linsys ! relative tolerance
      fpar_pcg(2) = 0.0        ! absolute tolerance, set to zero
      fpar_pcg(11) = 0.0       ! clearing the FLOPS counter
      iter = -1
 10   continue
      call pcg(n,rhs,sol,ipar_pcg,fpar_pcg,rwork)
      idx_in = ipar_pcg(8)
      idx_out = ipar_pcg(9)
      if ((ipar_pcg(1) .eq. 1) .and. (iout .ne. 0)
     .     .and. (iter .ge .0)) then
         write(iout,9000) iter, fpar_pcg(5)
 9000    format(' Iteration = ',i5,' Current res. norm = ',e16.8)
      endif
      if (ipar_pcg(1).eq.1) then
         iter = iter + 1
         call lmult(n,a,ja,ia,rwork(idx_in),rwork(idx_out))
         goto 10
      else if (ipar_pcg(1).eq.2) then
c        preconditioner solve
c        copy input vector to temporary location.
         do i = 1,n
            rwork(ltmp+i-1) = rwork(idx_in+i-1)
         enddo
c        code for row scaling.
         if (iscale .ne. 0) then
            do i = 1,n
               rwork(ltmp+i-1) = scale(i)*rwork(ltmp+i-1)
            enddo
         endif
         if (ireorder .ne. 0) then
            call dvperm (n,rwork(ltmp),perm) 
         endif
         call lusol_ict(n,plu,jlu,rwork(ltmp),rwork(idx_out))
         if (ireorder .ne. 0) then
            call dvperm (n,rwork(idx_out),qperm) 
         endif
c        code for column scaling.
         if (iscale .ne. 0) then
            do i = 1,n
               rwork(idx_out+i-1) = scale(i)*rwork(idx_out+i-1)
            enddo
         endif
         goto 10
      else if (ipar_pcg(1).gt.0) then
c         ipar_pcg(1) is an unspecified code
         print *, ' ipar_pcg(1) is an unspecified code'
         stop
      else
c         the iterative solver terminated with code = ipar_pcg(1)
      endif
      ier_pcg = ipar_pcg(1)
      if (iout .ne. 0) write(iout,9000) iter, fpar_pcg(5)
      if (ier_pcg .lt. 0) then
         if (iout .ne. 0) then
             write(iout,*)
     .           ' Nonzero ier value from pcg, ier = ',ier_pcg
            write(iout,9001)
         endif
 9001    format(
     ./' ier_pcg = -1  --> convergnce not achieved in maxits',
     ./'                     iterations.'
     .)
            return
      endif
      return
c-------------end-of-subroutine-pcg_driver------------------------------
c-----------------------------------------------------------------------
      end
