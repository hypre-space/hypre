      subroutine dshftab2(n,    sigmar, sigmai, aptr,  aind,  aval, 
     &                    bptr, bind,   bval,   abptr, abind, abval,
     &                    nnz,  work)
c
c     === Perform B = A-sigma*B ===
c
c     n      (input)  dimension of the matrix.
c     sigma  (input)  shift to be applied.
c     aptr   (input)  column pointers of the stiffness matrix.
c     aind   (input)  row indices of the stiffness matrix.
c     aval   (input)  nonzero values of the stiffness matrix.
c     bptr   (input)  column pointers of the mass matrix.
c     bind   (input)  row indices of the mass matrix.
c     bval   (input)  nonzero values of the mass matrix.
c     abptr  (output) column pointers of A - simga*B.
c     abind  (output) row indices of A - simga*B.
c     abval  (output) nonzero values of A - simga*B.
c     nnz    (output) number of nonzeros in A-sigma*B
c     work   (work)   work space
c
      integer            n,         aptr(n+1), aind(*), 
     &                   bptr(n+1), bind(*),   abptr(n+1),
     &                   abind(*)
      double precision   sigmar,    sigmai,    aval(*),
     &                   bval(*)
      complex*16         abval(*),  work(n)
c
c     === Local Scaler ===
c
      integer            nnz, jcol, irow, i, k1, k2, nzloc
      double precision   zero
      parameter          (zero = (0.0d0, 0.0d0))
c
c     === Executable statements ===
c
      nnz = 0
      do jcol = 1, n 
c
c        === clean up the buffer space ===
c
         do irow = 1, n
	   work(irow) = zero
         end do
c
         k1 = aptr(jcol)
         k2 = aptr(jcol+1) - 1
         do i = k1, k2
            irow = aind(i)
            work(irow) = dcmplx(aval(i))
         end do
c         
         k1 = bptr(jcol)
         k2 = bptr(jcol+1) - 1
         do i = k1, k2
            irow = bind(i)
            work(irow) = work(irow) - dcmplx(sigmar, sigmai)
     &                              * dcmplx(bval(i))
         end do
c
         abptr(jcol) = nnz+1
         nzloc = 0
         do i = 1, n
            if (work(i) .ne. zero) then
               nzloc = nzloc + 1
               nnz = nnz + 1
               abind(nnz) = i
               abval(nnz) = work(i)
            endif 
         end do
         if (nzloc .eq. 0) then
c
c           === a zero column, insert a 0.0 on the diagonal ===
c
            nnz = nnz + 1
            abind(nnz) = jcol
            abval(nnz) = zero
         end if
c
      end do
c
      abptr(n+1) = nnz+1
c
 9000 continue
      return
      end
     





