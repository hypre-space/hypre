      subroutine dnzcnt(n,   aptr,  aind, aval, bptr, bind, bval,
     &                  nnz, dwork, iwork)
c
c     === Count NNZ of B = A-sigma*B ===
c
c     n      (input)  dimension of the matrix.
c     aptr   (input)  column pointers of the A matrix.
c     aind   (input)  row indices of the A matrix.
c     aval   (input)  nonzero values of the A matrix.
c     bptr   (input)  column pointers of the B matrix.
c     bind   (input)  row indices of the B matrix.
c     bval   (input)  nonzero values of the B matrix.
c     
c
      integer            n, aptr(n+1), aind(*), bptr(n+1), bind(*),
     &                   iwork(n)
      double precision   aval(*), bval(*), dwork(n)
c
c     === Local Scaler ===
c
      integer            nnz, j, irow, i, k1, k2, nzloc
      double precision   zero, one
      parameter          (zero = 0.0d0, one = 1.0d0)
c
c     === Executable statements ===
c
      do i = 1, n
         dwork(i) = zero
      end do
c
      nnz = 0
      do j = 1, n 
         nzloc = 0
         k1 = aptr(j)
         k2 = aptr(j+1) - 1
         do i = k1, k2
            nzloc = nzloc + 1
            irow = aind(i)
            dwork(irow)  = one
            iwork(nzloc) = irow
         end do
c
c        === check to see if B's structure matches A's ===
c         
         k1 = bptr(j)
         k2 = bptr(j+1) - 1
         do i = k1, k2
            irow = bind(i)
            if (dwork(irow) .eq. zero) then
               dwork(irow) = one
               nzloc = nzloc + 1
               iwork(nzloc) = irow
            end if
         end do
c
         nnz = nnz+nzloc
c
c        === cleanup dwork ===
c
         do i = 1, nzloc
            irow = iwork(i)
            dwork(irow) = zero
         end do
      end do
      return
      end
     





