      subroutine ict(n,a,ja,ia,lfil,droptol,alu,jlu,jstart,link,jr,
     *   w,jw,iwk,ierr)
c-----------------------------------------------------------------------
      implicit none 
      integer n, ja(*),ia(n+1),jlu(*),jw(2*n),lfil,iwk,ierr
      integer jstart(n),link(*),jr(*)
      real*8 a(*),alu(*),w(n),droptol
c----------------------------------------------------------------------*
c                      *** ICT preconditioner ***                      *
c                      Symmetric version of ILUT                       *
c                                                                      *
c  The approximate factorization L D L^T is produced, L is unit triang.*
c  Elements in L are dropped if they are less than droptol.            *
c  A maximum of lfil elements are stored per row of L.                 *
c
c      function [l,d] = ict(a)
c
c      n = size(a,2);
c      l = speye(n,n);
c      d = speye(n,n);
c      d(1,1) = a(1,1);
c
c      for j = 2:n
c
c         w = a(j,1:j-1);
c
c         for k = 1:j-1
c
c            drop w(k) if abs(w(k)/d(k,k)) is small
c
c            for i = k+1:j-1
c                w(i) = w(i) - w(k) * l(i,k);
c            end
c
c            w(k) = w(k) / d(k,k);
c
c         end
c
c         drop all but lfil largest elements in w
c
c         d(j,j) = a(j,j) - w * d(1:j-1,1:j-1) * w';
c         l(j,1:j-1) = w;
c
c      end
c                                                                      *
c----------------------------------------------------------------------*
c     REVISION HISTORY
c     02-10-97  Created from ILUT.
c----------------------------------------------------------------------*
c PARAMETERS                                                           
c-----------                                                           
c
c on entry:
c========== 
c n       = integer. The row dimension of the matrix A. The matrix 
c
c a,ja,ia = lower part of matrix stored in CSR format
c
c lfil    = integer. The fill-in parameter. Each row of L 
c           will have a maximum of lfil elements (excluding the 
c           diagonal element). lfil must be .ge. 0.
c
c droptol = real*8. Sets the threshold for dropping small terms in the
c           factorization. See above for details on dropping strategy.
c
c iwk     = integer. The lengths of arrays alu and jlu. If the arrays
c           are not big enough to store the ILU factorizations, ict
c           will stop with an error message. 
c
c On return:
c===========
c
c alu,jlu = the factorization  L D L^T  stored in MSR format.
c           The unit lower triangular L is stored in jlu and alu(n+2:..).
c           The diagonal (stored in alu(1:n)) is inverted. 
c
c ierr    = integer. Error message with the following meaning.
c           ierr  = 0    --> successful return.
c           ierr .gt. 0  --> zero pivot encountered at step number ierr.
c           ierr  = -1   --> Error. input matrix may be wrong.
c           ierr  = -2   --> The matrix L overflows the array al.
c           ierr  = -4   --> Illegal value for lfil.
c           ierr  = -5   --> zero row encountered.
c
c work arrays:
c=============
c jw      = integer work array of length 2*n.
c w       = real work array of length n 
c jstart(n),link(lfil*n),jr(lfil*n) = companion structure
c  
c----------------------------------------------------------------------
c w, jw (1:n) store the working array
c jw(n+1:2n)  stores nonzero indicators
c 
c----------------------------------------------------------------------* 
      integer ju0,k,j1,j2,j,ii,lenl,jj,jrow,jpos,len 
      real*8 t, abs, s, fact 
c----------------------------------------------------------------------* 

      if (lfil .lt. 0) goto 998

c     initialize ju0 (points to next element to be added to alu,jlu)
c     and pointer array.
c
      ju0 = n+2
      jlu(1) = ju0
c
c     initialize nonzero indicator array
c     and companion structure
c
      do 1 j=1,n
         jw(n+j) = 0
         jstart(j) = 0
 1    continue
c-----------------------------------------------------------------------
c     beginning of main loop.
c-----------------------------------------------------------------------
      do 500 ii = 1, n
         j1 = ia(ii)
         j2 = ia(ii+1) - 1
c     
c     unpack L-part of row of A in array w 
c     diagonal element stored in alu(ii)
c     
         lenl = 0
         alu(ii) = 0.0
c
         do 170  j = j1, j2
            k = ja(j)
            t = a(j)
            if (k .lt. ii) then
               lenl = lenl+1
               jw(lenl) = k
               w(lenl) = t
               jw(n+k) = lenl
            else if (k .eq. ii) then
               alu(ii) = t
            else
	       goto 995
            endif
 170     continue
         jj = 0
         len = 0 
c     
c     eliminate previous rows
c     
 150     jj = jj+1
         if (jj .gt. lenl) goto 160
c-----------------------------------------------------------------------
c     in order to do the elimination in the correct order we must select
c     the smallest column index among jw(k), k=jj+1, ..., lenl.
c-----------------------------------------------------------------------
         jrow = jw(jj)
         k = jj
c     
c     determine smallest column index jrow
c     
         do 151 j=jj+1,lenl
            if (jw(j) .lt. jrow) then
               jrow = jw(j)
               k = j
            endif
 151     continue
c
         if (k .ne. jj) then
c     exchange in jw
            j = jw(jj)
            jw(jj) = jw(k)
            jw(k) = j
c     exchange in jr
            jw(n+jrow) = jj
            jw(n+j) = k
c     exchange in w
            s = w(jj)
            w(jj) = w(k)
            w(k) = s
         endif
c
c     zero out element in row by setting jw(n+jrow) to zero.
c     
         jw(n+jrow) = 0
c
c     get the multiplier for row to be eliminated (jrow)
c     and apply drop tolerance.
c     
         fact = w(jj)
         if (abs(fact/alu(jrow)) .le. droptol) goto 150
c     
c     combine current row and row jrow
c     loop over companion structure for column jrow
c
	 k = jstart(jrow)
         do while (k .ne. 0)

            s = fact*alu(k)
            j = jr(k)
            jpos = jw(n+j)

            if (j .ge. ii) then
                  ! dealing with upper part
		  goto 995
	    else

               ! dealing  with lower part
               if (jpos .eq. 0) then
 
                  ! this is a fill-in element
                  lenl = lenl+1
                  if (lenl .gt. n) goto 995
                  jw(lenl) = j
                  jw(n+j) = lenl
                  w(lenl) = - s
               else

                  ! this is not a fill-in element 
                  w(jpos) = w(jpos) - s
               endif
            endif

            k = link(k)
         enddo
c     
c     store this pivot element -- from left to right -- no danger of
c     overlap with the working elements in L (pivots). 
c     
         len = len+1
         w(len) = fact*alu(jrow)
         jw(len) = jrow
         goto 150
 160     continue
c     
c     update L-matrix
c     
         lenl = len
         len = min0(lenl,lfil)
c     
c     sort by quick-split
c
         call qsplit (w,jw,lenl,len)
c     
c     store inverse of diagonal element
c     
         do k = 1, len
            alu(ii) = alu(ii) - w(k)*w(k)/alu(jw(k))
         enddo

         if (alu(ii) .eq. 0.0) then
            ierr = ii
            return
         endif
c     
         alu(ii) = 1.d0/ alu(ii) 
c
c     store L-part
c 
         if (len + ju0 .gt. iwk) goto 996
         do 204 k=1, len 
            alu(ju0) =  w(k)
            jlu(ju0) = jw(k)
            ju0 = ju0+1
 204     continue
c     
c     update pointer to beginning of next row
c     
         jlu(ii+1) = ju0
c
c     update linked-list companion structure
c
         call cadd(ii, jlu, jlu, jstart, link, jr)
c-----------------------------------------------------------------------
c     end main loop
c-----------------------------------------------------------------------
 500  continue
      ierr = 0
      return
c
c     incomprehensible error. Matrix must be wrong.
c     
 995  ierr = -1
      return
c     
c     insufficient storage in L.
c     
 996  ierr = -2
      return
c     
c     illegal lfil entered.
c     
 998  ierr = -4
      return
c     
c     zero row encountered
c     
 999  ierr = -5
      return
c----------------end-of-ilut--------------------------------------------
c-----------------------------------------------------------------------
      end
c-----------------------------------------------------------------------
      subroutine cadd(ii, ia, ja, jstart,link,jr)
      implicit none
      integer ii, ia(*), ja(*), jstart(*), link(*), jr(*)
c-----------------------------------------------------------------------
c     Add row to linked-list companion structure.
c-----------------------------------------------------------------------
c     On entry:
c--------------
c     ii      = number of row being added
c     ia      = pointers to beginnings of rows
c     ja      = pointers to values in row
c     jstart,link,jr = linked-list companion structure
c
c     On return:
c--------------
c     jstart,link,jr = updated linked-list companion structure
c
c     Notes:
c--------------
c     Do not forget to initialize jstart(1:n) = 0 before first call.
c
c-----------------------------------------------------------------------
      integer k, j, last

      do 10 k = ia(ii),ia(ii+1)-1
         j = ja(k)
         last   = jstart(j)
         link(k) = last
         jstart(j) = k
         jr(k) = ii
 10   continue
      return
      end
c-----------------------------------------------------------------------
      subroutine lmult (n,am,jm,im,x,y)
      implicit none
      integer n,jm(*),im(*)
      real*8 am(*),y(n),x(n)
c-----------------------------------------------------------------------
c     y = A x
c     A is symmetric.
c     Only the lower triangular part is stored (by rows), with diagonal
c     element stored last in each row.
c-----------------------------------------------------------------------
      real*8 t
      integer ii,k

c-----diag part
      do ii = 1, n
         y(ii) = am(im(ii+1)-1) * x(ii)
      enddo

c-----upper part
      do ii=2,n
         t = x(ii)
         do k=im(ii),im(ii+1)-2
            y(jm(k)) = y(jm(k)) + t*am(k)
         enddo
      enddo

c-----lower part
      do ii = 2,n
         t = y(ii)
         do k=im(ii),im(ii+1)-2
            t = t + am(k)*x(jm(k))
         enddo
         y(ii) =  t
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine lusol_ict (n,alu,jlu,rhs,sol)
c
c     Solve with L D L^T produced by ict.
c
      implicit none
      integer n,jlu(*)
      real*8 alu(*),rhs(n),sol(n)
c-----------------------------------------------------------------------
      real*8 t
      integer ii,k
c
c     lower
c
      do ii=1, n
         t = rhs(ii)
         do k=jlu(ii),jlu(ii+1)-1
            t = t - alu(k)*sol(jlu(k))
         enddo
         sol(ii) =  t
      enddo
c
c     diag
c
      do ii=1, n
         sol(ii) = sol(ii)*alu(ii)
      enddo
c
c     lower transpose
c
      do ii=n,2,-1
         t = sol(ii)
         do k=jlu(ii),jlu(ii+1)-1
            sol(jlu(k)) = sol(jlu(k)) - t*alu(k)
         enddo
      enddo
      return
      end
c-----------------------------------------------------------------------



