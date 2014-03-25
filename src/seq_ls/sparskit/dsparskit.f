      subroutine atob (n, a, ja, ia, b, jb, ib)

c ... Copy matrix a,ja,ia to b,jb,ib.  Both matrices are in
c     compressed sparse row format.

      implicit none

c ... Input arguments:
      integer n
      real*8 a(*)
      integer ja(*)
      integer ia(n+1)

c ... Output arguments:
      real*8 b(*)
      integer jb(*)
      integer ib(n+1)

c ... Local variable:
      integer i

      do i = 1, ia(n+1) - 1
         b(i) = a(i)
         jb(i) = ja(i)
      enddo
      do i = 1, n+1
         ib(i) = ia(i)
      enddo

      return
      end
c  end of atob
c-----------------------------------------------------------------------
       subroutine ilut (n,a,ja,ia,lfil,tol,alu,jlu,ju,iwk,
     *                  wu,wl,jr,jwl,jwu,ierr) 
c----------------------------------------------------------------------- 
       implicit none
       integer n, ju0, j, ii, j1, j2, k, lenu, lenl, jj, nl, jrow
       integer jpos, len
       real*8 tnorm, t, s, fact
       real*8 a(*), alu(*), wu(n+1), wl(n), tol
       integer ja(*),ia(n+1),jlu(*),ju(n),jr(n), jwu(n), 
     *      jwl(n), lfil, iwk, ierr
c----------------------------------------------------------------------* 
c                      *** ILUT preconditioner ***                     *
c                      ---------------------------                     *
c      incomplete LU factorization with dual truncation mechanism      *
c      VERSION 2 : sorting  done for both L and U.                     *
c                                                                      *
c Bug Fix:  Version of 2-25-93.                                        *
c                                                                      *
c----------------------------------------------------------------------*
c---- coded by Youcef Saad May, 5, 1990. ------------------------------* 
c---- Dual drop-off strategy works as follows.                         *
c                                                                      *
c     1) Theresholding in L and U as set by tol. Any element whose size*
c        is less than some tolerance (relative to the norm of current  *
c        row in u) is dropped.                                         *
c                                                                      *
c     2) Keeping only the largest lfil elements in L and the largest   *
c        lfil elements in U.                                           *
c                                                                      *
c Flexibility: one can use tol=0 to get a strategy based on keeping the*
c largest elements in each row of L and U. Taking tol .ne. 0 but lfil=n*
c will give the usual threshold strategy (however, fill-in is then     *
c impredictible).                                                      *
c                                                                      *
c----------------------------------------------------------------------*
c PARAMETERS
c-----------
c
c on entry:
c========== 
c n       = integer. The dimension of the matrix A.
c
c a,ja,ia = matrix stored in Compressed Sparse Row format.
c
c lfil    = integer. The fill-in parameter. Each row of L and
c           each row of U will have a maximum of lfil elements 
c           in addition to the original number of nonzero elements.
c           Thus storage can be determined beforehand.
c           lfil must be .ge. 0. 
c
c iwk     = integer. The minimum length of arrays alu and jlu
c 
c On return:
c=========== 
c
c alu,jlu = matrix stored in Modified Sparse Row (MSR) format containing
c           the L and U factors together. The diagonal (stored in
c           alu(1:n) ) is inverted. Each i-th row of the alu,jlu matrix 
c           contains the i-th row of L (excluding the diagonal entry=1) 
c           followed by the i-th row of U.  
c                                                                        
c ju      = integer array of length n containing the pointers to        
c           the beginning of each row of U in the matrix alu,jlu. 
c                                                                       
c ierr    = integer. Error message with the following meaning.
c           ierr  = 0    --> successful return.
c           ierr .gt. 0  --> zero pivot encountered at step number ierr.
c           ierr  = -1   --> Error. input matrix may be wrong.
c                            (The elimination process has generated a 
c                            row in L or U whose length is .gt.  n.)
c           ierr  = -2   --> The matrix L overflows the array al.
c           ierr  = -3   --> The matrix U overflows the array alu.
c           ierr  = -4   --> Illegal value for lfil.
c           ierr  = -5   --> zero row encountered. 
c           
c work arrays:
c=============
c jr,jwu,jwl 	  = integer work arrays of length n.
c wu, wl          = real work arrays of length n+1, and n resp.
c
c Notes:
c ------ 
c A must have all nonzero diagonal elements.
c----------------------------------------------------------------------- 
        if (lfil .lt. 0) goto 998
c-------------------------------
c initialize ju0 (points to next element to be added to alu,jlu)
c and pointer.
c----------------------------------------------------------------------- 
	ju0 = n+2
	jlu(1) = ju0
c
c  integer double pointer array. 
c 
	do 1 j=1, n
           jr(j)  = 0
 1      continue
c-----------------------------------------------------------------------
c     beginning of main loop. 
c-----------------------------------------------------------------------
	do 500 ii = 1, n
           j1 = ia(ii)
           j2 = ia(ii+1) - 1
           tnorm = 0.
           do 501 k=j1,j2
              tnorm = tnorm+abs(a(k))
 501       continue
           if (tnorm .eq. 0.) goto 999 
           tnorm = tnorm/(j2-j1+1)
c
c     unpack L-part and U-part of row of A in arrays wl, wu 
c
           lenu = 1
           lenl = 0
           jwu(1) = ii
           wu(1) = 0.0 
           jr(ii) = 1
c
c     unpack lower and upper parts of row ii, in jwl-wl and 
c     jwu-wu compressed rows respectively. Ignore element if small
c          
       do 170  j = j1, j2
          k = ja(j)
          t = a(j)
          if (abs(t) .lt. tol*tnorm .and. k .ne. ii) goto 170
          if (k .lt. ii) then
             lenl = lenl+1
             jwl(lenl) = k
             wl(lenl) = t 
             jr(k) = lenl
          else if (k .eq. ii) then 
             wu(1) = t
          else 
             lenu = lenu+1
             jwu(lenu) = k
             wu(lenu) = t 
             jr(k) = lenu
          endif
 170   continue
       tnorm = tnorm/(j2-j1+1)
c     lenl0 = lenl
c     lenu0 = lenu
c-----------------------------------------------------------------------
       jj = 0
       nl = 0
c
c     eliminate previous rows
c     
 150    jj = jj+1
        if (jj .gt. lenl) goto 160
c------------------------------------------------------------------- 
c     in order to do the elimination in the correct order we need to 
c     exchange the current row number with the one that has
c     smallest column number, among jj,jj+1,...,lenl.
c------------------------------------------------------------------- 
        jrow = jwl(jj)
        k = jj
c     
c     determine smallest column index
c
        do 151 j=jj+1,lenl
           if (jwl(j) .lt. jrow) then
              jrow = jwl(j)
              k = j
           endif 
 151    continue
c     
c     exchange in jwl
c
        if (k .ne. jj) then
           j = jwl(jj)
           jwl(jj) = jwl(k) 
           jwl(k) = j
c     
c     exchange in jr
c
           jr(jrow) = jj
           jr(j) = k
c
c     exchange in wl 
c
           s = wl(jj) 
           wl(jj) = wl(k) 
           wl(k) = s
        endif
c
        if (jrow .ge. ii) goto 160
c
c     get the multiplier for row to be eliminated: jrow
c     
        fact = wl(jj)*alu(jrow)
c     zero out element in row by setting jr(jrow) = 0
        jr(jrow) = 0
        if (abs(fact)*wu(n+2-jrow) .le. tol*tnorm) goto 150
c
c     combine current row and row jrow
c     
        do 203 k = ju(jrow), jlu(jrow+1)-1
           s = fact*alu(k)      
           j = jlu(k)
           jpos = jr(j)
c
c if fill-in element is small then disregard:
c     
           if (abs(s) .lt. tol*tnorm .and. jpos .eq. 0) goto 203
           if (j .ge. ii) then
c
c     dealing with upper part.
c
              if (jpos .eq. 0) then
c     this is a fill-in element
                 lenu = lenu+1
                 if (lenu .gt. n) goto 995
                 jwu(lenu) = j
                 jr(j) = lenu
                 wu(lenu) = - s 
              else
c     no fill-in element --
                 wu(jpos) = wu(jpos) - s
              endif
           else 
c
c     dealing with lower part.
c
              if (jpos .eq. 0) then
c     this is a fill-in element
                 lenl = lenl+1
                 if (lenl .gt. n) goto 995
                 jwl(lenl) = j
                 jr(j) = lenl
                 wl(lenl) = - s 
              else
c     no fill-in element --
                 wl(jpos) = wl(jpos) - s
              endif
           endif
 203	continue
        nl = nl+1
        wl(nl) = fact
        jwl(nl)  = jrow
	goto 150
c
c     update l-matrix
c
c 160    len = min0(nl,lenl0+lfil) 
 160    len = min0(nl,lfil) 
  	call qsplit (wl,jwl,nl,len) 
c     
        do 204 k=1, len
           if (ju0 .gt. iwk) goto 996
           alu(ju0) =  wl(k)
           jlu(ju0) =  jwl(k)
           ju0 = ju0+1	
 204    continue
c
c     save pointer to beginning of row ii of U 
c 
        ju(ii) = ju0 
c
c     reset double-pointer jr to zero (L-part - except first 
c     jj-1 elements which have already been reset)
c
	do 306 k= jj, lenl
           jr(jwl(k)) = 0
 306	continue
c       len = min0(lenu,lenu0+lfil)
       len = min0(lenu,lfil) 
       call qsplit (wu(2), jwu(2), lenu-1,len) 
c
c     update u-matrix
c     
       t = abs(wu(1))                 
       if (len + ju0 .gt. iwk) goto 997
       do 302 k=2, len
          jlu(ju0) = jwu(k)
          alu(ju0) = wu(k)
          t = t + abs(wu(k) ) 
          ju0 = ju0+1 
 302   continue
c     
c     save norm (in fact the average abs value) in wu (backwards) 
c       
       wu(n+2-ii) = t / (len+1)
c     
c     store inverse of diagonal element of u
c
       if (wu(1) .eq. 0.0) wu(1) = (0.0001 + tol)*tnorm
c     
       alu(ii) = 1.0 / wu(1) 
c
c     update pointer to beginning of next row of U.
c     
       jlu(ii+1) = ju0
c
c     reset double-pointer jr to zero (U-part) 
c
       do 308 k=1, lenu
          jr(jwu(k)) = 0
 308   continue
c-----------------------------------------------------------------------
c     end main loop
c-----------------------------------------------------------------------
 500  continue
      ierr = 0
      return
c     
c     zero pivot :
c     
c 900    ierr = ii
c        return
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
c     insufficient storage in U.
c     
 997  ierr = -3
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
c----------------end of ilut  ----------------------------------------- 
c-----------------------------------------------------------------------
      end
c----------------------------------------------------------------------
        subroutine qsplit  (a, ind, n, ncut) 
        implicit none
        integer n, mid, j
        real*8 a(n) 
        integer ind(n), ncut 
c-----------------------------------------------------------------------
c     does a quick-sort split of a real array.
c     on input a(1:n). is a real array
c     on output a(1:n) is permuted such that its elements satisfy: 
c     a(i) .le. a(ncut) for i .le. ncut and
c     a(i) .ge. a(ncut) for i .ge. ncut 
c     ind(1:n) is an integer array which permuted in the same way as a(*).
c-----------------------------------------------------------------------
        real*8 tmp, abskey 
        integer itmp, first, last 
c-----
        first = 1
        last = n
        if (ncut .lt. first .or. ncut .gt. last) return
c     
c     outer loop -- while mid .ne. ncut do 
c     
 1      mid = first 
        abskey = abs(a(mid)) 
        do 2 j=first+1, last
           if (abs(a(j)) .gt. abskey) then
              mid = mid+1
c     interchange 
              tmp = a(mid) 
              itmp = ind(mid)
              a(mid) = a(j)
              ind(mid) = ind(j) 
              a(j)  = tmp
              ind(j) = itmp 
           endif
 2      continue 
c     
c     interchange 
c     
        tmp = a(mid) 
        a(mid) = a(first) 
        a(first)  = tmp
c     
        itmp = ind(mid)
        ind(mid) = ind(first) 
        ind(first) = itmp 
c     
c     test for while loop 
c     
        if (mid .eq. ncut) return 
        if (mid .gt. ncut) then
           last = mid-1
        else
           first = mid+1
        endif
        goto 1
c----------------end-of-qsplit------------------------------------------
c-----------------------------------------------------------------------
        end
c-----------------------------------------------------------------------
      subroutine aplb (nrow,ncol,job,a,ja,ia,b,jb,ib,
     *     c,jc,ic,nzmax,iw,ierr)
      real*8 a(*), b(*), c(*) 
      integer ja(*),jb(*),jc(*),ia(nrow+1),ib(nrow+1),ic(nrow+1),
     *     iw(ncol)
c-----------------------------------------------------------------------
c performs the matrix sum  C = A+B. 
c-----------------------------------------------------------------------
c on entry:
c ---------
c nrow	= integer. The row dimension of A and B
c ncol  = integer. The column dimension of A and B.
c job   = integer. Job indicator. When job = 0, only the structure
c                  (i.e. the arrays jc, ic) is computed and the
c                  real values are ignored.
c
c a,
c ja,
c ia   = Matrix A in compressed sparse row format.
c 
c b, 
c jb, 
c ib	=  Matrix B in compressed sparse row format.
c
c nzmax	= integer. The  length of the arrays c and jc.
c         amub will stop if the result matrix C  has a number 
c         of elements that exceeds exceeds nzmax. See ierr.
c 
c on return:
c----------
c c, 
c jc, 
c ic	= resulting matrix C in compressed sparse row sparse format.
c	    
c ierr	= integer. serving as error message. 
c         ierr = 0 means normal return,
c         ierr .gt. 0 means that amub stopped while computing the
c         i-th row  of C with i=ierr, because the number 
c         of elements in C exceeds nzmax.
c
c work arrays:
c------------
c iw	= integer work array of length equal to the number of
c         columns in A.
c
c-----------------------------------------------------------------------
      logical values
      values = (job .ne. 0) 
      ierr = 0
      len = 0
      ic(1) = 1 
      do 1 j=1, ncol
         iw(j) = 0
 1    continue
c     
      do 500 ii=1, nrow
c     row i 
         do 200 ka=ia(ii), ia(ii+1)-1 
            len = len+1
            jcol    = ja(ka)
            if (len .gt. nzmax) goto 999
            jc(len) = jcol 
            if (values) c(len)  = a(ka) 
            iw(jcol)= len
 200     continue
c     
         do 300 kb=ib(ii),ib(ii+1)-1
            jcol = jb(kb)
            jpos = iw(jcol)
            if (jpos .eq. 0) then
               len = len+1
               if (len .gt. nzmax) goto 999
               jc(len) = jcol
               if (values) c(len)  = b(kb)
               iw(jcol)= len
            else
               if (values) c(jpos) = c(jpos) + b(kb)
            endif
 300     continue
         do 301 k=ic(ii), len
	    iw(jc(k)) = 0
 301     continue
         ic(ii+1) = len+1
 500  continue
      return
 999  ierr = ii
      return
c------------end of aplb ----------------------------------------------- 
c-----------------------------------------------------------------------
      end
c-----------------------------------------------------------------------
      subroutine aplb1(nrow,ncol,job,a,ja,ia,b,jb,ib,c,jc,ic,nzmax,ierr)
      real*8 a(*), b(*), c(*) 
      integer ja(*),jb(*),jc(*),ia(nrow+1),ib(nrow+1),ic(nrow+1)
c-----------------------------------------------------------------------
c performs the matrix sum  C = A+B for matrices in sorted CSR format.
c the difference with aplb  is that the resulting matrix is such that
c the elements of each row are sorted with increasing column indices in
c each row, provided the original matrices are sorted in the same way. 
c-----------------------------------------------------------------------
c on entry:
c ---------
c nrow	= integer. The row dimension of A and B
c ncol  = integer. The column dimension of A and B.
c job   = integer. Job indicator. When job = 0, only the structure
c                  (i.e. the arrays jc, ic) is computed and the
c                  real values are ignored.
c
c a,
c ja,
c ia   = Matrix A in compressed sparse row format with entries sorted
c 
c b, 
c jb, 
c ib	=  Matrix B in compressed sparse row format with entries sorted
c        ascendly in each row   
c
c nzmax	= integer. The  length of the arrays c and jc.
c         amub will stop if the result matrix C  has a number 
c         of elements that exceeds exceeds nzmax. See ierr.
c 
c on return:
c----------
c c, 
c jc, 
c ic	= resulting matrix C in compressed sparse row sparse format
c         with entries sorted ascendly in each row. 
c	    
c ierr	= integer. serving as error message. 
c         ierr = 0 means normal return,
c         ierr .gt. 0 means that amub stopped while computing the
c         i-th row  of C with i=ierr, because the number 
c         of elements in C exceeds nzmax.
c
c Notes: 
c-------
c     this will not work if any of the two input matrices is not sorted
c-----------------------------------------------------------------------
      logical values
      values = (job .ne. 0) 
      ierr = 0
      kc = 1
      ic(1) = kc 
c
      do 6 i=1, nrow
         ka = ia(i)
         kb = ib(i)
         kamax = ia(i+1)-1
         kbmax = ib(i+1)-1 
 5       continue 
         if (ka .le. kamax) then
            j1 = ja(ka)
         else
            j1 = ncol+1
         endif
         if (kb .le. kbmax) then 
            j2 = jb(kb)         
         else 
            j2 = ncol+1
         endif
c
c     three cases
c     
         if (j1 .eq. j2) then 
            if (values) c(kc) = a(ka)+b(kb)
            jc(kc) = j1
            ka = ka+1
            kb = kb+1
            kc = kc+1
         else if (j1 .lt. j2) then
            jc(kc) = j1
            if (values) c(kc) = a(ka)
            ka = ka+1
            kc = kc+1
         else if (j1 .gt. j2) then
            jc(kc) = j2
            if (values) c(kc) = b(kb)
            kb = kb+1
            kc = kc+1
         endif
         if (kc .gt. nzmax) goto 999
         if (ka .le. kamax .or. kb .le. kbmax) goto 5
         ic(i+1) = kc
 6    continue
      return
 999  ierr = i 
      return
c------------end-of-aplb1----------------------------------------------- 
c-----------------------------------------------------------------------
      end
      subroutine aplsb (nrow,ncol,a,ja,ia,s,b,jb,ib,
     *     c,jc,ic,nzmax,iw,ierr)
      implicit none
      integer nrow, ncol, nzmax, ierr, len, j, ii, ka, jcol, kb
      integer jpos, k
      real*8 a(*),b(*),c(*),s
      integer ja(*),jb(*),jc(*),ia(nrow+1),ib(nrow+1),
     *     ic(nrow+1),iw(ncol)
c-----------------------------------------------------------------------
c performs the matrix linear combination  C = A+s*B  
c-----------------------------------------------------------------------
c on entry:
c ---------
c nrow	= integer. The row dimension of A
c ncol  = integer. The column dimension of A
c
c a,
c ja,
c ia   = Matrix A in compressed sparse row format.
c
c s	= real. scalar factor for B.
c
c b, 
c jb, 
c ib	=  Matrix B in compressed sparse row format.
c
c nzmax	= integer. The  length of the arrays c and jc.
c         amub will stop if the result matrix C  has a number 
c         of elements that exceeds exceeds nzmax. See ierr.
c 
c on return:
c----------
c c, 
c jc, 
c ic	= resulting matrix C in compressed sparse row sparse format.
c	    
c ierr	= integer. serving as error message. 
c         ierr = 0 means normal return,
c         ierr .gt. 0 means that amub stopped while computing the
c         i-th row  of C with i=ierr, because the number 
c         of elements in C exceeds nzmax.
c
c work arrays:
c------------
c iw	= integer work array of length equal to the number of
c         columns in A.
c
c-----------------------------------------------------------------------
      ierr = 0
      len = 0
      ic(1) = 1 
c     
      do 1 j=1, ncol
         iw(j) = 0
 1    continue
c     
      do 500 ii=1, nrow
c     row i 
         do 200 ka=ia(ii), ia(ii+1)-1 
            len = len+1
            jcol    = ja(ka)
            if (len .gt. nzmax) goto 999
            jc(len) = jcol 
            c(len)  = a(ka)
            iw(jcol)= len
 200     continue
         do 300 kb=ib(ii),ib(ii+1)-1
            jcol = jb(kb)
            jpos = iw(jcol)
            if (jpos .eq. 0) then
               len = len+1
               if (len .gt. nzmax) goto 999
               jc(len) = jcol
               c(len)  = s*b(kb)
               iw(jcol)= len
            else
               c(jpos) = c(jpos) + s*b(kb)
            endif
 300     continue
         do 301 k=ic(ii), len
            iw(jc(k)) = 0
 301     continue
         ic(ii+1) = len+1
 500  continue
      return
 999  ierr = ii
      return
c----- end of aplsb ----------------------------------------------------
c-----------------------------------------------------------------------
      end
c-----------------------------------------------------------------------
      subroutine aplsca (nrow, a, ja, ia, scal,iw)
      implicit none
      integer nrow, icount, j, ko, ii, k1, k2, k
      real*8 a(*), scal
      integer ja(*), ia(nrow+1),iw(*)
c-----------------------------------------------------------------------
c Adds a scalar to the diagonal entries of a sparse matrix A :=A + s I 
c-----------------------------------------------------------------------
c on entry:
c ---------
c nrow	= integer. The row dimension of A
c
c a,
c ja,
c ia    = Matrix A in compressed sparse row format.
c 
c scal  = real. scalar to add to the diagonal entries. 
c
c on return:
c----------
c
c a, 
c ja, 
c ia	= matrix A with diagonal elements shifted (or created).
c	    
c iw    = integer work array of length n. On return iw will
c         contain  the positions of the diagonal entries in the 
c         output matrix. (i.e., a(iw(k)), ja(iw(k)), k=1,...n,
c         are the values/column indices of the diagonal elements 
c         of the output matrix. ). 
c
c Notes:
c-------
c     The column dimension of A is not needed. 
c     important: the matrix a may be expanded slightly to allow for
c     additions of nonzero elements to previously nonexisting diagonals.
c     The is no checking as to whether there is enough space appended
c     to the arrays a and ja. if not sure allow for n additional 
c     elemnts. 
c coded by Y. Saad. Latest version July, 19, 1990
c-----------------------------------------------------------------------
      logical test
c
      call diapos (nrow,ja,ia,iw)
      icount = 0
      do 1 j=1, nrow
         if (iw(j) .eq. 0) then
            icount = icount+1
         else
            a(iw(j)) = a(iw(j)) + scal 
         endif
 1    continue
c
c     if no diagonal elements to insert in data structure return.
c
      if (icount .eq. 0) return
c
c shift the nonzero elements if needed, to allow for created 
c diagonal elements. 
c
      ko = ia(nrow+1)+icount
c
c     copy rows backward
c
      do 5 ii=nrow, 1, -1 
c     
c     go through  row ii
c     
         k1 = ia(ii)
         k2 = ia(ii+1)-1 
         ia(ii+1) = ko
         test = (iw(ii) .eq. 0) 
         do 4 k = k2,k1,-1 
            j = ja(k)
            if (test .and. (j .lt. ii)) then 
               test = .false. 
               ko = ko - 1
               a(ko) = scal 
               ja(ko) = ii
               iw(ii) = ko
            endif
            ko = ko-1
            a(ko) = a(k) 
            ja(ko) = j
 4       continue
c     diagonal element has not been added yet.
         if (test) then
            ko = ko-1
            a(ko) = scal 
            ja(ko) = ii
            iw(ii) = ko
         endif
 5    continue
      ia(1) = ko 
      return
c-----------------------------------------------------------------------
c----------end-of-aplsca------------------------------------------------ 
      end
c-----------------------------------------------------------------------
      subroutine diapos  (n,ja,ia,idiag)
      implicit none
      integer n, i, k
      integer ia(n+1), ja(*), idiag(n) 
c-----------------------------------------------------------------------
c this subroutine returns the positions of the diagonal elements of a
c sparse matrix a, ja, ia, in the array idiag.
c-----------------------------------------------------------------------
c on entry:
c---------- 
c
c n	= integer. row dimension of the matrix a.
c a,ja,
c    ia = matrix stored compressed sparse row format. a array skipped.
c
c on return:
c-----------
c idiag  = integer array of length n. The i-th entry of idiag 
c          points to the diagonal element a(i,i) in the arrays
c          a, ja. (i.e., a(idiag(i)) = element A(i,i) of matrix A)
c          if no diagonal element is found the entry is set to 0.
c----------------------------------------------------------------------c
c           Y. Saad, March, 1990
c----------------------------------------------------------------------c
      do 1 i=1, n 
         idiag(i) = 0
 1    continue
c     
c     sweep through data structure. 
c     
      do  6 i=1,n
         do 51 k= ia(i),ia(i+1) -1
            if (ja(k) .eq. i) idiag(i) = k
 51      continue
 6    continue
c----------- -end-of-diapos---------------------------------------------
c-----------------------------------------------------------------------
      return
      end
c-----------------------------------------------------------------------
      subroutine amux (n, x, y, a,ja,ia) 
      real*8  x(*), y(*), a(*) 
      integer n, ja(*), ia(*)
c-----------------------------------------------------------------------
c         A times a vector
c----------------------------------------------------------------------- 
c multiplies a matrix by a vector using the dot product form
c Matrix A is stored in compressed sparse row storage.
c
c on entry:
c----------
c n     = row dimension of A
c x     = real array of length equal to the column dimension of
c         the A matrix.
c a, ja,
c    ia = input matrix in compressed sparse row format.
c
c on return:
c-----------
c y     = real array of length n, containing the product y=Ax
c
c-----------------------------------------------------------------------
c local variables
c
      real*8 t
      integer i, k
c-----------------------------------------------------------------------
      do 100 i = 1,n
c
c     compute the inner product of row i with vector x
c 
         t = 0.
         do 99 k=ia(i), ia(i+1)-1 
            t = t + a(k)*x(ja(k))
 99      continue
c
c     store result in y(i) 
c
         y(i) = t
 100  continue
c
      return
c---------end-of-amux---------------------------------------------------
c-----------------------------------------------------------------------
      end
      subroutine lusol0 (n, y, x, alu, jlu, ju)
      implicit none
      integer n, jlu(*), ju(*)
      real*8 x(n), y(n), alu(*)
c-----------------------------------------------------------------------
c
c performs a forward followed by a backward solve 
c for LU matrix as produced by  ILUT
c 
c-----------------------------------------------------------------------
c local variables
c
      integer i,k 
c
c forward solve 
c
      do 40 i = 1, n 
         x(i) = y(i)
         do 41 k=jlu(i),ju(i)-1 
            x(i) = x(i) - alu(k)* x(jlu(k)) 
 41      continue
 40   continue
c     
c     backward solve.
c     
      do 90 i = n, 1, -1 
         do 91 k=ju(i),jlu(i+1)-1 
            x(i) = x(i) - alu(k)*x(jlu(k))
 91      continue
         x(i) = alu(i)*x(i)
 90   continue
c
      return
c----------------end of lusol0 -----------------------------------------
c-----------------------------------------------------------------------
      end
      subroutine coscal(nrow,job,nrm,a,ja,ia,diag,b,jb,ib,ierr) 
c----------------------------------------------------------------------- 
      real*8 a(*),b(*),diag(nrow) 
      integer nrow,job,ja(*),jb(*),ia(nrow+1),ib(nrow+1),ierr 
c-----------------------------------------------------------------------
c scales the columns of A such that their norms are one on return
c result matrix written on b, or overwritten on A.
c 3 choices of norms: 1-norm, 2-norm, max-norm. in place.
c-----------------------------------------------------------------------
c on entry:
c ---------
c nrow	= integer. The row dimension of A
c
c job   = integer. job indicator. Job=0 means get array b only
c         job = 1 means get b, and the integer arrays ib, jb.
c
c nrm   = integer. norm indicator. nrm = 1, means 1-norm, nrm =2
c                  means the 2-nrm, nrm = 0 means max norm
c
c a,
c ja,
c ia   = Matrix A in compressed sparse row format.
c 
c on return:
c----------
c
c diag = diagonal matrix stored as a vector containing the matrix
c        by which the columns have been scaled, i.e., on return 
c        we have B = A * Diag
c
c b, 
c jb, 
c ib	= resulting matrix B in compressed sparse row sparse format.
c
c ierr  = error message. ierr=0     : Normal return 
c                        ierr=i > 0 : Column number i is a zero row.
c Notes:
c-------
c 1)        The column dimension of A is not needed. 
c 2)       algorithm in place (B can take the place of A).
c-----------------------------------------------------------------
      call cnrms (nrow,nrm,a,ja,ia,diag)
      ierr = 0
      do 1 j=1, nrow
         if (diag(j) .eq. 0.0) then
            ierr = j 
            return
         else
            diag(j) = 1.0d0/diag(j)
         endif
 1    continue
      call amudia (nrow,job,a,ja,ia,diag,b,jb,ib)
      return
c--------end-of-coscal-------------------------------------------------- 
c-----------------------------------------------------------------------
      end
      subroutine cnrms   (nrow, nrm, a, ja, ia, diag) 
      real*8 a(*), diag(nrow) 
      integer ja(*), ia(nrow+1) 
c-----------------------------------------------------------------------
c gets the norms of each column of A. (choice of three norms)
c-----------------------------------------------------------------------
c on entry:
c ---------
c nrow	= integer. The row dimension of A
c
c nrm   = integer. norm indicator. nrm = 1, means 1-norm, nrm =2
c                  means the 2-nrm, nrm = 0 means max norm
c
c a,
c ja,
c ia   = Matrix A in compressed sparse row format.
c 
c on return:
c----------
c
c diag = real vector of length nrow containing the norms
c
c-----------------------------------------------------------------
      do 10 k=1, nrow 
         diag(k) = 0.0d0
 10   continue
      do 1 ii=1,nrow
         k1 = ia(ii)
         k2 = ia(ii+1)-1
         do 2 k=k1, k2
            j = ja(k) 
c     update the norm of each column
            if (nrm .eq. 0) then
               diag(j) = max(diag(j),abs(a(k) ) ) 
            elseif (nrm .eq. 1) then
               diag(j) = diag(j) + abs(a(k) ) 
            else
               diag(j) = diag(j)+a(k)**2
            endif 
 2       continue
 1    continue
      if (nrm .ne. 2) return
      do 3 k=1, nrow
         diag(k) = sqrt(diag(k))
 3    continue
      return
c-----------------------------------------------------------------------
c------------end-of-cnrms-----------------------------------------------
      end 
      subroutine roscal(nrow,job,nrm,a,ja,ia,diag,b,jb,ib,ierr) 
      real*8 a(*), b(*), diag(nrow) 
      integer nrow,job,nrm,ja(*),jb(*),ia(nrow+1),ib(nrow+1),ierr 
c-----------------------------------------------------------------------
c scales the rows of A such that their norms are one on return
c 3 choices of norms: 1-norm, 2-norm, max-norm.
c-----------------------------------------------------------------------
c on entry:
c ---------
c nrow	= integer. The row dimension of A
c
c job   = integer. job indicator. Job=0 means get array b only
c         job = 1 means get b, and the integer arrays ib, jb.
c
c nrm   = integer. norm indicator. nrm = 1, means 1-norm, nrm =2
c                  means the 2-nrm, nrm = 0 means max norm
c
c a,
c ja,
c ia   = Matrix A in compressed sparse row format.
c 
c on return:
c----------
c
c diag = diagonal matrix stored as a vector containing the matrix
c        by which the rows have been scaled, i.e., on return 
c        we have B = Diag*A.
c
c b, 
c jb, 
c ib	= resulting matrix B in compressed sparse row sparse format.
c	    
c ierr  = error message. ierr=0     : Normal return 
c                        ierr=i > 0 : Row number i is a zero row.
c Notes:
c-------
c 1)        The column dimension of A is not needed. 
c 2)        algorithm in place (B can take the place of A).
c-----------------------------------------------------------------
      call rnrms (nrow,nrm,a,ja,ia,diag)
      ierr = 0
      do 1 j=1, nrow
         if (diag(j) .eq. 0.0d0) then
            ierr = j 
            return
         else
            diag(j) = 1.0d0/diag(j)
         endif
 1    continue
      call diamua(nrow,job,a,ja,ia,diag,b,jb,ib)
      return
c-------end-of-roscal---------------------------------------------------
c-----------------------------------------------------------------------
      end
      subroutine rnrms   (nrow, nrm, a, ja, ia, diag)
      implicit none
      integer nrow, nrm, ii, k1, k2, k
      real*8 a(*), diag(nrow), scal
      integer ja(*), ia(nrow+1)
c-----------------------------------------------------------------------
c gets the norms of each row of A. (choice of three norms)
c-----------------------------------------------------------------------
c on entry:
c ---------
c nrow  = integer. The row dimension of A
c
c nrm   = integer. norm indicator. nrm = 1, means 1-norm, nrm =2
c                  means the 2-nrm, nrm = 0 means max norm
c
c a,
c ja,
c ia   = Matrix A in compressed sparse row format.
c
c on return:
c----------
c
c diag = real vector of length nrow containing the norms
c
c-----------------------------------------------------------------
      do 1 ii=1,nrow
c
c     compute the norm if each element.
c
         scal = 0.
         k1 = ia(ii)
         k2 = ia(ii+1)-1
         if (nrm .eq. 0) then
            do 2 k=k1, k2
               scal = max(scal,abs(a(k) ) )
 2          continue
         elseif (nrm .eq. 1) then
            do 3 k=k1, k2
               scal = scal + abs(a(k) )
 3          continue
         else
            do 4 k=k1, k2
               scal = scal+a(k)**2
 4          continue
         endif
         if (nrm .eq. 2) scal = sqrt(scal)
         diag(ii) = scal
 1    continue
      return
c-----------------------------------------------------------------------
c-------------end-of-rnrms----------------------------------------------
      end
      subroutine diamua (nrow,job, a, ja, ia, diag, b, jb, ib)
      implicit none
      integer nrow, job, ii, k1, k2, k
      real*8 a(*), b(*), diag(nrow), scal
      integer ja(*),jb(*), ia(nrow+1),ib(nrow+1) 
c-----------------------------------------------------------------------
c performs the matrix by matrix product B = Diag * A  (in place) 
c-----------------------------------------------------------------------
c on entry:
c ---------
c nrow  = integer. The row dimension of A
c
c job   = integer. job indicator. Job=0 means get array b only
c         job = 1 means get b, and the integer arrays ib, jb.
c
c a,
c ja,
c ia   = Matrix A in compressed sparse row format.
c 
c diag = diagonal matrix stored as a vector dig(1:n)
c
c on return:
c----------
c
c b, 
c jb, 
c ib    = resulting matrix B in compressed sparse row sparse format.
c           
c Notes:
c-------
c 1)        The column dimension of A is not needed. 
c 2)        algorithm in place (B can take the place of A).
c           in this case use job=0.
c-----------------------------------------------------------------
      do 1 ii=1,nrow
c     
c     normalize each row 
c     
         k1 = ia(ii)
         k2 = ia(ii+1)-1
         scal = diag(ii) 
         do 2 k=k1, k2
            b(k) = a(k)*scal
 2       continue
 1    continue
c     
      if (job .eq. 0) return
c     
      ib(1) = ia(1) 
      do 3 ii=1, nrow
         ib(ii) = ia(ii)
         do 31 k=ia(ii),ia(ii+1)-1
            jb(k) = ja(k)
 31      continue
 3    continue
      return
c----------end-of-diamua------------------------------------------------
c-----------------------------------------------------------------------
      end 
c-----------------------------------------------------------------------
      subroutine csrbnd (n,a,ja,ia,job,abd,nabd,lowd,ml,mu,ierr)
      implicit none
      integer n, job, nabd, lowd, ml, mu, ierr, m, i, ii, j, mdiag, k
      real*8 a(*),abd(nabd,n)
      integer ia(n+1),ja(*)
c----------------------------------------------------------------------- 
c   Compressed Sparse Row  to  Banded (Linpack ) format.
c----------------------------------------------------------------------- 
c this subroutine converts a general sparse matrix stored in
c compressed sparse row format into the banded format. for the
c banded format,the Linpack conventions are assumed (see below).
c----------------------------------------------------------------------- 
c on entry:
c----------
c n	= integer,the actual row dimension of the matrix.
c
c a,
c ja,
c ia    = input matrix stored in compressed sparse row format.
c
c job	= integer. if job=1 then the values of the lower bandwith ml 
c         and the upper bandwidth mu are determined internally. 
c         otherwise it is assumed that the values of ml and mu 
c         are the correct bandwidths on input. See ml and mu below.
c
c nabd  = integer. first dimension of array abd.
c
c lowd  = integer. this should be set to the row number in abd where
c         the lowest diagonal (leftmost) of A is located. 
c         lowd should be  ( 1  .le.  lowd  .le. nabd).
c         if it is not known in advance what lowd should be
c         enter lowd = 0 and the default value lowd = ml+mu+1
c         will be chosen. Alternative: call routine getbwd from unary
c         first to detrermione ml and mu then define lowd accordingly.
c         (Note: the banded solvers in linpack use lowd=2*ml+mu+1. )
c
c ml	= integer. equal to the bandwidth of the strict lower part of A
c mu	= integer. equal to the bandwidth of the strict upper part of A
c         thus the total bandwidth of A is ml+mu+1.
c         if ml+mu+1 is found to be larger than lowd then an error 
c         flag is raised (unless lowd = 0). see ierr.
c
c note:   ml and mu are assumed to have	 the correct bandwidth values
c         as defined above if job is set to zero on entry.
c
c on return:
c-----------
c
c abd   = real array of dimension abd(nabd,n).
c         on return contains the values of the matrix stored in
c         banded form. The j-th column of abd contains the elements
c         of the j-th column of  the original matrix comprised in the
c         band ( i in (j-ml,j+mu) ) with the lowest diagonal at
c         the bottom row (row lowd). See details below for this format.
c
c ml	= integer. equal to the bandwidth of the strict lower part of A
c mu	= integer. equal to the bandwidth of the strict upper part of A
c         if job=1 on entry then these two values are internally computed.
c
c lowd  = integer. row number in abd where the lowest diagonal 
c         (leftmost) of A is located on return. In case lowd = 0
c         on return, then it is defined to ml+mu+1 on return and the
c         lowd will contain this value on return. `
c
c ierr  = integer. used for error messages. On return:
c         ierr .eq. 0  :means normal return
c         ierr .eq. -1 : means invalid value for lowd. (either .lt. 0
c         or larger than nabd).
c         ierr .eq. -2 : means that lowd is not large enough and as 
c         result the matrix cannot be stored in array abd. 
c         lowd should be at least ml+mu+1, where ml and mu are as
c         provided on output.
c
c----------------------------------------------------------------------* 
c Additional details on banded format.  (this closely follows the      *
c format used in linpack. may be useful for converting a matrix into   *
c this storage format in order to use the linpack  banded solvers).    * 
c----------------------------------------------------------------------*
c             ---  band storage format  for matrix abd ---             * 
c uses ml+mu+1 rows of abd(nabd,*) to store the diagonals of           *
c a in rows of abd starting from the lowest (sub)-diagonal  which  is  *
c stored in row number lowd of abd. the minimum number of rows needed  *
c in abd is ml+mu+1, i.e., the minimum value for lowd is ml+mu+1. the  *
c j-th  column  of  abd contains the elements of the j-th column of a, *
c from bottom to top: the element a(j+ml,j) is stored in  position     *
c abd(lowd,j), then a(j+ml-1,j) in position abd(lowd-1,j) and so on.   *
c Generally, the element a(j+k,j) of original matrix a is stored in    *
c position abd(lowd+k-ml,j), for k=ml,ml-1,..,0,-1, -mu.               *
c The first dimension nabd of abd must be .ge. lowd                    *
c                                                                      *
c     example [from linpack ]:   if the original matrix is             *
c                                                                      *
c              11 12 13  0  0  0                                       *
c              21 22 23 24  0  0                                       *
c               0 32 33 34 35  0     original banded matrix            *
c               0  0 43 44 45 46                                       *
c               0  0  0 54 55 56                                       *
c               0  0  0  0 65 66                                       *
c                                                                      *
c then  n = 6, ml = 1, mu = 2. lowd should be .ge. 4 (=ml+mu+1)  and   *
c if lowd = 5 for example, abd  should be:                             *
c                                                                      *
c untouched --> x  x  x  x  x  x                                       *
c               *  * 13 24 35 46                                       *
c               * 12 23 34 45 56    resulting abd matrix in banded     *
c              11 22 33 44 55 66    format                             *
c  row lowd--> 21 32 43 54 65  *                                       *
c                                                                      *
c * = not used                                                         *
c                                                                      
*
c----------------------------------------------------------------------*
c first determine ml and mu.
c----------------------------------------------------------------------- 
      ierr = 0
c-----------
      if (job .eq. 1) call getbwd(n,a,ja,ia,ml,mu)
      m = ml+mu+1
      if (lowd .eq. 0) lowd = m
      if (m .gt. lowd)  ierr = -2
      if (lowd .gt. nabd .or. lowd .lt. 0) ierr = -1
      if (ierr .lt. 0) return
c------------
      do 15  i=1,m
         ii = lowd -i+1
         do 10 j=1,n
	    abd(ii,j) = 0.0d0
 10      continue
 15   continue
c---------------------------------------------------------------------	   
      mdiag = lowd-ml
      do 30 i=1,n
         do 20 k=ia(i),ia(i+1)-1
            j = ja(k)
            abd(i-j+mdiag,j) = a(k) 
 20      continue
 30   continue
      return
c------------- end of csrbnd ------------------------------------------- 
c----------------------------------------------------------------------- 
      end
c-----------------------------------------------------------------------
      subroutine getbwd(n,a,ja,ia,ml,mu)
c-----------------------------------------------------------------------
c gets the bandwidth of lower part and upper part of A.
c does not assume that A is sorted.
c-----------------------------------------------------------------------
c on entry:
c----------
c n	= integer = the row dimension of the matrix
c a, ja,
c    ia = matrix in compressed sparse row format.
c 
c on return:
c----------- 
c ml	= integer. The bandwidth of the strict lower part of A
c mu	= integer. The bandwidth of the strict upper part of A 
c
c Notes:
c ===== ml and mu are allowed to be negative or return. This may be 
c       useful since it will tell us whether a band is confined 
c       in the strict  upper/lower triangular part. 
c       indeed the definitions of ml and mu are
c
c       ml = max ( (i-j)  s.t. a(i,j) .ne. 0  )
c       mu = max ( (j-i)  s.t. a(i,j) .ne. 0  )
c----------------------------------------------------------------------c
c Y. Saad, Sep. 21 1989                                                c
c----------------------------------------------------------------------c
      implicit none
      integer n
      real*8 a(*) 
      integer ja(*),ia(n+1),ml,mu,ldist,i,k 
      ml = - n
      mu = - n
      do 3 i=1,n
         do 31 k=ia(i),ia(i+1)-1 
            ldist = i-ja(k)
            ml = max(ml,ldist)
            mu = max(mu,-ldist)
 31      continue
 3    continue
      return
c---------------end-of-getbwd ------------------------------------------ 
c----------------------------------------------------------------------- 
      end
c----------------------------------------------------------------------- 
      subroutine coicsr (n,nnz,job,a,ja,ia,iwk)
      implicit none
      integer n, nnz, job, i, k, init, j, ipos, inext, jnext
      integer ia(nnz),ja(nnz),iwk(n) 
      real*8 a(*)
c------------------------------------------------------------------------
c IN-PLACE coo-csr conversion routine.
c------------------------------------------------------------------------
c this subroutine converts a matrix stored in coordinate format into 
c the csr format. The conversion is done in place in that the arrays 
c a,ja,ia of the result are overwritten onto the original arrays.
c------------------------------------------------------------------------
c on entry:
c--------- 
c n	= integer. row dimension of A.
c nnz	= integer. number of nonzero elements in A.
c job   = integer. Job indicator. when job=1, the real values in a are
c         filled. Otherwise a is not touched and the structure of the
c         array only (i.e. ja, ia)  is obtained.
c a	= real array of size nnz (number of nonzero elements in A)
c         containing the nonzero elements 
c ja	= integer array of length nnz containing the column positions
c 	  of the corresponding elements in a.
c ia	= integer array of length nnz containing the row positions
c 	  of the corresponding elements in a.
c iwk	= integer work array of length n.
c on return:
c----------
c a
c ja 
c ia	= contains the compressed sparse row data structure for the 
c         resulting matrix.
c Note: 
c-------
c         the entries of the output matrix are not sorted (the column
c         indices in each are not in increasing order) use coocsr
c         if you want them sorted.
c----------------------------------------------------------------------c
c  Coded by Y. Saad, Sep. 26 1989                                      c
c----------------------------------------------------------------------c
      real*8 t,tnext
      logical values
c----------------------------------------------------------------------- 
      values = (job .eq. 1) 
c find pointer array for resulting matrix. 
      do 35 i=1,n+1
         iwk(i) = 0
 35   continue
      do 4 k=1,nnz
         i = ia(k)
         iwk(i+1) = iwk(i+1)+1
 4    continue 
c------------------------------------------------------------------------
      iwk(1) = 1 
      do 44 i=2,n
         iwk(i) = iwk(i-1) + iwk(i)
 44   continue 
c
c     loop for a cycle in chasing process. 
c
      init = 1
      k = 0
 5    if (values) t = a(init)
      i = ia(init)
      j = ja(init)
      ia(init) = -1
c------------------------------------------------------------------------
 6    k = k+1 		
c     current row number is i.  determine  where to go. 
      ipos = iwk(i)
c     save the chased element. 
      if (values) tnext = a(ipos)
      inext = ia(ipos)
      jnext = ja(ipos)
c     then occupy its location.
      if (values) a(ipos)  = t
      ja(ipos) = j
c     update pointer information for next element to come in row i. 
      iwk(i) = ipos+1
c     determine  next element to be chased,
      if (ia(ipos) .lt. 0) goto 65
      t = tnext
      i = inext
      j = jnext 
      ia(ipos) = -1
      if (k .lt. nnz) goto 6
      goto 70
 65   init = init+1
      if (init .gt. nnz) goto 70
      if (ia(init) .lt. 0) goto 65
c     restart chasing --	
      goto 5
 70   do 80 i=1,n 
         ia(i+1) = iwk(i)
 80   continue
      ia(1) = 1
      return
c----------------- end of coicsr ----------------------------------------
c------------------------------------------------------------------------
      end
c----------------------------------------------------------------------- 
      subroutine coocsr(nrow,nnz,a,ir,jc,ao,jao,iao)
c----------------------------------------------------------------------- 
      real*8 a(*),ao(*),x
      integer ir(*),jc(*),jao(*),iao(*)
c-----------------------------------------------------------------------
c  Coordinate     to   Compressed Sparse Row 
c----------------------------------------------------------------------- 
c converts a matrix that is stored in coordinate format
c  a, ir, jc into a row general sparse ao, jao, iao format.
c
c on entry:
c--------- 
c nrow	= dimension of the matrix 
c nnz	= number of nonzero elements in matrix
c a,
c ir, 
c jc    = matrix in coordinate format. a(k), ir(k), jc(k) store the nnz
c         nonzero elements of the matrix with a(k) = actual real value of
c 	  the elements, ir(k) = its row number and jc(k) = its column 
c	  number. The order of the elements is arbitrary. 
c
c on return:
c----------- 
c ir 	is destroyed
c
c ao, jao, iao = matrix in general sparse matrix format with ao 
c 	continung the real values, jao containing the column indices, 
c	and iao being the pointer to the beginning of the row, 
c	in arrays ao, jao.
c
c Notes:
c------ column indices for each row in jao are in ascending order
c       only if jc was sorted in that way
c------ This routine is NOT in place.  See coicsr
c
c------------------------------------------------------------------------
      do 1 k=1,nrow+1
         iao(k) = 0
 1    continue
c determine row-lengths.
      do 2 k=1, nnz
         iao(ir(k)) = iao(ir(k))+1
 2    continue
c starting position of each row..
      k = 1
      do 3 j=1,nrow+1
         k0 = iao(j)
         iao(j) = k
         k = k+k0
 3    continue
c go through the structure  once more. Fill in output matrix.
      do 4 k=1, nnz
         i = ir(k)
         j = jc(k)
         x = a(k)
         iad = iao(i)
         ao(iad) =  x
         jao(iad) = j
         iao(i) = iad+1
 4    continue
c shift back iao
      do 5 j=nrow,1,-1
         iao(j+1) = iao(j)
 5    continue
      iao(1) = 1
      return
c------------- end of coocsr ------------------------------------------- 
c----------------------------------------------------------------------- 
      end
c-----------------------------------------------------------------------
      subroutine csrcsc (n,job,ipos,a,ja,ia,ao,jao,iao)
      implicit none
      integer n, job, ipos, i, k, j, next
      integer ia(n+1),iao(n+1),ja(*),jao(*)
      real*8  a(*),ao(*)
c-----------------------------------------------------------------------
c Compressed Sparse Row     to      Compressed Sparse Column
c
c (transposition operation)   Not in place. 
c----------------------------------------------------------------------- 
c -- not in place --
c this subroutine transposes a matrix stored in a, ja, ia format.
c ---------------
c on entry:
c----------
c n	= dimension of A.
c job	= integer to indicate whether or not to fill the values of the
c         matrix ao or only the pattern (ia, and ja). Enter 1 for yes.
c ipos  = starting position in ao, jao of the transposed matrix.
c         the iao array takes this into account (thus iao(1) is set to ipos.)
c         Note: this may be useful if one needs to append the data structure
c         of the transpose to that of A. In this case use for example
c                call csrcsc (n,1,n+2,a,ja,ia,a,ja,ia(n+2)) 
c	  for any other normal usage, enter ipos=1.
c a	= real array of length nnz (nnz=number of nonzero elements in input 
c         matrix) containing the nonzero elements.
c ja	= integer array of length nnz containing the column positions
c 	  of the corresponding elements in a.
c ia	= integer of size n+1. ia(k) contains the position in a, ja of
c	  the beginning of the k-th row.
c
c on return:
c ---------- 
c output arguments:
c ao	= real array of size nzz containing the "a" part of the transpose
c jao	= integer array of size nnz containing the column indices.
c iao	= integer array of size n+1 containing the "ia" index array of
c	  the transpose. 
c
c----------------------------------------------------------------------- 
c----------------- compute lengths of rows of transp(A) ----------------
      do 1 i=1,n+1
         iao(i) = 0
 1    continue
      do 3 i=1, n
         do 2 k=ia(i), ia(i+1)-1 
            j = ja(k)+1
            iao(j) = iao(j)+1
 2       continue 
 3    continue
c---------- compute pointers from lengths ------------------------------
      iao(1) = ipos 
      do 4 i=1,n
         iao(i+1) = iao(i) + iao(i+1)
 4    continue
c--------------- now do the actual copying ----------------------------- 
      do 6 i=1,n
         do 62 k=ia(i),ia(i+1)-1 
            j = ja(k) 
            next = iao(j)
            if (job .eq. 1)  ao(next) = a(k)
            jao(next) = i
            iao(j) = next+1
 62      continue
 6    continue
c-------------------------- reshift iao and leave ---------------------- 
      do 7 i=n,1,-1
         iao(i+1) = iao(i)
 7    continue
      iao(1) = ipos
c--------------- end of csrcsc ----------------------------------------- 
c-----------------------------------------------------------------------
      end
c-----------------------------------------------------------------------
      subroutine csrdia (n,idiag,job,a,ja,ia,ndiag,
     *                   diag,ioff,ao,jao,iao,ind)
      implicit none
      integer n, idiag, job, ndiag, job1, job2, n2, idum, ii, jmax
      integer k, j, i, ko, l
      real*8 diag(ndiag,idiag), a(*), ao(*)
      integer ia(*), ind(*), ja(*), jao(*), iao(*), ioff(*)
c----------------------------------------------------------------------- 
c Compressed sparse row     to    diagonal format
c----------------------------------------------------------------------- 
c this subroutine extracts  idiag diagonals  from the  input matrix a,
c a, ia, and puts the rest of  the matrix  in the  output matrix ao,
c jao, iao.  The diagonals to be extracted depend  on the  value of job
c (see below for details.)  In  the first  case, the  diagonals to be
c extracted are simply identified by  their offsets  provided in ioff
c by the caller.  In the second case, the  code internally determines
c the idiag most significant diagonals, i.e., those  diagonals of the
c matrix which  have  the  largest  number  of  nonzero elements, and
c extracts them.
c----------------------------------------------------------------------- 
c on entry:
c---------- 
c n	= dimension of the matrix a.
c idiag = integer equal to the number of diagonals to be extracted. 
c         Note: on return idiag may be modified.
c a, ja, 			
c    ia = matrix stored in a, ja, ia, format
c job	= integer. serves as a job indicator.  Job is better thought 
c         of as a two-digit number job=xy. If the first (x) digit
c         is one on entry then the diagonals to be extracted are 
c         internally determined. In this case csrdia exctracts the
c         idiag most important diagonals, i.e. those having the largest
c         number on nonzero elements. If the first digit is zero
c         then csrdia assumes that ioff(*) contains the offsets 
c         of the diagonals to be extracted. there is no verification 
c         that ioff(*) contains valid entries.
c         The second (y) digit of job determines whether or not
c         the remainder of the matrix is to be written on ao,jao,iao.
c         If it is zero  then ao, jao, iao is not filled, i.e., 
c         the diagonals are found  and put in array diag and the rest is
c         is discarded. if it is one, ao, jao, iao contains matrix
c         of the remaining elements.
c         Thus:
c         job= 0 means do not select diagonals internally (pick those
c                defined by ioff) and do not fill ao,jao,iao
c         job= 1 means do not select diagonals internally 
c                      and fill ao,jao,iao
c         job=10 means  select diagonals internally 
c                      and do not fill ao,jao,iao
c         job=11 means select diagonals internally 
c                      and fill ao,jao,iao
c 
c ndiag = integer equal to the first dimension of array diag.
c
c on return:
c----------- 
c
c idiag = number of diagonals found. This may be smaller than its value 
c         on entry. 
c diag  = real array of size (ndiag x idiag) containing the diagonals
c         of A on return
c          
c ioff  = integer array of length idiag, containing the offsets of the
c   	  diagonals to be extracted.
c ao, jao
c  iao  = remainder of the matrix in a, ja, ia format.
c work arrays:
c------------ 
c ind   = integer array of length 2*n-1 used as integer work space.
c         needed only when job.ge.10 i.e., in case the diagonals are to
c         be selected internally.
c
c Notes:
c-------
c    1) The algorithm is in place: ao, jao, iao can be overwritten on 
c       a, ja, ia if desired 
c    2) When the code is required to select the diagonals (job .ge. 10) 
c       the selection of the diagonals is done from left to right 
c       as a result if several diagonals have the same weight (number 
c       of nonzero elemnts) the leftmost one is selected first.
c-----------------------------------------------------------------------
      job1 = job/10
      job2 = job-job1*10
      if (job1 .eq. 0) goto 50
      n2 = n+n-1
      call infdia(n,ja,ia,ind,idum)
c----------- determine diagonals to  accept.---------------------------- 
c----------------------------------------------------------------------- 
      ii = 0
 4    ii=ii+1
      jmax = 0
      do 41 k=1, n2
         j = ind(k)
         if (j .le. jmax) goto 41
         i = k
         jmax = j
 41   continue
      if (jmax .le. 0) then
         ii = ii-1
         goto 42
      endif
      ioff(ii) = i-n
      ind(i) = - jmax
      if (ii .lt.  idiag) goto 4
 42   idiag = ii
c---------------- initialize diago to zero ----------------------------- 
 50   continue
      do 55 j=1,idiag
         do 54 i=1,n
            diag(i,j) = 0.
 54      continue
 55   continue
c----------------------------------------------------------------------- 
      ko = 1
c----------------------------------------------------------------------- 
c extract diagonals and accumulate remaining matrix.
c----------------------------------------------------------------------- 
      do 6 i=1, n
         do 51 k=ia(i),ia(i+1)-1 
            j = ja(k)
            do 52 l=1,idiag
               if (j-i .ne. ioff(l)) goto 52
               diag(i,l) = a(k)
               goto 51
 52         continue
c--------------- append element not in any diagonal to ao,jao,iao ----- 
            if (job2 .eq. 0) goto 51
            ao(ko) = a(k)
            jao(ko) = j
            ko = ko+1
 51      continue
         if (job2 .ne. 0 ) ind(i+1) = ko
 6    continue
      if (job2 .eq. 0) return
c     finish with iao
      iao(1) = 1
      do 7 i=2,n+1
         iao(i) = ind(i)
 7    continue
      return
c----------- end of csrdia ---------------------------------------------
c----------------------------------------------------------------------- 
      end
      subroutine csrdns(nrow,ncol,a,ja,ia,dns,ndns,ierr)
      implicit none
      integer nrow, ncol, ndns, ierr, i, j, k
      real*8 dns(ndns,*),a(*)
      integer ja(*),ia(*)
c-----------------------------------------------------------------------
c Compressed Sparse Row    to    Dense 
c-----------------------------------------------------------------------
c
c converts a row-stored sparse matrix into a densely stored one
c
c On entry:
c---------- 
c
c nrow	= row-dimension of a
c ncol	= column dimension of a
c a, 
c ja, 
c ia    = input matrix in compressed sparse row format. 
c         (a=value array, ja=column array, ia=pointer array)
c dns   = array where to store dense matrix
c ndns	= first dimension of array dns 
c
c on return: 
c----------- 
c dns   = the sparse matrix a, ja, ia has been stored in dns(ndns,*)
c 
c ierr  = integer error indicator. 
c         ierr .eq. 0  means normal return
c         ierr .eq. i  means that the code has stopped when processing
c         row number i, because it found a column number .gt. ncol.
c 
c----------------------------------------------------------------------- 
      ierr = 0
      do 1 i=1, nrow
         do 2 j=1,ncol
	    dns(i,j) = 0.0d0
 2       continue
 1    continue
c     
      do 4 i=1,nrow
         do 3 k=ia(i),ia(i+1)-1
            j = ja(k) 
	    if (j .gt. ncol) then
               ierr = i
               return
	    endif
	    dns(i,j) = a(k)
 3       continue	   
 4    continue
      return
c---- end of csrdns ----------------------------------------------------
c-----------------------------------------------------------------------
      end
c----------------------------------------------------------------------- 
      subroutine infdia (n,ja,ia,ind,idiag)
      implicit none
      integer n, idiag, n2, i, k, j
      integer ia(*), ind(*), ja(*)
c-----------------------------------------------------------------------
c     obtains information on the diagonals of A. 
c----------------------------------------------------------------------- 
c this subroutine finds the lengths of each of the 2*n-1 diagonals of A
c it also outputs the number of nonzero diagonals found. 
c----------------------------------------------------------------------- 
c on entry:
c---------- 
c n	= dimension of the matrix a.
c
c a,    ..... not needed here.
c ja, 			
c ia    = matrix stored in csr format
c
c on return:
c----------- 
c
c idiag = integer. number of nonzero diagonals found. 
c 
c ind   = integer array of length at least 2*n-1. The k-th entry in
c         ind contains the number of nonzero elements in the diagonal
c         number k, the numbering beeing from the lowermost diagonal
c         (bottom-left). In other words ind(k) = length of diagonal
c         whose offset wrt the main diagonal is = - n + k.
c----------------------------------------------------------------------c
c           Y. Saad, Sep. 21 1989                                      c
c----------------------------------------------------------------------c
      n2= n+n-1
      do 1 i=1,n2
         ind(i) = 0
 1    continue
      do 3 i=1, n
         do 2 k=ia(i),ia(i+1)-1
            j = ja(k)
            ind(n+j-i) = ind(n+j-i) +1
 2       continue 
 3    continue
c     count the nonzero ones.
      idiag = 0 
      do 41 k=1, n2
         if (ind(k) .ne. 0) idiag = idiag+1
 41   continue
      return
c done
c------end-of-infdia ---------------------------------------------------
c-----------------------------------------------------------------------
      end
      subroutine ivperm (n, ix, perm)
      implicit none
      integer n, perm(n), ix(n)
      integer init, ii, k, next, j
c-----------------------------------------------------------------------
c this subroutine performs an in-place permutation of an integer vector 
c ix according to the permutation array perm(*), i.e., on return, 
c the vector x satisfies,
c
c	ix(perm(j)) :== ix(j), j=1,2,.., n
c
c-----------------------------------------------------------------------
c on entry:
c---------
c n 	= length of vector x.
c perm 	= integer array of length n containing the permutation  array.
c ix	= input vector
c
c on return:
c---------- 
c ix	= vector x permuted according to ix(perm(*)) :=  ix(*)
c
c----------------------------------------------------------------------c
c           Y. Saad, Sep. 21 1989                                      c
c----------------------------------------------------------------------c
c local variables
      integer tmp, tmp1
c
      init      = 1
      tmp	= ix(init)	
      ii        = perm(init)
      perm(init)= -perm(init)
      k         = 0
c     
c loop
c 
 6    k = k+1
c
c save the chased element --
c 
      tmp1	  = ix(ii) 
      ix(ii)     = tmp
      next	  = perm(ii) 
      if (next .lt. 0 ) goto 65
c     
c test for end 
c
      if (k .gt. n) goto 101
      tmp       = tmp1
      perm(ii)  = - perm(ii)
      ii        = next 
c
c end loop 
c
      goto 6
c
c reinitilaize cycle --
c
 65   init      = init+1
      if (init .gt. n) goto 101
      if (perm(init) .lt. 0) goto 65
      tmp	= ix(init)
      ii	= perm(init)
      perm(init)=-perm(init)
      goto 6
c     
 101  continue
      do 200 j=1, n
         perm(j) = -perm(j)
 200  continue 
c     
      return
c-------------------end-of-ivperm--------------------------------------- 
c-----------------------------------------------------------------------
      end
      subroutine dvperm (n, x, perm)
      implicit none
      integer n, perm(n)
      integer init, ii, k, next, j
      real*8 x(n)
c-----------------------------------------------------------------------
c this subroutine performs an in-place permutation of a real vector x 
c according to the permutation array perm(*), i.e., on return, 
c the vector x satisfies,
c
c	x(perm(j)) :== x(j), j=1,2,.., n
c
c-----------------------------------------------------------------------
c on entry:
c---------
c n 	= length of vector x.
c perm 	= integer array of length n containing the permutation  array.
c x	= input vector
c
c on return:
c---------- 
c x	= vector x permuted according to x(perm(*)) :=  x(*)
c
c----------------------------------------------------------------------c
c           Y. Saad, Sep. 21 1989                                      c
c----------------------------------------------------------------------c
c local variables 
      real*8 tmp, tmp1
c
      init      = 1
      tmp	= x(init)	
      ii        = perm(init)
      perm(init)= -perm(init)
      k         = 0
c     
c loop
c 
 6    k = k+1
c
c save the chased element --
c 
      tmp1	  = x(ii) 
      x(ii)     = tmp
      next	  = perm(ii) 
      if (next .lt. 0 ) goto 65
c     
c test for end 
c
      if (k .gt. n) goto 101
      tmp       = tmp1
      perm(ii)  = - perm(ii)
      ii        = next 
c
c end loop 
c
      goto 6
c
c reinitilaize cycle --
c
 65   init      = init+1
      if (init .gt. n) goto 101
      if (perm(init) .lt. 0) goto 65
      tmp	= x(init)
      ii	= perm(init)
      perm(init)=-perm(init)
      goto 6
c     
 101  continue
      do 200 j=1, n
         perm(j) = -perm(j)
 200  continue 
c     
      return
c-------------------end-of-dvperm--------------------------------------- 
c-----------------------------------------------------------------------
      end
      subroutine prtmt (nrow,ncol,a,ja,ia,rhs,guesol,title,key,type,
     1     ifmt,job,iounit)
c-----------------------------------------------------------------------
c writes a matrix in Harwell-Boeing format into a file.
c assumes that the matrix is stored in COMPRESSED SPARSE COLUMN FORMAT.
c some limited functionality for right hand sides. 
c Author: Youcef Saad - Date: Sept., 1989 - updated Oct. 31, 1989 to
c cope with new format. 
c-----------------------------------------------------------------------
c on entry:
c---------
c nrow   = number of rows in matrix
c ncol	 = number of columns in matrix 
c a	 = real array containing the values of the matrix stored 
c          columnwise
c ja 	 = integer array of the same length as a containing the column
c          indices of the corresponding matrix elements of array a.
c ia     = integer array of containing the pointers to the beginning of 
c	   the row in arrays a and ja.
c rhs    = real array  containing the right-hand-side (s) and optionally
c          the associated initial guesses and/or exact solutions
c          in this order. See also guesol for details. the vector rhs will
c          be used only if job .gt. 2 (see below). Only full storage for
c          the right hand sides is supported. 
c
c guesol = a 2-character string indicating whether an initial guess 
c          (1-st character) and / or the exact solution (2-nd)
c          character) is provided with the right hand side.
c	   if the first character of guesol is 'G' it means that an
c          an intial guess is provided for each right-hand sides. 
c          These are assumed to be appended to the right hand-sides in 
c          the array rhs.
c	   if the second character of guesol is 'X' it means that an
c          exact solution is provided for each right-hand side.
c          These are assumed to be appended to the right hand-sides 
c          and the initial guesses (if any) in the array rhs.
c
c title  = character*72 = title of matrix test ( character a*72 ).
c key    = character*8  = key of matrix 
c type   = charatcer*3  = type of matrix.
c
c ifmt	 = integer specifying the format chosen for the real values
c	   to be output (i.e., for a, and for rhs-guess-sol if 
c          applicable). The meaning of ifmt is as follows.
c	  * if (ifmt .lt. 100) then the D descriptor is used,
c           format Dd.m, in which the length (m) of the mantissa is 
c           precisely the integer ifmt (and d = ifmt+6)
c	  * if (ifmt .gt. 100) then prtmt will use the 
c           F- descriptor (format Fd.m) in which the length of the 
c           mantissa (m) is the integer mod(ifmt,100) and the length 
c           of the integer part is k=ifmt/100 (and d = k+m+2)
c	    Thus  ifmt= 4   means  D10.4  +.xxxxD+ee    while
c	          ifmt=104  means  F7.4   +x.xxxx
c	          ifmt=205  means  F9.5   +xx.xxxxx
c	    Note: formats for ja, and ia are internally computed.
c
c job	 = integer to indicate whether matrix values and
c	   a right-hand-side is available to be written
c          job = 1   write srtucture only, i.e., the arrays ja and ia.
c          job = 2   write matrix including values, i.e., a, ja, ia
c          job = 3   write matrix and one right hand side: a,ja,ia,rhs.
c	   job = nrhs+2 write matrix and nrhs successive right hand sides
c	   Note that there cannot be any right-hand-side if the matrix
c	   has no values. Also the initial guess and exact solutions when 
c          provided are for each right hand side. For example if nrhs=2 
c          and guesol='GX' there are 6 vectors to write.
c          
c
c iounit = logical unit number where to write the matrix into.
c
c on return:
c---------- 
c the matrix a, ja, ia will be written in output unit iounit
c in the Harwell-Boeing format. None of the inputs is modofied.
c  
c Notes: 1) This code attempts to pack as many elements as possible per
c        80-character line.
c           (1-14-94:  Gary R. Smith added one blank for legibility)
c        2) this code attempts to avoid as much as possible to put
c        blanks in the formats that are written in the 4-line header
c	 (This is done for purely esthetical reasons since blanks
c        are ignored in format descriptors.)
c        3) sparse formats for right hand sides and guesses are not
c        supported.
c-----------------------------------------------------------------------
      implicit none
      integer ifmt, job, iounit, ix, ihead, i, next, iend
      character title*72,key*8,type*3,ptrfmt*16,indfmt*16,valfmt*20,
     *	        guesol*2, rhstyp*3
      integer totcrd, ptrcrd, indcrd, valcrd, rhscrd, nrow, ncol,
     1     nnz, nrhs, len, nperli
      integer ja(*), ia(*) 	
      real*8 a(*),rhs(*)
c--------------
c     compute pointer format
c--------------
      nnz    = ia(ncol+1) -1
      len    = int ( log10(0.1+(nnz+1))) + 2
      nperli = 80/len
      ptrcrd = ncol/nperli + 1
      if (len .gt. 9) then
         assign 101 to ix
      else
         assign 100 to ix
      endif
      write (ptrfmt,ix) nperli,len
 100  format(1h(,i2,1HI,i1,1h) )
 101  format(1h(,i2,1HI,i2,1h) )
c----------------------------
c compute ROW index format
c----------------------------
      len    = int ( log10(0.1+(nrow) )) + 2
      nperli = min0(80/len,nnz)
      indcrd = (nnz-1)/nperli+1
      write (indfmt,100) nperli,len
c---------------
c compute values and rhs format (using the same for both)
c--------------- 
      valcrd	= 0
      rhscrd  = 0
c quit this part if no values provided.
      if (job .le. 1) goto 20
c     
      if (ifmt .ge. 100) then
         ihead = ifmt/100
         ifmt = ifmt-100*ihead
         len = ihead+ifmt+2
         nperli = 80/len
c     
         if (len .le. 9 ) then
            assign 102 to ix
         elseif (ifmt .le. 9) then
            assign 103 to ix
         else 
            assign 104 to ix
         endif
c     
         write(valfmt,ix) nperli,len,ifmt
 102     format(1h(,i2,1hF,i1,1h.,i1,1h) )
 103     format(1h(,i2,1hF,i2,1h.,i1,1h) )
 104     format(1h(,i2,1hF,i2,1h.,i2,1h) )
C
      else
         len = ifmt + 8
         nperli = 80/len
c     try to minimize the blanks in the format strings.
         if (nperli .le. 9) then
	    if (len .le. 9 ) then
	       assign 105 to ix
	    elseif (ifmt .le. 9) then
	       assign 106 to ix
	    else 
	       assign 107 to ix
	    endif
	 else 
	    if (len .le. 9 ) then
	       assign 108 to ix
	    elseif (ifmt .le. 9) then
	       assign 109 to ix
	    else 
               assign 110 to ix
            endif
         endif
c-----------
         write(valfmt,ix) nperli,len,ifmt
 105     format(3h(1p,i1,1hD,i1,1h.,i1,1h) )
 106     format(3h(1p,i1,1hD,i2,1h.,i1,1h) )
 107     format(3h(1p,i1,1hD,i2,1h.,i2,1h) )
 108     format(3h(1p,i2,1hD,i1,1h.,i1,1h) )
 109     format(3h(1p,i2,1hD,i2,1h.,i1,1h) )
 110     format(3h(1p,i2,1hD,i2,1h.,i2,1h) )
c     
      endif 	    
      valcrd = (nnz-1)/nperli+1
      nrhs   = job -2
      if (nrhs .ge. 1) then
         i = (nrhs*nrow-1)/nperli+1
         rhscrd = i
         if (guesol(1:1) .eq. 'G') rhscrd = rhscrd+i
         if (guesol(2:2) .eq. 'X') rhscrd = rhscrd+i
         rhstyp = 'F'//guesol
      endif 
 20   continue
c     
      totcrd = ptrcrd+indcrd+valcrd+rhscrd
c     write 4-line or five line header
      write(iounit,10) title,key,totcrd,ptrcrd,indcrd,valcrd,
     1     rhscrd,type,nrow,ncol,nnz,nrhs,ptrfmt,indfmt,valfmt,valfmt
c-----------------------------------------------------------------------
      if (nrhs .ge. 1) write (iounit,11) rhstyp, nrhs
 10   format (a72, a8 / 5i14 / a3, 11x, 4i14 / 2a16, 2a20)
 11   format(A3,11x,i4)
c     
      write(iounit,ptrfmt) (ia (i), i = 1, ncol+1)
      write(iounit,indfmt) (ja (i), i = 1, nnz)
      if (job .le. 1) return
      write(iounit,valfmt) (a(i), i = 1, nnz)
      if (job .le. 2) return 
      len = nrow*nrhs 
      next = 1
      iend = len
      write(iounit,valfmt) (rhs(i), i = next, iend)
c     
c     write initial guesses if available
c     
      if (guesol(1:1) .eq. 'G') then
         next = next+len
         iend = iend+ len
         write(iounit,valfmt) (rhs(i), i = next, iend)
      endif
c     
c     write exact solutions if available
c     
      if (guesol(2:2) .eq. 'X')then
         next = next+len
         iend = iend+ len
         write(iounit,valfmt) (rhs(i), i = next, iend)
      endif
c     
      return
c----------end of prtmt ------------------------------------------------ 
c-----------------------------------------------------------------------
      end
c----------------------------------------------------------------------c
c                          S P A R S K I T                             c
c----------------------------------------------------------------------c
c               ROERDERING ROUTINES -- REORD MODULE                    c
c----------------------------------------------------------------------c
c BSF     : Breadth-First Search traversal (Cuthill mc kee ordering)  c
c dblstr  : two-way dissection partitioning -- with equal size domains c
c stripes : routine used by dblstr to assign points                    c
c perphn  : finds a peripheral node and does a BFS search from it.     c
c add_lvst: routine for adding a new level set in BFS algorithm        c
c reversp : routine to reverse a given permuation (e.g., for RCMK)     c
c maskdeg : integer function to compute the `masked' of a node         c
c----------------------------------------------------------------------c
      subroutine BFS(n,ja,ia,nfirst,iperm,mask,maskval,riord,levels,
     *     nlev)
      implicit none 
      integer n,ja(*),ia(*),nfirst,iperm(n),mask(n),riord(*),levels(*),
     *     nlev,maskval 
c-----------------------------------------------------------------------
c finds the level-structure (breadth-first-search or CMK) ordering for a
c given sparse matrix. Uses add_lvst. Allows an set of nodes to be 
c the initial level (instead of just one node). Allows masked nodes.
c-------------------------parameters------------------------------------
c on entry:
c----------
c n      = number of nodes in the graph 
c ja, ia = pattern of matrix in CSR format (the ja,ia arrays of csr data
c          structure)
c nfirst = number of nodes in the first level that is input in riord
c iperm  = integer array indicating in which order to  traverse the graph
c          in order to generate all connected components. 
c          The nodes will be traversed in order iperm(1),....,iperm(n) 
c          Convention: 
c          if iperm(1) .eq. 0 on entry then BFS will traverse the 
c          nodes in the  order 1,2,...,n. 
c 
c riord  = (also an ouput argument). on entry riord contains the labels  
c          of the nfirst nodes that constitute the first level.      
c 
c mask   = array used to indicate whether or not a node should be 
c          condidered in the graph. see maskval.
c          mask is also used as a marker of  visited nodes. 
c 
c maskval= consider node i only when:  mask(i) .eq. maskval 
c          maskval must be .gt. 0. 
c          thus, to consider all nodes, take mask(1:n) = 1. 
c          maskval=1 (for example) 
c 
c on return
c ---------
c mask   = on return mask is restored to its initial state. 
c riord  = `reverse permutation array'. Contains the labels of the nodes
c          constituting all the levels found, from the first level to
c          the last. 
c levels = pointer array for the level structure. If lev is a level
c          number, and k1=levels(lev),k2=levels(lev+1)-1, then
c          all the nodes of level number lev are:
c          riord(k1),riord(k1+1),...,riord(k2) 
c nlev   = number of levels found
c-----------------------------------------------------------------------
c Notes on possible usage
c-------------------------
c 1. if you want a CMK ordering from a known node, say node init then
c    call BFS with nfirst=1,iperm(1) =0, mask(1:n) =1, maskval =1, 
c    riord(1) = init.
c 2. if you want the RCMK ordering and you have a preferred initial node
c     then use above call followed by reversp(n,riord)
c 3. Similarly to 1, and 2, but you know a good LEVEL SET to start from
c    (nfirst = number if nodes in the level, riord(1:nfirst) contains 
c    the nodes. 
c 4. If you do not know how to select a good initial node in 1 and 2, 
c    then you should use perphn instead. 
c
c-----------------------------------------------------------------------
c     local variables -- 
      integer j, ii, nod, istart, iend 
      logical permut
      permut = (iperm(1) .ne. 0) 
c     
c     start pointer structure to levels 
c     
      nlev   = 0 
c     
c     previous end
c     
      istart = 0 
      ii = 0
c     
c     current end 
c     
      iend = nfirst
c     
c     intialize masks to zero -- except nodes of first level -- 
c     
      do 12 j=1, nfirst 
         mask(riord(j)) = 0 
 12   continue
c-----------------------------------------------------------------------
 13   continue 
c     
 1    nlev = nlev+1
      levels(nlev) = istart + 1
      call add_lvst (istart,iend,nlev,riord,ja,ia,mask,maskval) 
      if (istart .lt. iend) goto 1
 2    ii = ii+1 
      if (ii .le. n) then
         nod = ii         
         if (permut) nod = iperm(nod)          
         if (mask(nod) .eq. maskval) then
c     
c     start a new level
c
            istart = iend
            iend = iend+1 
            riord(iend) = nod
            mask(nod) = 0
            goto 1
         else 
            goto 2
         endif
      endif
c----------------------------------------------------------------------- 
 3    levels(nlev+1) = iend+1 
      do j=1, iend
         mask(riord(j)) = maskval 
      enddo
      return
c----------------------------------------------------------------------- 
c-----end-of-BFS--------------------------------------------------------
      end
c-----------------------------------------------------------------------
      subroutine dblstr(n,ja,ia,ip1,ip2,nfirst,riord,ndom,map,mapptr,
     *     mask,levels,iwk) 
      implicit none
      integer ndom,ja(*),ia(*),ip1,ip2,nfirst,riord(*),map(*),mapptr(*),
     *     mask(*),levels(*),iwk(*),nextdom
c-----------------------------------------------------------------------
c this routine performs a two-way partitioning of a graph using 
c level sets recursively. First a coarse set is found by a
c simple cuthill-mc Kee type algorithm. Them each of the large
c domains is further partitioned into subsets using the same 
c technique. The ip1 and ip2 parameters indicate the desired number 
c number of partitions 'in each direction'. So the total number of
c partitions on return ought to be equal (or close) to ip1*ip2 
c----------------------parameters----------------------------------------
c on entry: 
c---------
c n      = row dimension of matrix == number of vertices in graph
c ja, ia = pattern of matrix in CSR format (the ja,ia arrays of csr data
c          structure)
c ip1    = integer indicating the number of large partitions ('number of
c          paritions in first direction') 
c ip2    = integer indicating the number of smaller partitions, per 
c          large partition, ('number of partitions in second direction') 
c nfirst = number of nodes in the first level that is input in riord 
c riord  = (also an ouput argument). on entry riord contains the labels  
c          of the nfirst nodes that constitute the first level.   
c on return:
c-----------
c ndom   = total number of partitions found 
c map    = list of nodes listed partition by pertition from partition 1
c          to paritition ndom.
c mapptr = pointer array for map. All nodes from position 
c          k1=mapptr(idom),to position k2=mapptr(idom+1)-1 in map belong
c          to partition idom.
c work arrays:
c-------------
c mask   = array of length n, used to hold the partition number of each 
c          node for the first (large) partitioning. 
c          mask is also used as a marker of  visited nodes. 
c levels = integer array of length .le. n used to hold the pointer 
c          arrays for the various level structures obtained from BFS. 
c-----------------------------------------------------------------------
      integer n, j,idom,kdom,jdom,maskval,k,nlev,init,ndp1,numnod
      maskval = 1 
      do j=1, n
         mask(j) = maskval 
      enddo
      iwk(1) = 0 
      call BFS(n,ja,ia,nfirst,iwk,mask,maskval,riord,levels,nlev)      
c      init = riord(1) 
c      call perphn (ja,ia,mask,maskval,init,nlev,riord,levels) 
      call stripes (nlev,riord,levels,ip1,map,mapptr,ndom)
c-----------------------------------------------------------------------
      if (ip2 .eq. 1) return      
      ndp1 = ndom+1
c     
c     pack info into array iwk 
c 
      do j = 1, ndom+1
         iwk(j) = ndp1+mapptr(j)  
      enddo
      do j=1, mapptr(ndom+1)-1
         iwk(ndp1+j) = map(j) 
      enddo
c----------------------------------------------------------------------- 
      do idom=1, ndom 
         do k=mapptr(idom),mapptr(idom+1)-1 
            mask(map(k)) = idom
         enddo
      enddo
      nextdom = 1 
c
c     jdom = counter for total number of (small) subdomains 
c     
      jdom = 1
      mapptr(jdom) = 1 
c----------------------------------------------------------------------- 
      do idom =1, ndom
         maskval = idom
         nfirst = 1
         numnod = iwk(idom+1) - iwk(idom) 
         j = iwk(idom) 
         init = iwk(j) 
         nextdom = mapptr(jdom) 
         call perphn(numnod,ja,ia,init,iwk(j),mask,maskval,
     *        nlev,riord,levels)
         call stripes (nlev,riord,levels,ip2,map(nextdom),
     *        mapptr(jdom),kdom)
         mapptr(jdom) = nextdom
         do j = jdom,jdom+kdom-1
            mapptr(j+1) = nextdom + mapptr(j+1)-1
         enddo
         jdom = jdom + kdom
      enddo
c
      ndom = jdom - 1
      return
      end 
c-----------------------------------------------------------------------
      subroutine perphn(n,ja,ia,init,iperm,mask,maskval,nlev,riord,
     *     levels) 
      implicit none
      integer n,ja(*),ia(*),init,iperm(*),mask(*),maskval,
     *     nlev,riord(*),levels(*)
c-----------------------------------------------------------------------
c     finds a pseudo-peripheral node and does a BFS search from it. 
c-----------------------------------------------------------------------
c see routine  dblstr for description of parameters
c input: 
c------- 
c ja, ia  = list pointer array for the adjacency graph 
c mask    = array used for masking nodes -- see maskval 
c maskval = value to be checked against for determing whether or
c           not a node is masked. If mask(k) .ne. maskval then
c           node k is not considered. 
c init    = init node in the pseudo-peripheral node algorithm. 
c
c output:
c-------
c init    = actual pseudo-peripherial node found. 
c nlev    = number of levels in the final BFS traversal. 
c riord   =  
c levels  = 
c-----------------------------------------------------------------------
      integer j,nlevp,deg,nfirst,mindeg,nod,maskdeg
      nlevp = 0 
 1    continue
      riord(1) = init
      nfirst = 1 
      call BFS(n,ja,ia,nfirst,iperm,mask,maskval,riord,levels,nlev)
      if (nlev .gt. nlevp) then 
         mindeg = levels(nlev+1)-1
         do j=levels(nlev),levels(nlev+1)-1
            nod = riord(j) 
            deg = maskdeg(ja,ia,nod,mask,maskval)
            if (deg .lt. mindeg) then
               init = nod
               mindeg = deg
            endif 
         enddo
         nlevp = nlev 
         goto 1 
      endif
      return
      end
c-----------------------------------------------------------------------
      subroutine add_lvst(istart,iend,nlev,riord,ja,ia,mask,maskval)
      implicit none
      integer istart, iend, maskval, ir, i, k, j
      integer nlev, nod, riord(*), ja(*), ia(*), mask(*) 
c---------------------------------------------------------------------- 
c adds one level set to the previous sets. span all nodes of previous 
c set. Uses Mask to mark those already visited. 
c----------------------------------------------------------------------- 
      nod = iend
      do 25 ir = istart+1,iend 
         i = riord(ir)		
         do 24 k=ia(i),ia(i+1)-1
            j = ja(k)
            if (mask(j) .eq. maskval) then
               nod = nod+1 
               mask(j) = 0
               riord(nod) = j
            endif 
 24      continue
 25   continue
      istart = iend 
      iend   = nod 
      return
c-----------------------------------------------------------------------
      end 
c----------------------------------------------------------------------- 
      subroutine stripes (nlev,riord,levels,ip,map,mapptr,ndom)
      implicit none
      integer nlev,riord(*),levels(nlev+1),ip,map(*),
     *    mapptr(*), ndom
c-----------------------------------------------------------------------
c    this is a post processor to BFS. stripes uses the output of BFS to 
c    find a decomposition of the adjacency graph by stripes. It fills 
c    the stripes level by level until a number of nodes .gt. ip is 
c    is reached. 
c---------------------------parameters-----------------------------------
c on entry: 
c --------
c nlev   = number of levels as found by BFS 
c riord  = reverse permutation array produced by BFS -- 
c levels = pointer array for the level structure as computed by BFS. If 
c          lev is a level number, and k1=levels(lev),k2=levels(lev+1)-1, 
c          then all the nodes of level number lev are:
c                      riord(k1),riord(k1+1),...,riord(k2) 
c  ip    = number of desired partitions (subdomains) of about equal size.
c 
c on return
c ---------
c ndom     = number of subgraphs (subdomains) found 
c map      = node per processor list. The nodes are listed contiguously
c            from proc 1 to nproc = mpx*mpy. 
c mapptr   = pointer array for array map. list for proc. i starts at 
c            mapptr(i) and ends at mapptr(i+1)-1 in array map.
c-----------------------------------------------------------------------
c local variables. 
c
      integer ib,ktr,ilev,k,nsiz,psiz 
      ndom = 1 
      ib = 1
c to add: if (ip .le. 1) then ...
      nsiz = levels(nlev+1) - levels(1) 
      psiz = (nsiz-ib)/max(1,(ip - ndom + 1)) + 1 
      mapptr(ndom) = ib 
      ktr = 0 
      do 10 ilev = 1, nlev
c
c     add all nodes of this level to domain
c     
         do 3 k=levels(ilev), levels(ilev+1)-1
            map(ib) = riord(k)
            ib = ib+1
            ktr = ktr + 1 
            if (ktr .ge. psiz  .or. k .ge. nsiz) then 
               ndom = ndom + 1
               mapptr(ndom) = ib 
               psiz = (nsiz-ib)/max(1,(ip - ndom + 1)) + 1 
               ktr = 0
            endif
c
 3       continue
 10   continue
      ndom = ndom-1
      return 
c-----------------------------------------------------------------------
c-----end-of-stripes----------------------------------------------------
      end
c----------------------------------------------------------------------- 
      subroutine reversp (n, riord) 
      integer n, riord(n) 
c-----------------------------------------------------------------------
c     this routine does an in-place reversing of the permutation array
c     riord -- 
c----------------------------------------------------------------------- 
      integer j, k 
      do 26 j=1,n/2
         k = riord(j) 
         riord(j) = riord(n-j+1)
         riord(n-j+1) = k 
 26   continue 
      return 
      end 
c----------------------------------------------------------------------- 
      integer function maskdeg  (ja,ia,nod,mask,maskval) 
      implicit none 
      integer ja(*),ia(*),nod,mask(*),maskval
c-----------------------------------------------------------------------
      integer deg, k
      deg = 0
      do k =ia(nod),ia(nod+1)-1
         if (mask(ja(k)) .eq. maskval) deg = deg+1 
      enddo
      maskdeg = deg 
      return
      end 
c-----------------------------------------------------------------------
      subroutine rperm (nrow,a,ja,ia,ao,jao,iao,perm,job)
      implicit none
      integer nrow,ja(*),ia(nrow+1),jao(*),iao(nrow+1),perm(nrow),job
      real*8 a(*),ao(*)
      integer j, i, ii, ko, k
c-----------------------------------------------------------------------
c this subroutine permutes the rows of a matrix in CSR format. 
c rperm  computes B = P A  where P is a permutation matrix.  
c the permutation P is defined through the array perm: for each j, 
c perm(j) represents the destination row number of row number j. 
c Youcef Saad -- recoded Jan 28, 1991.
c-----------------------------------------------------------------------
c on entry:
c----------
c n 	= dimension of the matrix
c a, ja, ia = input matrix in csr format
c perm 	= integer array of length nrow containing the permutation arrays
c	  for the rows: perm(i) is the destination of row i in the
c         permuted matrix. 
c         ---> a(i,j) in the original matrix becomes a(perm(i),j) 
c         in the output  matrix.
c
c job	= integer indicating the work to be done:
c 		job = 1	permute a, ja, ia into ao, jao, iao 
c                       (including the copying of real values ao and
c                       the array iao).
c 		job .ne. 1 :  ignore real values.
c                     (in which case arrays a and ao are not needed nor
c                      used).
c
c------------
c on return: 
c------------ 
c ao, jao, iao = input matrix in a, ja, ia format
c note : 
c        if (job.ne.1)  then the arrays a and ao are not used.
c----------------------------------------------------------------------c
c           Y. Saad, May  2, 1990                                      c
c----------------------------------------------------------------------c
      logical values
      values = (job .eq. 1) 
c     
c     determine pointers for output matix. 
c     
      do 50 j=1,nrow
         i = perm(j)
         iao(i+1) = ia(j+1) - ia(j)
 50   continue
c
c get pointers from lengths
c
      iao(1) = 1
      do 51 j=1,nrow
         iao(j+1)=iao(j+1)+iao(j)
 51   continue
c
c copying 
c
      do 100 ii=1,nrow
c
c old row = ii  -- new row = iperm(ii) -- ko = new pointer
c        
         ko = iao(perm(ii)) 
         do 60 k=ia(ii), ia(ii+1)-1 
            jao(ko) = ja(k) 
            if (values) ao(ko) = a(k)
            ko = ko+1
 60      continue
 100  continue
c
      return
c---------end-of-rperm ------------------------------------------------- 
c-----------------------------------------------------------------------
      end
c-----------------------------------------------------------------------
      subroutine cperm (nrow,a,ja,ia,ao,jao,iao,perm,job) 
      integer nrow,ja(*),ia(nrow+1),jao(*),iao(nrow+1),perm(*), job
      real*8 a(*), ao(*) 
c-----------------------------------------------------------------------
c this subroutine permutes the columns of a matrix a, ja, ia.
c the result is written in the output matrix  ao, jao, iao.
c cperm computes B = A P, where  P is a permutation matrix
c that maps column j into column perm(j), i.e., on return 
c      a(i,j) becomes a(i,perm(j)) in new matrix 
c Y. Saad, May 2, 1990 / modified Jan. 28, 1991. 
c-----------------------------------------------------------------------
c on entry:
c----------
c nrow 	= row dimension of the matrix
c
c a, ja, ia = input matrix in csr format. 
c
c perm	= integer array of length ncol (number of columns of A
c         containing the permutation array  the columns: 
c         a(i,j) in the original matrix becomes a(i,perm(j))
c         in the output matrix.
c
c job	= integer indicating the work to be done:
c 		job = 1	permute a, ja, ia into ao, jao, iao 
c                       (including the copying of real values ao and
c                       the array iao).
c 		job .ne. 1 :  ignore real values ao and ignore iao.
c
c------------
c on return: 
c------------ 
c ao, jao, iao = input matrix in a, ja, ia format (array ao not needed)
c
c Notes:
c------- 
c 1. if job=1 then ao, iao are not used.
c 2. This routine is in place: ja, jao can be the same. 
c 3. If the matrix is initially sorted (by increasing column number) 
c    then ao,jao,iao  may not be on return. 
c 
c----------------------------------------------------------------------c
c local parameters:
      integer k, i, nnz
c
      nnz = ia(nrow+1)-1
      do 100 k=1,nnz
         jao(k) = perm(ja(k)) 
 100  continue
c
c     done with ja array. return if no need to touch values.
c
      if (job .ne. 1) return
c
c else get new pointers -- and copy values too.
c 
      do 1 i=1, nrow+1
         iao(i) = ia(i)
 1    continue
c
      do 2 k=1, nnz
         ao(k) = a(k)
 2    continue
c
      return
c---------end-of-cperm-------------------------------------------------- 
c-----------------------------------------------------------------------
      end
c----------------------------------------------------------------------- 
      subroutine dperm (nrow,a,ja,ia,ao,jao,iao,perm,qperm,job)
      integer nrow,ja(*),ia(nrow+1),jao(*),iao(nrow+1),perm(nrow),
     +        qperm(*),job
      real*8 a(*),ao(*) 
c-----------------------------------------------------------------------
c This routine permutes the rows and columns of a matrix stored in CSR
c format. i.e., it computes P A Q, where P, Q are permutation matrices. 
c P maps row i into row perm(i) and Q maps column j into column qperm(j): 
c      a(i,j)    becomes   a(perm(i),qperm(j)) in new matrix
c In the particular case where Q is the transpose of P (symmetric 
c permutation of A) then qperm is not needed. 
c note that qperm should be of length ncol (number of columns) but this
c is not checked. 
c-----------------------------------------------------------------------
c Y. Saad, Sep. 21 1989 / recoded Jan. 28 1991. 
c-----------------------------------------------------------------------
c on entry: 
c---------- 
c n 	= dimension of the matrix
c a, ja, 
c    ia = input matrix in a, ja, ia format
c perm 	= integer array of length n containing the permutation arrays
c	  for the rows: perm(i) is the destination of row i in the
c         permuted matrix -- also the destination of column i in case
c         permutation is symmetric (job .le. 2) 
c
c qperm	= same thing for the columns. This should be provided only
c         if job=3 or job=4, i.e., only in the case of a nonsymmetric
c	  permutation of rows and columns. Otherwise qperm is a dummy
c
c job	= integer indicating the work to be done:
c * job = 1,2 permutation is symmetric  Ao :== P * A * transp(P)
c 		job = 1	permute a, ja, ia into ao, jao, iao 
c 		job = 2 permute matrix ignoring real values.
c * job = 3,4 permutation is non-symmetric  Ao :== P * A * Q 
c 		job = 3	permute a, ja, ia into ao, jao, iao 
c 		job = 4 permute matrix ignoring real values.
c		
c on return: 
c-----------
c ao, jao, iao = input matrix in a, ja, ia format
c
c in case job .eq. 2 or job .eq. 4, a and ao are never referred to 
c and can be dummy arguments. 
c Notes:
c------- 
c  1) algorithm is in place 
c  2) column indices may not be sorted on return even  though they may be 
c     on entry.
c----------------------------------------------------------------------c
c local variables 
      integer locjob, mod
c
c     locjob indicates whether or not real values must be copied. 
c     
      locjob = mod(job,2) 
c
c permute rows first 
c 
      call rperm (nrow,a,ja,ia,ao,jao,iao,perm,locjob)
c
c then permute columns
c
      locjob = 0
c
      if (job .le. 2) then
         call cperm (nrow,ao,jao,iao,ao,jao,iao,perm,locjob) 
      else 
         call cperm (nrow,ao,jao,iao,ao,jao,iao,qperm,locjob) 
      endif 
c     
      return
c-------end-of-dperm----------------------------------------------------
c-----------------------------------------------------------------------
      end
	subroutine csrcoo (nrow,job,nzmax,a,ja,ia,nnz,ao,ir,jc,ierr)
c----------------------------------------------------------------------- 
	real*8 a(*),ao(*) 
	integer ir(*),jc(*),ja(*),ia(*)
c----------------------------------------------------------------------- 
c  Compressed Sparse Row      to      Coordinate 
c----------------------------------------------------------------------- 
c converts a matrix that is stored in coordinate format
c  a, ir, jc into a row general sparse ao, jao, iao format.
c
c on entry: 
c---------
c nrow	= dimension of the matrix.
c job   = integer serving as a job indicator. 
c         if job = 1 fill in only the array ir, ignore jc, and ao.
c         if job = 2 fill in ir, and jc but not ao 
c         if job = 3 fill in everything.
c         The reason why these options are provided is that on return 
c         ao and jc are the same as a, ja. So when job = 3, a and ja are
c         simply copied into ao, jc.  When job=2, only jc and ir are
c         returned. With job=1 only the array ir is returned. Moreover,
c         the algorithm is in place:
c	     call csrcoo (nrow,1,nzmax,a,ja,ia,nnz,a,ia,ja,ierr) 
c         will write the output matrix in coordinate format on a, ja,ia.
c         (Important: note the order in the output arrays a, ja, ia. )
c         i.e., ao can be the same as a, ir can be the same as ia
c         and jc can be the same as ja.
c
c a,
c ja,
c ia    = matrix in compressed sparse row format.
c nzmax = length of space available in ao, ir, jc.
c         the code will stop immediatly if the number of
c         nonzero elements found in input matrix exceeds nzmax.
c 
c on return:
c----------- 
c ao, ir, jc = matrix in coordinate format.
c
c nnz        = number of nonzero elements in matrix.
c ierr       = integer error indicator.
c         ierr .eq. 0 means normal retur
c         ierr .eq. 1 means that the the code stopped 
c         because there was no space in ao, ir, jc 
c         (according to the value of  nzmax).
c 
c------------------------------------------------------------------------
	ierr = 0
	nnz = ia(nrow+1)-1
	if (nnz .gt. nzmax) then
	    ierr = 1
	    return
	endif
c------------------------------------------------------------------------
	goto (3,2,1) job
 1      do 10 k=1,nnz
	       ao(k) = a(k)
 10	    continue
 2       do 11 k=1,nnz
		jc(k) = ja(k)
 11     continue
c copy backward to allow 
 3	do 13 i=nrow,1,-1
	   k1 = ia(i+1)-1
	   k2 = ia(i)
	   do 12 k=k1,k2,-1
	   ir(k) = i
 12	continue
 13	continue
	return
c------------- end of csrcoo ------------------------------------------- 
c----------------------------------------------------------------------- 
	end
      subroutine amudia (nrow,job, a, ja, ia, diag, b, jb, ib)
      real*8 a(*), b(*), diag(nrow) 
      integer ja(*),jb(*), ia(nrow+1),ib(nrow+1) 
c-----------------------------------------------------------------------
c performs the matrix by matrix product B = A * Diag  (in place) 
c-----------------------------------------------------------------------
c on entry:
c ---------
c nrow	= integer. The row dimension of A
c
c job   = integer. job indicator. Job=0 means get array b only
c         job = 1 means get b, and the integer arrays ib, jb.
c
c a,
c ja,
c ia   = Matrix A in compressed sparse row format.
c 
c diag = diagonal matrix stored as a vector dig(1:n)
c
c on return:
c----------
c
c b, 
c jb, 
c ib	= resulting matrix B in compressed sparse row sparse format.
c	    
c Notes:
c-------
c 1)        The column dimension of A is not needed. 
c 2)        algorithm in place (B can take the place of A).
c-----------------------------------------------------------------
      do 1 ii=1,nrow
c     
c     scale each element 
c     
         k1 = ia(ii)
         k2 = ia(ii+1)-1
         do 2 k=k1, k2
            b(k) = a(k)*diag(ja(k)) 
 2       continue
 1    continue
c     
      if (job .eq. 0) return
c     
      ib(1) = ia(1) 
      do 3 ii=1, nrow
         ib(ii) = ia(ii)
         do 31 k=ia(ii),ia(ii+1)-1
            jb(k) = ja(k)
 31      continue
 3    continue
      return
c-----------------------------------------------------------------------
c-----------end-of-amudiag----------------------------------------------
      end 
      subroutine ilutp (n,a,ja,ia,lfil,droptol,permtol,mbloc,alu,jlu,
     *    ju,iwk,wu,wl,jr,jwl,jwu,iperm,ierr)
c-----------------------------------------------------------------------
c     SPARSKIT ROUTINE   --  ILUT with PIVOTING --
c-----------------------------------------------------------------------
c      implicit none
      integer n,ja(*),ia(n+1),lfil,jlu(*),ju(n),jr(n),jwu(n),
     *     jwl(n),iwk,iperm(2*n),ierr
      real*8 a(*), alu(*), wu(n+1), wl(n), droptol
c----------------------------------------------------------------------*
c                      *** ILUT preconditioner ***                     *
c                      ---------------------------                     *
c      incomplete LU factorization with dual truncation mechanism      *
c      VERSION 2 : sorting  done for both L and U.                     *
c                                                                      *
c----------------------------------------------------------------------*
c---- coded by Youcef Saad Sep 8, 1993 . ------------------------------*
c---- Dual drop-off strategy works as follows.                         *
c                                                                      *
c 1) Theresholding in L and U as set by droptol. Any element whose size*
c    is less than some tolerance (relative to the norm of current      *
c    row in u) is dropped.                                             *
c                                                                      *
c 2) Keeping only the largest lfil+il(i) elements in the i-th row      *
c    of L and the largest lfil+iu(i) elements in the i-th row of       *
c    U where il(i), iu(i) are the original number of nonzero           *
c    elements of the L-part and the U-part of the i-th row of A        *
c                                                                      *
c column permuting is used --                                          *
c  see also comments in ilut                                           *
c----------------------------------------------------------------------*
c PARAMETERS
c-----------
c
c on entry:
c==========
c n       = integer. The dimension of the matrix A.
c
c a,ja,ia = matrix stored in Compressed Sparse Row format.
c           ONE RETURN THE COLUMNS OF A ARE PERMUTED.
c
c lfil    = integer. The fill-in parameter. Each row of L and
c           each row of U will have a maximum of lfil elements
c           in addition to their original number of nonzero elements.
c           Thus storage can be determined beforehand.
c           lfil must be .ge. 0.
c
c droptol = tolerance used for dropping elements in L and U.
c           elements are dropped if they are .lt. norm(row) x droptol
c           row = row being eliminated
c
c permtol = tolerance ratio used for determning whether to permute
c           two columns. We will permute two columns only if
c           a(i,j)*permtol .gt. a(i,i) [good values 0.1 to 0.01]
c
c mbloc   = if desired, permuting can be done only within the diagonal
c           blocks of size mbloc. Useful for PDE problems with several
c           degrees of freedom.. If feature not wanted take mbloc=n.
c
c iwk     = integer. The minimum length of arrays alu and jlu
c           to work properly, the code requires that iwk be
c
c                      .ge. nnz + 2*lfil*n + 2
c
c           where nnz = original number of nonzero elements in A.
c           if iwk is not large enough the code will stop prematurely
c           with ierr = -2 or ierr = -3 (see below).
c
c On return:
c===========
c
c alu,jlu = matrix stored in Modified Sparse Row (MSR) format containing
c           the L and U factors together. The diagonal (stored in
c           alu(1:n) ) is inverted. Each i-th row of the alu,jlu matrix
c           contains the i-th row of L (excluding the diagonal entry=1)
c           followed by the i-th row of U.
c
c ju      = integer array of length n containing the pointers to
c           the beginning of each row of U in the matrix alu,jlu.
c iperm   = contains the permutation arrays ..
c           iperm(1:n) = old numbers of unknowns
c           iperm(n+1:2*n) = reverse permutation = new unknowns.
c
c ierr    = integer. Error message with the following meaning.
c           ierr  = 0    --> successful return.
c           ierr .gt. 0  --> zero pivot encountered at step number ierr.
c           ierr  = -1   --> Error. input matrix may be wrong.
c                            (The elimination process has generated a
c                            row in L or U whose length is .gt.  n.)
c           ierr  = -2   --> The matrix L overflows the array al.
c           ierr  = -3   --> The matrix U overflows the array alu.
c           ierr  = -4   --> Illegal value for lfil.
c           ierr  = -5   --> zero row encountered.
c
c work arrays:
c=============
c jr,jwu,jwl 	  = integer work arrays of length n.
c wu, wl          = real work arrays of length n+1, and n resp.
c
c Notes:
c ------
c A must have all nonzero diagonal elements.
c U -- MATRIX   STORED IN UNPERMUTED FORMAT TO AVOID PERMUTATION ARRAYS
c code working -- CODED BY Y. SAAD, SEPT 9, 1993.
c-----------------------------------------------------------------------
c     local variables
c
      integer k,i,j,jrow,ju0,ii,j1,j2,jpos,len,imax,lenu,lenl,
     *     nl,jj,lenl0, lenu0,mbloc,icut
      real*8 s, tmp, tnorm,xmax,xmax0, fact, abs, t, permtol
c
      if (lfil .lt. 0) goto 998
c-------------------------------
c initialize ju0 (points to next element to be added to alu,jlu)
c and pointer.
c-----------------------------------------------------------------------
      ju0 = n+2
      jlu(1) = ju0
c
c  integer double pointer array.
c
      do 1 j=1, n
         jr(j)  = 0
         iperm(j) = j
         iperm(n+j) = j
 1    continue
c-----------------------------------------------------------------------
c  beginning of main loop.
c-----------------------------------------------------------------------
      do 500 ii = 1, n
         j1 = ia(ii)
         j2 = ia(ii+1) - 1
         tnorm = 0.0d0
         do 501 k=j1,j2
            tnorm = tnorm+abs(a(k))
 501     continue
         if (tnorm .eq. 0.0) goto 999
         tnorm = tnorm/(j2-j1+1)
c
c--- unpack L-part and U-part of row of A in arrays wl, wu --
c
         lenu = 1
         lenl = 0
         jwu(1) = ii
         wu(1) = 0.0
         jr(ii) = 1
c-----------------------------------------------------------------------
         do 170  j = j1, j2
            k = iperm(n+ja(j))
            t = a(j)
            if (abs(t) .lt. droptol*tnorm .and. k .ne. ii) goto 170
            if (k .lt. ii) then
               lenl = lenl+1
               jwl(lenl) = k
               wl(lenl) = t
               jr(k) = lenl
            else if (k .eq. ii) then
               wu(1) = t
            else
               lenu = lenu+1
               jwu(lenu) = k
               wu(lenu) = t
               jr(k) = lenu
            endif
 170     continue
c-----------------------------------------------------------------------
         tnorm = tnorm/(j2-j1+1)
         lenl0 = lenl
         lenu0 = lenu
         jj = 0
         nl = 0
c-------------------------------------------------------------------
c---------------------- eliminate previous rows --------------------
c-------------------------------------------------------------------
 150     jj = jj+1
         if (jj .gt. lenl) goto 160
c-------------------------------------------------------------------
c in order to do the elimination in the correct order we need to
c exchange the current row number with the one that has
c smallest column number, among jj,jj+1,...,lenl.
c-------------------------------------------------------------------
         jrow = jwl(jj)
         k = jj
c
c determine smallest column index
c
         do 151 j=jj+1,lenl
            if (jwl(j) .lt. jrow) then
               jrow = jwl(j)
               k = j
            endif
 151     continue
c
c     exchange in jwl
c
         if (k .ne. jj) then
            j = jwl(jj)
            jwl(jj) = jwl(k)
            jwl(k) = j
c
c     exchange in jr
c
            jr(jrow) = jj
            jr(j) = k
c
c     exchange in wl
c
            s = wl(jj)
            wl(jj) = wl(k)
            wl(k) = s
         endif
c-----------------------------------------------------------------------
         if (jrow .ge. ii) goto 160
c
c     get the multiplier for row to be eliminated: jrow
c
         fact = wl(jj)*alu(jrow)
c     zero out element in row by setting jr(jrow) = 0
c
         jr(jrow) = 0
c
         if (abs(fact)*wu(n+2-jrow) .le. droptol*tnorm) goto 150
c-------------------------------------------------------------------
c------------combine current row and row jrow ---------------------
c-------------------------------------------------------------------
         do 203 k = ju(jrow), jlu(jrow+1)-1
            s = fact*alu(k)
c     new column number
            j = iperm(n+jlu(k))
            jpos = jr(j)
c
c     if fill-in element is small then disregard:
c
            if (abs(s) .lt. droptol*tnorm .and. jpos .eq. 0) goto 203
            if (j .ge. ii) then
c
c     dealing with upper part.
c
               if (jpos .eq. 0) then
c     this is a fill-in element
                  lenu = lenu+1
                  if (lenu .gt. n) goto 995
                  jwu(lenu) = j
                  jr(j) = lenu
                  wu(lenu) = - s
               else
c     no fill-in element --
                  wu(jpos) = wu(jpos) - s
               endif
            else
c
c     dealing with lower part.
c
               if (jpos .eq. 0) then
c     this is a fill-in element
                 lenl = lenl+1
                 if (lenl .gt. n) goto 995
                 jwl(lenl) = j
                 jr(j) = lenl
                 wl(lenl) = - s
              else
c     no fill-in element --
                 wl(jpos) = wl(jpos) - s
              endif
           endif
 203	continue
        nl = nl+1
        wl(nl) = fact
        jwl(nl)  = jrow
	goto 150
c----------------------------------------------------------
c------------ update l-matrix -----------------------------
c----------------------------------------------------------
 160    len = min(nl,lenl0+lfil)
c     160    len = min0(nl,lfil)

  	call qsplit (wl,jwl,nl,len)
c
c     store L-part -- in original coordinates ..
c
        do 204 k=1, len
           if (ju0 .gt. iwk) goto 996
           alu(ju0) =  wl(k)
           jlu(ju0) = iperm(jwl(k))
c     jlu(ju0) = jwl(k)
           ju0 = ju0+1
 204    continue
c
c     save pointer to beginning of row ii of U
c
        ju(ii) = ju0
c
c     reset double-pointer jr to zero (L-part - except first
c     jj-1 elements which have already been reset)
c
	do 306 k= jj, lenl
           jr(jwl(k)) = 0
 306	continue
c----------------------------------------------------------
c------------update u-matrix -----------------------------
c----------------------------------------------------------
        len = min(lenu,lenu0+lfil)
c     len = min0(lenu,lfil)
	call qsplit (wu(2), jwu(2), lenu-1,len)
c
        imax = 1
        xmax = abs(wu(imax))
        xmax0 = xmax
c
        icut = ii - 1 + mbloc - mod(ii-1,mbloc)
c
        do k=2,len
           t = abs(wu(k))
           if (t .gt. xmax .and. t*permtol .gt. xmax0 .and.
     *          jwu(k) .le. icut) then
              imax = k
              xmax = t
           endif
        enddo
c
c     exchange wu's
c
        tmp = wu(1)
        wu(1) = wu(imax)
        wu(imax) = tmp
c
c     update iperm and reverse iperm
c
        j = jwu(imax)
        i = iperm(ii)
        iperm(ii) = iperm(j)
        iperm(j) = i
c     reverse iperm
        iperm(n+iperm(ii)) = ii
        iperm(n+iperm(j)) = j
c
        t = abs(wu(k))
        if (len + ju0 .gt. iwk) goto 997
c
c     store U-part in original coordinates
c
        do 302 k=2, len
           jlu(ju0) = iperm(jwu(k))
           alu(ju0) = wu(k)
           t = t + abs(wu(k) )
           ju0 = ju0+1
 302	continue
c
c     save norm in wu (backwards). Norm is in fact average abs value
c
        wu(n+2-ii) = t / (len+1)
c
c     store inverse of diagonal element of u
c
        if (wu(1) .eq. 0.0) wu(1) = (1.0D-4 + droptol)*tnorm
c
        alu(ii) = 1.0d0/ wu(1)
c
c     update pointer to beginning of next row of U.
c
	jlu(ii+1) = ju0
c
c     reset double-pointer jr to zero (U-part)
c
	do 308 k=1, lenu
           jr(jwu(k)) = 0
 308	continue
c-----------------------------------------------------------------------
c     end main loop
c-----------------------------------------------------------------------
 500  continue
c
c     permute all column indices of LU ...
c
c
c     call dvperm(n,alu,iperm(n+1))
c
c     do ii =1, n
c     do k = ju(ii), jlu(ii+1)-1
c     jlu(k) = iperm(jlu(k))
c     enddo
c     enddo
c-----------------------------------------------------------------------
      do k = jlu(1),jlu(n+1)-1
         jlu(k) = iperm(n+jlu(k))
      enddo
c
c     ...and A
c
      do k=1, ia(n+1)-1
         ja(k) = iperm(n+ja(k))
      enddo
c
      ierr = 0
      return
c
c     zero pivot :
c
c     900    ierr = ii
c     return
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
c     insufficient storage in U.
c
 997  ierr = -3
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
c----------------end-of-ilutp-------------------------------------------
c-----------------------------------------------------------------------
      end
      function distdot(n,x,ix,y,iy)
      integer n, ix, iy
      real*8 distdot, x(*), y(*), ddot
      external ddot
      distdot = ddot(n,x,ix,y,iy)
      return
      end
      
