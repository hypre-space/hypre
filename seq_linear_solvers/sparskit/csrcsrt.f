      subroutine csrcsrt (n, a, ja, ia, ao, jao, iao, job, ierr)
      implicit none
      real*8 a(*), ao(*)
      integer ia(*), ja(*), iao(*), jao(*), job, ierr, n
      integer i, j, int_temp, last_len
      real*8 temp
c-----------------------------------------------------------------------
c Compressed Sparse Row to Compressed Symmetric Sparse Row Transpose
c----------------------------------------------------------------------- 
c This routine transposes a matrix in CSR format into another
c matrix in CSR format.
c Routine is out of place.
c Author: Andrew J. Cleary, May 1997
c----------------------------------------------------------------------- 
c On entry
c-----------
c n  = dimension of the matrix a.
c a, ja, 
c    ia = matrix stored in compressed row sparse format
c job: tells what operation should be performed
c    job =1 ==> numerical values should be put into ao
c    job =2 ==> only structure should be returned (in jao, iao)
c
c On return:
c----------- 
c ao, jao, 
c     iao = transpose of input matrix (a,ja,ia) stored in compressed sparse 
c          row format format.
c  
c ierr   = integer error indicator. 
c          ierr .eq. 0  means normal return
c          ierr .ne. 0  means an error
c
c----------------------------------------------------------------------- 
      ierr = 0
      if ( n .le. 0 ) then
         ierr = -1
         return
      endif
c
      do 10 i=1, n+1
         iao( i ) = 0
 10   continue
c
c     Pass through input matrix finding size of rows in transpose
c
      do 20 i=1, ia( n+1 )-1
         iao( ja( i ) ) = iao( ja( i ) ) + 1
 20   continue
c
c     Translate lengths into pointers
c
      last_len = iao( 1 )
      iao( 1 ) = 1
      do 30 i=2, n+1
         int_temp = iao( i )
         iao( i ) = iao( i-1 ) + last_len
         last_len = int_temp     
 30   continue
c
c     Pass through matrix again inputting into jao and ao (if job=1).
c     Use iao( i ) as pointer into jao and ao for next element of row i.
c
      do 40 i=1, n
         do 50 j=ia(i), ia(i+1)-1
            jao( iao( ja( j ) ) ) = i
            if( job .eq. 1 ) then
               ao( iao( ja( j ) ) ) = a( j )
            endif
            iao( ja( j ) ) = iao( ja( j ) ) + 1
 50      continue
 40   continue
c
c     reset iao pointers
c
      do 60 i=n, 2, -1
         iao( i ) = iao( i-1 )
 60   continue
      iao( 1 ) = 1
c
      return
c--------- end of csrcsrt ----------------------------------------------- 
c----------------------------------------------------------------------- 
      end
