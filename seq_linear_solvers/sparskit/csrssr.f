      subroutine csrssr (nrow,a,ja,ia,nzmax,ao,jao,iao,ierr)
      real*8 a(*), ao(*), t
      integer ia(*), ja(*), iao(*), jao(*)
c-----------------------------------------------------------------------
c Compressed Sparse Row     to     Symmetric Sparse Row
c----------------------------------------------------------------------- 
c this subroutine extracts the lower triangular part of a matrix.
c It can used as a means for converting a symmetric matrix for 
c which all the entries are stored in sparse format into one
c in which only the lower part is stored. The routine is in place in 
c that the output matrix ao, jao, iao can be overwritten on 
c the  input matrix  a, ja, ia if desired. Csrssr has been coded to
c put the diagonal elements of the matrix in the last position in
c each row (i.e. in position  ao(ia(i+1)-1   of ao and jao) 
c----------------------------------------------------------------------- 
c On entry
c-----------
c nrow  = dimension of the matrix a.
c a, ja, 
c    ia = matrix stored in compressed row sparse format
c
c nzmax = length of arrays ao,  and jao. 
c
c On return:
c----------- 
c ao, jao, 
c     iao = lower part of input matrix (a,ja,ia) stored in compressed sparse 
c          row format format.
c  
c ierr   = integer error indicator. 
c          ierr .eq. 0  means normal return
c          ierr .eq. i  means that the code has stopped when processing
c          row number i, because there is not enough space in ao, jao
c          (according to the value of nzmax) 
c
c----------------------------------------------------------------------- 
      ierr = 0
      ko = 0
c-----------------------------------------------------------------------
      do  7 i=1, nrow
         kold = ko
         kdiag = 0
         do 71 k = ia(i), ia(i+1) -1
            if (ja(k)  .gt. i) goto 71
            ko = ko+1
            if (ko .gt. nzmax) then
               ierr = i
               return
            endif
            ao(ko) = a(k)
            jao(ko) = ja(k)
            if (ja(k)  .eq. i) kdiag = ko
 71      continue
         if (kdiag .eq. 0 .or. kdiag .eq. ko) goto 72
c     
c     exchange
c     
         t = ao(kdiag)
         ao(kdiag) = ao(ko)
         ao(ko) = t
c     
         k = jao(kdiag)
         jao(kdiag) = jao(ko)
         jao(ko) = k
 72      iao(i) = kold+1
 7    continue
c     redefine iao(n+1)
      iao(nrow+1) = ko+1
      return
c--------- end of csrssr ----------------------------------------------- 
c----------------------------------------------------------------------- 
      end
