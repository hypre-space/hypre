c/*BHEADER*******************************************************************
c * (c) 1996   The Regents of the University of California
c *
c * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
c * notice, contact person, and disclaimer.
c *
c * $Revision$
c *******************************************************************EHEADER*/

      subroutine equate(m, n, 
     &                  matrix_old, matrix_new)
c
c -----------------------------------------------------------------------
c This is an example of a F77 code illustrating the indentation used
c for CASC codes. This code does not illustrate issues related to
c documentation or error handling.
c 
c The most important item here is consistent indentation of the following
c structures:
c    - do loops
c    - if-then-else-endif statements
c    - do-while statements
c In each of the above, we recommend indentation by 3 spaces - 2 is sometimes
c difficult to detect, and 4 can be too much in deeply nested statements.
c 
c The following are left to the discretion of a project. The only constraint
c is that the standard be applied consistently to all F77/F90 codes within
c a project.
c    - upper/lower case for codes
c    - upper/lower/mixed case for comments
c    - whether a code has blank lines or blank comments (a line with
c      c in the first column)
c    - where blank lines should be added to improve the modularity of the 
c      code
c 
c Note that this code does something nonsensical - it is mainly for the
c illustration of indentation.
c
c -----------------------------------------------------------------------
c 

      implicit none

      integer m, n
      integer i, j

      real*8 matrix_new(m,n), matrix_old(m,n)

      if (m .gt. n) then 
c
         do j = 1, n
            do i = 1, m
               matrix_new(i,j) = matrix_old(i,j)
            end do
         end do
c 
      else 
c
         do j = 1, n
            do i = 1, m
               matrix_new(i,j) = - matrix_old(i,j)
            end do
         end do
c
      end if

      return
      end
