c*****************************************************************************
c  Routines to test HYPRE_ParCSRMatrix Fortran interface
c*****************************************************************************

c--------------------------------------------------------------------------
c  fhypre_parcsrmatrixcreate
c--------------------------------------------------------------------------
      subroutine fhypre_parcsrmatrixcreate(fcomm, fglobal_num_rows, 
     1                                     fglobal_num_cols,
     1                                     frow_starts, fcol_starts, 
     3                                     fnum_cols_offd,
     2                                     fnum_nonzeros_diag, 
     5                                     fnum_nonzeros_offd, fmatrix)
      integer   ierr
      integer   fcomm
      integer   fglobal_num_rows
      integer   fglobal_num_cols
      integer   frow_starts
      integer   fcol_starts
      integer   fnum_cols_offd
      integer   fnum_nonzeros_diag
      integer   fnum_nonzeros_offd
      integer*8 fmatrix

      call HYPRE_ParCSRMatrixCreate(fcomm, fglobal_num_rows, 
     1                              fglobal_num_cols, frow_starts,
     2                              fcol_starts, fnum_cols_offd,
     3                              fnum_nonzeros_diag, 
     4                              fnum_nonzeros_offd, fmatrix, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_parcsrmatrixcreate: error = ', ierr
      endif
  
      return
      end

c--------------------------------------------------------------------------
c HYPRE_ParCSRMatrixDestroy
c--------------------------------------------------------------------------
      subroutine fhypre_parcsrmatrixdestroy (fmatrix)
      integer ierr
      integer*8 fmatrix
   
      call HYPRE_ParCSRMatrixDestroy(fmatrix, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_parcsrmatrixdestroy: error = ', ierr
      endif
  
      return
      end

c--------------------------------------------------------------------------
c HYPRE_ParCSRMatrixInitialize
c--------------------------------------------------------------------------
      subroutine fhypre_parcsrmatrixinitialize (fmatrix)
      integer ierr
      integer*8 fmatrix

      call HYPRE_ParCSRMatrixInitialize(fmatrix, ierr)

      if (ierr .ne. 0) then
         print *, 'fhypre_parcsrmatrixinitialize: error = ', ierr
      endif
  
      return
      end

c--------------------------------------------------------------------------
c HYPRE_ParCSRMatrixRead
c--------------------------------------------------------------------------
      subroutine fhypre_parcsrmatrixread (fcomm, ffile_name, fmatrix)

      integer fcomm
      character*(*) ffile_name
      integer*8 fmatrix
      integer ierr

      call HYPRE_ParCSRMatrixRead(fcomm, ffile_name, fmatrix, ierr)

      if (ierr .ne. 0) then
         print *, 'fhypre_parcsrmatrixread: error = ', ierr
      endif
  
      return
      end

c--------------------------------------------------------------------------
c HYPRE_ParCSRMatrixPrint
c--------------------------------------------------------------------------
      subroutine fhypre_parcsrmatrixprint (fmatrix, fname)

      integer ierr
      integer*8 fmatrix
      character*(*) fname

      call HYPRE_ParCSRMatrixPrint(fmatrix, fname, ierr)

      if (ierr .ne. 0) then
         print *, 'fhypre_parcsrmatrixprint: error = ', ierr
      endif
  
      return
      end

c--------------------------------------------------------------------------
c HYPRE_ParCSRMatrixGetComm
c--------------------------------------------------------------------------
      subroutine fhypre_parcsrmatrixgetcomm (fmatrix, fcomm)

      integer ierr
      integer fcomm
      integer*8 fmatrix

      call HYPRE_ParCSRMatrixGetComm(fmatrix, fcomm)

      if (ierr .ne. 0) then
         print *, 'fhypre_parcsrmatrixgetcomm: error = ', ierr
      endif
  
      return
      end

c--------------------------------------------------------------------------
c HYPRE_ParCSRMatrixGetDims
c--------------------------------------------------------------------------
      subroutine fhypre_parcsrmatrixgetdims (fmatrix, fM, fN)
      
      integer ierr
      integer fM
      integer fN
      integer*8 fmatrix

      call HYPRE_ParCSRMatrixGetDims(fmatrix, fM, fN, ierr)

      if (ierr .ne. 0) then
         print *, 'fhypre_parcsrmatrixgetdims: error = ', ierr
      endif
  
      return
      end

c--------------------------------------------------------------------------
c HYPRE_ParCSRMatrixGetLocalRange
c--------------------------------------------------------------------------
      subroutine fhypre_parcsrmatrixgetlocalrange (fmatrix, frow_start, 
     1                                             frow_end, fcol_start,
     2                                             fcol_end)

      integer ierr
      integer frow_start
      integer frow_end
      integer fcol_start
      integer fcol_end
      integer*8 fmatrix

      call HYPRE_ParCSRMatrixGetLocalRange(fmatrix, frow_start, 
     1                                     frow_end, fcol_start, 
     2                                     fcol_end, ierr)

      if (ierr .ne. 0) then
         print *, 'fhypre_parcsrmatrixgetlocalrange: error = ', ierr
      endif
  
      return
      end

c--------------------------------------------------------------------------
c HYPRE_ParCSRMatrixGetRow
c--------------------------------------------------------------------------
      subroutine fhypre_parcsrmatrixgetrow (fmatrix, frow, fsize, 
     1                                      fcolptr, fvalptr)

      integer ierr
      integer frow
      integer fsize
      integer*8 fcolptr
      integer*8 fvalptr
      integer*8 fmatrix

      call HYPRE_ParCSRMatrixGetRow(fmatrix, frow, fsize, fcolptr, 
     1                              fvalptr, ierr)

      if (ierr .ne. 0) then
         print *, 'fhypre_parcsrmatrixgetrow: error = ', ierr
      endif
  
      return
      end

c--------------------------------------------------------------------------
c HYPRE_ParCSRMatrixRestoreRow
c--------------------------------------------------------------------------
      subroutine fhypre_parcsrmatrixrestorerow (fmatrix, frow, fsize, 
     1                                          fcolptr, fvalptr)

      integer ierr
      integer frow
      integer fsize
      integer*8 fcolptr
      integer*8 fvalptr
      integer*8 fmatrix

      call HYPRE_ParCSRMatrixRestoreRow(fmatrix, frow, fsize, fcolptr,
     1                                  fvalptr, ierr)

      if (ierr .ne. 0) then
         print *, 'fhypre_parcsrmatrixrestorerow: error = ', ierr
      endif
  
      return
      end

c--------------------------------------------------------------------------
c HYPRE_ParCSRMatrixMatvec
c--------------------------------------------------------------------------
      subroutine fhypre_parcsrmatrixmatvec (falpha, fA, fx, fbeta, fy)

      integer ierr
      double precision falpha
      double precision fbeta
      integer*8 fA
      integer*8 fx
      integer*8 fy

      call HYPRE_ParCSRMatrixMatvec(falpha, fA, fx, fbeta, fy, ierr)

      if (ierr .ne. 0) then
         print *, 'fhypre_parcsrmatrixmatvec: error = ', ierr
      endif
  
      return
      end

c--------------------------------------------------------------------------
c HYPRE_ParCSRMatrixMatvecT
c--------------------------------------------------------------------------
      subroutine fhypre_parcsrmatrixmatvect (falpha, fA, fx, fbeta, fy)
      
      integer ierr
      double precision falpha
      double precision fbeta
      integer*8 fA
      integer*8 fx
      integer*8 fy

      call HYPRE_ParCSRMatrixMatvecT(falpha, fA, fx, fbeta, fy, ierr)

      if (ierr .ne. 0) then
         print *, 'fhypre_parcsrmatrixmatvect: error = ', ierr
      endif
  
      return
      end



c--------------------------------------------------------------------------
c HYPRE_ParVectorCreate
c--------------------------------------------------------------------------
      subroutine fhypre_parvectorcreate(fcomm, fsize, fpartion, fvector)

      integer ierr
      integer fcomm
      integer fsize
      integer*8 fvector
      integer*8 fpartion

      call HYPRE_ParVectorCreate(fcomm, fsize, fpartion, fvector, ierr)

      if (ierr .ne. 0) then
         print *, 'fhypre_parvectorcreate: error = ', ierr
      endif
  
      return
      end

c--------------------------------------------------------------------------
c HYPRE_ParVectorDestroy
c--------------------------------------------------------------------------
      subroutine fhypre_parvectordestroy (fvector)

      integer ierr
      integer*8 fvector

      call HYPRE_ParVectorDestroy(fvector, ierr)

      if (ierr .ne. 0) then
         print *, 'fhypre_parvectordestroy: error = ', ierr
      endif
  
      return
      end

c--------------------------------------------------------------------------
c HYPRE_ParVectorInitialize
c--------------------------------------------------------------------------
      subroutine fhypre_parvectorinitialize (fvector)
   
      integer ierr
      integer*8 fvector

      call HYPRE_ParVectorInitialize(fvector, ierr)

      if (ierr .ne. 0) then
         print *, 'fhypre_parvectorinitialize: error = ', ierr
      endif
  
      return
      end

c--------------------------------------------------------------------------
c HYPRE_ParVectorRead
c--------------------------------------------------------------------------
      subroutine fhypre_parvectorread (fcomm, fvector, fname)

      integer ierr
      integer fcomm
      character*(*) fname
      integer*8 fvector

      call HYPRE_ParVectorRead(fcomm, fname, fvector, ierr)

      if (ierr .ne. 0) then
         print *, 'fhypre_parvectorread: error = ', ierr
      endif
  
      return
      end

c--------------------------------------------------------------------------
c HYPRE_ParVectorPrint
c--------------------------------------------------------------------------
      subroutine fhypre_parvectorprint (fvector, fname, fsize)

      integer ierr
      integer fsize
      character*(*) fname
      integer*8 fvector

      call HYPRE_ParVectorPrint (fvector, fname, ierr)

      if (ierr .ne. 0) then
         print *, 'fhypre_parvectorprint: error = ', ierr
      endif

      return
      end
