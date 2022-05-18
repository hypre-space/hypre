!     Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
!     HYPRE Project Developers. See the top-level COPYRIGHT file for details.
!
!     SPDX-License-Identifier: (Apache-2.0 OR MIT)

!*****************************************************************************
!  Routines to test HYPRE_ParCSRMatrix Fortran interface
!*****************************************************************************

!--------------------------------------------------------------------------
!  fhypre_parcsrmatrixcreate
!--------------------------------------------------------------------------
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

!--------------------------------------------------------------------------
! HYPRE_ParCSRMatrixDestroy
!--------------------------------------------------------------------------
      subroutine fhypre_parcsrmatrixdestroy (fmatrix)
      integer ierr
      integer*8 fmatrix
   
      call HYPRE_ParCSRMatrixDestroy(fmatrix, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_parcsrmatrixdestroy: error = ', ierr
      endif
  
      return
      end

!--------------------------------------------------------------------------
! HYPRE_ParCSRMatrixInitialize
!--------------------------------------------------------------------------
      subroutine fhypre_parcsrmatrixinitialize (fmatrix)
      integer ierr
      integer*8 fmatrix

      call HYPRE_ParCSRMatrixInitialize(fmatrix, ierr)

      if (ierr .ne. 0) then
         print *, 'fhypre_parcsrmatrixinitialize: error = ', ierr
      endif
  
      return
      end

!--------------------------------------------------------------------------
! HYPRE_ParCSRMatrixRead
!--------------------------------------------------------------------------
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

!--------------------------------------------------------------------------
! HYPRE_ParCSRMatrixPrint
!--------------------------------------------------------------------------
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

!--------------------------------------------------------------------------
! HYPRE_ParCSRMatrixGetComm
!--------------------------------------------------------------------------
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

!--------------------------------------------------------------------------
! HYPRE_ParCSRMatrixGetDims
!--------------------------------------------------------------------------
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

!--------------------------------------------------------------------------
! HYPRE_ParCSRMatrixGetRowPartitioning
!--------------------------------------------------------------------------
      subroutine fhypre_parcsrmatrixgetrowpartit (fmatrix, frow_ptr) 

      integer ierr
      integer*8 fmatrix
      integer*8 frow_ptr

      call HYPRE_ParCSRMatrixGetRowPartiti(fmatrix, frow_ptr, 
     1                                          ierr)

      if (ierr .ne. 0) then
         print *, 'fhypre_parcsrmatrixgetrowpartitioning: error = ',
     1                                                     ierr
      endif
  
      return
      end

!--------------------------------------------------------------------------
! HYPRE_ParCSRMatrixGetColPartitioning
!--------------------------------------------------------------------------
      subroutine fhypre_parcsrmatrixgetcolpartit (fmatrix, fcol_ptr) 

      integer ierr
      integer*8 fmatrix
      integer*8 fcol_ptr

      call HYPRE_ParCSRMatrixGetColPartiti(fmatrix, fcol_ptr, 
     1                                          ierr)

      if (ierr .ne. 0) then
         print *, 'fhypre_parcsrmatrixgetcolpartitioning: error = ',
     1                                                     ierr
      endif
  
      return
      end

!--------------------------------------------------------------------------
! HYPRE_ParCSRMatrixGetLocalRange
!--------------------------------------------------------------------------
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

!--------------------------------------------------------------------------
! HYPRE_ParCSRMatrixGetRow
!--------------------------------------------------------------------------
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

!--------------------------------------------------------------------------
! HYPRE_ParCSRMatrixRestoreRow
!--------------------------------------------------------------------------
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

!--------------------------------------------------------------------------
! HYPRE_CSRMatrixtoParCSRMatrix
!--------------------------------------------------------------------------
      subroutine fhypre_csrmatrixtoparcsrmatrix (fcomm, fA_CSR, 
     1                                           frow_part, fcol_part,
     2                                           fmatrix)

      integer ierr
      integer fcomm
      integer frow_part
      integer fcol_part
      integer*8 fA_CSR
      integer*8 fmatrix

      call HYPRE_CSRMatriXToParCSRMatrix(fcomm, fA_CSR, frow_part, 
     1                                   fcol_part, fmatrix, ierr)

      if (ierr .ne. 0) then
         print *, 'fhypre_csrmatrixtoparcsrmatrix: error = ', ierr
      endif
  
      return
      end

!--------------------------------------------------------------------------
! HYPRE_ParCSRMatrixMatvec
!--------------------------------------------------------------------------
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

!--------------------------------------------------------------------------
! HYPRE_ParCSRMatrixMatvecT
!--------------------------------------------------------------------------
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



!--------------------------------------------------------------------------
! HYPRE_ParVectorCreate
!--------------------------------------------------------------------------
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

!--------------------------------------------------------------------------
! HYPRE_ParMultiVectorCreate
!--------------------------------------------------------------------------
      subroutine fhypre_parmultivectorcreate(fcomm, fsize, fpartion, 
     1                                       fnumvecs, fvector)

      integer ierr
      integer fcomm
      integer fsize
      integer fnumvecs
      integer*8 fvector
      integer*8 fpartion

      call HYPRE_ParMultiVectorCreate(fcomm, fsize, fpartion, fnumvecs,
     1                                fvector, ierr)

      if (ierr .ne. 0) then
         print *, 'fhypre_parmultivectorcreate: error = ', ierr
      endif
  
      return
      end

!--------------------------------------------------------------------------
! HYPRE_ParVectorDestroy
!--------------------------------------------------------------------------
      subroutine fhypre_parvectordestroy (fvector)

      integer ierr
      integer*8 fvector

      call HYPRE_ParVectorDestroy(fvector, ierr)

      if (ierr .ne. 0) then
         print *, 'fhypre_parvectordestroy: error = ', ierr
      endif
  
      return
      end

!--------------------------------------------------------------------------
! HYPRE_ParVectorInitialize
!--------------------------------------------------------------------------
      subroutine fhypre_parvectorinitialize (fvector)
   
      integer ierr
      integer*8 fvector

      call HYPRE_ParVectorInitialize(fvector, ierr)

      if (ierr .ne. 0) then
         print *, 'fhypre_parvectorinitialize: error = ', ierr
      endif
  
      return
      end

!--------------------------------------------------------------------------
! HYPRE_ParVectorRead
!--------------------------------------------------------------------------
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

!--------------------------------------------------------------------------
! HYPRE_ParVectorPrint
!--------------------------------------------------------------------------
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

!--------------------------------------------------------------------------
! HYPRE_ParVectorSetConstantValues
!--------------------------------------------------------------------------
      subroutine fhypre_parvectorsetconstantvalue (fvector, fvalue)

      integer ierr
      double precision fvalue
      integer*8 fvector

      call HYPRE_ParVectorSetConstantValue (fvector, fvalue, ierr)

      if (ierr .ne. 0) then
         print *, 'fhypre_parvectorconstantvalues: error = ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! HYPRE_ParVectorSetRandomValues
!--------------------------------------------------------------------------
      subroutine fhypre_parvectorsetrandomvalues (fvector, fseed)

      integer ierr
      integer fseed
      integer*8 fvector

      call HYPRE_ParVectorSetRandomValues (fvector, fvalue, ierr)

      if (ierr .ne. 0) then
         print *, 'fhypre_parvectorrandomvalues: error = ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! HYPRE_ParVectorCopy
!--------------------------------------------------------------------------
      subroutine fhypre_parvectorcopy (fx, fy)

      integer ierr
      integer*8 fx
      integer*8 fy

      call HYPRE_ParVectorCopy (fx, fy, ierr)

      if (ierr .ne. 0) then
         print *, 'fhypre_parvectorcopy: error = ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! HYPRE_ParVectorCloneShallow
!--------------------------------------------------------------------------
      subroutine fhypre_parvectorcloneshallow (fx)

      integer ierr
      integer*8 fx

      call HYPRE_ParVectorCloneShallow (fx, ierr)

      if (ierr .ne. 0) then
         print *, 'fhypre_parvectorcloneshallow: error = ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! HYPRE_ParVectorScale
!--------------------------------------------------------------------------
      subroutine fhypre_parvectorscale (fvalue, fx)

      integer ierr
      double precision fvalue
      integer*8 fx

      call HYPRE_ParVectorScale (fvalue, fx, ierr)

      if (ierr .ne. 0) then
         print *, 'fhypre_parvectorscale: error = ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! HYPRE_ParVectorAxpy
!--------------------------------------------------------------------------
      subroutine fhypre_parvectoraxpy (fvalue, fx, fy)

      integer ierr
      double precision fvalue
      integer*8 fx
      integer*8 fy

      call HYPRE_ParVectorAxpy (fvalue, fx, fy, ierr)

      if (ierr .ne. 0) then
         print *, 'fhypre_parvectoraxpy: error = ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! HYPRE_ParVectorInnerProd
!--------------------------------------------------------------------------
      subroutine fhypre_parvectorinnerprod (fx, fy, fprod)

      integer ierr
      double precision fprod
      integer*8 fx
      integer*8 fy

      call HYPRE_ParVectorInnerProd (fx, fy, fprod, ierr)

      if (ierr .ne. 0) then
         print *, 'fhypre_parvectorinnerprod: error = ', ierr
      endif

      return
      end



!--------------------------------------------------------------------------
! hypre_ParCSRMatrixGlobalNumRows
!--------------------------------------------------------------------------
      subroutine fhypre_parcsrmatrixglobalnumrow (fmatrix, fnrows)

      integer ierr
      integer fnrows
      integer*8 fmatrix

      call hypre_ParCSRMatrixGlobalNumRows (fmatrix, fnrows, ierr)

      if (ierr .ne. 0) then
         print *, 'fhypre_parcsrmatrixglobalnumrows: error = ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! hypre_ParCSRMatrixRowStarts
!--------------------------------------------------------------------------
      subroutine fhypre_parcsrmatrixrowstarts (fmatrix, frows)

      integer ierr
      integer*8 frows
      integer*8 fmatrix

      call hypre_ParCSRMatrixRowStarts (fmatrix, frows, ierr)

      if (ierr .ne. 0) then
         print *, 'fhypre_parcsrmatrixrowstarts: error = ', ierr
      endif

      return
      end



!--------------------------------------------------------------------------
! hypre_ParVectorSetDataOwner
!--------------------------------------------------------------------------
      subroutine fhypre_parvectorsetdataowner (fv, fown)

      integer ierr
      integer fown
      integer*8 fv

      call hypre_SetParVectorDataOwner (fv, fown, ierr)

      if (ierr .ne. 0) then
         print *, 'fhypre_parvectorsetdataowner: error = ', ierr
      endif

      return
      end
