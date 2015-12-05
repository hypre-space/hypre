cBHEADER**********************************************************************
c Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
c Produced at the Lawrence Livermore National Laboratory.
c This file is part of HYPRE.  See file COPYRIGHT for details.
c
c HYPRE is free software; you can redistribute it and/or modify it under the
c terms of the GNU Lesser General Public License (as published by the Free
c Software Foundation) version 2.1 dated February 1999.
c
c $Revision: 1.4 $
cEHEADER**********************************************************************

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
c HYPRE_ParCSRMatrixGetRowPartitioning
c--------------------------------------------------------------------------
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

c--------------------------------------------------------------------------
c HYPRE_ParCSRMatrixGetColPartitioning
c--------------------------------------------------------------------------
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
c HYPRE_CSRMatrixtoParCSRMatrix
c--------------------------------------------------------------------------
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
c HYPRE_ParMultiVectorCreate
c--------------------------------------------------------------------------
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

c--------------------------------------------------------------------------
c HYPRE_ParVectorSetConstantValues
c--------------------------------------------------------------------------
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

c--------------------------------------------------------------------------
c HYPRE_ParVectorSetRandomValues
c--------------------------------------------------------------------------
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

c--------------------------------------------------------------------------
c HYPRE_ParVectorCopy
c--------------------------------------------------------------------------
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

c--------------------------------------------------------------------------
c HYPRE_ParVectorCloneShallow
c--------------------------------------------------------------------------
      subroutine fhypre_parvectorcloneshallow (fx)

      integer ierr
      integer*8 fx

      call HYPRE_ParVectorCloneShallow (fx, ierr)

      if (ierr .ne. 0) then
         print *, 'fhypre_parvectorcloneshallow: error = ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_ParVectorScale
c--------------------------------------------------------------------------
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

c--------------------------------------------------------------------------
c HYPRE_ParVectorAxpy
c--------------------------------------------------------------------------
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

c--------------------------------------------------------------------------
c HYPRE_ParVectorInnerProd
c--------------------------------------------------------------------------
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



c--------------------------------------------------------------------------
c hypre_ParCSRMatrixGlobalNumRows
c--------------------------------------------------------------------------
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

c--------------------------------------------------------------------------
c hypre_ParCSRMatrixRowStarts
c--------------------------------------------------------------------------
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



c--------------------------------------------------------------------------
c hypre_ParVectorSetDataOwner
c--------------------------------------------------------------------------
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
