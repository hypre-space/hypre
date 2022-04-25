!     Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
!     HYPRE Project Developers. See the top-level COPYRIGHT file for details.
!
!     SPDX-License-Identifier: (Apache-2.0 OR MIT)

!*****************************************************************************
! HYPRE_IJMatrix Fortran interface
!*****************************************************************************

!--------------------------------------------------------------------------
! HYPRE_IJMatrixCreate
!--------------------------------------------------------------------------
      subroutine fhypre_ijmatrixcreate(fcomm, filower, fiupper,
     1                                 fjlower, fjupper, fmatrix)
      
      integer ierr
      integer*8 fmatrix
      integer*8 fcomm
      integer filower
      integer fiupper
      integer fjlower
      integer fjupper

      call HYPRE_IJMatrixCreate(fcomm, filower, fiupper, fjlower, 
     1                          fjupper, fmatrix, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_ijmatrixcreate error = ', ierr
      endif
 
      return
      end

!--------------------------------------------------------------------------
! HYPRE_IJMatrixDestroy
!--------------------------------------------------------------------------
      subroutine fhypre_ijmatrixdestroy(fmatrix)
      
      integer ierr
      integer*8 fmatrix

      call HYPRE_IJMatrixDestroy(fmatrix, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_ijmatrixdestroy error = ', ierr
      endif
 
      return
      end

!--------------------------------------------------------------------------
! HYPRE_IJMatrixInitialize
!--------------------------------------------------------------------------
      subroutine fhypre_ijmatrixinitialize(fmatrix)
      
      integer ierr
      integer*8 fmatrix

      call HYPRE_IJMatrixInitialize(fmatrix, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_ijmatrixinitialize error = ', ierr
      endif
 
      return
      end

!--------------------------------------------------------------------------
! HYPRE_IJMatrixAssemble
!--------------------------------------------------------------------------
      subroutine fhypre_ijmatrixassemble(fmatrix)
      
      integer ierr
      integer*8 fmatrix

      call HYPRE_IJMatrixAssemble(fmatrix, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_ijmatrixassemble error = ', ierr
      endif
 
      return
      end

!--------------------------------------------------------------------------
! HYPRE_IJMatrixSetRowSizes
!--------------------------------------------------------------------------
      subroutine fhypre_ijmatrixsetrowsizes(fmatrix, fizes)
      
      integer ierr
      integer*8 fmatrix
      integer fsizes

      call HYPRE_IJMatrixSetRowSizes(fmatrix, fsizes, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_ijmatrixsetrowsizes error = ', ierr
      endif
 
      return
      end

!--------------------------------------------------------------------------
! HYPRE_IJMatrixSetDiagOffdSizes
!--------------------------------------------------------------------------
      subroutine fhypre_ijmatrixsetdiagoffdsizes(fmatrix, fdiag_sizes,
     1                                           foffd_sizes)
      
      integer ierr
      integer*8 fmatrix
      integer fdiag_sizes
      integer foffd_sizes

      call HYPRE_IJMatrixSetDiagOffdSizes(fmatrix, fdiag_sizes, 
     1                                    foffd_sizes, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_ijmatrixsetdiagoffdsizes error = ', ierr
      endif
 
      return
      end

!--------------------------------------------------------------------------
! HYPRE_IJMatrixSetValues
!--------------------------------------------------------------------------
      subroutine fhypre_ijmatrixsetvalues(fmatrix, fnrows, fncols, 
     1                                    frows, fcols, fvalues)
      
      integer ierr
      integer*8 fmatrix
      integer fnrows
      integer fncols
      integer frows
      integer fcols
      double precision fvalues

      call HYPRE_IJMatrixSetValues(fmatrix, fnrows, fncols, frows, 
     1                             fcols, fvalues, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_ijmatrixsetvalues error = ', ierr
      endif
 
      return
      end

!--------------------------------------------------------------------------
! HYPRE_IJMatrixAddToValues
!--------------------------------------------------------------------------
      subroutine fhypre_ijmatrixaddtovalues(fmatrix, fnrows, fncols,
     1                                      frows, fcols, fvalues)
      
      integer ierr
      integer*8 fmatrix
      integer fnrows
      integer fncols
      integer frows
      integer fcols
      double precision fvalues

      call HYPRE_IJMatrixAddToValues(fmatrix, fnrows, fncols, frows,
     1                               fcols, fvalues, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_ijmatrixaddtovalues error = ', ierr
      endif
 
      return
      end

!--------------------------------------------------------------------------
! HYPRE_IJMatrixSetObjectType
!--------------------------------------------------------------------------
      subroutine fhypre_ijmatrixsetobjecttype(fmatrix, ftype)
      
      integer ierr
      integer*8 fmatrix
      integer ftype

      call HYPRE_IJMatrixSetObjectType(fmatrix, ftype, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_ijmatrixsetobjecttype error = ', ierr
      endif
 
      return
      end

!--------------------------------------------------------------------------
! HYPRE_IJMatrixGetObjectType
!--------------------------------------------------------------------------
      subroutine fhypre_ijmatrixgetobjecttype(fmatrix, ftype)
      
      integer ierr
      integer*8 fmatrix
      integer ftype

      call HYPRE_IJMatrixGetObjectType(fmatrix, ftype, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_ijmatrixgetobjecttype error = ', ierr
      endif
 
      return
      end

!--------------------------------------------------------------------------
! HYPRE_IJMatrixGetObject
!--------------------------------------------------------------------------
      subroutine fhypre_ijmatrixgetobject(fmatrix, fobject)
      
      integer ierr
      integer*8 fmatrix
      integer*8 fobject

      call HYPRE_IJMatrixGetObject(fmatrix, fobject, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_ijmatrixgetobject error = ', ierr
      endif
 
      return
      end

!--------------------------------------------------------------------------
! HYPRE_IJMatrixRead
!--------------------------------------------------------------------------
      subroutine fhypre_ijmatrixread(ffilename, fcomm, fobject_type,
     1                               fmatrix)
      
      integer ierr
      integer*8 fmatrix
      integer*8 fcomm
      integer fobject_type
      character*(*) ffilename

      call HYPRE_IJMatrixRead(ffilename, fcomm, fobject_type, fmatrix,
     1                        ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_ijmatrixread error = ', ierr
      endif
 
      return
      end

!--------------------------------------------------------------------------
! HYPRE_IJMatrixPrint
!--------------------------------------------------------------------------
      subroutine fhypre_ijmatrixprint(fmatrix, ffilename)
      
      integer ierr
      integer*8 fmatrix
      character*(*) ffilename

      call HYPRE_IJMatrixPrint(fmatrix, ffilename, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_ijmatrixprint error = ', ierr
      endif
 
      return
      end



!--------------------------------------------------------------------------
! hypre_IJMatrixSetObject
!--------------------------------------------------------------------------
      subroutine fhypre_ijmatrixsetobject(fmatrix, fobject)
      
      integer ierr
      integer*8 fmatrix
      integer*8 fobject

      call hypre_IJMatrixSetObject(fmatrix, fobject, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_ijmatrixsetobject error = ', ierr
      endif
 
      return
      end



!--------------------------------------------------------------------------
! HYPRE_IJVectorCreate
!--------------------------------------------------------------------------
      subroutine fhypre_ijvectorcreate(fcomm, fjlower, fjupper, fvector)
      
      integer ierr
      integer*8 fvector
      integer fcomm
      integer fjlower
      integer fjupper

      call HYPRE_IJVectorCreate(fcomm, fjlower, fjupper, fvector, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_ijvectorcreate error = ', ierr
      endif
 
      return
      end

!--------------------------------------------------------------------------
! HYPRE_IJVectorDestroy
!--------------------------------------------------------------------------
      subroutine fhypre_ijvectordestroy(fvector)
      
      integer ierr
      integer*8 fvector

      call HYPRE_IJVectorDestroy(fvector, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_ijvectordestroy error = ', ierr
      endif
 
      return
      end

!--------------------------------------------------------------------------
! HYPRE_IJVectorInitialize
!--------------------------------------------------------------------------
      subroutine fhypre_ijvectorinitialize(fvector)
      
      integer ierr
      integer*8 fvector

      call HYPRE_IJVectorInitialize(fvector, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_ijvectorinitialize error = ', ierr
      endif
 
      return
      end

!--------------------------------------------------------------------------
! HYPRE_IJVectorSetValues
!--------------------------------------------------------------------------
      subroutine fhypre_ijvectorsetvalues(fvector, fnum_values, 
     1                                    findices, fvalues)
      
      integer ierr
      integer*8 fvector
      integer fnum_values
      integer findices
      double precision fvalues

      call HYPRE_IJVectorSetValues(fvector, fnum_values, findices,
     1                             fvalues, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_ijvectorsetvalues error = ', ierr
      endif
 
      return
      end

!--------------------------------------------------------------------------
! HYPRE_IJVectorAddToValues
!--------------------------------------------------------------------------
      subroutine fhypre_ijvectoraddtovalues(fvector, fnum_values,
     1                                      findices, fvalues)
      
      integer ierr
      integer*8 fvector
      integer fnum_values
      integer findices
      double precision fvalues

      call HYPRE_IJVectorAddToValues(fvector, fnum_values, findices,
     1                               fvalues, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_ijvectoraddtovalues error = ', ierr
      endif
 
      return
      end

!--------------------------------------------------------------------------
! HYPRE_IJVectorAssemble
!--------------------------------------------------------------------------
      subroutine fhypre_ijvectorassemble(fvector)
      
      integer ierr
      integer*8 fvector

      call HYPRE_IJVectorAssemble(fvector , ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_ijvectorassemble error = ', ierr
      endif
 
      return
      end

!--------------------------------------------------------------------------
! HYPRE_IJVectorGetValues
!--------------------------------------------------------------------------
      subroutine fhypre_ijvectorgetvalues(fvector, fnum_values, 
     1                                    findices, fvalues)
      
      integer ierr
      integer*8 fvector
      integer fnum_values
      integer findices
      double precision fvalues

      call HYPRE_IJVectorGetValues(fvector, fnum_values, findices,
     1                             fvalues, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_ijvectorgetvalues error = ', ierr
      endif
 
      return
      end

!--------------------------------------------------------------------------
! HYPRE_IJVectorSetObjectType
!--------------------------------------------------------------------------
      subroutine fhypre_ijvectorsetobjecttype(fvector, ftype)
      
      integer ierr
      integer*8 fvector
      integer ftype

      call HYPRE_IJVectorSetObjectType(fvector, ftype, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_ijvectorsetobjecttype error = ', ierr
      endif
 
      return
      end

!--------------------------------------------------------------------------
! HYPRE_IJVectorGetObjectType
!--------------------------------------------------------------------------
      subroutine fhypre_ijvectorgetobjecttype(fvector, ftype)
      
      integer ierr
      integer*8 fvector
      integer ftype

      call HYPRE_IJVectorGetObjectType(fvector, ftype, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_ijvectorgetobjecttype error = ', ierr
      endif
 
      return
      end

!--------------------------------------------------------------------------
! HYPRE_IJVectorGetObject
!--------------------------------------------------------------------------
      subroutine fhypre_ijvectorgetobject(fvector, fobject)
      
      integer ierr
      integer*8 fvector
      integer*8 fobject

      call HYPRE_IJVectorGetObject(fvector, fobject, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_ijvectorgetobject error = ', ierr
      endif
 
      return
      end

!--------------------------------------------------------------------------
! HYPRE_IJVectorRead
!--------------------------------------------------------------------------
      subroutine fhypre_ijvectorread(ffilename, fcomm, fobject_type,
     1                               fvector)
      
      integer ierr
      integer*8 fvector
      integer*8 fcomm
      integer fobject_type
      character*(*) ffilename

      call HYPRE_IJVectorRead(ffilename, fcomm, fobject_type, fvector,
     1                        ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_ijvectorread error = ', ierr
      endif
 
      return
      end

!--------------------------------------------------------------------------
! HYPRE_IJVectorPrint
!--------------------------------------------------------------------------
      subroutine fhypre_ijvectorprint(fvector, ffilename)
      
      integer ierr
      integer*8 fvector
      character*(*) ffilename

      call HYPRE_IJVectorPrint(fvector, ffilename, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_ijvectorprint error = ', ierr
      endif
 
      return
      end
