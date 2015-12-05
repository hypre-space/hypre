cBHEADER**********************************************************************
c Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
c Produced at the Lawrence Livermore National Laboratory.
c This file is part of HYPRE.  See file COPYRIGHT for details.
c
c HYPRE is free software; you can redistribute it and/or modify it under the
c terms of the GNU Lesser General Public License (as published by the Free
c Software Foundation) version 2.1 dated February 1999.
c
c $Revision: 1.5 $
cEHEADER**********************************************************************

c**************************************************
c      Routines to test struct_mv fortran interface
c**************************************************


c**************************************************
c           HYPRE_StructStencil routines
c**************************************************

c******************************************
c      fhypre_structstencilcreate
c******************************************
      subroutine fhypre_structstencilcreate(fdim, fdim1, fstencil)
      integer ierr
      integer fdim
      integer fdim1
      integer*8 fstencil

      call HYPRE_StructStencilCreate(fdim, fdim1, fstencil, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structstencilcreate: error = ', ierr
      endif

      return
      end

c******************************************
c      fhypre_structstencilsetelement
c******************************************
      subroutine fhypre_structstencilsetelement(fstencil, findx,
     1                                          foffset)
      integer ierr
      integer findx
      integer foffset(*)
      integer*8 fstencil

      call HYPRE_StructStencilSetElement(fstencil, findx, foffset,
     1                                   ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structstencilsetelement: error = ', ierr
      endif

      return
      end

c******************************************
c      fhypre_structstencildestroy
c******************************************
      subroutine fhypre_structstencildestroy(fstencil)
      integer ierr
      integer*8 fstencil

      call HYPRE_StructStencilDestroy(fstencil, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structstencildestroy: error = ', ierr
      endif

      return
      end



c**************************************************
c           HYPRE_StructGrid routines
c**************************************************

c******************************************
c      fhypre_structgridcreate
c******************************************
      subroutine fhypre_structgridcreate(fcomm, fdim, fgrid)
      integer ierr
      integer fcomm
      integer fdim
      integer*8 fgrid

      call HYPRE_StructGridCreate(fcomm, fdim, fgrid, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structgridcreate: error = ', ierr
      endif

      return
      end

c******************************************
c      fhypre_structgriddestroy
c******************************************
      subroutine fhypre_structgriddestroy(fgrid)
      integer ierr
      integer*8 fgrid

      call HYPRE_StructGridDestroy(fgrid, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structgriddestroy: error = ', ierr
      endif

      return
      end

c******************************************
c      fhypre_structgridsetextents
c******************************************
      subroutine fhypre_structgridsetextents(fgrid, flower, fupper)
      integer ierr
      integer flower(*)
      integer fupper(*)
      integer*8 fgrid

      call HYPRE_StructGridSetExtents(fgrid, flower, fupper, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structgridsetelement: error = ', ierr
      endif

      return
      end

c******************************************
c      fhypre_structgridsetperiodic
c******************************************
      subroutine fhypre_structgridsetperiodic(fgrid, fperiod)
      integer ierr
      integer fperiod(*)
      integer*8 fgrid

      call HYPRE_StructGridSetPeriodic(fgrid, fperiod, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structgridsetperiodic: error = ', ierr
      endif

      return
      end

c******************************************
c      fhypre_structgridassemble
c******************************************
      subroutine fhypre_structgridassemble(fgrid)
      integer ierr
      integer*8 fgrid

      call HYPRE_StructGridAssemble(fgrid, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structgridassemble: error = ', ierr
      endif

      return
      end

c******************************************
c      fhypre_structgridsetnumghost
c******************************************
      subroutine fhypre_structgridsetnumghost(fgrid, fnumghost)
      integer ierr
      integer fnumghost
      integer*8 fgrid

      call HYPRE_StructGridSetNumGhost(fgrid, fnumghost, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structgridsetnumghost: error = ', ierr
      endif

      return
      end



c**************************************************
c           HYPRE_StructMatrix routines
c**************************************************

c******************************************
c      fhypre_structmatrixcreate
c******************************************
      subroutine fhypre_structmatrixcreate(fcomm, fgrid, fstencil, 
     1                                     fmatrix)
      integer ierr
      integer fcomm
      integer*8 fgrid
      integer*8 fstencil
      integer*8 fmatrix

      call HYPRE_StructMatrixCreate(fcomm, fgrid, fstencil, fmatrix,
     1                              ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structmatrixcreate: error = ', ierr
      endif

      return
      end

c******************************************
c      fhypre_structmatrixdestroy
c******************************************
      subroutine fhypre_structmatrixdestroy(fmatrix)
      integer ierr
      integer*8 fmatrix

      call HYPRE_StructMatrixDestroy(fmatrix, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structmatrixdestroy: error = ', ierr
      endif

      return
      end

c******************************************
c      fhypre_structmatrixinitialize
c******************************************
      subroutine fhypre_structmatrixinitialize(fmatrix)
      integer ierr
      integer*8 fmatrix

      call HYPRE_StructMatrixInitialize(fmatrix, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structmatrixinitialize: error = ', ierr
      endif

      return
      end

c******************************************
c      fhypre_structmatrixsetvalues
c******************************************
      subroutine fhypre_structmatrixsetvalues(fmatrix, fgridindx, 
     1                                        fnumsindx, fsindx, fvals)
      integer ierr
      integer fgridindx(*)
      integer fnumsindx(*)
      integer fsindx(*)
      double precision fvals(*)
      integer*8 fmatrix

      call HYPRE_StructMatrixSetValues(fmatrix, fgridindx, fnumsindx, 
     1                                 fsindx, fvals, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structmatrixsetvalues: error = ', ierr
      endif

      return
      end

c******************************************
c      fhypre_structmatrixsetboxvalues
c******************************************
      subroutine fhypre_structmatrixsetboxvalues(fmatrix, flower,
     1                                           fupper, fnumsindx,
     2                                           fsindx, fvals)
      integer ierr
      integer flower(*)
      integer fupper(*)
      integer fnumsindx(*)
      integer fsindx(*)
      double precision fvals(*)
      integer*8 fmatrix

      call HYPRE_StructMatrixSetBoxValues(fmatrix, flower, fupper,
     1                                    fnumsindx, fsindx, fvals,
     2                                    ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structmatrixsetboxvalues: error = ', ierr
      endif

      return
      end

c******************************************
c      fhypre_structmatrixgetboxvalues
c******************************************
      subroutine fhypre_structmatrixgetboxvalues(fmatrix, flower,
     1                                           fupper, fnumsindx,
     2                                           fsindx, fvals)
      integer ierr
      integer flower(*)
      integer fupper(*)
      integer fnumsindx(*)
      integer fsindx(*)
      double precision fvals(*)
      integer*8 fmatrix

      call HYPRE_StructMatrixGetBoxValues(fmatrix, flower, fupper,
     1                                    fnumsindx, fsindx, fvals,
     2                                    ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structmatrixgetboxvalues: error = ', ierr
      endif

      return
      end

c******************************************
c      fhypre_structmatrixsetconstantentries
c******************************************
      subroutine fhypre_structmatrixsetconstante(fmatrix, fnument,
     1                                           fentries)
      integer ierr
      integer fnument(*)
      integer fentries(*)
      integer*8 fmatrix

      call HYPRE_StructMatrixSetConstantEn(fmatrix, fnument,
     1                                     fentries, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structmatrixsetconstantentries: error =', ierr
      endif

      return
      end

c******************************************
c      fhypre_structmatrixsetconstantvalues
c******************************************
      subroutine fhypre_structmatrixsetconstantv(fmatrix,
     1                                           fnumsindx, fsindx,
     2                                           fvals)
      integer ierr
      integer fnumsindx(*)
      integer fsindx(*)
      double precision fvals(*)
      integer*8 fmatrix

      call HYPRE_StructMatrixSetConstantVa(fmatrix, fnumsindx, 
     1                                         fsindx, fvals, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structmatrixsetconstantvalues: error = ', ierr
      endif

      return
      end

c******************************************
c      fhypre_structmatrixaddtovalues
c******************************************
      subroutine fhypre_structmatrixaddtovalues(fmatrix, fgrdindx,
     1                                          fnumsindx, fsindx,
     2                                          fvals)
      integer ierr
      integer fgrdindx(*)
      integer fnumsindx(*)
      integer fsindx(*)
      double precision fvals(*)
      integer*8 fmatrix

      call HYPRE_StructMatrixAddToValues(fmatrix, fgrdindx,
     1                                   fnumsindx, fsindx, fvals,
     2                                   ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structmatrixaddtovalues: error = ', ierr
      endif

      return
      end

c******************************************
c      fhypre_structmatrixaddtoboxvalues
c******************************************
      subroutine fhypre_structmatrixaddtoboxvalues(fmatrix, filower,
     1                                             fiupper, fnumsindx,
     2                                             fsindx, fvals)
      integer ierr
      integer filower(*)
      integer fiupper(*)
      integer fnumsindx
      integer fsindx(*)
      double precision fvals(*)
      integer*8 fmatrix

      call HYPRE_StructMatrixAddToBoxValues(fmatrix, filower, fiupper,
     1                                      fnumsindx, fsindx, fvals,
     2                                      ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structmatrixaddtovalues: error = ', ierr
      endif

      return
      end

c******************************************
c      fhypre_structmatrixaddtoconstantvalues
c******************************************
      subroutine fhypre_structmatrixaddtoconstant(fmatrix, fnumsindx,
     2                                            fsindx, fvals)
      integer ierr
      integer fnumsindx
      integer fsindx(*)
      double precision fvals(*)
      integer*8 fmatrix

      call HYPRE_StructMatrixSetConstantVa(fmatrix, fnumsindx, 
     1                                         fsindx, fvals, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structmatrixaddtoconstantvalues: error = ',
     1                             ierr
      endif

      return
      end

c******************************************
c      fhypre_structmatrixassemble
c******************************************
      subroutine fhypre_structmatrixassemble(fmatrix)
      integer ierr
      integer*8 fmatrix

      call HYPRE_StructMatrixAssemble(fmatrix, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structmatrixassemble: error = ', ierr
      endif

      return
      end

c******************************************
c      fhypre_structmatrixsetnumghost
c******************************************
      subroutine fhypre_structmatrixsetnumghost(fmatrix, fnumghost)
      integer ierr
      integer fnumghost
      integer*8 fmatrix

      call HYPRE_StructMatrixSetNumGhost(fmatrix, fnumghost, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structmatrixsetnumghost: error = ', ierr
      endif

      return
      end

c******************************************
c      fhypre_structmatrixgetgrid
c******************************************
      subroutine fhypre_structmatrixgetgrid(fmatrix, fgrid)
      integer ierr
      integer*8 fmatrix
      integer*8 fgrid

      call HYPRE_StructMatrixGetGrid(fmatrix, fgrid, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structmatrixgetgrid: error = ', ierr
      endif

      return
      end

c******************************************
c      fhypre_structmatrixsetsymmetric
c******************************************
      subroutine fhypre_structmatrixsetsymmetric(fmatrix, fsymmetric)
      integer ierr
      integer fsymmetric
      integer*8 fmatrix

      call HYPRE_StructMatrixSetSymmetric(fmatrix, fsymmetric, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structmatrixsetsymmetric: error = ', ierr
      endif

      return
      end

c******************************************
c      fhypre_structmatrixprint
c******************************************
      subroutine fhypre_structmatrixprint(fmatrix, fall)
      integer ierr
      integer fall
      integer*8 fmatrix

      call HYPRE_StructMatrixPrint(fmatrix, fall, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structmatrixprint: error = ', ierr
      endif

      return
      end

c******************************************
c      fhypre_structmatrixmatvec
c******************************************
      subroutine fhypre_structmatrixmatvec(falpha, fA, fx, fbeta, fy)
      integer ierr
      integer falpha
      integer fbeta
      integer*8 fA
      integer*8 fx
      integer*8 fy

      call HYPRE_StructMatrixMatvec(falplah, fA, fx, fbeta, fy, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structmatrixmatvec: error = ', ierr
      endif

      return
      end



c**************************************************
c           HYPRE_StructVector routines
c**************************************************

c******************************************
c      fhypre_structvectorcreate
c******************************************
      subroutine fhypre_structvectorcreate(fcomm, fgrid, fvector)
      integer ierr
      integer fcomm
      integer*8 fgrid
      integer*8 fvector

      call HYPRE_StructVectorCreate(fcomm, fgrid, fvector, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structvectorcreate: error = ', ierr
      endif

      return
      end

c******************************************
c      fhypre_structvectordestroy
c******************************************
      subroutine fhypre_structvectordestroy(fvector)
      integer ierr
      integer*8 fvector

      call HYPRE_StructVectorDestroy(fvector, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structvectordestroy: error = ', ierr
      endif

      return
      end

c******************************************
c      fhypre_structvectorinitialize
c******************************************
      subroutine fhypre_structvectorinitialize(fvector)
      integer ierr
      integer*8 fvector

      call HYPRE_StructVectorInitialize(fvector, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structvectorinitialize: error = ', ierr
      endif

      return
      end

c******************************************
c      fhypre_structvectorsetvalues
c******************************************
      subroutine fhypre_structvectorsetvalues(fvector, fgridindx,
     1                                          fvals)
      integer ierr
      integer fgridindx(*)
      double precision fvals(*)
      integer*8 fvector

      call HYPRE_StructVectorSetValues(fvector, fgridindx, fvals, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structvectorsetvalues: error = ', ierr
      endif

      return
      end

c******************************************
c      fhypre_structvectorsetboxvalues
c******************************************
      subroutine fhypre_structvectorsetboxvalues(fvector, flower,
     1                                           fupper, fvals)
      integer ierr
      integer flower(*)
      integer fupper(*)
      double precision fvals(*)
      integer*8 fvector

      call HYPRE_StructVectorSetBoxValues(fvector, flower, fupper,
     1                                    fvals, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structvectorsetboxvalues: error = ', ierr
      endif

      return
      end

c******************************************
c      fhypre_structvectorsetconstantvalues
c******************************************
      subroutine fhypre_structvectorsetconstantv(fvector, fvals)
      integer ierr
      double precision fvals(*)
      integer*8 fvector

      call HYPRE_StructVectorSetConstantVa(fvector, fvals, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structvectorsetconstantvalues: error = ', ierr
      endif

      return
      end

c******************************************
c      fhypre_structvectoraddtovalues
c******************************************
      subroutine fhypre_structvectoraddtovalues(fvector, fgrdindx,
     1                                          fvals)
      integer ierr
      integer fgrdindx(*)
      double precision fvals(*)
      integer*8 fvector

      call HYPRE_StructVectorAddToValues(fvector, fgrdindx, fvals, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structvectoraddtovalues: error = ', ierr
      endif

      return
      end

c******************************************
c      fhypre_structvectoraddtoboxvalues
c******************************************
      subroutine fhypre_structvectoraddtoboxvalu(fvector, flower, 
     1                                             fupper, fvals)
      integer ierr
      integer flower(*)
      integer fupper(*)
      double precision fvals(*)
      integer*8 fvector

      call HYPRE_StructVectorAddToBoxValue(fvector, flower, fupper,
     1                                      fvals, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structvectoraddtoboxvalues: error = ', ierr
      endif

      return
      end

c******************************************
c      fhypre_structvectorscalevalues
c******************************************
      subroutine fhypre_structvectorscalevalues(fvector, ffactor)
      integer ierr
      double precision ffactor
      integer*8 fvector

      call HYPRE_StructVectorScaleValues(fvector, ffactor, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structvectorscalevalues: error = ', ierr
      endif

      return
      end

c******************************************
c      fhypre_structvectorgetvalues
c******************************************
      subroutine fhypre_structvectorgetvalues(fvector, fgrdindx,
     1                                          fvals)
      integer ierr
      integer fgrdindx(*)
      double precision fvals(*)
      integer*8 fvector

      call HYPRE_StructVectorGetValues(fvector, fgrdindx, fvals, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structvectorgetvalues: error = ', ierr
      endif

      return
      end

c******************************************
c      fhypre_structvectorgetboxvalues
c******************************************
      subroutine fhypre_structvectorgetboxvalues(fvector, flower, 
     1                                           fupper, fvals)
      integer ierr
      integer flower(*)
      integer fupper(*)
      double precision fvals(*)
      integer*8 fvector

      call HYPRE_StructVectorGetBoxValues(fvector, flower, fupper,
     1                                    fvals, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structvectorgetboxvalues: error = ', ierr
      endif

      return
      end

c******************************************
c      fhypre_structvectorassemble
c******************************************
      subroutine fhypre_structvectorassemble(fvector)
      integer ierr
      integer*8 fvector

      call HYPRE_StructVectorAssemble(fvector, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structvectorassemble: error = ', ierr
      endif

      return
      end

c******************************************
c      fhypre_structvectorsetnumghost
c******************************************
      subroutine fhypre_structvectorsetnumghost(fvector, fnumghost)
      integer ierr
      integer fnumghost
      integer*8 fvector

      call HYPRE_StructVectorSetNumGhost(fvector, fnumghost, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structvectorsetnumghost: error = ', ierr
      endif

      return
      end

c******************************************
c      fhypre_structvectorcopy
c******************************************
      subroutine fhypre_structvectorcopy(fx, fy)
      integer ierr
      integer*8 fx
      integer*8 fy

      call HYPRE_StructVectorCopy(fx, fy, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structvectorcopy: error = ', ierr
      endif

      return
      end

c******************************************
c      fhypre_structvectorgetmigratecommpkg
c******************************************
      subroutine fhypre_structvectorgetmigrateco(ffromvec, ftovec, 
     1                                                fcommpkg)
      integer ierr
      integer*8 ffromvec
      integer*8 ftovec
      integer*8 fcommpkg

      call HYPRE_StructVectorGetMigrateCom(ffromvec, ftovec, fcommpkg,
     1                                     ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structvectorgetmigratecommpkg: error = ', ierr
      endif

      return
      end

c******************************************
c      fhypre_structvectormigrate
c******************************************
      subroutine fhypre_structvectormigrate(fcommpkg, ffromvec,
     1                                        ftovec)
      integer ierr
      integer*8 ffromvec
      integer*8 ftovec
      integer*8 fcommpkg

      call HYPRE_StructVectorMigrate(fcommpkg, ffromvec, ftovec, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structvectormigrate: error = ', ierr
      endif

      return
      end

c******************************************
c      fhypre_commpkgdestroy
c******************************************
      subroutine fhypre_commpkgdestroy(fcommpkg)
      integer ierr
      integer*8 fcommpkg

      call HYPRE_DestroyCommPkg(fcommpkg, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_commpkgdestroy: error = ', ierr
      endif

      return
      end

c******************************************
c      fhypre_structvectorprint
c******************************************
      subroutine fhypre_structvectorprint(fvector, fall)
      integer ierr
      integer fall
      integer*8 fvector

      call HYPRE_StructVectorPrint(fvector, fall, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structvectorprint: error = ', ierr
      endif

      return
      end
