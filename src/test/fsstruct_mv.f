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

c***************************************************************************
c HYPRE_SStruct fortran interface
c***************************************************************************


c***************************************************************************
c              HYPRE_SStructGraph routines
c***************************************************************************

c-------------------------------------------------------------------------
c HYPRE_SStructGraphCreate
c-------------------------------------------------------------------------
      subroutine fhypre_sstructgraphcreate(fcomm, fgrid, fgraphptr)
     
      integer ierr
      integer fcomm
      integer*8 fgrid
      integer*8 fgraphptr

      call HYPRE_SStructGraphCreate(fcomm, fgrid, fgraphptr, ierr)

      if (ierr .ne. 0) then
         print *, ' fhypre_sstructgraphcreate error = ', ierr
      endif

      return
      end


c--------------------------------------------------------------------------
c HYPRE_SStructGraphDestroy
c--------------------------------------------------------------------------
      subroutine fhypre_sstructgraphdestroy(fgraph)
      
      integer ierr
      integer*8 fgraph

      call HYPRE_SStructGraphDestroy(fgraph, ierr)

      if (ierr .ne. 0) then
         print *, ' fhypre_sstructgraphdestroy error = ', ierr
      endif

      return
      end


c--------------------------------------------------------------------------
c HYPRE_SStructGraphSetStencil
c--------------------------------------------------------------------------
      subroutine fhypre_sstructgraphsetstencil(fgraph, fpart, fvar, 
     1                                         fstencil)

      integer ierr
      integer part
      integer var
      integer*8 fgraph
      integer*8 fstencil

      call HYPRE_SStructGraphSetStencil(fgraph, fpart, fvar, fstencil, 
     1                                  ierr)

      if (ierr .ne. 0) then
         print *, ' fhypre_sstructgraphsetstencil error = ', ierr
      endif

      return
      end


c--------------------------------------------------------------------------
c  HYPRE_SStructGraphAddEntries-
c    THIS IS FOR A NON-OVERLAPPING GRID GRAPH.
c--------------------------------------------------------------------------
      subroutine fhypre_sstructgraphaddentries(fgraph, fpart, findex, 
     1                                         fvar, fto_part,
     1                                         fto_index, fto_var)

      integer ierr
      integer fpart
      integer findex
      integer fvar
      integer fto_part
      integer fto_index
      integer fto_var
      integer*8 fgraph

      call HYPRE_SStructGraphAddEntries(fgraph, fpart, findex, fvar,
     1                                  fto_part, fto_index, fto_var, 
     2                                  ierr)

      if (ierr .ne. 0) then
         print *, ' fhypre_sstructgraphaddedntries error = ', ierr
      endif

      return
      end


c--------------------------------------------------------------------------
c  HYPRE_SStructGraphAssemble
c--------------------------------------------------------------------------
      subroutine fhypre_sstructgraphassemble(fgraph)
     
      integer ierr
      integer*8 fgraph

      call HYPRE_SStructGraphAssemble(fgraph, ierr)

      if (ierr .ne. 0) then
         print *, ' fhypre_sstructgraphassemble error = ', ierr
      endif

      return
      end


c--------------------------------------------------------------------------
c  HYPRE_SStructGraphSetObjectType
c--------------------------------------------------------------------------
      subroutine fhypre_sstructgraphsetobjecttyp(fgraph, ftype)
                                                 
      integer ierr
      integer ftype
      integer*8 fgraph

      call HYPRE_SStructGraphSetObjectType(fgraph, ftype, ierr)

      if (ierr .ne. 0) then
         print *, ' fhypre_sstructgraphsetobjecttype error = ', ierr
      endif

      return
      end




c***************************************************************************
c              HYPRE_SStructGrid routines
c***************************************************************************

c--------------------------------------------------------------------------
c  HYPRE_SStructGridCreate
c--------------------------------------------------------------------------
      subroutine fhypre_sstructgridcreate(fcomm, fndim, fnparts,
     1                                    fgridptr)
                                         
      integer ierr
      integer fcomm
      integer fndim
      integer fnparts
      integer*8 fgridptr

      call HYPRE_SStructGridCreate(fcomm, fndim, fnparts, fgridptr, 
     1                             ierr)

      if (ierr .ne. 0) then
         print *, ' fhypre_sstructgridcreate error = ', ierr
      endif

      return
      end


c--------------------------------------------------------------------------
c   HYPRE_SStructGridDestroy
c--------------------------------------------------------------------------
      subroutine fhypre_sstructgriddestroy(fgrid)

      integer ierr
      integer*8 fgrid

      call HYPRE_SStructGridDestroy(fgrid, ierr)

      if (ierr .ne. 0) then
         print *, ' fhypre_sstructgriddestroy error = ', ierr
      endif

      return
      end


c--------------------------------------------------------------------------
c   HYPRE_SStructGridSetExtents
c--------------------------------------------------------------------------
      subroutine fhypre_sstructgridsetextents(fgrid, fpart, filower, 
     1                                        fiupper)

      integer ierr
      integer fpart
      integer filower
      integer fiupper
      integer*8 fgrid

      call HYPRE_SStructGridSetExtents(fgrid, fpart, filower, fiupper, 
     1                                 ierr)

      if (ierr .ne. 0) then
         print *, ' fhypre_sstructgridsetextents error = ', ierr
      endif

      return
      end


c--------------------------------------------------------------------------
c   HYPRE_SStructGridSetVariables
c--------------------------------------------------------------------------
      subroutine fhypre_sstructgridsetvariables(fgrid, fpart, fnvars, 
     1                                          fvartypes)

      integer ierr
      integer fpart
      integer fnvars
      integer*8 fgrid
      integer*8 fvartypes

      call HYPRE_SStructGridSetVariables(fgrid, fpart, fnvars, 
     1                                   fvartypes, ierr)

      if (ierr .ne. 0) then
         print *, ' fhypre_sstructgridsetvariables error = ', ierr
      endif

      return
      end


c--------------------------------------------------------------------------
c   HYPRE_SStructGridSetVariable
c          Like HYPRE_SStructGridSetVariables except only one variable
c          is done at a time; fnvars needed for memory allocation.
c--------------------------------------------------------------------------
      subroutine fhypre_sstructgridsetvariable(fgrid, fpart, fvar, 
     1                                         fnvars, fvartype)

      integer ierr
      integer fpart
      integer fvar
      integer fnvars
      integer*8 fgrid
      integer*8 fvartype

      call HYPRE_SStructGridSetVariable(fgrid, fpart, fvar, fnvars, 
     1                                   fvartype, ierr)

      if (ierr .ne. 0) then
         print *, ' fhypre_sstructgridsetvariable error = ', ierr
      endif

      return
      end


c--------------------------------------------------------------------------
c   HYPRE_SStructGridAddVariables
c--------------------------------------------------------------------------
      subroutine fhypre_sstructgridaddvariables(fgrid, fpart, findex,
     1                                          fnvars, fvartypes)

      integer ierr
      integer fpart
      integer findex
      integer fnvars
      integer*8 fgrid
      integer*8 fvartypes

      call HYPRE_SStructGridAddVariables(fgrid, fpart, findex, fnvars,
     1                                   fvartypes, ierr)

      if (ierr .ne. 0) then
         print *, ' fhypre_sstructgridaddvariables error = ', ierr
      endif

      return
      end


c--------------------------------------------------------------------------
c   HYPRE_SStructGridAddVariable
c          Like HYPRE_SStructGridAddVariables except only one variable
c          is done at a time.
c--------------------------------------------------------------------------
      subroutine fhypre_sstructgridaddvariable(fgrid, fpart, findex,
     1                                         fvar,  fvartype)

      integer ierr
      integer fpart
      integer findex
      integer fvar
      integer*8 fgrid
      integer*8 fvartype

      call HYPRE_SStructGridAddVariable(fgrid, fpart, findex, fvar,
     1                                   fvartype, ierr)

      if (ierr .ne. 0) then
         print *, ' fhypre_sstructgridaddvariable error = ', ierr
      endif

      return
      end


c--------------------------------------------------------------------------
c   HYPRE_SStructGridSetNeighborBox
c--------------------------------------------------------------------------
      subroutine fhypre_sstructgridsetneighborbo(fgrid, fpart, filower,
     1                                            fiupper, fnbor_part,
     2                                            fnbor_ilower,
     3                                            fnbor_iupper,
     4                                            findex_map)

      integer ierr
      integer fpart
      integer filower
      integer fiupper
      integer fnbor_part
      integer fnbor_ilower
      integer fnbor_iupper
      integer findex_map
      integer*8 fgrid

      call HYPRE_SStructGridSetNeighborBox(fgrid, fpart, filower, 
     1                                     fiupper, fnbor_part,
     2                                     fnbor_ilower, fnbor_iupper,
     3                                     findex_map, ierr)

      if (ierr .ne. 0) then
         print *, ' fhypre_sstructgridsetneighborbox error = ', ierr
      endif

      return
      end


c--------------------------------------------------------------------------
c   HYPRE_SStructGridAssemble
c--------------------------------------------------------------------------
      subroutine fhypre_sstructgridassemble(fgrid)

      integer ierr
      integer*8 fgrid

      call HYPRE_SStructGridAssemble(fgrid, ierr)

      if (ierr .ne. 0) then
         print *, ' fhypre_sstructgridassemble error = ', ierr
      endif

      return
      end


c--------------------------------------------------------------------------
c   HYPRE_SStructGridSetPeriodic
c--------------------------------------------------------------------------
      subroutine fhypre_sstructgridsetperiodic(fgrid, fpart, fperiodic)

      integer ierr
      integer fpart
      integer fperiodic
      integer*8 fgrid

      call HYPRE_SStructGridSetPeriodic(fgrid, fpart, fperiodic, ierr)

      if (ierr .ne. 0) then
         print *, ' fhypre_sstructgridsetperiodic error = ', ierr
      endif

      return
      end


c--------------------------------------------------------------------------
c   HYPRE_SStructGridSetNumGhost
c--------------------------------------------------------------------------
      subroutine fhypre_sstructgridsetnumghost(fgrid, fnum_ghost)

      integer ierr
      integer fnumghost
      integer*8 fgrid

      call HYPRE_SStructGridSetNumGhost(fgrid, fnum_ghost, ierr)       

      if (ierr .ne. 0) then
         print *, ' fhypre_sstructgridsetnumghost error = ', ierr
      endif

      return
      end




c***************************************************************************
c              HYPRE_SStructMatrix routines
c***************************************************************************

c--------------------------------------------------------------------------
c   HYPRE_SStructMatrixCreate
c--------------------------------------------------------------------------
      subroutine fhypre_sstructmatrixcreate(fcomm, fgraph, fmatrix_ptr)

      integer ierr
      integer fcomm
      integer*8 fgraph
      integer*8 fmatrix_ptr

      call HYPRE_SStructMatrixCreate(fcomm, fgraph, fmatrix_ptr, ierr)

      if (ierr .ne. 0) then
         print *, ' fhypre_sstructmatrixcreate error = ', ierr
      endif

      return
      end


c--------------------------------------------------------------------------
c   HYPRE_SStructMatrixDestroy
c--------------------------------------------------------------------------
      subroutine fhypre_sstructmatrixdestroy(fmatrix)

      integer ierr
      integer*8 fmatrix

      call HYPRE_SStructMatrixDestroy(fmatrix, ierr)

      if (ierr .ne. 0) then
         print *, ' fhypre_sstructmatrixdestroy error = ', ierr
      endif

      return
      end


c--------------------------------------------------------------------------
c   HYPRE_SStructMatrixInitialize
c--------------------------------------------------------------------------
      subroutine fhypre_sstructmatrixinitialize(fmatrix)

      integer ierr
      integer*8 fmatrix

      call HYPRE_SStructMatrixInitialize(fmatrix, ierr)

      if (ierr .ne. 0) then
         print *, ' fhypre_sstructmatrixinitialize error = ', ierr
      endif

      return
      end


c--------------------------------------------------------------------------
c   HYPRE_SStructMatrixSetValues
c--------------------------------------------------------------------------
      subroutine fhypre_sstructmatrixsetvalues(fmatrix, fpart, findex, 
     1                                         fvar, fnentries, 
     2                                         fentries, fvalues)

      integer ierr
      integer fpart
      integer findex
      integer fvar
      integer fnentries
      integer fentries
      integer*8 fmatrix
      double precision fvalues

      call HYPRE_SStructMatrixSetValues(fmatrix, fpart, findex, fvar, 
     1                                  fnentries, fentries, fvalues, 
     2                                  ierr)

      if (ierr .ne. 0) then
         print *, ' fhypre_sstructmatrixsetvalues error = ', ierr
      endif

      return
      end


c--------------------------------------------------------------------------
c   HYPRE_SStructMatrixSetBoxValues
c--------------------------------------------------------------------------
      subroutine fhypre_sstructmatrixsetboxvalue(fmatrix, fpart, 
     1                                            filower, fiupper, 
     2                                            fvar, fnentries, 
     3                                            fentries, fvalues)

      integer ierr
      integer fpart
      integer filower
      integer fiupper
      integer fvar
      integer fnentries
      integer fentries
      integer*8 fmatrix
      double precision fvalues

      call HYPRE_SStructMatrixSetBoxValues(fmatrix, fpart, filower, 
     1                                     fiupper, fvar, fnentries, 
     2                                     fentries, fvalues, ierr)

      if (ierr .ne. 0) then
         print *, ' fhypre_sstructmatrixsetboxvalues error = ', ierr
      endif

      return
      end


c--------------------------------------------------------------------------
c   HYPRE_SStructMatrixGetValues
c--------------------------------------------------------------------------
      subroutine fhypre_sstructmatrixgetvalues(fmatrix, fpart, findex, 
     1                                         fvar, fnentries, 
     2                                         fentries, fvalues)

      integer ierr
      integer fpart
      integer findex
      integer fvar
      integer fnentries
      integer fentries
      integer*8 fmatrix
      double precision fvalues

      call HYPRE_SStructMatrixGetValues(fmatrix, fpart, findex, fvar, 
     1                                  fnentries, fentries, fvalues, 
     2                                  ierr)

      if (ierr .ne. 0) then
         print *, ' fhypre_sstructmatrixgetvalues error = ', ierr
      endif

      return
      end


c--------------------------------------------------------------------------
c   HYPRE_SStructMatrixGetBoxValues
c--------------------------------------------------------------------------
      subroutine fhypre_sstructmatrixgetboxvalue(fmatrix, fpart, 
     1                                            filower, fiupper, 
     2                                            fvar, fnentries,
     3                                            fentries, fvalues)
      integer ierr
      integer fpart
      integer filower
      integer fiupper
      integer fvar
      integer fnentries
      integer fentries
      integer*8 fmatrix
      double precision fvalues

      call HYPRE_SStructMatrixGetBoxValues(fmatrix, fpart, filower, 
     1                                     fiupper, fvar, fnentries, 
     2                                     fentries, fvalues, ierr)

      if (ierr .ne. 0) then
         print *, ' fhypre_sstructmatrixgetboxvalues error = ', ierr
      endif

      return
      end


c--------------------------------------------------------------------------
c   HYPRE_SStructMatrixAddToValues
c--------------------------------------------------------------------------
      subroutine fhypre_sstructmatrixaddtovalues(fmatrix, fpart, findex,
     1                                           fvar, fnentries, 
     2                                           fentries, fvalues)

      integer ierr
      integer fpart
      integer findex
      integer fvar
      integer fnentries
      integer fentries
      integer*8 fmatrix
      double precision fvalues

      call HYPRE_SStructMatrixAddToValues(fmatrix, fpart, findex, fvar, 
     1                                    fnentries, fentries, fvalues, 
     2                                    ierr)

      if (ierr .ne. 0) then
         print *, ' fhypre_sstructmatrixaddtovalues error = ', ierr
      endif

      return
      end


c--------------------------------------------------------------------------
c   HYPRE_SStructMatrixAddToBoxValues
c--------------------------------------------------------------------------
      subroutine fhypre_sstructmatrixaddtoboxval(fmatrix, fpart, 
     1                                             filower, fiupper,
     2                                             fvar, fnentries, 
     3                                             fentries, fvalues)
      integer ierr
      integer fpart
      integer filower
      integer fiupper
      integer fvar
      integer fnentries
      integer fentries
      integer*8 fmatrix
      double precision fvalues

      call HYPRE_SStructMatrixAddToBoxValu(fmatrix, fpart, filower, 
     1                                       fiupper, fvar, fnentries, 
     2                                       fentries, fvalues, ierr)

      if (ierr .ne. 0) then
         print *, ' fhypre_sstructmatrixaddtoboxvalues error = ', ierr
      endif

      return
      end


c--------------------------------------------------------------------------
c   HYPRE_SStructMatrixAssemble
c--------------------------------------------------------------------------
      subroutine fhypre_sstructmatrixassemble(fmatrix)

      integer ierr
      integer*8 fmatrix

      call HYPRE_SStructMatrixAssemble(fmatrix, ierr)

      if (ierr .ne. 0) then
         print *, ' fhypre_sstructmatrixassemble error = ', ierr
      endif

      return
      end


c--------------------------------------------------------------------------
c   HYPRE_SStructMatrixSetSymmetric
c--------------------------------------------------------------------------
      subroutine fhypre_sstructmatrixsetsymmetri(fmatrix, fpart, fvar,
     1                                            fto_var, fsymmetric)

      integer ierr
      integer fpart
      integer fvar
      integer fto_var
      integer fsymmetric
      integer*8 fmatrix

      call HYPRE_SStructMatrixSetSymmetric(fmatrix, fpart, fvar, 
     1                                     fto_var, fsymmetric, ierr)

      if (ierr .ne. 0) then
         print *, ' fhypre_sstructmatrixsetsymmetric error = ', ierr
      endif

      return
      end


c--------------------------------------------------------------------------
c   HYPRE_SStructMatrixSetNSSymmetric
c--------------------------------------------------------------------------
      subroutine fhypre_sstructmatrixsetnssymmet(fmatrix, fsymmetric)

      integer ierr
      integer fsymmetric
      integer*8 fmatrix

      call HYPRE_SStructMatrixSetNSSymmetr(fmatrix, fsymmetric, ierr)

      if (ierr .ne. 0) then
         print *, ' fhypre_sstructmatrixsetnssymmetric error = ', ierr
      endif

      return
      end


c--------------------------------------------------------------------------
c   HYPRE_SStructMatrixSetObjectType
c--------------------------------------------------------------------------
      subroutine fhypre_sstructmatrixsetobjectty(fmatrix, ftype)

      integer ierr
      integer ftype
      integer*8 fmatrix

      call HYPRE_SStructMatrixSetObjectTyp(fmatrix, ftype, ierr)

      if (ierr .ne. 0) then
         print *, ' fhypre_sstructmatrixsetobjecttype error = ', ierr
      endif

      return
      end


c--------------------------------------------------------------------------
c   HYPRE_SStructMatrixGetObject
c--------------------------------------------------------------------------
      subroutine fhypre_sstructmatrixgetobject(fmatrix, fobject)

      integer ierr
      integer*8 fobject
      integer*8 fmatrix

      call HYPRE_SStructMatrixGetObject(fmatrix, fobject, ierr)

      if (ierr .ne. 0) then
         print *, ' fhypre_sstructmatrixgetobject error = ', ierr
      endif

      return
      end


c--------------------------------------------------------------------------
c   HYPRE_SStructMatrixGetObject2
c--------------------------------------------------------------------------
      subroutine fhypre_sstructmatrixgetobject2(fmatrix, fobject)

      integer ierr
      integer*8 fobject
      integer*8 fmatrix

      call HYPRE_SStructMatrixGetObject2(fmatrix, fobject, ierr)

      if (ierr .ne. 0) then
         print *, ' fhypre_sstructmatrixgetobject2 error = ', ierr
      endif

      return
      end


c--------------------------------------------------------------------------
c   HYPRE_SStructMatrixPrint
c--------------------------------------------------------------------------
      subroutine fhypre_sstructmatrixprint(ffilename, fmatrix, fall)

      integer ierr
      integer fall
      integer*8 fmatrix
      character*(*) ffilename

      call HYPRE_SStructMatrixPrint(ffilename, fmatrix, fall, ierr)

      if (ierr .ne. 0) then
         print *, ' fhypre_sstructmatrixprint error = ', ierr
      endif

      return
      end


c--------------------------------------------------------------------------
c   HYPRE_SStructMatrixMatvec
c--------------------------------------------------------------------------
      subroutine fhypre_sstructmatrixmatvec(falpha, fA, fx, fbeta, fy)

      integer ierr
      integer*8 fA
      integer*8 fx
      integer*8 fy
      double precision falpha
      double precision fbeta

      call HYPRE_SStructMatrixMatvec(falpha, fA, fx, fbeta, fy, ierr)

      if (ierr .ne. 0) then
         print *, ' fhypre_sstructmatrixmatvec error = ', ierr
      endif

      return
      end




c***************************************************************************
c              HYPRE_SStructStencil routines
c***************************************************************************

c--------------------------------------------------------------------------
c  HYPRE_SStructStencilCreate
c--------------------------------------------------------------------------
      subroutine fhypre_sstructstencilcreate(fndim, fsize, fstencil_ptr)

      integer ierr
      integer fndim
      integer fsize
      integer*8 fstencil_ptr

      call HYPRE_SStructStencilCreate(fndim, fsize, fstencil_ptr, ierr)

      if (ierr .ne. 0) then
         print *, ' fhypre_sstructstencilcreate error = ', ierr
      endif

      return
      end


c--------------------------------------------------------------------------
c  HYPRE_SStructStencilDestroy
c--------------------------------------------------------------------------
      subroutine fhypre_sstructstencildestroy(fstencil)

      integer ierr
      integer*8 fstencil

      call HYPRE_SStructStencilDestroy(fstencil, ierr)

      if (ierr .ne. 0) then
         print *, ' fhypre_sstructstencildestroy error = ', ierr
      endif

      return
      end


c--------------------------------------------------------------------------
c  HYPRE_SStructStencilSetEntry
c--------------------------------------------------------------------------
      subroutine fhypre_sstructstencilsetentry(fstencil, fentry, 
     1                                         foffset, fvar)

      integer ierr
      integer fentry
      integer foffset
      integer fvar
      integer*8 fstencil

      call HYPRE_SStructStencilSetEntry(fstencil, fentry, foffset, fvar,
     1                                  ierr)

      if (ierr .ne. 0) then
         print *, ' fhypre_sstructstencilsetentry error = ', ierr
      endif

      return
      end




c***************************************************************************
c              HYPRE_SStructVector routines
c***************************************************************************

c--------------------------------------------------------------------------
c   HYPRE_SStructVectorCreate
c--------------------------------------------------------------------------
      subroutine fhypre_sstructvectorcreate(fcomm, fgrid, fvector_ptr)

      integer ierr
      integer fcomm
      integer*8 fvector_ptr
      integer*8 fgrid

      call HYPRE_SStructVectorCreate(fcomm, fgrid, fvector_ptr, ierr)

      if (ierr .ne. 0) then
         print *, ' fhypre_sstructvectorcreate error = ', ierr
      endif

      return
      end


c--------------------------------------------------------------------------
c  HYPRE_SStructVectorDestroy
c--------------------------------------------------------------------------
      subroutine fhypre_sstructvectordestroy(fvector)

      integer ierr
      integer*8 fvector

      call HYPRE_SStructVectorDestroy(fvector, ierr)

      if (ierr .ne. 0) then
         print *, ' fhypre_sstructvectordestroy error = ', ierr
      endif

      return
      end


c---------------------------------------------------------
c  HYPRE_SStructVectorInitialize
c---------------------------------------------------------
      subroutine fhypre_sstructvectorinitialize(fvector)

      integer ierr
      integer*8 fvector
   
      call HYPRE_SStructVectorInitialize(fvector, ierr)

      if (ierr .ne. 0) then
         print *, ' fhypre_sstructvectorinitialize error = ', ierr
      endif

      return
      end


c--------------------------------------------------------------------------
c   HYPRE_SStructVectorSetValues
c--------------------------------------------------------------------------
      subroutine fhypre_sstructvectorsetvalues(fvector, fpart, findex, 
     1                                         fvar, fvalue)

      integer ierr
      integer fpart
      integer findex
      integer fvar
      integer*8 fvector
      double precision fvalue

      call HYPRE_SStructVectorSetValues(fvector, fpart, findex, fvar,
     1                                  fvalue, ierr)

      if (ierr .ne. 0) then
         print *, ' fhypre_sstructvectorsetvalues error = ', ierr
      endif

      return
      end


c--------------------------------------------------------------------------
c   HYPRE_SStructVectorSetBoxValues
c--------------------------------------------------------------------------
      subroutine fhypre_sstructvectorsetboxvalue(fvector, fpart,
     1                                            filower, fiupper, 
     2                                            fvar, fvalues)

      integer ierr
      integer fpart
      integer filower
      integer fiupper
      integer fvar
      integer*8 fvector
      double precision fvalues

      call HYPRE_SStructVectorSetBoxValues(fvector, fpart, filower, 
     1                                     fiupper, fvar, fvalues, ierr)

      if (ierr .ne. 0) then
         print *, ' fhypre_sstructvectorsetboxvalues error = ', ierr
      endif

      return
      end


c--------------------------------------------------------------------------
c   HYPRE_SStructVectorAddToValues
c--------------------------------------------------------------------------
      subroutine fhypre_sstructvectoraddtovalues(fvector, fpart, findex,
     1                                           fvar, fvalue)

      integer ierr
      integer fpart
      integer findex
      integer fvar
      integer*8 fvector
      double precision fvalue

      call HYPRE_SStructVectorAddToValues(fvector, fpart, findex, fvar, 
     1                                    fvalue, ierr)

      if (ierr .ne. 0) then
         print *, ' fhypre_sstructvectoraddtovalues error = ', ierr
      endif

      return
      end


c--------------------------------------------------------------------------
c   HYPRE_SStructVectorAddToBoxValues
c--------------------------------------------------------------------------
      subroutine fhypre_sstructvectoraddtoboxval(fvector, fpart,
     1                                            filower, fiupper, 
     2                                            fvar, fvalues)
      integer ierr
      integer fpart
      integer filower
      integer fiupper
      integer fvar
      integer*8 fvector
      double precision fvalues

      call HYPRE_SStructVectorAddToBoxValu(fvector, fpart, filower,
     1                                       fiupper, fvar, fvalues,
     2                                       ierr)

      if (ierr .ne. 0) then
         print *, ' fhypre_sstructvectoraddtoboxvalues error = ', ierr
      endif

      return
      end


c--------------------------------------------------------------------------
c   HYPRE_SStructVectorAssemble
c--------------------------------------------------------------------------
      subroutine fhypre_sstructvectorassemble(fvector)

      integer ierr
      integer*8 fvector

      call HYPRE_SStructVectorAssemble(fvector, ierr)

      if (ierr .ne. 0) then
         print *, ' fhypre_sstructvectorassemble error = ', ierr
      endif

      return
      end


c--------------------------------------------------------------------------
c   HYPRE_SStructVectorGather
c--------------------------------------------------------------------------
      subroutine fhypre_sstructvectorgather(fvector)

      integer ierr
      integer*8 fvector

      call HYPRE_SStructVectorGather(fvector, ierr)

      if (ierr .ne. 0) then
         print *, ' fhypre_sstructvectorgather error = ', ierr
      endif

      return
      end


c--------------------------------------------------------------------------
c   HYPRE_SStructVectorGetValues
c--------------------------------------------------------------------------
      subroutine fhypre_sstructvectorgetvalues(fvector, fpart, findex, 
     1                                         fvar, fvalue)

      integer ierr
      integer fpart
      integer findex
      integer fvar
      integer*8 fvector
      double precision fvalue

      call HYPRE_SStructVectorGetValues(fvector, fpart, findex, fvar, 
     1                                  fvalue, ierr)

      if (ierr .ne. 0) then
         print *, ' fhypre_sstructvectorgetvalues error = ', ierr
      endif

      return
      end


c--------------------------------------------------------------------------
c   HYPRE_SStructVectorGetBoxValues
c--------------------------------------------------------------------------
      subroutine fhypre_sstructvectorgetboxvalue(fvector, fpart, 
     1                                            filower, fiupper, 
     2                                            fvar, fvalues)

      integer ierr
      integer fpart
      integer filower
      integer fiupper
      integer fvar
      integer*8 fvector
      double precision fvalues

      call HYPRE_SStructVectorGetBoxValues(fvector, fpart, filower,
     1                                     fiupper, fvar, fvalues, ierr)

      if (ierr .ne. 0) then
         print *, ' fhypre_sstructvectorgetboxvalues error = ', ierr
      endif

      return
      end


c--------------------------------------------------------------------------
c   HYPRE_SStructVectorSetConstantValues
c--------------------------------------------------------------------------
      subroutine fhypre_sstructvectorsetconstant(fvector, fvalue)

      integer ierr
      integer*8 fvector
      double precision fvalue

      call HYPRE_SStructVectorSetConstantV(fvector, fvalue, ierr)

      if (ierr .ne. 0) then
         print *, ' fhypre_sstructvectorsetconstantvalues error = ',
     1                                       ierr
      endif

      return
      end


c--------------------------------------------------------------------------
c   HYPRE_SStructVectorSetObjectType
c--------------------------------------------------------------------------
      subroutine fhypre_sstructvectorsetobjectty(fvector, ftype)

      integer ierr
      integer ftype
      integer*8 fvector

      call HYPRE_SStructVectorSetObjectTyp(fvector, ftype, ierr)

      if (ierr .ne. 0) then
         print *, ' fhypre_sstructvectorsetobjecttype error = ', ierr
      endif

      return
      end


c--------------------------------------------------------------------------
c   HYPRE_SStructVectorGetObject
c--------------------------------------------------------------------------
      subroutine fhypre_sstructvectorgetobject(fvector, fobject)

      integer ierr
      integer*8 fobject
      integer*8 fvector

      call HYPRE_SStructVectorGetObject(fvector, fobject, ierr)

      if (ierr .ne. 0) then
         print *, ' fhypre_sstructvectorgetobject error = ', ierr
      endif

      return
      end


c--------------------------------------------------------------------------
c   HYPRE_SStructVectorPrint
c--------------------------------------------------------------------------
      subroutine fhypre_sstructvectorprint(ffilename, fvector, fall)

      integer ierr
      integer fall
      integer*8 fvector
      character*(*) ffilename

      call HYPRE_SStructVectorPrint(ffilename, fvector, fall, ierr)

      if (ierr .ne. 0) then
         print *, ' fhypre_sstructvectorprint error = ', ierr
      endif

      return
      end


c--------------------------------------------------------------------------
c   HYPRE_SStructVectorCopy
c--------------------------------------------------------------------------
      subroutine fhypre_sstructvectorcopy(fx, fy)

      integer ierr
      integer*8 fx
      integer*8 fy

      call HYPRE_SStructVectorCopy(fx, fy, ierr)

      if (ierr .ne. 0) then
         print *, ' fhypre_sstructvectorcopy error = ', ierr
      endif

      return
      end


c--------------------------------------------------------------------------
c   HYPRE_SStructVectorScale
c--------------------------------------------------------------------------
      subroutine fhypre_sstructvectorscale(falpha, fy)

      integer ierr
      integer*8 fy
      double precision falpha

      call HYPRE_SStructVectorScale(falpha, fy, ierr)

      if (ierr .ne. 0) then
         print *, ' fhypre_sstructvectorscale error = ', ierr
      endif

      return
      end


c--------------------------------------------------------------------------
c   HYPRE_SStructInnerProd
c--------------------------------------------------------------------------
      subroutine fhypre_sstructinnerprod(fx, fy, fresult)

      integer ierr
      integer*8 fx
      integer*8 fy
      double precision fresult

      call HYPRE_SStructInnerProd(fx, fy, fresult, ierr)

      if (ierr .ne. 0) then
         print *, ' fhypre_sstructinnerprod error = ', ierr
      endif

      return
      end


c--------------------------------------------------------------------------
c   HYPRE_SStructAxpy
c--------------------------------------------------------------------------
      subroutine fhypre_sstructaxpy(falpha, fx, fy)

      integer ierr
      integer*8 fx
      integer*8 fy
      double precision falpha

      call HYPRE_SStructAxpy(falpha, fx, fy, ierr)

      if (ierr .ne. 0) then
         print *, ' fhypre_sstructaxpy error = ', ierr
      endif

      return
      end
