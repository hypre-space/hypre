/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.26 $
 ***********************************************************************EHEADER*/


/******************************************************************************
 *
 * Member functions for hypre_SStructVector class.
 *
 *****************************************************************************/

#include "headers.h"

/*==========================================================================
 * SStructPVector routines
 *==========================================================================*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructPVectorRef( hypre_SStructPVector  *vector,
                         hypre_SStructPVector **vector_ref )
{
   hypre_SStructPVectorRefCount(vector) ++;
   *vector_ref = vector;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructPVectorCreate( MPI_Comm               comm,
                            hypre_SStructPGrid    *pgrid,
                            hypre_SStructPVector **pvector_ptr)
{
   hypre_SStructPVector  *pvector;
   HYPRE_Int              nvars;
   hypre_StructVector   **svectors;
   hypre_CommPkg        **comm_pkgs;
   hypre_StructGrid      *sgrid;
   HYPRE_Int              var;
 
   pvector = hypre_TAlloc(hypre_SStructPVector, 1);

   hypre_SStructPVectorComm(pvector)  = comm;
   hypre_SStructPVectorPGrid(pvector) = pgrid;
   nvars = hypre_SStructPGridNVars(pgrid);
   hypre_SStructPVectorNVars(pvector) = nvars;
   svectors = hypre_TAlloc(hypre_StructVector *, nvars);

   for (var = 0; var < nvars; var++)
   {
      sgrid = hypre_SStructPGridSGrid(pgrid, var);
      svectors[var] = hypre_StructVectorCreate(comm, sgrid);
   }
   hypre_SStructPVectorSVectors(pvector) = svectors;
   comm_pkgs = hypre_TAlloc(hypre_CommPkg *, nvars);
   for (var = 0; var < nvars; var++)
   {
      comm_pkgs[var] = NULL;
   }
   hypre_SStructPVectorCommPkgs(pvector) = comm_pkgs;
   hypre_SStructPVectorRefCount(pvector) = 1;

   /* GEC inclusion of dataindices   */
   hypre_SStructPVectorDataIndices(pvector) = NULL ;

   *pvector_ptr = pvector;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int 
hypre_SStructPVectorDestroy( hypre_SStructPVector *pvector )
{
   HYPRE_Int            nvars;
   hypre_StructVector **svectors;
   hypre_CommPkg      **comm_pkgs;
   HYPRE_Int            var;

   /* GEC destroying dataindices and data in pvector   */

   HYPRE_Int          *dataindices;

   if (pvector)
   {
      hypre_SStructPVectorRefCount(pvector) --;
      if (hypre_SStructPVectorRefCount(pvector) == 0)
      {
         nvars     = hypre_SStructPVectorNVars(pvector);
         svectors = hypre_SStructPVectorSVectors(pvector);
         comm_pkgs = hypre_SStructPVectorCommPkgs(pvector);
         dataindices = hypre_SStructPVectorDataIndices(pvector);
         for (var = 0; var < nvars; var++)
         {
            hypre_StructVectorDestroy(svectors[var]);
            hypre_CommPkgDestroy(comm_pkgs[var]);
         }
           
         hypre_TFree(dataindices);
         hypre_TFree(svectors);
         hypre_TFree(comm_pkgs);
         hypre_TFree(pvector);
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int 
hypre_SStructPVectorInitialize( hypre_SStructPVector *pvector )
{
   hypre_SStructPGrid    *pgrid     = hypre_SStructPVectorPGrid(pvector);
   HYPRE_Int              nvars     = hypre_SStructPVectorNVars(pvector);
   HYPRE_SStructVariable *vartypes  = hypre_SStructPGridVarTypes(pgrid);
   hypre_StructVector    *svector;
   HYPRE_Int              var;

   for (var = 0; var < nvars; var++)
   {
      svector = hypre_SStructPVectorSVector(pvector, var);
      hypre_StructVectorInitialize(svector);
      if (vartypes[var] > 0)
      {
         /* needed to get AddTo accumulation correct between processors */
         hypre_StructVectorClearGhostValues(svector);
      }
   }

   hypre_SStructPVectorAccumulated(pvector) = 0;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * (action > 0): add-to values
 * (action = 0): set values
 * (action < 0): get values
 *--------------------------------------------------------------------------*/

HYPRE_Int 
hypre_SStructPVectorSetValues( hypre_SStructPVector *pvector,
                               hypre_Index           index,
                               HYPRE_Int             var,
                               double               *value,
                               HYPRE_Int             action )
{
   hypre_StructVector *svector = hypre_SStructPVectorSVector(pvector, var);
   hypre_BoxArray     *grid_boxes;
   hypre_Box          *box;
   HYPRE_Int           i;

   /* set values inside the grid */
   hypre_StructVectorSetValues(svector, index, value, action, -1, 0);

   /* set (AddTo/Get) or clear (Set) values outside the grid in ghost zones */
   if (action != 0)
   {
      /* AddTo/Get */
      hypre_SStructPGrid *pgrid = hypre_SStructPVectorPGrid(pvector);
      hypre_Index         varoffset;
      HYPRE_Int           done = 0;

      grid_boxes = hypre_StructGridBoxes(hypre_StructVectorGrid(svector));

      hypre_ForBoxI(i, grid_boxes)
      {
         box = hypre_BoxArrayBox(grid_boxes, i);
         if ((hypre_IndexX(index) >= hypre_BoxIMinX(box)) &&
             (hypre_IndexX(index) <= hypre_BoxIMaxX(box)) &&
             (hypre_IndexY(index) >= hypre_BoxIMinY(box)) &&
             (hypre_IndexY(index) <= hypre_BoxIMaxY(box)) &&
             (hypre_IndexZ(index) >= hypre_BoxIMinZ(box)) &&
             (hypre_IndexZ(index) <= hypre_BoxIMaxZ(box))   )
         {
            done = 1;
            break;
         }
      }

      if (!done)
      {
         hypre_SStructVariableGetOffset(hypre_SStructPGridVarType(pgrid, var),
                                        hypre_SStructPGridNDim(pgrid), varoffset);
         hypre_ForBoxI(i, grid_boxes)
         {
            box = hypre_BoxArrayBox(grid_boxes, i);
            if ((hypre_IndexX(index) >=
                 hypre_BoxIMinX(box) - hypre_IndexX(varoffset)) &&
                (hypre_IndexX(index) <=
                 hypre_BoxIMaxX(box) + hypre_IndexX(varoffset)) &&
                (hypre_IndexY(index) >=
                 hypre_BoxIMinY(box) - hypre_IndexY(varoffset)) &&
                (hypre_IndexY(index) <=
                 hypre_BoxIMaxY(box) + hypre_IndexY(varoffset)) &&
                (hypre_IndexZ(index) >=
                 hypre_BoxIMinZ(box) - hypre_IndexZ(varoffset)) &&
                (hypre_IndexZ(index) <=
                 hypre_BoxIMaxZ(box) + hypre_IndexZ(varoffset))   )
            {
               hypre_StructVectorSetValues(svector, index, value, action, i, 1);
               break;
            }
         }
      }
   }
   else
   {
      /* Set */
      grid_boxes = hypre_StructGridBoxes(hypre_StructVectorGrid(svector));

      hypre_ForBoxI(i, grid_boxes)
      {
         box = hypre_BoxArrayBox(grid_boxes, i);
         if ((hypre_IndexX(index) < hypre_BoxIMinX(box)) ||
             (hypre_IndexX(index) > hypre_BoxIMaxX(box)) ||
             (hypre_IndexY(index) < hypre_BoxIMinY(box)) ||
             (hypre_IndexY(index) > hypre_BoxIMaxY(box)) ||
             (hypre_IndexZ(index) < hypre_BoxIMinZ(box)) ||
             (hypre_IndexZ(index) > hypre_BoxIMaxZ(box))   )
         {
            hypre_StructVectorClearValues(svector, index, i, 1);
         }
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * (action > 0): add-to values
 * (action = 0): set values
 * (action < 0): get values
 *--------------------------------------------------------------------------*/

HYPRE_Int 
hypre_SStructPVectorSetBoxValues( hypre_SStructPVector *pvector,
                                  hypre_Index           ilower,
                                  hypre_Index           iupper,
                                  HYPRE_Int             var,
                                  double               *values,
                                  HYPRE_Int             action )
{
   hypre_StructVector *svector = hypre_SStructPVectorSVector(pvector, var);
   hypre_BoxArray     *grid_boxes;
   hypre_Box          *box;
   hypre_Box          *value_box;
   HYPRE_Int           i, j;

   box = hypre_BoxCreate();
   hypre_CopyIndex(ilower, hypre_BoxIMin(box));
   hypre_CopyIndex(iupper, hypre_BoxIMax(box));
   value_box = box;

   /* set values inside the grid */
   hypre_StructVectorSetBoxValues(svector, box, value_box, values, action, -1, 0);

   /* set (AddTo/Get) or clear (Set) values outside the grid in ghost zones */
   if (action != 0)
   {
      /* AddTo/Get */
      hypre_SStructPGrid  *pgrid = hypre_SStructPVectorPGrid(pvector);
      hypre_Index          varoffset;
      hypre_BoxArray      *left_boxes, *done_boxes, *temp_boxes;
      hypre_Box           *left_box, *done_box, *int_box;

      hypre_SStructVariableGetOffset(hypre_SStructPGridVarType(pgrid, var),
                                     hypre_SStructPGridNDim(pgrid), varoffset);
      grid_boxes = hypre_StructGridBoxes(hypre_StructVectorGrid(svector));

      left_boxes = hypre_BoxArrayCreate(1);
      done_boxes = hypre_BoxArrayCreate(2);
      temp_boxes = hypre_BoxArrayCreate(0);

      /* done_box always points to the first box in done_boxes */
      done_box = hypre_BoxArrayBox(done_boxes, 0);
      /* int_box always points to the second box in done_boxes */
      int_box = hypre_BoxArrayBox(done_boxes, 1);

      hypre_CopyBox(box, hypre_BoxArrayBox(left_boxes, 0));
      hypre_BoxArraySetSize(left_boxes, 1);
      hypre_SubtractBoxArrays(left_boxes, grid_boxes, temp_boxes);

      hypre_BoxArraySetSize(done_boxes, 0);
      hypre_ForBoxI(i, grid_boxes)
      {
         hypre_SubtractBoxArrays(left_boxes, done_boxes, temp_boxes);
         hypre_BoxArraySetSize(done_boxes, 1);
         hypre_CopyBox(hypre_BoxArrayBox(grid_boxes, i), done_box);
         hypre_BoxIMinX(done_box) -= hypre_IndexX(varoffset);
         hypre_BoxIMinY(done_box) -= hypre_IndexY(varoffset);
         hypre_BoxIMinZ(done_box) -= hypre_IndexZ(varoffset);
         hypre_BoxIMaxX(done_box) += hypre_IndexX(varoffset);
         hypre_BoxIMaxY(done_box) += hypre_IndexY(varoffset);
         hypre_BoxIMaxZ(done_box) += hypre_IndexZ(varoffset);
         hypre_ForBoxI(j, left_boxes)
         {
            left_box = hypre_BoxArrayBox(left_boxes, j);
            hypre_IntersectBoxes(left_box, done_box, int_box);
            hypre_StructVectorSetBoxValues(svector, int_box, value_box,
                                           values, action, i, 1);
         }
      }

      hypre_BoxArrayDestroy(left_boxes);
      hypre_BoxArrayDestroy(done_boxes);
      hypre_BoxArrayDestroy(temp_boxes);
   }
   else
   {
      /* Set */
      hypre_BoxArray  *diff_boxes;
      hypre_Box       *grid_box, *diff_box;

      grid_boxes = hypre_StructGridBoxes(hypre_StructVectorGrid(svector));
      diff_boxes = hypre_BoxArrayCreate(0);

      hypre_ForBoxI(i, grid_boxes)
      {
         grid_box = hypre_BoxArrayBox(grid_boxes, i);
         hypre_BoxArraySetSize(diff_boxes, 0);
         hypre_SubtractBoxes(box, grid_box, diff_boxes);

         hypre_ForBoxI(j, diff_boxes)
         {
            diff_box = hypre_BoxArrayBox(diff_boxes, j);
            hypre_StructVectorClearBoxValues(svector, diff_box, i, 1);
         }
      }
      hypre_BoxArrayDestroy(diff_boxes);
   }

   hypre_BoxDestroy(box);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int 
hypre_SStructPVectorAccumulate( hypre_SStructPVector *pvector )
{
   hypre_SStructPGrid    *pgrid     = hypre_SStructPVectorPGrid(pvector);
   HYPRE_Int              nvars     = hypre_SStructPVectorNVars(pvector);
   hypre_StructVector   **svectors  = hypre_SStructPVectorSVectors(pvector);
   hypre_CommPkg        **comm_pkgs = hypre_SStructPVectorCommPkgs(pvector);

   hypre_CommInfo        *comm_info;
   hypre_CommPkg         *comm_pkg;
   hypre_CommHandle      *comm_handle;

   HYPRE_Int              ndim      = hypre_SStructPGridNDim(pgrid);
   HYPRE_SStructVariable *vartypes  = hypre_SStructPGridVarTypes(pgrid);

   hypre_Index            varoffset;
   HYPRE_Int              num_ghost[6];
   hypre_StructGrid      *sgrid;
   HYPRE_Int              var, d;

   /* if values already accumulated, just return */
   if (hypre_SStructPVectorAccumulated(pvector))
   {
      return hypre_error_flag;
   }

   for (var = 0; var < nvars; var++)
   {
      if (vartypes[var] > 0)
      {
         sgrid = hypre_StructVectorGrid(svectors[var]);
         hypre_SStructVariableGetOffset(vartypes[var], ndim, varoffset);
         for (d = 0; d < 3; d++)
         {
            num_ghost[2*d]   = hypre_IndexD(varoffset, d);
            num_ghost[2*d+1] = hypre_IndexD(varoffset, d);
         }
         
         hypre_CreateCommInfoFromNumGhost(sgrid, num_ghost, &comm_info);
         hypre_CommPkgDestroy(comm_pkgs[var]);
         hypre_CommPkgCreate(comm_info,
                             hypre_StructVectorDataSpace(svectors[var]),
                             hypre_StructVectorDataSpace(svectors[var]),
                             1, NULL, 0, hypre_StructVectorComm(svectors[var]),
                             &comm_pkgs[var]);

         /* accumulate values from AddTo */
         hypre_CommPkgCreate(comm_info,
                             hypre_StructVectorDataSpace(svectors[var]),
                             hypre_StructVectorDataSpace(svectors[var]),
                             1, NULL, 1, hypre_StructVectorComm(svectors[var]),
                             &comm_pkg);
         hypre_InitializeCommunication(comm_pkg,
                                       hypre_StructVectorData(svectors[var]),
                                       hypre_StructVectorData(svectors[var]), 1, 0,
                                       &comm_handle);
         hypre_FinalizeCommunication(comm_handle);

         hypre_CommInfoDestroy(comm_info);
         hypre_CommPkgDestroy(comm_pkg);
      }
   }

   hypre_SStructPVectorAccumulated(pvector) = 1;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int 
hypre_SStructPVectorAssemble( hypre_SStructPVector *pvector )
{
   HYPRE_Int              nvars     = hypre_SStructPVectorNVars(pvector);
   hypre_StructVector   **svectors  = hypre_SStructPVectorSVectors(pvector);
   HYPRE_Int              var;

   hypre_SStructPVectorAccumulate(pvector);

   for (var = 0; var < nvars; var++)
   {
      hypre_StructVectorClearGhostValues(svectors[var]);
      hypre_StructVectorAssemble(svectors[var]);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int 
hypre_SStructPVectorGather( hypre_SStructPVector *pvector )
{
   HYPRE_Int              nvars     = hypre_SStructPVectorNVars(pvector);
   hypre_StructVector   **svectors  = hypre_SStructPVectorSVectors(pvector);
   hypre_CommPkg        **comm_pkgs = hypre_SStructPVectorCommPkgs(pvector);
   hypre_CommHandle      *comm_handle;
   HYPRE_Int              var;

   for (var = 0; var < nvars; var++)
   {
      if (comm_pkgs[var] != NULL)
      {
         hypre_InitializeCommunication(comm_pkgs[var],
                                       hypre_StructVectorData(svectors[var]),
                                       hypre_StructVectorData(svectors[var]), 0, 0,
                                       &comm_handle);
         hypre_FinalizeCommunication(comm_handle);
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int 
hypre_SStructPVectorGetValues( hypre_SStructPVector *pvector,
                               hypre_Index           index,
                               HYPRE_Int             var,
                               double               *value )
{
   hypre_SStructPGrid *pgrid     = hypre_SStructPVectorPGrid(pvector);
   hypre_StructVector *svector   = hypre_SStructPVectorSVector(pvector, var);
   hypre_StructGrid   *sgrid     = hypre_StructVectorGrid(svector);
   hypre_BoxArray     *iboxarray = hypre_SStructPGridIBoxArray(pgrid, var);
   hypre_BoxArray     *tboxarray;

   /* temporarily swap out sgrid boxes in order to get boundary data */
   tboxarray = hypre_StructGridBoxes(sgrid);
   hypre_StructGridBoxes(sgrid) = iboxarray;
   hypre_StructVectorSetValues(svector, index, value, -1, -1, 0);
   hypre_StructGridBoxes(sgrid) = tboxarray;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int 
hypre_SStructPVectorGetBoxValues( hypre_SStructPVector *pvector,
                                  hypre_Index           ilower,
                                  hypre_Index           iupper,
                                  HYPRE_Int             var,
                                  double               *values )
{
   hypre_SStructPGrid *pgrid     = hypre_SStructPVectorPGrid(pvector);
   hypre_StructVector *svector   = hypre_SStructPVectorSVector(pvector, var);
   hypre_StructGrid   *sgrid     = hypre_StructVectorGrid(svector);
   hypre_BoxArray     *iboxarray = hypre_SStructPGridIBoxArray(pgrid, var);
   hypre_BoxArray     *tboxarray;
   hypre_Box          *box;

   box = hypre_BoxCreate();
   hypre_CopyIndex(ilower, hypre_BoxIMin(box));
   hypre_CopyIndex(iupper, hypre_BoxIMax(box));
   /* temporarily swap out sgrid boxes in order to get boundary data */
   tboxarray = hypre_StructGridBoxes(sgrid);
   hypre_StructGridBoxes(sgrid) = iboxarray;
   hypre_StructVectorSetBoxValues(svector, box, box, values, -1, -1, 0);
   hypre_StructGridBoxes(sgrid) = tboxarray;
   hypre_BoxDestroy(box);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int 
hypre_SStructPVectorSetConstantValues( hypre_SStructPVector *pvector,
                                       double                value )
{
   HYPRE_Int           nvars = hypre_SStructPVectorNVars(pvector);
   hypre_StructVector *svector;
   HYPRE_Int           var;

   for (var = 0; var < nvars; var++)
   {
      svector = hypre_SStructPVectorSVector(pvector, var);
      hypre_StructVectorSetConstantValues(svector, value);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * For now, just print multiple files
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructPVectorPrint( const char           *filename,
                           hypre_SStructPVector *pvector,
                           HYPRE_Int             all )
{
   HYPRE_Int  nvars = hypre_SStructPVectorNVars(pvector);
   HYPRE_Int  var;
   char new_filename[255];

   for (var = 0; var < nvars; var++)
   {
      hypre_sprintf(new_filename, "%s.%02d", filename, var);
      hypre_StructVectorPrint(new_filename,
                              hypre_SStructPVectorSVector(pvector, var),
                              all);
   }

   return hypre_error_flag;
}

/*==========================================================================
 * SStructVector routines
 *==========================================================================*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructVectorRef( hypre_SStructVector  *vector,
                        hypre_SStructVector **vector_ref )
{
   hypre_SStructVectorRefCount(vector) ++;
   *vector_ref = vector;

   return hypre_error_flag;
   return 0;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int 
hypre_SStructVectorSetConstantValues( hypre_SStructVector *vector,
                                      double               value )
{
   HYPRE_Int             nparts = hypre_SStructVectorNParts(vector);
   hypre_SStructPVector *pvector;
   HYPRE_Int             part;

   for (part = 0; part < nparts; part++)
   {
      pvector = hypre_SStructVectorPVector(vector, part);
      hypre_SStructPVectorSetConstantValues(pvector, value);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Here the address of the parvector inside the semistructured vector
 * is provided to the "outside". It assumes that the vector type
 * is HYPRE_SSTRUCT
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructVectorConvert( hypre_SStructVector  *vector,
                            hypre_ParVector     **parvector_ptr )
{
  *parvector_ptr = hypre_SStructVectorParVector(vector);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Copy values from vector to parvector and provide the address
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructVectorParConvert( hypre_SStructVector  *vector,
                               hypre_ParVector     **parvector_ptr )
{
   hypre_ParVector      *parvector;
   double               *pardata;
   HYPRE_Int             pari;

   hypre_SStructPVector *pvector;
   hypre_StructVector   *y;
   hypre_Box            *y_data_box;
   HYPRE_Int             yi;
   double               *yp;
   hypre_BoxArray       *boxes;
   hypre_Box            *box;
   HYPRE_Int             bi;
   hypre_Index           loop_size;
   hypre_IndexRef        start;
   hypre_Index           stride;
                        
   HYPRE_Int             nparts, nvars;
   HYPRE_Int             part, var, i;
   HYPRE_Int             loopi, loopj, loopk;

   hypre_SetIndex(stride, 1, 1, 1);

   parvector = hypre_SStructVectorParVector(vector);
   pardata = hypre_VectorData(hypre_ParVectorLocalVector(parvector));
   pari = 0;
   nparts = hypre_SStructVectorNParts(vector);
   for (part = 0; part < nparts; part++)
   {
      pvector = hypre_SStructVectorPVector(vector, part);
      nvars = hypre_SStructPVectorNVars(pvector);
      for (var = 0; var < nvars; var++)
      {
         y = hypre_SStructPVectorSVector(pvector, var);

         boxes = hypre_StructGridBoxes(hypre_StructVectorGrid(y));
         hypre_ForBoxI(i, boxes)
            {
               box   = hypre_BoxArrayBox(boxes, i);
               start = hypre_BoxIMin(box);

               y_data_box =
                  hypre_BoxArrayBox(hypre_StructVectorDataSpace(y), i);
               yp = hypre_StructVectorBoxData(y, i);

               hypre_BoxGetSize(box, loop_size);
               hypre_BoxLoop2Begin(loop_size,
                                   y_data_box, start, stride, yi,
                                   box,        start, stride, bi);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,yi,bi
#include "hypre_box_smp_forloop.h"
               hypre_BoxLoop2For(loopi, loopj, loopk, yi, bi)
                  {
                     pardata[pari+bi] = yp[yi];
                  }
               hypre_BoxLoop2End(yi, bi);
               pari +=
                  hypre_IndexX(loop_size)*
                  hypre_IndexY(loop_size)*
                  hypre_IndexZ(loop_size);
            }
      }
   }

   *parvector_ptr = hypre_SStructVectorParVector(vector);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Used for HYPRE_SSTRUCT type semi structured vectors.
 * A dummy function to indicate that the struct vector part will be used.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructVectorRestore( hypre_SStructVector *vector,
                            hypre_ParVector     *parvector )
{
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Copy values from parvector to vector
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructVectorParRestore( hypre_SStructVector *vector,
                            hypre_ParVector     *parvector )
{
   double               *pardata;
   HYPRE_Int             pari;

   hypre_SStructPVector *pvector;
   hypre_StructVector   *y;
   hypre_Box            *y_data_box;
   HYPRE_Int             yi;
   double               *yp;
   hypre_BoxArray       *boxes;
   hypre_Box            *box;
   HYPRE_Int             bi;
   hypre_Index           loop_size;
   hypre_IndexRef        start;
   hypre_Index           stride;
                        
   HYPRE_Int             nparts, nvars;
   HYPRE_Int             part, var, i;
   HYPRE_Int             loopi, loopj, loopk;

   if (parvector != NULL)
   {
      hypre_SetIndex(stride, 1, 1, 1);

      parvector = hypre_SStructVectorParVector(vector);
      pardata = hypre_VectorData(hypre_ParVectorLocalVector(parvector));
      pari = 0;
      nparts = hypre_SStructVectorNParts(vector);
      for (part = 0; part < nparts; part++)
      {
         pvector = hypre_SStructVectorPVector(vector, part);
         nvars = hypre_SStructPVectorNVars(pvector);
         for (var = 0; var < nvars; var++)
         {
            y = hypre_SStructPVectorSVector(pvector, var);

            boxes = hypre_StructGridBoxes(hypre_StructVectorGrid(y));
            hypre_ForBoxI(i, boxes)
               {
                  box   = hypre_BoxArrayBox(boxes, i);
                  start = hypre_BoxIMin(box);

                  y_data_box =
                     hypre_BoxArrayBox(hypre_StructVectorDataSpace(y), i);
                  yp = hypre_StructVectorBoxData(y, i);

                  hypre_BoxGetSize(box, loop_size);
                  hypre_BoxLoop2Begin(loop_size,
                                      y_data_box, start, stride, yi,
                                      box,        start, stride, bi);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,yi,bi
#include "hypre_box_smp_forloop.h"
                  hypre_BoxLoop2For(loopi, loopj, loopk, yi, bi)
                     {
                        yp[yi] = pardata[pari+bi];
                     }
                  hypre_BoxLoop2End(yi, bi);
                  pari +=
                     hypre_IndexX(loop_size)*
                     hypre_IndexY(loop_size)*
                     hypre_IndexZ(loop_size);
               }
         }
      }
   }

   return hypre_error_flag;
}
/*------------------------------------------------------------------
 *  GEC1002 shell initialization of a pvector 
 *   if the pvector exists. This function will set the dataindices
 *  and datasize of the pvector. Datasize is the sum of the sizes
 *  of each svector and dataindices is defined as
 *  dataindices[var]= aggregated initial size of the pvector[var]
 *  When ucvars are present we need to modify adding nucvars.
 *----------------------------------------------------------------*/   
HYPRE_Int 
hypre_SStructPVectorInitializeShell( hypre_SStructPVector *pvector)
{
   HYPRE_Int            nvars = hypre_SStructPVectorNVars(pvector);
   HYPRE_Int            var;
   HYPRE_Int            pdatasize;
   HYPRE_Int            svectdatasize;
   HYPRE_Int           *pdataindices;
   HYPRE_Int            nucvars = 0;
   hypre_StructVector  *svector;

   pdatasize = 0;
   pdataindices = hypre_CTAlloc(HYPRE_Int, nvars);

   for (var =0; var < nvars; var++)
   {
      svector = hypre_SStructPVectorSVector(pvector, var);
      hypre_StructVectorInitializeShell(svector);
      pdataindices[var] = pdatasize ;
      svectdatasize = hypre_StructVectorDataSize(svector); 
      pdatasize += svectdatasize;    
   }

   /* GEC1002 assuming that the ucvars are located at the end, after the
    * the size of the vars has been included we add the number of uvar 
    * for this part                                                  */

   hypre_SStructPVectorDataIndices(pvector) = pdataindices;
   hypre_SStructPVectorDataSize(pvector) = pdatasize+nucvars ;

   hypre_SStructPVectorAccumulated(pvector) = 0;

   return hypre_error_flag;
}
     
/*------------------------------------------------------------------
 *  GEC1002 shell initialization of a sstructvector
 *  if the vector exists. This function will set the
 *  dataindices and datasize of the vector. When ucvars
 *  are present at the end of all the parts we need to modify adding pieces
 *  for ucvars.
 *----------------------------------------------------------------*/  
HYPRE_Int 
hypre_SStructVectorInitializeShell( hypre_SStructVector *vector)
{
   HYPRE_Int                part  ;
   HYPRE_Int                datasize;
   HYPRE_Int                pdatasize;
   HYPRE_Int                nparts = hypre_SStructVectorNParts(vector); 
   hypre_SStructPVector    *pvector;
   HYPRE_Int               *dataindices;

   datasize = 0;
   dataindices = hypre_CTAlloc(HYPRE_Int, nparts);
   for (part = 0; part < nparts; part++)
   {
      pvector = hypre_SStructVectorPVector(vector, part) ;
      hypre_SStructPVectorInitializeShell(pvector);
      pdatasize = hypre_SStructPVectorDataSize(pvector);
      dataindices[part] = datasize ;
      datasize        += pdatasize ;  
   } 
   hypre_SStructVectorDataIndices(vector) = dataindices;
   hypre_SStructVectorDataSize(vector) = datasize ;

   return hypre_error_flag;
}   


HYPRE_Int
hypre_SStructVectorClearGhostValues(hypre_SStructVector *vector)
{
   HYPRE_Int              nparts= hypre_SStructVectorNParts(vector);
   hypre_SStructPVector  *pvector;
   hypre_StructVector    *svector;

   HYPRE_Int    part;
   HYPRE_Int    nvars, var;

   for (part= 0; part< nparts; part++)
   {
      pvector= hypre_SStructVectorPVector(vector, part);
      nvars  = hypre_SStructPVectorNVars(pvector);

      for (var= 0; var< nvars; var++)
      {
         svector= hypre_SStructPVectorSVector(pvector, var);
         hypre_StructVectorClearGhostValues(svector);
      }
   }

   return hypre_error_flag;
}   
