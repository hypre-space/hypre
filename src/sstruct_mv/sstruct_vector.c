/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Member functions for hypre_SStructVector class.
 *
 *****************************************************************************/

#include "_hypre_sstruct_mv.h"
#include "_hypre_struct_mv.hpp"

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

   pvector = hypre_TAlloc(hypre_SStructPVector,  1, HYPRE_MEMORY_HOST);

   hypre_SStructPVectorComm(pvector)  = comm;
   hypre_SStructPVectorPGrid(pvector) = pgrid;
   nvars = hypre_SStructPGridNVars(pgrid);
   hypre_SStructPVectorNVars(pvector) = nvars;
   svectors = hypre_TAlloc(hypre_StructVector *,  nvars, HYPRE_MEMORY_HOST);

   for (var = 0; var < nvars; var++)
   {
      sgrid = hypre_SStructPGridSGrid(pgrid, var);
      svectors[var] = hypre_StructVectorCreate(comm, sgrid);
   }
   hypre_SStructPVectorSVectors(pvector) = svectors;
   comm_pkgs = hypre_TAlloc(hypre_CommPkg *,  nvars, HYPRE_MEMORY_HOST);
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

         hypre_TFree(dataindices, HYPRE_MEMORY_HOST);
         hypre_TFree(svectors, HYPRE_MEMORY_HOST);
         hypre_TFree(comm_pkgs, HYPRE_MEMORY_HOST);
         hypre_TFree(pvector, HYPRE_MEMORY_HOST);
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
                               HYPRE_Complex        *value,
                               HYPRE_Int             action )
{
   hypre_StructVector *svector = hypre_SStructPVectorSVector(pvector, var);
   HYPRE_Int           ndim = hypre_StructVectorNDim(svector);
   hypre_BoxArray     *grid_boxes;
   hypre_Box          *box, *grow_box;
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
         if (hypre_IndexInBox(index, box))
         {
            done = 1;
            break;
         }
      }

      if (!done)
      {
         grow_box = hypre_BoxCreate(ndim);
         hypre_SStructVariableGetOffset(
            hypre_SStructPGridVarType(pgrid, var), ndim, varoffset);
         hypre_ForBoxI(i, grid_boxes)
         {
            box = hypre_BoxArrayBox(grid_boxes, i);
            hypre_CopyBox(box, grow_box);
            hypre_BoxGrowByIndex(grow_box, varoffset);
            if (hypre_IndexInBox(index, grow_box))
            {
               hypre_StructVectorSetValues(svector, index, value, action, i, 1);
               break;
            }
         }
         hypre_BoxDestroy(grow_box);
      }
   }
   else
   {
      /* Set */
      grid_boxes = hypre_StructGridBoxes(hypre_StructVectorGrid(svector));

      hypre_ForBoxI(i, grid_boxes)
      {
         box = hypre_BoxArrayBox(grid_boxes, i);
         if (!hypre_IndexInBox(index, box))
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
                                  hypre_Box            *set_box,
                                  HYPRE_Int             var,
                                  hypre_Box            *value_box,
                                  HYPRE_Complex        *values,
                                  HYPRE_Int             action )
{
   hypre_StructVector *svector = hypre_SStructPVectorSVector(pvector, var);
   HYPRE_Int           ndim = hypre_StructVectorNDim(svector);
   hypre_BoxArray     *grid_boxes;
   HYPRE_Int           i, j;

   /* set values inside the grid */
   hypre_StructVectorSetBoxValues(svector, set_box, value_box, values, action, -1, 0);

   /* TODO: Why need DeviceSync? */
#if defined(HYPRE_USING_GPU)
   hypre_SyncCudaDevice(hypre_handle());
#endif
   /* set (AddTo/Get) or clear (Set) values outside the grid in ghost zones */
   if (action != 0)
   {
      /* AddTo/Get */
      hypre_SStructPGrid  *pgrid = hypre_SStructPVectorPGrid(pvector);
      hypre_Index          varoffset;
      hypre_BoxArray      *left_boxes, *done_boxes, *temp_boxes;
      hypre_Box           *left_box, *done_box, *int_box;

      hypre_SStructVariableGetOffset(
         hypre_SStructPGridVarType(pgrid, var), ndim, varoffset);
      grid_boxes = hypre_StructGridBoxes(hypre_StructVectorGrid(svector));

      left_boxes = hypre_BoxArrayCreate(1, ndim);
      done_boxes = hypre_BoxArrayCreate(2, ndim);
      temp_boxes = hypre_BoxArrayCreate(0, ndim);

      /* done_box always points to the first box in done_boxes */
      done_box = hypre_BoxArrayBox(done_boxes, 0);
      /* int_box always points to the second box in done_boxes */
      int_box = hypre_BoxArrayBox(done_boxes, 1);

      hypre_CopyBox(set_box, hypre_BoxArrayBox(left_boxes, 0));
      hypre_BoxArraySetSize(left_boxes, 1);
      hypre_SubtractBoxArrays(left_boxes, grid_boxes, temp_boxes);

      hypre_BoxArraySetSize(done_boxes, 0);
      hypre_ForBoxI(i, grid_boxes)
      {
         hypre_SubtractBoxArrays(left_boxes, done_boxes, temp_boxes);
         hypre_BoxArraySetSize(done_boxes, 1);
         hypre_CopyBox(hypre_BoxArrayBox(grid_boxes, i), done_box);
         hypre_BoxGrowByIndex(done_box, varoffset);
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
      diff_boxes = hypre_BoxArrayCreate(0, ndim);

      hypre_ForBoxI(i, grid_boxes)
      {
         grid_box = hypre_BoxArrayBox(grid_boxes, i);
         hypre_BoxArraySetSize(diff_boxes, 0);
         hypre_SubtractBoxes(set_box, grid_box, diff_boxes);

         hypre_ForBoxI(j, diff_boxes)
         {
            diff_box = hypre_BoxArrayBox(diff_boxes, j);
            hypre_StructVectorClearBoxValues(svector, diff_box, i, 1);
         }
      }
      hypre_BoxArrayDestroy(diff_boxes);
   }

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
   HYPRE_Int              num_ghost[2 * HYPRE_MAXDIM];
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
         for (d = 0; d < ndim; d++)
         {
            num_ghost[2 * d]   = num_ghost[2 * d + 1] = hypre_IndexD(varoffset, d);
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
                               HYPRE_Complex        *value )
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
                                  hypre_Box            *set_box,
                                  HYPRE_Int             var,
                                  hypre_Box            *value_box,
                                  HYPRE_Complex        *values )
{
   hypre_SStructPGrid *pgrid     = hypre_SStructPVectorPGrid(pvector);
   hypre_StructVector *svector   = hypre_SStructPVectorSVector(pvector, var);
   hypre_StructGrid   *sgrid     = hypre_StructVectorGrid(svector);
   hypre_BoxArray     *iboxarray = hypre_SStructPGridIBoxArray(pgrid, var);
   hypre_BoxArray     *tboxarray;

   /* temporarily swap out sgrid boxes in order to get boundary data */
   tboxarray = hypre_StructGridBoxes(sgrid);
   hypre_StructGridBoxes(sgrid) = iboxarray;
   hypre_StructVectorSetBoxValues(svector, set_box, value_box, values, -1, -1, 0);
   hypre_StructGridBoxes(sgrid) = tboxarray;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructPVectorSetConstantValues( hypre_SStructPVector *pvector,
                                       HYPRE_Complex         value )
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
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructVectorSetConstantValues( hypre_SStructVector *vector,
                                      HYPRE_Complex        value )
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
   HYPRE_Complex        *pardata;
   HYPRE_Int             pari;

   hypre_SStructPVector *pvector;
   hypre_StructVector   *y;
   hypre_Box            *y_data_box;
   HYPRE_Complex        *yp;
   hypre_BoxArray       *boxes;
   hypre_Box            *box;
   hypre_Index           loop_size;
   hypre_IndexRef        start;
   hypre_Index           stride;

   HYPRE_Int             nparts, nvars;
   HYPRE_Int             part, var, i;

   hypre_SetIndex(stride, 1);

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

#undef DEVICE_VAR
#define DEVICE_VAR is_device_ptr(pardata,yp)
            hypre_BoxLoop2Begin(hypre_SStructVectorNDim(vector), loop_size,
                                y_data_box, start, stride, yi,
                                box,        start, stride, bi);
            {
               pardata[pari + bi] = yp[yi];
            }
            hypre_BoxLoop2End(yi, bi);
#undef DEVICE_VAR
#define DEVICE_VAR

            pari += hypre_BoxVolume(box);
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
   HYPRE_UNUSED_VAR(vector);
   HYPRE_UNUSED_VAR(parvector);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Copy values from parvector to vector
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructVectorParRestore( hypre_SStructVector *vector,
                               hypre_ParVector     *parvector )
{
   HYPRE_Complex        *pardata;
   HYPRE_Int             pari;

   hypre_SStructPVector *pvector;
   hypre_StructVector   *y;
   hypre_Box            *y_data_box;
   HYPRE_Complex        *yp;
   hypre_BoxArray       *boxes;
   hypre_Box            *box;
   hypre_Index           loop_size;
   hypre_IndexRef        start;
   hypre_Index           stride;

   HYPRE_Int             nparts, nvars;
   HYPRE_Int             part, var, i;

   if (parvector != NULL)
   {
      hypre_SetIndex(stride, 1);

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

#undef DEVICE_VAR
#define DEVICE_VAR is_device_ptr(yp,pardata)
               hypre_BoxLoop2Begin(hypre_SStructVectorNDim(vector), loop_size,
                                   y_data_box, start, stride, yi,
                                   box,        start, stride, bi);
               {
                  yp[yi] = pardata[pari + bi];
               }
               hypre_BoxLoop2End(yi, bi);
#undef DEVICE_VAR
#define DEVICE_VAR

               pari += hypre_BoxVolume(box);
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
   pdataindices = hypre_CTAlloc(HYPRE_Int,  nvars, HYPRE_MEMORY_HOST);

   for (var = 0; var < nvars; var++)
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
   hypre_SStructPVectorDataSize(pvector) = pdatasize + nucvars ;

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
   dataindices = hypre_CTAlloc(HYPRE_Int,  nparts, HYPRE_MEMORY_HOST);
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
   HYPRE_Int              nparts = hypre_SStructVectorNParts(vector);
   hypre_SStructPVector  *pvector;
   hypre_StructVector    *svector;

   HYPRE_Int    part;
   HYPRE_Int    nvars, var;

   for (part = 0; part < nparts; part++)
   {
      pvector = hypre_SStructVectorPVector(vector, part);
      nvars  = hypre_SStructPVectorNVars(pvector);

      for (var = 0; var < nvars; var++)
      {
         svector = hypre_SStructPVectorSVector(pvector, var);
         hypre_StructVectorClearGhostValues(svector);
      }
   }

   return hypre_error_flag;
}

HYPRE_MemoryLocation
hypre_SStructVectorMemoryLocation(hypre_SStructVector *vector)
{
   HYPRE_Int type = hypre_SStructVectorObjectType(vector);

   if (type == HYPRE_SSTRUCT)
   {
      hypre_ParVector *parvector;
      hypre_SStructVectorConvert(vector, &parvector);
      return hypre_ParVectorMemoryLocation(parvector);
   }

   void *object;
   HYPRE_SStructVectorGetObject(vector, &object);

   if (type == HYPRE_PARCSR)
   {
      return hypre_ParVectorMemoryLocation((hypre_ParVector *) object);
   }

   if (type == HYPRE_STRUCT)
   {
      return hypre_StructVectorMemoryLocation((hypre_StructVector *) object);
   }

   return HYPRE_MEMORY_UNDEFINED;
}
