/*BHEADER**********************************************************************
 * (c) 2000   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
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
 * hypre_SStructPVectorCreate
 *--------------------------------------------------------------------------*/

int
hypre_SStructPVectorCreate( MPI_Comm               comm,
                            hypre_SStructPGrid    *pgrid,
                            hypre_SStructPVector **pvector_ptr)
{
   int ierr = 0;

   hypre_SStructPVector  *pvector;
   int                    nvars;
   hypre_StructVector   **svectors;
   hypre_CommPkg        **comm_pkgs;
   hypre_StructGrid      *sgrid;
   int                    var;

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

   *pvector_ptr = pvector;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SStructPVectorDestroy
 *--------------------------------------------------------------------------*/

int 
hypre_SStructPVectorDestroy( hypre_SStructPVector *pvector )
{
   int ierr = 0;

   int                  nvars;
   hypre_StructVector **svectors;
   hypre_CommPkg      **comm_pkgs;
   int                  var;

   if (pvector)
   {
      nvars     = hypre_SStructPVectorNVars(pvector);
      svectors = hypre_SStructPVectorSVectors(pvector);
      comm_pkgs = hypre_SStructPVectorCommPkgs(pvector);
      for (var = 0; var < nvars; var++)
      {
         hypre_StructVectorDestroy(svectors[var]);
         hypre_CommPkgDestroy(comm_pkgs[var]);
      }
      hypre_TFree(svectors);
      hypre_TFree(comm_pkgs);
      hypre_TFree(pvector);
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SStructPVectorInitialize
 *--------------------------------------------------------------------------*/

int 
hypre_SStructPVectorInitialize( hypre_SStructPVector *pvector )
{
   int ierr = 0;
   int nvars = hypre_SStructPVectorNVars(pvector);
   int var;

   for (var = 0; var < nvars; var++)
   {
      hypre_StructVectorInitialize(hypre_SStructPVectorSVector(pvector, var));
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SStructPVectorSetValues
 *--------------------------------------------------------------------------*/

int 
hypre_SStructPVectorSetValues( hypre_SStructPVector *pvector,
                               hypre_Index           index,
                               int                   var,
                               double                value,
                               int                   add_to )
{
   int ierr = 0;
   hypre_StructVector *svector = hypre_SStructPVectorSVector(pvector, var);
   int                 d;

   ierr = hypre_StructVectorSetValues(svector, index, value, add_to);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SStructPVectorSetBoxValues
 *--------------------------------------------------------------------------*/

int 
hypre_SStructPVectorSetBoxValues( hypre_SStructPVector *pvector,
                                  hypre_Index           ilower,
                                  hypre_Index           iupper,
                                  int                   var,
                                  double               *values,
                                  int                   add_to )
{
   int ierr = 0;
   hypre_StructVector *svector = hypre_SStructPVectorSVector(pvector, var);
   hypre_Box          *box;
   int                 d;

   box = hypre_BoxCreate();
   hypre_CopyIndex(ilower, hypre_BoxIMin(box));
   hypre_CopyIndex(iupper, hypre_BoxIMax(box));
   ierr = hypre_StructVectorSetBoxValues(svector, box, values, add_to );
   hypre_BoxDestroy(box);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SStructPVectorAssemble
 *--------------------------------------------------------------------------*/

int 
hypre_SStructPVectorAssemble( hypre_SStructPVector *pvector )
{
   int ierr = 0;

   hypre_SStructPGrid    *pgrid     = hypre_SStructPVectorPGrid(pvector);
   int                    nvars     = hypre_SStructPVectorNVars(pvector);
   hypre_StructVector   **svectors  = hypre_SStructPVectorSVectors(pvector);
   hypre_CommPkg        **comm_pkgs = hypre_SStructPVectorCommPkgs(pvector);

   int                    ndim      = hypre_SStructPGridNDim(pgrid);
   HYPRE_SStructVariable *vartypes  = hypre_SStructPGridVarTypes(pgrid);

   hypre_Index            varoffset;
   int                    num_ghost[6];
   hypre_StructGrid      *sgrid;
   hypre_BoxArrayArray   *send_boxes;
   hypre_BoxArrayArray   *recv_boxes;
   int                  **send_processes;
   int                  **recv_processes;
   hypre_Index            unit_stride;
   int                    var, d;

   hypre_SetIndex(unit_stride, 1, 1, 1);

   for (var = 0; var < nvars; var++)
   {
      hypre_StructVectorAssemble(svectors[var]);

      if (vartypes[var] > 0)
      {
         sgrid = hypre_StructVectorGrid(svectors[var]);
         hypre_SStructVariableGetOffset(vartypes[var], ndim, varoffset);
         for (d = 0; d < 3; d++)
         {
            num_ghost[2*d]   = hypre_IndexD(varoffset, d);
            num_ghost[2*d+1] = hypre_IndexD(varoffset, d);
         }
         
         hypre_CreateCommInfoFromNumGhost(sgrid, num_ghost,
                                          &send_boxes, &recv_boxes,
                                          &send_processes, &recv_processes);
         
         hypre_CommPkgDestroy(comm_pkgs[var]);
         comm_pkgs[var] =
            hypre_CommPkgCreate(send_boxes, recv_boxes,
                                unit_stride, unit_stride,
                                hypre_StructVectorDataSpace(svectors[var]),
                                hypre_StructVectorDataSpace(svectors[var]),
                                send_processes, recv_processes, 1,
                                hypre_StructVectorComm(svectors[var]),
                                hypre_StructGridPeriodic(sgrid));
      }
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SStructPVectorGather
 *--------------------------------------------------------------------------*/

int 
hypre_SStructPVectorGather( hypre_SStructPVector *pvector )
{
   int ierr = 0;
   int                    nvars     = hypre_SStructPVectorNVars(pvector);
   hypre_StructVector   **svectors  = hypre_SStructPVectorSVectors(pvector);
   hypre_CommPkg        **comm_pkgs = hypre_SStructPVectorCommPkgs(pvector);
   hypre_CommHandle      *comm_handle;
   int                    var;

   for (var = 0; var < nvars; var++)
   {
      if (comm_pkgs[var] != NULL)
      {
         hypre_InitializeCommunication(comm_pkgs[var],
                                       hypre_StructVectorData(svectors[var]),
                                       hypre_StructVectorData(svectors[var]),
                                       &comm_handle);
         hypre_FinalizeCommunication(comm_handle);
      }
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SStructPVectorGetValues
 *--------------------------------------------------------------------------*/

int 
hypre_SStructPVectorGetValues( hypre_SStructPVector *pvector,
                               hypre_Index           index,
                               int                   var,
                               double               *value )
{
   int ierr = 0;
   hypre_SStructPGrid *pgrid     = hypre_SStructPVectorPGrid(pvector);
   hypre_StructVector *svector   = hypre_SStructPVectorSVector(pvector, var);
   hypre_StructGrid   *sgrid     = hypre_StructVectorGrid(svector);
   hypre_BoxArray     *iboxarray = hypre_SStructPGridIBoxArray(pgrid, var);
   hypre_BoxArray     *tboxarray;

   /* temporarily swap out sgrid boxes in order to get boundary data */
   tboxarray = hypre_StructGridBoxes(sgrid);
   hypre_StructGridBoxes(sgrid) = iboxarray;
   ierr = hypre_StructVectorGetValues(svector, index, value);
   hypre_StructGridBoxes(sgrid) = tboxarray;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SStructPVectorGetBoxValues
 *--------------------------------------------------------------------------*/

int 
hypre_SStructPVectorGetBoxValues( hypre_SStructPVector *pvector,
                                  hypre_Index           ilower,
                                  hypre_Index           iupper,
                                  int                   var,
                                  double               *values )
{
   int ierr = 0;
   hypre_SStructPGrid *pgrid     = hypre_SStructPVectorPGrid(pvector);
   hypre_StructVector *svector   = hypre_SStructPVectorSVector(pvector, var);
   hypre_StructGrid   *sgrid     = hypre_StructVectorGrid(svector);
   hypre_BoxArray     *iboxarray = hypre_SStructPGridIBoxArray(pgrid, var);
   hypre_BoxArray     *tboxarray;
   hypre_Box          *box;
   int                 d;

   box = hypre_BoxCreate();
   hypre_CopyIndex(ilower, hypre_BoxIMin(box));
   hypre_CopyIndex(iupper, hypre_BoxIMax(box));
   /* temporarily swap out sgrid boxes in order to get boundary data */
   tboxarray = hypre_StructGridBoxes(sgrid);
   hypre_StructGridBoxes(sgrid) = iboxarray;
   ierr = hypre_StructVectorGetBoxValues(svector, box, values);
   hypre_StructGridBoxes(sgrid) = tboxarray;
   hypre_BoxDestroy(box);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SStructPVectorSetConstantValues
 *--------------------------------------------------------------------------*/

int 
hypre_SStructPVectorSetConstantValues( hypre_SStructPVector *pvector,
                                       double                value )
{
   int ierr = 0;
   int                 nvars = hypre_SStructPVectorNVars(pvector);
   hypre_StructVector *svector;
   int                 var;

   for (var = 0; var < nvars; var++)
   {
      svector = hypre_SStructPVectorSVector(pvector, var);
      hypre_StructVectorSetConstantValues(svector, value);
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SStructPVectorPrint: For now, just print multiple files
 *--------------------------------------------------------------------------*/

int
hypre_SStructPVectorPrint( char                 *filename,
                           hypre_SStructPVector *pvector,
                           int                   all )
{
   int ierr = 0;
   int  nvars = hypre_SStructPVectorNVars(pvector);
   int  var;
   char new_filename[255];

   for (var = 0; var < nvars; var++)
   {
      sprintf(new_filename, "%s.%02d", filename, var);
      hypre_StructVectorPrint(new_filename,
                              hypre_SStructPVectorSVector(pvector, var),
                              all);
   }

   return ierr;
}

/*==========================================================================
 * SStructVector routines
 *==========================================================================*/

/*--------------------------------------------------------------------------
 * hypre_SStructVectorRef
 *--------------------------------------------------------------------------*/

int
hypre_SStructVectorRef( hypre_SStructVector  *vector,
                        hypre_SStructVector **vector_ref )
{
   hypre_SStructVectorRefCount(vector) ++;
   *vector_ref = vector;

   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_SStructVectorSetConstantValues
 *--------------------------------------------------------------------------*/

int 
hypre_SStructVectorSetConstantValues( hypre_SStructVector *vector,
                                      double               value )
{
   int ierr = 0;
   int                   nparts = hypre_SStructVectorNParts(vector);
   hypre_SStructPVector *pvector;
   int                   part;

   for (part = 0; part < nparts; part++)
   {
      pvector = hypre_SStructVectorPVector(vector, part);
      hypre_SStructPVectorSetConstantValues(pvector, value);
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SStructVectorConvert
 *
 * Copy values from vector to parvector
 *--------------------------------------------------------------------------*/

int
hypre_SStructVectorConvert( hypre_SStructVector  *vector,
                            hypre_ParVector     **parvector_ptr )
{
   int ierr = 0;

   hypre_ParVector      *parvector;
   double               *pardata;
   int                   pari;

   hypre_SStructPVector *pvector;
   hypre_StructVector   *y;
   hypre_Box            *y_data_box;
   int                   yi;
   double               *yp;
   hypre_BoxArray       *boxes;
   hypre_Box            *box;
   hypre_Index           loop_size;
   hypre_IndexRef        start;
   hypre_Index           stride;
                        
   int                   nparts, nvars;
   int                   part, var, i;
   int                   loopi, loopj, loopk;

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
               hypre_BoxLoop1Begin(loop_size, y_data_box, start, stride, yi);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,yi
#include "hypre_box_smp_forloop.h"
               hypre_BoxLoop1For(loopi, loopj, loopk, yi)
                  {
                     pardata[pari++] = yp[yi];
                  }
               hypre_BoxLoop1End(yi);
            }
      }
   }

   *parvector_ptr = hypre_SStructVectorParVector(vector);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SStructVectorRestore
 *
 * Copy values from parvector to vector
 *--------------------------------------------------------------------------*/

int
hypre_SStructVectorRestore( hypre_SStructVector *vector,
                            hypre_ParVector     *parvector )
{
   int ierr = 0;

   double               *pardata;
   int                   pari;

   hypre_SStructPVector *pvector;
   hypre_StructVector   *y;
   hypre_Box            *y_data_box;
   int                   yi;
   double               *yp;
   hypre_BoxArray       *boxes;
   hypre_Box            *box;
   hypre_Index           loop_size;
   hypre_IndexRef        start;
   hypre_Index           stride;
                        
   int                   nparts, nvars;
   int                   part, var, i;
   int                   loopi, loopj, loopk;

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
                  hypre_BoxLoop1Begin(loop_size, y_data_box, start, stride, yi);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,yi
#include "hypre_box_smp_forloop.h"
                  hypre_BoxLoop1For(loopi, loopj, loopk, yi)
                     {
                        yp[yi] = pardata[pari++];
                     }
                  hypre_BoxLoop1End(yi);
               }
         }
      }
   }

   return ierr;
}

