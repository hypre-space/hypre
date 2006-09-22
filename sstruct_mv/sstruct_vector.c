/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team. UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer, contact information and the GNU Lesser General Public License.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the 
 * terms of the GNU General Public License (as published by the Free Software
 * Foundation) version 2.1 dated February 1999.
 *
 * HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
 * WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
 * FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
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
 * hypre_SStructVectorRef
 *--------------------------------------------------------------------------*/

int
hypre_SStructPVectorRef( hypre_SStructPVector  *vector,
                         hypre_SStructPVector **vector_ref )
{
   hypre_SStructPVectorRefCount(vector) ++;
   *vector_ref = vector;

   return 0;
}

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
   HYPRE_SStructVariable *vartypes= hypre_SStructPGridVarTypes(pgrid);
   int                    ndim    = hypre_SStructPGridNDim(pgrid);
   hypre_Index            varoffset;
   int                    var, d;
 
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

      /* set the Add_num_ghost layer */
      if (vartypes[var] > 0)
      {
         sgrid = hypre_StructVectorGrid(svectors[var]);
         hypre_SStructVariableGetOffset(vartypes[var], ndim, varoffset);
         for (d = 0; d < 3; d++)
         {
            hypre_StructVectorAddNumGhost(svectors[var])[2*d]= 
                                           hypre_IndexD(varoffset, d);
            hypre_StructVectorAddNumGhost(svectors[var])[2*d+1]= 
                                           hypre_IndexD(varoffset, d);
         }
      }
         
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

   /* GEC destroying dataindices and data in pvector   */

   int                *dataindices;

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
                               double               *value,
                               int                   add_to )
{
   int ierr = 0;
   hypre_StructVector *svector = hypre_SStructPVectorSVector(pvector, var);

   ierr = hypre_StructVectorSetValues(svector, index, *value, add_to);

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

   hypre_CommInfo        *comm_info;

   int                    ndim      = hypre_SStructPGridNDim(pgrid);
   HYPRE_SStructVariable *vartypes  = hypre_SStructPGridVarTypes(pgrid);

   hypre_Index            varoffset;
   int                    num_ghost[6];
   hypre_StructGrid      *sgrid;
   int                    var, d;

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
         
         hypre_CreateCommInfoFromNumGhost(sgrid, num_ghost, &comm_info);
         hypre_CommPkgDestroy(comm_pkgs[var]);
         hypre_CommPkgCreate(comm_info,
                             hypre_StructVectorDataSpace(svectors[var]),
                             hypre_StructVectorDataSpace(svectors[var]),
                             1, hypre_StructVectorComm(svectors[var]),
                             &comm_pkgs[var]);
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
hypre_SStructPVectorPrint( const char           *filename,
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
 * Here the address of the parvector inside the semistructured vector
 * is provided to the "outside". It assumes that the vector type
 * is HYPRE_SSTRUCT
 *--------------------------------------------------------------------------*/

int
hypre_SStructVectorConvert( hypre_SStructVector  *vector,
                            hypre_ParVector     **parvector_ptr )
{
   int ierr = 0;

  *parvector_ptr = hypre_SStructVectorParVector(vector);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SStructParVectorConvert
 *
 * Copy values from vector to parvector and provide the address
 *--------------------------------------------------------------------------*/

int
hypre_SStructVectorParConvert( hypre_SStructVector  *vector,
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
   int                   bi;
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

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SStructVectorRestore
 * used for HYPRE_SSTRUCT type semi structured vectors.
 * A dummy function to indicate that the struct vector part will be used.
 *--------------------------------------------------------------------------*/

int
hypre_SStructVectorRestore( hypre_SStructVector *vector,
                            hypre_ParVector     *parvector )
{
   int ierr = 0;
   
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SStructVectorParRestore
 *
 * Copy values from parvector to vector
 *--------------------------------------------------------------------------*/

int
hypre_SStructVectorParRestore( hypre_SStructVector *vector,
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
   int                   bi;
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

   return ierr;
}
/*------------------------------------------------------------------
 *  GEC1002 shell initialization of a pvector 
 *   if the pvector exists. This function will set the dataindices
 *  and datasize of the pvector. Datasize is the sum of the sizes
 *  of each svector and dataindices is defined as
 *  dataindices[var]= aggregated initial size of the pvector[var]
 *  When ucvars are present we need to modify adding nucvars.
 *----------------------------------------------------------------*/   
int 
hypre_SStructPVectorInitializeShell( hypre_SStructPVector *pvector)
{
  int   ierr=0;
  int   nvars = hypre_SStructPVectorNVars(pvector);
  int   var  ;
  int   pdatasize;
  int   svectdatasize;
  int   *pdataindices;
  int   nucvars = 0;

  hypre_StructVector  *svector;

  pdatasize = 0;
  pdataindices = hypre_CTAlloc(int, nvars);

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

  return ierr;
}
     
/*------------------------------------------------------------------
 *  GEC1002 shell initialization of a sstructvector
 *  if the vector exists. This function will set the
 *  dataindices and datasize of the vector. When ucvars
 *  are present at the end of all the parts we need to modify adding pieces
 *  for ucvars.
 *----------------------------------------------------------------*/  
int 
hypre_SStructVectorInitializeShell( hypre_SStructVector *vector)
{
  int                      ierr = 0;
  int                      part  ;
  int                      datasize;
  int                      pdatasize;
  int                      nparts = hypre_SStructVectorNParts(vector); 
  hypre_SStructPVector    *pvector;
  int                     *dataindices;

  datasize = 0;
  dataindices = hypre_CTAlloc(int, nparts);
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

  return ierr;
}   


int
hypre_SStructVectorClearGhostValues(hypre_SStructVector *vector)
{
  int                    ierr= 0;

  int                    nparts= hypre_SStructVectorNParts(vector);
  hypre_SStructPVector  *pvector;
  hypre_StructVector    *svector;

  int    part;
  int    nvars, var;

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

  return ierr;
}   



