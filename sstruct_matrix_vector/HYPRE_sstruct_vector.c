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
 * HYPRE_SStructVector interface
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * HYPRE_SStructVectorCreate
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructVectorCreate( MPI_Comm              comm,
                           HYPRE_SStructGrid     grid,
                           HYPRE_SStructVector  *vector_ptr )
{
   int ierr = 0;

   hypre_SStructVector   *vector;
   int                    nparts;
   hypre_SStructPVector **pvectors;
   MPI_Comm               pcomm;
   hypre_SStructPGrid    *pgrid;
   int                    part;

   vector = hypre_TAlloc(hypre_SStructVector, 1);

   hypre_SStructVectorComm(vector) = comm;
   hypre_SStructVectorNDim(vector) = hypre_SStructGridNDim(grid);
   hypre_SStructVectorGrid(vector) = grid;
   nparts = hypre_SStructGridNParts(grid);
   hypre_SStructVectorNParts(vector) = nparts;
   pvectors = hypre_TAlloc(hypre_SStructPVector *, nparts);
   for (part = 0; part < nparts; part++)
   {
      pgrid = hypre_SStructGridPGrid(grid, part);
      pcomm = hypre_SStructPGridComm(pgrid);
      ierr = hypre_SStructPVectorCreate(pcomm, pgrid, &pvectors[part]);
   }
   hypre_SStructVectorPVectors(vector)   = pvectors;
   hypre_SStructVectorIJVector(vector)   = NULL;
   HYPRE_IJVectorCreate(comm, &hypre_SStructVectorIJVector(vector),
                        hypre_SStructGridGlobalSize(grid));
   hypre_SStructVectorParVector(vector)  = NULL;
   hypre_SStructVectorGlobalSize(vector) = 0;
   hypre_SStructVectorRefCount(vector)   = 1;

   *vector_ptr = vector;

   return ierr;
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructVectorDestroy TODO
 *--------------------------------------------------------------------------*/

int 
HYPRE_SStructVectorDestroy( HYPRE_SStructVector vector )
{
   int ierr = 0;

   return ierr;
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructVectorInitialize
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructVectorInitialize( HYPRE_SStructVector vector )
{
   int ierr = 0;
   int            nparts   = hypre_SStructVectorNParts(vector);
   HYPRE_IJVector ijvector = hypre_SStructVectorIJVector(vector);
   int            part;

   for (part = 0; part < nparts; part++)
   {
      hypre_SStructPVectorInitialize(hypre_SStructVectorPVector(vector, part));
   }

   /* u-vector */
   ierr  = HYPRE_IJVectorSetLocalStorageType(ijvector, HYPRE_PARCSR);
   ierr += HYPRE_IJVectorInitialize(ijvector);

   return ierr;
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructVectorSetValues
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructVectorSetValues( HYPRE_SStructVector  vector,
                              int                  part,
                              int                 *index,
                              int                  var,
                              double               value )
{
   int ierr = 0;
   int                   ndim    = hypre_SStructVectorNDim(vector);
   hypre_SStructPVector *pvector = hypre_SStructVectorPVector(vector, part);
   hypre_Index           cindex;

   hypre_CopyToCleanIndex(index, ndim, cindex);

   if (var < hypre_SStructPVectorNVars(pvector))
   {
      ierr = hypre_SStructPVectorSetValues(pvector, cindex, var, value, 0);
   }
   else
   {
      /* TODO */
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructVectorSetBoxValues
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructVectorSetBoxValues( HYPRE_SStructVector  vector,
                                 int                  part,
                                 int                 *ilower,
                                 int                 *iupper,
                                 int                  var,
                                 double              *values )
{
   int ierr = 0;
   int                   ndim    = hypre_SStructVectorNDim(vector);
   hypre_SStructPVector *pvector = hypre_SStructVectorPVector(vector, part);
   hypre_Index           cilower;
   hypre_Index           ciupper;

   hypre_CopyToCleanIndex(ilower, ndim, cilower);
   hypre_CopyToCleanIndex(iupper, ndim, ciupper);

   ierr = hypre_SStructPVectorSetBoxValues(pvector, cilower, ciupper,
                                           var, values, 0);

   return ierr;
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructVectorAddToValues
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructVectorAddToValues( HYPRE_SStructVector  vector,
                                int                  part,
                                int                 *index,
                                int                  var,
                                double               value )
{
   int ierr = 0;
   int                   ndim    = hypre_SStructVectorNDim(vector);
   hypre_SStructPVector *pvector = hypre_SStructVectorPVector(vector, part);
   hypre_Index           cindex;

   hypre_CopyToCleanIndex(index, ndim, cindex);

   if (var < hypre_SStructPVectorNVars(pvector))
   {
      ierr = hypre_SStructPVectorSetValues(pvector, cindex, var, value, 1);
   }
   else
   {
      /* TODO */
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructVectorAddToBoxValues
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructVectorAddToBoxValues( HYPRE_SStructVector  vector,
                                   int                  part,
                                   int                 *ilower,
                                   int                 *iupper,
                                   int                  var,
                                   double              *values )
{
   int ierr = 0;
   int                   ndim    = hypre_SStructVectorNDim(vector);
   hypre_SStructPVector *pvector = hypre_SStructVectorPVector(vector, part);
   hypre_Index           cilower;
   hypre_Index           ciupper;

   hypre_CopyToCleanIndex(ilower, ndim, cilower);
   hypre_CopyToCleanIndex(iupper, ndim, ciupper);

   ierr = hypre_SStructPVectorSetBoxValues(pvector, cilower, ciupper,
                                           var, values, 1);

   return ierr;
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructVectorAssemble
 *--------------------------------------------------------------------------*/

int 
HYPRE_SStructVectorAssemble( HYPRE_SStructVector vector )
{
   int ierr = 0;
   int            nparts   = hypre_SStructVectorNParts(vector);
   HYPRE_IJVector ijvector = hypre_SStructVectorIJVector(vector);
   int            part;

   for (part = 0; part < nparts; part++)
   {
      hypre_SStructPVectorAssemble(hypre_SStructVectorPVector(vector, part));
   }

   /* u-vector */
   ierr = HYPRE_IJVectorAssemble(ijvector);
   hypre_SStructVectorParVector(vector) =
      (hypre_ParVector *) HYPRE_IJVectorGetLocalStorage(ijvector);

   return ierr;
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructVectorGetValues
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructVectorGetValues( HYPRE_SStructVector  vector,
                              int                  part,
                              int                 *index,
                              int                  var,
                              double              *value )
{
   int ierr = 0;
   int                   ndim    = hypre_SStructVectorNDim(vector);
   hypre_SStructPVector *pvector = hypre_SStructVectorPVector(vector, part);
   hypre_Index           cindex;

   hypre_CopyToCleanIndex(index, ndim, cindex);

   /* TODO: migrate data? */

   if (var < hypre_SStructPVectorNVars(pvector))
   {
      ierr = hypre_SStructPVectorGetValues(pvector, cindex, var, value);
   }
   else
   {
      /* TODO */
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructVectorGetBoxValues
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructVectorGetBoxValues(HYPRE_SStructVector  vector,
                                int                  part,
                                int                 *ilower,
                                int                 *iupper,
                                int                  var,
                                double              *values )
{
   int ierr = 0;
   int                   ndim    = hypre_SStructVectorNDim(vector);
   hypre_SStructPVector *pvector = hypre_SStructVectorPVector(vector, part);
   hypre_Index           cilower;
   hypre_Index           ciupper;

   hypre_CopyToCleanIndex(ilower, ndim, cilower);
   hypre_CopyToCleanIndex(iupper, ndim, ciupper);

   /* TODO: migrate data? */

   ierr = hypre_SStructPVectorGetBoxValues(pvector, cilower, ciupper,
                                           var, values);

   return ierr;
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructVectorPrint: For now, just print multiple files
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructVectorPrint( char                *filename,
                          HYPRE_SStructVector  vector,
                          int                  all )
{
   int  ierr = 0;
   int  nparts = hypre_SStructVectorNParts(vector);
   int  part;
   char new_filename[255];

   for (part = 0; part < nparts; part++)
   {
      sprintf(new_filename, "%s.%02d", filename, part);
      hypre_SStructPVectorPrint(new_filename,
                                hypre_SStructVectorPVector(vector, part),
                                all);
   }

   return ierr;
}

