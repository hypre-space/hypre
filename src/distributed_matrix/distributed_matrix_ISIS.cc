/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.5 $
 ***********************************************************************EHEADER*/





/* THIS IS A C++ FILE, since it needs to call ISIS++ with objects */

#ifdef ISIS_AVAILABLE
#include "iostream.h"
#include "RowMatrix.h"  // ISIS++ header file
#endif

#include "./distributed_matrix.h"

#ifdef ISIS_AVAILABLE
extern "C" {

typedef struct
{
    HYPRE_Int *ind;
    double *val;
} 
RowBuf;
#endif

/*--------------------------------------------------------------------------
 * hypre_InitializeDistributedMatrixISIS
 *--------------------------------------------------------------------------*/

  /* matrix must be set before calling this function*/

HYPRE_Int 
hypre_InitializeDistributedMatrixISIS(hypre_DistributedMatrix *dm)
{
#ifdef ISIS_AVAILABLE
   RowMatrix *mat = (RowMatrix *) hypre_DistributedMatrixLocalStorage(dm);

   const Map& map = mat->getMap();

   HYPRE_Int num_rows = mat->getMap().n();
   HYPRE_Int num_cols = mat->getMap().n();
   
   hypre_DistributedMatrixM(dm) = num_rows;
   hypre_DistributedMatrixN(dm) = num_cols;

   /* allocate space for row buffers */

   RowBuf *rowbuf = new RowBuf;
   rowbuf->ind = new HYPRE_Int[num_cols];
   rowbuf->val = new double[num_cols];

   dm->auxiliary_data = (void *) rowbuf;
#endif

   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_FreeDistributedMatrixISIS
 *--------------------------------------------------------------------------*/

HYPRE_Int 
hypre_FreeDistributedMatrixISIS( hypre_DistributedMatrix *dm)
{
#ifdef ISIS_AVAILABLE
   RowBuf *rowbuf = (RowBuf *) dm->auxiliary_data;

   delete rowbuf->ind;
   delete rowbuf->val;
   delete rowbuf;
#endif

   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_PrintDistributedMatrixISIS
 *--------------------------------------------------------------------------*/

HYPRE_Int 
hypre_PrintDistributedMatrixISIS( hypre_DistributedMatrix *matrix )
{
#ifdef ISIS_AVAILABLE
   cout << "hypre_PrintDistributedMatrixISIS not implemented" << endl;
#endif

   return -1;
}

/*--------------------------------------------------------------------------
 * hypre_GetDistributedMatrixLocalRangeISIS
 *--------------------------------------------------------------------------*/

HYPRE_Int 
hypre_GetDistributedMatrixLocalRangeISIS( hypre_DistributedMatrix *dm,
                             HYPRE_Int *start,
                             HYPRE_Int *end )
{
#ifdef ISIS_AVAILABLE
   RowMatrix *mat = (RowMatrix *) hypre_DistributedMatrixLocalStorage(dm);

   *start = mat->getMap().startRow() - 1;  // convert to 0-based
   *end = mat->getMap().endRow(); // endRow actually returns 1 less
   
   cout << "LocalRangeISIS " << *start << "  " << *end << endl;
#endif
   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_GetDistributedMatrixRowISIS
 *--------------------------------------------------------------------------*/

/* semantics: buffers returned will be overwritten on next call to 
// this get function */

HYPRE_Int 
hypre_GetDistributedMatrixRowISIS( hypre_DistributedMatrix *dm,
                             HYPRE_Int row,
                             HYPRE_Int *size,
                             HYPRE_Int **col_ind,
                             double **values )
{
#ifdef ISIS_AVAILABLE
   RowMatrix *mat = (RowMatrix *) hypre_DistributedMatrixLocalStorage(dm);
   RowBuf *rowbuf;
   HYPRE_Int i, temp;

   rowbuf = (RowBuf *) dm->auxiliary_data;

   mat->getRow(row+1, temp, rowbuf->val, rowbuf->ind);

#if 0
   // add diagonal element if necessary
   {
       HYPRE_Int *p;
       HYPRE_Int found = 0;

       for (i=0, p=rowbuf->ind; i<temp; i++, p++)
       {
	   if (*p == row+1) 
	       found = 1;
       }

       if (!found)
       {
	   rowbuf->ind[temp] = row+1;
	   rowbuf->val[temp] = 1.; // pick a value
	   temp++;
       }
   }
#endif

   // set pointers to local buffers
   if (col_ind != NULL)
   {
       HYPRE_Int *p;

       *size = temp;
       *col_ind = rowbuf->ind;

       // need to convert to 0-based indexing for output
       for (i=0, p=*col_ind; i<temp; i++, p++)
	   (*p)--;
   }

   if (values != NULL)
   {
       *values = rowbuf->val;
       *size = temp;
   }

#endif

   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_RestoreDistributedMatrixRowISIS
 *--------------------------------------------------------------------------*/

HYPRE_Int 
hypre_RestoreDistributedMatrixRowISIS( hypre_DistributedMatrix *dm,
                             HYPRE_Int row,
                             HYPRE_Int *size,
                             HYPRE_Int **col_ind,
                             double **values )
{
  /* does nothing, since we use local buffers */

   return 0;
}

#ifdef ISIS_AVAILABLE
} // extern "C"
#endif
