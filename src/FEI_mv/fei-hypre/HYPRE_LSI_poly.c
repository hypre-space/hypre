/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * HYPRE_LSI_POLY interface
 *
 *****************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "utilities/_hypre_utilities.h"
#include "HYPRE.h"
#include "IJ_mv/HYPRE_IJ_mv.h"
#include "parcsr_mv/HYPRE_parcsr_mv.h"
#include "parcsr_mv/_hypre_parcsr_mv.h"
#include "parcsr_ls/HYPRE_parcsr_ls.h"
#include "HYPRE_MHMatrix.h"
#include "HYPRE_FEI.h"

typedef struct HYPRE_LSI_Poly_Struct
{
   MPI_Comm  comm;
   int       order;
   double    *coefficients;
   int       Nrows;
   int       outputLevel;
}
HYPRE_LSI_Poly;

#define habs(x) ((x > 0) ? (x) : -(x))

/*--------------------------------------------------------------------------
 * HYPRE_LSI_PolyCreate - Return a polynomial preconditioner object "solver".
 *--------------------------------------------------------------------------*/

int HYPRE_LSI_PolyCreate( MPI_Comm comm, HYPRE_Solver *solver )
{
   HYPRE_LSI_Poly *poly_ptr;

   poly_ptr = hypre_TAlloc(HYPRE_LSI_Poly, 1, HYPRE_MEMORY_HOST);

   if (poly_ptr == NULL) return 1;

   poly_ptr->comm         = comm;
   poly_ptr->order        = 0;
   poly_ptr->coefficients = NULL;
   poly_ptr->Nrows        = 0;
   poly_ptr->outputLevel  = 0;

   *solver = (HYPRE_Solver) poly_ptr;

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_LSI_PolyDestroy - Destroy a Poly object.
 *--------------------------------------------------------------------------*/

int HYPRE_LSI_PolyDestroy( HYPRE_Solver solver )
{
   HYPRE_LSI_Poly *poly_ptr;

   poly_ptr = (HYPRE_LSI_Poly *) solver;
   hypre_TFree(poly_ptr->coefficients, HYPRE_MEMORY_HOST);
   hypre_TFree(poly_ptr, HYPRE_MEMORY_HOST);

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_LSI_PolySetOrder - Set the order of the polynomial.
 *--------------------------------------------------------------------------*/

int HYPRE_LSI_PolySetOrder(HYPRE_Solver solver, int order )
{
   HYPRE_LSI_Poly *poly_ptr = (HYPRE_LSI_Poly *) solver;

   poly_ptr->order = order;
   if ( poly_ptr->order < 0 ) poly_ptr->order = 0;
   if ( poly_ptr->order > 8 ) poly_ptr->order = 8;
   hypre_TFree(poly_ptr->coefficients, HYPRE_MEMORY_HOST);
   poly_ptr->coefficients = NULL;

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_LSI_PolySetOutputLevel - Set debug level
 *--------------------------------------------------------------------------*/

int HYPRE_LSI_PolySetOutputLevel(HYPRE_Solver solver, int level)
{
   HYPRE_LSI_Poly *poly_ptr = (HYPRE_LSI_Poly *) solver;

   poly_ptr->outputLevel = level;

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_LSI_PolySolve - Solve function for Polynomial.
 *--------------------------------------------------------------------------*/

int HYPRE_LSI_PolySolve( HYPRE_Solver solver, HYPRE_ParCSRMatrix A,
                         HYPRE_ParVector b,   HYPRE_ParVector x )
{
   int            i, j, order, Nrows;
   double         *rhs, *soln, *orig_rhs, mult, *coefs;
   HYPRE_LSI_Poly *poly_ptr = (HYPRE_LSI_Poly *) solver;

   rhs  = hypre_VectorData(hypre_ParVectorLocalVector((hypre_ParVector *) b));
   soln = hypre_VectorData(hypre_ParVectorLocalVector((hypre_ParVector *) x));

   order = poly_ptr->order;
   Nrows = poly_ptr->Nrows;
   coefs = poly_ptr->coefficients;
   if ( coefs == NULL )
   {
      printf("HYPRE_LSI_PolySolve ERROR : PolySetup not called.\n");
      exit(1);
   }
   orig_rhs = hypre_TAlloc(double,  Nrows , HYPRE_MEMORY_HOST);
   for ( i = 0; i < Nrows; i++ )
   {
      orig_rhs[i] = rhs[i];
      soln[i] = rhs[i] * coefs[order];
   }
   for (i = order - 1; i >= 0; i-- )
   {
      HYPRE_ParCSRMatrixMatvec(1.0, A, x, 0.0, b);
      mult = coefs[i];
      for ( j = 0; j < Nrows; j++ )
         soln[j] = mult * orig_rhs[j] + rhs[j];
   }
   for ( i = 0; i < Nrows; i++ ) rhs[i] = orig_rhs[i];
   hypre_TFree(orig_rhs, HYPRE_MEMORY_HOST);

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_LSI_PolySetup - Set up function for LSI_Poly.                      *
 * abridged from AZTEC                                                      *
 *--------------------------------------------------------------------------*/

int HYPRE_LSI_PolySetup(HYPRE_Solver solver, HYPRE_ParCSRMatrix A_csr,
                        HYPRE_ParVector b,   HYPRE_ParVector x )
{
   int            i, j, my_id, startRow, endRow, order;
   int            pos_diag, neg_diag;
   int            rowLeng, *colInd, *row_partition;
   double         *coefs=NULL, rowsum, max_norm, *colVal;
   HYPRE_LSI_Poly *poly_ptr = (HYPRE_LSI_Poly *) solver;
#ifndef HYPRE_SEQUENTIAL
   double         dtemp;
#endif

   /* ---------------------------------------------------------------- */
   /* initialize structure                                             */
   /* ---------------------------------------------------------------- */

   order = poly_ptr->order;
   coefs = hypre_TAlloc(double, (order+1) , HYPRE_MEMORY_HOST);
   poly_ptr->coefficients = coefs;

   /* ---------------------------------------------------------------- */
   /* compute matrix norm                                              */
   /* ---------------------------------------------------------------- */

   HYPRE_ParCSRMatrixGetRowPartitioning( A_csr, &row_partition );
#ifdef HYPRE_SEQUENTIAL
   my_id = 0;
#else
   MPI_Comm_rank(poly_ptr->comm, &my_id);
#endif

   startRow  = row_partition[my_id];
   endRow    = row_partition[my_id+1] - 1;
   hypre_TFree( row_partition , HYPRE_MEMORY_HOST);
   poly_ptr->Nrows = endRow - startRow + 1;

   max_norm = 0.0;
   pos_diag = neg_diag = 0;
   for ( i = startRow; i <= endRow; i++ )
   {
      HYPRE_ParCSRMatrixGetRow(A_csr, i, &rowLeng, &colInd, &colVal);
      rowsum = 0.0;
      for (j = 0; j < rowLeng; j++)
      {
         rowsum += habs(colVal[j]);
         if ( colInd[j] == i && colVal[j] > 0.0 ) pos_diag++;
         if ( colInd[j] == i && colVal[j] < 0.0 ) neg_diag++;
      }
      if ( rowsum > max_norm ) max_norm = rowsum;
      HYPRE_ParCSRMatrixRestoreRow(A_csr, i, &rowLeng, &colInd, &colVal);
   }
#ifndef HYPRE_SEQUENTIAL
   MPI_Allreduce(&max_norm, &dtemp, 1, MPI_DOUBLE, MPI_MAX, poly_ptr->comm);
#endif
   if ( pos_diag == 0 && neg_diag > 0 ) max_norm = - max_norm;

   /* ---------------------------------------------------------------- */
   /* fill in the coefficient table                                    */
   /* ---------------------------------------------------------------- */

   switch ( order )
   {
       case 0: coefs[0] = 1.0;     break;
       case 1: coefs[0] = 5.0;     coefs[1] = -1.0;   break;
       case 2: coefs[0] = 14.0;    coefs[1] = -7.0;   coefs[2] = 1.0;
               break;
       case 3: coefs[0] = 30.0;    coefs[1] = -27.0;  coefs[2] = 9.0;
               coefs[3] = -1.0;    break;
       case 4: coefs[0] = 55.0;    coefs[1] = -77.0;   coefs[2] = 44.0;
               coefs[3] = -11.0;   coefs[4] = 1.0;     break;
       case 5: coefs[0] = 91.0;    coefs[1] = -182.0;  coefs[2] = 156.0;
               coefs[3] = -65.0;   coefs[4] = 13.0;    coefs[5] = -1.0;
               break;
       case 6: coefs[0] = 140.0;   coefs[1] = -378.0;  coefs[2] = 450.0;
               coefs[3] = -275.0;  coefs[4] = 90.0;    coefs[5] = -15.0;
               coefs[6] = 1.0;     break;
       case 7: coefs[0] = 204.0;   coefs[1] = -714.0;  coefs[2] = 1122.0;
               coefs[3] = -935.0;  coefs[4] = 442.0;   coefs[5] = -119.0;
               coefs[6] = 17.0;    coefs[7] = -1.0;    break;
       case 8: coefs[0] = 285.0;   coefs[1] = -1254.0; coefs[2] = 2508.0;
               coefs[3] = -2717.0; coefs[4] = 1729.0;  coefs[5] = -665.0;
               coefs[6] = 152.0;   coefs[7] = -19.0;   coefs[8] = 1.0;
               break;
   }
   for( i = 0; i <= order; i++ )
      coefs[i] *= pow( 4.0 / max_norm, (double) i);

   return 0;
}

