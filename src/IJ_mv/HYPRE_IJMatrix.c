/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * HYPRE_IJMatrix interface
 *
 *****************************************************************************/

#include "./_hypre_IJ_mv.h"

#include "../HYPRE.h"

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixCreate
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_IJMatrixCreate( MPI_Comm        comm,
                      HYPRE_BigInt    ilower,
                      HYPRE_BigInt    iupper,
                      HYPRE_BigInt    jlower,
                      HYPRE_BigInt    jupper,
                      HYPRE_IJMatrix *matrix )
{
   HYPRE_BigInt info[2];
   HYPRE_Int num_procs;
   HYPRE_Int myid;

   hypre_IJMatrix *ijmatrix;

   HYPRE_BigInt  row0, col0, rowN, colN;

   ijmatrix = hypre_CTAlloc(hypre_IJMatrix, 1, HYPRE_MEMORY_HOST);

   hypre_IJMatrixComm(ijmatrix)           = comm;
   hypre_IJMatrixObject(ijmatrix)         = NULL;
   hypre_IJMatrixTranslator(ijmatrix)     = NULL;
   hypre_IJMatrixAssumedPart(ijmatrix)    = NULL;
   hypre_IJMatrixObjectType(ijmatrix)     = HYPRE_UNITIALIZED;
   hypre_IJMatrixAssembleFlag(ijmatrix)   = 0;
   hypre_IJMatrixPrintLevel(ijmatrix)     = 0;
   hypre_IJMatrixOMPFlag(ijmatrix)        = 0;

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &myid);


   if (ilower > iupper + 1 || ilower < 0)
   {
      hypre_error_in_arg(2);
      hypre_TFree(ijmatrix, HYPRE_MEMORY_HOST);
      return hypre_error_flag;
   }

   if (iupper < -1)
   {
      hypre_error_in_arg(3);
      hypre_TFree(ijmatrix, HYPRE_MEMORY_HOST);
      return hypre_error_flag;
   }

   if (jlower > jupper + 1 || jlower < 0)
   {
      hypre_error_in_arg(4);
      hypre_TFree(ijmatrix, HYPRE_MEMORY_HOST);
      return hypre_error_flag;
   }

   if (jupper < -1)
   {
      hypre_error_in_arg(5);
      hypre_TFree(ijmatrix, HYPRE_MEMORY_HOST);
      return hypre_error_flag;
   }

   hypre_IJMatrixRowPartitioning(ijmatrix)[0] = ilower;
   hypre_IJMatrixRowPartitioning(ijmatrix)[1] = iupper + 1;
   hypre_IJMatrixColPartitioning(ijmatrix)[0] = jlower;
   hypre_IJMatrixColPartitioning(ijmatrix)[1] = jupper + 1;

   /* now we need the global number of rows and columns as well
      as the global first row and column index */

   /* proc 0 has the first row and col */
   if (myid == 0)
   {
      info[0] = ilower;
      info[1] = jlower;
   }
   hypre_MPI_Bcast(info, 2, HYPRE_MPI_BIG_INT, 0, comm);
   row0 = info[0];
   col0 = info[1];

   /* proc (num_procs-1) has the last row and col */
   if (myid == (num_procs - 1))
   {
      info[0] = iupper;
      info[1] = jupper;
   }
   hypre_MPI_Bcast(info, 2, HYPRE_MPI_BIG_INT, num_procs - 1, comm);

   rowN = info[0];
   colN = info[1];

   hypre_IJMatrixGlobalFirstRow(ijmatrix) = row0;
   hypre_IJMatrixGlobalFirstCol(ijmatrix) = col0;
   hypre_IJMatrixGlobalNumRows(ijmatrix) = rowN - row0 + 1;
   hypre_IJMatrixGlobalNumCols(ijmatrix) = colN - col0 + 1;

   *matrix = (HYPRE_IJMatrix) ijmatrix;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_IJMatrixDestroy( HYPRE_IJMatrix matrix )
{
   hypre_IJMatrix *ijmatrix = (hypre_IJMatrix *) matrix;

   if (!ijmatrix)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (ijmatrix)
   {
      if hypre_IJMatrixAssumedPart(ijmatrix)
      {
         hypre_AssumedPartitionDestroy((hypre_IJAssumedPart*)hypre_IJMatrixAssumedPart(ijmatrix));
      }
      if ( hypre_IJMatrixObjectType(ijmatrix) == HYPRE_PARCSR )
      {
         hypre_IJMatrixDestroyParCSR( ijmatrix );
      }
      else if ( hypre_IJMatrixObjectType(ijmatrix) != -1 )
      {
         hypre_error_in_arg(1);
         return hypre_error_flag;
      }
   }

   hypre_TFree(ijmatrix, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_IJMatrixInitialize( HYPRE_IJMatrix matrix )
{
   hypre_IJMatrix *ijmatrix = (hypre_IJMatrix *) matrix;

   if (!ijmatrix)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if ( hypre_IJMatrixObjectType(ijmatrix) == HYPRE_PARCSR )
   {
      hypre_IJMatrixInitializeParCSR( ijmatrix ) ;
   }
   else
   {
      hypre_error_in_arg(1);
   }

   return hypre_error_flag;

}

HYPRE_Int
HYPRE_IJMatrixInitialize_v2( HYPRE_IJMatrix matrix, HYPRE_MemoryLocation memory_location )
{
   hypre_IJMatrix *ijmatrix = (hypre_IJMatrix *) matrix;

   if (!ijmatrix)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if ( hypre_IJMatrixObjectType(ijmatrix) == HYPRE_PARCSR )
   {
      hypre_IJMatrixInitializeParCSR_v2( ijmatrix, memory_location ) ;
   }
   else
   {
      hypre_error_in_arg(1);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_IJMatrixSetPrintLevel( HYPRE_IJMatrix matrix,
                             HYPRE_Int print_level )
{
   hypre_IJMatrix *ijmatrix = (hypre_IJMatrix *) matrix;

   if (!ijmatrix)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   hypre_IJMatrixPrintLevel(ijmatrix) = (print_level > 0) ? print_level : 0;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * This is a helper routine to compute a prefix sum of integer values.
 *
 * The current implementation is okay for modest numbers of threads.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PrefixSumInt(HYPRE_Int   nvals,
                   HYPRE_Int  *vals,
                   HYPRE_Int  *sums)
{
   HYPRE_Int  j, nthreads, bsize;

   nthreads = hypre_NumThreads();
   bsize = (nvals + nthreads - 1) / nthreads; /* This distributes the remainder */

   if (nvals < nthreads || bsize == 1)
   {
      sums[0] = 0;
      for (j = 1; j < nvals; j++)
      {
         sums[j] += sums[j - 1] + vals[j - 1];
      }
   }
   else
   {

      /* Compute preliminary partial sums (in parallel) within each interval */
#ifdef HYPRE_USING_OPENMP
      #pragma omp parallel for private(j) HYPRE_SMP_SCHEDULE
#endif
      for (j = 0; j < nvals; j += bsize)
      {
         HYPRE_Int  i, n = hypre_min((j + bsize), nvals);

         sums[j] = 0;
         for (i = j + 1; i < n; i++)
         {
            sums[i] = sums[i - 1] + vals[i - 1];
         }
      }

      /* Compute final partial sums (in serial) for the first entry of every interval */
      for (j = bsize; j < nvals; j += bsize)
      {
         sums[j] = sums[j - bsize] + sums[j - 1] + vals[j - 1];
      }

      /* Compute final partial sums (in parallel) for the remaining entries */
#ifdef HYPRE_USING_OPENMP
      #pragma omp parallel for private(j) HYPRE_SMP_SCHEDULE
#endif
      for (j = bsize; j < nvals; j += bsize)
      {
         HYPRE_Int  i, n = hypre_min((j + bsize), nvals);

         for (i = j + 1; i < n; i++)
         {
            sums[i] += sums[j];
         }
      }
   }

   return hypre_error_flag;
}


/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_IJMatrixSetValues( HYPRE_IJMatrix       matrix,
                         HYPRE_Int            nrows,
                         HYPRE_Int           *ncols,
                         const HYPRE_BigInt  *rows,
                         const HYPRE_BigInt  *cols,
                         const HYPRE_Complex *values )
{
   hypre_IJMatrix *ijmatrix = (hypre_IJMatrix *) matrix;

   if (nrows == 0)
   {
      return hypre_error_flag;
   }

   if (!ijmatrix)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   /*
   if (!ncols)
   {
      hypre_error_in_arg(3);
      return hypre_error_flag;
   }
   */

   if (!rows)
   {
      hypre_error_in_arg(4);
      return hypre_error_flag;
   }

   if (!cols)
   {
      hypre_error_in_arg(5);
      return hypre_error_flag;
   }

   if (!values)
   {
      hypre_error_in_arg(6);
      return hypre_error_flag;
   }

   if ( hypre_IJMatrixObjectType(ijmatrix) != HYPRE_PARCSR )
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   HYPRE_IJMatrixSetValues2(matrix, nrows, ncols, rows, NULL, cols, values);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_IJMatrixSetValues2( HYPRE_IJMatrix       matrix,
                          HYPRE_Int            nrows,
                          HYPRE_Int           *ncols,
                          const HYPRE_BigInt  *rows,
                          const HYPRE_Int     *row_indexes,
                          const HYPRE_BigInt  *cols,
                          const HYPRE_Complex *values )
{
   hypre_IJMatrix *ijmatrix = (hypre_IJMatrix *) matrix;

   if (nrows == 0)
   {
      return hypre_error_flag;
   }

   if (!ijmatrix)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (nrows < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   /*
   if (!ncols)
   {
      hypre_error_in_arg(3);
      return hypre_error_flag;
   }
   */

   if (!rows)
   {
      hypre_error_in_arg(4);
      return hypre_error_flag;
   }

   if (!cols)
   {
      hypre_error_in_arg(6);
      return hypre_error_flag;
   }

   if (!values)
   {
      hypre_error_in_arg(7);
      return hypre_error_flag;
   }

   if ( hypre_IJMatrixObjectType(ijmatrix) != HYPRE_PARCSR )
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

#if defined(HYPRE_USING_GPU)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1( hypre_IJMatrixMemoryLocation(matrix) );

   if (exec == HYPRE_EXEC_DEVICE)
   {
      hypre_IJMatrixSetAddValuesParCSRDevice(ijmatrix, nrows, ncols, rows, row_indexes, cols, values,
                                             "set");
   }
   else
#endif
   {
      HYPRE_Int *row_indexes_tmp = (HYPRE_Int *) row_indexes;
      HYPRE_Int *ncols_tmp = ncols;

      if (!ncols_tmp)
      {
         HYPRE_Int i;
         ncols_tmp = hypre_TAlloc(HYPRE_Int, nrows, HYPRE_MEMORY_HOST);
         for (i = 0; i < nrows; i++)
         {
            ncols_tmp[i] = 1;
         }
      }

      if (!row_indexes)
      {
         row_indexes_tmp = hypre_CTAlloc(HYPRE_Int, nrows, HYPRE_MEMORY_HOST);
         hypre_PrefixSumInt(nrows, ncols_tmp, row_indexes_tmp);
      }

      if (hypre_IJMatrixOMPFlag(ijmatrix))
      {
         hypre_IJMatrixSetValuesOMPParCSR(ijmatrix, nrows, ncols_tmp, rows, row_indexes_tmp, cols, values);
      }
      else
      {
         hypre_IJMatrixSetValuesParCSR(ijmatrix, nrows, ncols_tmp, rows, row_indexes_tmp, cols, values);
      }

      if (!ncols)
      {
         hypre_TFree(ncols_tmp, HYPRE_MEMORY_HOST);
      }

      if (!row_indexes)
      {
         hypre_TFree(row_indexes_tmp, HYPRE_MEMORY_HOST);
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_IJMatrixSetConstantValues( HYPRE_IJMatrix matrix, HYPRE_Complex value)
{
   hypre_IJMatrix *ijmatrix = (hypre_IJMatrix *) matrix;

   if (!ijmatrix)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if ( hypre_IJMatrixObjectType(ijmatrix) == HYPRE_PARCSR )
   {
      return ( hypre_IJMatrixSetConstantValuesParCSR( ijmatrix, value));
   }
   else
   {
      hypre_error_in_arg(1);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_IJMatrixAddToValues( HYPRE_IJMatrix       matrix,
                           HYPRE_Int            nrows,
                           HYPRE_Int           *ncols,
                           const HYPRE_BigInt  *rows,
                           const HYPRE_BigInt  *cols,
                           const HYPRE_Complex *values )
{
   hypre_IJMatrix *ijmatrix = (hypre_IJMatrix *) matrix;

   if (nrows == 0)
   {
      return hypre_error_flag;
   }

   if (!ijmatrix)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (nrows < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   /*
   if (!ncols)
   {
      hypre_error_in_arg(3);
      return hypre_error_flag;
   }
   */

   if (!rows)
   {
      hypre_error_in_arg(4);
      return hypre_error_flag;
   }

   if (!cols)
   {
      hypre_error_in_arg(5);
      return hypre_error_flag;
   }

   if (!values)
   {
      hypre_error_in_arg(6);
      return hypre_error_flag;
   }

   if ( hypre_IJMatrixObjectType(ijmatrix) != HYPRE_PARCSR )
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   HYPRE_IJMatrixAddToValues2(matrix, nrows, ncols, rows, NULL, cols, values);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_IJMatrixAddToValues2( HYPRE_IJMatrix       matrix,
                            HYPRE_Int            nrows,
                            HYPRE_Int           *ncols,
                            const HYPRE_BigInt  *rows,
                            const HYPRE_Int     *row_indexes,
                            const HYPRE_BigInt  *cols,
                            const HYPRE_Complex *values )
{
   hypre_IJMatrix *ijmatrix = (hypre_IJMatrix *) matrix;

   if (nrows == 0)
   {
      return hypre_error_flag;
   }

   if (!ijmatrix)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (nrows < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   /*
   if (!ncols)
   {
      hypre_error_in_arg(3);
      return hypre_error_flag;
   }
   */

   if (!rows)
   {
      hypre_error_in_arg(4);
      return hypre_error_flag;
   }

   if (!cols)
   {
      hypre_error_in_arg(6);
      return hypre_error_flag;
   }

   if (!values)
   {
      hypre_error_in_arg(7);
      return hypre_error_flag;
   }

   if ( hypre_IJMatrixObjectType(ijmatrix) != HYPRE_PARCSR )
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

#if defined(HYPRE_USING_GPU)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1( hypre_IJMatrixMemoryLocation(matrix) );

   if (exec == HYPRE_EXEC_DEVICE)
   {
      hypre_IJMatrixSetAddValuesParCSRDevice(ijmatrix, nrows, ncols, rows, row_indexes, cols, values,
                                             "add");
   }
   else
#endif
   {
      HYPRE_Int *row_indexes_tmp = (HYPRE_Int *) row_indexes;
      HYPRE_Int *ncols_tmp = ncols;

      if (!ncols_tmp)
      {
         HYPRE_Int i;
         ncols_tmp = hypre_TAlloc(HYPRE_Int, nrows, HYPRE_MEMORY_HOST);
         for (i = 0; i < nrows; i++)
         {
            ncols_tmp[i] = 1;
         }
      }

      if (!row_indexes)
      {
         row_indexes_tmp = hypre_CTAlloc(HYPRE_Int, nrows, HYPRE_MEMORY_HOST);
         hypre_PrefixSumInt(nrows, ncols_tmp, row_indexes_tmp);
      }

      if (hypre_IJMatrixOMPFlag(ijmatrix))
      {
         hypre_IJMatrixAddToValuesOMPParCSR(ijmatrix, nrows, ncols_tmp, rows, row_indexes_tmp, cols, values);
      }
      else
      {
         hypre_IJMatrixAddToValuesParCSR(ijmatrix, nrows, ncols_tmp, rows, row_indexes_tmp, cols, values);
      }

      if (!ncols)
      {
         hypre_TFree(ncols_tmp, HYPRE_MEMORY_HOST);
      }

      if (!row_indexes)
      {
         hypre_TFree(row_indexes_tmp, HYPRE_MEMORY_HOST);
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_IJMatrixAssemble( HYPRE_IJMatrix matrix )
{
   hypre_IJMatrix *ijmatrix = (hypre_IJMatrix *) matrix;

   if (!ijmatrix)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if ( hypre_IJMatrixObjectType(ijmatrix) == HYPRE_PARCSR )
   {
#if defined(HYPRE_USING_GPU)
      HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1( hypre_IJMatrixMemoryLocation(matrix) );

      if (exec == HYPRE_EXEC_DEVICE)
      {
         return ( hypre_IJMatrixAssembleParCSRDevice( ijmatrix ) );
      }
      else
#endif
      {
         return ( hypre_IJMatrixAssembleParCSR( ijmatrix ) );
      }
   }
   else
   {
      hypre_error_in_arg(1);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_IJMatrixGetRowCounts( HYPRE_IJMatrix matrix,
                            HYPRE_Int      nrows,
                            HYPRE_BigInt  *rows,
                            HYPRE_Int     *ncols )
{
   hypre_IJMatrix *ijmatrix = (hypre_IJMatrix *) matrix;

   if (nrows == 0)
   {
      return hypre_error_flag;
   }

   if (!ijmatrix)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (nrows < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   if (!rows)
   {
      hypre_error_in_arg(3);
      return hypre_error_flag;
   }

   if (!ncols)
   {
      hypre_error_in_arg(4);
      return hypre_error_flag;
   }

   if ( hypre_IJMatrixObjectType(ijmatrix) == HYPRE_PARCSR )
   {
      hypre_IJMatrixGetRowCountsParCSR( ijmatrix, nrows, rows, ncols );
   }
   else
   {
      hypre_error_in_arg(1);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_IJMatrixGetValues( HYPRE_IJMatrix matrix,
                         HYPRE_Int      nrows,
                         HYPRE_Int     *ncols,
                         HYPRE_BigInt  *rows,
                         HYPRE_BigInt  *cols,
                         HYPRE_Complex *values )
{
   hypre_IJMatrix *ijmatrix = (hypre_IJMatrix *) matrix;

   if (nrows == 0)
   {
      return hypre_error_flag;
   }

   if (!ijmatrix)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (!ncols)
   {
      hypre_error_in_arg(3);
      return hypre_error_flag;
   }

   if (!rows)
   {
      hypre_error_in_arg(4);
      return hypre_error_flag;
   }

   if (!cols)
   {
      hypre_error_in_arg(5);
      return hypre_error_flag;
   }

   if (!values)
   {
      hypre_error_in_arg(6);
      return hypre_error_flag;
   }

   if ( hypre_IJMatrixObjectType(ijmatrix) == HYPRE_PARCSR )
   {
      hypre_IJMatrixGetValuesParCSR( ijmatrix, nrows, ncols,
                                     rows, cols, values );
   }
   else
   {
      hypre_error_in_arg(1);
   }

   return hypre_error_flag;

}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_IJMatrixSetObjectType( HYPRE_IJMatrix matrix,
                             HYPRE_Int      type )
{
   hypre_IJMatrix *ijmatrix = (hypre_IJMatrix *) matrix;

   if (!ijmatrix)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   hypre_IJMatrixObjectType(ijmatrix) = type;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_IJMatrixGetObjectType( HYPRE_IJMatrix  matrix,
                             HYPRE_Int      *type )
{
   hypre_IJMatrix *ijmatrix = (hypre_IJMatrix *) matrix;

   if (!ijmatrix)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *type = hypre_IJMatrixObjectType(ijmatrix);
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_IJMatrixGetLocalRange( HYPRE_IJMatrix  matrix,
                             HYPRE_BigInt   *ilower,
                             HYPRE_BigInt   *iupper,
                             HYPRE_BigInt   *jlower,
                             HYPRE_BigInt   *jupper )
{
   hypre_IJMatrix  *ijmatrix = (hypre_IJMatrix *) matrix;
   HYPRE_BigInt    *row_partitioning;
   HYPRE_BigInt    *col_partitioning;

   if (!ijmatrix)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   row_partitioning = hypre_IJMatrixRowPartitioning(ijmatrix);
   col_partitioning = hypre_IJMatrixColPartitioning(ijmatrix);

   *ilower = row_partitioning[0];
   *iupper = row_partitioning[1] - 1;
   *jlower = col_partitioning[0];
   *jupper = col_partitioning[1] - 1;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
   Returns a pointer to an underlying ijmatrix type used to implement IJMatrix.
   Assumes that the implementation has an underlying matrix, so it would not
   work with a direct implementation of IJMatrix.

   @return integer error code
   @param IJMatrix [IN]
   The ijmatrix to be pointed to.
*/

HYPRE_Int
HYPRE_IJMatrixGetObject( HYPRE_IJMatrix   matrix,
                         void           **object )
{
   hypre_IJMatrix *ijmatrix = (hypre_IJMatrix *) matrix;

   if (!ijmatrix)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *object = hypre_IJMatrixObject( ijmatrix );

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_IJMatrixSetRowSizes( HYPRE_IJMatrix   matrix,
                           const HYPRE_Int *sizes )
{
   hypre_IJMatrix *ijmatrix = (hypre_IJMatrix *) matrix;

   if (!ijmatrix)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if ( hypre_IJMatrixObjectType(ijmatrix) == HYPRE_PARCSR )
   {
      return ( hypre_IJMatrixSetRowSizesParCSR( ijmatrix, sizes ) );
   }
   else
   {
      hypre_error_in_arg(1);
   }

   return hypre_error_flag;
}


/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_IJMatrixSetDiagOffdSizes( HYPRE_IJMatrix   matrix,
                                const HYPRE_Int *diag_sizes,
                                const HYPRE_Int *offdiag_sizes )
{
   hypre_IJMatrix *ijmatrix = (hypre_IJMatrix *) matrix;

   if (!ijmatrix)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if ( hypre_IJMatrixObjectType(ijmatrix) == HYPRE_PARCSR )
   {
      hypre_IJMatrixSetDiagOffdSizesParCSR( ijmatrix, diag_sizes, offdiag_sizes );
   }
   else
   {
      hypre_error_in_arg(1);
   }
   return hypre_error_flag;

}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_IJMatrixSetMaxOffProcElmts( HYPRE_IJMatrix matrix,
                                  HYPRE_Int      max_off_proc_elmts)
{
   hypre_IJMatrix *ijmatrix = (hypre_IJMatrix *) matrix;

   if (!ijmatrix)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if ( hypre_IJMatrixObjectType(ijmatrix) == HYPRE_PARCSR )
   {
      return ( hypre_IJMatrixSetMaxOffProcElmtsParCSR(ijmatrix,
                                                      max_off_proc_elmts) );
   }
   else
   {
      hypre_error_in_arg(1);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixRead
 *
 * Reads data from file in ASCII format and creates an IJMatrix on host memory
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_IJMatrixRead( const char     *filename,
                    MPI_Comm        comm,
                    HYPRE_Int       type,
                    HYPRE_IJMatrix *matrix_ptr )
{
   hypre_IJMatrixRead(filename, comm, type, matrix_ptr, 0);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixReadBinary
 *
 * Reads data from file in binary format and creates an IJMatrix
 * on host memory.
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_IJMatrixReadBinary( const char     *filename,
                          MPI_Comm        comm,
                          HYPRE_Int       type,
                          HYPRE_IJMatrix *matrix_ptr )
{
   hypre_IJMatrixReadBinary(filename, comm, type, matrix_ptr);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixReadMM
 *
 * Reads matrix-market data from file in ASCII format and creates an
 * IJMatrix on host memory.
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_IJMatrixReadMM( const char     *filename,
                      MPI_Comm        comm,
                      HYPRE_Int       type,
                      HYPRE_IJMatrix *matrix_ptr )
{
   hypre_IJMatrixRead(filename, comm, type, matrix_ptr, 1);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixPrint
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_IJMatrixPrint( HYPRE_IJMatrix  matrix,
                     const char     *filename )
{
   if (!matrix)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if ( (hypre_IJMatrixObjectType(matrix) != HYPRE_PARCSR) )
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   void *object;
   HYPRE_IJMatrixGetObject(matrix, &object);
   hypre_ParCSRMatrix *par_csr = (hypre_ParCSRMatrix*) object;

   hypre_ParCSRMatrixPrintIJ(par_csr, 0, 0, filename);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixPrintBinary
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_IJMatrixPrintBinary( HYPRE_IJMatrix  matrix,
                           const char     *filename )
{
   void    *object;

   if (!matrix)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if ( (hypre_IJMatrixObjectType(matrix) != HYPRE_PARCSR) )
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   HYPRE_IJMatrixGetObject(matrix, &object);
   hypre_ParCSRMatrixPrintBinaryIJ((hypre_ParCSRMatrix*) object, 0, 0, filename);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixSetOMPFlag
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_IJMatrixSetOMPFlag( HYPRE_IJMatrix matrix,
                          HYPRE_Int      omp_flag )
{
   hypre_IJMatrix *ijmatrix = (hypre_IJMatrix *) matrix;

   if (!ijmatrix)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   hypre_IJMatrixOMPFlag(ijmatrix) = omp_flag;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixTranspose
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_IJMatrixTranspose( HYPRE_IJMatrix  matrix_A,
                         HYPRE_IJMatrix *matrix_AT )
{
   hypre_IJMatrix   *ij_A = (hypre_IJMatrix *) matrix_A;
   hypre_IJMatrix   *ij_AT;
   HYPRE_Int         i;

   if (!ij_A)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   ij_AT = hypre_CTAlloc(hypre_IJMatrix, 1, HYPRE_MEMORY_HOST);

   hypre_IJMatrixComm(ij_AT)           = hypre_IJMatrixComm(ij_A);
   hypre_IJMatrixObject(ij_AT)         = NULL;
   hypre_IJMatrixTranslator(ij_AT)     = NULL;
   hypre_IJMatrixAssumedPart(ij_AT)    = NULL;
   hypre_IJMatrixObjectType(ij_AT)     = hypre_IJMatrixObjectType(ij_A);
   hypre_IJMatrixAssembleFlag(ij_AT)   = 1;
   hypre_IJMatrixPrintLevel(ij_AT)     = hypre_IJMatrixPrintLevel(ij_A);
   hypre_IJMatrixGlobalFirstRow(ij_AT) = hypre_IJMatrixGlobalFirstCol(ij_A);
   hypre_IJMatrixGlobalFirstCol(ij_AT) = hypre_IJMatrixGlobalFirstRow(ij_A);
   hypre_IJMatrixGlobalNumRows(ij_AT)  = hypre_IJMatrixGlobalNumCols(ij_A);
   hypre_IJMatrixGlobalNumCols(ij_AT)  = hypre_IJMatrixGlobalNumRows(ij_A);

   for (i = 0; i < 2; i++)
   {
      hypre_IJMatrixRowPartitioning(ij_AT)[i] = hypre_IJMatrixColPartitioning(ij_A)[i];
      hypre_IJMatrixColPartitioning(ij_AT)[i] = hypre_IJMatrixRowPartitioning(ij_A)[i];
   }

   if (hypre_IJMatrixObjectType(ij_A) == HYPRE_PARCSR)
   {
      hypre_IJMatrixTransposeParCSR(ij_A, ij_AT);
   }
   else
   {
      hypre_error_in_arg(1);
   }

   *matrix_AT = (HYPRE_IJMatrix) ij_AT;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixNorm
 *
 *  TODO: Add other norms
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_IJMatrixNorm( HYPRE_IJMatrix  matrix,
                    HYPRE_Real     *norm )
{
   hypre_IJMatrix *ijmatrix = (hypre_IJMatrix *) matrix;

   if (!ijmatrix)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (hypre_IJMatrixObjectType(ijmatrix) == HYPRE_PARCSR)
   {
      hypre_IJMatrixNormParCSR(ijmatrix, norm);
   }
   else
   {
      hypre_error_in_arg(1);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixAdd
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_IJMatrixAdd( HYPRE_Complex    alpha,
                   HYPRE_IJMatrix   matrix_A,
                   HYPRE_Complex    beta,
                   HYPRE_IJMatrix   matrix_B,
                   HYPRE_IJMatrix  *matrix_C )
{
   hypre_IJMatrix   *ij_A = (hypre_IJMatrix *) matrix_A;
   hypre_IJMatrix   *ij_B = (hypre_IJMatrix *) matrix_B;
   hypre_IJMatrix   *ij_C;

   HYPRE_BigInt     *row_partitioning_A;
   HYPRE_BigInt     *col_partitioning_A;
   HYPRE_BigInt     *row_partitioning_B;
   HYPRE_BigInt     *col_partitioning_B;
   HYPRE_Int         i;

   if (!ij_A)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   /* Check if A and B have the same row/col partitionings */
   row_partitioning_A = hypre_IJMatrixRowPartitioning(ij_A);
   row_partitioning_B = hypre_IJMatrixRowPartitioning(ij_B);
   col_partitioning_A = hypre_IJMatrixColPartitioning(ij_A);
   col_partitioning_B = hypre_IJMatrixColPartitioning(ij_B);
   for (i = 0; i < 2; i++)
   {
      if (row_partitioning_A[i] != row_partitioning_B[i])
      {
         hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                           "Input matrices must have same row partitioning!");
         return hypre_error_flag;
      }

      if (col_partitioning_A[i] != col_partitioning_B[i])
      {
         hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                           "Input matrices must have same col partitioning!");
         return hypre_error_flag;
      }
   }

   ij_C = hypre_CTAlloc(hypre_IJMatrix, 1, HYPRE_MEMORY_HOST);

   hypre_IJMatrixComm(ij_C)            = hypre_IJMatrixComm(ij_A);
   hypre_IJMatrixObject(ij_C)          = NULL;
   hypre_IJMatrixTranslator(ij_C)      = NULL;
   hypre_IJMatrixAssumedPart(ij_C)     = NULL;
   hypre_IJMatrixObjectType(ij_C)      = hypre_IJMatrixObjectType(ij_A);
   hypre_IJMatrixAssembleFlag(ij_C)    = 1;
   hypre_IJMatrixPrintLevel(ij_C)      = hypre_IJMatrixPrintLevel(ij_A);

   /* Copy row/col partitioning of A to C */
   for (i = 0; i < 2; i++)
   {
      hypre_IJMatrixRowPartitioning(ij_C)[i] = row_partitioning_A[i];
      hypre_IJMatrixColPartitioning(ij_C)[i] = col_partitioning_A[i];
   }

   if (hypre_IJMatrixObjectType(ij_A) == HYPRE_PARCSR)
   {
      hypre_IJMatrixAddParCSR(alpha, ij_A, beta, ij_B, ij_C);
   }
   else
   {
      hypre_error_in_arg(1);
   }

   *matrix_C = (HYPRE_IJMatrix) ij_C;

   return hypre_error_flag;
}
