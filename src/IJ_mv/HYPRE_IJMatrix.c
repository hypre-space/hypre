/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
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
   HYPRE_BigInt *row_partitioning;
   HYPRE_BigInt *col_partitioning;
   HYPRE_BigInt *info;
   HYPRE_Int num_procs;
   HYPRE_Int myid;

   hypre_IJMatrix *ijmatrix;

   HYPRE_BigInt  row0, col0, rowN, colN;

   ijmatrix = hypre_CTAlloc(hypre_IJMatrix,  1, HYPRE_MEMORY_HOST);

   hypre_IJMatrixComm(ijmatrix)           = comm;
   hypre_IJMatrixObject(ijmatrix)         = NULL;
   hypre_IJMatrixTranslator(ijmatrix)     = NULL;
   hypre_IJMatrixAssumedPart(ijmatrix)    = NULL;
   hypre_IJMatrixObjectType(ijmatrix)     = HYPRE_UNITIALIZED;
   hypre_IJMatrixAssembleFlag(ijmatrix)   = 0;
   hypre_IJMatrixPrintLevel(ijmatrix)     = 0;
   hypre_IJMatrixOMPFlag(ijmatrix)        = 0;

   hypre_MPI_Comm_size(comm,&num_procs);
   hypre_MPI_Comm_rank(comm, &myid);


   if (ilower > iupper+1 || ilower < 0)
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

   if (jlower > jupper+1 || jlower < 0)
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

   info = hypre_CTAlloc(HYPRE_BigInt, 2, HYPRE_MEMORY_HOST);

   row_partitioning = hypre_CTAlloc(HYPRE_BigInt, 2, HYPRE_MEMORY_HOST);
   col_partitioning = hypre_CTAlloc(HYPRE_BigInt, 2, HYPRE_MEMORY_HOST);

   row_partitioning[0] = ilower;
   row_partitioning[1] = iupper+1;
   col_partitioning[0] = jlower;
   col_partitioning[1] = jupper+1;

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
   if (myid == (num_procs-1))
   {
      info[0] = iupper;
      info[1] = jupper;
   }
   hypre_MPI_Bcast(info, 2, HYPRE_MPI_BIG_INT, num_procs-1, comm);

   rowN = info[0];
   colN = info[1];

   hypre_IJMatrixGlobalFirstRow(ijmatrix) = row0;
   hypre_IJMatrixGlobalFirstCol(ijmatrix) = col0;
   hypre_IJMatrixGlobalNumRows(ijmatrix) = rowN - row0 + 1;
   hypre_IJMatrixGlobalNumCols(ijmatrix) = colN - col0 + 1;

   hypre_TFree(info, HYPRE_MEMORY_HOST);

   hypre_IJMatrixRowPartitioning(ijmatrix) = row_partitioning;
   hypre_IJMatrixColPartitioning(ijmatrix) = col_partitioning;

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
      if (hypre_IJMatrixRowPartitioning(ijmatrix) ==
          hypre_IJMatrixColPartitioning(ijmatrix))
      {
         hypre_TFree(hypre_IJMatrixRowPartitioning(ijmatrix), HYPRE_MEMORY_HOST);
      }
      else
      {
         hypre_TFree(hypre_IJMatrixRowPartitioning(ijmatrix), HYPRE_MEMORY_HOST);
         hypre_TFree(hypre_IJMatrixColPartitioning(ijmatrix), HYPRE_MEMORY_HOST);
      }
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

   hypre_IJMatrixPrintLevel(ijmatrix) = 1;
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
      for (j=1; j < nvals; j++)
         sums[j] += sums[j-1] + vals[j-1];
   }
   else
   {

      /* Compute preliminary partial sums (in parallel) within each interval */
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(j) HYPRE_SMP_SCHEDULE
#endif
      for (j = 0; j < nvals; j += bsize)
      {
         HYPRE_Int  i, n = hypre_min((j+bsize), nvals);

         sums[0] = 0;
         for (i = j+1; i < n; i++)
         {
            sums[i] = sums[i-1] + vals[i-1];
         }
      }

      /* Compute final partial sums (in serial) for the first entry of every interval */
      for (j = bsize; j < nvals; j += bsize)
      {
         sums[j] = sums[j-bsize] + sums[j-1] + vals[j-1];
      }

      /* Compute final partial sums (in parallel) for the remaining entries */
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(j) HYPRE_SMP_SCHEDULE
#endif
      for (j = bsize; j < nvals; j += bsize)
      {
         HYPRE_Int  i, n = hypre_min((j+bsize), nvals);

         for (i = j+1; i < n; i++)
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

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1( hypre_IJMatrixMemoryLocation(matrix) );

   if (exec == HYPRE_EXEC_DEVICE)
   {
      hypre_IJMatrixSetAddValuesParCSRDevice(ijmatrix, nrows, ncols, rows, row_indexes, cols, values, "set");
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
      return( hypre_IJMatrixSetConstantValuesParCSR( ijmatrix, value));
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

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1( hypre_IJMatrixMemoryLocation(matrix) );

   if (exec == HYPRE_EXEC_DEVICE)
   {
      hypre_IJMatrixSetAddValuesParCSRDevice(ijmatrix, nrows, ncols, rows, row_indexes, cols, values, "add");
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
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
      HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1( hypre_IJMatrixMemoryLocation(matrix) );

      if (exec == HYPRE_EXEC_DEVICE)
      {
         return( hypre_IJMatrixAssembleParCSRDevice( ijmatrix ) );
      }
      else
#endif
      {
         return( hypre_IJMatrixAssembleParCSR( ijmatrix ) );
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
   hypre_IJMatrix *ijmatrix = (hypre_IJMatrix *) matrix;
   MPI_Comm comm;
   HYPRE_BigInt *row_partitioning;
   HYPRE_BigInt *col_partitioning;
   HYPRE_Int my_id;

   if (!ijmatrix)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   comm = hypre_IJMatrixComm(ijmatrix);
   row_partitioning = hypre_IJMatrixRowPartitioning(ijmatrix);
   col_partitioning = hypre_IJMatrixColPartitioning(ijmatrix);

   hypre_MPI_Comm_rank(comm, &my_id);

   *ilower = row_partitioning[0];
   *iupper = row_partitioning[1]-1;
   *jlower = col_partitioning[0];
   *jupper = col_partitioning[1]-1;

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
      return( hypre_IJMatrixSetRowSizesParCSR( ijmatrix , sizes ) );
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
      return( hypre_IJMatrixSetMaxOffProcElmtsParCSR(ijmatrix,
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
 * create IJMatrix on host memory
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_IJMatrixRead( const char     *filename,
                    MPI_Comm        comm,
                    HYPRE_Int       type,
                    HYPRE_IJMatrix *matrix_ptr )
{
   HYPRE_IJMatrix  matrix;
   HYPRE_BigInt    ilower, iupper, jlower, jupper;
   HYPRE_BigInt    I, J;
   HYPRE_Int       ncols;
   HYPRE_Complex   value;
   HYPRE_Int       myid, ret;
   char            new_filename[255];
   FILE           *file;

   hypre_MPI_Comm_rank(comm, &myid);

   hypre_sprintf(new_filename,"%s.%05d", filename, myid);

   if ((file = fopen(new_filename, "r")) == NULL)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   hypre_fscanf(file, "%b %b %b %b", &ilower, &iupper, &jlower, &jupper);
   HYPRE_IJMatrixCreate(comm, ilower, iupper, jlower, jupper, &matrix);

   HYPRE_IJMatrixSetObjectType(matrix, type);

   HYPRE_IJMatrixInitialize_v2(matrix, HYPRE_MEMORY_HOST);

   /* It is important to ensure that whitespace follows the index value to help
    * catch mistakes in the input file.  See comments in IJVectorRead(). */
   ncols = 1;
   while ( (ret = hypre_fscanf(file, "%b %b%*[ \t]%le", &I, &J, &value)) != EOF )
   {
      if (ret != 3)
      {
         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Error in IJ matrix input file.");
         return hypre_error_flag;
      }
      if (I < ilower || I > iupper)
      {
         HYPRE_IJMatrixAddToValues(matrix, 1, &ncols, &I, &J, &value);
      }
      else
      {
         HYPRE_IJMatrixSetValues(matrix, 1, &ncols, &I, &J, &value);
      }
   }

   HYPRE_IJMatrixAssemble(matrix);

   fclose(file);

   *matrix_ptr = matrix;

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
   HYPRE_ParCSRMatrix par_csr = (HYPRE_ParCSRMatrix) object;

   HYPRE_MemoryLocation memory_location = hypre_IJMatrixMemoryLocation(matrix);

   if ( hypre_GetActualMemLocation(memory_location) == hypre_MEMORY_HOST )
   {
      hypre_ParCSRMatrixPrintIJ(par_csr, 0, 0, filename);
   }
   else
   {
      HYPRE_ParCSRMatrix par_csr2 = hypre_ParCSRMatrixClone_v2(par_csr, 1, HYPRE_MEMORY_HOST);
      hypre_ParCSRMatrixPrintIJ(par_csr2, 0, 0, filename);
      hypre_ParCSRMatrixDestroy(par_csr2);
   }

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

