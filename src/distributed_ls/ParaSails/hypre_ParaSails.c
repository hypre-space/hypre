/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * hypre_ParaSails
 *
 *****************************************************************************/

#include "Common.h"
#include "HYPRE_distributed_matrix_types.h"
#include "HYPRE_distributed_matrix_protos.h"
#include "hypre_ParaSails.h"
#include "Matrix.h"
#include "ParaSails.h"

/* these includes required for hypre_ParaSailsIJMatrix */
#include "../../IJ_mv/HYPRE_IJ_mv.h"
#include "../../HYPRE.h"
#include "../../utilities/_hypre_utilities.h"

typedef struct
{
   MPI_Comm   comm;
   ParaSails *ps;
}
   hypre_ParaSails_struct;

/*--------------------------------------------------------------------------
 * balance_info - Dump out information about the partitioning of the 
 * matrix, which affects load balance
 *--------------------------------------------------------------------------*/

#ifdef BALANCE_INFO
static void balance_info(MPI_Comm comm, Matrix *mat)
{
   HYPRE_Int mype, num_local, i, total;

   hypre_MPI_Comm_rank(comm, &mype);
   num_local = mat->end_row - mat->beg_row + 1;

   /* compute number of nonzeros on local matrix */
   total = 0;
   for (i=0; i<num_local; i++)
      total += mat->lens[i];

   /* each processor prints out its own info */
   hypre_printf("%4d: nrows %d, nnz %d, send %d (%d), recv %d (%d)\n",
                mype, num_local, total, mat->num_send, mat->sendlen,
                mat->num_recv, mat->recvlen);
}

static void matvec_timing(MPI_Comm comm, Matrix *mat)
{
   HYPRE_Real time0, time1;
   HYPRE_Real trial1, trial2, trial3, trial4, trial5, trial6;
   HYPRE_Real *temp1, *temp2;
   HYPRE_Int i, mype;
   HYPRE_Int n = mat->end_row - mat->beg_row + 1;

   temp1 = hypre_CTAlloc(HYPRE_Real, n, HYPRE_MEMORY_HOST);
   temp2 = hypre_CTAlloc(HYPRE_Real, n, HYPRE_MEMORY_HOST);

   /* warm-up */
   hypre_MPI_Barrier(comm);
   for (i=0; i<100; i++)
      MatrixMatvec(mat, temp1, temp2);

   hypre_MPI_Barrier(comm);
   time0 = hypre_MPI_Wtime();
   for (i=0; i<100; i++)
      MatrixMatvec(mat, temp1, temp2);
   hypre_MPI_Barrier(comm);
   time1 = hypre_MPI_Wtime();
   trial1 = time1-time0;

   hypre_MPI_Barrier(comm);
   time0 = hypre_MPI_Wtime();
   for (i=0; i<100; i++)
      MatrixMatvec(mat, temp1, temp2);
   hypre_MPI_Barrier(comm);
   time1 = hypre_MPI_Wtime();
   trial2 = time1-time0;

   hypre_MPI_Barrier(comm);
   time0 = hypre_MPI_Wtime();
   for (i=0; i<100; i++)
      MatrixMatvec(mat, temp1, temp2);
   hypre_MPI_Barrier(comm);
   time1 = hypre_MPI_Wtime();
   trial3 = time1-time0;

   hypre_MPI_Barrier(comm);
   time0 = hypre_MPI_Wtime();
   for (i=0; i<100; i++)
      MatrixMatvecSerial(mat, temp1, temp2);
   hypre_MPI_Barrier(comm);
   time1 = hypre_MPI_Wtime();
   trial4 = time1-time0;

   hypre_MPI_Barrier(comm);
   time0 = hypre_MPI_Wtime();
   for (i=0; i<100; i++)
      MatrixMatvecSerial(mat, temp1, temp2);
   hypre_MPI_Barrier(comm);
   time1 = hypre_MPI_Wtime();
   trial5 = time1-time0;

   hypre_MPI_Barrier(comm);
   time0 = hypre_MPI_Wtime();
   for (i=0; i<100; i++)
      MatrixMatvecSerial(mat, temp1, temp2);
   hypre_MPI_Barrier(comm);
   time1 = hypre_MPI_Wtime();
   trial6 = time1-time0;

   hypre_MPI_Comm_rank(comm, &mype);
   if (mype == 0)
      hypre_printf("Timings: %f %f %f Serial: %f %f %f\n", 
                   trial1, trial2, trial3, trial4, trial5, trial6);

   fflush(stdout);

   /* this is all we wanted, so don't waste any more cycles */
   exit(0);
}
#endif

/*--------------------------------------------------------------------------
 * convert_matrix - Create and convert distributed matrix to native 
 * data structure of ParaSails
 *--------------------------------------------------------------------------*/

static Matrix *convert_matrix(MPI_Comm comm, HYPRE_DistributedMatrix distmat)
{
   HYPRE_Int beg_row, end_row, row, dummy;
   HYPRE_Int len, *ind;
   HYPRE_Real *val;
   Matrix *mat;

   HYPRE_DistributedMatrixGetLocalRange(distmat, &beg_row, &end_row,
                                        &dummy, &dummy);

   mat = MatrixCreate(comm, beg_row, end_row);

   for (row=beg_row; row<=end_row; row++)
   {
      HYPRE_DistributedMatrixGetRow(distmat, row, &len, &ind, &val);
      MatrixSetRow(mat, row, len, ind, val);
      HYPRE_DistributedMatrixRestoreRow(distmat, row, &len, &ind, &val);
   }

   MatrixComplete(mat);

#ifdef BALANCE_INFO
   matvec_timing(comm, mat);
   balance_info(comm, mat);
#endif

   return mat;
}

/*--------------------------------------------------------------------------
 * hypre_ParaSailsCreate - Return a ParaSails preconditioner object "obj"
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_ParaSailsCreate(MPI_Comm comm, hypre_ParaSails *obj)
{
   hypre_ParaSails_struct *internal;

   internal = (hypre_ParaSails_struct *)
      hypre_CTAlloc(hypre_ParaSails_struct,  1, HYPRE_MEMORY_HOST);

   internal->comm = comm;
   internal->ps   = NULL;

   *obj = (hypre_ParaSails) internal;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParaSailsDestroy - Destroy a ParaSails object "ps".
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_ParaSailsDestroy(hypre_ParaSails obj)
{
   hypre_ParaSails_struct *internal = (hypre_ParaSails_struct *) obj;

   ParaSailsDestroy(internal->ps);

   hypre_TFree(internal, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParaSailsSetup - This function should be used if the preconditioner
 * pattern and values are set up with the same distributed matrix.
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_ParaSailsSetup(hypre_ParaSails obj,
                               HYPRE_DistributedMatrix distmat, HYPRE_Int sym, HYPRE_Real thresh, HYPRE_Int nlevels,
                               HYPRE_Real filter, HYPRE_Real loadbal, HYPRE_Int logging)
{
   /* HYPRE_Real cost; */
   Matrix *mat;
   hypre_ParaSails_struct *internal = (hypre_ParaSails_struct *) obj;
   HYPRE_Int err;

   mat = convert_matrix(internal->comm, distmat);

   ParaSailsDestroy(internal->ps);

   internal->ps = ParaSailsCreate(internal->comm, 
                                  mat->beg_row, mat->end_row, sym);

   ParaSailsSetupPattern(internal->ps, mat, thresh, nlevels);

   if (logging)
      /* cost = */ ParaSailsStatsPattern(internal->ps, mat);

   internal->ps->loadbal_beta = loadbal;

   err = ParaSailsSetupValues(internal->ps, mat, filter);

   if (logging)
      ParaSailsStatsValues(internal->ps, mat);

   MatrixDestroy(mat);

   if (err)
   {
      hypre_error(HYPRE_ERROR_GENERIC);
   }
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParaSailsSetupPattern - Set up pattern using a distributed matrix.
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_ParaSailsSetupPattern(hypre_ParaSails obj,
                                      HYPRE_DistributedMatrix distmat, HYPRE_Int sym, HYPRE_Real thresh, HYPRE_Int nlevels,
                                      HYPRE_Int logging)
{
   /* HYPRE_Real cost; */
   Matrix *mat;
   hypre_ParaSails_struct *internal = (hypre_ParaSails_struct *) obj;

   mat = convert_matrix(internal->comm, distmat);

   ParaSailsDestroy(internal->ps);

   internal->ps = ParaSailsCreate(internal->comm, 
                                  mat->beg_row, mat->end_row, sym);

   ParaSailsSetupPattern(internal->ps, mat, thresh, nlevels);

   if (logging)
      /* cost = */ ParaSailsStatsPattern(internal->ps, mat);

   MatrixDestroy(mat);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParaSailsSetupValues - Set up values using a distributed matrix.
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_ParaSailsSetupValues(hypre_ParaSails obj,
                                     HYPRE_DistributedMatrix distmat, HYPRE_Real filter, HYPRE_Real loadbal,
                                     HYPRE_Int logging)
{
   Matrix *mat;
   hypre_ParaSails_struct *internal = (hypre_ParaSails_struct *) obj;
   HYPRE_Int err;

   mat = convert_matrix(internal->comm, distmat);

   internal->ps->loadbal_beta = loadbal;
   internal->ps->setup_pattern_time = 0.0;

   err = ParaSailsSetupValues(internal->ps, mat, filter);

   if (logging)
      ParaSailsStatsValues(internal->ps, mat);

   MatrixDestroy(mat);

   if (err)
   {
      hypre_error(HYPRE_ERROR_GENERIC);
   }
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParaSailsApply - Apply the ParaSails preconditioner to an array 
 * "u", and return the result in the array "v".
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_ParaSailsApply(hypre_ParaSails obj, HYPRE_Real *u, HYPRE_Real *v)
{
   hypre_ParaSails_struct *internal = (hypre_ParaSails_struct *) obj;

   ParaSailsApply(internal->ps, u, v);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParaSailsApplyTrans - Apply the ParaSails preconditioner, transposed
 * to an array "u", and return the result in the array "v".
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_ParaSailsApplyTrans(hypre_ParaSails obj, HYPRE_Real *u, HYPRE_Real *v)
{
   hypre_ParaSails_struct *internal = (hypre_ParaSails_struct *) obj;

   ParaSailsApplyTrans(internal->ps, u, v);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParaSailsIJMatrix - Return the IJ matrix which is the sparse
 * approximate inverse (or its factor).  This matrix is a copy of the
 * matrix that is in ParaSails Matrix format.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParaSailsBuildIJMatrix(hypre_ParaSails obj, HYPRE_IJMatrix *pij_A)
{
   hypre_ParaSails_struct *internal = (hypre_ParaSails_struct *) obj;
   ParaSails *ps = internal->ps;
   Matrix *mat = internal->ps->M;

   HYPRE_Int *diag_sizes, *offdiag_sizes, local_row, i, j;
   HYPRE_Int size;
   HYPRE_Int *col_inds;
   HYPRE_Real *values;

   HYPRE_IJMatrixCreate( ps->comm, ps->beg_row, ps->end_row,
                         ps->beg_row, ps->end_row,
                         pij_A );

   HYPRE_IJMatrixSetObjectType( *pij_A, HYPRE_PARCSR );

   diag_sizes = hypre_CTAlloc(HYPRE_Int,  ps->end_row - ps->beg_row + 1, HYPRE_MEMORY_HOST);
   offdiag_sizes = hypre_CTAlloc(HYPRE_Int,  ps->end_row - ps->beg_row + 1, HYPRE_MEMORY_HOST);
   local_row = 0;
   for (i=ps->beg_row; i<= ps->end_row; i++)
   {
      MatrixGetRow(mat, local_row, &size, &col_inds, &values);
      NumberingLocalToGlobal(ps->numb, size, col_inds, col_inds);

      for (j=0; j < size; j++)
      {
         if (col_inds[j] < ps->beg_row || col_inds[j] > ps->end_row)
            offdiag_sizes[local_row]++;
         else
            diag_sizes[local_row]++;
      }

      local_row++;
   }
   HYPRE_IJMatrixSetDiagOffdSizes( *pij_A, (const HYPRE_Int *) diag_sizes,
                                   (const HYPRE_Int *) offdiag_sizes );
   hypre_TFree(diag_sizes, HYPRE_MEMORY_HOST);
   hypre_TFree(offdiag_sizes, HYPRE_MEMORY_HOST);

   HYPRE_IJMatrixInitialize( *pij_A );

   local_row = 0;
   for (i=ps->beg_row; i<= ps->end_row; i++)
   {
      MatrixGetRow(mat, local_row, &size, &col_inds, &values);

      HYPRE_IJMatrixSetValues( *pij_A, 1, &size, &i, (const HYPRE_Int *) col_inds,
                               (const HYPRE_Real *) values );

      NumberingGlobalToLocal(ps->numb, size, col_inds, col_inds);

      local_row++;
   }

   HYPRE_IJMatrixAssemble( *pij_A );

   return hypre_error_flag;
}
