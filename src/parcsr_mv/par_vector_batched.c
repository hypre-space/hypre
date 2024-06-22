/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Member functions for hypre_Vector class.
 *
 *****************************************************************************/

#include "_hypre_parcsr_mv.h"

/*--------------------------------------------------------------------------
 * hypre_ParVectorMassAxpy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParVectorMassAxpy( HYPRE_Complex    *alpha,
                         hypre_ParVector **x,
                         hypre_ParVector  *y,
                         HYPRE_Int         k,
                         HYPRE_Int         unroll )
{
   HYPRE_Int i;
   hypre_Vector **x_local;
   hypre_Vector *y_local = hypre_ParVectorLocalVector(y);
   x_local = hypre_TAlloc(hypre_Vector *, k, HYPRE_MEMORY_HOST);

   for (i = 0; i < k; i++)
   {
      x_local[i] = hypre_ParVectorLocalVector(x[i]);
   }

   hypre_SeqVectorMassAxpy( alpha, x_local, y_local, k, unroll);

   hypre_TFree(x_local, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParVectorMassInnerProd
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParVectorMassInnerProd( hypre_ParVector  *x,
                              hypre_ParVector **y,
                              HYPRE_Int         k,
                              HYPRE_Int         unroll,
                              HYPRE_Real       *result )
{
   MPI_Comm      comm    = hypre_ParVectorComm(x);
   hypre_Vector *x_local = hypre_ParVectorLocalVector(x);
   HYPRE_Real *local_result;
   HYPRE_Int i;
   hypre_Vector **y_local;
   y_local = hypre_TAlloc(hypre_Vector *, k, HYPRE_MEMORY_HOST);

   for (i = 0; i < k; i++)
   {
      y_local[i] = (hypre_Vector *) hypre_ParVectorLocalVector(y[i]);
   }

   local_result = hypre_CTAlloc(HYPRE_Real, k, HYPRE_MEMORY_HOST);

   hypre_SeqVectorMassInnerProd(x_local, y_local, k, unroll, local_result);

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_ALL_REDUCE] -= hypre_MPI_Wtime();
#endif
   hypre_MPI_Allreduce(local_result, result, k, HYPRE_MPI_REAL,
                       hypre_MPI_SUM, comm);
#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_ALL_REDUCE] += hypre_MPI_Wtime();
#endif

   hypre_TFree(y_local, HYPRE_MEMORY_HOST);
   hypre_TFree(local_result, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParVectorMassDotpTwo
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParVectorMassDotpTwo ( hypre_ParVector  *x,
                             hypre_ParVector  *y,
                             hypre_ParVector **z,
                             HYPRE_Int         k,
                             HYPRE_Int         unroll,
                             HYPRE_Real       *result_x,
                             HYPRE_Real       *result_y )
{
   MPI_Comm      comm    = hypre_ParVectorComm(x);
   hypre_Vector *x_local = hypre_ParVectorLocalVector(x);
   hypre_Vector *y_local = hypre_ParVectorLocalVector(y);
   HYPRE_Real *local_result, *result;
   HYPRE_Int i;
   hypre_Vector **z_local;
   z_local = hypre_TAlloc(hypre_Vector*, k, HYPRE_MEMORY_HOST);

   for (i = 0; i < k; i++)
   {
      z_local[i] = (hypre_Vector *) hypre_ParVectorLocalVector(z[i]);
   }

   local_result = hypre_CTAlloc(HYPRE_Real, 2 * k, HYPRE_MEMORY_HOST);
   result = hypre_CTAlloc(HYPRE_Real, 2 * k, HYPRE_MEMORY_HOST);

   hypre_SeqVectorMassDotpTwo(x_local, y_local, z_local, k, unroll, &local_result[0],
                              &local_result[k]);

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_ALL_REDUCE] -= hypre_MPI_Wtime();
#endif
   hypre_MPI_Allreduce(local_result, result, 2 * k, HYPRE_MPI_REAL,
                       hypre_MPI_SUM, comm);
#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_ALL_REDUCE] += hypre_MPI_Wtime();
#endif

   for (i = 0; i < k; i++)
   {
      result_x[i] = result[i];
      result_y[i] = result[k + i];
   }
   hypre_TFree(z_local, HYPRE_MEMORY_HOST);
   hypre_TFree(local_result, HYPRE_MEMORY_HOST);
   hypre_TFree(result, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}

