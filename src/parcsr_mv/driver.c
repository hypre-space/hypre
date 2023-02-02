/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_mv.h"

/*--------------------------------------------------------------------------
 * Test driver for unstructured matrix interface
 *--------------------------------------------------------------------------*/

HYPRE_Int
main( HYPRE_Int   argc,
      char *argv[] )
{
   hypre_ParVector   *vector1;
   hypre_ParVector   *vector2;
   hypre_ParVector   *tmp_vector;

   HYPRE_Int          num_procs, my_id;
   HYPRE_BigInt   global_size = 20;
   HYPRE_Int      local_size;
   HYPRE_BigInt      first_index;
   HYPRE_Int      i;
   HYPRE_BigInt   *partitioning;
   HYPRE_Complex  prod;
   HYPRE_Complex  *data, *data2;
   hypre_Vector *vector;
   hypre_Vector *local_vector;
   hypre_Vector *local_vector2;

   /* Initialize MPI */
   hypre_MPI_Init(&argc, &argv);

   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs );
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &my_id );

   hypre_printf(" my_id: %d num_procs: %d\n", my_id, num_procs);

   partitioning = NULL;
   vector1 = hypre_ParVectorCreate(hypre_MPI_COMM_WORLD, global_size, partitioning);
   partitioning = hypre_ParVectorPartitioning(vector1);
   hypre_ParVectorInitialize(vector1);
   local_vector = hypre_ParVectorLocalVector(vector1);
   data = hypre_VectorData(local_vector);
   local_size = hypre_VectorSize(local_vector);
   first_index = partitioning[my_id];

   for (i = 0; i < local_size; i++)
   {
      data[i] = first_index + i;
   }
   /*
      hypre_ParVectorPrint(vector1, "Vector");
   */
   local_vector2 = hypre_SeqVectorCreate(global_size);
   hypre_SeqVectorInitialize(local_vector2);
   data2 = hypre_VectorData(local_vector2);
   for (i = 0; i < global_size; i++)
   {
      data2[i] = i + 1;
   }

   /*   partitioning = hypre_CTAlloc(HYPRE_Int,4);
      partitioning[0] = 0;
      partitioning[1] = 10;
      partitioning[2] = 10;
      partitioning[3] = 20;
   */
   vector2 = hypre_VectorToParVector(hypre_MPI_COMM_WORLD, local_vector2, partitioning);

   hypre_ParVectorPrint(vector2, "Convert");

   vector = hypre_ParVectorToVectorAll(vector2);

   /*-----------------------------------------------------------
    * Copy the vector into tmp_vector
    *-----------------------------------------------------------*/

   tmp_vector = hypre_ParVectorRead(hypre_MPI_COMM_WORLD, "Convert");
   /*
      tmp_vector = hypre_ParVectorCreate(hypre_MPI_COMM_WORLD,global_size,partitioning);
      hypre_ParVectorInitialize(tmp_vector);
      hypre_ParVectorCopy(vector1, tmp_vector);

      hypre_ParVectorPrint(tmp_vector,"Copy");
   */
   /*-----------------------------------------------------------
    * Scale tmp_vector
    *-----------------------------------------------------------*/

   hypre_ParVectorScale(2.0, tmp_vector);
   /*
      hypre_ParVectorPrint(tmp_vector,"Scale");
   */
   /*-----------------------------------------------------------
    * Do an Axpy (2*vector - vector) = vector
    *-----------------------------------------------------------*/

   hypre_ParVectorAxpy(-1.0, vector1, tmp_vector);
   /*
      hypre_ParVectorPrint(tmp_vector,"Axpy");
   */
   /*-----------------------------------------------------------
    * Do an inner product vector* tmp_vector
    *-----------------------------------------------------------*/

   prod = hypre_ParVectorInnerProd(vector1, tmp_vector);

   hypre_printf (" prod: %8.2f \n", prod);

   /*-----------------------------------------------------------
    * Finalize things
    *-----------------------------------------------------------*/

   hypre_ParVectorDestroy(vector1);
   hypre_ParVectorDestroy(vector2);
   hypre_ParVectorDestroy(tmp_vector);
   hypre_SeqVectorDestroy(local_vector2);
   if (vector) { hypre_SeqVectorDestroy(vector); }

   /* Finalize MPI */
   hypre_MPI_Finalize();

   return 0;
}
