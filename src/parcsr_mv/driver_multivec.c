/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.6 $
 ***********************************************************************EHEADER*/




#include "headers.h"
 
/*--------------------------------------------------------------------------
 * Test driver for PAR multivectors (under construction)
 *--------------------------------------------------------------------------*/
 
HYPRE_Int
main( HYPRE_Int   argc,
      char *argv[] )
{
   hypre_ParVector   *vector1;
   hypre_ParVector   *vector2;
   hypre_ParVector   *tmp_vector;

   HYPRE_Int          num_procs, my_id;
   HYPRE_Int	 	global_size = 20;
   HYPRE_Int		local_size;
   HYPRE_Int		first_index;
   HYPRE_Int          num_vectors, vecstride, idxstride;
   HYPRE_Int 		i, j;
   HYPRE_Int 		*partitioning;
   double	prod;
   double 	*data, *data2;
   hypre_Vector *vector; 
   hypre_Vector *local_vector; 
   hypre_Vector *local_vector2;
 
   /* Initialize MPI */
   hypre_MPI_Init(&argc, &argv);

   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs );
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &my_id );

   hypre_printf(" my_id: %d num_procs: %d\n", my_id, num_procs);
 
   partitioning = NULL;
   num_vectors = 3;
   vector1 = hypre_ParMultiVectorCreate
      ( hypre_MPI_COMM_WORLD, global_size, partitioning, num_vectors );
   partitioning = hypre_ParVectorPartitioning(vector1);

   hypre_ParVectorInitialize(vector1);
   local_vector = hypre_ParVectorLocalVector(vector1);
   data = hypre_VectorData(local_vector);
   local_size = hypre_VectorSize(local_vector);
   vecstride = hypre_VectorVectorStride(local_vector);
   idxstride = hypre_VectorIndexStride(local_vector);
   first_index = partitioning[my_id];

   hypre_printf("vecstride=%i idxstride=%i local_size=%i num_vectors=%i",
          vecstride, idxstride, local_size, num_vectors );
   for (j=0; j<num_vectors; ++j )
      for (i=0; i < local_size; i++)
         data[ j*vecstride + i*idxstride ] = first_index+i + 100*j;

   hypre_ParVectorPrint(vector1, "Vector");

   local_vector2 = hypre_SeqMultiVectorCreate( global_size, num_vectors );
   hypre_SeqVectorInitialize(local_vector2);
   data2 = hypre_VectorData(local_vector2);
   vecstride = hypre_VectorVectorStride(local_vector2);
   idxstride = hypre_VectorIndexStride(local_vector2);
   for (j=0; j<num_vectors; ++j )
      for (i=0; i < global_size; i++)
         data2[ j*vecstride + i*idxstride ] = i + 100*j;

/*   partitioning = hypre_CTAlloc(HYPRE_Int,4);
   partitioning[0] = 0;
   partitioning[1] = 10;
   partitioning[2] = 10;
   partitioning[3] = 20;
*/
   partitioning = hypre_CTAlloc(HYPRE_Int,1+num_procs);
   hypre_GeneratePartitioning( global_size, num_procs, &partitioning );

   vector2 = hypre_VectorToParVector(hypre_MPI_COMM_WORLD,local_vector2,partitioning);
   hypre_ParVectorSetPartitioningOwner(vector2,0);

   hypre_ParVectorPrint(vector2, "Convert");

   vector = hypre_ParVectorToVectorAll(vector2);

   /*-----------------------------------------------------------
    * Copy the vector into tmp_vector
    *-----------------------------------------------------------*/

/* Read doesn't work for multivectors yet...
   tmp_vector = hypre_ParVectorRead(hypre_MPI_COMM_WORLD, "Convert");*/
   tmp_vector = hypre_ParMultiVectorCreate
      ( hypre_MPI_COMM_WORLD, global_size, partitioning, num_vectors );
   hypre_ParVectorInitialize( tmp_vector );
   hypre_ParVectorCopy( vector2, tmp_vector );
/*
   tmp_vector = hypre_ParVectorCreate(hypre_MPI_COMM_WORLD,global_size,partitioning);
   hypre_ParVectorSetPartitioningOwner(tmp_vector,0);
   hypre_ParVectorInitialize(tmp_vector);
   hypre_ParVectorCopy(vector1, tmp_vector);

   hypre_ParVectorPrint(tmp_vector,"Copy");
*/
   /*-----------------------------------------------------------
    * Scale tmp_vector
    *-----------------------------------------------------------*/

   hypre_ParVectorScale(2.0, tmp_vector);
   hypre_ParVectorPrint(tmp_vector,"Scale");

   /*-----------------------------------------------------------
    * Do an Axpy (2*vector - vector) = vector
    *-----------------------------------------------------------*/

   hypre_ParVectorAxpy(-1.0, vector1, tmp_vector);
   hypre_ParVectorPrint(tmp_vector,"Axpy");

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
   if (vector) hypre_SeqVectorDestroy(vector); 

   /* Finalize MPI */
   hypre_MPI_Finalize();

   return 0;
}

