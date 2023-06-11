/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * hypre_IJVector interface
 *
 *****************************************************************************/

#include "./_hypre_IJ_mv.h"

#include "../HYPRE.h"

/*--------------------------------------------------------------------------
 * hypre_IJVectorDistribute
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_IJVectorDistribute( HYPRE_IJVector vector, const HYPRE_Int *vec_starts )
{
   hypre_IJVector *vec = (hypre_IJVector *) vector;

   if (vec == NULL)
   {
      hypre_printf("Vector variable is NULL -- hypre_IJVectorDistribute\n");
      exit(1);
   }

   if ( hypre_IJVectorObjectType(vec) == HYPRE_PARCSR )

   {
      return ( hypre_IJVectorDistributePar(vec, vec_starts) );
   }

   else
   {
      hypre_printf("Unrecognized object type -- hypre_IJVectorDistribute\n");
      exit(1);
   }

   return -99;
}

/*--------------------------------------------------------------------------
 * hypre_IJVectorZeroValues
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_IJVectorZeroValues( HYPRE_IJVector vector )
{
   hypre_IJVector *vec = (hypre_IJVector *) vector;

   if (vec == NULL)
   {
      hypre_printf("Vector variable is NULL -- hypre_IJVectorZeroValues\n");
      exit(1);
   }

   /*  if ( hypre_IJVectorObjectType(vec) == HYPRE_PETSC )

      return( hypre_IJVectorZeroValuesPETSc(vec) );

   else if ( hypre_IJVectorObjectType(vec) == HYPRE_ISIS )

      return( hypre_IJVectorZeroValuesISIS(vec) );

   else */

   if ( hypre_IJVectorObjectType(vec) == HYPRE_PARCSR )
   {
      return ( hypre_IJVectorZeroValuesPar(vec) );
   }
   else
   {
      hypre_printf("Unrecognized object type -- hypre_IJVectorZeroValues\n");
      exit(1);
   }

   return -99;
}

/*--------------------------------------------------------------------------
 * hypre_IJVectorReadBinary
 *
 * Reads a vector from file stored in binary format.
 * The resulting IJMatrix is stored on host memory.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_IJVectorReadBinary( MPI_Comm         comm,
                          const char      *filename,
                          HYPRE_Int        type,
                          HYPRE_IJVector  *vector_ptr )
{
   /* Vector variables */
   HYPRE_IJVector         vector;
   HYPRE_BigInt           partitioning[2];
   HYPRE_BigInt           global_size;
   HYPRE_Int              size;
   HYPRE_Int              num_components;
   HYPRE_Int              total_size;
   HYPRE_Int              storage_method;

   /* Buffers */
   hypre_float           *f32buffer = NULL;
   hypre_double          *f64buffer = NULL;
   HYPRE_Complex         *buffer;

   /* Local variables */
   FILE                  *fp;
   char                   new_filename[HYPRE_MAX_FILE_NAME_LEN];
   hypre_uint64           header[8];
   HYPRE_Int              myid;
   size_t                 count;
   HYPRE_Int              i, c;

   /* Open binary file */
   hypre_MPI_Comm_rank(comm, &myid);
   hypre_sprintf(new_filename, "%s.%05d.bin", filename, myid);
   if ((fp = fopen(new_filename, "r")) == NULL)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Could not open input file!");
      return hypre_error_flag;
   }

   /*---------------------------------------------
    * Read header (64 bytes)
    *---------------------------------------------*/

   count = 8;
   if (fread(header, sizeof(hypre_uint64), count, fp) != count)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Could not read header entries\n");
      return hypre_error_flag;
   }
   partitioning[0] = (HYPRE_BigInt) header[1];
   partitioning[1] = (HYPRE_BigInt) header[2];
   global_size     = (HYPRE_BigInt) header[3];
   size            = (HYPRE_Int) header[4];
   num_components  = (HYPRE_Int) header[5];
   total_size      = size * num_components;
   storage_method  = (HYPRE_Int) header[6];

   /* Sanity checks */
   if (storage_method == 1)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Does not support row-wise ordering!\n");
      return hypre_error_flag;
   }

   if (size > global_size)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Invalid vector size!\n");
      return hypre_error_flag;
   }

   /*---------------------------------------------
    * Read data
    *---------------------------------------------*/

   /* Allocate memory for buffers */
   count  = total_size;
   buffer = hypre_TAlloc(HYPRE_Complex, total_size, HYPRE_MEMORY_HOST);
   if (header[0] == sizeof(hypre_float))
   {
      f32buffer = hypre_TAlloc(hypre_float, header[0], HYPRE_MEMORY_HOST);
   }
   else if (header[0] == sizeof(hypre_double))
   {
      f64buffer = hypre_TAlloc(hypre_double, header[0], HYPRE_MEMORY_HOST);
   }
   else
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Unsupported data type for matrix coefficients");
      return hypre_error_flag;
   }

   /* Read data */
   if (f32buffer)
   {
      if (fread(f32buffer, sizeof(hypre_float), count, fp) != count)
      {
         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Could not read all matrix coefficients");
         return hypre_error_flag;
      }

#ifdef HYPRE_USING_OPENMP
      #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < total_size; i++)
      {
         buffer[i] = (HYPRE_Complex) f32buffer[i];
      }
   }
   else if (f64buffer)
   {
      if (fread(f64buffer, sizeof(hypre_double), count, fp) != count)
      {
         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Could not read all matrix coefficients");
         return hypre_error_flag;
      }

#ifdef HYPRE_USING_OPENMP
      #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < total_size; i++)
      {
         buffer[i] = (HYPRE_Complex) f64buffer[i];
      }
   }
   else
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Unsupported data type for vector entries");
      return hypre_error_flag;
   }

   /* Close file */
   fclose(fp);

   /*---------------------------------------------
    * Create vector
    *---------------------------------------------*/

   HYPRE_IJVectorCreate(comm, partitioning[0], partitioning[1] - 1, &vector);
   HYPRE_IJVectorSetObjectType(vector, type);
   HYPRE_IJVectorSetNumComponents(vector, num_components);
   HYPRE_IJVectorInitialize_v2(vector, HYPRE_MEMORY_HOST);
   for (c = 0; c < num_components; c++)
   {
      HYPRE_IJVectorSetComponent(vector, c);
      HYPRE_IJVectorSetValues(vector, size, NULL, buffer + c*size);
   }
   HYPRE_IJVectorAssemble(vector);

   *vector_ptr = vector;

   /*---------------------------------------------
    * Finalize
    *---------------------------------------------*/

   hypre_TFree(f32buffer, HYPRE_MEMORY_HOST);
   hypre_TFree(f64buffer, HYPRE_MEMORY_HOST);
   hypre_TFree(buffer, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}
