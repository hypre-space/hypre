/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * IJVector_Par interface
 *
 *****************************************************************************/

#include "_hypre_IJ_mv.h"
#include "../HYPRE.h"

/******************************************************************************
 *
 * hypre_IJVectorCreatePar
 *
 * creates ParVector if necessary, and leaves a pointer to it as the
 * hypre_IJVector object
 *
 *****************************************************************************/

HYPRE_Int
hypre_IJVectorCreatePar(hypre_IJVector *vector,
                        HYPRE_BigInt   *IJpartitioning)
{
   MPI_Comm      comm = hypre_IJVectorComm(vector);

   HYPRE_BigInt  global_n, partitioning[2], jmin;
   HYPRE_Int     j;

   jmin = hypre_IJVectorGlobalFirstRow(vector);
   global_n = hypre_IJVectorGlobalNumRows(vector);

   /* Shift to zero-based partitioning for ParVector object */
   for (j = 0; j < 2; j++)
   {
      partitioning[j] = IJpartitioning[j] - jmin;
   }

   hypre_IJVectorObject(vector) = (void*) hypre_ParVectorCreate(comm, global_n, partitioning);

   return hypre_error_flag;
}

/******************************************************************************
 *
 * hypre_IJVectorDestroyPar
 *
 * frees ParVector local storage of an IJVectorPar
 *
 *****************************************************************************/

HYPRE_Int
hypre_IJVectorDestroyPar(hypre_IJVector *vector)
{
   return hypre_ParVectorDestroy((hypre_ParVector*)hypre_IJVectorObject(vector));
}

/******************************************************************************
 *
 * hypre_IJVectorInitializePar
 *
 * initializes ParVector of IJVectorPar
 *
 *****************************************************************************/
HYPRE_Int
hypre_IJVectorInitializePar(hypre_IJVector *vector)
{
   return hypre_IJVectorInitializePar_v2(vector, hypre_IJVectorMemoryLocation(vector));
}

HYPRE_Int
hypre_IJVectorInitializePar_v2(hypre_IJVector *vector, HYPRE_MemoryLocation memory_location)
{
   MPI_Comm            comm         = hypre_IJVectorComm(vector);
   hypre_ParVector    *par_vector   = (hypre_ParVector*) hypre_IJVectorObject(vector);
   hypre_AuxParVector *aux_vector   = (hypre_AuxParVector*) hypre_IJVectorTranslator(vector);
   HYPRE_Int           print_level  = hypre_IJVectorPrintLevel(vector);
   HYPRE_Int           num_vectors  = hypre_IJVectorNumComponents(vector);

   HYPRE_BigInt       *partitioning = hypre_ParVectorPartitioning(par_vector);
   hypre_Vector       *local_vector = hypre_ParVectorLocalVector(par_vector);

   HYPRE_Int           my_id;

   HYPRE_MemoryLocation memory_location_aux =
      hypre_GetExecPolicy1(memory_location) == HYPRE_EXEC_HOST ? HYPRE_MEMORY_HOST : HYPRE_MEMORY_DEVICE;

   hypre_MPI_Comm_rank(comm, &my_id);

   if (!partitioning)
   {
      if (print_level)
      {
         hypre_printf("No ParVector partitioning for initialization -- ");
         hypre_printf("hypre_IJVectorInitializePar\n");
      }
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   hypre_VectorNumVectors(local_vector) = num_vectors;
   hypre_VectorSize(local_vector) = (HYPRE_Int)(partitioning[1] - partitioning[0]);

   hypre_ParVectorInitialize_v2(par_vector, memory_location);

   if (!aux_vector)
   {
      hypre_AuxParVectorCreate(&aux_vector);
      hypre_IJVectorTranslator(vector) = aux_vector;
   }
   hypre_AuxParVectorInitialize_v2(aux_vector, memory_location_aux);

   return hypre_error_flag;
}

/******************************************************************************
 *
 * hypre_IJVectorSetMaxOffProcElmtsPar
 *
 *****************************************************************************/

HYPRE_Int
hypre_IJVectorSetMaxOffProcElmtsPar(hypre_IJVector *vector,
                                    HYPRE_Int       max_off_proc_elmts)
{
   hypre_AuxParVector *aux_vector;

   aux_vector = (hypre_AuxParVector*) hypre_IJVectorTranslator(vector);
   if (!aux_vector)
   {
      hypre_AuxParVectorCreate(&aux_vector);
      hypre_IJVectorTranslator(vector) = aux_vector;
   }
   hypre_AuxParVectorMaxOffProcElmts(aux_vector) = max_off_proc_elmts;

#if defined(HYPRE_USING_GPU)
   hypre_AuxParVectorUsrOffProcElmts(aux_vector) = max_off_proc_elmts;
#endif

   return hypre_error_flag;
}

/******************************************************************************
 *
 * hypre_IJVectorDistributePar
 *
 * takes an IJVector generated for one processor and distributes it
 * across many processors according to vec_starts,
 * if vec_starts is NULL, it distributes them evenly?
 *
 *****************************************************************************/

HYPRE_Int
hypre_IJVectorDistributePar(hypre_IJVector  *vector,
                            const HYPRE_Int *vec_starts)
{
   hypre_ParVector *old_vector = (hypre_ParVector*) hypre_IJVectorObject(vector);
   hypre_ParVector *par_vector;
   HYPRE_Int print_level = hypre_IJVectorPrintLevel(vector);

   if (!old_vector)
   {
      if (print_level)
      {
         hypre_printf("old_vector == NULL -- ");
         hypre_printf("hypre_IJVectorDistributePar\n");
         hypre_printf("**** Vector storage is either unallocated or orphaned ****\n");
      }
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   par_vector = hypre_VectorToParVector(hypre_ParVectorComm(old_vector),
                                        hypre_ParVectorLocalVector(old_vector),
                                        (HYPRE_BigInt *)vec_starts);
   if (!par_vector)
   {
      if (print_level)
      {
         hypre_printf("par_vector == NULL -- ");
         hypre_printf("hypre_IJVectorDistributePar\n");
         hypre_printf("**** Vector storage is unallocated ****\n");
      }
      hypre_error_in_arg(1);
   }

   hypre_ParVectorDestroy(old_vector);

   hypre_IJVectorObject(vector) = par_vector;

   return hypre_error_flag;
}

/******************************************************************************
 *
 * hypre_IJVectorZeroValuesPar
 *
 * zeroes all local components of an IJVectorPar
 *
 *****************************************************************************/

HYPRE_Int
hypre_IJVectorZeroValuesPar(hypre_IJVector *vector)
{
   HYPRE_Int my_id;
   HYPRE_BigInt vec_start, vec_stop;

   hypre_ParVector *par_vector = (hypre_ParVector*) hypre_IJVectorObject(vector);
   MPI_Comm comm = hypre_IJVectorComm(vector);
   HYPRE_BigInt *partitioning;
   hypre_Vector *local_vector;
   HYPRE_Int print_level = hypre_IJVectorPrintLevel(vector);

   hypre_MPI_Comm_rank(comm, &my_id);

   /* If par_vector == NULL or partitioning == NULL or local_vector == NULL
      let user know of catastrophe and exit */

   if (!par_vector)
   {
      if (print_level)
      {
         hypre_printf("par_vector == NULL -- ");
         hypre_printf("hypre_IJVectorZeroValuesPar\n");
         hypre_printf("**** Vector storage is either unallocated or orphaned ****\n");
      }
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   partitioning = hypre_ParVectorPartitioning(par_vector);
   local_vector = hypre_ParVectorLocalVector(par_vector);
   if (!local_vector)
   {
      if (print_level)
      {
         hypre_printf("local_vector == NULL -- ");
         hypre_printf("hypre_IJVectorZeroValuesPar\n");
         hypre_printf("**** Vector local data is either unallocated or orphaned ****\n");
      }
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   vec_start = partitioning[0];
   vec_stop  = partitioning[1];

   if (vec_start > vec_stop)
   {
      if (print_level)
      {
         hypre_printf("vec_start > vec_stop -- ");
         hypre_printf("hypre_IJVectorZeroValuesPar\n");
         hypre_printf("**** This vector partitioning should not occur ****\n");
      }
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   hypre_assert(hypre_VectorSize(local_vector) == (HYPRE_Int)(vec_stop - vec_start));

   hypre_SeqVectorSetConstantValues(local_vector, 0.0);

   return hypre_error_flag;
}

/******************************************************************************
 *
 * hypre_IJVectorSetComponentPar
 *
 * Set the component identifier of a vector with multiple components
 * (multivector)
 *
 *****************************************************************************/

HYPRE_Int
hypre_IJVectorSetComponentPar(hypre_IJVector *vector,
                              HYPRE_Int       component)
{
   HYPRE_Int        print_level = hypre_IJVectorPrintLevel(vector);
   hypre_ParVector *par_vector  = (hypre_ParVector*) hypre_IJVectorObject(vector);
   HYPRE_Int        num_vectors = hypre_ParVectorNumVectors(par_vector);

   if (component < 0 || component > num_vectors)
   {
      if (print_level)
      {
         hypre_printf("component < 0 || component > num_vectors -- ");
         hypre_printf("hypre_IJVectorSetComponentPar\n");
      }
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }
   else
   {
      hypre_ParVectorSetComponent(par_vector, component);
   }

   return hypre_error_flag;
}

/******************************************************************************
 *
 * hypre_IJVectorSetValuesPar
 *
 * sets a potentially noncontiguous set of components of an IJVectorPar
 *
 *****************************************************************************/

HYPRE_Int
hypre_IJVectorSetValuesPar(hypre_IJVector       *vector,
                           HYPRE_Int             num_values,
                           const HYPRE_BigInt   *indices,
                           const HYPRE_Complex  *values)
{
   HYPRE_Int my_id;
   HYPRE_Int j, k;
   HYPRE_BigInt i, vec_start, vec_stop;
   HYPRE_Complex *data;
   HYPRE_Int print_level = hypre_IJVectorPrintLevel(vector);

   HYPRE_BigInt *IJpartitioning = hypre_IJVectorPartitioning(vector);
   hypre_ParVector *par_vector = (hypre_ParVector*) hypre_IJVectorObject(vector);
   MPI_Comm comm = hypre_IJVectorComm(vector);
   HYPRE_Int component;
   hypre_Vector *local_vector;
   HYPRE_Int vecoffset;
   HYPRE_Int vecstride;
   HYPRE_Int idxstride;

   /* If no components are to be set, perform no checking and return */
   if (num_values < 1) { return 0; }

   hypre_MPI_Comm_rank(comm, &my_id);

   /* If par_vector == NULL or partitioning == NULL or local_vector == NULL
      let user know of catastrophe and exit */

   if (!par_vector)
   {
      if (print_level)
      {
         hypre_printf("par_vector == NULL -- ");
         hypre_printf("hypre_IJVectorSetValuesPar\n");
         hypre_printf("**** Vector storage is either unallocated or orphaned ****\n");
      }
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   local_vector = hypre_ParVectorLocalVector(par_vector);
   if (!local_vector)
   {
      if (print_level)
      {
         hypre_printf("local_vector == NULL -- ");
         hypre_printf("hypre_IJVectorSetValuesPar\n");
         hypre_printf("**** Vector local data is either unallocated or orphaned ****\n");
      }
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   vec_start = IJpartitioning[0];
   vec_stop  = IJpartitioning[1] - 1;

   if (vec_start > vec_stop)
   {
      if (print_level)
      {
         hypre_printf("vec_start > vec_stop -- ");
         hypre_printf("hypre_IJVectorSetValuesPar\n");
         hypre_printf("**** This vector partitioning should not occur ****\n");
      }
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   /* Determine whether indices points to local indices only, and if not, store
      indices and values in auxiliary vector structure.  If indices == NULL,
      assume that num_values components are to be set in a block starting at
      vec_start.  NOTE: If indices == NULL off proc values are ignored!!! */

   data = hypre_VectorData(local_vector);
   component = hypre_VectorComponent(local_vector);
   vecstride = hypre_VectorVectorStride(local_vector);
   idxstride = hypre_VectorIndexStride(local_vector);
   vecoffset = component * vecstride;
   if (indices)
   {
      for (j = 0; j < num_values; j++)
      {
         i = indices[j];
         if (vec_start <= i && i <= vec_stop)
         {
            k = (HYPRE_Int)(i - vec_start);
            data[vecoffset + k * idxstride] = values[j];
         }
      }
   }
   else
   {
      if (num_values > (HYPRE_Int)(vec_stop - vec_start) + 1)
      {
         if (print_level)
         {
            hypre_printf("Warning! Indices beyond local range  not identified!\n ");
            hypre_printf("Off processor values have been ignored!\n");
         }
         num_values = (HYPRE_Int)(vec_stop - vec_start) + 1;
      }
#ifdef HYPRE_USING_OPENMP
      #pragma omp parallel for private(j) HYPRE_SMP_SCHEDULE
#endif
      for (j = 0; j < num_values; j++)
      {
         data[vecoffset + j * idxstride] = values[j];
      }
   }

   return hypre_error_flag;
}

/******************************************************************************
 *
 * hypre_IJVectorAddToValuesPar
 *
 * adds to a potentially noncontiguous set of IJVectorPar components
 *
 *****************************************************************************/

HYPRE_Int
hypre_IJVectorAddToValuesPar(hypre_IJVector       *vector,
                             HYPRE_Int             num_values,
                             const HYPRE_BigInt   *indices,
                             const HYPRE_Complex  *values)
{
   MPI_Comm            comm = hypre_IJVectorComm(vector);
   hypre_ParVector    *par_vector = (hypre_ParVector*) hypre_IJVectorObject(vector);
   hypre_AuxParVector *aux_vector = (hypre_AuxParVector*) hypre_IJVectorTranslator(vector);
   HYPRE_BigInt       *IJpartitioning = hypre_IJVectorPartitioning(vector);
   HYPRE_Int           print_level = hypre_IJVectorPrintLevel(vector);

   hypre_Vector       *local_vector;
   HYPRE_Int           idxstride, vecstride;
   HYPRE_Int           component, vecoffset;
   HYPRE_Int           num_vectors;
   HYPRE_Int           my_id;
   HYPRE_Int           j;
   HYPRE_BigInt        big_i, vec_start, vec_stop;
   HYPRE_Complex      *data;

   /* If no components are to be retrieved, perform no checking and return */
   if (num_values < 1)
   {
      return hypre_error_flag;
   }

   hypre_MPI_Comm_rank(comm, &my_id);

   /* If par_vector == NULL or partitioning == NULL or local_vector == NULL
      let user know of catastrophe and exit */

   if (!par_vector)
   {
      if (print_level)
      {
         hypre_printf("par_vector == NULL -- ");
         hypre_printf("hypre_IJVectorAddToValuesPar\n");
         hypre_printf("**** Vector storage is either unallocated or orphaned ****\n");
      }
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   local_vector = hypre_ParVectorLocalVector(par_vector);
   if (!local_vector)
   {
      if (print_level)
      {
         hypre_printf("local_vector == NULL -- ");
         hypre_printf("hypre_IJVectorAddToValuesPar\n");
         hypre_printf("**** Vector local data is either unallocated or orphaned ****\n");
      }
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   vec_start = IJpartitioning[0];
   vec_stop  = IJpartitioning[1] - 1;

   if (vec_start > vec_stop)
   {
      if (print_level)
      {
         hypre_printf("vec_start > vec_stop -- ");
         hypre_printf("hypre_IJVectorAddToValuesPar\n");
         hypre_printf("**** This vector partitioning should not occur ****\n");
      }
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   data = hypre_VectorData(local_vector);
   num_vectors = hypre_VectorNumVectors(local_vector);
   component   = hypre_VectorComponent(local_vector);
   vecstride   = hypre_VectorVectorStride(local_vector);
   idxstride   = hypre_VectorIndexStride(local_vector);
   vecoffset   = component * vecstride;

   if (indices)
   {
      HYPRE_Int current_num_elmts
         = hypre_AuxParVectorCurrentOffProcElmts(aux_vector);
      HYPRE_Int max_off_proc_elmts
         = hypre_AuxParVectorMaxOffProcElmts(aux_vector);
      HYPRE_BigInt *off_proc_i = hypre_AuxParVectorOffProcI(aux_vector);
      HYPRE_Complex *off_proc_data = hypre_AuxParVectorOffProcData(aux_vector);
      HYPRE_Int k;

      for (j = 0; j < num_values; j++)
      {
         big_i = indices[j];
         if (big_i < vec_start || big_i > vec_stop)
         {
            /* if elements outside processor boundaries, store in off processor
               stash */
            if (!max_off_proc_elmts)
            {
               max_off_proc_elmts = 100;
               hypre_AuxParVectorMaxOffProcElmts(aux_vector) = max_off_proc_elmts;
               hypre_AuxParVectorOffProcI(aux_vector) = hypre_CTAlloc(HYPRE_BigInt,
                                                                      max_off_proc_elmts,
                                                                      HYPRE_MEMORY_HOST);
               hypre_AuxParVectorOffProcData(aux_vector) = hypre_CTAlloc(HYPRE_Complex,
                                                                         max_off_proc_elmts,
                                                                         HYPRE_MEMORY_HOST);
               off_proc_i = hypre_AuxParVectorOffProcI(aux_vector);
               off_proc_data = hypre_AuxParVectorOffProcData(aux_vector);
            }
            else if (current_num_elmts + 1 > max_off_proc_elmts)
            {
               max_off_proc_elmts += 10;
               off_proc_i = hypre_TReAlloc(off_proc_i, HYPRE_BigInt, max_off_proc_elmts,
                                           HYPRE_MEMORY_HOST);
               off_proc_data = hypre_TReAlloc(off_proc_data, HYPRE_Complex,
                                              max_off_proc_elmts, HYPRE_MEMORY_HOST);
               hypre_AuxParVectorMaxOffProcElmts(aux_vector) = max_off_proc_elmts;
               hypre_AuxParVectorOffProcI(aux_vector) = off_proc_i;
               hypre_AuxParVectorOffProcData(aux_vector) = off_proc_data;
            }
            off_proc_i[current_num_elmts] = big_i;
            off_proc_data[current_num_elmts++] = values[j];
            hypre_AuxParVectorCurrentOffProcElmts(aux_vector) = current_num_elmts;
         }
         else /* local values are added to the vector */
         {
            k = (HYPRE_Int)(big_i - vec_start);
            data[vecoffset + k * idxstride] += values[j];
         }
      }

      if (current_num_elmts > 0 && num_vectors > 1)
      {
         hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                           "Off processor AddToValues not implemented for multivectors!\n");
         return hypre_error_flag;
      }
   }
   else
   {
      if (num_values > (HYPRE_Int)(vec_stop - vec_start) + 1)
      {
         if (print_level)
         {
            hypre_printf("Warning! Indices beyond local range  not identified!\n ");
            hypre_printf("Off processor values have been ignored!\n");
         }
         num_values = (HYPRE_Int)(vec_stop - vec_start) + 1;
      }

#ifdef HYPRE_USING_OPENMP
      #pragma omp parallel for private(j) HYPRE_SMP_SCHEDULE
#endif
      for (j = 0; j < num_values; j++)
      {
         data[vecoffset + j * idxstride] += values[j];
      }
   }

   return hypre_error_flag;
}

/******************************************************************************
 *
 * hypre_IJVectorAssemblePar
 *
 * currently tests existence of of ParVector object and its partitioning
 *
 *****************************************************************************/

HYPRE_Int
hypre_IJVectorAssemblePar(hypre_IJVector *vector)
{
   hypre_ParVector     *par_vector = (hypre_ParVector*) hypre_IJVectorObject(vector);
   hypre_AuxParVector  *aux_vector = (hypre_AuxParVector*) hypre_IJVectorTranslator(vector);
   MPI_Comm             comm = hypre_IJVectorComm(vector);
   HYPRE_Int            print_level = hypre_IJVectorPrintLevel(vector);

   if (!par_vector)
   {
      if (print_level)
      {
         hypre_printf("par_vector == NULL -- ");
         hypre_printf("hypre_IJVectorAssemblePar\n");
         hypre_printf("**** Vector storage is either unallocated or orphaned ****\n");
      }
      hypre_error_in_arg(1);
   }

   if (aux_vector)
   {
      HYPRE_Int off_proc_elmts, current_num_elmts;
      HYPRE_Int max_off_proc_elmts;
      HYPRE_BigInt *off_proc_i;
      HYPRE_Complex *off_proc_data;
      current_num_elmts = hypre_AuxParVectorCurrentOffProcElmts(aux_vector);
      hypre_MPI_Allreduce(&current_num_elmts, &off_proc_elmts, 1, HYPRE_MPI_INT,
                          hypre_MPI_SUM, comm);
      if (off_proc_elmts)
      {
         max_off_proc_elmts = hypre_AuxParVectorMaxOffProcElmts(aux_vector);
         off_proc_i = hypre_AuxParVectorOffProcI(aux_vector);
         off_proc_data = hypre_AuxParVectorOffProcData(aux_vector);
         hypre_IJVectorAssembleOffProcValsPar(vector, max_off_proc_elmts,
                                              current_num_elmts, HYPRE_MEMORY_HOST,
                                              off_proc_i, off_proc_data);
         hypre_TFree(hypre_AuxParVectorOffProcI(aux_vector), HYPRE_MEMORY_HOST);
         hypre_TFree(hypre_AuxParVectorOffProcData(aux_vector), HYPRE_MEMORY_HOST);
         hypre_AuxParVectorMaxOffProcElmts(aux_vector) = 0;
         hypre_AuxParVectorCurrentOffProcElmts(aux_vector) = 0;
      }
   }

   return hypre_error_flag;
}

/******************************************************************************
 *
 * hypre_IJVectorGetValuesPar
 *
 * get a potentially noncontiguous set of IJVectorPar components
 *
 *****************************************************************************/

HYPRE_Int
hypre_IJVectorGetValuesPar(hypre_IJVector      *vector,
                           HYPRE_Int            num_values,
                           const HYPRE_BigInt  *indices,
                           HYPRE_Complex       *values)
{
   HYPRE_Int        my_id;
   MPI_Comm         comm           = hypre_IJVectorComm(vector);
   HYPRE_BigInt    *IJpartitioning = hypre_IJVectorPartitioning(vector);
   HYPRE_BigInt     vec_start;
   HYPRE_BigInt     vec_stop;
   HYPRE_BigInt     jmin           = hypre_IJVectorGlobalFirstRow(vector);
   hypre_ParVector *par_vector     = (hypre_ParVector*) hypre_IJVectorObject(vector);
   HYPRE_Int        print_level    = hypre_IJVectorPrintLevel(vector);

   /* If no components are to be retrieved, perform no checking and return */
   if (num_values < 1)
   {
      return 0;
   }

   hypre_MPI_Comm_rank(comm, &my_id);

   /* If par_vector == NULL or partitioning == NULL or local_vector == NULL
      let user know of catastrophe and exit */

   if (!par_vector)
   {
      if (print_level)
      {
         hypre_printf("par_vector == NULL -- ");
         hypre_printf("hypre_IJVectorGetValuesPar\n");
         hypre_printf("**** Vector storage is either unallocated or orphaned ****\n");
      }
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   hypre_Vector *local_vector = hypre_ParVectorLocalVector(par_vector);
   if (!local_vector)
   {
      if (print_level)
      {
         hypre_printf("local_vector == NULL -- ");
         hypre_printf("hypre_IJVectorGetValuesPar\n");
         hypre_printf("**** Vector local data is either unallocated or orphaned ****\n");
      }
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   vec_start = IJpartitioning[0];
   vec_stop  = IJpartitioning[1];

   if (vec_start > vec_stop)
   {
      if (print_level)
      {
         hypre_printf("vec_start > vec_stop -- ");
         hypre_printf("hypre_IJVectorGetValuesPar\n");
         hypre_printf("**** This vector partitioning should not occur ****\n");
      }
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   hypre_ParVectorGetValues2(par_vector, num_values, (HYPRE_BigInt *) indices, jmin, values);

   return hypre_error_flag;
}

/******************************************************************************
 * hypre_IJVectorAssembleOffProcValsPar
 *
 * This is for handling set and get values calls to off-proc. entries - it is
 * called from assemble.  There is an alternate version for when the assumed
 * partition is being used.
 *****************************************************************************/

HYPRE_Int
hypre_IJVectorAssembleOffProcValsPar( hypre_IJVector       *vector,
                                      HYPRE_Int             max_off_proc_elmts,
                                      HYPRE_Int             current_num_elmts,
                                      HYPRE_MemoryLocation  memory_location,
                                      HYPRE_BigInt         *off_proc_i,
                                      HYPRE_Complex        *off_proc_data)
{
   HYPRE_UNUSED_VAR(max_off_proc_elmts);

   HYPRE_Int myid;
   HYPRE_BigInt global_first_row, global_num_rows;
   HYPRE_Int i, j, in, k;
   HYPRE_Int proc_id, last_proc, prev_id, tmp_id;
   HYPRE_Int max_response_size;
   HYPRE_Int ex_num_contacts = 0;
   HYPRE_BigInt range_start, range_end;
   HYPRE_Int storage;
   HYPRE_Int indx;
   HYPRE_BigInt row;
   HYPRE_Int num_ranges, row_count;
   HYPRE_Int num_recvs;
   HYPRE_Int counter;
   HYPRE_BigInt upper_bound;
   HYPRE_Int num_real_procs;

   HYPRE_BigInt *row_list = NULL;
   HYPRE_Int *a_proc_id = NULL, *orig_order = NULL;
   HYPRE_Int *real_proc_id = NULL, *us_real_proc_id = NULL;
   HYPRE_Int *ex_contact_procs = NULL, *ex_contact_vec_starts = NULL;
   HYPRE_Int *recv_starts = NULL;
   HYPRE_BigInt *response_buf = NULL;
   HYPRE_Int *response_buf_starts = NULL;
   HYPRE_Int *num_rows_per_proc = NULL;
   HYPRE_Int  tmp_int;
   HYPRE_Int  obj_size_bytes, big_int_size, complex_size;
   HYPRE_Int  first_index;

   void *void_contact_buf = NULL;
   void *index_ptr;
   void *recv_data_ptr;

   HYPRE_Complex tmp_complex;
   HYPRE_BigInt *ex_contact_buf = NULL;
   HYPRE_Complex *vector_data;
   HYPRE_Complex value;

   hypre_DataExchangeResponse      response_obj1, response_obj2;
   hypre_ProcListElements          send_proc_obj;

   MPI_Comm comm = hypre_IJVectorComm(vector);
   hypre_ParVector *par_vector = (hypre_ParVector*) hypre_IJVectorObject(vector);

   hypre_IJAssumedPart   *apart;

   hypre_MPI_Comm_rank(comm, &myid);

   global_num_rows = hypre_IJVectorGlobalNumRows(vector);
   global_first_row = hypre_IJVectorGlobalFirstRow(vector);

   if (memory_location == HYPRE_MEMORY_DEVICE)
   {
      HYPRE_BigInt  *off_proc_i_h    = hypre_TAlloc(HYPRE_BigInt,  current_num_elmts, HYPRE_MEMORY_HOST);
      HYPRE_Complex *off_proc_data_h = hypre_TAlloc(HYPRE_Complex, current_num_elmts, HYPRE_MEMORY_HOST);

      hypre_TMemcpy(off_proc_i_h,    off_proc_i,    HYPRE_BigInt,  current_num_elmts, HYPRE_MEMORY_HOST,
                    HYPRE_MEMORY_DEVICE);
      hypre_TMemcpy(off_proc_data_h, off_proc_data, HYPRE_Complex, current_num_elmts, HYPRE_MEMORY_HOST,
                    HYPRE_MEMORY_DEVICE);

      off_proc_i    = off_proc_i_h;
      off_proc_data = off_proc_data_h;
   }

   /* call hypre_IJVectorAddToValuesParCSR directly inside this function
    * with one chunk of data */
   HYPRE_Int      off_proc_nelm_recv_cur = 0;
   HYPRE_Int      off_proc_nelm_recv_max = 0;
   HYPRE_BigInt  *off_proc_i_recv = NULL;
   HYPRE_Complex *off_proc_data_recv = NULL;
   HYPRE_BigInt  *off_proc_i_recv_d = NULL;
   HYPRE_Complex *off_proc_data_recv_d = NULL;

   /* verify that we have created the assumed partition */
   if  (hypre_IJVectorAssumedPart(vector) == NULL)
   {
      hypre_IJVectorCreateAssumedPartition(vector);
   }

   apart = (hypre_IJAssumedPart*) hypre_IJVectorAssumedPart(vector);

   /* get the assumed processor id for each row */
   a_proc_id = hypre_CTAlloc(HYPRE_Int,  current_num_elmts, HYPRE_MEMORY_HOST);
   orig_order =  hypre_CTAlloc(HYPRE_Int,  current_num_elmts, HYPRE_MEMORY_HOST);
   real_proc_id = hypre_CTAlloc(HYPRE_Int,  current_num_elmts, HYPRE_MEMORY_HOST);
   row_list =   hypre_CTAlloc(HYPRE_BigInt,  current_num_elmts, HYPRE_MEMORY_HOST);

   if (current_num_elmts > 0)
   {
      for (i = 0; i < current_num_elmts; i++)
      {
         row = off_proc_i[i];
         row_list[i] = row;
         hypre_GetAssumedPartitionProcFromRow(comm, row, global_first_row,
                                              global_num_rows, &proc_id);
         a_proc_id[i] = proc_id;
         orig_order[i] = i;
      }

      /* now we need to find the actual order of each row  - sort on row -
         this will result in proc ids sorted also...*/

      hypre_BigQsortb2i(row_list, a_proc_id, orig_order, 0, current_num_elmts - 1);

      /* calculate the number of contacts */
      ex_num_contacts = 1;
      last_proc = a_proc_id[0];
      for (i = 1; i < current_num_elmts; i++)
      {
         if (a_proc_id[i] > last_proc)
         {
            ex_num_contacts++;
            last_proc = a_proc_id[i];
         }
      }

   }

   /* now we will go through a create a contact list - need to contact
      assumed processors and find out who the actual row owner is - we
      will contact with a range (2 numbers) */

   ex_contact_procs = hypre_CTAlloc(HYPRE_Int,  ex_num_contacts, HYPRE_MEMORY_HOST);
   ex_contact_vec_starts =  hypre_CTAlloc(HYPRE_Int,  ex_num_contacts + 1, HYPRE_MEMORY_HOST);
   ex_contact_buf =  hypre_CTAlloc(HYPRE_BigInt,  ex_num_contacts * 2, HYPRE_MEMORY_HOST);

   counter = 0;
   range_end = -1;
   for (i = 0; i < current_num_elmts; i++)
   {
      if (row_list[i] > range_end)
      {
         /* assumed proc */
         proc_id = a_proc_id[i];

         /* end of prev. range */
         if (counter > 0) { ex_contact_buf[counter * 2 - 1] = row_list[i - 1]; }

         /*start new range*/
         ex_contact_procs[counter] = proc_id;
         ex_contact_vec_starts[counter] = counter * 2;
         ex_contact_buf[counter * 2] =  row_list[i];
         counter++;

         hypre_GetAssumedPartitionRowRange(comm, proc_id, global_first_row,
                                           global_num_rows, &range_start, &range_end);
      }
   }

   /*finish the starts*/
   ex_contact_vec_starts[counter] =  counter * 2;
   /*finish the last range*/
   if (counter > 0)
   {
      ex_contact_buf[counter * 2 - 1] = row_list[current_num_elmts - 1];
   }

   /* create response object - can use same fill response as used in the commpkg
      routine */
   response_obj1.fill_response = hypre_RangeFillResponseIJDetermineRecvProcs;
   response_obj1.data1 =  apart; /* this is necessary so we can fill responses*/
   response_obj1.data2 = NULL;

   max_response_size = 6;  /* 6 means we can fit 3 ranges*/

   hypre_DataExchangeList(ex_num_contacts, ex_contact_procs,
                          ex_contact_buf, ex_contact_vec_starts, sizeof(HYPRE_BigInt),
                          sizeof(HYPRE_BigInt), &response_obj1, max_response_size, 4,
                          comm, (void**) &response_buf, &response_buf_starts);

   /* now response_buf contains a proc_id followed by an upper bound for the
      range.  */

   hypre_TFree(ex_contact_procs, HYPRE_MEMORY_HOST);
   hypre_TFree(ex_contact_buf, HYPRE_MEMORY_HOST);
   hypre_TFree(ex_contact_vec_starts, HYPRE_MEMORY_HOST);

   hypre_TFree(a_proc_id, HYPRE_MEMORY_HOST);
   a_proc_id = NULL;

   /*how many ranges were returned?*/
   num_ranges = response_buf_starts[ex_num_contacts];
   num_ranges = num_ranges / 2;

   prev_id = -1;
   j = 0;
   counter = 0;
   num_real_procs = 0;

   /* loop through ranges - create a list of actual processor ids*/
   for (i = 0; i < num_ranges; i++)
   {
      upper_bound = response_buf[i * 2 + 1];
      counter = 0;
      tmp_id = (HYPRE_Int)response_buf[i * 2];

      /* loop through row_list entries - counting how many are in the range */
      while (j < current_num_elmts && row_list[j] <= upper_bound)
      {
         real_proc_id[j] = tmp_id;
         j++;
         counter++;
      }
      if (counter > 0 && tmp_id != prev_id)
      {
         num_real_procs++;
      }
      prev_id = tmp_id;
   }

   /* now we have the list of real procesors ids (real_proc_id) - and the number
      of distinct ones - so now we can set up data to be sent - we have
      HYPRE_Int and HYPRE_Complex data.  (row number and value) - we will send
      everything as a void since we may not know the rel sizes of ints and
      doubles */

   /* first find out how many elements to send per proc - so we can do
      storage */

   complex_size = sizeof(HYPRE_Complex);
   big_int_size = sizeof(HYPRE_BigInt);

   obj_size_bytes = hypre_max(big_int_size, complex_size);

   ex_contact_procs = hypre_CTAlloc(HYPRE_Int,  num_real_procs, HYPRE_MEMORY_HOST);
   num_rows_per_proc = hypre_CTAlloc(HYPRE_Int,  num_real_procs, HYPRE_MEMORY_HOST);

   counter = 0;

   if (num_real_procs > 0 )
   {
      ex_contact_procs[0] = real_proc_id[0];
      num_rows_per_proc[0] = 1;

      /* loop through real procs - these are sorted (row_list is sorted also)*/
      for (i = 1; i < current_num_elmts; i++)
      {
         if (real_proc_id[i] == ex_contact_procs[counter]) /* same processor */
         {
            num_rows_per_proc[counter] += 1; /*another row */
         }
         else /* new processor */
         {
            counter++;
            ex_contact_procs[counter] = real_proc_id[i];
            num_rows_per_proc[counter] = 1;
         }
      }
   }

   /* calculate total storage and make vec_starts arrays */
   storage = 0;
   ex_contact_vec_starts = hypre_CTAlloc(HYPRE_Int,  num_real_procs + 1, HYPRE_MEMORY_HOST);
   ex_contact_vec_starts[0] = -1;

   for (i = 0; i < num_real_procs; i++)
   {
      storage += 1 + 2 *  num_rows_per_proc[i];
      ex_contact_vec_starts[i + 1] = -storage - 1; /* need negative for next loop */
   }

   /*void_contact_buf = hypre_TAlloc(char, storage*obj_size_bytes);*/
   void_contact_buf = hypre_CTAlloc(char, storage * obj_size_bytes, HYPRE_MEMORY_HOST);
   index_ptr = void_contact_buf; /* step through with this index */

   /* set up data to be sent to send procs */
   /* for each proc, ex_contact_buf_d contains #rows, row #, data, etc. */

   /* un-sort real_proc_id  - we want to access data arrays in order */

   us_real_proc_id =  hypre_CTAlloc(HYPRE_Int,  current_num_elmts, HYPRE_MEMORY_HOST);
   for (i = 0; i < current_num_elmts; i++)
   {
      us_real_proc_id[orig_order[i]] = real_proc_id[i];
   }
   hypre_TFree(real_proc_id, HYPRE_MEMORY_HOST);

   prev_id = -1;
   for (i = 0; i < current_num_elmts; i++)
   {
      proc_id = us_real_proc_id[i];
      /* can't use row list[i] - you loose the negative signs that differentiate
         add/set values */
      row = off_proc_i[i];
      /* find position of this processor */
      indx = hypre_BinarySearch(ex_contact_procs, proc_id, num_real_procs);
      in =  ex_contact_vec_starts[indx];

      index_ptr = (void *) ((char *) void_contact_buf + in * obj_size_bytes);

      /* first time for this processor - add the number of rows to the buffer */
      if (in < 0)
      {
         in = -in - 1;
         /* re-calc. index_ptr since in_i was negative */
         index_ptr = (void *) ((char *) void_contact_buf + in * obj_size_bytes);

         tmp_int = num_rows_per_proc[indx];
         hypre_TMemcpy(index_ptr, &tmp_int,  HYPRE_Int, 1, HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
         index_ptr = (void *) ((char *) index_ptr + obj_size_bytes);

         in++;
      }
      /* add row # */
      hypre_TMemcpy(index_ptr, &row, HYPRE_BigInt, 1, HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
      index_ptr = (void *) ((char *) index_ptr + obj_size_bytes);
      in++;

      /* add value */
      tmp_complex = off_proc_data[i];
      hypre_TMemcpy(index_ptr, &tmp_complex, HYPRE_Complex, 1,
                    HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
      index_ptr = (void *) ((char *) index_ptr + obj_size_bytes);
      in++;

      /* increment the indexes to keep track of where we are - fix later */
      ex_contact_vec_starts[indx] = in;
   }

   /* some clean up */

   hypre_TFree(response_buf, HYPRE_MEMORY_HOST);
   hypre_TFree(response_buf_starts, HYPRE_MEMORY_HOST);

   hypre_TFree(us_real_proc_id, HYPRE_MEMORY_HOST);
   hypre_TFree(orig_order, HYPRE_MEMORY_HOST);
   hypre_TFree(row_list, HYPRE_MEMORY_HOST);
   hypre_TFree(num_rows_per_proc, HYPRE_MEMORY_HOST);

   for (i = num_real_procs; i > 0; i--)
   {
      ex_contact_vec_starts[i] =   ex_contact_vec_starts[i - 1];
   }

   ex_contact_vec_starts[0] = 0;

   /* now send the data */

   /***********************************/
   /* now get the info in send_proc_obj_d */

   /* the response we expect is just a confirmation*/
   response_buf = NULL;
   response_buf_starts = NULL;

   /*build the response object*/

   /* use the send_proc_obj for the info kept from contacts */
   /*estimate inital storage allocation */

   send_proc_obj.length = 0;
   send_proc_obj.storage_length = num_real_procs + 5;
   send_proc_obj.id = NULL; /* don't care who sent it to us */
   send_proc_obj.vec_starts =
      hypre_CTAlloc(HYPRE_Int,  send_proc_obj.storage_length + 1, HYPRE_MEMORY_HOST);
   send_proc_obj.vec_starts[0] = 0;
   send_proc_obj.element_storage_length = storage + 20;
   send_proc_obj.v_elements =
      hypre_TAlloc(char, obj_size_bytes * send_proc_obj.element_storage_length, HYPRE_MEMORY_HOST);

   response_obj2.fill_response = hypre_FillResponseIJOffProcVals;
   response_obj2.data1 = NULL;
   response_obj2.data2 = &send_proc_obj;

   max_response_size = 0;

   hypre_DataExchangeList(num_real_procs, ex_contact_procs,
                          void_contact_buf, ex_contact_vec_starts, obj_size_bytes,
                          0, &response_obj2, max_response_size, 5,
                          comm,  (void **) &response_buf, &response_buf_starts);

   /***********************************/

   hypre_TFree(response_buf, HYPRE_MEMORY_HOST);
   hypre_TFree(response_buf_starts, HYPRE_MEMORY_HOST);

   hypre_TFree(ex_contact_procs, HYPRE_MEMORY_HOST);
   hypre_TFree(void_contact_buf, HYPRE_MEMORY_HOST);
   hypre_TFree(ex_contact_vec_starts, HYPRE_MEMORY_HOST);

   /* Now we can unpack the send_proc_objects and either set or add to the
      vector data */

   num_recvs = send_proc_obj.length;

   /* alias */
   recv_data_ptr = send_proc_obj.v_elements;
   recv_starts = send_proc_obj.vec_starts;

   vector_data = hypre_VectorData(hypre_ParVectorLocalVector(par_vector));
   first_index = hypre_ParVectorFirstIndex(par_vector);

   for (i = 0; i < num_recvs; i++)
   {
      indx = recv_starts[i];

      /* get the number of rows for  this recv */
      hypre_TMemcpy(&row_count, recv_data_ptr, HYPRE_Int, 1,
                    HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
      recv_data_ptr = (void *) ((char *)recv_data_ptr + obj_size_bytes);
      indx++;

      for (j = 0; j < row_count; j++) /* for each row: unpack info */
      {
         /* row # */
         hypre_TMemcpy(&row, recv_data_ptr, HYPRE_BigInt, 1,
                       HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
         recv_data_ptr = (void *) ((char *)recv_data_ptr + obj_size_bytes);
         indx++;

         /* value */
         hypre_TMemcpy(&value, recv_data_ptr, HYPRE_Complex, 1,
                       HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
         recv_data_ptr = (void *) ((char *)recv_data_ptr + obj_size_bytes);
         indx++;

         if (memory_location == HYPRE_MEMORY_HOST)
         {
            k = (HYPRE_Int)(row - first_index - global_first_row);
            vector_data[k] += value;
         }
         else
         {
            if (off_proc_nelm_recv_cur >= off_proc_nelm_recv_max)
            {
               off_proc_nelm_recv_max = 2 * (off_proc_nelm_recv_cur + 1);
               off_proc_i_recv    = hypre_TReAlloc(off_proc_i_recv,    HYPRE_BigInt,  off_proc_nelm_recv_max,
                                                   HYPRE_MEMORY_HOST);
               off_proc_data_recv = hypre_TReAlloc(off_proc_data_recv, HYPRE_Complex, off_proc_nelm_recv_max,
                                                   HYPRE_MEMORY_HOST);
            }
            off_proc_i_recv[off_proc_nelm_recv_cur] = row;
            off_proc_data_recv[off_proc_nelm_recv_cur] = value;
            off_proc_nelm_recv_cur ++;
         }
      }
   }

   if (memory_location == HYPRE_MEMORY_DEVICE)
   {
      off_proc_i_recv_d    = hypre_TAlloc(HYPRE_BigInt,  off_proc_nelm_recv_cur, HYPRE_MEMORY_DEVICE);
      off_proc_data_recv_d = hypre_TAlloc(HYPRE_Complex, off_proc_nelm_recv_cur, HYPRE_MEMORY_DEVICE);

      hypre_TMemcpy(off_proc_i_recv_d,    off_proc_i_recv,    HYPRE_BigInt,  off_proc_nelm_recv_cur,
                    HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
      hypre_TMemcpy(off_proc_data_recv_d, off_proc_data_recv, HYPRE_Complex, off_proc_nelm_recv_cur,
                    HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);

#if defined(HYPRE_USING_GPU)
      hypre_IJVectorSetAddValuesParDevice(vector, off_proc_nelm_recv_cur, off_proc_i_recv_d,
                                          off_proc_data_recv_d, "add");
#endif
   }

   hypre_TFree(send_proc_obj.v_elements, HYPRE_MEMORY_HOST);
   hypre_TFree(send_proc_obj.vec_starts, HYPRE_MEMORY_HOST);

   if (memory_location == HYPRE_MEMORY_DEVICE)
   {
      hypre_TFree(off_proc_i,    HYPRE_MEMORY_HOST);
      hypre_TFree(off_proc_data, HYPRE_MEMORY_HOST);
   }

   hypre_TFree(off_proc_i_recv,    HYPRE_MEMORY_HOST);
   hypre_TFree(off_proc_data_recv, HYPRE_MEMORY_HOST);

   hypre_TFree(off_proc_i_recv_d,    HYPRE_MEMORY_DEVICE);
   hypre_TFree(off_proc_data_recv_d, HYPRE_MEMORY_DEVICE);

   return hypre_error_flag;
}
