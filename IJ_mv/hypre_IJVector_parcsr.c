/*BHEADER**********************************************************************
 * (c) 1999   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * IJVector_Par interface
 *
 *****************************************************************************/
 
#include "IJ_mv.h"
#include "aux_parcsr_matrix.h"

/******************************************************************************
 *
 * hypre_IJVectorCreatePar
 *
 * creates ParVector if necessary, and leaves a pointer to it in the
 * hypre_IJVector local_storage
 *
 *****************************************************************************/
int
hypre_IJVectorCreatePar(hypre_IJVector *vector, const int *partitioning)
{
   MPI_Comm comm = hypre_IJVectorContext(vector);
   int global_n = hypre_IJVectorN(vector); 
   hypre_ParVector *par_vector = hypre_IJVectorLocalStorage(vector);
   int ierr = 0;

   int my_id;
   MPI_Comm_rank(comm, &my_id);

   if (!par_vector)
   {
      hypre_IJVectorLocalStorage(vector) = hypre_ParVectorCreate(comm,
               global_n, (int *) partitioning); 
   } 

   return ierr;
}

/******************************************************************************
 *
 * hypre_IJVectorDestroyPar
 *
 * frees ParVector local storage of an IJVectorPar 
 *
 *****************************************************************************/
int
hypre_IJVectorDestroyPar(hypre_IJVector *vector)
{
   return hypre_ParVectorDestroy(hypre_IJVectorLocalStorage(vector));
}

/******************************************************************************
 *
 * hypre_IJVectorSetPartitioningPar
 *
 * initializes IJVectorPar ParVector partitioning
 *
 *****************************************************************************/

int
hypre_IJVectorSetPartitioningPar(hypre_IJVector *vector,
                                 const int      *partitioning )
{
   int ierr = 0;
   hypre_ParVector *par_vector = hypre_IJVectorLocalStorage(vector);

   if (!par_vector)
   {
      MPI_Comm comm = hypre_IJVectorContext(vector);
      int global_n = hypre_IJVectorN(vector); 
      hypre_IJVectorLocalStorage(vector) = hypre_ParVectorCreate(comm,
               global_n, (int *) partitioning); 
   }
   else
      hypre_ParVectorPartitioning(par_vector) = (int *) partitioning;

   return ierr;
}

/******************************************************************************
 *
 * hypre_IJVectorSetLocalPartitioningPar
 *
 * initializes IJVectorPar ParVector local partitioning
 *
 *****************************************************************************/

int
hypre_IJVectorSetLocalPartitioningPar(hypre_IJVector *vector,
                                      int             vec_start_this_proc,
                                      int             vec_start_next_proc  )
{
   int ierr = 0;
   hypre_ParVector *par_vector = hypre_IJVectorLocalStorage(vector);
   int *partitioning = hypre_ParVectorPartitioning(par_vector);
   int num_procs, my_id;
   MPI_Comm comm = hypre_IJVectorContext(vector);

   MPI_Comm_size(comm, &num_procs);
   MPI_Comm_rank(comm, &my_id);

   if (vec_start_this_proc > vec_start_next_proc) 
   {
      printf("vec_start_this_proc > vec_start_next_proc -- ");
      printf("hypre_IJVectorSetLocalPartitioningPar\n");
      printf("**** This vector partitioning should not occur ****\n");
      exit(1);
   }

   if (!partitioning)
      partitioning = hypre_CTAlloc(int, num_procs);

   if (partitioning)
   {   
      partitioning[my_id] = vec_start_this_proc;
      partitioning[my_id+1] = vec_start_next_proc;

      hypre_ParVectorPartitioning(par_vector) = partitioning;
   }
   else
      ++ierr;

   return ierr;
}

/******************************************************************************
 *
 * hypre_IJVectorInitializePar
 *
 * initializes ParVector of IJVectorPar
 *
 *****************************************************************************/

int
hypre_IJVectorInitializePar(hypre_IJVector *vector)
{
   int ierr = 0;
   hypre_ParVector *par_vector = hypre_IJVectorLocalStorage(vector);
   int *partitioning = hypre_ParVectorPartitioning(par_vector);
   hypre_Vector *local_vector = hypre_ParVectorLocalVector(par_vector);
   int my_id;
   MPI_Comm  comm = hypre_IJVectorContext(vector);

   MPI_Comm_rank(comm,&my_id);
  
   if (partitioning)
   {
      hypre_VectorSize(local_vector) = partitioning[my_id+1] -
                                       partitioning[my_id];
      ierr += hypre_ParVectorInitialize(par_vector);
   }
   else
      ++ierr;

   return ierr;
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
int
hypre_IJVectorDistributePar(hypre_IJVector *vector,
			    const int	   *vec_starts)
{
   int ierr = 0;

   hypre_ParVector *old_vector = hypre_IJVectorLocalStorage(vector);
   hypre_ParVector *par_vector;
   
   if (!old_vector)
   {
      printf("old_vector == NULL -- ");
      printf("hypre_IJVectorDistributePar\n");
      printf("**** Vector storage is either unallocated or orphaned ****\n");
      exit(1);
   }

   par_vector = hypre_VectorToParVector(hypre_ParVectorComm(old_vector),
		                        hypre_ParVectorLocalVector(old_vector),
                                        (int *)vec_starts);
   if (!par_vector)
   {
      printf("par_vector == NULL -- ");
      printf("hypre_IJVectorDistributePar\n");
      printf("**** Vector storage is unallocated ****\n");
      exit(1);
   }

   ierr = hypre_ParVectorDestroy(old_vector);

   hypre_IJVectorLocalStorage(vector) = par_vector;

   return ierr;
}

/******************************************************************************
 *
 * hypre_IJVectorZeroLocalComponentsPar
 *
 * zeroes all local components of an IJVectorPar
 *
 *****************************************************************************/
int
hypre_IJVectorZeroLocalComponentsPar(hypre_IJVector *vector)
{
   int ierr = 0;
   int my_id;
   int i, vec_start, vec_stop;
   double *data;

   hypre_ParVector *par_vector = hypre_IJVectorLocalStorage(vector);
   MPI_Comm comm = hypre_IJVectorContext(vector);
   int *partitioning = hypre_ParVectorPartitioning(par_vector);
   hypre_Vector *local_vector = hypre_ParVectorLocalVector(par_vector);

   MPI_Comm_rank(comm, &my_id);

/* If par_vector == NULL or partitioning == NULL or local_vector == NULL 
   let user know of catastrophe and exit */

   if (!par_vector)
   {
      printf("par_vector == NULL -- ");
      printf("hypre_IJVectorZeroLocalComponentsPar\n");
      printf("**** Vector storage is either unallocated or orphaned ****\n");
      exit(1);
   }
   if (!partitioning)
   {
      printf("partitioning == NULL -- ");
      printf("hypre_IJVectorZeroLocalComponentsPar\n");
      printf("**** Vector partitioning is either unallocated or orphaned ****\n");
      exit(1);
   }
   if (!local_vector)
   {
      printf("local_vector == NULL -- ");
      printf("hypre_IJVectorZeroLocalComponentsPar\n");
      printf("**** Vector local data is either unallocated or orphaned ****\n");
      exit(1);
   }

   vec_start = partitioning[my_id];
   vec_stop  = partitioning[my_id+1];
   
   if (vec_start > vec_stop) 
   {
      printf("vec_start > vec_stop -- ");
      printf("hypre_IJVectorZeroLocalComponentsPar\n");
      printf("**** This vector partitioning should not occur ****\n");
      exit(1);
   }

   data = hypre_VectorData( local_vector );
   for (i = 0; i < vec_stop - vec_start; i++)
      data[i] = 0.;
  
   return ierr;
}

/******************************************************************************
 *
 * hypre_IJVectorSetLocalComponentsPar
 *
 * sets a potentially noncontiguous set of components of an IJVectorPar
 *
 *****************************************************************************/
int
hypre_IJVectorSetLocalComponentsPar(hypre_IJVector *vector,
                                    int             num_values,
                                    const int      *glob_vec_indices,
                                    const int      *value_indices,
                                    const double   *values            )
{
   int ierr = 0;
   int my_id;
   int i, j, vec_start, vec_stop;
   double *data;

   hypre_ParVector *par_vector = hypre_IJVectorLocalStorage(vector);
   MPI_Comm comm = hypre_IJVectorContext(vector);
   int *partitioning = hypre_ParVectorPartitioning(par_vector);
   hypre_Vector *local_vector = hypre_ParVectorLocalVector(par_vector);

/* If no components are to be set, perform no checking and return */
   if (num_values < 1) return ierr;

   MPI_Comm_rank(comm, &my_id);

/* If par_vector == NULL or partitioning == NULL or local_vector == NULL 
   let user know of catastrophe and exit */

   if (!par_vector)
   {
      printf("par_vector == NULL -- ");
      printf("hypre_IJVectorSetLocalComponentsPar\n");
      printf("**** Vector storage is either unallocated or orphaned ****\n");
      exit(1);
   }
   if (!partitioning)
   {
      printf("partitioning == NULL -- ");
      printf("hypre_IJVectorSetLocalComponentsPar\n");
      printf("**** Vector partitioning is either unallocated or orphaned ****\n");
      exit(1);
   }
   if (!local_vector)
   {
      printf("local_vector == NULL -- ");
      printf("hypre_IJVectorSetLocalComponentsPar\n");
      printf("**** Vector local data is either unallocated or orphaned ****\n");
      exit(1);
   }

   vec_start = partitioning[my_id];
   vec_stop  = partitioning[my_id+1];
  
   if (vec_start > vec_stop) 
   {
      printf("vec_start > vec_stop -- ");
      printf("hypre_IJVectorSetLocalComponentsPar\n");
      printf("**** This vector partitioning should not occur ****\n");
      exit(1);
   }

/* Determine whether glob_vec_indices points to local indices only,
   and if not, let user know of catastrophe and exit.
   If glob_vec_indices == NULL, assume that num_values components are to be
   set in a block starting at vec_start */

   if (glob_vec_indices)
   {
      for (i = 0; i < num_values; i++)
      { 	
	ierr += (glob_vec_indices[i] <  vec_start);
        ierr += (glob_vec_indices[i] >= vec_stop);
      }
   }

   if (ierr)
   {
      printf("glob_vec_indices beyond local range -- ");
      printf("hypre_IJVectorSetLocalComponentsPar\n");
      printf("**** Glob_vec_indices specified are unusable ****\n");
      exit(1);
   }
    
   data = hypre_VectorData(local_vector);
   if (!value_indices)
   {
      if (glob_vec_indices)
      {
         for (j = 0; j < num_values; j++)
         {
            i = glob_vec_indices[j] - vec_start;
            data[i] = values[j];
         } 
      }
      else
      {
         for (j = 0; j < num_values; j++)
            data[j] = values[j];
      } 
   } 
   else if (value_indices)
   {
      if (glob_vec_indices)
      {
         for (j = 0; j < num_values; j++)
         {
            i = glob_vec_indices[j] - vec_start;
            data[i] = values[value_indices[j]];
         } 
      }
      else
      {
         for (j = 0; j < num_values; j++)
            data[j] = values[value_indices[j]];
      } 
   }  
  
   return ierr;
}

/******************************************************************************
 *
 * hypre_IJVectorSetLocalComponentsInBlockPar
 *
 * sets a contiguous set of components of an IJVectorPar
 *
 *****************************************************************************/
int
hypre_IJVectorSetLocalComponentsInBlockPar(hypre_IJVector *vector,
                                           int             glob_vec_index_start,
                                           int             glob_vec_index_stop,
                                           const int      *value_indices,
                                           const double   *values                )
{
   int ierr = 0;
   int my_id;
   int i, vec_start, vec_stop, local_n, local_start, local_stop;
   double *data;

   hypre_ParVector *par_vector = hypre_IJVectorLocalStorage(vector);
   MPI_Comm comm = hypre_IJVectorContext(vector);
   int *partitioning = hypre_ParVectorPartitioning(par_vector);
   hypre_Vector *local_vector = hypre_ParVectorLocalVector(par_vector);

/* If no components are to be set, perform no checking and return */
   if (glob_vec_index_start > glob_vec_index_stop) return ierr;

   MPI_Comm_rank(comm, &my_id);

/* If par_vector == NULL or partitioning == NULL or local_vector == NULL 
   let user know of catastrophe and exit */

   if (!par_vector)
   {
      printf("par_vector == NULL -- ");
      printf("hypre_IJVectorSetLocalComponentsInBlockPar\n");
      printf("**** Vector storage is either unallocated or orphaned ****\n");
      exit(1);
   }
   if (!partitioning)
   {
      printf("partitioning == NULL -- ");
      printf("hypre_IJVectorSetLocalComponentsInBlockPar\n");
      printf("**** Vector partitioning is either unallocated or orphaned ****\n");
      exit(1);
   }
   if (!local_vector)
   {
      printf("local_vector == NULL -- ");
      printf("hypre_IJVectorSetLocalComponentsInBlockPar\n");
      printf("**** Vector local data is either unallocated or orphaned ****\n");
      exit(1);
   }

   vec_start = partitioning[my_id];
   vec_stop  = partitioning[my_id+1];

   if (vec_start > vec_stop) 
   {
      printf("vec_start > vec_stop -- ");
      printf("hypre_IJVectorSetLocalComponentsInBlockPar\n");
      printf("**** This vector partitioning should not occur ****\n");
      exit(1);
   }
   else if (glob_vec_index_start < vec_start)
   {
      printf("glob_vec_index_start below local range -- ");
      printf("hypre_IJVectorSetLocalComponentsInBlockPar\n");
      printf("**** The specified glob_vec_index_start is invalid ****\n");
      exit(1);
   }
   else if (glob_vec_index_start > vec_stop-1)
   {
      printf("glob_vec_index_start above local range -- ");
      printf("hypre_IJVectorSetLocalComponentsInBlockPar\n");
      printf("**** The specified glob_vec_index_start is invalid ****\n");
      exit(1);
   }
   if (glob_vec_index_stop < vec_start)
   {
      printf("glob_vec_index_stop below local range -- ");
      printf("hypre_IJVectorSetLocalComponentsInBlockPar\n");
      printf("**** The specified glob_vec_index_stop is invalid ****\n");
      exit(1);
   }
   else if (glob_vec_index_stop > vec_stop-1)
   {
      printf("glob_vec_index_stop above local range -- ");
      printf("hypre_IJVectorSetLocalComponentsInBlockPar\n");
      printf("**** The specified glob_vec_index_stop is invalid ****\n");
      exit(1);
   }

   local_n = vec_stop - vec_start;
   local_start = glob_vec_index_start - vec_start;
   local_stop  = glob_vec_index_stop  - vec_start;

   data = hypre_VectorData(local_vector);
   if (!value_indices)
   {   
      for (i = local_start; i <= local_stop; i++)
         data[i] = values[i];
   }
   else if (value_indices)
   {   
      for (i = local_start; i <= local_stop; i++)
         data[i] = values[value_indices[i-local_start]];
   }
  
   return ierr;
}

/******************************************************************************
 *
 * hypre_IJVectorAddToLocalComponentsPar
 *
 * adds to a potentially noncontiguous set of IJVectorPar components
 *
 *****************************************************************************/
int
hypre_IJVectorAddToLocalComponentsPar(hypre_IJVector *vector,
                                      int             num_values,
                                      const int      *glob_vec_indices,
                                      const int      *value_indices,
                                      const double   *values            )
{
   int ierr = 0;
   int my_id;
   int i, j, vec_start, vec_stop;
   double *data;

   hypre_ParVector *par_vector = hypre_IJVectorLocalStorage(vector);
   MPI_Comm comm = hypre_IJVectorContext(vector);
   int *partitioning = hypre_ParVectorPartitioning(par_vector);
   hypre_Vector *local_vector = hypre_ParVectorLocalVector(par_vector);

/* If no components are to be retrieved, perform no checking and return */
   if (num_values < 1) return ierr;

   MPI_Comm_rank(comm, &my_id);

/* If par_vector == NULL or partitioning == NULL or local_vector == NULL 
   let user know of catastrophe and exit */

   if (!par_vector)
   {
      printf("par_vector == NULL -- ");
      printf("hypre_IJVectorAddToLocalComponentsPar\n");
      printf("**** Vector storage is either unallocated or orphaned ****\n");
      exit(1);
   }
   if (!partitioning)
   {
      printf("partitioning == NULL -- ");
      printf("hypre_IJVectorAddToLocalComponentsPar\n");
      printf("**** Vector partitioning is either unallocated or orphaned ****\n");
      exit(1);
   }
   if (!local_vector)
   {
      printf("local_vector == NULL -- ");
      printf("hypre_IJVectorAddToLocalComponentsPar\n");
      printf("**** Vector local data is either unallocated or orphaned ****\n");
      exit(1);
   }

   vec_start = partitioning[my_id];
   vec_stop  = partitioning[my_id+1];

   if (vec_start > vec_stop) 
   {
      printf("vec_start > vec_stop -- ");
      printf("hypre_IJVectorAddToLocalComponentsPar\n");
      printf("**** This vector partitioning should not occur ****\n");
      exit(1);
   }

/* Determine whether glob_vec_indices points to local indices only,
   and if not, let user know of catastrophe and exit.
   If glob_vec_indices == NULL, assume that num_values components are to
   be affected in a block starting at vec_start */

   if (glob_vec_indices)
   {
      for (i = 0; i < num_values; i++)
      { 	
	ierr += (glob_vec_indices[i] <  vec_start);
        ierr += (glob_vec_indices[i] >= vec_stop);
      }
   }

   if (ierr)
   {
      printf("glob_vec_indices beyond local range -- ");
      printf("hypre_IJVectorAddToLocalComponentsPar\n");
      printf("**** Glob_vec_indices specified are unusable ****\n");
      exit(1);
   }
    
   data = hypre_VectorData(local_vector);
   if (!value_indices)
   {
      if (glob_vec_indices)
      {
         for (j = 0; j < num_values; j++)
         {
            i = glob_vec_indices[j] - vec_start;
            data[i] += values[j];
         } 
      }
      else
      {
         for (j = 0; j < num_values; j++)
            data[j] += values[j];
      } 
   } 
   else if (value_indices)
   {
      if (glob_vec_indices)
      {
         for (j = 0; j < num_values; j++)
         {
            i = glob_vec_indices[j] - vec_start;
            data[i] += values[value_indices[j]];
         } 
      }
      else
      {
         for (j = 0; j < num_values; j++)
            data[j] += values[value_indices[j]];
      } 
   }  
  
   return ierr;
}

/******************************************************************************
 *
 * hypre_IJVectorAddToLocalComponentsInBlockPar
 *
 * adds to a contiguous set of components in an IJVectorPar
 *
 *****************************************************************************/

int
hypre_IJVectorAddToLocalComponentsInBlockPar(hypre_IJVector *vector,
                                             int             glob_vec_index_start,
                                             int             glob_vec_index_stop,
                                             const int      *value_indices,
                                             const double   *values                ) 
{
   int ierr = 0;
   int my_id;
   int i, vec_start, vec_stop, local_n, local_start, local_stop;
   double *data;

   hypre_ParVector *par_vector = hypre_IJVectorLocalStorage(vector);
   MPI_Comm comm = hypre_IJVectorContext(vector);
   int *partitioning = hypre_ParVectorPartitioning(par_vector);
   hypre_Vector *local_vector = hypre_ParVectorLocalVector(par_vector);

/* If no components are to be affected, perform no checking and return */
   if (glob_vec_index_start > glob_vec_index_stop) return ierr;

   MPI_Comm_rank(comm, &my_id);

/* If par_vector == NULL or partitioning == NULL or local_vector == NULL 
   let user know of catastrophe and exit */

   if (!par_vector)
   {
      printf("par_vector == NULL -- ");
      printf("hypre_IJVectorAddToLocalComponentsInBlockPar\n");
      printf("**** Vector storage is either unallocated or orphaned ****\n");
      exit(1);
   }
   if (!partitioning)
   {
      printf("partitioning == NULL -- ");
      printf("hypre_IJVectorAddToLocalComponentsInBlockPar\n");
      printf("**** Vector partitioning is either unallocated or orphaned ****\n");
      exit(1);
   }
   if (!local_vector)
   {
      printf("local_vector == NULL -- ");
      printf("hypre_IJVectorAddToLocalComponentsInBlockPar\n");
      printf("**** Vector local data is either unallocated or orphaned ****\n");
      exit(1);
   }

   vec_start = partitioning[my_id];
   vec_stop  = partitioning[my_id+1];
 
   if (vec_start > vec_stop) 
   {
      printf("vec_start > vec_stop -- ");
      printf("hypre_IJVectorAddToLocalComponentsInBlockPar\n");
      printf("**** This vector partitioning should not occur ****\n");
      exit(1);
   }
   else if (glob_vec_index_start < vec_start)
   {
      printf("glob_vec_index_start below local range -- ");
      printf("hypre_IJVectorAddToLocalComponentsInBlockPar\n");
      exit(1);
   }
   else if (glob_vec_index_start > vec_stop-1)
   {
      printf("glob_vec_index_start above local range -- ");
      printf("hypre_IJVectorAddToLocalComponentsInBlockPar\n");
      exit(1);
   }
   if (glob_vec_index_stop < vec_start)
   {
      printf("glob_vec_index_stop below local range -- ");
      printf("hypre_IJVectorAddToLocalComponentsInBlockPar\n");
      exit(1);
   }
   else if (glob_vec_index_stop > vec_stop-1)
   {
      printf("glob_vec_index_stop above local range -- ");
      printf("hypre_IJVectorAddToLocalComponentsInBlockPar\n");
      exit(1);
   }

   local_n = vec_stop - vec_start;
   local_start = glob_vec_index_start - vec_start;
   local_stop  = glob_vec_index_stop  - vec_start;

   data = hypre_VectorData(local_vector);
   if (!value_indices)
   {
      for (i = local_start; i <= local_stop; i++)
      {
         data[i] += values[i];
      }
   }
   else if (value_indices)
   {
      for (i = local_start; i <= local_stop; i++)
      {
         data[i] += values[value_indices[i-local_start]];
      }
   }
  
   return ierr;

}

/******************************************************************************
 *
 * hypre_IJVectorAssemblePar
 *
 * assemble the partitioning of the vector
 *
 *****************************************************************************/

int
hypre_IJVectorAssemblePar(hypre_IJVector *vector)
{
   int ierr = 0;
   int my_id;

   hypre_ParVector *par_vector = hypre_IJVectorLocalStorage(vector);
   MPI_Comm comm = hypre_IJVectorContext(vector);
   int *partitioning = hypre_ParVectorPartitioning(par_vector);

   MPI_Comm_rank(comm, &my_id);

   if (!par_vector)
   {
      printf("par_vector == NULL -- ");
      printf("hypre_IJVectorAssemblePar\n");
      printf("**** Vector storage is either unallocated or orphaned ****\n");
      exit(1);
   } 
   if (!partitioning)
   { 
      printf("partitioning == NULL -- ");
      printf("hypre_IJVectorAssemblePar\n");
      printf("**** Vector partitioning is either unallocated or orphaned ****\n");
      exit(1);
   }

   ierr += MPI_Allgather(&partitioning[my_id], 1, MPI_INT,
                         partitioning, 1, MPI_INT, comm);

   return ierr;
}
                                 
/******************************************************************************
 *
 * hypre_IJVectorGetLocalComponentsPar
 *
 * get a potentially noncontiguous set of IJVectorPar components
 *
 *****************************************************************************/

int
hypre_IJVectorGetLocalComponentsPar(hypre_IJVector *vector,
                                    int             num_values,
                                    const int      *glob_vec_indices,
                                    const int      *value_indices,
                                    double         *values            )
{
   int ierr = 0;
   int my_id;
   int i, j, vec_start, vec_stop;
   double *data;

   hypre_ParVector *par_vector = hypre_IJVectorLocalStorage(vector);
   MPI_Comm comm = hypre_IJVectorContext(vector);
   int *partitioning = hypre_ParVectorPartitioning(par_vector);
   hypre_Vector *local_vector = hypre_ParVectorLocalVector(par_vector);

/* If no components are to be retrieved, perform no checking and return */
   if (num_values < 1) return ierr;

   MPI_Comm_rank(comm, &my_id);

/* If par_vector == NULL or partitioning == NULL or local_vector == NULL 
   let user know of catastrophe and exit */

   if (!par_vector)
   {
      printf("par_vector == NULL -- ");
      printf("hypre_IJVectorGetLocalComponentsPar\n");
      printf("**** Vector storage is either unallocated or orphaned ****\n");
      exit(1);
   }
   if (!partitioning)
   {
      printf("partitioning == NULL -- ");
      printf("hypre_IJVectorGetLocalComponentsPar\n");
      printf("**** Vector partitioning is either unallocated or orphaned ****\n");
      exit(1);
   }
   if (!local_vector)
   {
      printf("local_vector == NULL -- ");
      printf("hypre_IJVectorGetLocalComponentsPar\n");
      printf("**** Vector local data is either unallocated or orphaned ****\n");
      exit(1);
   }

   vec_start = partitioning[my_id];
   vec_stop  = partitioning[my_id+1];

   if (vec_start > vec_stop) 
   {
      printf("vec_start > vec_stop -- ");
      printf("hypre_IJVectorGetLocalComponentsPar\n");
      printf("**** This vector partitioning should not occur ****\n");
      exit(1);
   }

/* Determine whether glob_vec_indices points to local indices only,
   and if not, let user know of catastrophe and exit.
   If glob_vec_indices == NULL, assume that num_values components are to be
   retrieved from block starting at vec_start */

   if (glob_vec_indices)
   {
      for (i = 0; i < num_values; i++)
      { 	
	ierr += (glob_vec_indices[i] <  vec_start);
        ierr += (glob_vec_indices[i] >= vec_stop);
      }
   }

   if (ierr)
   {
      printf("glob_vec_indices beyond local range -- ");
      printf("hypre_IJVectorGetLocalComponentsPar\n");
      printf("**** Glob_vec_indices specified are unusable ****\n");
      exit(1);
   }
    
   data = hypre_VectorData(local_vector);
   if (!value_indices)
   {
      if (glob_vec_indices)
      {
         for (j = 0; j < num_values; j++)
         {
            i = glob_vec_indices[j] - vec_start;
            values[j] = data[i];
         }
      }
      else
      {
         for (j = 0; j < num_values; j++)
         {
            values[j] = data[j];
         }
      }
   } 
   else if (value_indices)
   {
      if (glob_vec_indices)
      {
         for (j = 0; j < num_values; j++)
         {
            i = glob_vec_indices[j] - vec_start;
            values[value_indices[j]] = data[i];
         }
      }
      else
      {
         for (j = 0; j < num_values; j++)
         {
            values[value_indices[j]] = data[j];
         }
      }
   } 

   return ierr;
}

/******************************************************************************
 *
 * hypre_IJVectorGetLocalComponentsInBlockPar
 *
 * gets a contiguous set of components in an IJVectorPar
 *
 *****************************************************************************/

int
hypre_IJVectorGetLocalComponentsInBlockPar(hypre_IJVector *vector,
                                           int             glob_vec_index_start,
                                           int             glob_vec_index_stop,
                                           const int      *value_indices,
                                           double         *values                )
{
   int ierr = 0;
   int my_id;
   int i, vec_start, vec_stop, local_n, local_start, local_stop;
   double *data;

   hypre_ParVector *par_vector = hypre_IJVectorLocalStorage(vector);
   MPI_Comm comm = hypre_IJVectorContext(vector);
   int *partitioning = hypre_ParVectorPartitioning(par_vector);
   hypre_Vector *local_vector = hypre_ParVectorLocalVector(par_vector);

/* If no components are to be retrieved, perform no checking and return */
   if (glob_vec_index_start > glob_vec_index_stop) return ierr;

   MPI_Comm_rank(comm, &my_id);

/* If par_vector == NULL or partitioning == NULL or local_vector == NULL 
   let user know of catastrophe and exit */

   if (!par_vector)
   {
      printf("par_vector == NULL -- ");
      printf("hypre_IJVectorGetLocalComponentsInBlockPar\n");
      printf("**** Vector storage is either unallocated or orphaned ****\n");
      exit(1);
   }
   if (!partitioning)
   {
      printf("partitioning == NULL -- ");
      printf("hypre_IJVectorGetLocalComponentsInBlockPar\n");
      printf("**** Vector partitioning is either unallocated or orphaned ****\n");
      exit(1);
   }
   if (!local_vector)
   {
      printf("local_vector == NULL -- ");
      printf("hypre_IJVectorGetLocalComponentsInBlockPar\n");
      printf("**** Vector local data is either unallocated or orphaned ****\n");
      exit(1);
   }

   vec_start = partitioning[my_id];
   vec_stop  = partitioning[my_id+1];

   if (vec_start > vec_stop) 
   {
      printf("vec_start > vec_stop -- ");
      printf("hypre_IJVectorGetLocalComponentsInBlockPar\n");
      printf("**** This vector partitioning should not occur ****\n");
      exit(1);
   }
   else if (glob_vec_index_start < vec_start)
   {
      printf("glob_vec_index_start below local range -- ");
      printf("hypre_IJVectorGetLocalComponentsInBlockPar\n");
      exit(1);
   }
   else if (glob_vec_index_start > vec_stop-1)
   {
      printf("glob_vec_index_start above local range -- ");
      printf("hypre_IJVectorGetLocalComponentsInBlockPar\n");
      exit(1);
   }
   if (glob_vec_index_stop < vec_start)
   {
      printf("glob_vec_index_stop below local range -- ");
      printf("hypre_IJVectorGetLocalComponentsInBlockPar\n");
      exit(1);
   }
   else if (glob_vec_index_stop > vec_stop-1)
   {
      printf("glob_vec_index_stop above local range -- ");
      printf("hypre_IJVectorGetLocalComponentsInBlockPar\n");
      exit(1);
   }

   local_n = vec_stop - vec_start;
   local_start = glob_vec_index_start - vec_start;
   local_stop  = glob_vec_index_stop  - vec_start;

   data = hypre_VectorData(local_vector);
   if (!value_indices)
   {
      for (i = local_start; i <= local_stop; i++)
         values[i] = data[i];
   }
   else if (value_indices)
   {
      for (i = local_start; i <= local_stop; i++)
         values[value_indices[i-local_start]] = data[i];
   } 
  
   return ierr;

}
