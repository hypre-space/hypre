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
 
#include "IJ_matrix_vector.h"
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
   };

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

   if (vec_start_this_proc >= vec_start_next_proc) ++ierr;

   if (!ierr)
   {
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
   };

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
   
   if (!old_vector) ++ierr;

   par_vector = hypre_VectorToParVector(hypre_ParVectorComm(old_vector),
		                        hypre_ParVectorLocalVector(old_vector),
                                        (int *)vec_starts);
   if (!par_vector) ++ierr;

   ierr = hypre_ParVectorDestroy(old_vector);

   hypre_IJVectorLocalStorage(vector) = par_vector;

   return ierr;
}

/******************************************************************************
 *
 * hypre_IJVectorZeroLocalComponentsPar
 *
 * inserts a potentially noncontiguous set of rows into an IJVectorPar
 *
 *****************************************************************************/
int
hypre_IJVectorZeroLocalComponentsPar(hypre_IJVector *vector)
{
   int ierr = 0;
   int my_id, num_procs;
   int i, vec_start, vec_stop;
   double *data;

   hypre_ParVector *par_vector = hypre_IJVectorLocalStorage(vector);
   MPI_Comm comm = hypre_IJVectorContext(vector);
   int *partitioning = hypre_ParVectorPartitioning(par_vector);
   hypre_Vector *local_vector = hypre_ParVectorLocalVector(par_vector);

   MPI_Comm_rank(comm, &my_id);

   if (!par_vector) ++ierr;
   if (!partitioning) ++ierr;
   if (!local_vector) ++ierr;

   vec_start = partitioning[my_id];
   vec_stop  = partitioning[my_id+1];
   
   if (vec_start > vec_stop) ++ierr;

   if (!ierr)
   {
      data = hypre_VectorData( local_vector );
      for (i = 0; i < vec_stop - vec_start; i++)
      {
         data[i] = 0.;
      };
   }; 
  
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
   int my_id, num_procs;
   int i, j, vec_start, vec_stop;
   double *data;

   hypre_ParVector *par_vector = hypre_IJVectorLocalStorage(vector);
   MPI_Comm comm = hypre_IJVectorContext(vector);
   int *partitioning = hypre_ParVectorPartitioning(par_vector);
   hypre_Vector *local_vector = hypre_ParVectorLocalVector(par_vector);

   MPI_Comm_rank(comm, &my_id);

   if (!par_vector) ++ierr;
   if (!partitioning) ++ierr;
   if (!local_vector) ++ierr;

   vec_start = partitioning[my_id];
   vec_stop  = partitioning[my_id+1];
  
/* Determine whether *glob_vec_indices points to local indices only */
   for (i = 0; i < num_values; i++)
   { ierr += !(glob_vec_indices[i] >= vec_start);
     ierr += !(glob_vec_indices[i] <  vec_stop);
   };
    
   data = hypre_VectorData(local_vector);
   if (!ierr && !value_indices)
   {
       if (glob_vec_indices)
       {
          for (j = 0; j < num_values; j++)
          {
             i = glob_vec_indices[j] - vec_start;
             data[i] = values[j];
          };
        }
        else
        {
          for (j = 0; j < num_values; j++)
          {
             data[j] = values[j];
          };
        };
   } 
   else if (!ierr && value_indices)
   {
       if (glob_vec_indices)
       {
          for (j = 0; j < num_values; j++)
          {
             i = glob_vec_indices[j] - vec_start;
             data[i] = values[value_indices[j]];
          };
       }
       else
       {
          for (j = 0; j < num_values; j++)
          {
             data[j] = values[value_indices[j]];
          };
       };
   }; 
  
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
   int my_id, num_procs;
   int i, vec_start, vec_stop, local_n, local_start, local_stop;
   double *data;

   hypre_ParVector *par_vector = hypre_IJVectorLocalStorage(vector);
   MPI_Comm comm = hypre_IJVectorContext(vector);
   int *partitioning = hypre_ParVectorPartitioning(par_vector);
   hypre_Vector *local_vector = hypre_ParVectorLocalVector(par_vector);

   MPI_Comm_rank(comm, &my_id);

   if (!par_vector) ++ierr;
   if (!partitioning) ++ierr;
   if (!local_vector) ++ierr;

   vec_start = partitioning[my_id];
   vec_stop  = partitioning[my_id+1];

   local_n = vec_stop - vec_start;
   local_start = glob_vec_index_start - vec_start;
   local_stop  = glob_vec_index_stop  - vec_start;

   if (local_start > local_stop) ++ierr;
   if (local_start < 0 || local_start >= local_n) ++ierr;
   if (local_stop >= local_n) ++ierr;

   data = hypre_VectorData(local_vector);
   if (!ierr && !value_indices)
   {   
       for (i = 0; i <= local_stop - local_start; i++)
       {
          data[i] = values[i];
       };
   }
   else if (!ierr && value_indices)
   {   
       for (i = 0; i <= local_stop - local_start; i++)
       {
          data[i] = values[value_indices[i]];
       };
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
   int my_id, num_procs;
   int i, j, vec_start, vec_stop;
   double *data;

   hypre_ParVector *par_vector = hypre_IJVectorLocalStorage(vector);
   MPI_Comm comm = hypre_IJVectorContext(vector);
   int *partitioning = hypre_ParVectorPartitioning(par_vector);
   hypre_Vector *local_vector = hypre_ParVectorLocalVector(par_vector);

   MPI_Comm_rank(comm, &my_id);

   if (!par_vector) ++ierr;
   if (!partitioning) ++ierr;
   if (!local_vector) ++ierr;

   vec_start = partitioning[my_id];
   vec_stop  = partitioning[my_id+1];

/* Determine whether *glob_vec_indices points to local indices only */
   for (i = 0; i < num_values; i++)
   { ierr += !(glob_vec_indices[i] >= vec_start);
     ierr += !(glob_vec_indices[i] <  vec_stop);
   };
    
   data = hypre_VectorData(local_vector);
   if (!ierr && !value_indices)
   {
      if (glob_vec_indices)
      {
         for (j = 0; j < num_values; j++)
         {
            i = glob_vec_indices[j] - vec_start;
            data[i] += values[j];
         };
      }
      else
      {
         for (j = 0; j < num_values; j++)
         {
            data[j] += values[j];
         };
      };
   } 
   else if (!ierr && value_indices)
   {
      if (glob_vec_indices)
      {
         for (j = 0; j < num_values; j++)
         {
            i = glob_vec_indices[j] - vec_start;
            data[i] += values[value_indices[j]];
         };
      }
      else
      {
         for (j = 0; j < num_values; j++)
         {
            data[j] += values[value_indices[j]];
         };
      };
   }; 
  
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
   int i, j, vec_start, vec_stop, local_n, local_start, local_stop;
   double *data;

   hypre_ParVector *par_vector = hypre_IJVectorLocalStorage(vector);
   MPI_Comm comm = hypre_IJVectorContext(vector);
   int *partitioning = hypre_ParVectorPartitioning(par_vector);
   hypre_Vector *local_vector = hypre_ParVectorLocalVector(par_vector);

   MPI_Comm_rank(comm, &my_id);

   if (!par_vector) ++ierr;
   if (!partitioning) ++ierr;
   if (!local_vector) ++ierr;

   vec_start = partitioning[my_id];
   vec_stop  = partitioning[my_id+1];

   local_n = vec_stop - vec_start;
   local_start = glob_vec_index_start - vec_start;
   local_stop  = glob_vec_index_stop  - vec_start;

   if (local_start > local_stop) ++ierr; 
   if (local_start < 0 || local_start >= local_n) ++ierr;
   if (local_stop >= local_n) ++ierr;

   data = hypre_VectorData(local_vector);
   if (!ierr && !value_indices)
   {
      for (i = 0; i <= local_stop - local_start; i++)
      {
         data[i] += values[i];
      };
   }
   else if (!ierr && value_indices)
   {
      for (i = 0; i <= local_stop - local_start; i++)
      {
         data[i] += values[value_indices[i]];
      };
   };
  
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

   if (!par_vector) ++ierr;
   if (!partitioning) ++ierr;

   ierr = MPI_Allgather(&partitioning[my_id], 1, MPI_INT,
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
   int my_id, num_procs;
   int i, j, vec_start, vec_stop;
   double *data;

   hypre_ParVector *par_vector = hypre_IJVectorLocalStorage(vector);
   MPI_Comm comm = hypre_IJVectorContext(vector);
   int *partitioning = hypre_ParVectorPartitioning(par_vector);
   hypre_Vector *local_vector = hypre_ParVectorLocalVector(par_vector);

   MPI_Comm_rank(comm, &my_id);

   if (!par_vector) ++ierr;
   if (!partitioning) ++ierr;
   if (!local_vector) ++ierr;

   vec_start = partitioning[my_id];
   vec_stop  = partitioning[my_id+1];

/* Determine whether *glob_vec_indices points to local indices only */
   for (i = 0; i < num_values; i++)
   { ierr += !(glob_vec_indices[i] >= vec_start);
     ierr += !(glob_vec_indices[i] <  vec_stop);
   };
    
   data = hypre_VectorData(local_vector);
   if (!ierr && !value_indices)
   {
      if (glob_vec_indices)
      {
         for (j = 0; j < num_values; j++)
         {
            i = glob_vec_indices[j] - vec_start;
            values[j] = data[i];
         };
      }
      else
      {
         for (j = 0; j < num_values; j++)
         {
            values[j] = data[j];
         };
      };
   } 
   else if (!ierr && value_indices)
   {
      if (glob_vec_indices)
      {
         for (j = 0; j < num_values; j++)
         {
            i = glob_vec_indices[j] - vec_start;
            values[value_indices[j]] = data[i];
         };
      }
      else
      {
         for (j = 0; j < num_values; j++)
         {
            values[value_indices[j]] = data[j];
         };
      };
   }; 

  
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
   int my_id, num_procs;
   int i, j, vec_start, vec_stop, local_n, local_start, local_stop;
   double *data;

   hypre_ParVector *par_vector = hypre_IJVectorLocalStorage(vector);
   MPI_Comm comm = hypre_IJVectorContext(vector);
   int *partitioning = hypre_ParVectorPartitioning(par_vector);
   hypre_Vector *local_vector = hypre_ParVectorLocalVector(par_vector);

   MPI_Comm_rank(comm, &my_id);

   if (!par_vector) ++ierr;
   if (!partitioning) ++ierr;
   if (!local_vector) ++ierr;

   vec_start = partitioning[my_id];
   vec_stop  = partitioning[my_id+1];

   local_n = vec_stop - vec_start;
   local_start = glob_vec_index_start - vec_start;
   local_stop  = glob_vec_index_stop  - vec_start;

   if (local_start > local_stop) ++ierr; 
   if (local_start < 0 || local_start >= local_n) ++ierr;
   if (local_stop >= local_n) ++ierr;

   data = hypre_VectorData(local_vector);
   if (!ierr && !value_indices)
   {
      for (i = 0; i <= local_stop - local_start; i++)
      {
         values[i] = data[i];
      };
   }
   else if (!ierr && value_indices)
   {
      for (i = 0; i <= local_stop - local_start; i++)
      {
         values[value_indices[i]] = data[i];
      };
   };
  
   return ierr;

}
