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
 * hypre_NewIJVectorPar
 *
 * creates ParVector if necessary,
 *
 *****************************************************************************/
int
hypre_NewIJVectorPar(hypre_IJVector *vector)
{
   MPI_Comm comm = hypre_IJVectorContext(vector);
   int global_n = hypre_IJVectorN(vector); 
   hypre_ParVector *par_vector = hypre_IJVectorLocalStorage(vector);
   int ierr = 0;

   int my_id;
   MPI_Comm_rank(comm, &my_id);

   if (!par_vector)
   {
      hypre_IJVectorLocalStorage(vector) = hypre_CreateParVector(comm,
               global_n,NULL); 
   };

   return ierr;
}

/******************************************************************************
 *
 * hypre_InitializeIJVectorPar
 *
 * initializes ParVector of IJVectorPar
 *
 *****************************************************************************/

int
hypre_InitializeIJVectorPar(hypre_IJVector *vector)
{
   int ierr = 0;
   hypre_ParVector *par_vector = hypre_IJVectorLocalStorage(vector);
   int *partitioning = hypre_ParVectorPartitioning(par_vector);
   hypre_Vector *local_vector = hypre_ParVectorLocalVector(par_vector);
   int local_num_rows;
   int my_id;
   MPI_Comm  comm = hypre_IJVectorContext(vector);

   MPI_Comm_rank(comm,&my_id);
  
   hypre_VectorSize(local_vector) = partitioning[my_id+1] -
                                    partitioning[my_id];

   ierr += hypre_InitializeParVector(par_vector);

   return ierr;
}

/******************************************************************************
 *
 * hypre_SetIJVectorParLocalComponents
 *
 * inserts a potentially noncontiguous set of rows into an IJVectorPar
 *
 *****************************************************************************/
int
hypre_SetIJVectorParLocalComponents(hypre_IJVector *vector,
                                    int             num_values,
                                    int            *glob_vec_indices,
                                    double          value       )
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

   vec_start = partitioning[my_id];
   vec_stop  = partitioning[my_id+1];
   
/* Determine whether *glob_vec_indices points to local indices only */
   for (i = 0; i < num_values; i++)
   { ierr += (glob_vec_indices[i] >= vec_start);
     ierr += (glob_vec_indices[i] <  vec_stop);
   }
    
   if (!ierr)
   {
      data = hypre_VectorData( local_vector );
      for (j = 0; j < num_values; j++)
      {
         i = glob_vec_indices[j] - vec_start;
         data[i] = value;
      };
   }; 
  
   return ierr;
}

/******************************************************************************
 *
 * hypre_SetIJVectorParLocalComponentsInBlock
 *
 * inserts a potentially noncontiguous set of rows into an IJVectorPar
 *
 *****************************************************************************/
int
hypre_SetIJVectorParLocalComponentsInBlock(hypre_IJVector *vector,
                                           int             glob_vec_index_start,
                                           int             glob_vec_index_stop,
                                           double          value    )
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

   vec_start = partitioning[my_id];
   vec_stop  = partitioning[my_id+1];
   
   local_n = vec_stop - vec_start;
   local_start = glob_vec_index_start - vec_start;
   local_stop  = glob_vec_index_stop  - vec_start;

   if (local_start > local_stop) ++ierr;
   if (local_start < 0 || local_start >= local_n) ++ierr;
   if (local_stop >= local_n) ++ierr;

   if (!ierr)
   {
      data = hypre_VectorData(local_vector);
      for (i = 0; i <= local_stop - local_start; i++)
      {
         data[i] = value;
      };
   };
  
   return ierr;
}

/******************************************************************************
 *
 * hypre_InsertIJVectorParLocalComponents
 *
 * inserts a potentially noncontiguous set of rows into an IJVectorPar
 *
 *****************************************************************************/
int
hypre_InsertIJVectorParLocalComponents(hypre_IJVector *vector,
                                       int             num_values,
                                       int            *glob_vec_indices,
                                       int            *value_indices,
                                       double         *values      )
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

   vec_start = partitioning[my_id];
   vec_stop  = partitioning[my_id+1];
  
/* Determine whether *glob_vec_indices points to local indices only */
   for (i = 0; i < num_values; i++)
   { ierr += (glob_vec_indices[i] >= vec_start);
     ierr += (glob_vec_indices[i] <  vec_stop);
   };
    
   data = hypre_VectorData(local_vector);
   if (!ierr && !value_indices)
   {
       for (j = 0; j < num_values; j++)
       {
          i = glob_vec_indices[j] - vec_start;
          data[i] = values[j];
       };
   } 
   else if (!ierr && value_indices)
   {
       for (j = 0; j < num_values; j++)
       {
          i = glob_vec_indices[j] - vec_start;
          data[i] = values[value_indices[j]];
       };
   }; 
  
   return ierr;
}

/******************************************************************************
 *
 * hypre_InsertIJVectorParLocalComponentsInBlock
 *
 * inserts a contiguous set of components into an IJVectorPar
 *
 *****************************************************************************/
int
hypre_InsertIJVectorParLocalComponentsInBlock(hypre_IJVector *vector,
                                              int             glob_vec_index_start,
                                              int             glob_vec_index_stop,
                                              int            *value_indices,
                                              double         *values   )
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
 * hypre_AddToIJVectorParLocalComponents
 *
 * adds to a potentially noncontiguous set of IJVectorPar components
 *
 *****************************************************************************/
int
hypre_AddToIJVectorParLocalComponents(hypre_IJVector *vector,
                                      int             num_values,
                                      int            *glob_vec_indices,
                                      int            *value_indices,
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

   vec_start = partitioning[my_id];
   vec_stop  = partitioning[my_id+1];

/* Determine whether *glob_vec_indices points to local indices only */
   for (i = 0; i < num_values; i++)
   { ierr += (glob_vec_indices[i] >= vec_start);
     ierr += (glob_vec_indices[i] <  vec_stop);
   };
    
   data = hypre_VectorData(local_vector);
   if (!ierr && !value_indices)
   {
      for (j = 0; j < num_values; j++)
      {
         i = glob_vec_indices[j] - vec_start;
         data[i] += values[i];
      };
   } 
   else if (!ierr && value_indices)
   {
      for (j = 0; j < num_values; j++)
      {
         i = glob_vec_indices[j] - vec_start;
         data[i] += values[value_indices[i]];
      };
   }; 
  
   return ierr;
}

/******************************************************************************
 *
 * hypre_AddToIJVectorParLocalComponentsInBlock
 *
 * adds to a contiguous set of components in an IJVectorPar
 *
 *****************************************************************************/

int
hypre_AddToIJVectorParLocalComponentsInBlock(hypre_IJVector *vector,
                                             int             glob_vec_index_start,
                                             int             glob_vec_index_stop,
                                             int            *value_indices,
                                             double         *values   )
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
 * hypre_GetIJVectorParLocalComponents
 *
 * get a potentially noncontiguous set of IJVectorPar components
 *
 *****************************************************************************/
int
hypre_GetIJVectorParLocalComponents(hypre_IJVector *vector,
                                    int             num_values,
                                    int            *glob_vec_indices,
                                    int            *value_indices,
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

   vec_start = partitioning[my_id];
   vec_stop  = partitioning[my_id+1];

/* Determine whether *glob_vec_indices points to local indices only */
   for (i = 0; i < num_values; i++)
   { ierr += (glob_vec_indices[i] >= vec_start);
     ierr += (glob_vec_indices[i] <  vec_stop);
   };
    
   data = hypre_VectorData(local_vector);
   if (!ierr && !value_indices)
   {
      for (j = 0; j < num_values; j++)
      {
         i = glob_vec_indices[j] - vec_start;
         values[i] = data[i];
      };
   } 
   else if (!ierr && value_indices)
   {
      for (j = 0; j < num_values; j++)
      {
         i = glob_vec_indices[j] - vec_start;
         values[value_indices[i]] = data[i];
      };
   }; 
  
   return ierr;
}

/******************************************************************************
 *
 * hypre_GetIJVectorParLocalComponentsInBlock
 *
 * gets a contiguous set of components in an IJVectorPar
 *
 *****************************************************************************/

int
hypre_GerIJVectorParLocalComponentsInBlock(hypre_IJVector *vector,
                                           int             glob_vec_index_start,
                                           int             glob_vec_index_stop,
                                           int            *value_indices,
                                           double         *values   )
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
/******************************************************************************
 *
 * hypre_DistributeIJVectorPar
 *
 * takes an IJVector generated for one processor and distributes it
 * across many processors according to vec_starts,
 * if vec_starts is NULL, it distributes them evenly.
 *
 *****************************************************************************/
int
hypre_DistributeIJVectorPar(hypre_IJVector *vector,
			    int	           *vec_starts)
{
   int ierr = 0;

   hypre_ParVector *old_vector = hypre_IJVectorLocalStorage(vector);
   hypre_ParVector *par_vector;
   hypre_Vector *local_vector = hypre_ParVectorLocalVector(par_vector);
   par_vector = hypre_VectorToParVector(hypre_ParVectorComm(old_vector),
		                        hypre_ParVectorLocalVector(old_vector),
                                        vec_starts);
   ierr = hypre_DestroyParVector(old_vector);
   hypre_IJVectorLocalStorage(vector) = par_vector;

   return ierr;
}

/******************************************************************************
 *
 * hypre_FreeIJVectorPar
 *
 * frees ParVector local storage of an IJVectorPar 
 *
 *****************************************************************************/
int
hypre_FreeIJVectorPar(hypre_IJVector *vector)
{
   return hypre_DestroyParVector(hypre_IJVectorLocalStorage(vector));
}

