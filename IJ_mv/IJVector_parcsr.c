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
 * creates ParVector if necessary, and leaves a pointer to it as the
 * hypre_IJVector object
 *
 *****************************************************************************/
int
hypre_IJVectorCreatePar(hypre_IJVector *vector, int *IJpartitioning)
{
   MPI_Comm comm = hypre_IJVectorComm(vector);

#if 0
   if (par_vector)
   {
      printf("Creating a ParVector object will orphan an old one -- ");
      printf("hypreIJVectorCreatePar\n");
      exit(1);
   }
#endif

   int num_procs, jmin, global_n, *partitioning, j;
   MPI_Comm_size(comm, &num_procs);

   jmin = IJpartitioning[0];
   global_n = IJpartitioning[num_procs] - jmin;
   
   partitioning = hypre_CTAlloc(int, num_procs+1); 

/* Shift to zero-based partitioning for ParVector object */
   for (j = 0; j < num_procs+1; j++) 
      partitioning[j] = IJpartitioning[j] - jmin;

   hypre_IJVectorObject(vector) = hypre_ParVectorCreate(comm,
            global_n, (int *) partitioning); 

   return 0;
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
   return hypre_ParVectorDestroy(hypre_IJVectorObject(vector));
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
   hypre_ParVector *par_vector = hypre_IJVectorObject(vector);
   int *partitioning = hypre_ParVectorPartitioning(par_vector);
   hypre_Vector *local_vector = hypre_ParVectorLocalVector(par_vector);
   int my_id;
   MPI_Comm  comm = hypre_IJVectorComm(vector);

   MPI_Comm_rank(comm,&my_id);
  
   if (!partitioning)
   {
      printf("No ParVector partitioning for initialization -- ");
      printf("hypre_IJVectorInitializePar\n"); 
      exit(1);
   }

   hypre_VectorSize(local_vector) = partitioning[my_id+1] -
                                    partitioning[my_id];

   return( hypre_ParVectorInitialize(par_vector) );
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

   hypre_ParVector *old_vector = hypre_IJVectorObject(vector);
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

   hypre_IJVectorObject(vector) = par_vector;

   return ierr;
}

/******************************************************************************
 *
 * hypre_IJVectorZeroValuesPar
 *
 * zeroes all local components of an IJVectorPar
 *
 *****************************************************************************/
int
hypre_IJVectorZeroValuesPar(hypre_IJVector *vector)
{
   int ierr = 0;
   int my_id;
   int i, vec_start, vec_stop;
   double *data;

   hypre_ParVector *par_vector = hypre_IJVectorObject(vector);
   MPI_Comm comm = hypre_IJVectorComm(vector);
   int *partitioning = hypre_ParVectorPartitioning(par_vector);
   hypre_Vector *local_vector = hypre_ParVectorLocalVector(par_vector);

   MPI_Comm_rank(comm, &my_id);

/* If par_vector == NULL or partitioning == NULL or local_vector == NULL 
   let user know of catastrophe and exit */

   if (!par_vector)
   {
      printf("par_vector == NULL -- ");
      printf("hypre_IJVectorZeroValuesPar\n");
      printf("**** Vector storage is either unallocated or orphaned ****\n");
      exit(1);
   }
   if (!partitioning)
   {
      printf("partitioning == NULL -- ");
      printf("hypre_IJVectorZeroValuesPar\n");
      printf("**** Vector partitioning is either unallocated or orphaned ****\n");
      exit(1);
   }
   if (!local_vector)
   {
      printf("local_vector == NULL -- ");
      printf("hypre_IJVectorZeroValuesPar\n");
      printf("**** Vector local data is either unallocated or orphaned ****\n");
      exit(1);
   }

   vec_start = partitioning[my_id];
   vec_stop  = partitioning[my_id+1];
   
   if (vec_start > vec_stop) 
   {
      printf("vec_start > vec_stop -- ");
      printf("hypre_IJVectorZeroValuesPar\n");
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
 * hypre_IJVectorSetValuesPar
 *
 * sets a potentially noncontiguous set of components of an IJVectorPar
 *
 *****************************************************************************/
int
hypre_IJVectorSetValuesPar(hypre_IJVector *vector,
                                    int             num_values,
                                    const int      *indices,
                                    const double   *values            )
{
   int ierr = 0;
   int my_id;
   int i, j, vec_start, vec_stop;
   double *data;

   int *IJpartitioning = hypre_IJVectorPartitioning(vector);
   hypre_ParVector *par_vector = hypre_IJVectorObject(vector);
   MPI_Comm comm = hypre_IJVectorComm(vector);
   hypre_Vector *local_vector = hypre_ParVectorLocalVector(par_vector);

/* If no components are to be set, perform no checking and return */
   if (num_values < 1) return 0;

   MPI_Comm_rank(comm, &my_id);

/* If par_vector == NULL or partitioning == NULL or local_vector == NULL 
   let user know of catastrophe and exit */

   if (!par_vector)
   {
      printf("par_vector == NULL -- ");
      printf("hypre_IJVectorSetValuesPar\n");
      printf("**** Vector storage is either unallocated or orphaned ****\n");
      exit(1);
   }
   if (!IJpartitioning)
   {
      printf("IJpartitioning == NULL -- ");
      printf("hypre_IJVectorSetValuesPar\n");
      printf("**** IJVector partitioning is either unallocated or orphaned ****\n");
      exit(1);
   }
   if (!local_vector)
   {
      printf("local_vector == NULL -- ");
      printf("hypre_IJVectorSetValuesPar\n");
      printf("**** Vector local data is either unallocated or orphaned ****\n");
      exit(1);
   }

   vec_start = IJpartitioning[my_id];
   vec_stop  = IJpartitioning[my_id+1];
  
   if (vec_start > vec_stop) 
   {
      printf("vec_start > vec_stop -- ");
      printf("hypre_IJVectorSetValuesPar\n");
      printf("**** This vector partitioning should not occur ****\n");
      exit(1);
   }

/* Determine whether indices points to local indices only,
   and if not, let user know of catastrophe and exit.
   If indices == NULL, assume that num_values components are to be
   set in a block starting at vec_start */

   if (indices)
   {
      for (i = 0; i < num_values; i++)
      { 	
	ierr += (indices[i] <  vec_start);
        ierr += (indices[i] >= vec_stop);
      }
   }

   if (ierr)
   {
      printf("indices beyond local range -- ");
      printf("hypre_IJVectorSetValuesPar\n");
      printf("**** Indices specified are unusable ****\n");
      exit(1);
   }
    
   data = hypre_VectorData(local_vector);

   if (indices)
   {
      for (j = 0; j < num_values; j++)
      {
         i = indices[j] - vec_start;
         data[i] = values[j];
      } 
   }
   else
   {
      for (j = 0; j < num_values; j++)
         data[j] = values[j];
   } 
  
   return ierr;
}

/******************************************************************************
 *
 * hypre_IJVectorAddToValuesPar
 *
 * adds to a potentially noncontiguous set of IJVectorPar components
 *
 *****************************************************************************/
int
hypre_IJVectorAddToValuesPar(hypre_IJVector *vector,
                             int             num_values,
                             const int      *indices,
                             const double   *values      )
{
   int ierr = 0;
   int my_id;
   int i, j, vec_start, vec_stop;
   double *data;

   int *IJpartitioning = hypre_IJVectorPartitioning(vector);
   hypre_ParVector *par_vector = hypre_IJVectorObject(vector);
   MPI_Comm comm = hypre_IJVectorComm(vector);
   hypre_Vector *local_vector = hypre_ParVectorLocalVector(par_vector);

/* If no components are to be retrieved, perform no checking and return */
   if (num_values < 1) return 0;

   MPI_Comm_rank(comm, &my_id);

/* If par_vector == NULL or partitioning == NULL or local_vector == NULL 
   let user know of catastrophe and exit */

   if (!par_vector)
   {
      printf("par_vector == NULL -- ");
      printf("hypre_IJVectorAddToValuesPar\n");
      printf("**** Vector storage is either unallocated or orphaned ****\n");
      exit(1);
   }
   if (!IJpartitioning)
   {
      printf("IJpartitioning == NULL -- ");
      printf("hypre_IJVectorAddToValuesPar\n");
      printf("**** IJVector partitioning is either unallocated or orphaned ****\n");
      exit(1);
   }
   if (!local_vector)
   {
      printf("local_vector == NULL -- ");
      printf("hypre_IJVectorAddToValuesPar\n");
      printf("**** Vector local data is either unallocated or orphaned ****\n");
      exit(1);
   }

   vec_start = IJpartitioning[my_id];
   vec_stop  = IJpartitioning[my_id+1];

   if (vec_start > vec_stop) 
   {
      printf("vec_start > vec_stop -- ");
      printf("hypre_IJVectorAddToValuesPar\n");
      printf("**** This vector partitioning should not occur ****\n");
      exit(1);
   }

/* Determine whether indices points to local indices only,
   and if not, let user know of catastrophe and exit.
   If indices == NULL, assume that num_values components are to
   be affected in a block starting at vec_start */

   if (indices)
   {
      for (i = 0; i < num_values; i++)
      { 	
	ierr += (indices[i] <  vec_start);
        ierr += (indices[i] >= vec_stop);
      }
   }

   if (ierr)
   {
      printf("indices beyond local range -- ");
      printf("hypre_IJVectorAddToValuesPar\n");
      printf("**** Indices specified are unusable ****\n");
      exit(1);
   }
    
   data = hypre_VectorData(local_vector);

   if (indices)
   {
      for (j = 0; j < num_values; j++)
      {
         i = indices[j] - vec_start;
         data[i] += values[j];
      } 
   }
   else
   {
      for (j = 0; j < num_values; j++)
         data[j] += values[j];
   } 
  
   return ierr;
}

/******************************************************************************
 *
 * hypre_IJVectorAssemblePar
 *
 * currently tests existence of of ParVector object and its partitioning
 *
 *****************************************************************************/
int
hypre_IJVectorAssemblePar(hypre_IJVector *vector)
{
   int *IJpartitioning = hypre_IJVectorPartitioning(vector);
   hypre_ParVector *par_vector = hypre_IJVectorObject(vector);
   int *partitioning = hypre_ParVectorPartitioning(par_vector);

   if (!par_vector)
   {
      printf("par_vector == NULL -- ");
      printf("hypre_IJVectorAssemblePar\n");
      printf("**** Vector storage is either unallocated or orphaned ****\n");
      exit(1);
   } 
   if (!IJpartitioning)
   { 
      printf("IJpartitioning == NULL -- ");
      printf("hypre_IJVectorAssemblePar\n");
      printf("**** IJVector partitioning is either unallocated or orphaned ****\n");
      exit(1);
   }
   if (!partitioning)
   { 
      printf("partitioning == NULL -- ");
      printf("hypre_IJVectorAssemblePar\n");
      printf("**** ParVector partitioning is either unallocated or orphaned ****\n");
      exit(1);
   }

   return 0;
}
                                 
/******************************************************************************
 *
 * hypre_IJVectorGetValuesPar
 *
 * get a potentially noncontiguous set of IJVectorPar components
 *
 *****************************************************************************/
int
hypre_IJVectorGetValuesPar(hypre_IJVector *vector,
                           int             num_values,
                           const int      *indices,
                           double         *values      )
{
   int ierr = 0;
   int my_id;
   int i, j, vec_start, vec_stop;
   double *data;

   int *IJpartitioning = hypre_IJVectorPartitioning(vector);
   hypre_ParVector *par_vector = hypre_IJVectorObject(vector);
   MPI_Comm comm = hypre_IJVectorComm(vector);
   hypre_Vector *local_vector = hypre_ParVectorLocalVector(par_vector);

/* If no components are to be retrieved, perform no checking and return */
   if (num_values < 1) return 0;

   MPI_Comm_rank(comm, &my_id);

/* If par_vector == NULL or partitioning == NULL or local_vector == NULL 
   let user know of catastrophe and exit */

   if (!par_vector)
   {
      printf("par_vector == NULL -- ");
      printf("hypre_IJVectorGetValuesPar\n");
      printf("**** Vector storage is either unallocated or orphaned ****\n");
      exit(1);
   }
   if (!IJpartitioning)
   {
      printf("IJpartitioning == NULL -- ");
      printf("hypre_IJVectorGetValuesPar\n");
      printf("**** IJVector partitioning is either unallocated or orphaned ****\n");
      exit(1);
   }
   if (!local_vector)
   {
      printf("local_vector == NULL -- ");
      printf("hypre_IJVectorGetValuesPar\n");
      printf("**** Vector local data is either unallocated or orphaned ****\n");
      exit(1);
   }

   vec_start = IJpartitioning[my_id];
   vec_stop  = IJpartitioning[my_id+1];

   if (vec_start > vec_stop) 
   {
      printf("vec_start > vec_stop -- ");
      printf("hypre_IJVectorGetValuesPar\n");
      printf("**** This vector partitioning should not occur ****\n");
      exit(1);
   }

/* Determine whether indices points to local indices only,
   and if not, let user know of catastrophe and exit.
   If indices == NULL, assume that num_values components are to be
   retrieved from block starting at vec_start */

   if (indices)
   {
      for (i = 0; i < num_values; i++)
      { 	
	ierr += (indices[i] <  vec_start);
        ierr += (indices[i] >= vec_stop);
      }
   }

   if (ierr)
   {
      printf("indices beyond local range -- ");
      printf("hypre_IJVectorGetValuesPar\n");
      printf("**** Indices specified are unusable ****\n");
      exit(1);
   }
    
   data = hypre_VectorData(local_vector);

   if (indices)
   {
      for (j = 0; j < num_values; j++)
      {
         i = indices[j] - vec_start;
         values[j] = data[i];
      }
   }
   else
   {
      for (j = 0; j < num_values; j++)
         values[j] = data[j];
   }

   return ierr;
}

/******************************************************************************
 *
 * hypre_IJVectorPrintPar
 *
 * prints an IJVector for debugging purposes
 *
 *****************************************************************************/
int
hypre_IJVectorPrintPar(hypre_IJVector *vec,
                       const char     *filename)
{
   int *partitioning = hypre_IJVectorPartitioning(vec);

   return( hypre_ParVectorPrintIJ( hypre_IJVectorObject(vec),
                                   partitioning[0],
                                   filename ) );
}

/******************************************************************************
 *
 * hypre_IJVectorReadPar
 *
 * reads an IJVector from files generated by hypre_IJVectorPrintPar
 *
 *****************************************************************************/
int
hypre_IJVectorReadPar(MPI_Comm         comm,
                      const char      *filename,
                      hypre_IJVector **vec)
{
   int ierr = 0;
   hypre_ParVector *par_vector;
   int base_j;
   int num_procs, i;
   int *col_starts;
   int *partitioning;
   MPI_Comm_size(comm,&num_procs);
   ierr = hypre_ParVectorReadIJ( comm, filename, &base_j, &par_vector);
   col_starts = hypre_ParVectorPartitioning(par_vector);

   *vec = hypre_CTAlloc(hypre_IJVector,1);
   hypre_IJVectorComm(*vec) = comm;
   hypre_IJVectorObject(*vec) = par_vector;

   partitioning = hypre_CTAlloc(int, num_procs+1);
   for (i=0; i < num_procs+1; i++)
   {
      partitioning[i] = col_starts[i] + base_j;
   }

   hypre_IJVectorPartitioning(*vec) = partitioning;

   return ierr;
}
