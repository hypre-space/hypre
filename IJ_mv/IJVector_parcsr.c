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
 
#include "headers.h"
#include "../HYPRE.h"

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
   hypre_AuxParVector *aux_vector = hypre_IJVectorTranslator(vector);
   int *partitioning = hypre_ParVectorPartitioning(par_vector);
   hypre_Vector *local_vector = hypre_ParVectorLocalVector(par_vector);
   int my_id, ierr = 0;
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

   ierr += hypre_ParVectorInitialize(par_vector);

   if (!aux_vector)
   {  
      ierr = hypre_AuxParVectorCreate(&aux_vector);
      hypre_IJVectorTranslator(vector) = aux_vector;
   }
   ierr += hypre_AuxParVectorInitialize(aux_vector);

   return ierr;
}

/******************************************************************************
 *
 * hypre_IJVectorSetMaxOffProcElmtsPar
 *
 *****************************************************************************/

int
hypre_IJVectorSetMaxOffProcElmtsPar(hypre_IJVector *vector,
                                    int max_off_proc_elmts)
{
   int ierr = 0;
   hypre_AuxParVector *aux_vector;

   aux_vector = hypre_IJVectorTranslator(vector);
   if (!aux_vector)
   {
      ierr = hypre_AuxParVectorCreate(&aux_vector);
      hypre_IJVectorTranslator(vector) = aux_vector;
   }
   hypre_AuxParVectorMaxOffProcElmts(aux_vector) = max_off_proc_elmts;
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
   hypre_AuxParVector *aux_vector = hypre_IJVectorTranslator(vector);
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
   vec_stop  = IJpartitioning[my_id+1]-1;
  
   if (vec_start > vec_stop) 
   {
      printf("vec_start > vec_stop -- ");
      printf("hypre_IJVectorSetValuesPar\n");
      printf("**** This vector partitioning should not occur ****\n");
      exit(1);
   }

/* Determine whether indices points to local indices only,
   and if not, store indices and values into auxiliary vector structure
   If indices == NULL, assume that num_values components are to be
   set in a block starting at vec_start. 
   NOTE: If indices == NULL off processor values are ignored!!! */

   data = hypre_VectorData(local_vector);

   if (indices)
   {
      int current_num_elmts
                = hypre_AuxParVectorCurrentNumElmts(aux_vector);
      int max_off_proc_elmts
                = hypre_AuxParVectorMaxOffProcElmts(aux_vector);
      int *off_proc_i = hypre_AuxParVectorOffProcI(aux_vector);
      double *off_proc_data = hypre_AuxParVectorOffProcData(aux_vector);

      for (j = 0; j < num_values; j++)
      {
	 i = indices[j];
	 if (i < vec_start || i > vec_stop)
         {
	 /* if elements outside processor boundaries, store in off processor
		stash */
	    if (!max_off_proc_elmts)
            {
               max_off_proc_elmts = 100;
               hypre_AuxParVectorMaxOffProcElmts(aux_vector) =
                        max_off_proc_elmts;
               hypre_AuxParVectorOffProcI(aux_vector)
                        = hypre_CTAlloc(int,max_off_proc_elmts);
               hypre_AuxParVectorOffProcData(aux_vector)
                        = hypre_CTAlloc(double,max_off_proc_elmts);
               off_proc_i = hypre_AuxParVectorOffProcI(aux_vector);
               off_proc_data = hypre_AuxParVectorOffProcData(aux_vector);
            }
            else if (current_num_elmts + 1 > max_off_proc_elmts)
            {
               max_off_proc_elmts += 10;
               off_proc_i = hypre_TReAlloc(off_proc_i,int,max_off_proc_elmts);
               off_proc_data = hypre_TReAlloc(off_proc_data,double,
                                max_off_proc_elmts);
               hypre_AuxParVectorMaxOffProcElmts(aux_vector)
                        = max_off_proc_elmts;
               hypre_AuxParVectorOffProcI(aux_vector) = off_proc_i;
               hypre_AuxParVectorOffProcData(aux_vector) = off_proc_data;
            }
            off_proc_i[current_num_elmts] = i;
            off_proc_data[current_num_elmts++] = values[j];
            hypre_AuxParVectorCurrentNumElmts(aux_vector)=current_num_elmts;
         }
         else /* local values are inserted into the vector */
         {
            i -= vec_start;
            data[i] = values[j];
         }
      } 
   }
   else 
   {
      if (num_values > vec_stop - vec_start + 1)
      {
         printf("Warning! Indices beyond local range  not identified!\n ");
         printf("Off processor values have been ignored!\n");
	 num_values = vec_stop - vec_start +1;
      }

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
   hypre_AuxParVector *aux_vector = hypre_IJVectorTranslator(vector);
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
   vec_stop  = IJpartitioning[my_id+1]-1;

   if (vec_start > vec_stop) 
   {
      printf("vec_start > vec_stop -- ");
      printf("hypre_IJVectorAddToValuesPar\n");
      printf("**** This vector partitioning should not occur ****\n");
      exit(1);
   }

/* Determine whether indices points to local indices only,
   and if not, store indices and values into auxiliary vector structure
   If indices == NULL, assume that num_values components are to be
   set in a block starting at vec_start. 
   NOTE: If indices == NULL off processor values are ignored!!! */

   /* if (indices)
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
   } */
    
   data = hypre_VectorData(local_vector);

   if (indices)
   {
      int current_num_elmts
                = hypre_AuxParVectorCurrentNumElmts(aux_vector);
      int max_off_proc_elmts
                = hypre_AuxParVectorMaxOffProcElmts(aux_vector);
      int *off_proc_i = hypre_AuxParVectorOffProcI(aux_vector);
      double *off_proc_data = hypre_AuxParVectorOffProcData(aux_vector);

      for (j = 0; j < num_values; j++)
      {
	 i = indices[j];
	 if (i < vec_start || i > vec_stop)
         {
	 /* if elements outside processor boundaries, store in off processor
		stash */
	    if (!max_off_proc_elmts)
            {
               max_off_proc_elmts = 100;
               hypre_AuxParVectorMaxOffProcElmts(aux_vector) =
                        max_off_proc_elmts;
               hypre_AuxParVectorOffProcI(aux_vector)
                        = hypre_CTAlloc(int,max_off_proc_elmts);
               hypre_AuxParVectorOffProcData(aux_vector)
                        = hypre_CTAlloc(double,max_off_proc_elmts);
               off_proc_i = hypre_AuxParVectorOffProcI(aux_vector);
               off_proc_data = hypre_AuxParVectorOffProcData(aux_vector);
            }
            else if (current_num_elmts + 1 > max_off_proc_elmts)
            {
               max_off_proc_elmts += 10;
               off_proc_i = hypre_TReAlloc(off_proc_i,int,max_off_proc_elmts);
               off_proc_data = hypre_TReAlloc(off_proc_data,double,
                                max_off_proc_elmts);
               hypre_AuxParVectorMaxOffProcElmts(aux_vector)
                        = max_off_proc_elmts;
               hypre_AuxParVectorOffProcI(aux_vector) = off_proc_i;
               hypre_AuxParVectorOffProcData(aux_vector) = off_proc_data;
            }
            off_proc_i[current_num_elmts] = -i-1;
            off_proc_data[current_num_elmts++] = values[j];
            hypre_AuxParVectorCurrentNumElmts(aux_vector)=current_num_elmts;
         }
         else /* local values are added to the vector */
         {
            i -= vec_start;
            data[i] += values[j];
         }
      } 
   }
   else 
   {
      if (num_values > vec_stop - vec_start + 1)
      {
         printf("Warning! Indices beyond local range  not identified!\n ");
         printf("Off processor values have been ignored!\n");
	 num_values = vec_stop - vec_start +1;
      }

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
   hypre_AuxParVector *aux_vector = hypre_IJVectorTranslator(vector);
   int *partitioning = hypre_ParVectorPartitioning(par_vector);
   MPI_Comm comm = hypre_IJVectorComm(vector);

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

   if (aux_vector)
   {
      int off_proc_elmts, current_num_elmts;
      int max_off_proc_elmts;
      int *off_proc_i;
      double *off_proc_data;
      current_num_elmts = hypre_AuxParVectorCurrentNumElmts(aux_vector);
      MPI_Allreduce(&current_num_elmts,&off_proc_elmts,1,MPI_INT, MPI_SUM,comm);
      if (off_proc_elmts)
      {
         max_off_proc_elmts=hypre_AuxParVectorMaxOffProcElmts(aux_vector);
         off_proc_i=hypre_AuxParVectorOffProcI(aux_vector);
         off_proc_data=hypre_AuxParVectorOffProcData(aux_vector);
         hypre_IJVectorAssembleOffProcValsPar(vector, max_off_proc_elmts, 
		current_num_elmts, off_proc_i, off_proc_data);
	 hypre_TFree(hypre_AuxParVectorOffProcI(aux_vector));
	 hypre_TFree(hypre_AuxParVectorOffProcData(aux_vector));
	 hypre_AuxParVectorMaxOffProcElmts(aux_vector) = 0;
	 hypre_AuxParVectorCurrentNumElmts(aux_vector) = 0;
      }
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

int
hypre_IJVectorAssembleOffProcValsPar( hypre_IJVector *vector, 
   				      int max_off_proc_elmts,
   				      int current_num_elmts,
   				      int *off_proc_i,
   			     	      double *off_proc_data)
{
   int ierr = 0;
   MPI_Comm comm = hypre_IJMatrixComm(vector);
   hypre_ParVector *par_vector = hypre_IJMatrixObject(vector);
   MPI_Request *requests;
   MPI_Status *status;
   int i, j, j2, row;
   int iii, indx, ip, first_index;
   int proc_id, num_procs, my_id;
   int num_sends, num_sends2;
   int num_recvs;
   int num_requests;
   int vec_start, vec_len;
   int *send_procs;
   int *send_i;
   int *send_map_starts;
   int *recv_procs;
   int *recv_i;
   int *recv_vec_starts;
   int *info;
   int *int_buffer;
   int *proc_id_mem;
   int *partitioning;
   int *displs;
   int *recv_buf;
   double *send_data;
   double *recv_data;
   double *data = hypre_VectorData(hypre_ParVectorLocalVector(par_vector));

   MPI_Comm_size(comm,&num_procs);
   MPI_Comm_rank(comm, &my_id);
   partitioning = hypre_IJVectorPartitioning(vector);
   first_index = partitioning[my_id];
   
   info = hypre_CTAlloc(int,num_procs);  
   proc_id_mem = hypre_CTAlloc(int,current_num_elmts);
   for (i=0; i < current_num_elmts; i++)
   {
      row = off_proc_i[i]; 
      if (row < 0) row = -row-1;
      proc_id = hypre_FindProc(partitioning,row,num_procs);
      proc_id_mem[i] = proc_id; 
      info[proc_id]++;
   }

   /* determine send_procs and amount of data to be sent */   
   num_sends = 0;
   for (i=0; i < num_procs; i++)
   {
      if (info[i])
      {
         num_sends++;
      }
   }
   num_sends2 = 2*num_sends;
   send_procs =  hypre_CTAlloc(int,num_sends);
   send_map_starts =  hypre_CTAlloc(int,num_sends+1);
   int_buffer =  hypre_CTAlloc(int,num_sends2);
   j = 0;
   j2 = 0;
   send_map_starts[0] = 0;
   for (i=0; i < num_procs; i++)
   {
      if (info[i])
      {
         send_procs[j++] = i;
         send_map_starts[j] = send_map_starts[j-1]+info[i];
         int_buffer[j2++] = i;
	 int_buffer[j2++] = info[i];
      }
   }

   MPI_Allgather(&num_sends2,1,MPI_INT,info,1,MPI_INT,comm);

   displs = hypre_CTAlloc(int, num_procs+1);
   displs[0] = 0;
   for (i=1; i < num_procs+1; i++)
        displs[i] = displs[i-1]+info[i-1];
   recv_buf = hypre_CTAlloc(int, displs[num_procs]);

   MPI_Allgatherv(int_buffer,num_sends2,MPI_INT,recv_buf,info,displs,
			MPI_INT,comm);

   hypre_TFree(int_buffer);
   hypre_TFree(info);

   /* determine recv procs and amount of data to be received */
   num_recvs = 0;
   for (j=0; j < displs[num_procs]; j+=2)
   {
      if (recv_buf[j] == my_id)
	 num_recvs++;
   }

   recv_procs = hypre_CTAlloc(int,num_recvs);
   recv_vec_starts = hypre_CTAlloc(int,num_recvs+1);

   j2 = 0;
   recv_vec_starts[0] = 0;
   for (i=0; i < num_procs; i++)
   {
      for (j=displs[i]; j < displs[i+1]; j+=2)
      {
         if (recv_buf[j] == my_id)
         {
	    recv_procs[j2++] = i;
	    recv_vec_starts[j2] = recv_vec_starts[j2-1]+recv_buf[j+1];
         }
         if (j2 == num_recvs) break;
      }
   }
   hypre_TFree(recv_buf);
   hypre_TFree(displs);

   /* set up data to be sent to send procs */
   /* send_i contains for each send proc 
      indices, send_data contains corresponding values */
      
   send_i = hypre_CTAlloc(int,send_map_starts[num_sends]);
   send_data = hypre_CTAlloc(double,send_map_starts[num_sends]);
   recv_i = hypre_CTAlloc(int,recv_vec_starts[num_recvs]);
   recv_data = hypre_CTAlloc(double,recv_vec_starts[num_recvs]);
    
   for (i=0; i < current_num_elmts; i++)
   {
      proc_id = proc_id_mem[i];
      indx = hypre_BinarySearch(send_procs,proc_id,num_sends);
      iii = send_map_starts[indx];
      send_i[iii] = off_proc_i[i]; 
      send_data[iii] = off_proc_data[i];
      send_map_starts[indx]++;
   }

   hypre_TFree(proc_id_mem);

   for (i=num_sends; i > 0; i--)
   {
      send_map_starts[i] = send_map_starts[i-1];
   }
   send_map_starts[0] = 0;

   num_requests = num_recvs+num_sends;

   requests = hypre_CTAlloc(MPI_Request, num_requests);
   status = hypre_CTAlloc(MPI_Status, num_requests);

   j=0; 
   for (i=0; i < num_recvs; i++)
   {
       vec_start = recv_vec_starts[i];
       vec_len = recv_vec_starts[i+1] - vec_start;
       ip = recv_procs[i];
       MPI_Irecv(&recv_i[vec_start], vec_len, MPI_INT, ip, 0, comm, 
			&requests[j++]);
   }

   for (i=0; i < num_sends; i++)
   {
       vec_start = send_map_starts[i];
       vec_len = send_map_starts[i+1] - vec_start;
       ip = send_procs[i];
       MPI_Isend(&send_i[vec_start], vec_len, MPI_INT, ip, 0, comm, 
			&requests[j++]);
   }
  
   if (num_requests)
   {
      MPI_Waitall(num_requests, requests, status);
   }

   j=0;
   for (i=0; i < num_recvs; i++)
   {
       vec_start = recv_vec_starts[i];
       vec_len = recv_vec_starts[i+1] - vec_start;
       ip = recv_procs[i];
       MPI_Irecv(&recv_data[vec_start], vec_len, MPI_DOUBLE, ip, 0, comm, 
			&requests[j++]);
   }

   for (i=0; i < num_sends; i++)
   {
       vec_start = send_map_starts[i];
       vec_len = send_map_starts[i+1] - vec_start;
       ip = send_procs[i];
       MPI_Isend(&send_data[vec_start], vec_len, MPI_DOUBLE, ip, 0, comm, 
			&requests[j++]);
   }
  
   if (num_requests)
   {
      MPI_Waitall(num_requests, requests, status);
      hypre_TFree(requests);
      hypre_TFree(status);
   }

   hypre_TFree(send_i);
   hypre_TFree(send_data);
   hypre_TFree(send_procs);
   hypre_TFree(send_map_starts);
   hypre_TFree(recv_procs);

   for (i=0; i < recv_vec_starts[num_recvs]; i++)
   {
      row = recv_i[i];
      if (row < 0)
      {
         row = -row-1;
         j = row - first_index;
         data[j] += recv_data[i];
      }
      else
      {
         j = row - first_index;
         data[j] = recv_data[i];
      }
   }

   hypre_TFree(recv_vec_starts);
   hypre_TFree(recv_i);
   hypre_TFree(recv_data);

   return ierr;
}
