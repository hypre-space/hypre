/*BHEADER**********************************************************************
 * (c) 1998   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * Member functions for hypre_Vector class.
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_ParVectorCreate
 *--------------------------------------------------------------------------*/

hypre_ParVector *
hypre_ParVectorCreate(  MPI_Comm comm,
			int global_size, 
			int *partitioning)
{
   hypre_ParVector  *vector;
   int num_procs, my_id;

   vector = hypre_CTAlloc(hypre_ParVector, 1);
   MPI_Comm_rank(comm,&my_id);

   if (!partitioning)
   {
     MPI_Comm_size(comm,&num_procs);
     hypre_GeneratePartitioning(global_size, num_procs, &partitioning);
   }

   hypre_ParVectorComm(vector) = comm;
   hypre_ParVectorGlobalSize(vector) = global_size;
   hypre_ParVectorFirstIndex(vector) = partitioning[my_id];
   hypre_ParVectorPartitioning(vector) = partitioning;
   hypre_ParVectorLocalVector(vector) = 
		hypre_SeqVectorCreate(partitioning[my_id+1]-partitioning[my_id]);

   /* set defaults */
   hypre_ParVectorOwnsData(vector) = 1;
   hypre_ParVectorOwnsPartitioning(vector) = 1;

   return vector;
}

/*--------------------------------------------------------------------------
 * hypre_ParVectorDestroy
 *--------------------------------------------------------------------------*/

int 
hypre_ParVectorDestroy( hypre_ParVector *vector )
{
   int  ierr=0;

   if (vector)
   {
      if ( hypre_ParVectorOwnsData(vector) )
      {
         hypre_SeqVectorDestroy(hypre_ParVectorLocalVector(vector));
      }
      if ( hypre_ParVectorOwnsPartitioning(vector) )
      {
         hypre_TFree(hypre_ParVectorPartitioning(vector));
      }
      hypre_TFree(vector);
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_ParVectorInitialize
 *--------------------------------------------------------------------------*/

int 
hypre_ParVectorInitialize( hypre_ParVector *vector )
{
   int  ierr = 0;

   ierr = hypre_SeqVectorInitialize(hypre_ParVectorLocalVector(vector));

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_ParVectorSetDataOwner
 *--------------------------------------------------------------------------*/

int 
hypre_ParVectorSetDataOwner( hypre_ParVector *vector,
                             int           owns_data   )
{
   int    ierr=0;

   hypre_ParVectorOwnsData(vector) = owns_data;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_ParVectorSetPartitioningOwner
 *--------------------------------------------------------------------------*/

int 
hypre_ParVectorSetPartitioningOwner( hypre_ParVector *vector,
                             	     int owns_partitioning)
{
   int    ierr=0;

   hypre_ParVectorOwnsPartitioning(vector) = owns_partitioning;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_ParVectorRead
 *--------------------------------------------------------------------------*/

hypre_ParVector
*hypre_ParVectorRead( MPI_Comm comm, char *file_name )
{
   char 	new_file_name[80];
   hypre_ParVector *par_vector;
   int  	my_id, num_procs;
   int		*partitioning;
   int		global_size, i;
   FILE		*fp;

   MPI_Comm_rank(comm,&my_id); 
   MPI_Comm_size(comm,&num_procs); 

   partitioning = hypre_CTAlloc(int,num_procs+1);

   sprintf(new_file_name,"%s.INFO.%d",file_name,my_id); 
   fp = fopen(new_file_name, "r");
   fscanf(fp, "%d\n", &global_size);
   for (i=0; i < num_procs; i++)
	fscanf(fp, "%d\n", &partitioning[i]);
   fclose (fp);
   partitioning[num_procs] = global_size; 

   par_vector = hypre_CTAlloc(hypre_ParVector, 1);
	
   hypre_ParVectorComm(par_vector) = comm;
   hypre_ParVectorGlobalSize(par_vector) = global_size;
   hypre_ParVectorFirstIndex(par_vector) = partitioning[my_id];
   hypre_ParVectorPartitioning(par_vector) = partitioning;

   hypre_ParVectorOwnsData(par_vector) = 1;
   hypre_ParVectorOwnsPartitioning(par_vector) = 1;

   sprintf(new_file_name,"%s.%d",file_name,my_id); 
   hypre_ParVectorLocalVector(par_vector) = hypre_SeqVectorRead(new_file_name);

   return par_vector;
}

/*--------------------------------------------------------------------------
 * hypre_ParVectorPrint
 *--------------------------------------------------------------------------*/

int
hypre_ParVectorPrint( hypre_ParVector  *vector, 
                      char             *file_name )
{
   char 	new_file_name[80];
   hypre_Vector *local_vector = hypre_ParVectorLocalVector(vector); 
   int  	ierr = 0;
   MPI_Comm 	comm = hypre_ParVectorComm(vector);
   int  	my_id, num_procs, i;
   int		*partitioning = hypre_ParVectorPartitioning(vector); 
   int		global_size = hypre_ParVectorGlobalSize(vector); 
   FILE		*fp;

   MPI_Comm_rank(comm,&my_id); 
   MPI_Comm_size(comm,&num_procs); 
   sprintf(new_file_name,"%s.%d",file_name,my_id); 
   ierr = hypre_SeqVectorPrint(local_vector,new_file_name);
   sprintf(new_file_name,"%s.INFO.%d",file_name,my_id); 
   fp = fopen(new_file_name, "w");
   fprintf(fp, "%d\n", global_size);
   for (i=0; i < num_procs; i++)
	fprintf(fp, "%d\n", partitioning[i]);
   fclose (fp);
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_ParVectorSetConstantValues
 *--------------------------------------------------------------------------*/

int
hypre_ParVectorSetConstantValues( hypre_ParVector *v,
                                  double        value )
{
   hypre_Vector *v_local = hypre_ParVectorLocalVector(v);
           
   return hypre_SeqVectorSetConstantValues(v_local,value);
}

/*--------------------------------------------------------------------------
 * hypre_ParVectorSetRandomValues
 *--------------------------------------------------------------------------*/

int
hypre_ParVectorSetRandomValues( hypre_ParVector *v,
                                int            seed )
{
   int my_id;
   hypre_Vector *v_local = hypre_ParVectorLocalVector(v);

   MPI_Comm 	comm = hypre_ParVectorComm(v);
   MPI_Comm_rank(comm,&my_id); 

   seed *= (my_id+1);
           
   return hypre_SeqVectorSetRandomValues(v_local,seed);
}

/*--------------------------------------------------------------------------
 * hypre_ParVectorCopy
 *--------------------------------------------------------------------------*/

int
hypre_ParVectorCopy( hypre_ParVector *x,
                     hypre_ParVector *y )
{
   hypre_Vector *x_local = hypre_ParVectorLocalVector(x);
   hypre_Vector *y_local = hypre_ParVectorLocalVector(y);
   return hypre_SeqVectorCopy(x_local, y_local);
}

/*--------------------------------------------------------------------------
 * hypre_ParVectorScale
 *--------------------------------------------------------------------------*/

int
hypre_ParVectorScale( double        alpha,
                      hypre_ParVector *y     )
{
   hypre_Vector *y_local = hypre_ParVectorLocalVector(y);

   return hypre_SeqVectorScale( alpha, y_local);
}

/*--------------------------------------------------------------------------
 * hypre_ParVectorAxpy
 *--------------------------------------------------------------------------*/

int
hypre_ParVectorAxpy( double        alpha,
               hypre_ParVector *x,
               hypre_ParVector *y     )
{
   hypre_Vector *x_local = hypre_ParVectorLocalVector(x);
   hypre_Vector *y_local = hypre_ParVectorLocalVector(y);
           
   return hypre_SeqVectorAxpy( alpha, x_local, y_local);
}

/*--------------------------------------------------------------------------
 * hypre_ParVectorInnerProd
 *--------------------------------------------------------------------------*/

double
hypre_ParVectorInnerProd( hypre_ParVector *x,
                    hypre_ParVector *y )
{
   MPI_Comm      comm    = hypre_ParVectorComm(x);
   hypre_Vector *x_local = hypre_ParVectorLocalVector(x);
   hypre_Vector *y_local = hypre_ParVectorLocalVector(y);
           
   double result = 0.0;
   double local_result = hypre_SeqVectorInnerProd(x_local, y_local);
   
   MPI_Allreduce(&local_result, &result, 1, MPI_DOUBLE, MPI_SUM, comm);
   
   return result;
}

/*--------------------------------------------------------------------------
 * hypre_VectorToParVector:
 * generates a ParVector from a Vector on proc 0 and distributes the pieces
 * to the other procs in comm
 *--------------------------------------------------------------------------*/

hypre_ParVector *
hypre_VectorToParVector (MPI_Comm comm, hypre_Vector *v, int *vec_starts)
{
   int 			global_size;
   int 			local_size;
   int  		num_procs, my_id;
   hypre_ParVector  	*par_vector;
   hypre_Vector     	*local_vector;
   double          	*v_data;
   double		*local_data;
   MPI_Request		*requests;
   MPI_Status		*status, status0;
   int			i, j;

   MPI_Comm_size(comm,&num_procs);
   MPI_Comm_rank(comm,&my_id);

   if (my_id == 0)
   {
        global_size = hypre_VectorSize(v);
        v_data = hypre_VectorData(v);
   }

   MPI_Bcast(&global_size,1,MPI_INT,0,comm);

   par_vector = hypre_ParVectorCreate(comm, global_size, vec_starts);

   vec_starts = hypre_ParVectorPartitioning(par_vector);

   local_size = vec_starts[my_id+1] - vec_starts[my_id];
   hypre_ParVectorInitialize(par_vector);
   local_vector = hypre_ParVectorLocalVector(par_vector);
   local_data = hypre_VectorData(local_vector);

   if (my_id == 0)
   {
	requests = hypre_CTAlloc(MPI_Request,num_procs-1);
	status = hypre_CTAlloc(MPI_Status,num_procs-1);
	j = 0;
	for (i=1; i < num_procs; i++)
		MPI_Isend(&v_data[vec_starts[i]],vec_starts[i+1]-vec_starts[i],
		MPI_DOUBLE, i, 0, comm, &requests[j++]);
	for (i=0; i < local_size; i++)
		local_data[i] = v_data[i];
	MPI_Waitall(num_procs-1,requests, status);
	hypre_TFree(requests);
	hypre_TFree(status);
   }
   else
   {
	MPI_Recv(local_data,local_size,MPI_DOUBLE,0,0,comm,&status0);
   }

   return par_vector;
}
   
/*--------------------------------------------------------------------------
 * hypre_ParVectorToVectorAll:
 * generates a Vector on every proc which has a piece of the data
 * from a ParVector on several procs in comm,
 * vec_starts needs to contain the partitioning across all procs in comm 
 *--------------------------------------------------------------------------*/

hypre_Vector *
hypre_ParVectorToVectorAll (hypre_ParVector *par_v)
{
   MPI_Comm		comm = hypre_ParVectorComm(par_v);
   int 			global_size = hypre_ParVectorGlobalSize(par_v);
   int 			*vec_starts = hypre_ParVectorPartitioning(par_v);
   hypre_Vector     	*local_vector = hypre_ParVectorLocalVector(par_v);
   int  		num_procs, my_id;
   hypre_Vector  	*vector;
   double		*vector_data;
   double		*local_data;
   int 			local_size;
   MPI_Request		*requests;
   MPI_Status		*status;
   int			i, j;
   int			*used_procs;
   int			num_types, num_requests;
   int			vec_len, proc_id;

   MPI_Comm_size(comm, &num_procs);
   MPI_Comm_rank(comm, &my_id);

   local_size = vec_starts[my_id+1] - vec_starts[my_id];

/* if my_id contains no data, return NULL  */

   if (!local_size)
	return NULL;
 
   local_data = hypre_VectorData(local_vector);
   vector = hypre_SeqVectorCreate(global_size);
   hypre_SeqVectorInitialize(vector);
   vector_data = hypre_VectorData(vector);

/* determine procs which hold data of par_v and store ids in used_procs */

   num_types = -1;
   for (i=0; i < num_procs; i++)
        if (vec_starts[i+1]-vec_starts[i])
                num_types++;
   num_requests = 2*num_types;
 
   used_procs = hypre_CTAlloc(int, num_types);
   j = 0;
   for (i=0; i < num_procs; i++)
        if (vec_starts[i+1]-vec_starts[i] && i-my_id)
                used_procs[j++] = i;
 
   requests = hypre_CTAlloc(MPI_Request, num_requests);
   status = hypre_CTAlloc(MPI_Status, num_requests);

/* initialize data exchange among used_procs and generate vector */
 
   j = 0;
   for (i = 0; i < num_types; i++)
   {
        proc_id = used_procs[i];
        vec_len = vec_starts[proc_id+1] - vec_starts[proc_id];
        MPI_Irecv(&vector_data[vec_starts[proc_id]], vec_len, MPI_DOUBLE,
                                proc_id, 0, comm, &requests[j++]);
   }
   for (i = 0; i < num_types; i++)
   {
        MPI_Isend(local_data, local_size, MPI_DOUBLE, used_procs[i],
                          0, comm, &requests[j++]);
   }
 
   for (i=0; i < local_size; i++)
        vector_data[vec_starts[my_id]+i] = local_data[i];
 
   MPI_Waitall(num_requests, requests, status);

   if (num_requests)
   {
   	hypre_TFree(used_procs);
   	hypre_TFree(requests);
   	hypre_TFree(status); 
   }

   return vector;
}

/*--------------------------------------------------------------------------
 * hypre_ParVectorPrintIJ
 *--------------------------------------------------------------------------*/

int
hypre_ParVectorPrintIJ( hypre_ParVector *vector,
                        int              base_j,
                        char            *filename )
{
   int ierr = 0;
   MPI_Comm          comm         = hypre_ParVectorComm(vector);
   int               global_size  = hypre_ParVectorGlobalSize(vector);
   int              *partitioning = hypre_ParVectorPartitioning(vector);
   double           *local_data;
   int               myid, num_procs, i, j, part0;
   char              new_filename[255];
   FILE             *file;

   MPI_Comm_rank(comm, &myid);
   MPI_Comm_size(comm, &num_procs);
  
   sprintf(new_filename,"%s.%05d", filename, myid);

   if ((file = fopen(new_filename, "w")) == NULL)
   {
      printf("Error: can't open output file %s\n", new_filename);
      exit(1);
   }

   local_data = hypre_VectorData(hypre_ParVectorLocalVector(vector));

   fprintf(file, "%d \n", global_size);

   for (i=0; i <= num_procs; i++)
   {
      fprintf(file, "%d \n", partitioning[i] + base_j);
   }

   part0 = partitioning[myid];
   for (j = part0; j < partitioning[myid+1]; j++)
   {
      fprintf(file, "%d %le\n", j + base_j, local_data[j-part0]);
   }

   fclose(file);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_ParVectorReadIJ
 *--------------------------------------------------------------------------*/

int
hypre_ParVectorReadIJ( MPI_Comm             comm,
                       char                *filename,
                       int                 *base_j_ptr,
                       hypre_ParVector    **vector_ptr)
{
   int ierr = 0;
   int               global_size;
   hypre_ParVector  *vector;
   hypre_Vector     *local_vector;
   double           *local_data;
   int              *partitioning;
   int               base_j;

   int               myid, num_procs, i, j, J;
   char              new_filename[255];
   FILE             *file;

   MPI_Comm_size(comm, &num_procs);
   MPI_Comm_rank(comm, &myid);
  
   sprintf(new_filename,"%s.%05d", filename, myid);

   if ((file = fopen(new_filename, "r")) == NULL)
   {
      printf("Error: can't open output file %s\n", new_filename);
      exit(1);
   }

   fscanf(file, "%d", &global_size);

   partitioning = hypre_CTAlloc(int,num_procs+1);

   fscanf(file, "%d", partitioning);
   for (i = 1; i <= num_procs; i++)
   {
      fscanf(file, "%d", partitioning+i);
      partitioning[i] -= partitioning[0];
   }
   base_j = partitioning[0];
   partitioning[0] = 0;

   vector = hypre_ParVectorCreate(comm, global_size,
                                  partitioning);

   hypre_ParVectorInitialize(vector);

   local_vector = hypre_ParVectorLocalVector(vector);
   local_data   = hypre_VectorData(local_vector);

   for (j = 0; j < partitioning[myid+1] - partitioning[myid]; j++)
   {
      fscanf(file, "%d %le", &J, local_data + j);
   }

   fclose(file);

   *base_j_ptr = base_j;
   *vector_ptr = vector;

   return ierr;
}
