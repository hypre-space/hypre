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
 * hypre_CreateParVector
 *--------------------------------------------------------------------------*/

hypre_ParVector *
hypre_CreateParVector(  MPI_Comm comm,
			int global_size, 
			int first_index, 
			int local_size) 
{
   hypre_ParVector  *vector;
   int num_procs, my_id;

   vector = hypre_CTAlloc(hypre_ParVector, 1);

   if (!first_index && !local_size)
   {
	MPI_Comm_size(comm,&num_procs);
	MPI_Comm_rank(comm,&my_id);
	MPE_Decomp1d(global_size,num_procs,my_id,&first_index,&local_size);
	first_index--;
	local_size = local_size-first_index;
   }
	
   hypre_ParVectorComm(vector) = comm;
   hypre_ParVectorGlobalSize(vector) = global_size;
   hypre_ParVectorFirstIndex(vector) = first_index;
   hypre_ParVectorLocalVector(vector) = 
		hypre_CreateVector(local_size);
   hypre_ParVectorCommPkg(vector) = NULL;

   /* set defaults */
   hypre_ParVectorOwnsData(vector) = 1;

   return vector;
}

/*--------------------------------------------------------------------------
 * hypre_DestroyParVector
 *--------------------------------------------------------------------------*/

int 
hypre_DestroyParVector( hypre_ParVector *vector )
{
   int  ierr=0;

   if (vector)
   {
      if ( hypre_ParVectorOwnsData(vector) )
      {
         hypre_DestroyVector(hypre_ParVectorLocalVector(vector));
	 if (hypre_ParVectorCommPkg(vector))
		hypre_DestroyVectorCommPkg(hypre_ParVectorCommPkg(vector));
      }
      hypre_TFree(vector);
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_InitializeParVector
 *--------------------------------------------------------------------------*/

int 
hypre_InitializeParVector( hypre_ParVector *vector )
{
   int  ierr = 0;

   ierr = hypre_InitializeVector(hypre_ParVectorLocalVector(vector));

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SetParVectorDataOwner
 *--------------------------------------------------------------------------*/

int 
hypre_SetParVectorDataOwner( hypre_ParVector *vector,
                             int           owns_data   )
{
   int    ierr=0;

   hypre_ParVectorOwnsData(vector) = owns_data;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_PrintParVector
 *--------------------------------------------------------------------------*/

int
hypre_PrintParVector( hypre_ParVector  *vector, 
                      char             *file_name )
{
   char 	new_file_name[80];
   hypre_Vector *local_vector = hypre_ParVectorLocalVector(vector); 
   int  	ierr = 0;
   MPI_Comm 	comm = hypre_ParVectorComm(vector);
   int  	my_id; 

   MPI_Comm_rank(comm,&my_id); 
   sprintf(new_file_name,"%s.%d",file_name,my_id); 
   ierr = hypre_PrintVector(local_vector,new_file_name);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SetParVectorConstantValues
 *--------------------------------------------------------------------------*/

int
hypre_SetParVectorConstantValues( hypre_ParVector *v,
                                  double        value )
{
   hypre_Vector *v_local = hypre_ParVectorLocalVector(v);
           
   return hypre_SetVectorConstantValues(v_local,value);
}

/*--------------------------------------------------------------------------
 * hypre_CopyParVector
 *--------------------------------------------------------------------------*/

int
hypre_CopyParVector( hypre_ParVector *x,
                     hypre_ParVector *y )
{
   hypre_Vector *x_local = hypre_ParVectorLocalVector(x);
   hypre_Vector *y_local = hypre_ParVectorLocalVector(y);

   return hypre_CopyVector(x_local, y_local);
}

/*--------------------------------------------------------------------------
 * hypre_ScaleParVector
 *--------------------------------------------------------------------------*/

int
hypre_ScaleParVector( double        alpha,
                      hypre_ParVector *y     )
{
   hypre_Vector *y_local = hypre_ParVectorLocalVector(y);

   return hypre_ScaleVector( alpha, y_local);
}

/*--------------------------------------------------------------------------
 * hypre_ParAxpy
 *--------------------------------------------------------------------------*/

int
hypre_ParAxpy( double        alpha,
               hypre_ParVector *x,
               hypre_ParVector *y     )
{
   hypre_Vector *x_local = hypre_ParVectorLocalVector(x);
   hypre_Vector *y_local = hypre_ParVectorLocalVector(y);
           
   return hypre_Axpy( alpha, x_local, y_local);
}

/*--------------------------------------------------------------------------
 * hypre_ParInnerProd
 *--------------------------------------------------------------------------*/

double   hypre_ParInnerProd( MPI_Comm comm,
			     hypre_ParVector *x,
                             hypre_ParVector *y )
{
   hypre_Vector *x_local = hypre_ParVectorLocalVector(x);
   hypre_Vector *y_local = hypre_ParVectorLocalVector(y);
           
   double result = 0.0;
   double local_result = hypre_InnerProd(x_local, y_local);
   
   MPI_Allreduce(&local_result, &result, 1, MPI_DOUBLE, MPI_SUM, comm);
   
   return result;
}

/*--------------------------------------------------------------------------
 * hypre_VectorToParVector
 *--------------------------------------------------------------------------*/

hypre_ParVector *
hypre_VectorToParVector (MPI_Comm comm, hypre_Vector *v,
			 int *vec_starts)
{
   int 			global_size = hypre_VectorSize(v);
   int 			local_size;
   int  		num_procs, my_id;
   MPI_Datatype   	*vector_mpi_types;
   hypre_ParVector  	*par_vector;
   hypre_Vector     	*local_vector;
   double          	*v_data = hypre_VectorData(v);
   double		*local_data;
   MPI_Request		*requests;
   MPI_Status		*status, status0;
   int			i, j;
   hypre_VectorCommPkg  *vector_comm_pkg;

   MPI_Comm_size(comm,&num_procs);
   MPI_Comm_rank(comm,&my_id);

   vector_comm_pkg = hypre_InitializeVectorCommPkg(comm, global_size,
		vec_starts);

   vector_mpi_types = hypre_VectorCommPkgVectorMPITypes(vector_comm_pkg);
   vec_starts = hypre_VectorCommPkgVecStarts(vector_comm_pkg);
	
   local_size = vec_starts[my_id+1] - vec_starts[my_id];
   par_vector = hypre_CreateParVector(comm, global_size, vec_starts[my_id], 
		local_size);
   hypre_InitializeParVector(par_vector);
   local_vector = hypre_ParVectorLocalVector(par_vector);
   local_data = hypre_VectorData(local_vector);

   if (my_id == 0)
   {
	requests = hypre_CTAlloc(MPI_Request,num_procs-1);
	status = hypre_CTAlloc(MPI_Status,num_procs-1);
	j = 0;
	for (i=1; i < num_procs; i++)
		MPI_Isend(&v_data[vec_starts[i]], 1, vector_mpi_types[i], i, 0,
				comm, &requests[j++]);
	for (i=0; i < local_size; i++)
		local_data[i] = v_data[i];
	MPI_Waitall(num_procs-1,requests, status);
	hypre_TFree(requests);
	hypre_TFree(status);
   }
   else
   {
	MPI_Recv(local_data,1,vector_mpi_types[my_id],0,0,comm,&status0);
   }
   hypre_ParVectorCommPkg(par_vector) = vector_comm_pkg;

   return par_vector;
}
   

int
hypre_BuildParVectorMPITypes (MPI_Comm  comm,
			      int	vec_len,
			      int	*vec_starts,
			      MPI_Datatype   *vector_mpi_types)
{
   int		i;
   int		ierr = 0;
   int		num_procs;
   int		*len;

   MPI_Comm_size( comm, &num_procs);
   len = hypre_CTAlloc(int,num_procs);

   for (i=0; i < num_procs; i++)
   {
	MPE_Decomp1d(vec_len, num_procs, i, &vec_starts[i], &len[i]);
        vec_starts[i] = vec_starts[i]-1;
	len[i] = len[i]-vec_starts[i];
	MPI_Type_vector(len[i],1,1,MPI_DOUBLE, &vector_mpi_types[i]);
	MPI_Type_commit(&vector_mpi_types[i]);
   }
   hypre_TFree(len);

   return ierr;
}
