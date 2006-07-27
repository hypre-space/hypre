/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team <hypre-users@llnl.gov>, UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer and the GNU Lesser General Public License.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the terms and conditions of the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
 ***********************************************************************EHEADER*/

/******************************************************************************
 *
 * Member functions for hypre_Vector class.
 *
 *****************************************************************************/

#include "headers.h"
#include <assert.h>

#ifdef HYPRE_NO_GLOBAL_PARTITION
int hypre_FillResponseParToVectorAll(void*, int, int, void*, MPI_Comm, void**, int*);
#endif


/*--------------------------------------------------------------------------
 * hypre_ParVectorCreate
 *--------------------------------------------------------------------------*/

/* If create is called for HYPRE_NO_GLOBAL_PARTITION and partitioning is NOT null,
   then it is assumed that it is array of length 2 containing the start row of 
   the calling processor followed by the start row of the next processor - AHB 6/05 */


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
#ifdef HYPRE_NO_GLOBAL_PARTITION
     hypre_GenerateLocalPartitioning(global_size, num_procs, my_id, &partitioning);
#else
     hypre_GeneratePartitioning(global_size, num_procs, &partitioning);
#endif
   }

   hypre_ParVectorComm(vector) = comm;
   hypre_ParVectorGlobalSize(vector) = global_size;
#ifdef HYPRE_NO_GLOBAL_PARTITION
   hypre_ParVectorFirstIndex(vector) = partitioning[0];
   hypre_ParVectorLastIndex(vector) = partitioning[1]-1;
   hypre_ParVectorPartitioning(vector) = partitioning;
   hypre_ParVectorLocalVector(vector) = 
		hypre_SeqVectorCreate(partitioning[1]-partitioning[0]);
#else
   hypre_ParVectorFirstIndex(vector) = partitioning[my_id];
   hypre_ParVectorLastIndex(vector) = partitioning[my_id+1] -1;
   hypre_ParVectorPartitioning(vector) = partitioning;
   hypre_ParVectorLocalVector(vector) = 
		hypre_SeqVectorCreate(partitioning[my_id+1]-partitioning[my_id]);
#endif

   /* set defaults */
   hypre_ParVectorOwnsData(vector) = 1;
   hypre_ParVectorOwnsPartitioning(vector) = 1;

   return vector;
}

/*--------------------------------------------------------------------------
 * hypre_ParMultiVectorCreate
 *--------------------------------------------------------------------------*/

hypre_ParVector *
hypre_ParMultiVectorCreate(  MPI_Comm comm,
			int global_size, 
			int *partitioning,
                        int num_vectors )
{
   /* note that global_size is the global length of a single vector */
   hypre_ParVector * vector = hypre_ParVectorCreate( comm, global_size, partitioning );
   hypre_ParVectorNumVectors(vector) = num_vectors;
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
 * hypre_ParVectorSetNumVectors
 * call before calling hypre_ParVectorInitialize
 * probably this will do more harm than good, use hypre_ParMultiVectorCreate
 *--------------------------------------------------------------------------*/
#if 0
int 
hypre_ParVectorSetNumVectors( hypre_ParVector *vector,
                              int num_vectors )
{
   int    ierr=0;
   hypre_Vector *local_vector = hypre_ParVectorLocalVector(v);

   hypre_SeqVectorSetNumVectors( local_vector, num_vectors );

   return ierr;
}
#endif
/*--------------------------------------------------------------------------
 * hypre_ParVectorRead
 *--------------------------------------------------------------------------*/

hypre_ParVector
*hypre_ParVectorRead( MPI_Comm    comm,
                      const char *file_name )
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
#ifdef HYPRE_NO_GLOBAL_PARTITION
   for (i=0; i < 2; i++)
	fscanf(fp, "%d\n", &partitioning[i]);
   fclose (fp);
#else
   for (i=0; i < num_procs; i++)
	fscanf(fp, "%d\n", &partitioning[i]);
   fclose (fp);
   partitioning[num_procs] = global_size; 
#endif
   par_vector = hypre_CTAlloc(hypre_ParVector, 1);
	
   hypre_ParVectorComm(par_vector) = comm;
   hypre_ParVectorGlobalSize(par_vector) = global_size;

#ifdef HYPRE_NO_GLOBAL_PARTITION
   hypre_ParVectorFirstIndex(par_vector) = partitioning[0];
   hypre_ParVectorLastIndex(par_vector) = partitioning[1]-1;
#else
   hypre_ParVectorFirstIndex(par_vector) = partitioning[my_id];
   hypre_ParVectorLastIndex(par_vector) = partitioning[my_id+1]-1;
#endif

   hypre_ParVectorPartitioning(par_vector) = partitioning;

   hypre_ParVectorOwnsData(par_vector) = 1;
   hypre_ParVectorOwnsPartitioning(par_vector) = 1;

   sprintf(new_file_name,"%s.%d",file_name,my_id); 
   hypre_ParVectorLocalVector(par_vector) = hypre_SeqVectorRead(new_file_name);

   /* multivector code not written yet >>> */
   hypre_assert( hypre_ParVectorNumVectors(par_vector) == 1 );

   return par_vector;
}

/*--------------------------------------------------------------------------
 * hypre_ParVectorPrint
 *--------------------------------------------------------------------------*/

int
hypre_ParVectorPrint( hypre_ParVector  *vector, 
                      const char       *file_name )
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
#ifdef HYPRE_NO_GLOBAL_PARTITION
   for (i=0; i < 2; i++)
	fprintf(fp, "%d\n", partitioning[i]);
#else
  for (i=0; i < num_procs; i++)
	fprintf(fp, "%d\n", partitioning[i]);
#endif

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
 * hypre_ParVectorCloneShallow
 * returns a complete copy of a hypre_ParVector x - a shallow copy, re-using
 * the partitioning and data arrays of x
 *--------------------------------------------------------------------------*/

hypre_ParVector *
hypre_ParVectorCloneShallow( hypre_ParVector *x )
{
   hypre_ParVector * y = hypre_ParVectorCreate(
      hypre_ParVectorComm(x), hypre_ParVectorGlobalSize(x), hypre_ParVectorPartitioning(x) );

   hypre_ParVectorOwnsData(y) = 1;
   /* ...This vector owns its local vector, although the local vector doesn't own _its_ data */
   hypre_ParVectorOwnsPartitioning(y) = 0;
   hypre_SeqVectorDestroy( hypre_ParVectorLocalVector(y) );
   hypre_ParVectorLocalVector(y) = hypre_SeqVectorCloneShallow(
      hypre_ParVectorLocalVector(x) );
   hypre_ParVectorFirstIndex(y) = hypre_ParVectorFirstIndex(x);

   return y;
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
 *
 * this is not being optimized to use HYPRE_NO_GLOBAL_PARTITION
 *--------------------------------------------------------------------------*/

hypre_ParVector *
hypre_VectorToParVector (MPI_Comm comm, hypre_Vector *v, int *vec_starts)
{
   int 			global_size;
   int 			local_size;
   int                  num_vectors;
   int  		num_procs, my_id;
   int                  global_vecstride, vecstride, idxstride;
   hypre_ParVector  	*par_vector;
   hypre_Vector     	*local_vector;
   double          	*v_data;
   double		*local_data;
   MPI_Request		*requests;
   MPI_Status		*status, status0;
   int			i, j, k, p;

   MPI_Comm_size(comm,&num_procs);
   MPI_Comm_rank(comm,&my_id);

   if (my_id == 0)
   {
        global_size = hypre_VectorSize(v);
        v_data = hypre_VectorData(v);
        num_vectors = hypre_VectorNumVectors(v); /* for multivectors */
        global_vecstride = hypre_VectorVectorStride(v);

   }

   MPI_Bcast(&global_size,1,MPI_INT,0,comm);
   MPI_Bcast(&num_vectors,1,MPI_INT,0,comm);
   MPI_Bcast(&global_vecstride,1,MPI_INT,0,comm);

   if ( num_vectors==1 )
      par_vector = hypre_ParVectorCreate(comm, global_size, vec_starts);
   else
      par_vector = hypre_ParMultiVectorCreate(comm, global_size, vec_starts, num_vectors);

   vec_starts = hypre_ParVectorPartitioning(par_vector);

   local_size = vec_starts[my_id+1] - vec_starts[my_id];

   hypre_ParVectorInitialize(par_vector);
   local_vector = hypre_ParVectorLocalVector(par_vector);
   local_data = hypre_VectorData(local_vector);
   vecstride = hypre_VectorVectorStride(local_vector);
   idxstride = hypre_VectorIndexStride(local_vector);
   hypre_assert( idxstride==1 );  /* <<< so far only the only implemented multivector StorageMethod is 0 <<< */

   if (my_id == 0)
   {
	requests = hypre_CTAlloc(MPI_Request,num_vectors*(num_procs-1));
	status = hypre_CTAlloc(MPI_Status,num_vectors*(num_procs-1));
	k = 0;
	for ( p=1; p<num_procs; p++)
           for ( j=0; j<num_vectors; ++j )
           {
		MPI_Isend( &v_data[vec_starts[p]]+j*global_vecstride,
                          (vec_starts[p+1]-vec_starts[p]),
                          MPI_DOUBLE, p, 0, comm, &requests[k++] );
           }
        if ( num_vectors==1 )
        {
           for (i=0; i < local_size; i++)
              local_data[i] = v_data[i];
        }
        else
           for ( j=0; j<num_vectors; ++j )
           {
              for (i=0; i < local_size; i++)
                 local_data[i+j*vecstride] = v_data[i+j*global_vecstride];
           }
	MPI_Waitall(num_procs-1,requests, status);
	hypre_TFree(requests);
	hypre_TFree(status);
   }
   else
   {
      for ( j=0; j<num_vectors; ++j )
	MPI_Recv( local_data+j*vecstride, local_size, MPI_DOUBLE, 0, 0, comm,&status0 );
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
#ifndef HYPRE_NO_GLOBAL_PARTITION
   int 			*vec_starts = hypre_ParVectorPartitioning(par_v);
#endif
   hypre_Vector     	*local_vector = hypre_ParVectorLocalVector(par_v);
   int  		num_procs, my_id;
   int                  num_vectors = hypre_ParVectorNumVectors(par_v);
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

#ifdef HYPRE_NO_GLOBAL_PARTITION

   int *new_vec_starts;
   
   int num_contacts;
   int contact_proc_list[1];
   int contact_send_buf[1];
   int contact_send_buf_starts[2];
   int max_response_size;
   int *response_recv_buf=NULL;
   int *response_recv_buf_starts = NULL;
   hypre_DataExchangeResponse response_obj;
   hypre_ProcListElements send_proc_obj;
   
   int *send_info = NULL;
   MPI_Status  status1;
   int count, tag1 = 112, tag2 = 223;
   int start;
   
#endif


   MPI_Comm_size(comm, &num_procs);
   MPI_Comm_rank(comm, &my_id);

#ifdef HYPRE_NO_GLOBAL_PARTITION

  local_size = hypre_ParVectorLastIndex(par_v) - 
     hypre_ParVectorFirstIndex(par_v) + 1;

 

/* determine procs which hold data of par_v and store ids in used_procs */
/* we need to do an exchange data for this.  If I own row then I will contact
   processor 0 with the endpoint of my local range */


   if (local_size > 0)
   {
      num_contacts = 1;
      contact_proc_list[0] = 0;
      contact_send_buf[0] =  hypre_ParVectorLastIndex(par_v);
      contact_send_buf_starts[0] = 0;
      contact_send_buf_starts[1] = 1;
   }
   else
   {
      num_contacts = 0;
      contact_send_buf_starts[0] = 0;
      contact_send_buf_starts[1] = 0;
   }

   /*build the response object*/
   /*send_proc_obj will  be for saving info from contacts */
   send_proc_obj.length = 0;
   send_proc_obj.storage_length = 10;
   send_proc_obj.id = hypre_CTAlloc(int, send_proc_obj.storage_length);
   send_proc_obj.vec_starts = hypre_CTAlloc(int, send_proc_obj.storage_length + 1); 
   send_proc_obj.vec_starts[0] = 0;
   send_proc_obj.element_storage_length = 10;
   send_proc_obj.elements = hypre_CTAlloc(int, send_proc_obj.element_storage_length);

   max_response_size = 0; /* each response is null */
   response_obj.fill_response = hypre_FillResponseParToVectorAll;
   response_obj.data1 = NULL;
   response_obj.data2 = &send_proc_obj; /*this is where we keep info from contacts*/
  
   
   hypre_DataExchangeList(num_contacts, 
                          contact_proc_list, contact_send_buf, 
                          contact_send_buf_starts, sizeof(int), 
                          sizeof(int), &response_obj, 
                          max_response_size, 1,
                          comm, (void**) &response_recv_buf,	   
                          &response_recv_buf_starts);

 /* now processor 0 should have a list of ranges for processors that have rows -
      these are in send_proc_obj - it needs to create the new list of processors
      and also an array of vec starts - and send to those who own row*/
   if (my_id)
   {
      if (local_size)      
      {
         /* look for a message from processor 0 */         
         MPI_Probe(0, tag1, comm, &status1);
         MPI_Get_count(&status1, MPI_INT, &count);
         
         send_info = hypre_CTAlloc(int, count);
         MPI_Recv(send_info, count, MPI_INT, 0, tag1, comm, &status1);

         /* now unpack */  
         num_types = send_info[0];
         used_procs =  hypre_CTAlloc(int, num_types);  
         new_vec_starts = hypre_CTAlloc(int, num_types+1);

         for (i=1; i<= num_types; i++)
         {
            used_procs[i-1] = send_info[i];
         }
         for (i=num_types+1; i< count; i++)
         {
            new_vec_starts[i-num_types-1] = send_info[i] ;
         }
      }
      else /* clean up and exit */
      {
         hypre_TFree(send_proc_obj.vec_starts);
         hypre_TFree(send_proc_obj.id);
         hypre_TFree(send_proc_obj.elements);
         if(response_recv_buf)        hypre_TFree(response_recv_buf);
         if(response_recv_buf_starts) hypre_TFree(response_recv_buf_starts);
         return NULL;
      }
   }
   else /* my_id ==0 */
   {
      num_types = send_proc_obj.length;
      used_procs =  hypre_CTAlloc(int, num_types);  
      new_vec_starts = hypre_CTAlloc(int, num_types+1);
      
      new_vec_starts[0] = 0;
      for (i=0; i< num_types; i++)
      {
         used_procs[i] = send_proc_obj.id[i];
         new_vec_starts[i+1] = send_proc_obj.elements[i]+1;
      }
      qsort0(used_procs, 0, num_types-1);
      qsort0(new_vec_starts, 0, num_types);
      /*now we need to put into an array to send */
      count =  2*num_types+2;
      send_info = hypre_CTAlloc(int, count);
      send_info[0] = num_types;
      for (i=1; i<= num_types; i++)
      {
         send_info[i] = used_procs[i-1];
      }
      for (i=num_types+1; i< count; i++)
      {
         send_info[i] = new_vec_starts[i-num_types-1];
      }
      requests = hypre_CTAlloc(MPI_Request, num_types);
      status =  hypre_CTAlloc(MPI_Status, num_types);

      /* don't send to myself  - these are sorted so my id would be first*/
      start = 0;
      if (used_procs[0] == 0)
      {
         start = 1;
      }
   
      
      for (i=start; i < num_types; i++)
      {
         MPI_Isend(send_info, count, MPI_INT, used_procs[i], tag1, comm, &requests[i-start]);
      }
      MPI_Waitall(num_types-start, requests, status);

      hypre_TFree(status);
      hypre_TFree(requests);
   }

   /* clean up */
   hypre_TFree(send_proc_obj.vec_starts);
   hypre_TFree(send_proc_obj.id);
   hypre_TFree(send_proc_obj.elements);
   hypre_TFree(send_info);
   if(response_recv_buf)        hypre_TFree(response_recv_buf);
   if(response_recv_buf_starts) hypre_TFree(response_recv_buf_starts);

   /* now proc 0 can exit if it has no rows */
   if (!local_size) {
      hypre_TFree(used_procs);
      hypre_TFree(new_vec_starts);
      return NULL;
   }
   
   /* everyone left has rows and knows: new_vec_starts, num_types, and used_procs */

  /* this vector should be rather small */

   local_data = hypre_VectorData(local_vector);
   vector = hypre_SeqVectorCreate(global_size);
   hypre_VectorNumVectors(vector) = num_vectors;
   hypre_SeqVectorInitialize(vector);
   vector_data = hypre_VectorData(vector);

   num_requests = 2*num_types;

   requests = hypre_CTAlloc(MPI_Request, num_requests);
   status = hypre_CTAlloc(MPI_Status, num_requests);

/* initialize data exchange among used_procs and generate vector  - here we 
   send to ourself also*/
 
   j = 0;
   for (i = 0; i < num_types; i++)
   {
        proc_id = used_procs[i];
        vec_len = new_vec_starts[i+1] - new_vec_starts[i];
        MPI_Irecv(&vector_data[new_vec_starts[i]], num_vectors*vec_len, MPI_DOUBLE,
                                proc_id, tag2, comm, &requests[j++]);
   }
   for (i = 0; i < num_types; i++)
   {
        MPI_Isend(local_data, num_vectors*local_size, MPI_DOUBLE, used_procs[i],
                          tag2, comm, &requests[j++]);
   }
 
   MPI_Waitall(num_requests, requests, status);


   if (num_requests)
   {
   	hypre_TFree(requests);
   	hypre_TFree(status); 
        hypre_TFree(used_procs);
   }

   hypre_TFree(new_vec_starts);
   


#else
   local_size = vec_starts[my_id+1] - vec_starts[my_id];

/* if my_id contains no data, return NULL  */

   if (!local_size)
	return NULL;
 
   local_data = hypre_VectorData(local_vector);
   vector = hypre_SeqVectorCreate(global_size);
   hypre_VectorNumVectors(vector) = num_vectors;
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
        MPI_Irecv(&vector_data[vec_starts[proc_id]], num_vectors*vec_len, MPI_DOUBLE,
                                proc_id, 0, comm, &requests[j++]);
   }
   for (i = 0; i < num_types; i++)
   {
        MPI_Isend(local_data, num_vectors*local_size, MPI_DOUBLE, used_procs[i],
                          0, comm, &requests[j++]);
   }
 
   for (i=0; i < num_vectors*local_size; i++)
        vector_data[vec_starts[my_id]+i] = local_data[i];
 
   MPI_Waitall(num_requests, requests, status);

   if (num_requests)
   {
   	hypre_TFree(used_procs);
   	hypre_TFree(requests);
   	hypre_TFree(status); 
   }


#endif

   return vector;
}

/*--------------------------------------------------------------------------
 * hypre_ParVectorPrintIJ
 *--------------------------------------------------------------------------*/

int
hypre_ParVectorPrintIJ( hypre_ParVector *vector,
                        int              base_j,
                        const char      *filename )
{
   int ierr = 0;
   MPI_Comm          comm         = hypre_ParVectorComm(vector);
   int               global_size  = hypre_ParVectorGlobalSize(vector);
   int              *partitioning = hypre_ParVectorPartitioning(vector);
   double           *local_data;
   int               myid, num_procs, i, j, part0;
   char              new_filename[255];
   FILE             *file;

   /* multivector code not written yet >>> */
   hypre_assert( hypre_ParVectorNumVectors(vector) == 1 );
   if ( hypre_ParVectorNumVectors(vector) != 1 ) ++ierr;

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
#ifdef HYPRE_NO_GLOBAL_PARTITION
   for (i=0; i <= 2; i++)
#else
   for (i=0; i <= num_procs; i++)
#endif
   {
      fprintf(file, "%d \n", partitioning[i] + base_j);
   }

#ifdef HYPRE_NO_GLOBAL_PARTITION
   part0 = partitioning[0];
   for (j = part0; j < partitioning[1]; j++)
#else
   part0 = partitioning[myid];
   for (j = part0; j < partitioning[myid+1]; j++)
#endif
   {
      fprintf(file, "%d %e\n", j + base_j, local_data[j-part0]);
   }

   fclose(file);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_ParVectorReadIJ
 *--------------------------------------------------------------------------*/

int
hypre_ParVectorReadIJ( MPI_Comm             comm,
                       const char          *filename,
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
#ifdef HYPRE_NO_GLOBAL_PARTITION
/* this may need to be changed so that the base is available in the file! */
   partitioning = hypre_CTAlloc(int,2);

   fscanf(file, "%d", partitioning);
   for (i = 0; i < 2; i++)
   {
      fscanf(file, "%d", partitioning+i);
   }
#else
   partitioning = hypre_CTAlloc(int,num_procs+1);

   fscanf(file, "%d", partitioning);
   for (i = 1; i <= num_procs; i++)
   {
      fscanf(file, "%d", partitioning+i);
      partitioning[i] -= partitioning[0];
   }
   base_j = partitioning[0];
   partitioning[0] = 0;
#endif
   vector = hypre_ParVectorCreate(comm, global_size,
                                  partitioning);

   hypre_ParVectorInitialize(vector);

   local_vector = hypre_ParVectorLocalVector(vector);
   local_data   = hypre_VectorData(local_vector);

#ifdef HYPRE_NO_GLOBAL_PARTITION
   for (j = 0; j < partitioning[1] - partitioning[0]; j++)
#else
   for (j = 0; j < partitioning[myid+1] - partitioning[myid]; j++)
#endif
   {
      fscanf(file, "%d %le", &J, local_data + j);
   }

   fclose(file);

   *base_j_ptr = base_j;
   *vector_ptr = vector;

   /* multivector code not written yet >>> */
   hypre_assert( hypre_ParVectorNumVectors(vector) == 1 );
   if ( hypre_ParVectorNumVectors(vector) != 1 ) ++ierr;

   return ierr;
}


/*--------------------------------------------------------------------
 * hypre_FillResponseParToVectorAll
 * Fill response function for determining the send processors
 * data exchange
 *--------------------------------------------------------------------*/

int
hypre_FillResponseParToVectorAll(void *p_recv_contact_buf, 
                                 int contact_size, int contact_proc, void *ro, 
                                 MPI_Comm comm, void **p_send_response_buf, 
                                 int *response_message_size )
{
   int    myid;
   int    i, index, count, elength;

   int    *recv_contact_buf = (int * ) p_recv_contact_buf;

   hypre_DataExchangeResponse  *response_obj = ro;  

   hypre_ProcListElements      *send_proc_obj = response_obj->data2;   


   MPI_Comm_rank(comm, &myid );


   /*check to see if we need to allocate more space in send_proc_obj for ids*/
   if (send_proc_obj->length == send_proc_obj->storage_length)
   {
      send_proc_obj->storage_length +=10; /*add space for 10 more processors*/
      send_proc_obj->id = hypre_TReAlloc(send_proc_obj->id,int, 
					 send_proc_obj->storage_length);
      send_proc_obj->vec_starts = hypre_TReAlloc(send_proc_obj->vec_starts,int, 
                                  send_proc_obj->storage_length + 1);
   }
  
   /*initialize*/ 
   count = send_proc_obj->length;
   index = send_proc_obj->vec_starts[count]; /*this is the number of elements*/

   /*send proc*/ 
   send_proc_obj->id[count] = contact_proc; 

   /*do we need more storage for the elements?*/
     if (send_proc_obj->element_storage_length < index + contact_size)
   {
      elength = hypre_max(contact_size, 10);   
      elength += index;
      send_proc_obj->elements = hypre_TReAlloc(send_proc_obj->elements, 
					       int, elength);
      send_proc_obj->element_storage_length = elength; 
   }
   /*populate send_proc_obj*/
   for (i=0; i< contact_size; i++) 
   { 
      send_proc_obj->elements[index++] = recv_contact_buf[i];
   }
   send_proc_obj->vec_starts[count+1] = index;
   send_proc_obj->length++;
   

  /*output - no message to return (confirmation) */
   *response_message_size = 0; 
  
   
   return(0);

}

/* -----------------------------------------------------------------------------
 * return the sum of all local elements of the vector
 * ----------------------------------------------------------------------------- */

double hypre_ParVectorLocalSumElts( hypre_ParVector * vector )
{
   return hypre_VectorSumElts( hypre_ParVectorLocalVector(vector) );
}
