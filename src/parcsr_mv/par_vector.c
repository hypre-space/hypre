/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.23 $
 ***********************************************************************EHEADER*/




/******************************************************************************
 *
 * Member functions for hypre_Vector class.
 *
 *****************************************************************************/

#include "headers.h"
#include <assert.h>

#ifdef HYPRE_NO_GLOBAL_PARTITION
HYPRE_Int hypre_FillResponseParToVectorAll(void*, HYPRE_Int, HYPRE_Int, void*, MPI_Comm, void**, HYPRE_Int*);
#endif


/*--------------------------------------------------------------------------
 * hypre_ParVectorCreate
 *--------------------------------------------------------------------------*/

/* If create is called for HYPRE_NO_GLOBAL_PARTITION and partitioning is NOT null,
   then it is assumed that it is array of length 2 containing the start row of 
   the calling processor followed by the start row of the next processor - AHB 6/05 */


hypre_ParVector *
hypre_ParVectorCreate(  MPI_Comm comm,
			HYPRE_Int global_size, 
			HYPRE_Int *partitioning)
{
   hypre_ParVector  *vector;
   HYPRE_Int num_procs, my_id;

   if (global_size < 0)
   {
      hypre_error_in_arg(2);
      return NULL;
   }
   vector = hypre_CTAlloc(hypre_ParVector, 1);
   hypre_MPI_Comm_rank(comm,&my_id);

   if (!partitioning)
   {
     hypre_MPI_Comm_size(comm,&num_procs);
#ifdef HYPRE_NO_GLOBAL_PARTITION
     hypre_GenerateLocalPartitioning(global_size, num_procs, my_id, &partitioning);
#else
     hypre_GeneratePartitioning(global_size, num_procs, &partitioning);
#endif
   }


   hypre_ParVectorAssumedPartition(vector) = NULL;
   

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
			HYPRE_Int global_size, 
			HYPRE_Int *partitioning,
                        HYPRE_Int num_vectors )
{
   /* note that global_size is the global length of a single vector */
   hypre_ParVector * vector = hypre_ParVectorCreate( comm, global_size, partitioning );
   hypre_ParVectorNumVectors(vector) = num_vectors;
   return vector;
}


/*--------------------------------------------------------------------------
 * hypre_ParVectorDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int 
hypre_ParVectorDestroy( hypre_ParVector *vector )
{
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

      if (hypre_ParVectorAssumedPartition(vector))
         hypre_ParVectorDestroyAssumedPartition(vector);



      hypre_TFree(vector);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParVectorInitialize
 *--------------------------------------------------------------------------*/

HYPRE_Int 
hypre_ParVectorInitialize( hypre_ParVector *vector )
{
   if (!vector)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   hypre_SeqVectorInitialize(hypre_ParVectorLocalVector(vector));

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParVectorSetDataOwner
 *--------------------------------------------------------------------------*/

HYPRE_Int 
hypre_ParVectorSetDataOwner( hypre_ParVector *vector,
                             HYPRE_Int           owns_data   )
{

   if (!vector)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   hypre_ParVectorOwnsData(vector) = owns_data;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParVectorSetPartitioningOwner
 *--------------------------------------------------------------------------*/

HYPRE_Int 
hypre_ParVectorSetPartitioningOwner( hypre_ParVector *vector,
                             	     HYPRE_Int owns_partitioning)
{
   if (!vector)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   hypre_ParVectorOwnsPartitioning(vector) = owns_partitioning;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParVectorSetNumVectors
 * call before calling hypre_ParVectorInitialize
 * probably this will do more harm than good, use hypre_ParMultiVectorCreate
 *--------------------------------------------------------------------------*/
#if 0
HYPRE_Int 
hypre_ParVectorSetNumVectors( hypre_ParVector *vector,
                              HYPRE_Int num_vectors )
{
   HYPRE_Int    ierr=0;
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
   HYPRE_Int  	my_id, num_procs;
   HYPRE_Int		*partitioning;
   HYPRE_Int		global_size, i;
   FILE		*fp;

   hypre_MPI_Comm_rank(comm,&my_id); 
   hypre_MPI_Comm_size(comm,&num_procs); 

   partitioning = hypre_CTAlloc(HYPRE_Int,num_procs+1);

   hypre_sprintf(new_file_name,"%s.INFO.%d",file_name,my_id); 
   fp = fopen(new_file_name, "r");
   hypre_fscanf(fp, "%d\n", &global_size);
#ifdef HYPRE_NO_GLOBAL_PARTITION
   for (i=0; i < 2; i++)
	hypre_fscanf(fp, "%d\n", &partitioning[i]);
   fclose (fp);
#else
   for (i=0; i < num_procs; i++)
	hypre_fscanf(fp, "%d\n", &partitioning[i]);
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

   hypre_sprintf(new_file_name,"%s.%d",file_name,my_id); 
   hypre_ParVectorLocalVector(par_vector) = hypre_SeqVectorRead(new_file_name);

   /* multivector code not written yet >>> */
   hypre_assert( hypre_ParVectorNumVectors(par_vector) == 1 );

   return par_vector;
}

/*--------------------------------------------------------------------------
 * hypre_ParVectorPrint
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParVectorPrint( hypre_ParVector  *vector, 
                      const char       *file_name )
{
   char 	new_file_name[80];
   hypre_Vector *local_vector;
   MPI_Comm 	comm;
   HYPRE_Int  	my_id, num_procs, i;
   HYPRE_Int		*partitioning;
   HYPRE_Int		global_size;
   FILE		*fp;
   if (!vector)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   local_vector = hypre_ParVectorLocalVector(vector); 
   comm = hypre_ParVectorComm(vector);
   partitioning = hypre_ParVectorPartitioning(vector); 
   global_size = hypre_ParVectorGlobalSize(vector); 

   hypre_MPI_Comm_rank(comm,&my_id); 
   hypre_MPI_Comm_size(comm,&num_procs); 
   hypre_sprintf(new_file_name,"%s.%d",file_name,my_id); 
   hypre_SeqVectorPrint(local_vector,new_file_name);
   hypre_sprintf(new_file_name,"%s.INFO.%d",file_name,my_id); 
   fp = fopen(new_file_name, "w");
   hypre_fprintf(fp, "%d\n", global_size);
#ifdef HYPRE_NO_GLOBAL_PARTITION
   for (i=0; i < 2; i++)
	hypre_fprintf(fp, "%d\n", partitioning[i]);
#else
  for (i=0; i < num_procs; i++)
	hypre_fprintf(fp, "%d\n", partitioning[i]);
#endif

   fclose (fp);
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParVectorSetConstantValues
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParVectorSetConstantValues( hypre_ParVector *v,
                                  double        value )
{
   hypre_Vector *v_local = hypre_ParVectorLocalVector(v);
           
   return hypre_SeqVectorSetConstantValues(v_local,value);
}

/*--------------------------------------------------------------------------
 * hypre_ParVectorSetRandomValues
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParVectorSetRandomValues( hypre_ParVector *v,
                                HYPRE_Int            seed )
{
   HYPRE_Int my_id;
   hypre_Vector *v_local = hypre_ParVectorLocalVector(v);

   MPI_Comm 	comm = hypre_ParVectorComm(v);
   hypre_MPI_Comm_rank(comm,&my_id); 

   seed *= (my_id+1);
           
   return hypre_SeqVectorSetRandomValues(v_local,seed);
}

/*--------------------------------------------------------------------------
 * hypre_ParVectorCopy
 *--------------------------------------------------------------------------*/

HYPRE_Int
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

HYPRE_Int
hypre_ParVectorScale( double        alpha,
                      hypre_ParVector *y     )
{
   hypre_Vector *y_local = hypre_ParVectorLocalVector(y);

   return hypre_SeqVectorScale( alpha, y_local);
}

/*--------------------------------------------------------------------------
 * hypre_ParVectorAxpy
 *--------------------------------------------------------------------------*/

HYPRE_Int
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
   
   hypre_MPI_Allreduce(&local_result, &result, 1, hypre_MPI_DOUBLE, hypre_MPI_SUM, comm);
   
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
hypre_VectorToParVector (MPI_Comm comm, hypre_Vector *v, HYPRE_Int *vec_starts)
{
   HYPRE_Int 			global_size;
   HYPRE_Int 			local_size;
   HYPRE_Int                  num_vectors;
   HYPRE_Int  		num_procs, my_id;
   HYPRE_Int                  global_vecstride, vecstride, idxstride;
   hypre_ParVector  	*par_vector;
   hypre_Vector     	*local_vector;
   double          	*v_data;
   double		*local_data;
   hypre_MPI_Request		*requests;
   hypre_MPI_Status		*status, status0;
   HYPRE_Int			i, j, k, p;

   hypre_MPI_Comm_size(comm,&num_procs);
   hypre_MPI_Comm_rank(comm,&my_id);

   if (my_id == 0)
   {
        global_size = hypre_VectorSize(v);
        v_data = hypre_VectorData(v);
        num_vectors = hypre_VectorNumVectors(v); /* for multivectors */
        global_vecstride = hypre_VectorVectorStride(v);

   }

   hypre_MPI_Bcast(&global_size,1,HYPRE_MPI_INT,0,comm);
   hypre_MPI_Bcast(&num_vectors,1,HYPRE_MPI_INT,0,comm);
   hypre_MPI_Bcast(&global_vecstride,1,HYPRE_MPI_INT,0,comm);

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
	requests = hypre_CTAlloc(hypre_MPI_Request,num_vectors*(num_procs-1));
	status = hypre_CTAlloc(hypre_MPI_Status,num_vectors*(num_procs-1));
	k = 0;
	for ( p=1; p<num_procs; p++)
           for ( j=0; j<num_vectors; ++j )
           {
		hypre_MPI_Isend( &v_data[vec_starts[p]]+j*global_vecstride,
                          (vec_starts[p+1]-vec_starts[p]),
                          hypre_MPI_DOUBLE, p, 0, comm, &requests[k++] );
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
	hypre_MPI_Waitall(num_procs-1,requests, status);
	hypre_TFree(requests);
	hypre_TFree(status);
   }
   else
   {
      for ( j=0; j<num_vectors; ++j )
	hypre_MPI_Recv( local_data+j*vecstride, local_size, hypre_MPI_DOUBLE, 0, 0, comm,&status0 );
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
   HYPRE_Int 			global_size = hypre_ParVectorGlobalSize(par_v);
#ifndef HYPRE_NO_GLOBAL_PARTITION
   HYPRE_Int 			*vec_starts = hypre_ParVectorPartitioning(par_v);
#endif
   hypre_Vector     	*local_vector = hypre_ParVectorLocalVector(par_v);
   HYPRE_Int  		num_procs, my_id;
   HYPRE_Int                  num_vectors = hypre_ParVectorNumVectors(par_v);
   hypre_Vector  	*vector;
   double		*vector_data;
   double		*local_data;
   HYPRE_Int 			local_size;
   hypre_MPI_Request		*requests;
   hypre_MPI_Status		*status;
   HYPRE_Int			i, j;
   HYPRE_Int			*used_procs;
   HYPRE_Int			num_types, num_requests;
   HYPRE_Int			vec_len, proc_id;

#ifdef HYPRE_NO_GLOBAL_PARTITION

   HYPRE_Int *new_vec_starts;
   
   HYPRE_Int num_contacts;
   HYPRE_Int contact_proc_list[1];
   HYPRE_Int contact_send_buf[1];
   HYPRE_Int contact_send_buf_starts[2];
   HYPRE_Int max_response_size;
   HYPRE_Int *response_recv_buf=NULL;
   HYPRE_Int *response_recv_buf_starts = NULL;
   hypre_DataExchangeResponse response_obj;
   hypre_ProcListElements send_proc_obj;
   
   HYPRE_Int *send_info = NULL;
   hypre_MPI_Status  status1;
   HYPRE_Int count, tag1 = 112, tag2 = 223;
   HYPRE_Int start;
   
#endif


   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

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
   send_proc_obj.id = hypre_CTAlloc(HYPRE_Int, send_proc_obj.storage_length);
   send_proc_obj.vec_starts = hypre_CTAlloc(HYPRE_Int, send_proc_obj.storage_length + 1); 
   send_proc_obj.vec_starts[0] = 0;
   send_proc_obj.element_storage_length = 10;
   send_proc_obj.elements = hypre_CTAlloc(HYPRE_Int, send_proc_obj.element_storage_length);

   max_response_size = 0; /* each response is null */
   response_obj.fill_response = hypre_FillResponseParToVectorAll;
   response_obj.data1 = NULL;
   response_obj.data2 = &send_proc_obj; /*this is where we keep info from contacts*/
  
   
   hypre_DataExchangeList(num_contacts, 
                          contact_proc_list, contact_send_buf, 
                          contact_send_buf_starts, sizeof(HYPRE_Int), 
                          sizeof(HYPRE_Int), &response_obj, 
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
         hypre_MPI_Probe(0, tag1, comm, &status1);
         hypre_MPI_Get_count(&status1, HYPRE_MPI_INT, &count);
         
         send_info = hypre_CTAlloc(HYPRE_Int, count);
         hypre_MPI_Recv(send_info, count, HYPRE_MPI_INT, 0, tag1, comm, &status1);

         /* now unpack */  
         num_types = send_info[0];
         used_procs =  hypre_CTAlloc(HYPRE_Int, num_types);  
         new_vec_starts = hypre_CTAlloc(HYPRE_Int, num_types+1);

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
      used_procs =  hypre_CTAlloc(HYPRE_Int, num_types);  
      new_vec_starts = hypre_CTAlloc(HYPRE_Int, num_types+1);
      
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
      send_info = hypre_CTAlloc(HYPRE_Int, count);
      send_info[0] = num_types;
      for (i=1; i<= num_types; i++)
      {
         send_info[i] = used_procs[i-1];
      }
      for (i=num_types+1; i< count; i++)
      {
         send_info[i] = new_vec_starts[i-num_types-1];
      }
      requests = hypre_CTAlloc(hypre_MPI_Request, num_types);
      status =  hypre_CTAlloc(hypre_MPI_Status, num_types);

      /* don't send to myself  - these are sorted so my id would be first*/
      start = 0;
      if (used_procs[0] == 0)
      {
         start = 1;
      }
   
      
      for (i=start; i < num_types; i++)
      {
         hypre_MPI_Isend(send_info, count, HYPRE_MPI_INT, used_procs[i], tag1, comm, &requests[i-start]);
      }
      hypre_MPI_Waitall(num_types-start, requests, status);

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

   requests = hypre_CTAlloc(hypre_MPI_Request, num_requests);
   status = hypre_CTAlloc(hypre_MPI_Status, num_requests);

/* initialize data exchange among used_procs and generate vector  - here we 
   send to ourself also*/
 
   j = 0;
   for (i = 0; i < num_types; i++)
   {
        proc_id = used_procs[i];
        vec_len = new_vec_starts[i+1] - new_vec_starts[i];
        hypre_MPI_Irecv(&vector_data[new_vec_starts[i]], num_vectors*vec_len, hypre_MPI_DOUBLE,
                                proc_id, tag2, comm, &requests[j++]);
   }
   for (i = 0; i < num_types; i++)
   {
        hypre_MPI_Isend(local_data, num_vectors*local_size, hypre_MPI_DOUBLE, used_procs[i],
                          tag2, comm, &requests[j++]);
   }
 
   hypre_MPI_Waitall(num_requests, requests, status);


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
 
   used_procs = hypre_CTAlloc(HYPRE_Int, num_types);
   j = 0;
   for (i=0; i < num_procs; i++)
        if (vec_starts[i+1]-vec_starts[i] && i-my_id)
                used_procs[j++] = i;
 
   requests = hypre_CTAlloc(hypre_MPI_Request, num_requests);
   status = hypre_CTAlloc(hypre_MPI_Status, num_requests);

/* initialize data exchange among used_procs and generate vector */
 
   j = 0;
   for (i = 0; i < num_types; i++)
   {
        proc_id = used_procs[i];
        vec_len = vec_starts[proc_id+1] - vec_starts[proc_id];
        hypre_MPI_Irecv(&vector_data[vec_starts[proc_id]], num_vectors*vec_len, hypre_MPI_DOUBLE,
                                proc_id, 0, comm, &requests[j++]);
   }
   for (i = 0; i < num_types; i++)
   {
        hypre_MPI_Isend(local_data, num_vectors*local_size, hypre_MPI_DOUBLE, used_procs[i],
                          0, comm, &requests[j++]);
   }
 
   for (i=0; i < num_vectors*local_size; i++)
        vector_data[vec_starts[my_id]+i] = local_data[i];
 
   hypre_MPI_Waitall(num_requests, requests, status);

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

HYPRE_Int
hypre_ParVectorPrintIJ( hypre_ParVector *vector,
                        HYPRE_Int              base_j,
                        const char      *filename )
{
   MPI_Comm          comm;
   HYPRE_Int               global_size;
   HYPRE_Int              *partitioning;
   double           *local_data;
   HYPRE_Int               myid, num_procs, i, j, part0;
   char              new_filename[255];
   FILE             *file;
   if (!vector)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   comm         = hypre_ParVectorComm(vector);
   global_size  = hypre_ParVectorGlobalSize(vector);
   partitioning = hypre_ParVectorPartitioning(vector);

   /* multivector code not written yet >>> */
   hypre_assert( hypre_ParVectorNumVectors(vector) == 1 );
   if ( hypre_ParVectorNumVectors(vector) != 1 ) hypre_error_in_arg(1);

   hypre_MPI_Comm_rank(comm, &myid);
   hypre_MPI_Comm_size(comm, &num_procs);
  
   hypre_sprintf(new_filename,"%s.%05d", filename, myid);

   if ((file = fopen(new_filename, "w")) == NULL)
   {
      hypre_printf("Error: can't open output file %s\n", new_filename);
      hypre_error_in_arg(3);
      return hypre_error_flag;
   }

   local_data = hypre_VectorData(hypre_ParVectorLocalVector(vector));

   hypre_fprintf(file, "%d \n", global_size);
#ifdef HYPRE_NO_GLOBAL_PARTITION
   for (i=0; i <= 2; i++)
#else
   for (i=0; i <= num_procs; i++)
#endif
   {
      hypre_fprintf(file, "%d \n", partitioning[i] + base_j);
   }

#ifdef HYPRE_NO_GLOBAL_PARTITION
   part0 = partitioning[0];
   for (j = part0; j < partitioning[1]; j++)
#else
   part0 = partitioning[myid];
   for (j = part0; j < partitioning[myid+1]; j++)
#endif
   {
      hypre_fprintf(file, "%d %.14e\n", j + base_j, local_data[j-part0]);
   }

   fclose(file);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParVectorReadIJ
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParVectorReadIJ( MPI_Comm             comm,
                       const char          *filename,
                       HYPRE_Int                 *base_j_ptr,
                       hypre_ParVector    **vector_ptr)
{
   HYPRE_Int               global_size;
   hypre_ParVector  *vector;
   hypre_Vector     *local_vector;
   double           *local_data;
   HYPRE_Int              *partitioning;
   HYPRE_Int               base_j;

   HYPRE_Int               myid, num_procs, i, j, J;
   char              new_filename[255];
   FILE             *file;

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &myid);
  
   hypre_sprintf(new_filename,"%s.%05d", filename, myid);

   if ((file = fopen(new_filename, "r")) == NULL)
   {
      hypre_printf("Error: can't open output file %s\n", new_filename);
      hypre_error(HYPRE_ERROR_GENERIC);
      return hypre_error_flag;
   }

   hypre_fscanf(file, "%d", &global_size);
#ifdef HYPRE_NO_GLOBAL_PARTITION
/* this may need to be changed so that the base is available in the file! */
   partitioning = hypre_CTAlloc(HYPRE_Int,2);

   hypre_fscanf(file, "%d", partitioning);
   for (i = 0; i < 2; i++)
   {
      hypre_fscanf(file, "%d", partitioning+i);
   }
#else
   partitioning = hypre_CTAlloc(HYPRE_Int,num_procs+1);

   hypre_fscanf(file, "%d", partitioning);
   for (i = 1; i <= num_procs; i++)
   {
      hypre_fscanf(file, "%d", partitioning+i);
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
      hypre_fscanf(file, "%d %le", &J, local_data + j);
   }

   fclose(file);

   *base_j_ptr = base_j;
   *vector_ptr = vector;

   /* multivector code not written yet >>> */
   hypre_assert( hypre_ParVectorNumVectors(vector) == 1 );
   if ( hypre_ParVectorNumVectors(vector) != 1 ) hypre_error(HYPRE_ERROR_GENERIC);

   return hypre_error_flag;
}


/*--------------------------------------------------------------------
 * hypre_FillResponseParToVectorAll
 * Fill response function for determining the send processors
 * data exchange
 *--------------------------------------------------------------------*/

HYPRE_Int
hypre_FillResponseParToVectorAll(void *p_recv_contact_buf, 
                                 HYPRE_Int contact_size, HYPRE_Int contact_proc, void *ro, 
                                 MPI_Comm comm, void **p_send_response_buf, 
                                 HYPRE_Int *response_message_size )
{
   HYPRE_Int    myid;
   HYPRE_Int    i, index, count, elength;

   HYPRE_Int    *recv_contact_buf = (HYPRE_Int * ) p_recv_contact_buf;

   hypre_DataExchangeResponse  *response_obj = ro;  

   hypre_ProcListElements      *send_proc_obj = response_obj->data2;   


   hypre_MPI_Comm_rank(comm, &myid );


   /*check to see if we need to allocate more space in send_proc_obj for ids*/
   if (send_proc_obj->length == send_proc_obj->storage_length)
   {
      send_proc_obj->storage_length +=10; /*add space for 10 more processors*/
      send_proc_obj->id = hypre_TReAlloc(send_proc_obj->id,HYPRE_Int, 
					 send_proc_obj->storage_length);
      send_proc_obj->vec_starts = hypre_TReAlloc(send_proc_obj->vec_starts,HYPRE_Int, 
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
					       HYPRE_Int, elength);
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
  
   
   return hypre_error_flag;

}

/* -----------------------------------------------------------------------------
 * return the sum of all local elements of the vector
 * ----------------------------------------------------------------------------- */

double hypre_ParVectorLocalSumElts( hypre_ParVector * vector )
{
   return hypre_VectorSumElts( hypre_ParVectorLocalVector(vector) );
}
