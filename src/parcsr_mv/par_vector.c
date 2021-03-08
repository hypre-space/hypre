/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Member functions for hypre_Vector class.
 *
 *****************************************************************************/

#include "_hypre_parcsr_mv.h"

HYPRE_Int hypre_FillResponseParToVectorAll(void*, HYPRE_Int, HYPRE_Int, void*, MPI_Comm, void**, HYPRE_Int*);

/*--------------------------------------------------------------------------
 * hypre_ParVectorCreate
 *--------------------------------------------------------------------------*/

/* If create is called and partitioning is NOT null, then it is assumed that it
   is array of length 2 containing the start row of the calling processor
   followed by the start row of the next processor - AHB 6/05 */

hypre_ParVector *
hypre_ParVectorCreate( MPI_Comm      comm,
                       HYPRE_BigInt  global_size,
                       HYPRE_BigInt *partitioning )
{
   hypre_ParVector *vector;
   HYPRE_Int num_procs, my_id;

   if (global_size < 0)
   {
      hypre_error_in_arg(2);
      return NULL;
   }
   vector = hypre_CTAlloc(hypre_ParVector, 1, HYPRE_MEMORY_HOST);
   hypre_MPI_Comm_rank(comm,&my_id);

   if (!partitioning)
   {
      hypre_MPI_Comm_size(comm,&num_procs);
      hypre_GenerateLocalPartitioning(global_size, num_procs, my_id, &partitioning);
   }

   hypre_ParVectorAssumedPartition(vector) = NULL;

   hypre_ParVectorComm(vector)         = comm;
   hypre_ParVectorGlobalSize(vector)   = global_size;
   hypre_ParVectorFirstIndex(vector)   = partitioning[0];
   hypre_ParVectorLastIndex(vector)    = partitioning[1]-1;
   hypre_ParVectorPartitioning(vector) = partitioning;
   hypre_ParVectorLocalVector(vector)  = hypre_SeqVectorCreate(partitioning[1] - partitioning[0]);

   /* set defaults */
   hypre_ParVectorOwnsData(vector)         = 1;
   hypre_ParVectorOwnsPartitioning(vector) = 1;
   hypre_ParVectorActualLocalSize(vector)  = 0;

   return vector;
}

/*--------------------------------------------------------------------------
 * hypre_ParMultiVectorCreate
 *--------------------------------------------------------------------------*/

hypre_ParVector *
hypre_ParMultiVectorCreate( MPI_Comm      comm,
                            HYPRE_BigInt  global_size,
                            HYPRE_BigInt *partitioning,
                            HYPRE_Int     num_vectors )
{
   /* note that global_size is the global length of a single vector */
   hypre_ParVector *vector = hypre_ParVectorCreate( comm, global_size, partitioning );
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
         hypre_TFree(hypre_ParVectorPartitioning(vector), HYPRE_MEMORY_HOST);
      }

      if (hypre_ParVectorAssumedPartition(vector))
      {
         hypre_AssumedPartitionDestroy(hypre_ParVectorAssumedPartition(vector));
      }

      hypre_TFree(vector, HYPRE_MEMORY_HOST);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParVectorInitialize
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParVectorInitialize_v2( hypre_ParVector *vector, HYPRE_MemoryLocation memory_location )
{
   if (!vector)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   hypre_SeqVectorInitialize_v2(hypre_ParVectorLocalVector(vector), memory_location);

   hypre_ParVectorActualLocalSize(vector) = hypre_VectorSize(hypre_ParVectorLocalVector(vector));

   return hypre_error_flag;
}

HYPRE_Int
hypre_ParVectorInitialize( hypre_ParVector *vector )
{
   return hypre_ParVectorInitialize_v2(vector, hypre_ParVectorMemoryLocation(vector));
}

/*--------------------------------------------------------------------------
 * hypre_ParVectorSetDataOwner
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParVectorSetDataOwner( hypre_ParVector *vector,
                             HYPRE_Int        owns_data )
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
                                     HYPRE_Int        owns_partitioning )
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
                              HYPRE_Int        num_vectors )
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
   char             new_file_name[80];
   hypre_ParVector *par_vector;
   HYPRE_Int        my_id, num_procs;
   HYPRE_BigInt    *partitioning;
   HYPRE_BigInt     global_size;
   HYPRE_Int        i;
   FILE            *fp;

   hypre_MPI_Comm_rank(comm,&my_id);
   hypre_MPI_Comm_size(comm,&num_procs);

   partitioning = hypre_CTAlloc(HYPRE_BigInt, num_procs+1, HYPRE_MEMORY_HOST);

   hypre_sprintf(new_file_name,"%s.INFO.%d",file_name,my_id);
   fp = fopen(new_file_name, "r");
   hypre_fscanf(fp, "%b\n", &global_size);
   for (i=0; i < 2; i++)
      hypre_fscanf(fp, "%b\n", &partitioning[i]);
   fclose (fp);
   par_vector = hypre_CTAlloc(hypre_ParVector, 1, HYPRE_MEMORY_HOST);

   hypre_ParVectorComm(par_vector) = comm;
   hypre_ParVectorGlobalSize(par_vector) = global_size;

   hypre_ParVectorFirstIndex(par_vector) = partitioning[0];
   hypre_ParVectorLastIndex(par_vector) = partitioning[1]-1;

   hypre_ParVectorPartitioning(par_vector) = partitioning;

   hypre_ParVectorOwnsData(par_vector) = 1;
   hypre_ParVectorOwnsPartitioning(par_vector) = 1;

   hypre_sprintf(new_file_name,"%s.%d",file_name,my_id);
   hypre_ParVectorLocalVector(par_vector) = hypre_SeqVectorRead(new_file_name);

   /* multivector code not written yet */
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
   char          new_file_name[80];
   hypre_Vector *local_vector;
   MPI_Comm      comm;
   HYPRE_Int     my_id, num_procs, i;
   HYPRE_BigInt *partitioning;
   HYPRE_BigInt  global_size;
   FILE         *fp;
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
   hypre_fprintf(fp, "%b\n", global_size);
   for (i=0; i < 2; i++)
      hypre_fprintf(fp, "%b\n", partitioning[i]);

   fclose (fp);
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParVectorSetConstantValues
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParVectorSetConstantValues( hypre_ParVector *v,
                                  HYPRE_Complex    value )
{
   hypre_Vector *v_local = hypre_ParVectorLocalVector(v);

   return hypre_SeqVectorSetConstantValues(v_local,value);
}

/*--------------------------------------------------------------------------
 * hypre_ParVectorSetRandomValues
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParVectorSetRandomValues( hypre_ParVector *v,
                                HYPRE_Int        seed )
{
   HYPRE_Int     my_id;
   hypre_Vector *v_local = hypre_ParVectorLocalVector(v);

   MPI_Comm     comm = hypre_ParVectorComm(v);
   hypre_MPI_Comm_rank(comm,&my_id);

   seed *= (my_id+1);

   return hypre_SeqVectorSetRandomValues(v_local, seed);
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
   hypre_ParVector * y =
      hypre_ParVectorCreate(hypre_ParVectorComm(x), hypre_ParVectorGlobalSize(x),
                            hypre_ParVectorPartitioning(x));

   hypre_ParVectorOwnsData(y) = 1;
   /* ...This vector owns its local vector, although the local vector doesn't
    * own _its_ data */
   hypre_ParVectorOwnsPartitioning(y) = 0;
   hypre_SeqVectorDestroy( hypre_ParVectorLocalVector(y) );
   hypre_ParVectorLocalVector(y) = hypre_SeqVectorCloneShallow(hypre_ParVectorLocalVector(x) );
   hypre_ParVectorFirstIndex(y) = hypre_ParVectorFirstIndex(x);

   return y;
}

hypre_ParVector *
hypre_ParVectorCloneDeep_v2( hypre_ParVector *x, HYPRE_MemoryLocation memory_location )
{
   hypre_ParVector *y =
      hypre_ParVectorCreate(hypre_ParVectorComm(x), hypre_ParVectorGlobalSize(x),
                            hypre_ParVectorPartitioning(x));

   hypre_ParVectorOwnsData(y) = 1;
   hypre_ParVectorOwnsPartitioning(y) = 0;
   hypre_SeqVectorDestroy( hypre_ParVectorLocalVector(y) );
   hypre_ParVectorLocalVector(y) = hypre_SeqVectorCloneDeep_v2( hypre_ParVectorLocalVector(x),
                                                                memory_location );
   hypre_ParVectorFirstIndex(y) = hypre_ParVectorFirstIndex(x); //RL: WHY HERE?

   return y;
}

HYPRE_Int
hypre_ParVectorMigrate(hypre_ParVector *x, HYPRE_MemoryLocation memory_location)
{
   if (!x)
   {
      return hypre_error_flag;
   }

   if ( hypre_GetActualMemLocation(memory_location) !=
        hypre_GetActualMemLocation(hypre_ParVectorMemoryLocation(x)) )
   {
      hypre_Vector *x_local = hypre_SeqVectorCloneDeep_v2(hypre_ParVectorLocalVector(x), memory_location);
      hypre_SeqVectorDestroy(hypre_ParVectorLocalVector(x));
      hypre_ParVectorLocalVector(x) = x_local;
   }
   else
   {
      hypre_VectorMemoryLocation(hypre_ParVectorLocalVector(x)) = memory_location;
   }

   return hypre_error_flag;
}


/*--------------------------------------------------------------------------
 * hypre_ParVectorScale
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParVectorScale( HYPRE_Complex    alpha,
                      hypre_ParVector *y )
{
   hypre_Vector *y_local = hypre_ParVectorLocalVector(y);

   return hypre_SeqVectorScale( alpha, y_local);
}

/*--------------------------------------------------------------------------
 * hypre_ParVectorAxpy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParVectorAxpy( HYPRE_Complex    alpha,
                     hypre_ParVector *x,
                     hypre_ParVector *y )
{
   hypre_Vector *x_local = hypre_ParVectorLocalVector(x);
   hypre_Vector *y_local = hypre_ParVectorLocalVector(y);

   return hypre_SeqVectorAxpy( alpha, x_local, y_local);
}

/*--------------------------------------------------------------------------
 * hypre_ParVectorInnerProd
 *--------------------------------------------------------------------------*/

HYPRE_Real
hypre_ParVectorInnerProd( hypre_ParVector *x,
                          hypre_ParVector *y )
{
   MPI_Comm      comm    = hypre_ParVectorComm(x);
   hypre_Vector *x_local = hypre_ParVectorLocalVector(x);
   hypre_Vector *y_local = hypre_ParVectorLocalVector(y);

   HYPRE_Real result = 0.0;
   HYPRE_Real local_result = hypre_SeqVectorInnerProd(x_local, y_local);

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_ALL_REDUCE] -= hypre_MPI_Wtime();
#endif
   hypre_MPI_Allreduce(&local_result, &result, 1, HYPRE_MPI_REAL,
                       hypre_MPI_SUM, comm);
#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_ALL_REDUCE] += hypre_MPI_Wtime();
#endif

   return result;
}

/*--------------------------------------------------------------------------
 * hypre_ParVectorElmdivpy
 * y = y + x ./ b [MATLAB Notation]
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParVectorElmdivpy( hypre_ParVector *x,
                         hypre_ParVector *b,
                         hypre_ParVector *y )
{
   hypre_Vector *x_local = hypre_ParVectorLocalVector(x);
   hypre_Vector *b_local = hypre_ParVectorLocalVector(b);
   hypre_Vector *y_local = hypre_ParVectorLocalVector(y);

   return hypre_SeqVectorElmdivpy(x_local, b_local, y_local);
}

/*--------------------------------------------------------------------------
 * hypre_VectorToParVector:
 * generates a ParVector from a Vector on proc 0 and distributes the pieces
 * to the other procs in comm
 *--------------------------------------------------------------------------*/

hypre_ParVector *
hypre_VectorToParVector ( MPI_Comm      comm,
                          hypre_Vector *v,
                          HYPRE_BigInt *vec_starts )
{
   HYPRE_BigInt        global_size;
   HYPRE_BigInt       *global_vec_starts = NULL;
   HYPRE_BigInt        first_index;
   HYPRE_BigInt        last_index;
   HYPRE_Int           local_size;
   HYPRE_Int           num_vectors;
   HYPRE_Int           num_procs, my_id;
   HYPRE_Int           global_vecstride, vecstride, idxstride;
   hypre_ParVector    *par_vector;
   hypre_Vector       *local_vector;
   HYPRE_Complex      *v_data;
   HYPRE_Complex      *local_data;
   hypre_MPI_Request  *requests;
   hypre_MPI_Status   *status, status0;
   HYPRE_Int           i, j, k, p;

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   if (my_id == 0)
   {
      global_size = (HYPRE_BigInt)hypre_VectorSize(v);
      v_data = hypre_VectorData(v);
      num_vectors = hypre_VectorNumVectors(v); /* for multivectors */
      global_vecstride = hypre_VectorVectorStride(v);
   }

   hypre_MPI_Bcast(&global_size,1,HYPRE_MPI_INT,0,comm);
   hypre_MPI_Bcast(&num_vectors,1,HYPRE_MPI_INT,0,comm);
   hypre_MPI_Bcast(&global_vecstride,1,HYPRE_MPI_INT,0,comm);

   if ( num_vectors == 1 )
      par_vector = hypre_ParVectorCreate(comm, global_size, vec_starts);
   else
      par_vector = hypre_ParMultiVectorCreate(comm, global_size, vec_starts, num_vectors);

   vec_starts  = hypre_ParVectorPartitioning(par_vector);
   first_index = hypre_ParVectorFirstIndex(par_vector);
   last_index  = hypre_ParVectorLastIndex(par_vector);
   local_size  = (HYPRE_Int)(last_index - first_index) + 1;

   if (my_id == 0)
   {
      global_vec_starts = hypre_CTAlloc(HYPRE_BigInt, num_procs+1, HYPRE_MEMORY_HOST);
   }
   hypre_MPI_Gather(&first_index, 1, HYPRE_MPI_BIG_INT, global_vec_starts,
                     1, HYPRE_MPI_BIG_INT, 0, comm);
   if (my_id == 0)
   {
      global_vec_starts[num_procs] = hypre_ParVectorGlobalSize(par_vector);
   }

   hypre_ParVectorInitialize(par_vector);
   local_vector = hypre_ParVectorLocalVector(par_vector);
   local_data = hypre_VectorData(local_vector);
   vecstride = hypre_VectorVectorStride(local_vector);
   idxstride = hypre_VectorIndexStride(local_vector);
   /* so far the only implemented multivector StorageMethod is 0 */
   hypre_assert( idxstride==1 );

   if (my_id == 0)
   {
      requests = hypre_CTAlloc(hypre_MPI_Request, num_vectors*(num_procs-1), HYPRE_MEMORY_HOST);
      status = hypre_CTAlloc(hypre_MPI_Status, num_vectors*(num_procs-1), HYPRE_MEMORY_HOST);
      k = 0;
      for (p = 1; p<num_procs; p++)
         for (j = 0; j<num_vectors; ++j)
         {
            hypre_MPI_Isend( &v_data[(HYPRE_Int) global_vec_starts[p]] + j*global_vecstride,
                             (HYPRE_Int)(global_vec_starts[p+1] - global_vec_starts[p]),
                             HYPRE_MPI_COMPLEX, p, 0, comm, &requests[k++] );
         }
      if (num_vectors == 1)
      {
         for (i = 0; i < local_size; i++)
            local_data[i] = v_data[i];
      }
      else
      {
         for (j = 0; j<num_vectors; ++j)
         {
            for (i = 0; i < local_size; i++)
               local_data[i+j*vecstride] = v_data[i+j*global_vecstride];
         }
      }
      hypre_MPI_Waitall(num_procs-1,requests, status);
      hypre_TFree(requests, HYPRE_MEMORY_HOST);
      hypre_TFree(status, HYPRE_MEMORY_HOST);
   }
   else
   {
      for ( j=0; j<num_vectors; ++j )
         hypre_MPI_Recv( local_data+j*vecstride, local_size, HYPRE_MPI_COMPLEX,
                         0, 0, comm,&status0 );
   }

   if (global_vec_starts)
   {
      hypre_TFree(global_vec_starts, HYPRE_MEMORY_HOST);
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
hypre_ParVectorToVectorAll( hypre_ParVector *par_v )
{
   MPI_Comm             comm = hypre_ParVectorComm(par_v);
   HYPRE_BigInt         global_size = hypre_ParVectorGlobalSize(par_v);
   hypre_Vector        *local_vector = hypre_ParVectorLocalVector(par_v);
   HYPRE_Int            num_procs, my_id;
   HYPRE_Int            num_vectors = hypre_ParVectorNumVectors(par_v);
   hypre_Vector        *vector;
   HYPRE_Complex       *vector_data;
   HYPRE_Complex       *local_data;
   HYPRE_Int            local_size;
   hypre_MPI_Request   *requests;
   hypre_MPI_Status    *status;
   HYPRE_Int            i, j;
   HYPRE_Int           *used_procs;
   HYPRE_Int            num_types, num_requests;
   HYPRE_Int            vec_len, proc_id;

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

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   local_size = (HYPRE_Int)(hypre_ParVectorLastIndex(par_v) -
      hypre_ParVectorFirstIndex(par_v) + 1);

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
   send_proc_obj.id = hypre_CTAlloc(HYPRE_Int,  send_proc_obj.storage_length, HYPRE_MEMORY_HOST);
   send_proc_obj.vec_starts =
      hypre_CTAlloc(HYPRE_Int,  send_proc_obj.storage_length + 1, HYPRE_MEMORY_HOST);
   send_proc_obj.vec_starts[0] = 0;
   send_proc_obj.element_storage_length = 10;
   send_proc_obj.elements =
      hypre_CTAlloc(HYPRE_BigInt,  send_proc_obj.element_storage_length, HYPRE_MEMORY_HOST);

   max_response_size = 0; /* each response is null */
   response_obj.fill_response = hypre_FillResponseParToVectorAll;
   response_obj.data1 = NULL;
   response_obj.data2 = &send_proc_obj; /*this is where we keep info from contacts*/


   hypre_DataExchangeList(num_contacts,
                          contact_proc_list, contact_send_buf,
                          contact_send_buf_starts, sizeof(HYPRE_Int),
                          //0, &response_obj,
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

         send_info = hypre_CTAlloc(HYPRE_Int,  count, HYPRE_MEMORY_HOST);
         hypre_MPI_Recv(send_info, count, HYPRE_MPI_INT, 0, tag1, comm, &status1);

         /* now unpack */
         num_types = send_info[0];
         used_procs =  hypre_CTAlloc(HYPRE_Int,  num_types, HYPRE_MEMORY_HOST);
         new_vec_starts = hypre_CTAlloc(HYPRE_Int,  num_types+1, HYPRE_MEMORY_HOST);

         for (i=1; i<= num_types; i++)
         {
            used_procs[i-1] = (HYPRE_Int)send_info[i];
         }
         for (i=num_types+1; i< count; i++)
         {
            new_vec_starts[i-num_types-1] = send_info[i] ;
         }
      }
      else /* clean up and exit */
      {
         hypre_TFree(send_proc_obj.vec_starts, HYPRE_MEMORY_HOST);
         hypre_TFree(send_proc_obj.id, HYPRE_MEMORY_HOST);
         hypre_TFree(send_proc_obj.elements, HYPRE_MEMORY_HOST);
         if(response_recv_buf)        hypre_TFree(response_recv_buf, HYPRE_MEMORY_HOST);
         if(response_recv_buf_starts) hypre_TFree(response_recv_buf_starts, HYPRE_MEMORY_HOST);
         return NULL;
      }
   }
   else /* my_id ==0 */
   {
      num_types = send_proc_obj.length;
      used_procs =  hypre_CTAlloc(HYPRE_Int,  num_types, HYPRE_MEMORY_HOST);
      new_vec_starts = hypre_CTAlloc(HYPRE_Int,  num_types+1, HYPRE_MEMORY_HOST);

      new_vec_starts[0] = 0;
      for (i=0; i< num_types; i++)
      {
         used_procs[i] = send_proc_obj.id[i];
         new_vec_starts[i+1] = send_proc_obj.elements[i]+1;
      }
      hypre_qsort0(used_procs, 0, num_types-1);
      hypre_qsort0(new_vec_starts, 0, num_types);
      /*now we need to put into an array to send */
      count =  2*num_types+2;
      send_info = hypre_CTAlloc(HYPRE_Int,  count, HYPRE_MEMORY_HOST);
      send_info[0] = num_types;
      for (i=1; i<= num_types; i++)
      {
         send_info[i] = (HYPRE_Int)used_procs[i-1];
      }
      for (i=num_types+1; i< count; i++)
      {
         send_info[i] = new_vec_starts[i-num_types-1];
      }
      requests = hypre_CTAlloc(hypre_MPI_Request,  num_types, HYPRE_MEMORY_HOST);
      status =  hypre_CTAlloc(hypre_MPI_Status,  num_types, HYPRE_MEMORY_HOST);

      /* don't send to myself  - these are sorted so my id would be first*/
      start = 0;
      if (used_procs[0] == 0)
      {
         start = 1;
      }


      for (i=start; i < num_types; i++)
      {
         hypre_MPI_Isend(send_info, count, HYPRE_MPI_INT, used_procs[i],
                         tag1, comm, &requests[i-start]);
      }
      hypre_MPI_Waitall(num_types-start, requests, status);

      hypre_TFree(status, HYPRE_MEMORY_HOST);
      hypre_TFree(requests, HYPRE_MEMORY_HOST);
   }

   /* clean up */
   hypre_TFree(send_proc_obj.vec_starts, HYPRE_MEMORY_HOST);
   hypre_TFree(send_proc_obj.id, HYPRE_MEMORY_HOST);
   hypre_TFree(send_proc_obj.elements, HYPRE_MEMORY_HOST);
   hypre_TFree(send_info, HYPRE_MEMORY_HOST);
   if(response_recv_buf)        hypre_TFree(response_recv_buf, HYPRE_MEMORY_HOST);
   if(response_recv_buf_starts) hypre_TFree(response_recv_buf_starts, HYPRE_MEMORY_HOST);

   /* now proc 0 can exit if it has no rows */
   if (!local_size) {
      hypre_TFree(used_procs, HYPRE_MEMORY_HOST);
      hypre_TFree(new_vec_starts, HYPRE_MEMORY_HOST);
      return NULL;
   }

   /* everyone left has rows and knows: new_vec_starts, num_types, and used_procs */

   /* this vector should be rather small */

   local_data = hypre_VectorData(local_vector);
   vector = hypre_SeqVectorCreate((HYPRE_Int)global_size);
   hypre_VectorNumVectors(vector) = num_vectors;
   hypre_SeqVectorInitialize(vector);
   vector_data = hypre_VectorData(vector);

   num_requests = 2*num_types;

   requests = hypre_CTAlloc(hypre_MPI_Request,  num_requests, HYPRE_MEMORY_HOST);
   status = hypre_CTAlloc(hypre_MPI_Status,  num_requests, HYPRE_MEMORY_HOST);

   /* initialize data exchange among used_procs and generate vector  - here we
      send to ourself also*/

   j = 0;
   for (i = 0; i < num_types; i++)
   {
      proc_id = used_procs[i];
      vec_len = (HYPRE_Int)(new_vec_starts[i+1] - new_vec_starts[i]);
      hypre_MPI_Irecv(&vector_data[(HYPRE_Int)new_vec_starts[i]], num_vectors*vec_len,
                      HYPRE_MPI_COMPLEX, proc_id, tag2, comm, &requests[j++]);
   }
   for (i = 0; i < num_types; i++)
   {
      hypre_MPI_Isend(local_data, num_vectors*local_size, HYPRE_MPI_COMPLEX,
                      used_procs[i], tag2, comm, &requests[j++]);
   }

   hypre_MPI_Waitall(num_requests, requests, status);

   if (num_requests)
   {
      hypre_TFree(requests, HYPRE_MEMORY_HOST);
      hypre_TFree(status, HYPRE_MEMORY_HOST);
      hypre_TFree(used_procs, HYPRE_MEMORY_HOST);
   }

   hypre_TFree(new_vec_starts, HYPRE_MEMORY_HOST);

   return vector;
}

/*--------------------------------------------------------------------------
 * hypre_ParVectorPrintIJ
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParVectorPrintIJ( hypre_ParVector *vector,
                        HYPRE_Int        base_j,
                        const char      *filename )
{
   MPI_Comm          comm;
   HYPRE_BigInt      global_size, j;
   HYPRE_BigInt     *partitioning;
   HYPRE_Complex    *local_data;
   HYPRE_Int         myid, num_procs, i, part0;
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

   /* multivector code not written yet */
   hypre_assert( hypre_ParVectorNumVectors(vector) == 1 );
   if ( hypre_ParVectorNumVectors(vector) != 1 ) hypre_error_in_arg(1);

   hypre_MPI_Comm_rank(comm, &myid);
   hypre_MPI_Comm_size(comm, &num_procs);

   hypre_sprintf(new_filename,"%s.%05d", filename, myid);

   if ((file = fopen(new_filename, "w")) == NULL)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC,"Error: can't open output file %s\n");
      return hypre_error_flag;
   }

   local_data = hypre_VectorData(hypre_ParVectorLocalVector(vector));

   hypre_fprintf(file, "%b \n", global_size);
   for (i=0; i < 2; i++)
   {
      hypre_fprintf(file, "%b ", partitioning[i] + base_j);
   }
   hypre_fprintf(file, "\n");

   part0 = partitioning[0];
   for (j = part0; j < partitioning[1]; j++)
   {
      hypre_fprintf(file, "%b %.14e\n", j + base_j, local_data[(HYPRE_Int)(j-part0)]);
   }

   fclose(file);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParVectorReadIJ
 * Warning: wrong base for assumed partition if base > 0
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParVectorReadIJ( MPI_Comm          comm,
                       const char       *filename,
                       HYPRE_Int        *base_j_ptr,
                       hypre_ParVector **vector_ptr )
{
   HYPRE_BigInt      global_size, J;
   hypre_ParVector  *vector;
   hypre_Vector     *local_vector;
   HYPRE_Complex    *local_data;
   HYPRE_BigInt     *partitioning;
   HYPRE_Int         base_j;

   HYPRE_Int         myid, num_procs, i, j;
   char              new_filename[255];
   FILE             *file;

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &myid);

   hypre_sprintf(new_filename,"%s.%05d", filename, myid);

   if ((file = fopen(new_filename, "r")) == NULL)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC,"Error: can't open output file %s\n");
      return hypre_error_flag;
   }

   hypre_fscanf(file, "%b", &global_size);
   /* this may need to be changed so that the base is available in the file! */
   partitioning = hypre_CTAlloc(HYPRE_BigInt, 2, HYPRE_MEMORY_HOST);

   hypre_fscanf(file, "%b", partitioning);
   for (i = 0; i < 2; i++)
   {
      hypre_fscanf(file, "%b", partitioning+i);
   }
   /* This is not yet implemented correctly! */
   base_j = 0;
   vector = hypre_ParVectorCreate(comm, global_size,
                                  partitioning);

   hypre_ParVectorInitialize(vector);

   local_vector = hypre_ParVectorLocalVector(vector);
   local_data   = hypre_VectorData(local_vector);

   for (j = 0; j < (HYPRE_Int)(partitioning[1] - partitioning[0]); j++)
   {
      hypre_fscanf(file, "%b %le", &J, local_data + j);
   }

   fclose(file);

   *base_j_ptr = base_j;
   *vector_ptr = vector;

   /* multivector code not written yet */
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
hypre_FillResponseParToVectorAll( void       *p_recv_contact_buf,
                                  HYPRE_Int   contact_size,
                                  HYPRE_Int   contact_proc,
                                  void       *ro,
                                  MPI_Comm    comm,
                                  void      **p_send_response_buf,
                                  HYPRE_Int  *response_message_size )
{
   HYPRE_Int     myid;
   HYPRE_Int     i, index, count, elength;

   HYPRE_BigInt    *recv_contact_buf = (HYPRE_BigInt * ) p_recv_contact_buf;

   hypre_DataExchangeResponse  *response_obj = (hypre_DataExchangeResponse*)ro;

   hypre_ProcListElements      *send_proc_obj = (hypre_ProcListElements*)response_obj->data2;
   hypre_MPI_Comm_rank(comm, &myid );

   /*check to see if we need to allocate more space in send_proc_obj for ids*/
   if (send_proc_obj->length == send_proc_obj->storage_length)
   {
      send_proc_obj->storage_length +=10; /*add space for 10 more processors*/
      send_proc_obj->id = hypre_TReAlloc(send_proc_obj->id, HYPRE_Int,
                                         send_proc_obj->storage_length, HYPRE_MEMORY_HOST);
      send_proc_obj->vec_starts =
         hypre_TReAlloc(send_proc_obj->vec_starts, HYPRE_Int,
                        send_proc_obj->storage_length + 1, HYPRE_MEMORY_HOST);
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
                                               HYPRE_BigInt,  elength, HYPRE_MEMORY_HOST);
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

HYPRE_Complex hypre_ParVectorLocalSumElts( hypre_ParVector * vector )
{
   return hypre_SeqVectorSumElts( hypre_ParVectorLocalVector(vector) );
}

HYPRE_Int
hypre_ParVectorGetValuesHost(hypre_ParVector *vector,
                             HYPRE_Int        num_values,
                             HYPRE_BigInt    *indices,
                             HYPRE_Complex   *values)
{
   HYPRE_Int     i, ierr = 0;
   HYPRE_BigInt  first_index = hypre_ParVectorFirstIndex(vector);
   HYPRE_BigInt  last_index = hypre_ParVectorLastIndex(vector);
   hypre_Vector *local_vector = hypre_ParVectorLocalVector(vector);
   HYPRE_Complex *data = hypre_VectorData(local_vector);

   /*
   if (hypre_VectorOwnsData(local_vector) == 0)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC,"Vector does not own data! -- hypre_ParVectorGetValues.");
      return hypre_error_flag;
   }
   */

   if (indices)
   {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) reduction(+:ierr) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < num_values; i++)
      {
         HYPRE_BigInt index = indices[i];
         if (index < first_index || index > last_index)
         {
            ierr ++;
         }
         else
         {
            HYPRE_Int local_index = (HYPRE_Int) (index - first_index);
            values[i] = data[local_index];
         }
      }

      if (ierr)
      {
         hypre_error_in_arg(3);
         hypre_error_w_msg(HYPRE_ERROR_GENERIC,"Index out of range! -- hypre_ParVectorGetValues.");
         hypre_printf("Index out of range! -- hypre_ParVectorGetValues\n");
      }
   }
   else
   {
      if (num_values > hypre_VectorSize(local_vector))
      {
         hypre_error_in_arg(2);
         return hypre_error_flag;
      }

#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < num_values; i++)
      {
         values[i] = data[i];
      }
   }

   return hypre_error_flag;
}

HYPRE_Int
hypre_ParVectorGetValues(hypre_ParVector *vector,
                         HYPRE_Int        num_values,
                         HYPRE_BigInt    *indices,
                         HYPRE_Complex   *values)
{
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
   if (HYPRE_EXEC_DEVICE == hypre_GetExecPolicy1( hypre_ParVectorMemoryLocation(vector) ))
   {
      hypre_ParVectorGetValuesDevice(vector, num_values, indices, values);
   }
   else
#endif
   {
      hypre_ParVectorGetValuesHost(vector, num_values, indices, values);
   }

   return hypre_error_flag;
}
