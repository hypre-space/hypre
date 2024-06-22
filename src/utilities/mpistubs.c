/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_utilities.h"

/******************************************************************************
 * This routine is the same in both the sequential and normal cases
 *
 * The 'comm' argument for MPI_Comm_f2c is MPI_Fint, which is always the size of
 * a Fortran integer and hence usually the size of hypre_int.
 ****************************************************************************/

MPI_Comm
hypre_MPI_Comm_f2c( hypre_int comm )
{
#ifdef HYPRE_HAVE_MPI_COMM_F2C
   return (MPI_Comm) MPI_Comm_f2c(comm);
#else
   return (MPI_Comm) (size_t)comm;
#endif
}

/******************************************************************************
 * MPI stubs to generate serial codes without mpi
 *****************************************************************************/

#ifdef HYPRE_SEQUENTIAL

HYPRE_Int
hypre_MPI_Init( hypre_int   *argc,
                char      ***argv )
{
   return (0);
}

HYPRE_Int
hypre_MPI_Finalize( void )
{
   return (0);
}

HYPRE_Int
hypre_MPI_Abort( hypre_MPI_Comm comm,
                 HYPRE_Int      errorcode )
{
   return (0);
}

HYPRE_Real
hypre_MPI_Wtime( void )
{
   return (0.0);
}

HYPRE_Real
hypre_MPI_Wtick( void )
{
   return (0.0);
}

HYPRE_Int
hypre_MPI_Barrier( hypre_MPI_Comm comm )
{
   return (0);
}

HYPRE_Int
hypre_MPI_Comm_create( hypre_MPI_Comm   comm,
                       hypre_MPI_Group  group,
                       hypre_MPI_Comm  *newcomm )
{
   *newcomm = hypre_MPI_COMM_NULL;
   return (0);
}

HYPRE_Int
hypre_MPI_Comm_dup( hypre_MPI_Comm  comm,
                    hypre_MPI_Comm *newcomm )
{
   *newcomm = comm;
   return (0);
}

HYPRE_Int
hypre_MPI_Comm_size( hypre_MPI_Comm  comm,
                     HYPRE_Int      *size )
{
   *size = 1;
   return (0);
}

HYPRE_Int
hypre_MPI_Comm_rank( hypre_MPI_Comm  comm,
                     HYPRE_Int      *rank )
{
   *rank = 0;
   return (0);
}

HYPRE_Int
hypre_MPI_Comm_free( hypre_MPI_Comm *comm )
{
   return 0;
}

HYPRE_Int
hypre_MPI_Comm_group( hypre_MPI_Comm   comm,
                      hypre_MPI_Group *group )
{
   return (0);
}

HYPRE_Int
hypre_MPI_Comm_split( hypre_MPI_Comm  comm,
                      HYPRE_Int       n,
                      HYPRE_Int       m,
                      hypre_MPI_Comm *comms )
{
   return (0);
}

HYPRE_Int
hypre_MPI_Group_incl( hypre_MPI_Group  group,
                      HYPRE_Int        n,
                      HYPRE_Int       *ranks,
                      hypre_MPI_Group *newgroup )
{
   return (0);
}

HYPRE_Int
hypre_MPI_Group_free( hypre_MPI_Group *group )
{
   return 0;
}

HYPRE_Int
hypre_MPI_Address( void           *location,
                   hypre_MPI_Aint *address )
{
   return (0);
}

HYPRE_Int
hypre_MPI_Get_count( hypre_MPI_Status   *status,
                     hypre_MPI_Datatype  datatype,
                     HYPRE_Int          *count )
{
   return (0);
}

HYPRE_Int
hypre_MPI_Alltoall( void               *sendbuf,
                    HYPRE_Int           sendcount,
                    hypre_MPI_Datatype  sendtype,
                    void               *recvbuf,
                    HYPRE_Int           recvcount,
                    hypre_MPI_Datatype  recvtype,
                    hypre_MPI_Comm      comm )
{
   return (0);
}

HYPRE_Int
hypre_MPI_Allgather( void               *sendbuf,
                     HYPRE_Int           sendcount,
                     hypre_MPI_Datatype  sendtype,
                     void               *recvbuf,
                     HYPRE_Int           recvcount,
                     hypre_MPI_Datatype  recvtype,
                     hypre_MPI_Comm      comm )
{
   HYPRE_Int i;

   switch (sendtype)
   {
      case hypre_MPI_INT:
      {
         HYPRE_Int *crecvbuf = (HYPRE_Int *)recvbuf;
         HYPRE_Int *csendbuf = (HYPRE_Int *)sendbuf;
         for (i = 0; i < sendcount; i++)
         {
            crecvbuf[i] = csendbuf[i];
         }
      }
      break;

      case hypre_MPI_LONG_LONG_INT:
      {
         HYPRE_BigInt *crecvbuf = (HYPRE_BigInt *)recvbuf;
         HYPRE_BigInt *csendbuf = (HYPRE_BigInt *)sendbuf;
         for (i = 0; i < sendcount; i++)
         {
            crecvbuf[i] = csendbuf[i];
         }
      }
      break;

      case hypre_MPI_FLOAT:
      {
         float *crecvbuf = (float *)recvbuf;
         float *csendbuf = (float *)sendbuf;
         for (i = 0; i < sendcount; i++)
         {
            crecvbuf[i] = csendbuf[i];
         }
      }
      break;

      case hypre_MPI_DOUBLE:
      {
         double *crecvbuf = (double *)recvbuf;
         double *csendbuf = (double *)sendbuf;
         for (i = 0; i < sendcount; i++)
         {
            crecvbuf[i] = csendbuf[i];
         }
      }
      break;

      case hypre_MPI_LONG_DOUBLE:
      {
         long double *crecvbuf = (long double *)recvbuf;
         long double *csendbuf = (long double *)sendbuf;
         for (i = 0; i < sendcount; i++)
         {
            crecvbuf[i] = csendbuf[i];
         }
      }
      break;

      case hypre_MPI_CHAR:
      {
         char *crecvbuf = (char *)recvbuf;
         char *csendbuf = (char *)sendbuf;
         for (i = 0; i < sendcount; i++)
         {
            crecvbuf[i] = csendbuf[i];
         }
      }
      break;

      case hypre_MPI_LONG:
      {
         hypre_longint *crecvbuf = (hypre_longint *)recvbuf;
         hypre_longint *csendbuf = (hypre_longint *)sendbuf;
         for (i = 0; i < sendcount; i++)
         {
            crecvbuf[i] = csendbuf[i];
         }
      }
      break;

      case hypre_MPI_BYTE:
      {
         hypre_TMemcpy(recvbuf, sendbuf, char, sendcount, HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
      }
      break;

      case hypre_MPI_REAL:
      {
         HYPRE_Real *crecvbuf = (HYPRE_Real *)recvbuf;
         HYPRE_Real *csendbuf = (HYPRE_Real *)sendbuf;
         for (i = 0; i < sendcount; i++)
         {
            crecvbuf[i] = csendbuf[i];
         }
      }
      break;

      case hypre_MPI_COMPLEX:
      {
         HYPRE_Complex *crecvbuf = (HYPRE_Complex *)recvbuf;
         HYPRE_Complex *csendbuf = (HYPRE_Complex *)sendbuf;
         for (i = 0; i < sendcount; i++)
         {
            crecvbuf[i] = csendbuf[i];
         }
      }
      break;
   }

   return (0);
}

HYPRE_Int
hypre_MPI_Allgatherv( void               *sendbuf,
                      HYPRE_Int           sendcount,
                      hypre_MPI_Datatype  sendtype,
                      void               *recvbuf,
                      HYPRE_Int          *recvcounts,
                      HYPRE_Int          *displs,
                      hypre_MPI_Datatype  recvtype,
                      hypre_MPI_Comm      comm )
{
   return ( hypre_MPI_Allgather(sendbuf, sendcount, sendtype,
                                recvbuf, *recvcounts, recvtype, comm) );
}

HYPRE_Int
hypre_MPI_Gather( void               *sendbuf,
                  HYPRE_Int           sendcount,
                  hypre_MPI_Datatype  sendtype,
                  void               *recvbuf,
                  HYPRE_Int           recvcount,
                  hypre_MPI_Datatype  recvtype,
                  HYPRE_Int           root,
                  hypre_MPI_Comm      comm )
{
   return ( hypre_MPI_Allgather(sendbuf, sendcount, sendtype,
                                recvbuf, recvcount, recvtype, comm) );
}

HYPRE_Int
hypre_MPI_Gatherv( void              *sendbuf,
                   HYPRE_Int           sendcount,
                   hypre_MPI_Datatype  sendtype,
                   void               *recvbuf,
                   HYPRE_Int          *recvcounts,
                   HYPRE_Int          *displs,
                   hypre_MPI_Datatype  recvtype,
                   HYPRE_Int           root,
                   hypre_MPI_Comm      comm )
{
   return ( hypre_MPI_Allgather(sendbuf, sendcount, sendtype,
                                recvbuf, *recvcounts, recvtype, comm) );
}

HYPRE_Int
hypre_MPI_Scatter( void               *sendbuf,
                   HYPRE_Int           sendcount,
                   hypre_MPI_Datatype  sendtype,
                   void               *recvbuf,
                   HYPRE_Int           recvcount,
                   hypre_MPI_Datatype  recvtype,
                   HYPRE_Int           root,
                   hypre_MPI_Comm      comm )
{
   return ( hypre_MPI_Allgather(sendbuf, sendcount, sendtype,
                                recvbuf, recvcount, recvtype, comm) );
}

HYPRE_Int
hypre_MPI_Scatterv( void               *sendbuf,
                    HYPRE_Int           *sendcounts,
                    HYPRE_Int           *displs,
                    hypre_MPI_Datatype   sendtype,
                    void                *recvbuf,
                    HYPRE_Int            recvcount,
                    hypre_MPI_Datatype   recvtype,
                    HYPRE_Int            root,
                    hypre_MPI_Comm       comm )
{
   return ( hypre_MPI_Allgather(sendbuf, *sendcounts, sendtype,
                                recvbuf, recvcount, recvtype, comm) );
}

HYPRE_Int
hypre_MPI_Bcast( void               *buffer,
                 HYPRE_Int           count,
                 hypre_MPI_Datatype  datatype,
                 HYPRE_Int           root,
                 hypre_MPI_Comm      comm )
{
   return (0);
}

HYPRE_Int
hypre_MPI_Send( void               *buf,
                HYPRE_Int           count,
                hypre_MPI_Datatype  datatype,
                HYPRE_Int           dest,
                HYPRE_Int           tag,
                hypre_MPI_Comm      comm )
{
   return (0);
}

HYPRE_Int
hypre_MPI_Recv( void               *buf,
                HYPRE_Int           count,
                hypre_MPI_Datatype  datatype,
                HYPRE_Int           source,
                HYPRE_Int           tag,
                hypre_MPI_Comm      comm,
                hypre_MPI_Status   *status )
{
   return (0);
}

HYPRE_Int
hypre_MPI_Isend( void               *buf,
                 HYPRE_Int           count,
                 hypre_MPI_Datatype  datatype,
                 HYPRE_Int           dest,
                 HYPRE_Int           tag,
                 hypre_MPI_Comm      comm,
                 hypre_MPI_Request  *request )
{
   return (0);
}

HYPRE_Int
hypre_MPI_Irecv( void               *buf,
                 HYPRE_Int           count,
                 hypre_MPI_Datatype  datatype,
                 HYPRE_Int           source,
                 HYPRE_Int           tag,
                 hypre_MPI_Comm      comm,
                 hypre_MPI_Request  *request )
{
   return (0);
}

HYPRE_Int
hypre_MPI_Send_init( void               *buf,
                     HYPRE_Int           count,
                     hypre_MPI_Datatype  datatype,
                     HYPRE_Int           dest,
                     HYPRE_Int           tag,
                     hypre_MPI_Comm      comm,
                     hypre_MPI_Request  *request )
{
   return 0;
}

HYPRE_Int
hypre_MPI_Recv_init( void               *buf,
                     HYPRE_Int           count,
                     hypre_MPI_Datatype  datatype,
                     HYPRE_Int           dest,
                     HYPRE_Int           tag,
                     hypre_MPI_Comm      comm,
                     hypre_MPI_Request  *request )
{
   return 0;
}

HYPRE_Int
hypre_MPI_Irsend( void               *buf,
                  HYPRE_Int           count,
                  hypre_MPI_Datatype  datatype,
                  HYPRE_Int           dest,
                  HYPRE_Int           tag,
                  hypre_MPI_Comm      comm,
                  hypre_MPI_Request  *request )
{
   return 0;
}

HYPRE_Int
hypre_MPI_Startall( HYPRE_Int          count,
                    hypre_MPI_Request *array_of_requests )
{
   return 0;
}

HYPRE_Int
hypre_MPI_Probe( HYPRE_Int         source,
                 HYPRE_Int         tag,
                 hypre_MPI_Comm    comm,
                 hypre_MPI_Status *status )
{
   return 0;
}

HYPRE_Int
hypre_MPI_Iprobe( HYPRE_Int         source,
                  HYPRE_Int         tag,
                  hypre_MPI_Comm    comm,
                  HYPRE_Int        *flag,
                  hypre_MPI_Status *status )
{
   return 0;
}

HYPRE_Int
hypre_MPI_Test( hypre_MPI_Request *request,
                HYPRE_Int         *flag,
                hypre_MPI_Status  *status )
{
   *flag = 1;
   return (0);
}

HYPRE_Int
hypre_MPI_Testall( HYPRE_Int          count,
                   hypre_MPI_Request *array_of_requests,
                   HYPRE_Int         *flag,
                   hypre_MPI_Status  *array_of_statuses )
{
   *flag = 1;
   return (0);
}

HYPRE_Int
hypre_MPI_Wait( hypre_MPI_Request *request,
                hypre_MPI_Status  *status )
{
   return (0);
}

HYPRE_Int
hypre_MPI_Waitall( HYPRE_Int          count,
                   hypre_MPI_Request *array_of_requests,
                   hypre_MPI_Status  *array_of_statuses )
{
   return (0);
}

HYPRE_Int
hypre_MPI_Waitany( HYPRE_Int          count,
                   hypre_MPI_Request *array_of_requests,
                   HYPRE_Int         *index,
                   hypre_MPI_Status  *status )
{
   return (0);
}

HYPRE_Int
hypre_MPI_Allreduce( void              *sendbuf,
                     void              *recvbuf,
                     HYPRE_Int          count,
                     hypre_MPI_Datatype datatype,
                     hypre_MPI_Op       op,
                     hypre_MPI_Comm     comm )
{
   HYPRE_Int i;

   switch (datatype)
   {
      case hypre_MPI_INT:
      {
         HYPRE_Int *crecvbuf = (HYPRE_Int *)recvbuf;
         HYPRE_Int *csendbuf = (HYPRE_Int *)sendbuf;
         for (i = 0; i < count; i++)
         {
            crecvbuf[i] = csendbuf[i];
         }
      }
      break;

      case hypre_MPI_LONG_LONG_INT:
      {
         HYPRE_BigInt *crecvbuf = (HYPRE_BigInt *)recvbuf;
         HYPRE_BigInt *csendbuf = (HYPRE_BigInt *)sendbuf;
         for (i = 0; i < count; i++)
         {
            crecvbuf[i] = csendbuf[i];
         }
      }
      break;

      case hypre_MPI_FLOAT:
      {
         float *crecvbuf = (float *)recvbuf;
         float *csendbuf = (float *)sendbuf;
         for (i = 0; i < count; i++)
         {
            crecvbuf[i] = csendbuf[i];
         }
      }
      break;

      case hypre_MPI_DOUBLE:
      {
         double *crecvbuf = (double *)recvbuf;
         double *csendbuf = (double *)sendbuf;
         for (i = 0; i < count; i++)
         {
            crecvbuf[i] = csendbuf[i];
         }
      }
      break;

      case hypre_MPI_LONG_DOUBLE:
      {
         long double *crecvbuf = (long double *)recvbuf;
         long double *csendbuf = (long double *)sendbuf;
         for (i = 0; i < count; i++)
         {
            crecvbuf[i] = csendbuf[i];
         }
      }
      break;

      case hypre_MPI_CHAR:
      {
         char *crecvbuf = (char *)recvbuf;
         char *csendbuf = (char *)sendbuf;
         for (i = 0; i < count; i++)
         {
            crecvbuf[i] = csendbuf[i];
         }
      }
      break;

      case hypre_MPI_LONG:
      {
         hypre_longint *crecvbuf = (hypre_longint *)recvbuf;
         hypre_longint *csendbuf = (hypre_longint *)sendbuf;
         for (i = 0; i < count; i++)
         {
            crecvbuf[i] = csendbuf[i];
         }
      }
      break;

      case hypre_MPI_BYTE:
      {
         hypre_TMemcpy(recvbuf, sendbuf, char, count, HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
      }
      break;

      case hypre_MPI_REAL:
      {
         HYPRE_Real *crecvbuf = (HYPRE_Real *)recvbuf;
         HYPRE_Real *csendbuf = (HYPRE_Real *)sendbuf;
         for (i = 0; i < count; i++)
         {
            crecvbuf[i] = csendbuf[i];
         }
      }
      break;

      case hypre_MPI_COMPLEX:
      {
         HYPRE_Complex *crecvbuf = (HYPRE_Complex *)recvbuf;
         HYPRE_Complex *csendbuf = (HYPRE_Complex *)sendbuf;
         for (i = 0; i < count; i++)
         {
            crecvbuf[i] = csendbuf[i];
         }
      }
      break;
   }

   return 0;
}

HYPRE_Int
hypre_MPI_Reduce( void               *sendbuf,
                  void               *recvbuf,
                  HYPRE_Int           count,
                  hypre_MPI_Datatype  datatype,
                  hypre_MPI_Op        op,
                  HYPRE_Int           root,
                  hypre_MPI_Comm      comm )
{
   hypre_MPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
   return 0;
}

HYPRE_Int
hypre_MPI_Scan( void               *sendbuf,
                void               *recvbuf,
                HYPRE_Int           count,
                hypre_MPI_Datatype  datatype,
                hypre_MPI_Op        op,
                hypre_MPI_Comm      comm )
{
   hypre_MPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
   return 0;
}

HYPRE_Int
hypre_MPI_Request_free( hypre_MPI_Request *request )
{
   return 0;
}

HYPRE_Int
hypre_MPI_Type_contiguous( HYPRE_Int           count,
                           hypre_MPI_Datatype  oldtype,
                           hypre_MPI_Datatype *newtype )
{
   return (0);
}

HYPRE_Int
hypre_MPI_Type_vector( HYPRE_Int           count,
                       HYPRE_Int           blocklength,
                       HYPRE_Int           stride,
                       hypre_MPI_Datatype  oldtype,
                       hypre_MPI_Datatype *newtype )
{
   return (0);
}

HYPRE_Int
hypre_MPI_Type_hvector( HYPRE_Int           count,
                        HYPRE_Int           blocklength,
                        hypre_MPI_Aint      stride,
                        hypre_MPI_Datatype  oldtype,
                        hypre_MPI_Datatype *newtype )
{
   return (0);
}

HYPRE_Int
hypre_MPI_Type_struct( HYPRE_Int           count,
                       HYPRE_Int          *array_of_blocklengths,
                       hypre_MPI_Aint     *array_of_displacements,
                       hypre_MPI_Datatype *array_of_types,
                       hypre_MPI_Datatype *newtype )
{
   return (0);
}

HYPRE_Int
hypre_MPI_Type_commit( hypre_MPI_Datatype *datatype )
{
   return (0);
}

HYPRE_Int
hypre_MPI_Type_free( hypre_MPI_Datatype *datatype )
{
   return (0);
}

HYPRE_Int
hypre_MPI_Op_create( hypre_MPI_User_function *function, hypre_int commute, hypre_MPI_Op *op )
{
   return (0);
}

HYPRE_Int
hypre_MPI_Op_free( hypre_MPI_Op *op )
{
   return (0);
}

#if defined(HYPRE_USING_GPU)
HYPRE_Int hypre_MPI_Comm_split_type( hypre_MPI_Comm comm, HYPRE_Int split_type, HYPRE_Int key,
                                     hypre_MPI_Info info, hypre_MPI_Comm *newcomm )
{
   return (0);
}

HYPRE_Int hypre_MPI_Info_create( hypre_MPI_Info *info )
{
   return (0);
}

HYPRE_Int hypre_MPI_Info_free( hypre_MPI_Info *info )
{
   return (0);
}
#endif

/******************************************************************************
 * MPI stubs to do casting of HYPRE_Int and hypre_int correctly
 *****************************************************************************/

#else

hypre_MPI_Request
hypre_MPI_RequestFromMPI_Request(MPI_Request request)
{
   hypre_MPI_Request hrequest;
   hypre_Memset(&hrequest, 0, sizeof(hypre_MPI_Request), HYPRE_MEMORY_HOST);
   hypre_MPI_RequestMPI_Request(hrequest) = request;

   return hrequest;
}

HYPRE_Int
hypre_MPI_RequestClear(hypre_MPI_Request *request)
{
   HYPRE_Int i;
   for (i = 0; i < 2; i++)
   {
      hypre_MPI_Request_Action *action = &hypre_MPI_RequestAction(*request, i);
      hypre_MPI_Request_ActionCount(action) = 0;
      hypre_MPI_Request_ActionDataSize(action) = 0;
      hypre_TFree(hypre_MPI_Request_ActionData(action), HYPRE_MEMORY_HOST);
   }

   return hypre_error_flag;
}

HYPRE_Int
hypre_MPI_RequestSetActionFree(HYPRE_Int             i,
                               void                 *ptr,
                               hypre_MemoryLocation  ptr_location,
                               hypre_MPI_Request    *request)
{
   HYPRE_Int action_id = HYPRE_MPI_REQUEST_FREE;
   hypre_MPI_Request_Action *action = &hypre_MPI_RequestAction(*request, i);

   HYPRE_Int nb = sizeof(HYPRE_Int) + sizeof(void *) + sizeof(hypre_MemoryLocation);
   HYPRE_Int data_size = hypre_MPI_Request_ActionDataSize(action);

   hypre_MPI_Request_ActionCount(action) ++;
   hypre_MPI_Request_ActionDataSize(action) = data_size + nb;
   hypre_MPI_Request_ActionData(action) = hypre_TReAlloc(hypre_MPI_Request_ActionData(action),
                                                         char,
                                                         hypre_MPI_Request_ActionDataSize(action),
                                                         HYPRE_MEMORY_HOST);

   char *data = hypre_MPI_Request_ActionData(action) + data_size;
   hypre_TMemcpy(data, &action_id, HYPRE_Int, 1, HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
   data += sizeof(HYPRE_Int);
   hypre_TMemcpy(data, &ptr, void *, 1, HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
   data += sizeof(void *);
   hypre_TMemcpy(data, &ptr_location, hypre_MemoryLocation, 1, HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
   data += sizeof(hypre_MemoryLocation);

   hypre_assert(data == hypre_MPI_Request_ActionData(action) + hypre_MPI_Request_ActionDataSize(action));

   return hypre_error_flag;
}

HYPRE_Int
hypre_MPI_RequestSetActionCopy(HYPRE_Int             i,
                               void                 *dest,
                               hypre_MemoryLocation  dest_location,
                               void                 *src,
                               hypre_MemoryLocation  src_location,
                               HYPRE_Int             num_bytes,
                               hypre_MPI_Request    *request)
{
   HYPRE_Int action_id = HYPRE_MPI_REQUEST_COPY;
   hypre_MPI_Request_Action *action = &hypre_MPI_RequestAction(*request, i);

   HYPRE_Int nb = 2 * (sizeof(HYPRE_Int) + sizeof(void *) + sizeof(hypre_MemoryLocation));
   HYPRE_Int data_size = hypre_MPI_Request_ActionDataSize(action);

   hypre_MPI_Request_ActionCount(action) ++;
   hypre_MPI_Request_ActionDataSize(action) = data_size + nb;
   hypre_MPI_Request_ActionData(action) = hypre_TReAlloc(hypre_MPI_Request_ActionData(action),
                                                         char,
                                                         hypre_MPI_Request_ActionDataSize(action),
                                                         HYPRE_MEMORY_HOST);

   char *data = hypre_MPI_Request_ActionData(action) + data_size;
   hypre_TMemcpy(data, &action_id, HYPRE_Int, 1, HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
   data += sizeof(HYPRE_Int);
   hypre_TMemcpy(data, &num_bytes, HYPRE_Int, 1, HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
   data += sizeof(HYPRE_Int);
   hypre_TMemcpy(data, &dest, void *, 1, HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
   data += sizeof(void *);
   hypre_TMemcpy(data, &src, void *, 1, HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
   data += sizeof(void *);
   hypre_TMemcpy(data, &dest_location, hypre_MemoryLocation, 1, HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
   data += sizeof(hypre_MemoryLocation);
   hypre_TMemcpy(data, &src_location, hypre_MemoryLocation, 1, HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
   data += sizeof(hypre_MemoryLocation);

   hypre_assert(data == hypre_MPI_Request_ActionData(action) + hypre_MPI_Request_ActionDataSize(action));

   return hypre_error_flag;
}

HYPRE_Int
hypre_MPI_RequestProcessAction(HYPRE_Int          i,
                               hypre_MPI_Request *request)
{
   hypre_MPI_Request_Action *action = &hypre_MPI_RequestAction(*request, i);
   HYPRE_Int count = hypre_MPI_Request_ActionCount(action);
   char *data = hypre_MPI_Request_ActionData(action);
   HYPRE_Int k;

   for (k = 0; k < count; k ++)
   {
      HYPRE_Int action_id;

      hypre_TMemcpy(&action_id, data, HYPRE_Int, 1, HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
      data += sizeof(HYPRE_Int);

      if (action_id == HYPRE_MPI_REQUEST_FREE)
      {
         void *ptr;
         hypre_MemoryLocation ptr_location;
         hypre_TMemcpy(&ptr, data, void *, 1, HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
         data += sizeof(void *);
         hypre_TMemcpy(&ptr_location, data, hypre_MemoryLocation, 1, HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
         data += sizeof(hypre_MemoryLocation);
         // action!
         _hypre_TFree(ptr, ptr_location);
      }
      else if (action_id == HYPRE_MPI_REQUEST_COPY)
      {
         void *dest, *src;
         HYPRE_Int num_bytes;
         hypre_MemoryLocation dest_location, src_location;
         hypre_TMemcpy(&num_bytes, data, HYPRE_Int, 1, HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
         data += sizeof(HYPRE_Int);
         hypre_TMemcpy(&dest, data, void *, 1, HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
         data += sizeof(void *);
         hypre_TMemcpy(&src, data, void *, 1, HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
         data += sizeof(void *);
         hypre_TMemcpy(&dest_location, data, hypre_MemoryLocation, 1, HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
         data += sizeof(hypre_MemoryLocation);
         hypre_TMemcpy(&src_location, data, hypre_MemoryLocation, 1, HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
         data += sizeof(hypre_MemoryLocation);
         // action!
         hypre_GpuProfilingPushRange("MPI-H2D");
         _hypre_TMemcpy(dest, src, char, num_bytes, dest_location, src_location);
         hypre_GpuProfilingPopRange();
      }
   }

   hypre_assert(data == hypre_MPI_Request_ActionData(action) + hypre_MPI_Request_ActionDataSize(action));

   return hypre_error_flag;
}

HYPRE_Int
hypre_MPI_Init( hypre_int   *argc,
                char      ***argv )
{
   return (HYPRE_Int) MPI_Init(argc, argv);
}

HYPRE_Int
hypre_MPI_Finalize( void )
{
   return (HYPRE_Int) MPI_Finalize();
}

HYPRE_Int
hypre_MPI_Abort( MPI_Comm  comm,
                 HYPRE_Int errorcode )
{
   return (HYPRE_Int) MPI_Abort(comm, (hypre_int)errorcode);
}

HYPRE_Real
hypre_MPI_Wtime( void )
{
   return (HYPRE_Real)MPI_Wtime();
}

HYPRE_Real
hypre_MPI_Wtick( void )
{
   return (HYPRE_Real)MPI_Wtick();
}

HYPRE_Int
hypre_MPI_Barrier( MPI_Comm comm )
{
   return (HYPRE_Int) MPI_Barrier(comm);
}

HYPRE_Int
hypre_MPI_Comm_create( MPI_Comm        comm,
                       hypre_MPI_Group group,
                       MPI_Comm       *newcomm )
{
   return (HYPRE_Int) MPI_Comm_create(comm, group, newcomm);
}

HYPRE_Int
hypre_MPI_Comm_dup( MPI_Comm  comm,
                    MPI_Comm *newcomm )
{
   return (HYPRE_Int) MPI_Comm_dup(comm, newcomm);
}

HYPRE_Int
hypre_MPI_Comm_size( MPI_Comm   comm,
                     HYPRE_Int *size )
{
   hypre_int mpi_size;
   HYPRE_Int ierr;
   ierr = (HYPRE_Int) MPI_Comm_size(comm, &mpi_size);
   *size = (HYPRE_Int) mpi_size;
   return ierr;
}

HYPRE_Int
hypre_MPI_Comm_rank( MPI_Comm   comm,
                     HYPRE_Int *rank )
{
   hypre_int mpi_rank;
   HYPRE_Int ierr;
   ierr = (HYPRE_Int) MPI_Comm_rank(comm, &mpi_rank);
   *rank = (HYPRE_Int) mpi_rank;
   return ierr;
}

HYPRE_Int
hypre_MPI_Comm_free( MPI_Comm *comm )
{
   return (HYPRE_Int) MPI_Comm_free(comm);
}

HYPRE_Int
hypre_MPI_Comm_group( MPI_Comm   comm,
                      hypre_MPI_Group *group )
{
   return (HYPRE_Int) MPI_Comm_group(comm, group);
}

HYPRE_Int
hypre_MPI_Comm_split( MPI_Comm  comm,
                      HYPRE_Int color,
                      HYPRE_Int key,
                      MPI_Comm *newcomm )
{
   return (HYPRE_Int) MPI_Comm_split(comm, (hypre_int) color, (hypre_int) key, newcomm);
}

HYPRE_Int
hypre_MPI_Group_incl( hypre_MPI_Group  group,
                      HYPRE_Int        n,
                      HYPRE_Int       *ranks,
                      hypre_MPI_Group *newgroup )
{
   hypre_int *mpi_ranks;
   HYPRE_Int  i;
   HYPRE_Int  ierr;

   mpi_ranks = hypre_TAlloc(hypre_int,  n, HYPRE_MEMORY_HOST);
   for (i = 0; i < n; i++)
   {
      mpi_ranks[i] = (hypre_int) ranks[i];
   }
   ierr = (HYPRE_Int) MPI_Group_incl(group, (hypre_int)n, mpi_ranks, newgroup);
   hypre_TFree(mpi_ranks, HYPRE_MEMORY_HOST);

   return ierr;
}

HYPRE_Int
hypre_MPI_Group_free( hypre_MPI_Group *group )
{
   return (HYPRE_Int) MPI_Group_free(group);
}

HYPRE_Int
hypre_MPI_Address( void           *location,
                   hypre_MPI_Aint *address )
{
#if MPI_VERSION > 1
   return (HYPRE_Int) MPI_Get_address(location, address);
#else
   return (HYPRE_Int) MPI_Address(location, address);
#endif
}

HYPRE_Int
hypre_MPI_Get_count( hypre_MPI_Status   *status,
                     hypre_MPI_Datatype  datatype,
                     HYPRE_Int          *count )
{
   hypre_int mpi_count;
   HYPRE_Int ierr;
   ierr = (HYPRE_Int) MPI_Get_count(status, datatype, &mpi_count);
   *count = (HYPRE_Int) mpi_count;
   return ierr;
}

HYPRE_Int
hypre_MPI_Alltoall( void               *sendbuf,
                    HYPRE_Int           sendcount,
                    hypre_MPI_Datatype  sendtype,
                    void               *recvbuf,
                    HYPRE_Int           recvcount,
                    hypre_MPI_Datatype  recvtype,
                    hypre_MPI_Comm      comm )
{
   return (HYPRE_Int) MPI_Alltoall(sendbuf, (hypre_int)sendcount, sendtype,
                                   recvbuf, (hypre_int)recvcount, recvtype, comm);
}

HYPRE_Int
hypre_MPI_Allgather( void               *sendbuf,
                     HYPRE_Int           sendcount,
                     hypre_MPI_Datatype  sendtype,
                     void               *recvbuf,
                     HYPRE_Int           recvcount,
                     hypre_MPI_Datatype  recvtype,
                     hypre_MPI_Comm      comm )
{
   return (HYPRE_Int) MPI_Allgather(sendbuf, (hypre_int)sendcount, sendtype,
                                    recvbuf, (hypre_int)recvcount, recvtype, comm);
}

HYPRE_Int
hypre_MPI_Allgatherv( void               *sendbuf,
                      HYPRE_Int           sendcount,
                      hypre_MPI_Datatype  sendtype,
                      void               *recvbuf,
                      HYPRE_Int          *recvcounts,
                      HYPRE_Int          *displs,
                      hypre_MPI_Datatype  recvtype,
                      hypre_MPI_Comm      comm )
{
   hypre_int *mpi_recvcounts, *mpi_displs, csize;
   HYPRE_Int  i;
   HYPRE_Int  ierr;

   MPI_Comm_size(comm, &csize);
   mpi_recvcounts = hypre_TAlloc(hypre_int, csize, HYPRE_MEMORY_HOST);
   mpi_displs = hypre_TAlloc(hypre_int, csize, HYPRE_MEMORY_HOST);
   for (i = 0; i < csize; i++)
   {
      mpi_recvcounts[i] = (hypre_int) recvcounts[i];
      mpi_displs[i] = (hypre_int) displs[i];
   }
   ierr = (HYPRE_Int) MPI_Allgatherv(sendbuf, (hypre_int)sendcount, sendtype,
                                     recvbuf, mpi_recvcounts, mpi_displs,
                                     recvtype, comm);
   hypre_TFree(mpi_recvcounts, HYPRE_MEMORY_HOST);
   hypre_TFree(mpi_displs, HYPRE_MEMORY_HOST);

   return ierr;
}

HYPRE_Int
hypre_MPI_Gather( void               *sendbuf,
                  HYPRE_Int           sendcount,
                  hypre_MPI_Datatype  sendtype,
                  void               *recvbuf,
                  HYPRE_Int           recvcount,
                  hypre_MPI_Datatype  recvtype,
                  HYPRE_Int           root,
                  hypre_MPI_Comm      comm )
{
   return (HYPRE_Int) MPI_Gather(sendbuf, (hypre_int) sendcount, sendtype,
                                 recvbuf, (hypre_int) recvcount, recvtype,
                                 (hypre_int)root, comm);
}

HYPRE_Int
hypre_MPI_Gatherv(void               *sendbuf,
                  HYPRE_Int           sendcount,
                  hypre_MPI_Datatype  sendtype,
                  void               *recvbuf,
                  HYPRE_Int          *recvcounts,
                  HYPRE_Int          *displs,
                  hypre_MPI_Datatype  recvtype,
                  HYPRE_Int           root,
                  hypre_MPI_Comm      comm )
{
   hypre_int *mpi_recvcounts = NULL;
   hypre_int *mpi_displs = NULL;
   hypre_int csize, croot;
   HYPRE_Int  i;
   HYPRE_Int  ierr;

   MPI_Comm_size(comm, &csize);
   MPI_Comm_rank(comm, &croot);
   if (croot == (hypre_int) root)
   {
      mpi_recvcounts = hypre_TAlloc(hypre_int,  csize, HYPRE_MEMORY_HOST);
      mpi_displs = hypre_TAlloc(hypre_int,  csize, HYPRE_MEMORY_HOST);
      for (i = 0; i < csize; i++)
      {
         mpi_recvcounts[i] = (hypre_int) recvcounts[i];
         mpi_displs[i] = (hypre_int) displs[i];
      }
   }
   ierr = (HYPRE_Int) MPI_Gatherv(sendbuf, (hypre_int)sendcount, sendtype,
                                  recvbuf, mpi_recvcounts, mpi_displs,
                                  recvtype, (hypre_int) root, comm);
   hypre_TFree(mpi_recvcounts, HYPRE_MEMORY_HOST);
   hypre_TFree(mpi_displs, HYPRE_MEMORY_HOST);

   return ierr;
}

HYPRE_Int
hypre_MPI_Scatter( void               *sendbuf,
                   HYPRE_Int           sendcount,
                   hypre_MPI_Datatype  sendtype,
                   void               *recvbuf,
                   HYPRE_Int           recvcount,
                   hypre_MPI_Datatype  recvtype,
                   HYPRE_Int           root,
                   hypre_MPI_Comm      comm )
{
   return (HYPRE_Int) MPI_Scatter(sendbuf, (hypre_int)sendcount, sendtype,
                                  recvbuf, (hypre_int)recvcount, recvtype,
                                  (hypre_int)root, comm);
}

HYPRE_Int
hypre_MPI_Scatterv(void               *sendbuf,
                   HYPRE_Int          *sendcounts,
                   HYPRE_Int          *displs,
                   hypre_MPI_Datatype  sendtype,
                   void               *recvbuf,
                   HYPRE_Int           recvcount,
                   hypre_MPI_Datatype  recvtype,
                   HYPRE_Int           root,
                   hypre_MPI_Comm      comm )
{
   hypre_int *mpi_sendcounts = NULL;
   hypre_int *mpi_displs = NULL;
   hypre_int csize, croot;
   HYPRE_Int  i;
   HYPRE_Int  ierr;

   MPI_Comm_size(comm, &csize);
   MPI_Comm_rank(comm, &croot);
   if (croot == (hypre_int) root)
   {
      mpi_sendcounts = hypre_TAlloc(hypre_int,  csize, HYPRE_MEMORY_HOST);
      mpi_displs = hypre_TAlloc(hypre_int,  csize, HYPRE_MEMORY_HOST);
      for (i = 0; i < csize; i++)
      {
         mpi_sendcounts[i] = (hypre_int) sendcounts[i];
         mpi_displs[i] = (hypre_int) displs[i];
      }
   }
   ierr = (HYPRE_Int) MPI_Scatterv(sendbuf, mpi_sendcounts, mpi_displs, sendtype,
                                   recvbuf, (hypre_int) recvcount,
                                   recvtype, (hypre_int) root, comm);
   hypre_TFree(mpi_sendcounts, HYPRE_MEMORY_HOST);
   hypre_TFree(mpi_displs, HYPRE_MEMORY_HOST);

   return ierr;
}

HYPRE_Int
hypre_MPI_Bcast( void               *buffer,
                 HYPRE_Int           count,
                 hypre_MPI_Datatype  datatype,
                 HYPRE_Int           root,
                 hypre_MPI_Comm      comm )
{
   return (HYPRE_Int) MPI_Bcast(buffer, (hypre_int)count, datatype,
                                (hypre_int)root, comm);
}

HYPRE_Int
hypre_MPI_Send( void               *buf,
                HYPRE_Int           count,
                hypre_MPI_Datatype  datatype,
                HYPRE_Int           dest,
                HYPRE_Int           tag,
                hypre_MPI_Comm      comm )
{
   return (HYPRE_Int) MPI_Send(buf, (hypre_int)count, datatype,
                               (hypre_int)dest, (hypre_int)tag, comm);
}

HYPRE_Int
hypre_MPI_Recv( void               *buf,
                HYPRE_Int           count,
                hypre_MPI_Datatype  datatype,
                HYPRE_Int           source,
                HYPRE_Int           tag,
                hypre_MPI_Comm      comm,
                hypre_MPI_Status   *status )
{
   return (HYPRE_Int) MPI_Recv(buf, (hypre_int)count, datatype,
                               (hypre_int)source, (hypre_int)tag, comm, status);
}

HYPRE_Int
hypre_MPI_Isend( void               *buf,
                 HYPRE_Int           count,
                 hypre_MPI_Datatype  datatype,
                 HYPRE_Int           dest,
                 HYPRE_Int           tag,
                 hypre_MPI_Comm      comm,
                 hypre_MPI_Request  *request )
{
   return (HYPRE_Int) MPI_Isend(buf, (hypre_int)count, datatype,
                                (hypre_int)dest, (hypre_int)tag, comm,
                                &hypre_MPI_RequestMPI_Request(*request));
}

HYPRE_Int
hypre_MPI_Irecv( void               *buf,
                 HYPRE_Int           count,
                 hypre_MPI_Datatype  datatype,
                 HYPRE_Int           source,
                 HYPRE_Int           tag,
                 hypre_MPI_Comm      comm,
                 hypre_MPI_Request  *request )
{
   return (HYPRE_Int) MPI_Irecv(buf, (hypre_int)count, datatype,
                                (hypre_int)source, (hypre_int)tag, comm,
                                &hypre_MPI_RequestMPI_Request(*request));
}

#define TYPE_MACRO_SEND      0
#define TYPE_MACRO_RECV      1
#define TYPE_MACRO_SEND_INIT 2
#define TYPE_MACRO_RECV_INIT 3

#define TYPE_MACRO(MPI_CMD, SEND_RECV, HYPRE_DTYPE, HYPRE_MPI_DTYPE)                  \
{                                                                                     \
   if (datatype == HYPRE_MPI_DTYPE)                                                   \
   {                                                                                  \
      if (!num)                                                                       \
      {                                                                               \
         return hypre_error_flag;                                                     \
      }                                                                               \
      HYPRE_Int i, ntot = displs[num];                                                \
      void *cbuf = NULL;                                                              \
      if (SEND_RECV == TYPE_MACRO_SEND || SEND_RECV == TYPE_MACRO_SEND_INIT)          \
      {                                                                               \
         cbuf = hypre_MPICommGetSendCopy(comm);                                       \
      }                                                                               \
      else if (SEND_RECV == TYPE_MACRO_RECV || SEND_RECV == TYPE_MACRO_RECV_INIT)     \
      {                                                                               \
         cbuf = hypre_MPICommGetRecvCopy(comm);                                       \
      }                                                                               \
      HYPRE_DTYPE *_buf = (HYPRE_DTYPE *) (cbuf ? cbuf : buf);                        \
      if (SEND_RECV == TYPE_MACRO_SEND && _buf != buf)                                \
      {                                                                               \
         hypre_GpuProfilingPushRange("MPI-D2H");                                      \
         _hypre_TMemcpy(_buf, buf, HYPRE_DTYPE, ntot,                                 \
                        hypre_MPICommGetSendCopyLocation(comm), memory_location);     \
         hypre_GpuProfilingPopRange();                                                \
      }                                                                               \
      for (i = 0; i < num; i++)                                                       \
      {                                                                               \
         HYPRE_Int ip = procs[i];                                                     \
         HYPRE_Int start = displs[i];                                                 \
         HYPRE_Int len = counts ? counts[i] : displs[i + 1] - start;                  \
         MPI_CMD(_buf + start, len, HYPRE_MPI_DTYPE,                                  \
                 ip, tag, comm,                                                       \
                 &hypre_MPI_RequestMPI_Request(requests[i]));                         \
      }                                                                               \
      if (_buf != buf)                                                                \
      {                                                                               \
         /* register pre/post action in the first request */                          \
         if (SEND_RECV == TYPE_MACRO_SEND_INIT)                                       \
         {                                                                            \
            hypre_MPI_RequestSetActionCopy(0, _buf,                                   \
                                           hypre_MPICommGetSendCopyLocation(comm),    \
                                           buf,                                       \
                                           memory_location,                           \
                                           ntot * sizeof(HYPRE_DTYPE),                \
                                           &requests[0]);                             \
         }                                                                            \
         else if (SEND_RECV == TYPE_MACRO_RECV || SEND_RECV == TYPE_MACRO_RECV_INIT)  \
         {                                                                            \
            hypre_MPI_RequestSetActionCopy(1, buf,                                    \
                                           memory_location,                           \
                                           _buf,                                      \
                                           hypre_MPICommGetRecvCopyLocation(comm),    \
                                           ntot * sizeof(HYPRE_DTYPE),                \
                                           &requests[0]);                             \
         }                                                                            \
      }                                                                               \
      return hypre_error_flag;                                                        \
   }                                                                                  \
}

HYPRE_Int
hypre_MPI_Isend_Multiple( void               *buf,
                          HYPRE_Int           num,
                          HYPRE_Int          *displs,
                          HYPRE_Int          *counts,
                          hypre_MPI_Datatype  datatype,
                          HYPRE_Int          *procs,
                          HYPRE_Int           tag,
                          hypre_MPI_Comm      comm,
                          hypre_MPI_Request  *requests )
{
   hypre_MemoryLocation memory_location = hypre_MPICommGetSendLocation(comm);

   TYPE_MACRO(MPI_Isend, TYPE_MACRO_SEND, HYPRE_Complex, HYPRE_MPI_COMPLEX);
   TYPE_MACRO(MPI_Isend, TYPE_MACRO_SEND, HYPRE_Int,     HYPRE_MPI_INT);
   TYPE_MACRO(MPI_Isend, TYPE_MACRO_SEND, HYPRE_BigInt,  HYPRE_MPI_BIG_INT);

   return hypre_error_flag;
}

HYPRE_Int
hypre_MPI_Irecv_Multiple( void               *buf,
                          HYPRE_Int           num,
                          HYPRE_Int          *displs,
                          HYPRE_Int          *counts,
                          hypre_MPI_Datatype  datatype,
                          HYPRE_Int          *procs,
                          HYPRE_Int           tag,
                          hypre_MPI_Comm      comm,
                          hypre_MPI_Request  *requests )
{
   hypre_MemoryLocation memory_location = hypre_MPICommGetRecvLocation(comm);

   TYPE_MACRO(MPI_Irecv, TYPE_MACRO_RECV, HYPRE_Complex, HYPRE_MPI_COMPLEX);
   TYPE_MACRO(MPI_Irecv, TYPE_MACRO_RECV, HYPRE_Int,     HYPRE_MPI_INT);
   TYPE_MACRO(MPI_Irecv, TYPE_MACRO_RECV, HYPRE_BigInt,  HYPRE_MPI_BIG_INT);

   return hypre_error_flag;
}

HYPRE_Int
hypre_MPI_Send_init( void               *buf,
                     HYPRE_Int           count,
                     hypre_MPI_Datatype  datatype,
                     HYPRE_Int           dest,
                     HYPRE_Int           tag,
                     hypre_MPI_Comm      comm,
                     hypre_MPI_Request  *request )
{
   return (HYPRE_Int) MPI_Send_init(buf, (hypre_int)count, datatype,
                                    (hypre_int)dest, (hypre_int)tag, comm,
                                    &hypre_MPI_RequestMPI_Request(*request));
}

HYPRE_Int
hypre_MPI_Send_init_Multiple( void               *buf,
                              HYPRE_Int           num,
                              HYPRE_Int          *displs,
                              HYPRE_Int          *counts,
                              hypre_MPI_Datatype  datatype,
                              HYPRE_Int          *procs,
                              HYPRE_Int           tag,
                              hypre_MPI_Comm      comm,
                              hypre_MPI_Request  *requests )
{
   hypre_MemoryLocation memory_location = hypre_MPICommGetSendLocation(comm);

   TYPE_MACRO(MPI_Send_init, TYPE_MACRO_SEND_INIT, HYPRE_Complex, HYPRE_MPI_COMPLEX);
   TYPE_MACRO(MPI_Send_init, TYPE_MACRO_SEND_INIT, HYPRE_Int,     HYPRE_MPI_INT);
   TYPE_MACRO(MPI_Send_init, TYPE_MACRO_SEND_INIT, HYPRE_BigInt,  HYPRE_MPI_BIG_INT);

   return hypre_error_flag;
}

HYPRE_Int
hypre_MPI_Recv_init( void               *buf,
                     HYPRE_Int           count,
                     hypre_MPI_Datatype  datatype,
                     HYPRE_Int           dest,
                     HYPRE_Int           tag,
                     hypre_MPI_Comm      comm,
                     hypre_MPI_Request  *request )
{
   return (HYPRE_Int) MPI_Recv_init(buf, (hypre_int)count, datatype,
                                    (hypre_int)dest, (hypre_int)tag, comm,
                                    &hypre_MPI_RequestMPI_Request(*request));
}

HYPRE_Int
hypre_MPI_Recv_init_Multiple( void               *buf,
                              HYPRE_Int           num,
                              HYPRE_Int          *displs,
                              HYPRE_Int          *counts,
                              hypre_MPI_Datatype  datatype,
                              HYPRE_Int          *procs,
                              HYPRE_Int           tag,
                              hypre_MPI_Comm      comm,
                              hypre_MPI_Request  *requests )
{
   hypre_MemoryLocation memory_location = hypre_MPICommGetRecvLocation(comm);

   TYPE_MACRO(MPI_Recv_init, TYPE_MACRO_RECV_INIT, HYPRE_Complex, HYPRE_MPI_COMPLEX);
   TYPE_MACRO(MPI_Recv_init, TYPE_MACRO_RECV_INIT, HYPRE_Int,     HYPRE_MPI_INT);
   TYPE_MACRO(MPI_Recv_init, TYPE_MACRO_RECV_INIT, HYPRE_BigInt,  HYPRE_MPI_BIG_INT);

   return hypre_error_flag;
}

HYPRE_Int
hypre_MPI_Irsend( void               *buf,
                  HYPRE_Int           count,
                  hypre_MPI_Datatype  datatype,
                  HYPRE_Int           dest,
                  HYPRE_Int           tag,
                  hypre_MPI_Comm      comm,
                  hypre_MPI_Request  *request )
{
   return (HYPRE_Int) MPI_Irsend(buf, (hypre_int)count, datatype,
                                 (hypre_int)dest, (hypre_int)tag, comm,
                                 &hypre_MPI_RequestMPI_Request(*request));
}

HYPRE_Int
hypre_MPI_Startall( HYPRE_Int          count,
                    hypre_MPI_Request *array_of_requests )
{
   HYPRE_Int i, ierr;
   MPI_Request *array_of_mpi_requests = hypre_CTAlloc(MPI_Request, count, HYPRE_MEMORY_HOST);

   for (i = 0; i < count; i++)
   {
      array_of_mpi_requests[i] = hypre_MPI_RequestMPI_Request(array_of_requests[i]);
      hypre_MPI_RequestProcessAction(0, &array_of_requests[i]);
   }

   ierr = (HYPRE_Int) MPI_Startall((hypre_int)count, array_of_mpi_requests);

   for (i = 0; i < count; i++)
   {
      hypre_MPI_RequestMPI_Request(array_of_requests[i]) = array_of_mpi_requests[i];
   }

   hypre_TFree(array_of_mpi_requests, HYPRE_MEMORY_HOST);

   return ierr;
}

HYPRE_Int
hypre_MPI_Probe( HYPRE_Int         source,
                 HYPRE_Int         tag,
                 hypre_MPI_Comm    comm,
                 hypre_MPI_Status *status )
{
   return (HYPRE_Int) MPI_Probe((hypre_int)source, (hypre_int)tag, comm, status);
}

HYPRE_Int
hypre_MPI_Iprobe( HYPRE_Int         source,
                  HYPRE_Int         tag,
                  hypre_MPI_Comm    comm,
                  HYPRE_Int        *flag,
                  hypre_MPI_Status *status )
{
   hypre_int mpi_flag;
   HYPRE_Int ierr;
   ierr = (HYPRE_Int) MPI_Iprobe((hypre_int)source, (hypre_int)tag, comm,
                                 &mpi_flag, status);
   *flag = (HYPRE_Int) mpi_flag;
   return ierr;
}

HYPRE_Int
hypre_MPI_Test( hypre_MPI_Request *request,
                HYPRE_Int         *flag,
                hypre_MPI_Status  *status )
{
   hypre_int mpi_flag;
   HYPRE_Int ierr;
   ierr = (HYPRE_Int) MPI_Test(&hypre_MPI_RequestMPI_Request(*request), &mpi_flag, status);
   *flag = (HYPRE_Int) mpi_flag;
   return ierr;
}

HYPRE_Int
hypre_MPI_Testall( HYPRE_Int          count,
                   hypre_MPI_Request *array_of_requests,
                   HYPRE_Int         *flag,
                   hypre_MPI_Status  *array_of_statuses )
{
   hypre_int mpi_flag;
   HYPRE_Int i, ierr;

   MPI_Request *array_of_mpi_requests = hypre_TAlloc(MPI_Request, count, HYPRE_MEMORY_HOST);
   for (i = 0; i < count; i++)
   {
      array_of_mpi_requests[i] = hypre_MPI_RequestMPI_Request(array_of_requests[i]);
   }

   ierr = (HYPRE_Int) MPI_Testall((hypre_int)count, array_of_mpi_requests,
                                  &mpi_flag, array_of_statuses);
   *flag = (HYPRE_Int) mpi_flag;

   for (i = 0; i < count; i++)
   {
      hypre_MPI_RequestMPI_Request(array_of_requests[i]) = array_of_mpi_requests[i];
   }

   hypre_TFree(array_of_mpi_requests, HYPRE_MEMORY_HOST);

   return ierr;
}

HYPRE_Int
hypre_MPI_Wait( hypre_MPI_Request *request,
                hypre_MPI_Status  *status )
{
   return (HYPRE_Int) MPI_Wait(&hypre_MPI_RequestMPI_Request(*request), status);
}

HYPRE_Int
hypre_MPI_Waitall( HYPRE_Int          count,
                   hypre_MPI_Request *array_of_requests,
                   hypre_MPI_Status  *array_of_statuses )
{
   hypre_GpuProfilingPushRange("hypre_MPI_Waitall");

   HYPRE_Int i, ierr;

   MPI_Request *array_of_mpi_requests = hypre_TAlloc(MPI_Request, count, HYPRE_MEMORY_HOST);
   for (i = 0; i < count; i++)
   {
      array_of_mpi_requests[i] = hypre_MPI_RequestMPI_Request(array_of_requests[i]);
   }

   ierr = (HYPRE_Int) MPI_Waitall((hypre_int)count,
                                   array_of_mpi_requests, array_of_statuses);

   for (i = 0; i < count; i++)
   {
      hypre_MPI_RequestMPI_Request(array_of_requests[i]) = array_of_mpi_requests[i];
      hypre_MPI_RequestProcessAction(1, &array_of_requests[i]);
   }

   hypre_TFree(array_of_mpi_requests, HYPRE_MEMORY_HOST);

   hypre_GpuProfilingPopRange();

   return ierr;
}

HYPRE_Int
hypre_MPI_Waitany( HYPRE_Int          count,
                   hypre_MPI_Request *array_of_requests,
                   HYPRE_Int         *index,
                   hypre_MPI_Status  *status )
{
   hypre_int mpi_index;
   HYPRE_Int i, ierr;

   MPI_Request *array_of_mpi_requests = hypre_TAlloc(MPI_Request, count, HYPRE_MEMORY_HOST);
   for (i = 0; i < count; i++)
   {
      array_of_mpi_requests[i] = hypre_MPI_RequestMPI_Request(array_of_requests[i]);
   }

   ierr = (HYPRE_Int) MPI_Waitany((hypre_int)count, array_of_mpi_requests,
                                  &mpi_index, status);
   *index = (HYPRE_Int) mpi_index;

   for (i = 0; i < count; i++)
   {
      hypre_MPI_RequestMPI_Request(array_of_requests[i]) = array_of_mpi_requests[i];
   }

   hypre_TFree(array_of_mpi_requests, HYPRE_MEMORY_HOST);

   return ierr;
}

HYPRE_Int
hypre_MPI_Allreduce( void              *sendbuf,
                     void              *recvbuf,
                     HYPRE_Int          count,
                     hypre_MPI_Datatype datatype,
                     hypre_MPI_Op       op,
                     hypre_MPI_Comm     comm )
{
   hypre_GpuProfilingPushRange("MPI_Allreduce");

   HYPRE_Int result = MPI_Allreduce(sendbuf, recvbuf, (hypre_int)count,
                                    datatype, op, comm);

   hypre_GpuProfilingPopRange();

   return result;
}

HYPRE_Int
hypre_MPI_Reduce( void               *sendbuf,
                  void               *recvbuf,
                  HYPRE_Int           count,
                  hypre_MPI_Datatype  datatype,
                  hypre_MPI_Op        op,
                  HYPRE_Int           root,
                  hypre_MPI_Comm      comm )
{
   return (HYPRE_Int) MPI_Reduce(sendbuf, recvbuf, (hypre_int)count,
                                 datatype, op, (hypre_int)root, comm);
}

HYPRE_Int
hypre_MPI_Scan( void               *sendbuf,
                void               *recvbuf,
                HYPRE_Int           count,
                hypre_MPI_Datatype  datatype,
                hypre_MPI_Op        op,
                hypre_MPI_Comm      comm )
{
   return (HYPRE_Int) MPI_Scan(sendbuf, recvbuf, (hypre_int)count,
                               datatype, op, comm);
}

HYPRE_Int
hypre_MPI_Request_free( hypre_MPI_Request *request )
{
   return (HYPRE_Int) MPI_Request_free(&hypre_MPI_RequestMPI_Request(*request));
}

HYPRE_Int
hypre_MPI_Type_contiguous( HYPRE_Int           count,
                           hypre_MPI_Datatype  oldtype,
                           hypre_MPI_Datatype *newtype )
{
   return (HYPRE_Int) MPI_Type_contiguous((hypre_int)count, oldtype, newtype);
}

HYPRE_Int
hypre_MPI_Type_vector( HYPRE_Int           count,
                       HYPRE_Int           blocklength,
                       HYPRE_Int           stride,
                       hypre_MPI_Datatype  oldtype,
                       hypre_MPI_Datatype *newtype )
{
   return (HYPRE_Int) MPI_Type_vector((hypre_int)count, (hypre_int)blocklength,
                                      (hypre_int)stride, oldtype, newtype);
}

HYPRE_Int
hypre_MPI_Type_hvector( HYPRE_Int           count,
                        HYPRE_Int           blocklength,
                        hypre_MPI_Aint      stride,
                        hypre_MPI_Datatype  oldtype,
                        hypre_MPI_Datatype *newtype )
{
#if MPI_VERSION > 1
   return (HYPRE_Int) MPI_Type_create_hvector((hypre_int)count, (hypre_int)blocklength,
                                              stride, oldtype, newtype);
#else
   return (HYPRE_Int) MPI_Type_hvector((hypre_int)count, (hypre_int)blocklength,
                                       stride, oldtype, newtype);
#endif
}

HYPRE_Int
hypre_MPI_Type_struct( HYPRE_Int           count,
                       HYPRE_Int          *array_of_blocklengths,
                       hypre_MPI_Aint     *array_of_displacements,
                       hypre_MPI_Datatype *array_of_types,
                       hypre_MPI_Datatype *newtype )
{
   hypre_int *mpi_array_of_blocklengths;
   HYPRE_Int  i;
   HYPRE_Int  ierr;

   mpi_array_of_blocklengths = hypre_TAlloc(hypre_int,  count, HYPRE_MEMORY_HOST);
   for (i = 0; i < count; i++)
   {
      mpi_array_of_blocklengths[i] = (hypre_int) array_of_blocklengths[i];
   }

#if MPI_VERSION > 1
   ierr = (HYPRE_Int) MPI_Type_create_struct((hypre_int)count, mpi_array_of_blocklengths,
                                             array_of_displacements, array_of_types,
                                             newtype);
#else
   ierr = (HYPRE_Int) MPI_Type_struct((hypre_int)count, mpi_array_of_blocklengths,
                                      array_of_displacements, array_of_types,
                                      newtype);
#endif

   hypre_TFree(mpi_array_of_blocklengths, HYPRE_MEMORY_HOST);

   return ierr;
}

HYPRE_Int
hypre_MPI_Type_commit( hypre_MPI_Datatype *datatype )
{
   return (HYPRE_Int) MPI_Type_commit(datatype);
}

HYPRE_Int
hypre_MPI_Type_free( hypre_MPI_Datatype *datatype )
{
   return (HYPRE_Int) MPI_Type_free(datatype);
}

HYPRE_Int
hypre_MPI_Op_free( hypre_MPI_Op *op )
{
   return (HYPRE_Int) MPI_Op_free(op);
}

HYPRE_Int
hypre_MPI_Op_create( hypre_MPI_User_function *function, hypre_int commute, hypre_MPI_Op *op )
{
   return (HYPRE_Int) MPI_Op_create(function, commute, op);
}

#if defined(HYPRE_USING_GPU) || defined(HYPRE_USING_DEVICE_OPENMP)
HYPRE_Int
hypre_MPI_Comm_split_type( MPI_Comm comm, HYPRE_Int split_type, HYPRE_Int key,
                           hypre_MPI_Info info, MPI_Comm *newcomm )
{
   return (HYPRE_Int) MPI_Comm_split_type(comm, split_type, key, info, newcomm );
}

HYPRE_Int
hypre_MPI_Info_create( hypre_MPI_Info *info )
{
   return (HYPRE_Int) MPI_Info_create(info);
}

HYPRE_Int
hypre_MPI_Info_free( hypre_MPI_Info *info )
{
   return (HYPRE_Int) MPI_Info_free(info);
}
#endif

HYPRE_Int
hypre_MPINeedHostBuffer(hypre_MemoryLocation memory_location)
{
#if defined(HYPRE_USING_GPU)
   return !hypre_GetGpuAwareMPI()                       &&
           memory_location != hypre_MEMORY_HOST         &&
           memory_location != hypre_MEMORY_HOST_PINNED;
#else
   /* RL: return 1 for debugging purpose */
   return 1;
#endif
}

#endif
