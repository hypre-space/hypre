/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
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

hypre_MPI_Comm
hypre_MPI_Comm_f2c( hypre_int comm )
{
#ifdef HYPRE_HAVE_MPI_COMM_F2C
   return (hypre_MPI_Comm) MPI_Comm_f2c(comm);
#else
   return (hypre_MPI_Comm) (size_t)comm;
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
   return(0);
}

HYPRE_Int
hypre_MPI_Finalize( )
{
   return(0);
}

HYPRE_Int
hypre_MPI_Abort( hypre_MPI_Comm comm,
                 HYPRE_Int      errorcode )
{
   return(0);
}

HYPRE_Real
hypre_MPI_Wtime( )
{
   return(0.0);
}

HYPRE_Real
hypre_MPI_Wtick( )
{
   return(0.0);
}

HYPRE_Int
hypre_MPI_Barrier( hypre_MPI_Comm comm )
{
   return(0);
}

HYPRE_Int
hypre_MPI_Comm_create( hypre_MPI_Comm   comm,
                       hypre_MPI_Group  group,
                       hypre_MPI_Comm  *newcomm )
{
   return(0);
}

HYPRE_Int
hypre_MPI_Comm_dup( hypre_MPI_Comm  comm,
                    hypre_MPI_Comm *newcomm )
{
   return(0);
}

HYPRE_Int
hypre_MPI_Comm_size( hypre_MPI_Comm  comm,
                     HYPRE_Int      *size )
{
   *size = 1;
   return(0);
}

HYPRE_Int
hypre_MPI_Comm_rank( hypre_MPI_Comm  comm,
                     HYPRE_Int      *rank )
{
   *rank = 0;
   return(0);
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
   return(0);
}

HYPRE_Int
hypre_MPI_Comm_split( hypre_MPI_Comm  comm,
                      HYPRE_Int       n,
                      HYPRE_Int       m,
                      hypre_MPI_Comm *comms )
{
   return(0);
}

HYPRE_Int
hypre_MPI_Group_incl( hypre_MPI_Group  group,
                      HYPRE_Int        n,
                      HYPRE_Int       *ranks,
                      hypre_MPI_Group *newgroup )
{
   return(0);
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
   return(0);
}

HYPRE_Int
hypre_MPI_Get_count( hypre_MPI_Status   *status,
                     hypre_MPI_Datatype  datatype,
                     HYPRE_Int          *count )
{
   return(0);
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
   return(0);
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

      case hypre_MPI_BYTE:
      {
         hypre_Memcpy(recvbuf,  sendbuf,  sendcount, HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
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

   return(0);
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
   return(0);
}

HYPRE_Int
hypre_MPI_Send( void               *buf,
                HYPRE_Int           count,
                hypre_MPI_Datatype  datatype,
                HYPRE_Int           dest,
                HYPRE_Int           tag,
                hypre_MPI_Comm      comm )
{
   return(0);
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
   return(0);
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
   return(0);
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
   return(0);
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
   return(0);
}

HYPRE_Int
hypre_MPI_Testall( HYPRE_Int          count,
                   hypre_MPI_Request *array_of_requests,
                   HYPRE_Int         *flag,
                   hypre_MPI_Status  *array_of_statuses )
{
   *flag = 1;
   return(0);
}

HYPRE_Int
hypre_MPI_Wait( hypre_MPI_Request *request,
                hypre_MPI_Status  *status )
{
   return(0);
}

HYPRE_Int
hypre_MPI_Waitall( HYPRE_Int          count,
                   hypre_MPI_Request *array_of_requests,
                   hypre_MPI_Status  *array_of_statuses )
{
   return(0);
}

HYPRE_Int
hypre_MPI_Waitany( HYPRE_Int          count,
                   hypre_MPI_Request *array_of_requests,
                   HYPRE_Int         *index,
                   hypre_MPI_Status  *status )
{
   return(0);
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

      case hypre_MPI_BYTE:
      {
         hypre_Memcpy(recvbuf,  sendbuf,  count, HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
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
   return(0);
}

HYPRE_Int
hypre_MPI_Type_vector( HYPRE_Int           count,
                       HYPRE_Int           blocklength,
                       HYPRE_Int           stride,
                       hypre_MPI_Datatype  oldtype,
                       hypre_MPI_Datatype *newtype )
{
   return(0);
}

HYPRE_Int
hypre_MPI_Type_hvector( HYPRE_Int           count,
                        HYPRE_Int           blocklength,
                        hypre_MPI_Aint      stride,
                        hypre_MPI_Datatype  oldtype,
                        hypre_MPI_Datatype *newtype )
{
   return(0);
}

HYPRE_Int
hypre_MPI_Type_struct( HYPRE_Int           count,
                       HYPRE_Int          *array_of_blocklengths,
                       hypre_MPI_Aint     *array_of_displacements,
                       hypre_MPI_Datatype *array_of_types,
                       hypre_MPI_Datatype *newtype )
{
   return(0);
}

HYPRE_Int
hypre_MPI_Type_commit( hypre_MPI_Datatype *datatype )
{
   return(0);
}

HYPRE_Int
hypre_MPI_Type_free( hypre_MPI_Datatype *datatype )
{
   return(0);
}

HYPRE_Int
hypre_MPI_Op_create( hypre_MPI_User_function *function, hypre_int commute, hypre_MPI_Op *op )
{
   return(0);
}

HYPRE_Int
hypre_MPI_Op_free( hypre_MPI_Op *op )
{
   return(0);
}

#if defined(HYPRE_USING_GPU)
HYPRE_Int hypre_MPI_Comm_split_type( hypre_MPI_Comm comm, HYPRE_Int split_type, HYPRE_Int key, hypre_MPI_Info info, hypre_MPI_Comm *newcomm )
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

HYPRE_Int
hypre_MPI_CheckCommMatrix( hypre_MPI_Comm   comm,
                           HYPRE_Int        num_recvs,
                           HYPRE_Int       *recvs,
                           HYPRE_Int        num_sends,
                           HYPRE_Int       *sends )
{
   return(0);
}

/******************************************************************************
 * MPI stubs to do casting of HYPRE_Int and hypre_int correctly
 *****************************************************************************/

#else

HYPRE_Int
hypre_MPI_Init( hypre_int   *argc,
                char      ***argv )
{
   return (HYPRE_Int) MPI_Init(argc, argv);
}

HYPRE_Int
hypre_MPI_Finalize( )
{
   return (HYPRE_Int) MPI_Finalize();
}

HYPRE_Int
hypre_MPI_Abort( hypre_MPI_Comm comm,
                 HYPRE_Int      errorcode )
{
   return (HYPRE_Int) MPI_Abort(comm, (hypre_int)errorcode);
}

HYPRE_Real
hypre_MPI_Wtime( )
{
   return MPI_Wtime();
}

HYPRE_Real
hypre_MPI_Wtick( )
{
   return MPI_Wtick();
}

HYPRE_Int
hypre_MPI_Barrier( hypre_MPI_Comm comm )
{
   return (HYPRE_Int) MPI_Barrier(comm);
}

HYPRE_Int
hypre_MPI_Comm_create( hypre_MPI_Comm   comm,
                       hypre_MPI_Group  group,
                       hypre_MPI_Comm  *newcomm )
{
   return (HYPRE_Int) MPI_Comm_create(comm, group, newcomm);
}

HYPRE_Int
hypre_MPI_Comm_dup( hypre_MPI_Comm  comm,
                    hypre_MPI_Comm *newcomm )
{
   return (HYPRE_Int) MPI_Comm_dup(comm, newcomm);
}

HYPRE_Int
hypre_MPI_Comm_size( hypre_MPI_Comm  comm,
                     HYPRE_Int      *size )
{
   hypre_int mpi_size;
   HYPRE_Int ierr;
   ierr = (HYPRE_Int) MPI_Comm_size(comm, &mpi_size);
   *size = (HYPRE_Int) mpi_size;
   return ierr;
}

HYPRE_Int
hypre_MPI_Comm_rank( hypre_MPI_Comm  comm,
                     HYPRE_Int      *rank )
{
   hypre_int mpi_rank;
   HYPRE_Int ierr;
   ierr = (HYPRE_Int) MPI_Comm_rank(comm, &mpi_rank);
   *rank = (HYPRE_Int) mpi_rank;
   return ierr;
}

HYPRE_Int
hypre_MPI_Comm_free( hypre_MPI_Comm *comm )
{
   return (HYPRE_Int) MPI_Comm_free(comm);
}

HYPRE_Int
hypre_MPI_Comm_group( hypre_MPI_Comm   comm,
                      hypre_MPI_Group *group )
{
   return (HYPRE_Int) MPI_Comm_group(comm, group);
}

HYPRE_Int
hypre_MPI_Comm_split( hypre_MPI_Comm  comm,
                      HYPRE_Int       n,
                      HYPRE_Int       m,
                      hypre_MPI_Comm *comms )
{
   return (HYPRE_Int) MPI_Comm_split(comm, (hypre_int)n, (hypre_int)m, comms);
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
                                (hypre_int)dest, (hypre_int)tag, comm, request);
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
                                (hypre_int)source, (hypre_int)tag, comm, request);
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
                                    (hypre_int)dest, (hypre_int)tag,
                                    comm, request);
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
                                    (hypre_int)dest, (hypre_int)tag,
                                    comm, request);
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
                                 (hypre_int)dest, (hypre_int)tag, comm, request);
}

HYPRE_Int
hypre_MPI_Startall( HYPRE_Int          count,
                    hypre_MPI_Request *array_of_requests )
{
   return (HYPRE_Int) MPI_Startall((hypre_int)count, array_of_requests);
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
   ierr = (HYPRE_Int) MPI_Test(request, &mpi_flag, status);
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
   HYPRE_Int ierr;
   ierr = (HYPRE_Int) MPI_Testall((hypre_int)count, array_of_requests,
                                  &mpi_flag, array_of_statuses);
   *flag = (HYPRE_Int) mpi_flag;
   return ierr;
}

HYPRE_Int
hypre_MPI_Wait( hypre_MPI_Request *request,
                hypre_MPI_Status  *status )
{
   return (HYPRE_Int) MPI_Wait(request, status);
}

HYPRE_Int
hypre_MPI_Waitall( HYPRE_Int          count,
                   hypre_MPI_Request *array_of_requests,
                   hypre_MPI_Status  *array_of_statuses )
{
   return (HYPRE_Int) MPI_Waitall((hypre_int)count,
                                  array_of_requests, array_of_statuses);
}

HYPRE_Int
hypre_MPI_Waitany( HYPRE_Int          count,
                   hypre_MPI_Request *array_of_requests,
                   HYPRE_Int         *index,
                   hypre_MPI_Status  *status )
{
   hypre_int mpi_index;
   HYPRE_Int ierr;
   ierr = (HYPRE_Int) MPI_Waitany((hypre_int)count, array_of_requests,
                                  &mpi_index, status);
   *index = (HYPRE_Int) mpi_index;
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
   return (HYPRE_Int) MPI_Allreduce(sendbuf, recvbuf, (hypre_int)count,
                                    datatype, op, comm);
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
   return (HYPRE_Int) MPI_Request_free(request);
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

#if defined(HYPRE_USING_GPU)
HYPRE_Int
hypre_MPI_Comm_split_type( hypre_MPI_Comm comm, HYPRE_Int split_type, HYPRE_Int key, hypre_MPI_Info info, hypre_MPI_Comm *newcomm )
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
hypre_MPI_CheckCommMatrix( hypre_MPI_Comm   comm,
                           HYPRE_Int        num_recvs,
                           HYPRE_Int       *recvs,
                           HYPRE_Int        num_sends,
                           HYPRE_Int       *sends )
{
   HYPRE_Int    nprocs, myid;

   HYPRE_Int   *displs         = NULL;
   HYPRE_Int   *num_recvs_proc = NULL;
   HYPRE_Int   *num_sends_proc = NULL;
   HYPRE_Int   *global_recvs   = NULL;
   HYPRE_Int   *global_sends   = NULL;

   HYPRE_Int    global_num_recvs;
   HYPRE_Int    global_num_sends;
   HYPRE_Int    from_proc, to_proc;
   HYPRE_Int    i, entry, proc, cnt;

   hypre_MPI_Comm_rank(comm, &myid);
   hypre_MPI_Comm_size(comm, &nprocs);

   if (myid == 0)
   {
      num_recvs_proc = hypre_CTAlloc(HYPRE_Int, nprocs, HYPRE_MEMORY_HOST);
      num_sends_proc = hypre_CTAlloc(HYPRE_Int, nprocs, HYPRE_MEMORY_HOST);
      displs = hypre_CTAlloc(HYPRE_Int, nprocs + 1, HYPRE_MEMORY_HOST);
   }

   /* Gather receives */
   hypre_MPI_Gather(&num_recvs, 1, HYPRE_MPI_INT,
                    num_recvs_proc, 1, HYPRE_MPI_INT,
                    0, comm);
   if (myid == 0)
   {
      global_num_recvs = displs[0] = 0;
      for (proc = 0; proc < nprocs; proc++)
      {
         global_num_recvs += num_recvs_proc[proc];
         displs[proc + 1] = displs[proc] + num_recvs_proc[proc];
      }
      global_recvs = hypre_CTAlloc(HYPRE_Int, global_num_recvs, HYPRE_MEMORY_HOST);
   }
   hypre_MPI_Gatherv(recvs, num_recvs, HYPRE_MPI_INT,
                     global_recvs, num_recvs_proc, displs, HYPRE_MPI_INT,
                     0, comm);

   /* Gather sends */
   hypre_MPI_Gather(&num_sends, 1, HYPRE_MPI_INT,
                    num_sends_proc, 1, HYPRE_MPI_INT,
                    0, comm);
   if (myid == 0)
   {
      global_num_sends = displs[0] = 0;
      for (proc = 0; proc < nprocs; proc++)
      {
         global_num_sends += num_sends_proc[proc];
         displs[proc + 1] = displs[proc] + num_sends_proc[proc];
      }
      global_sends = hypre_CTAlloc(HYPRE_Int, global_num_sends, HYPRE_MEMORY_HOST);
   }
   hypre_MPI_Gatherv(sends, num_sends, HYPRE_MPI_INT,
                     global_sends, num_sends_proc, displs, HYPRE_MPI_INT,
                     0, comm);

   /* Check for matching send/recv. */
   if (myid == 0)
   {
      cnt = 0;
      for (from_proc = 0; from_proc < nprocs; from_proc++)
      {
         for (i = 0; i < num_sends_proc[from_proc]; i++)
         {
            to_proc = global_sends[cnt++];
            entry = hypre_BinarySearch(&global_recvs[to_proc], from_proc,
                                       num_recvs_proc[to_proc]);
            if (entry == -1)
            {
               hypre_printf("Proc %d posts a send to proc %d without a matching recv!\n",
                            from_proc, to_proc);
            }
         }
      }

      cnt = 0;
      for (from_proc = 0; from_proc < nprocs; from_proc++)
      {
         for (i = 0; i < num_recvs_proc[from_proc]; i++)
         {
            to_proc = global_recvs[cnt++];
            entry = hypre_BinarySearch(&global_sends[to_proc], from_proc,
                                       num_sends_proc[to_proc]);
            if (entry == -1)
            {
               hypre_printf("Proc %d posts a recv to proc %d without a matching send!\n",
                            from_proc, to_proc);
            }
         }
      }
   }

   if (myid == 0)
   {
      hypre_TFree(displs, HYPRE_MEMORY_HOST);
      hypre_TFree(num_recvs_proc, HYPRE_MEMORY_HOST);
      hypre_TFree(num_sends_proc, HYPRE_MEMORY_HOST);
      hypre_TFree(global_recvs, HYPRE_MEMORY_HOST);
      hypre_TFree(global_sends, HYPRE_MEMORY_HOST);
   }

   return hypre_error_flag;
}

#endif
