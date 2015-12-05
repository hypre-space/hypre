/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.13 $
 ***********************************************************************EHEADER*/

#include "_hypre_utilities.h"

/******************************************************************************
 * This routine is the same in both the sequential and normal cases
 *
 * The 'comm' argument for MPI_Comm_f2c is MPI_Fint, which is always the size of
 * a Fortran integer and hence usually the size of hypre_int.
 *****************************************************************************/

hypre_MPI_Comm
hypre_MPI_Comm_f2c( hypre_int comm )
{
#ifdef HYPRE_HAVE_MPI_COMM_F2C
   return (hypre_MPI_Comm) MPI_Comm_f2c(comm);
#else
   return (hypre_MPI_Comm) comm;
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

double
hypre_MPI_Wtime( )
{
   return(0.0);
}

double
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
      case HYPRE_MPI_INT:
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
         memcpy(recvbuf, sendbuf, sendcount);
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
      case HYPRE_MPI_INT:
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
         memcpy(recvbuf, sendbuf, count);
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

double
hypre_MPI_Wtime( )
{
   return MPI_Wtime();
}

double
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

   mpi_ranks = hypre_TAlloc(hypre_int, n);
   for (i = 0; i < n; i++)
   {
      mpi_ranks[i] = (hypre_int) ranks[i];
   }
   ierr = (HYPRE_Int) MPI_Group_incl(group, (hypre_int)n, mpi_ranks, newgroup);
   hypre_TFree(mpi_ranks);

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
   return (HYPRE_Int) MPI_Address(location, address);
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
   mpi_recvcounts = hypre_TAlloc(hypre_int, csize);
   mpi_displs = hypre_TAlloc(hypre_int, csize);
   for (i = 0; i < csize; i++)
   {
      mpi_recvcounts[i] = (hypre_int) recvcounts[i];
      mpi_displs[i] = (hypre_int) displs[i];
   }
   ierr = (HYPRE_Int) MPI_Allgatherv(sendbuf, (hypre_int)sendcount, sendtype,
                                     recvbuf, mpi_recvcounts, mpi_displs, 
                                     recvtype, comm);
   hypre_TFree(mpi_recvcounts);
   hypre_TFree(mpi_displs);

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
   return (HYPRE_Int) MPI_Type_hvector((hypre_int)count, (hypre_int)blocklength,
                                       stride, oldtype, newtype);
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

   mpi_array_of_blocklengths = hypre_TAlloc(hypre_int, count);
   for (i = 0; i < count; i++)
   {
      mpi_array_of_blocklengths[i] = (hypre_int) array_of_blocklengths[i];
   }
   ierr = (HYPRE_Int) MPI_Type_struct((hypre_int)count, mpi_array_of_blocklengths,
                                      array_of_displacements, array_of_types,
                                      newtype);
   hypre_TFree(mpi_array_of_blocklengths);

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

#endif
