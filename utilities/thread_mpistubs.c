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
 *  Fake mpi stubs to generate serial codes without mpi
 *
 *****************************************************************************/

#ifdef HYPRE_USE_PTHREADS

#define HYPRE_USING_THREAD_MPISTUBS 

#include "utilities.h"
#include <mpi.h>

int
hypre_thread_MPI_Init( int    *argc,
          char ***argv)
{
  int returnval;
  int unthreaded = pthread_equal(initial_thread,pthread_self());
  int I_call_mpi = unthreaded || pthread_equal(hypre_thread[0],pthread_self());
  if (I_call_mpi)
  {
    returnval=MPI_Init(argc,argv);
  }
  else
  {
    returnval=0;
  }

  return returnval;
}

double
hypre_thread_MPI_Wtime( )
{
  double returnval;
  int unthreaded = pthread_equal(initial_thread,pthread_self());
  int I_call_mpi = unthreaded || pthread_equal(hypre_thread[0],pthread_self());
  if (I_call_mpi)
  {
    returnval=MPI_Wtime();
  }
  else
  {
    returnval=0.0;
  }

  return returnval;
}

double
hypre_thread_MPI_Wtick( )
{
  double returnval;
  int unthreaded = pthread_equal(initial_thread,pthread_self());
  int I_call_mpi = unthreaded || pthread_equal(hypre_thread[0],pthread_self());
  if (I_call_mpi)
  {
    returnval=MPI_Wtick();
  }
  else
  {
    returnval=0.0;
  }
  return returnval;
}

int
hypre_thread_MPI_Barrier( MPI_Comm comm )
{
  int returnval;
  int unthreaded = pthread_equal(initial_thread,pthread_self());
  int I_call_mpi = unthreaded || pthread_equal(hypre_thread[0],pthread_self());
  if (I_call_mpi)
  {
    returnval=MPI_Barrier(comm);
  }
  else
  {
  returnval=0;
  }
  hypre_barrier(&mpi_mtx, unthreaded);
  return returnval;
}

int
hypre_thread_MPI_Finalize( )
{
  int returnval;
  int unthreaded = pthread_equal(initial_thread,pthread_self());
  int I_call_mpi = unthreaded || pthread_equal(hypre_thread[0],pthread_self());
  hypre_barrier(&mpi_mtx, unthreaded);
  if (I_call_mpi)
  {
    returnval=MPI_Finalize();
  }
  else
  {
    returnval=0;
  }
  return returnval;
}

int
hypre_thread_MPI_Comm_group( MPI_Comm   comm,
                MPI_Group *group )
{
  int returnval;
  int unthreaded = pthread_equal(initial_thread,pthread_self());
  int I_call_mpi = unthreaded || pthread_equal(hypre_thread[0],pthread_self());
  if (I_call_mpi)
  {
    returnval=MPI_Comm_group(comm,group );
  }
  else
  {
   returnval=0;
  }
  return returnval;
}

int
hypre_thread_MPI_Comm_dup( MPI_Comm  comm,
              MPI_Comm *newcomm )
{
  int returnval;
  int unthreaded = pthread_equal(initial_thread,pthread_self());
  int I_call_mpi = unthreaded || pthread_equal(hypre_thread[0],pthread_self());
  if (I_call_mpi)
  {
    returnval=MPI_Comm_dup(comm,newcomm);
  }
  else
  {
    returnval=0;
  }
  return returnval;
}

int
hypre_thread_MPI_Group_incl( MPI_Group  group,
                int        n,
                int       *ranks,
                MPI_Group *newgroup )
{
  int returnval;
  int unthreaded = pthread_equal(initial_thread,pthread_self());
  int I_call_mpi = unthreaded || pthread_equal(hypre_thread[0],pthread_self());
  if (I_call_mpi)
  {
    returnval=MPI_Group_incl(group,n,ranks,newgroup );
  }
  else
  {
    returnval=0;
  }
  return returnval;
}

int
hypre_thread_MPI_Comm_create( MPI_Comm  comm,
                 MPI_Group group,
                 MPI_Comm *newcomm )
{
  int returnval;
  int unthreaded = pthread_equal(initial_thread,pthread_self());
  int I_call_mpi = unthreaded || pthread_equal(hypre_thread[0],pthread_self());
  if (I_call_mpi)
  {
    returnval=MPI_Comm_create(comm,group,newcomm );
  }
  else
  {
    returnval=0;
  }
  return returnval;
}

int
hypre_thread_MPI_Allgather( void        *sendbuf,
               int          sendcount,
               MPI_Datatype sendtype,
               void        *recvbuf,
               int          recvcount,
               MPI_Datatype recvtype,
               MPI_Comm     comm      ) 
{
  int returnval;
  int i;
  int unthreaded = pthread_equal(initial_thread,pthread_self());
  int I_call_mpi = unthreaded || pthread_equal(hypre_thread[0],pthread_self());
  hypre_barrier(&mpi_mtx, unthreaded);
  if (I_call_mpi)
  {
     returnval=MPI_Allgather(sendbuf,sendcount,sendtype,recvbuf,recvcount,
			     recvtype,comm);
  }
  else
  { 
     returnval=0;
  }
  hypre_barrier(&mpi_mtx, unthreaded);
  return returnval;
}

int
hypre_thread_MPI_Allgatherv( void        *sendbuf,
                int          sendcount,
                MPI_Datatype sendtype,
                void        *recvbuf,
                int         *recvcounts,
                int         *displs, 
                MPI_Datatype recvtype,
                MPI_Comm     comm       ) 
{ 
  int i,returnval;
  int unthreaded = pthread_equal(initial_thread,pthread_self());
  int I_call_mpi = unthreaded || pthread_equal(hypre_thread[0],pthread_self());
  hypre_barrier(&mpi_mtx, unthreaded);
  if (I_call_mpi)
  {
    returnval=MPI_Allgatherv(sendbuf,sendcount,sendtype,recvbuf,recvcounts,
			     displs,recvtype,comm);
  }
  else
  {
     returnval=0;
  }
  hypre_barrier(&mpi_mtx, unthreaded);
  return returnval;
}

int
hypre_thread_MPI_Bcast( void        *buffer,
           int          count,
           MPI_Datatype datatype,
           int          root,
           MPI_Comm     comm     ) 
{ 
  int returnval;
  int unthreaded = pthread_equal(initial_thread,pthread_self());
  int I_call_mpi = unthreaded || pthread_equal(hypre_thread[0],pthread_self());
  hypre_barrier(&mpi_mtx, unthreaded);
  if (I_call_mpi)
  {
    returnval=MPI_Bcast(buffer,count,datatype,root,comm);
  }
  else
  {
   returnval=0;
  }
  hypre_barrier(&mpi_mtx, unthreaded);
  return returnval;
}

int
hypre_thread_MPI_Send( void        *buf,
          int          count,
          MPI_Datatype datatype,
          int          dest,
          int          tag,
          MPI_Comm     comm     ) 
{ 
  int returnval;
  int unthreaded = pthread_equal(initial_thread,pthread_self());
  int I_call_mpi = unthreaded || pthread_equal(hypre_thread[0],pthread_self());
  hypre_barrier(&mpi_mtx, unthreaded);
  if (I_call_mpi)
  {
    returnval=MPI_Send(buf,count,datatype,dest,tag,comm);
  }
  else
  {
   returnval=0;
  }
  hypre_barrier(&mpi_mtx, unthreaded);
  return returnval;
}

int
hypre_thread_MPI_Recv( void        *buf,
          int          count,
          MPI_Datatype datatype,
          int          source,
          int          tag,
          MPI_Comm     comm,
          MPI_Status  *status   )
{ 
  int returnval;
  int unthreaded = pthread_equal(initial_thread,pthread_self());
  int I_call_mpi = unthreaded || pthread_equal(hypre_thread[0],pthread_self());
  hypre_barrier(&mpi_mtx, unthreaded);
  if (I_call_mpi)
  {
    returnval=MPI_Recv(buf,count,datatype,source,tag,comm,status);
  }
  else
  {
    returnval=0;
  }
  hypre_barrier(&mpi_mtx, unthreaded);
  return returnval;
}

int
hypre_thread_MPI_Isend( void        *buf,
           int          count,
           MPI_Datatype datatype,
           int          dest,
           int          tag,
           MPI_Comm     comm,
           MPI_Request *request  )
{ 
  int returnval;
  int unthreaded = pthread_equal(initial_thread,pthread_self());
  int I_call_mpi = unthreaded || pthread_equal(hypre_thread[0],pthread_self());
  hypre_barrier(&mpi_mtx, unthreaded);
  if (I_call_mpi)
  {
    returnval=MPI_Isend(buf,count,datatype,dest,tag,comm,request);
  }
  else
  {
    returnval=0;
  }
  return returnval;
}

int
hypre_thread_MPI_Irecv( void        *buf,
           int          count,
           MPI_Datatype datatype,
           int          source,
           int          tag,
           MPI_Comm     comm,
           MPI_Request *request  )
{ 
  int returnval;
  int unthreaded = pthread_equal(initial_thread,pthread_self());
  int I_call_mpi = unthreaded || pthread_equal(hypre_thread[0],pthread_self());
  hypre_barrier(&mpi_mtx, unthreaded);
  if (I_call_mpi)
  {
    returnval=MPI_Irecv(buf,count,datatype,source,tag,comm,request);
  }
  else
  {
    returnval=0;
  }
  return returnval;
}

int
hypre_thread_MPI_Wait( MPI_Request *request,
          MPI_Status  *status  )
{
  int returnval;
  int unthreaded = pthread_equal(initial_thread,pthread_self());
  int I_call_mpi = unthreaded || pthread_equal(hypre_thread[0],pthread_self());
  if (I_call_mpi)
  {
    returnval=MPI_Wait(request,status);
  }
  else
  {
   returnval=0;
  }
  hypre_barrier(&mpi_mtx, unthreaded);
  return returnval;
}

int
hypre_thread_MPI_Waitall( int          count,
             MPI_Request *array_of_requests,
             MPI_Status  *array_of_statuses )
{
  int returnval;
  int unthreaded = pthread_equal(initial_thread,pthread_self());
  int I_call_mpi = unthreaded || pthread_equal(hypre_thread[0],pthread_self());
  if (I_call_mpi)
  {
    returnval=MPI_Waitall(count,array_of_requests,array_of_statuses);
  }
  else
  {
   returnval=0;
  }
  hypre_barrier(&mpi_mtx, unthreaded);
  return returnval;
}

int
hypre_thread_MPI_Waitany( int          count,
             MPI_Request *array_of_requests,
             int         *index,
             MPI_Status  *status            )
{
  int returnval;
  int unthreaded = pthread_equal(initial_thread,pthread_self());
  int I_call_mpi = unthreaded || pthread_equal(hypre_thread[0],pthread_self());
  if (I_call_mpi)
  {
    returnval=MPI_Waitany(count,array_of_requests,index,status);
  }
  else
  {
   returnval=0;
  }
  hypre_barrier(&mpi_mtx, unthreaded);
  return returnval;
}

int
hypre_thread_MPI_Comm_size( MPI_Comm comm,
               int     *size )
{ 
  int returnval;
  int unthreaded = pthread_equal(initial_thread,pthread_self());
  int I_call_mpi = unthreaded || pthread_equal(hypre_thread[0],pthread_self());
  if (I_call_mpi)
  {
    returnval=MPI_Comm_size(comm,size);
  }
  else
  {
   returnval=0;
  }
  hypre_barrier(&mpi_mtx, unthreaded);
  return returnval;
}

int
hypre_thread_MPI_Comm_rank( MPI_Comm comm,
               int     *rank )
{ 
  int returnval;
  int unthreaded = pthread_equal(initial_thread,pthread_self());
  int I_call_mpi = unthreaded || pthread_equal(hypre_thread[0],pthread_self());
  if (I_call_mpi)
  {
    returnval=MPI_Comm_rank(comm,rank);
  }
  else
  {
    returnval=0;
  }
  hypre_barrier(&mpi_mtx, unthreaded);
  return returnval;
}

int
hypre_thread_MPI_Allreduce( void        *sendbuf,
               void        *recvbuf,
               int          count,
               MPI_Datatype datatype,
               MPI_Op       op,
               MPI_Comm     comm     )
{ 
  int returnval;
  int unthreaded = pthread_equal(initial_thread,pthread_self());
  int I_call_mpi = unthreaded || pthread_equal(hypre_thread[0],pthread_self());
  hypre_barrier(&mpi_mtx, unthreaded);
  if (I_call_mpi)
  {
    returnval=MPI_Allreduce(sendbuf,recvbuf,count,datatype,op,comm);
  }
  else
  {
    returnval=0;
  }
  hypre_barrier(&mpi_mtx, unthreaded);
  return returnval;
}

int
hypre_thread_MPI_Type_hvector( int           count,
                  int           blocklength,
                  MPI_Aint      stride,
                  MPI_Datatype  oldtype,
                  MPI_Datatype *newtype     )
{
  int returnval;
  int unthreaded = pthread_equal(initial_thread,pthread_self());
  int I_call_mpi = unthreaded || pthread_equal(hypre_thread[0],pthread_self());
  if (I_call_mpi)
  {
    returnval=MPI_Type_hvector(count,blocklength,stride,oldtype,newtype);
  }
  else
  {
   returnval=0;
  }
  return returnval;
}

int
hypre_thread_MPI_Type_struct( int           count,
                 int          *array_of_blocklengths,
                 MPI_Aint     *array_of_displacements,
                 MPI_Datatype *array_of_types,
                 MPI_Datatype *newtype                )
{
  int returnval;
  int unthreaded = pthread_equal(initial_thread,pthread_self());
  int I_call_mpi = unthreaded || pthread_equal(hypre_thread[0],pthread_self());
  if (I_call_mpi)
  {
    returnval=MPI_Type_struct(count,array_of_blocklengths,array_of_displacements,
		    array_of_types,newtype);
  }
  else
  {
   returnval=0;
  }
  return returnval;
}

int
hypre_thread_MPI_Type_free( MPI_Datatype *datatype )
{
  int returnval;
  int unthreaded = pthread_equal(initial_thread,pthread_self());
  int I_call_mpi = unthreaded || pthread_equal(hypre_thread[0],pthread_self());
  if (I_call_mpi)
  {
    returnval=MPI_Type_free(datatype);
  }
  else
  {
   returnval=0;
  }
  return returnval;
}

int
hypre_thread_MPI_Type_commit( MPI_Datatype *datatype )
{
  int returnval;
  int unthreaded = pthread_equal(initial_thread,pthread_self());
  int I_call_mpi = unthreaded || pthread_equal(hypre_thread[0],pthread_self());
  if (I_call_mpi)
  {
    returnval=MPI_Type_commit(datatype);
  }
  else
  {
   returnval=0;
  }
  return returnval;
}

#endif
