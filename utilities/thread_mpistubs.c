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

#define MPI_DEFINED 

#include "thread_mpistubs.h"
#include "hypre_utilities.h"
#include <mpi.h>
#include "threading.h"

int th_sem;

int
thread_MPI_Init( int    *argc,
          char ***argv)
{
  if (pthread_equal(pthread_self(),hypre_thread[0]))
  {
    MPI_Init(argc,argv);
  }
  else
  {
    return(0);
  }
}

double
thread_MPI_Wtime( )
{
  if (pthread_equal(pthread_self(),hypre_thread[0]))
  {
    MPI_Wtime();
  }
  else
  {
    return(0.0);
  }
}

double
thread_MPI_Wtick( )
{
  if (pthread_equal(pthread_self(),hypre_thread[0]))
  {
    MPI_Wtick();
  }
  else
  {
    return(0.0);
  }
}

int
thread_MPI_Barrier( MPI_Comm comm )
{
  hypre_barrier(&mpi_mtx,&mpi_cnd,&th_sem);
  if (pthread_equal(pthread_self(),hypre_thread[0]))
  {
    MPI_Barrier(comm);
  }
  else
  {
  return(0);
  }
  hypre_barrier(&mpi_mtx,&mpi_cnd,&th_sem);
}

int
thread_MPI_Finalize( )
{
  hypre_barrier(&mpi_mtx,&mpi_cnd,&th_sem);
  if (pthread_equal(pthread_self(),hypre_thread[0]))
  {
    MPI_Finalize();
  }
  else
  {
    return(0);
  }
}

int
thread_MPI_Comm_group( MPI_Comm   comm,
                MPI_Group *group )
{
  if (pthread_equal(pthread_self(),hypre_thread[0]))
  {
    MPI_Comm_group(comm,group );
  }
  else
  {
   return(0);
  }
}

int
thread_MPI_Comm_dup( MPI_Comm  comm,
              MPI_Comm *newcomm )
{
  if (pthread_equal(pthread_self(),hypre_thread[0]))
  {
    MPI_Comm_dup(comm,newcomm);
  }
  else
  {
    return(0);
  }
}

int
thread_MPI_Group_incl( MPI_Group  group,
                int        n,
                int       *ranks,
                MPI_Group *newgroup )
{
  if (pthread_equal(pthread_self(),hypre_thread[0]))
  {
    MPI_Group_incl(group,n,ranks,newgroup );
  }
  else
  {
    return(0);
  }
}

int
thread_MPI_Comm_create( MPI_Comm  comm,
                 MPI_Group group,
                 MPI_Comm *newcomm )
{
  if (pthread_equal(pthread_self(),hypre_thread[0]))
  {
    MPI_Comm_create(comm,group,newcomm );
  }
  else
  {
    return(0);
  }
}

int
thread_MPI_Allgather( void        *sendbuf,
               int          sendcount,
               MPI_Datatype sendtype,
               void        *recvbuf,
               int          recvcount,
               MPI_Datatype recvtype,
               MPI_Comm     comm      ) 
{
  int i;
  hypre_barrier(&mpi_mtx,&mpi_cnd,&th_sem);
  if (pthread_equal(pthread_self(),hypre_thread[0]))
  {
     MPI_Allgather(sendbuf,sendcount,sendtype,recvbuf,recvcount,recvtype,comm);
  }
  else
  { 
   switch (sendtype)
   {
      case MPI_INT:
      {
         int *crecvbuf = (int *)recvbuf;
         int *csendbuf = (int *)sendbuf;
         for (i = 0; i < sendcount; i++)
         {
	    crecvbuf[i] = csendbuf[i];
         }
      } 
      break;

      case MPI_DOUBLE:
      {
         double *crecvbuf = (double *)recvbuf;
         double *csendbuf = (double *)sendbuf;
         for (i = 0; i < sendcount; i++)
         {
	    crecvbuf[i] = csendbuf[i];
         }
      } 
      break;

      case MPI_CHAR:
      {
         char *crecvbuf = (char *)recvbuf;
         char *csendbuf = (char *)sendbuf;
         for (i = 0; i < sendcount; i++)
         {
	    crecvbuf[i] = csendbuf[i];
         }
      } 
      break;
   }

   return(0);
  }
  hypre_barrier(&mpi_mtx,&mpi_cnd,&th_sem);
}

int
thread_MPI_Allgatherv( void        *sendbuf,
                int          sendcount,
                MPI_Datatype sendtype,
                void        *recvbuf,
                int         *recvcounts,
                int         *displs, 
                MPI_Datatype recvtype,
                MPI_Comm     comm       ) 
{ 
  int i;
  hypre_barrier(&mpi_mtx,&mpi_cnd,&th_sem);
  if(pthread_equal(pthread_self(),hypre_thread[0]))
  {
    MPI_Allgatherv(sendbuf,sendcount,sendtype,recvbuf,recvcounts,displs,
		   recvtype,comm);
  }
  else
  {

 switch (sendtype)
   {
      case MPI_INT:
      {
         int *crecvbuf = (int *)recvbuf;
         int *csendbuf = (int *)sendbuf;
         for (i = 0; i < sendcount; i++)
         {
	    crecvbuf[i] = csendbuf[i];
         }
      } 
      break;

      case MPI_DOUBLE:
      {
         double *crecvbuf = (double *)recvbuf;
         double *csendbuf = (double *)sendbuf;
         for (i = 0; i < sendcount; i++)
         {
	    crecvbuf[i] = csendbuf[i];
         }
      } 
      break;

      case MPI_CHAR:
      {
         char *crecvbuf = (char *)recvbuf;
         char *csendbuf = (char *)sendbuf;
         for (i = 0; i < sendcount; i++)
         {
	    crecvbuf[i] = csendbuf[i];
         }
      } 
      break;
   }

   return(0);

  }
  hypre_barrier(&mpi_mtx,&mpi_cnd,&th_sem);
}

int
thread_MPI_Bcast( void        *buffer,
           int          count,
           MPI_Datatype datatype,
           int          root,
           MPI_Comm     comm     ) 
{ 
  hypre_barrier(&mpi_mtx,&mpi_cnd,&th_sem);
  if (pthread_equal(pthread_self(),hypre_thread[0]))
  {
    MPI_Bcast(buffer,count,datatype,root,comm);
  }
  else
  {
   return(0);
  }

  hypre_barrier(&mpi_mtx,&mpi_cnd,&th_sem);
}

int
thread_MPI_Send( void        *buf,
          int          count,
          MPI_Datatype datatype,
          int          dest,
          int          tag,
          MPI_Comm     comm     ) 
{ 
  hypre_barrier(&mpi_mtx,&mpi_cnd,&th_sem); 
  if (pthread_equal(pthread_self(),hypre_thread[0]))
  {
    MPI_Send(buf,count,datatype,dest,tag,comm);
  }
  else
  {
   return(0);
  }
  hypre_barrier(&mpi_mtx,&mpi_cnd,&th_sem); 
}

int
thread_MPI_Recv( void        *buf,
          int          count,
          MPI_Datatype datatype,
          int          source,
          int          tag,
          MPI_Comm     comm,
          MPI_Status  *status   )
{ 
  hypre_barrier(&mpi_mtx,&mpi_cnd,&th_sem); 
  if (pthread_equal(pthread_self(),hypre_thread[0]))
  {
    MPI_Recv(buf,count,datatype,source,tag,comm,status);
  }
  else
  {
    return(0);
  }
  hypre_barrier(&mpi_mtx,&mpi_cnd,&th_sem);
}

int
thread_MPI_Isend( void        *buf,
           int          count,
           MPI_Datatype datatype,
           int          dest,
           int          tag,
           MPI_Comm     comm,
           MPI_Request *request  )
{ 
  hypre_barrier(&mpi_mtx,&mpi_cnd,&th_sem);
  if (pthread_equal(pthread_self(),hypre_thread[0]))
  {
    MPI_Isend(buf,count,datatype,dest,tag,comm,request);
  }
  else
  {
    return(0);
  }
  hypre_barrier(&mpi_mtx,&mpi_cnd,&th_sem);
}

int
thread_MPI_Irecv( void        *buf,
           int          count,
           MPI_Datatype datatype,
           int          source,
           int          tag,
           MPI_Comm     comm,
           MPI_Request *request  )
{ 
  hypre_barrier(&mpi_mtx,&mpi_cnd,&th_sem);
  if (pthread_equal(pthread_self(),hypre_thread[0]))
  {
    MPI_Irecv(buf,count,datatype,source,tag,comm,request);
  }
  else
  {
    return(0);
  }
  hypre_barrier(&mpi_mtx,&mpi_cnd,&th_sem);
}

int
thread_MPI_Wait( MPI_Request *request,
          MPI_Status  *status  )
{
  hypre_barrier(&mpi_mtx,&mpi_cnd,&th_sem);
  if (pthread_equal(pthread_self(),hypre_thread[0]))
  {
    MPI_Wait;
  }
  else
  {
   return(0);
  }
  hypre_barrier(&mpi_mtx,&mpi_cnd,&th_sem);
}

int
thread_MPI_Waitall( int          count,
             MPI_Request *array_of_requests,
             MPI_Status  *array_of_statuses )
{
  hypre_barrier(&mpi_mtx,&mpi_cnd,&th_sem);
  if (pthread_equal(pthread_self(),hypre_thread[0]))
  {
    MPI_Waitall(count,array_of_requests,array_of_statuses);
  }
  else
  {
   return(0);
  }
  hypre_barrier(&mpi_mtx,&mpi_cnd,&th_sem);
}

int
thread_MPI_Waitany( int          count,
             MPI_Request *array_of_requests,
             int         *index,
             MPI_Status  *status            )
{
  hypre_barrier(&mpi_mtx,&mpi_cnd,&th_sem);
 if (pthread_equal(pthread_self(),hypre_thread[0]))
 {
   MPI_Waitany(count,array_of_requests,index,status);
 }
  else
  {
   return(0);
  }
  hypre_barrier(&mpi_mtx,&mpi_cnd,&th_sem);
}

int
thread_MPI_Comm_size( MPI_Comm comm,
               int     *size )
{ 
  *size = 1;
  if (pthread_equal(pthread_self(),hypre_thread[0]))
  {
    MPI_Comm_size(comm,size);
  }
  else
  {
   return(0);
  }
}

int
thread_MPI_Comm_rank( MPI_Comm comm,
               int     *rank )
{ 
  *rank=0;
  if (pthread_equal(pthread_self(),hypre_thread[0]))
  {
    MPI_Comm_rank(comm,rank);
  }
  else
  {
   return(0);
  }
}

int
thread_MPI_Allreduce( void        *sendbuf,
               void        *recvbuf,
               int          count,
               MPI_Datatype datatype,
               MPI_Op       op,
               MPI_Comm     comm     )
{ 
  hypre_barrier(&mpi_mtx,&mpi_cnd,&th_sem);
  if (pthread_equal(pthread_self(),hypre_thread[0]))
  {
    MPI_Allreduce(sendbuf,recvbuf,count,datatype,op,comm);
  }
  else
  {
   switch (datatype)
   {
      case MPI_INT:
      {
         int *crecvbuf = (int *)recvbuf;
         int *csendbuf = (int *)sendbuf;
         crecvbuf[0] = csendbuf[0];
      } 
      break;

      case MPI_DOUBLE:
      {
         double *crecvbuf = (double *)recvbuf;
         double *csendbuf = (double *)sendbuf;
         crecvbuf[0] = csendbuf[0];
      } 
      break;

      case MPI_CHAR:
      {
         char *crecvbuf = (char *)recvbuf;
         char *csendbuf = (char *)sendbuf;
         crecvbuf[0] = csendbuf[0];
      } 
      break;
   }

   return(0);
  }
  hypre_barrier(&mpi_mtx,&mpi_cnd,&th_sem);
}

int
thread_MPI_Type_hvector( int           count,
                  int           blocklength,
                  MPI_Aint      stride,
                  MPI_Datatype  oldtype,
                  MPI_Datatype *newtype     )
{
 if (pthread_equal(pthread_self(),hypre_thread[0]))
 {
   MPI_Type_hvector(count,blocklength,stride,oldtype,newtype);
 }
  else
  {
   return(0);
  }
  hypre_barrier(&mpi_mtx,&mpi_cnd,&th_sem);
}

int
thread_MPI_Type_struct( int           count,
                 int          *array_of_blocklengths,
                 MPI_Aint     *array_of_displacements,
                 MPI_Datatype *array_of_types,
                 MPI_Datatype *newtype                )
{
  if (pthread_equal(pthread_self(),hypre_thread[0]))
  {
    MPI_Type_struct(count,array_of_blocklengths,array_of_displacements,
		    array_of_types,newtype);
  }
  else
  {
   return(0);
  }
  hypre_barrier(&mpi_mtx,&mpi_cnd,&th_sem);
}

int
thread_MPI_Type_free( MPI_Datatype *datatype )
{
  hypre_barrier(&mpi_mtx,&mpi_cnd,&th_sem);
  if (pthread_equal(pthread_self(),hypre_thread[0]))
  {
    MPI_Type_free(datatype);
  }
  else
  {
   return(0);
  }
}

int
thread_MPI_Type_commit( MPI_Datatype *datatype )
{
  if (pthread_equal(pthread_self(),hypre_thread[0]))
  {
    MPI_Type_commit(datatype);
  }
  else
  {
   return(0);
  }
  hypre_barrier(&mpi_mtx,&mpi_cnd,&th_sem);
}

