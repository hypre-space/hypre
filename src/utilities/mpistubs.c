/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team, UCRL-CODE-222953.
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
 * $Revision: 2.6 $
 ***********************************************************************EHEADER*/

/******************************************************************************
 *
 *  Fake mpi stubs to generate serial codes without mpi
 *
 *****************************************************************************/

#include "utilities.h"

#ifdef HYPRE_SEQUENTIAL

int
hypre_MPI_Init( int    *argc,
                char ***argv )
{
   return(0);
}

int
hypre_MPI_Finalize( )
{
   return(0);
}

int
hypre_MPI_Abort( hypre_MPI_Comm  comm,
                 int             errorcode )
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

int
hypre_MPI_Barrier( hypre_MPI_Comm comm )
{
   return(0);
}

int
hypre_MPI_Comm_create( hypre_MPI_Comm  comm,
                       hypre_MPI_Group group,
                       hypre_MPI_Comm *newcomm )
{
   return(0);
}

int
hypre_MPI_Comm_dup( hypre_MPI_Comm  comm,
                    hypre_MPI_Comm *newcomm )
{
   return(0);
}

int
hypre_MPI_Comm_size( hypre_MPI_Comm comm,
                     int           *size )
{ 
   *size = 1;
   return(0);
}

int
hypre_MPI_Comm_rank( hypre_MPI_Comm comm,
                     int           *rank )
{ 
   *rank = 0;
   return(0);
}

int
hypre_MPI_Comm_free( hypre_MPI_Comm *comm )
{
   return 0;
}

int
hypre_MPI_Comm_group( hypre_MPI_Comm   comm,
                      hypre_MPI_Group *group )
{
   return(0);
}

int
hypre_MPI_Comm_split( hypre_MPI_Comm   comm,
                      int n, int m,
                      hypre_MPI_Comm * comms )
{
   return(0);
}

int
hypre_MPI_Group_incl( hypre_MPI_Group  group,
                      int              n,
                      int             *ranks,
                      hypre_MPI_Group *newgroup )
{
   return(0);
}

int
hypre_MPI_Group_free( hypre_MPI_Group *group )
{
   return 0;
}

int
hypre_MPI_Address( void           *location,
                   hypre_MPI_Aint *address )
{
   return(0);
}

int
hypre_MPI_Get_count( hypre_MPI_Status   *status,
                     hypre_MPI_Datatype  datatype,
                     int                *count )
{
   return(0);
}

int
hypre_MPI_Alltoall( void              *sendbuf,
                    int                sendcount,
                    hypre_MPI_Datatype sendtype,
                    void              *recvbuf,
                    int                recvcount,
                    hypre_MPI_Datatype recvtype,
                    hypre_MPI_Comm     comm )
{
   return(0);
}

int
hypre_MPI_Allgather( void              *sendbuf,
                     int                sendcount,
                     hypre_MPI_Datatype sendtype,
                     void              *recvbuf,
                     int                recvcount,
                     hypre_MPI_Datatype recvtype,
                     hypre_MPI_Comm     comm ) 
{
   int i;

   switch (sendtype)
   {
      case hypre_MPI_INT:
      {
         int *crecvbuf = (int *)recvbuf;
         int *csendbuf = (int *)sendbuf;
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

int
hypre_MPI_Allgatherv( void              *sendbuf,
                      int                sendcount,
                      hypre_MPI_Datatype sendtype,
                      void              *recvbuf,
                      int               *recvcounts,
                      int               *displs, 
                      hypre_MPI_Datatype recvtype,
                      hypre_MPI_Comm     comm ) 
{ 
   return ( hypre_MPI_Allgather(sendbuf, sendcount, sendtype,
                                recvbuf, *recvcounts, recvtype, comm) );
}

int
hypre_MPI_Gather( void              *sendbuf,
                  int                sendcount,
                  hypre_MPI_Datatype sendtype,
                  void              *recvbuf,
                  int                recvcount,
                  hypre_MPI_Datatype recvtype,
                  int                root,
                  hypre_MPI_Comm     comm )
{
   return ( hypre_MPI_Allgather(sendbuf, sendcount, sendtype,
                                recvbuf, recvcount, recvtype, comm) );
}

int
hypre_MPI_Scatter( void              *sendbuf,
                   int                sendcount,
                   hypre_MPI_Datatype sendtype,
                   void              *recvbuf,
                   int                recvcount,
                   hypre_MPI_Datatype recvtype,
                   int                root,
                   hypre_MPI_Comm     comm )
{
   return ( hypre_MPI_Allgather(sendbuf, sendcount, sendtype,
                                recvbuf, recvcount, recvtype, comm) );
}

int
hypre_MPI_Bcast( void              *buffer,
                 int                count,
                 hypre_MPI_Datatype datatype,
                 int                root,
                 hypre_MPI_Comm     comm ) 
{ 
   return(0);
}

int
hypre_MPI_Send( void              *buf,
                int                count,
                hypre_MPI_Datatype datatype,
                int                dest,
                int                tag,
                hypre_MPI_Comm     comm ) 
{ 
   return(0);
}

int
hypre_MPI_Recv( void              *buf,
                int                count,
                hypre_MPI_Datatype datatype,
                int                source,
                int                tag,
                hypre_MPI_Comm     comm,
                hypre_MPI_Status  *status )
{ 
   return(0);
}

int
hypre_MPI_Isend( void              *buf,
                 int                count,
                 hypre_MPI_Datatype datatype,
                 int                dest,
                 int                tag,
                 hypre_MPI_Comm     comm,
                 hypre_MPI_Request *request )
{ 
   return(0);
}

int
hypre_MPI_Irecv( void              *buf,
                 int                count,
                 hypre_MPI_Datatype datatype,
                 int                source,
                 int                tag,
                 hypre_MPI_Comm     comm,
                 hypre_MPI_Request *request )
{ 
   return(0);
}

int
hypre_MPI_Send_init( void              *buf,
                     int                count,
                     hypre_MPI_Datatype datatype,
                     int                dest,
                     int                tag, 
                     hypre_MPI_Comm     comm,
                     hypre_MPI_Request *request )
{
   return 0;
}

int
hypre_MPI_Recv_init( void              *buf,
                     int                count,
                     hypre_MPI_Datatype datatype,
                     int                dest,
                     int                tag, 
                     hypre_MPI_Comm     comm,
                     hypre_MPI_Request *request )
{
   return 0;
}

int
hypre_MPI_Irsend( void              *buf,
                  int                count,
                  hypre_MPI_Datatype datatype,
                  int                dest,
                  int                tag, 
                  hypre_MPI_Comm     comm,
                  hypre_MPI_Request *request )
{
   return 0;
}

int
hypre_MPI_Startall( int                count,
                    hypre_MPI_Request *array_of_requests )
{
   return 0;
}

int
hypre_MPI_Probe( int               source,
                 int               tag,
                 hypre_MPI_Comm    comm,
                 hypre_MPI_Status *status )
{
   return 0;
}

int
hypre_MPI_Iprobe( int               source,
                  int               tag,
                  hypre_MPI_Comm    comm,
                  int              *flag,
                  hypre_MPI_Status *status )
{
   return 0;
}

int
hypre_MPI_Test( hypre_MPI_Request *request,
                int               *flag,
                hypre_MPI_Status  *status )
{
   *flag = 1;
   return(0);
}

int
hypre_MPI_Testall( int                count,
                   hypre_MPI_Request *array_of_requests,
                   int               *flag,
                   hypre_MPI_Status  *array_of_statuses )
{
   *flag = 1;
   return(0);
}

int
hypre_MPI_Wait( hypre_MPI_Request *request,
                hypre_MPI_Status  *status )
{
   return(0);
}

int
hypre_MPI_Waitall( int                count,
                   hypre_MPI_Request *array_of_requests,
                   hypre_MPI_Status  *array_of_statuses )
{
   return(0);
}

int
hypre_MPI_Waitany( int                count,
                   hypre_MPI_Request *array_of_requests,
                   int               *index,
                   hypre_MPI_Status  *status )
{
   return(0);
}

int
hypre_MPI_Allreduce( void              *sendbuf,
                     void              *recvbuf,
                     int                count,
                     hypre_MPI_Datatype datatype,
                     hypre_MPI_Op       op,
                     hypre_MPI_Comm     comm )
{ 
   switch (datatype)
   {
      case hypre_MPI_INT:
      {
         int *crecvbuf = (int *)recvbuf;
         int *csendbuf = (int *)sendbuf;
         crecvbuf[0] = csendbuf[0];
      } 
      break;

      case hypre_MPI_DOUBLE:
      {
         double *crecvbuf = (double *)recvbuf;
         double *csendbuf = (double *)sendbuf;
         crecvbuf[0] = csendbuf[0];
      } 
      break;

      case hypre_MPI_CHAR:
      {
         char *crecvbuf = (char *)recvbuf;
         char *csendbuf = (char *)sendbuf;
         crecvbuf[0] = csendbuf[0];
      } 
      break;

      case hypre_MPI_BYTE:
      {
         memcpy(recvbuf, sendbuf, 1);
      } 
      break;
   }

   return 0;
}

int
hypre_MPI_Reduce( void              *sendbuf,
                  void              *recvbuf,
                  int                count,
                  hypre_MPI_Datatype datatype,
                  hypre_MPI_Op       op,
                  int                root,
                  hypre_MPI_Comm     comm )
{ 
   hypre_MPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
   return 0;
}

int
hypre_MPI_Request_free( hypre_MPI_Request *request )
{
   return 0;
}

int
hypre_MPI_Type_contiguous( int                 count,
                           hypre_MPI_Datatype  oldtype,
                           hypre_MPI_Datatype *newtype )
{
   return(0);
}

int
hypre_MPI_Type_vector( int                 count,
                       int                 blocklength,
                       int                 stride,
                       hypre_MPI_Datatype  oldtype,
                       hypre_MPI_Datatype *newtype )
{
   return(0);
}

int
hypre_MPI_Type_hvector( int                 count,
                        int                 blocklength,
                        hypre_MPI_Aint      stride,
                        hypre_MPI_Datatype  oldtype,
                        hypre_MPI_Datatype *newtype )
{
   return(0);
}

int
hypre_MPI_Type_struct( int                 count,
                       int                *array_of_blocklengths,
                       hypre_MPI_Aint     *array_of_displacements,
                       hypre_MPI_Datatype *array_of_types,
                       hypre_MPI_Datatype *newtype )
{
   return(0);
}

int
hypre_MPI_Type_commit( hypre_MPI_Datatype *datatype )
{
   return(0);
}

int
hypre_MPI_Type_free( hypre_MPI_Datatype *datatype )
{
   return(0);
}

#else

/* this is used only to eliminate compiler warnings */
int hypre_empty2;

#endif
