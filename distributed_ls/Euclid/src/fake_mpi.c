#ifndef USING_MPI

#include "fake_mpi.h"

MPI_Comm MPI_COMM_WORLD = 0;

int MPI_Comm_size(MPI_Comm comm, int *np) { *np = 1;  return 0; }

int MPI_Comm_rank(MPI_Comm comm, int *np) { *np = 0; return 0;}

int MPI_Waitall(int num, MPI_Request *requests, MPI_Status *statuses) 
{return 0; }

int MPI_Barrier(MPI_Comm comm) { return 0; }

int MPI_Alltoall(void *sendbuf, int sendcount, MPI_Datatype sendtype,
                void *rcvbuf,  int rcvcount,  MPI_Datatype recvtype,
                MPI_Comm comm) { return 0; }

int MPI_Request_free(MPI_Request *request) { return 0; }

int MPI_Allreduce(void *sendbuf, void *recvbuf,
                  int count, MPI_Datatype datatype,
                  MPI_Op op, MPI_Comm comm) { return 0; }

int MPI_Send_init(void *buf, int count, MPI_Datatype datatype,
                  int dest, int tag, MPI_Comm comm, 
                  MPI_Request *request) { return 0; }

extern int MPI_Recv_init(void *buf, int  count, MPI_Datatype datatype,
                         int source, int tag, MPI_Comm comm,
                         MPI_Request *request) { return 0; }

int MPI_Startall(int count, MPI_Request array_of_requests[]) { return 0; }

int MPI_Isend(void *buf, int count, MPI_Datatype datatype, int dest,
              int tag, MPI_Comm comm, MPI_Request *request) { return 0; }

int MPI_Irecv(void *buf, int count, MPI_Datatype datatype,
               int source, int tag, MPI_Comm comm, MPI_Request *request) 
{ return 0; }

int MPI_Allgather(void *sendbuf, int  sendcount, MPI_Datatype sendtype,
                  void *recvbuf, int recvcount, MPI_Datatype recvtype, 
                  MPI_Comm comm) { return 0; }


#endif
