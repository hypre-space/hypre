/* 
Except for MPI_Comm_size() and MPI_Comm_rank(), the functions
below do nothing --- they're included so Euclid will compile
in sequential mode, without using lots of "#ifdef USING_MPI."

Only those portions of the MPI spec needed to get Euclid
to compile are included below.
*/

#ifndef FAKE_MPI_H
#define FAKE_MPI_H

typedef int MPI_Comm;
extern MPI_Comm MPI_COMM_WORLD;

typedef int MPI_Request;
typedef int MPI_Status;
typedef int MPI_Datatype;
typedef int MPI_Op;

enum{ MPI_SUM, MPI_MAX, MPI_MIN };

#define MPI_INT     0
#define MPI_DOUBLE  1

#if defined(__cplusplus)
extern "C" {
#endif

extern int MPI_Alltoall(void *sendbuf, int sendcount, MPI_Datatype sendtype,
                       void *rcvbuf,  int rcvcount,  MPI_Datatype recvtype,
                       MPI_Comm comm);

extern int MPI_Request_free(MPI_Request *request);

extern int MPI_Allreduce(void *sendbuf, void *recvbuf,
                         int count, MPI_Datatype datatype,
                         MPI_Op op, MPI_Comm comm);

extern int MPI_Send_init(void *buf, int count, MPI_Datatype datatype,
                         int dest, int tag, MPI_Comm comm, 
                         MPI_Request *request);

extern int MPI_Startall(int count, MPI_Request array_of_requests[]);

extern int MPI_Isend(void *buf, int count, MPI_Datatype datatype, int dest,
                     int tag, MPI_Comm comm, MPI_Request *request);

extern int MPI_Irecv(void *buf, int count, MPI_Datatype datatype,
                      int source, int tag, MPI_Comm comm, MPI_Request *request);


extern int MPI_Allgather(void *sendbuf, int  sendcount, MPI_Datatype sendtype,
                         void *recvbuf, int recvcount, MPI_Datatype recvtype, 
                         MPI_Comm comm);

extern int MPI_Recv_init(void *buf, int  count, MPI_Datatype datatype,
                         int source, int tag, MPI_Comm comm,
                         MPI_Request *request);

extern int MPI_Barrier(MPI_Comm comm);
extern int MPI_Comm_size(MPI_Comm comm, int *np);
extern int MPI_Comm_rank(MPI_Comm comm, int *np);

extern int MPI_Waitall(int num, MPI_Request *requests, MPI_Status *statuses);


#if defined(__cplusplus)
}
#endif 


#endif
