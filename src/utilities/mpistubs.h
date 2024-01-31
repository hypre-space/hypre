/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 *  Fake mpi stubs to generate serial codes without mpi
 *
 *****************************************************************************/

#ifndef hypre_MPISTUBS
#define hypre_MPISTUBS

#ifdef __cplusplus
extern "C" {
#endif

#ifdef HYPRE_SEQUENTIAL

/******************************************************************************
 * MPI stubs to generate serial codes without mpi
 *****************************************************************************/

/*--------------------------------------------------------------------------
 * Change all MPI names to hypre_MPI names to avoid link conflicts.
 *
 * NOTE: MPI_Comm is the only MPI symbol in the HYPRE user interface,
 * and is defined in `HYPRE_utilities.h'.
 *--------------------------------------------------------------------------*/

#define MPI_Comm            hypre_MPI_Comm
#define MPI_Group           hypre_MPI_Group
#define MPI_Request         hypre_MPI_Request
#define MPI_Datatype        hypre_MPI_Datatype
#define MPI_Status          hypre_MPI_Status
#define MPI_Op              hypre_MPI_Op
#define MPI_Aint            hypre_MPI_Aint
#define MPI_Info            hypre_MPI_Info

#define MPI_COMM_WORLD       hypre_MPI_COMM_WORLD
#define MPI_COMM_NULL        hypre_MPI_COMM_NULL
#define MPI_COMM_SELF        hypre_MPI_COMM_SELF
#define MPI_COMM_TYPE_SHARED hypre_MPI_COMM_TYPE_SHARED

#define MPI_BOTTOM          hypre_MPI_BOTTOM

#define MPI_FLOAT           hypre_MPI_FLOAT
#define MPI_DOUBLE          hypre_MPI_DOUBLE
#define MPI_LONG_DOUBLE     hypre_MPI_LONG_DOUBLE
#define MPI_INT             hypre_MPI_INT
#define MPI_LONG_LONG_INT   hypre_MPI_LONG_LONG_INT
#define MPI_CHAR            hypre_MPI_CHAR
#define MPI_LONG            hypre_MPI_LONG
#define MPI_BYTE            hypre_MPI_BYTE

#define MPI_C_FLOAT_COMPLEX hypre_MPI_COMPLEX
#define MPI_C_LONG_DOUBLE_COMPLEX hypre_MPI_COMPLEX
#define MPI_C_DOUBLE_COMPLEX hypre_MPI_COMPLEX

#define MPI_SUM             hypre_MPI_SUM
#define MPI_MIN             hypre_MPI_MIN
#define MPI_MAX             hypre_MPI_MAX
#define MPI_LOR             hypre_MPI_LOR
#define MPI_LAND            hypre_MPI_LAND
#define MPI_BOR             hypre_MPI_BOR
#define MPI_SUCCESS         hypre_MPI_SUCCESS
#define MPI_STATUSES_IGNORE hypre_MPI_STATUSES_IGNORE

#define MPI_UNDEFINED       hypre_MPI_UNDEFINED
#define MPI_REQUEST_NULL    hypre_MPI_REQUEST_NULL
#define MPI_INFO_NULL       hypre_MPI_INFO_NULL
#define MPI_ANY_SOURCE      hypre_MPI_ANY_SOURCE
#define MPI_ANY_TAG         hypre_MPI_ANY_TAG
#define MPI_SOURCE          hypre_MPI_SOURCE
#define MPI_TAG             hypre_MPI_TAG

#define MPI_Init            hypre_MPI_Init
#define MPI_Finalize        hypre_MPI_Finalize
#define MPI_Abort           hypre_MPI_Abort
#define MPI_Wtime           hypre_MPI_Wtime
#define MPI_Wtick           hypre_MPI_Wtick
#define MPI_Barrier         hypre_MPI_Barrier
#define MPI_Comm_create     hypre_MPI_Comm_create
#define MPI_Comm_dup        hypre_MPI_Comm_dup
#define MPI_Comm_f2c        hypre_MPI_Comm_f2c
#define MPI_Comm_group      hypre_MPI_Comm_group
#define MPI_Comm_size       hypre_MPI_Comm_size
#define MPI_Comm_rank       hypre_MPI_Comm_rank
#define MPI_Comm_free       hypre_MPI_Comm_free
#define MPI_Comm_split      hypre_MPI_Comm_split
#define MPI_Comm_split_type hypre_MPI_Comm_split_type
#define MPI_Group_incl      hypre_MPI_Group_incl
#define MPI_Group_free      hypre_MPI_Group_free
#define MPI_Address         hypre_MPI_Address
#define MPI_Get_count       hypre_MPI_Get_count
#define MPI_Alltoall        hypre_MPI_Alltoall
#define MPI_Allgather       hypre_MPI_Allgather
#define MPI_Allgatherv      hypre_MPI_Allgatherv
#define MPI_Gather          hypre_MPI_Gather
#define MPI_Gatherv         hypre_MPI_Gatherv
#define MPI_Scatter         hypre_MPI_Scatter
#define MPI_Scatterv        hypre_MPI_Scatterv
#define MPI_Bcast           hypre_MPI_Bcast
#define MPI_Send            hypre_MPI_Send
#define MPI_Recv            hypre_MPI_Recv
#define MPI_Isend           hypre_MPI_Isend
#define MPI_Irecv           hypre_MPI_Irecv
#define MPI_Send_init       hypre_MPI_Send_init
#define MPI_Recv_init       hypre_MPI_Recv_init
#define MPI_Irsend          hypre_MPI_Irsend
#define MPI_Startall        hypre_MPI_Startall
#define MPI_Probe           hypre_MPI_Probe
#define MPI_Iprobe          hypre_MPI_Iprobe
#define MPI_Test            hypre_MPI_Test
#define MPI_Testall         hypre_MPI_Testall
#define MPI_Wait            hypre_MPI_Wait
#define MPI_Waitall         hypre_MPI_Waitall
#define MPI_Waitany         hypre_MPI_Waitany
#define MPI_Allreduce       hypre_MPI_Allreduce
#define MPI_Reduce          hypre_MPI_Reduce
#define MPI_Scan            hypre_MPI_Scan
#define MPI_Request_free    hypre_MPI_Request_free
#define MPI_Type_contiguous hypre_MPI_Type_contiguous
#define MPI_Type_vector     hypre_MPI_Type_vector
#define MPI_Type_hvector    hypre_MPI_Type_hvector
#define MPI_Type_struct     hypre_MPI_Type_struct
#define MPI_Type_commit     hypre_MPI_Type_commit
#define MPI_Type_free       hypre_MPI_Type_free
#define MPI_Op_free         hypre_MPI_Op_free
#define MPI_Op_create       hypre_MPI_Op_create
#define MPI_User_function   hypre_MPI_User_function
#define MPI_Info_create     hypre_MPI_Info_create

/*--------------------------------------------------------------------------
 * Types, etc.
 *--------------------------------------------------------------------------*/

/* These types have associated creation and destruction routines */
typedef HYPRE_Int hypre_MPI_Comm;
typedef HYPRE_Int hypre_MPI_Group;
typedef HYPRE_Int hypre_MPI_Request;
typedef HYPRE_Int hypre_MPI_Datatype;
typedef void (hypre_MPI_User_function) (void);

typedef struct
{
   HYPRE_Int hypre_MPI_SOURCE;
   HYPRE_Int hypre_MPI_TAG;
} hypre_MPI_Status;

typedef HYPRE_Int  hypre_MPI_Op;
typedef HYPRE_Int  hypre_MPI_Aint;
typedef HYPRE_Int  hypre_MPI_Info;

#define  hypre_MPI_COMM_SELF   1
#define  hypre_MPI_COMM_WORLD  0
#define  hypre_MPI_COMM_NULL  -1

#define  hypre_MPI_COMM_TYPE_SHARED 0

#define  hypre_MPI_BOTTOM  0x0

#define  hypre_MPI_FLOAT 0
#define  hypre_MPI_DOUBLE 1
#define  hypre_MPI_LONG_DOUBLE 2
#define  hypre_MPI_INT 3
#define  hypre_MPI_CHAR 4
#define  hypre_MPI_LONG 5
#define  hypre_MPI_BYTE 6
#define  hypre_MPI_REAL 7
#define  hypre_MPI_COMPLEX 8
#define  hypre_MPI_LONG_LONG_INT 9

#define  hypre_MPI_SUM 0
#define  hypre_MPI_MIN 1
#define  hypre_MPI_MAX 2
#define  hypre_MPI_LOR 3
#define  hypre_MPI_LAND 4
#define  hypre_MPI_BOR 5
#define  hypre_MPI_SUCCESS 0
#define  hypre_MPI_STATUSES_IGNORE 0

#define  hypre_MPI_UNDEFINED -9999
#define  hypre_MPI_REQUEST_NULL  0
#define  hypre_MPI_INFO_NULL     0
#define  hypre_MPI_ANY_SOURCE    1
#define  hypre_MPI_ANY_TAG       1

#else

/******************************************************************************
 * MPI stubs to do casting of HYPRE_Int and hypre_int correctly
 *****************************************************************************/

typedef MPI_Comm     hypre_MPI_Comm;
typedef MPI_Group    hypre_MPI_Group;
typedef MPI_Request  hypre_MPI_Request;
typedef MPI_Datatype hypre_MPI_Datatype;
typedef MPI_Status   hypre_MPI_Status;
typedef MPI_Op       hypre_MPI_Op;
typedef MPI_Aint     hypre_MPI_Aint;
typedef MPI_Info     hypre_MPI_Info;
typedef MPI_User_function    hypre_MPI_User_function;

#define  hypre_MPI_COMM_WORLD         MPI_COMM_WORLD
#define  hypre_MPI_COMM_NULL          MPI_COMM_NULL
#define  hypre_MPI_BOTTOM             MPI_BOTTOM
#define  hypre_MPI_COMM_SELF          MPI_COMM_SELF
#define  hypre_MPI_COMM_TYPE_SHARED   MPI_COMM_TYPE_SHARED

#define  hypre_MPI_FLOAT   MPI_FLOAT
#define  hypre_MPI_DOUBLE  MPI_DOUBLE
#define  hypre_MPI_LONG_DOUBLE  MPI_LONG_DOUBLE
/* HYPRE_MPI_INT is defined in HYPRE_utilities.h */
#define  hypre_MPI_INT     HYPRE_MPI_INT
#define  hypre_MPI_CHAR    MPI_CHAR
#define  hypre_MPI_LONG    MPI_LONG
#define  hypre_MPI_BYTE    MPI_BYTE
/* HYPRE_MPI_REAL is defined in HYPRE_utilities.h */
#define  hypre_MPI_REAL    HYPRE_MPI_REAL
/* HYPRE_MPI_COMPLEX is defined in HYPRE_utilities.h */
#define  hypre_MPI_COMPLEX HYPRE_MPI_COMPLEX

#define  hypre_MPI_SUM MPI_SUM
#define  hypre_MPI_MIN MPI_MIN
#define  hypre_MPI_MAX MPI_MAX
#define  hypre_MPI_LOR MPI_LOR
#define  hypre_MPI_BOR MPI_BOR
#define  hypre_MPI_SUCCESS MPI_SUCCESS
#define  hypre_MPI_STATUSES_IGNORE MPI_STATUSES_IGNORE

#define  hypre_MPI_UNDEFINED       MPI_UNDEFINED
#define  hypre_MPI_REQUEST_NULL    MPI_REQUEST_NULL
#define  hypre_MPI_INFO_NULL       MPI_INFO_NULL
#define  hypre_MPI_ANY_SOURCE      MPI_ANY_SOURCE
#define  hypre_MPI_ANY_TAG         MPI_ANY_TAG
#define  hypre_MPI_SOURCE          MPI_SOURCE
#define  hypre_MPI_TAG             MPI_TAG
#define  hypre_MPI_LAND            MPI_LAND

#endif

/******************************************************************************
 * Everything below this applies to both ifdef cases above
 *****************************************************************************/

/*--------------------------------------------------------------------------
 * Prototypes
 *--------------------------------------------------------------------------*/

/* mpistubs.c */
HYPRE_Int hypre_MPI_Init( hypre_int *argc, char ***argv );
HYPRE_Int hypre_MPI_Finalize( void );
HYPRE_Int hypre_MPI_Abort( hypre_MPI_Comm comm, HYPRE_Int errorcode );
HYPRE_Real hypre_MPI_Wtime( void );
HYPRE_Real hypre_MPI_Wtick( void );
HYPRE_Int hypre_MPI_Barrier( hypre_MPI_Comm comm );
HYPRE_Int hypre_MPI_Comm_create( hypre_MPI_Comm comm, hypre_MPI_Group group,
                                 hypre_MPI_Comm *newcomm );
HYPRE_Int hypre_MPI_Comm_dup( hypre_MPI_Comm comm, hypre_MPI_Comm *newcomm );
hypre_MPI_Comm hypre_MPI_Comm_f2c( hypre_int comm );
HYPRE_Int hypre_MPI_Comm_size( hypre_MPI_Comm comm, HYPRE_Int *size );
HYPRE_Int hypre_MPI_Comm_rank( hypre_MPI_Comm comm, HYPRE_Int *rank );
HYPRE_Int hypre_MPI_Comm_free( hypre_MPI_Comm *comm );
HYPRE_Int hypre_MPI_Comm_group( hypre_MPI_Comm comm, hypre_MPI_Group *group );
HYPRE_Int hypre_MPI_Comm_split( hypre_MPI_Comm comm, HYPRE_Int n, HYPRE_Int m,
                                hypre_MPI_Comm * comms );
HYPRE_Int hypre_MPI_Group_incl( hypre_MPI_Group group, HYPRE_Int n, HYPRE_Int *ranks,
                                hypre_MPI_Group *newgroup );
HYPRE_Int hypre_MPI_Group_free( hypre_MPI_Group *group );
HYPRE_Int hypre_MPI_Address( void *location, hypre_MPI_Aint *address );
HYPRE_Int hypre_MPI_Get_count( hypre_MPI_Status *status, hypre_MPI_Datatype datatype,
                               HYPRE_Int *count );
HYPRE_Int hypre_MPI_Alltoall( void *sendbuf, HYPRE_Int sendcount, hypre_MPI_Datatype sendtype,
                              void *recvbuf, HYPRE_Int recvcount, hypre_MPI_Datatype recvtype, hypre_MPI_Comm comm );
HYPRE_Int hypre_MPI_Allgather( void *sendbuf, HYPRE_Int sendcount, hypre_MPI_Datatype sendtype,
                               void *recvbuf, HYPRE_Int recvcount, hypre_MPI_Datatype recvtype, hypre_MPI_Comm comm );
HYPRE_Int hypre_MPI_Allgatherv( void *sendbuf, HYPRE_Int sendcount, hypre_MPI_Datatype sendtype,
                                void *recvbuf, HYPRE_Int *recvcounts, HYPRE_Int *displs, hypre_MPI_Datatype recvtype,
                                hypre_MPI_Comm comm );
HYPRE_Int hypre_MPI_Gather( void *sendbuf, HYPRE_Int sendcount, hypre_MPI_Datatype sendtype,
                            void *recvbuf, HYPRE_Int recvcount, hypre_MPI_Datatype recvtype, HYPRE_Int root,
                            hypre_MPI_Comm comm );
HYPRE_Int hypre_MPI_Gatherv( void *sendbuf, HYPRE_Int sendcount, hypre_MPI_Datatype sendtype,
                             void *recvbuf, HYPRE_Int *recvcounts, HYPRE_Int *displs, hypre_MPI_Datatype recvtype,
                             HYPRE_Int root, hypre_MPI_Comm comm );
HYPRE_Int hypre_MPI_Scatter( void *sendbuf, HYPRE_Int sendcount, hypre_MPI_Datatype sendtype,
                             void *recvbuf, HYPRE_Int recvcount, hypre_MPI_Datatype recvtype, HYPRE_Int root,
                             hypre_MPI_Comm comm );
HYPRE_Int hypre_MPI_Scatterv( void *sendbuf, HYPRE_Int *sendcounts, HYPRE_Int *displs,
                              hypre_MPI_Datatype sendtype, void *recvbuf, HYPRE_Int recvcount, hypre_MPI_Datatype recvtype,
                              HYPRE_Int root, hypre_MPI_Comm comm );
HYPRE_Int hypre_MPI_Bcast( void *buffer, HYPRE_Int count, hypre_MPI_Datatype datatype,
                           HYPRE_Int root, hypre_MPI_Comm comm );
HYPRE_Int hypre_MPI_Send( void *buf, HYPRE_Int count, hypre_MPI_Datatype datatype, HYPRE_Int dest,
                          HYPRE_Int tag, hypre_MPI_Comm comm );
HYPRE_Int hypre_MPI_Recv( void *buf, HYPRE_Int count, hypre_MPI_Datatype datatype, HYPRE_Int source,
                          HYPRE_Int tag, hypre_MPI_Comm comm, hypre_MPI_Status *status );
HYPRE_Int hypre_MPI_Isend( void *buf, HYPRE_Int count, hypre_MPI_Datatype datatype, HYPRE_Int dest,
                           HYPRE_Int tag, hypre_MPI_Comm comm, hypre_MPI_Request *request );
HYPRE_Int hypre_MPI_Irecv( void *buf, HYPRE_Int count, hypre_MPI_Datatype datatype,
                           HYPRE_Int source, HYPRE_Int tag, hypre_MPI_Comm comm, hypre_MPI_Request *request );
HYPRE_Int hypre_MPI_Send_init( void *buf, HYPRE_Int count, hypre_MPI_Datatype datatype,
                               HYPRE_Int dest, HYPRE_Int tag, hypre_MPI_Comm comm, hypre_MPI_Request *request );
HYPRE_Int hypre_MPI_Recv_init( void *buf, HYPRE_Int count, hypre_MPI_Datatype datatype,
                               HYPRE_Int dest, HYPRE_Int tag, hypre_MPI_Comm comm, hypre_MPI_Request *request );
HYPRE_Int hypre_MPI_Irsend( void *buf, HYPRE_Int count, hypre_MPI_Datatype datatype, HYPRE_Int dest,
                            HYPRE_Int tag, hypre_MPI_Comm comm, hypre_MPI_Request *request );
HYPRE_Int hypre_MPI_Startall( HYPRE_Int count, hypre_MPI_Request *array_of_requests );
HYPRE_Int hypre_MPI_Probe( HYPRE_Int source, HYPRE_Int tag, hypre_MPI_Comm comm,
                           hypre_MPI_Status *status );
HYPRE_Int hypre_MPI_Iprobe( HYPRE_Int source, HYPRE_Int tag, hypre_MPI_Comm comm, HYPRE_Int *flag,
                            hypre_MPI_Status *status );
HYPRE_Int hypre_MPI_Test( hypre_MPI_Request *request, HYPRE_Int *flag, hypre_MPI_Status *status );
HYPRE_Int hypre_MPI_Testall( HYPRE_Int count, hypre_MPI_Request *array_of_requests, HYPRE_Int *flag,
                             hypre_MPI_Status *array_of_statuses );
HYPRE_Int hypre_MPI_Wait( hypre_MPI_Request *request, hypre_MPI_Status *status );
HYPRE_Int hypre_MPI_Waitall( HYPRE_Int count, hypre_MPI_Request *array_of_requests,
                             hypre_MPI_Status *array_of_statuses );
HYPRE_Int hypre_MPI_Waitany( HYPRE_Int count, hypre_MPI_Request *array_of_requests,
                             HYPRE_Int *index, hypre_MPI_Status *status );
HYPRE_Int hypre_MPI_Allreduce( void *sendbuf, void *recvbuf, HYPRE_Int count,
                               hypre_MPI_Datatype datatype, hypre_MPI_Op op, hypre_MPI_Comm comm );
HYPRE_Int hypre_MPI_Reduce( void *sendbuf, void *recvbuf, HYPRE_Int count,
                            hypre_MPI_Datatype datatype, hypre_MPI_Op op, HYPRE_Int root, hypre_MPI_Comm comm );
HYPRE_Int hypre_MPI_Scan( void *sendbuf, void *recvbuf, HYPRE_Int count,
                          hypre_MPI_Datatype datatype, hypre_MPI_Op op, hypre_MPI_Comm comm );
HYPRE_Int hypre_MPI_Request_free( hypre_MPI_Request *request );
HYPRE_Int hypre_MPI_Type_contiguous( HYPRE_Int count, hypre_MPI_Datatype oldtype,
                                     hypre_MPI_Datatype *newtype );
HYPRE_Int hypre_MPI_Type_vector( HYPRE_Int count, HYPRE_Int blocklength, HYPRE_Int stride,
                                 hypre_MPI_Datatype oldtype, hypre_MPI_Datatype *newtype );
HYPRE_Int hypre_MPI_Type_hvector( HYPRE_Int count, HYPRE_Int blocklength, hypre_MPI_Aint stride,
                                  hypre_MPI_Datatype oldtype, hypre_MPI_Datatype *newtype );
HYPRE_Int hypre_MPI_Type_struct( HYPRE_Int count, HYPRE_Int *array_of_blocklengths,
                                 hypre_MPI_Aint *array_of_displacements, hypre_MPI_Datatype *array_of_types,
                                 hypre_MPI_Datatype *newtype );
HYPRE_Int hypre_MPI_Type_commit( hypre_MPI_Datatype *datatype );
HYPRE_Int hypre_MPI_Type_free( hypre_MPI_Datatype *datatype );
HYPRE_Int hypre_MPI_Op_free( hypre_MPI_Op *op );
HYPRE_Int hypre_MPI_Op_create( hypre_MPI_User_function *function, hypre_int commute,
                               hypre_MPI_Op *op );
#if defined(HYPRE_USING_GPU) || defined(HYPRE_USING_DEVICE_OPENMP)
HYPRE_Int hypre_MPI_Comm_split_type(hypre_MPI_Comm comm, HYPRE_Int split_type, HYPRE_Int key,
                                    hypre_MPI_Info info, hypre_MPI_Comm *newcomm);
HYPRE_Int hypre_MPI_Info_create(hypre_MPI_Info *info);
HYPRE_Int hypre_MPI_Info_free( hypre_MPI_Info *info );
#endif

#ifdef __cplusplus
}
#endif

#endif
