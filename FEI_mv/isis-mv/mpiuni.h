/* $Id$ */

/*
   This is a special set of bindings for uni-processor use of MPI by the PETSc library.
 
   NOT ALL THE MPI CALLS ARE IMPLEMENTED CORRECTLY! Only those needed in PETSc.

   For example,
   * Does not implement send to self.
   * Does not implement attributes correctly.

   Changes for ISIS++ by Ben Allan, 1/1999:
	added c++ C wrapper
	added missing MPI_Aint, per the linux mpich value.
	added MPIUNI_DEBUG flag which echos each mpi call made that is being
		faked to return MPI_SUCCESS.
Unfortunately this addition is not yet completed.
		Sources compiled -DMPIUNI_DEBUG will chatter.
	added message output just prior to all MPI_Abort calls giving context info.
*/

#if !defined(__MPI_H)
#define __MPI_H

/* keep c++ compilers happy */
#if defined(__cplusplus)
extern "C" {
#endif

#define USING_MPIUNI

#define MPI_COMM_WORLD       1
#define MPI_COMM_SELF        MPI_COMM_WORLD
#define MPI_COMM_NULL        0
#define MPI_IDENT            0
#define MPI_SUCCESS          0
#define MPI_UNEQUAL          3
#define MPI_ANY_SOURCE     (-2)
#define MPI_KEYVAL_INVALID   0
#define MPI_ERR_UNKNOWN     18
#define MPI_ERR_INTERN      21
#define MPI_ERR_OTHER        1
#define MPI_TAG_UB           0
#define MPI_ERRORS_RETURN    0

/* In order to handle datatypes, we make them into "sizeof(raw-type)";
    this allows us to do the MPIUNI_Memcpy's easily */
#define MPI_Datatype      int
#define MPI_FLOAT         sizeof(float)
#define MPI_DOUBLE        sizeof(double)
#define MPI_CHAR          sizeof(char)
#define MPI_BYTE          sizeof(char)
#define MPI_INT           sizeof(int)
#define MPI_UNSIGNED_LONG sizeof(unsigned long)
#define MPIU_PLOGDOUBLE   sizeof(PLogDouble)
/* questionable whether the following addition needs to be long on some
 * platforms. */
#define MPI_Aint      int

/* from intel/linux mpich */
#define MPI_BOTTOM      (void *)0

#define MPI_SUM 0

#ifndef MPIUNI_extern_guard
/* MPIUNI_extern_guard is here to facilitate testing of the macros */
#define MPIUNI_extern_guard

/* External types */
typedef int MPI_Op;
typedef int    MPI_Comm;  
typedef void   *MPI_Request;
typedef void   *MPI_Group;
typedef struct {int MPI_TAG, MPI_SOURCE, MPI_ERROR;} MPI_Status;
typedef char*   MPI_Errhandler;

extern void *MPIUNI_TMP;
extern int MPIUNI_Memcpy(void*,void*,int);

/*
  Prototypes of some functions which are implemented in mpi.c
*/

extern int    MPI_Abort(MPI_Comm,int);
extern int    MPI_Attr_get(MPI_Comm comm, int keyval, void *attribute_val, int *flag);
extern int    MPI_Keyval_free(int*);
extern int    MPI_Attr_put(MPI_Comm, int, void *);
extern int    MPI_Attr_delete(MPI_Comm, int);
typedef int   (MPI_Copy_function)( MPI_Comm, int, void *, void *, void *, int *);
typedef int   (MPI_Delete_function)( MPI_Comm, int, void *, void * );
extern int    MPI_Keyval_create(MPI_Copy_function *,MPI_Delete_function *,int *,void *);
extern int    MPI_Comm_free(MPI_Comm*);
extern int    MPI_Initialized(int *);
extern int    MPI_Comm_dup(MPI_Comm,MPI_Comm *);
extern int	MPI_Finalize(void);
extern double	MPI_Wtime();

extern int 	mpiuni_msg(char *, char *,int);
#endif /* extern guard */
#define MSG(s) mpiuni_msg((s),__FILE__,__LINE__)

#ifdef MPIUNI_DEBUG
#define DMSG(s) mpiuni_msg((s),__FILE__,__LINE__)
#else
#define DMSG(s) 0
#endif

/* 
    Routines we have replace with macros that do nothing 
    Some return error codes others return success
*/
#define MPI_Send( buf, count, datatype, dest, tag, comm)  \
     (MPIUNI_TMP = (void *) (long) (buf), \
      MPIUNI_TMP = (void *) (long) (count), \
      MPIUNI_TMP = (void *) (long) (datatype), \
      MPIUNI_TMP = (void *) (long) (dest), \
      MPIUNI_TMP = (void *) (long) (tag), \
      MPIUNI_TMP = (void *) (long) (comm), \
      DMSG("MPI_Send"), \
      MPI_SUCCESS)
#define MPI_Recv( buf, count, datatype, source, tag, comm, status) \
     (MPIUNI_TMP = (void *) (long) (buf), \
      MPIUNI_TMP = (void *) (long) (count), \
      MPIUNI_TMP = (void *) (long) (datatype), \
      MPIUNI_TMP = (void *) (long) (source), \
      MPIUNI_TMP = (void *) (long) (tag), \
      MPIUNI_TMP = (void *) (long) (comm), \
      MPIUNI_TMP = (void *) (long) (status), \
      MPIUNI_TMP = (void *) (long) (MSG("MPI_Recv")), \
      MPI_Abort(MPI_COMM_WORLD,0))
#define MPI_Get_count(status,  datatype, count) \
     (MPIUNI_TMP = (void *) (long) (status), \
      MPIUNI_TMP = (void *) (long) (datatype), \
      MPIUNI_TMP = (void *) (long) (count), \
      MPIUNI_TMP = (void *) (long) (MSG("MPI_Get_count")), \
      MPI_Abort(MPI_COMM_WORLD,0))
#define MPI_Bsend( buf, count, datatype, dest, tag, comm)  \
     (MPIUNI_TMP = (void *) (long) (buf), \
      MPIUNI_TMP = (void *) (long) (count), \
      MPIUNI_TMP = (void *) (long) (datatype), \
      MPIUNI_TMP = (void *) (long) (dest), \
      MPIUNI_TMP = (void *) (long) (tag), \
      MPIUNI_TMP = (void *) (long) (comm), \
      DMSG("MPI_Bsend"), \
      MPI_SUCCESS)
#define MPI_Ssend( buf, count,  datatype, dest, tag, comm) \
     (MPIUNI_TMP = (void *) (long) (buf), \
      MPIUNI_TMP = (void *) (long) (count), \
      MPIUNI_TMP = (void *) (long) (datatype), \
      MPIUNI_TMP = (void *) (long) (dest), \
      MPIUNI_TMP = (void *) (long) (tag), \
      MPIUNI_TMP = (void *) (long) (comm), \
      DMSG("MPI_Ssend"), \
      MPI_SUCCESS)
#define MPI_Rsend( buf, count,  datatype, dest, tag, comm) \
     (MPIUNI_TMP = (void *) (long) (buf), \
      MPIUNI_TMP = (void *) (long) (count), \
      MPIUNI_TMP = (void *) (long) (datatype), \
      MPIUNI_TMP = (void *) (long) (dest), \
      MPIUNI_TMP = (void *) (long) (tag), \
      MPIUNI_TMP = (void *) (long) (comm), \
      DMSG("MPI_Rsend"), \
      MPI_SUCCESS)
#define MPI_Buffer_attach( buffer, size) \
     (MPIUNI_TMP = (void *) (long) (buffer), \
      MPIUNI_TMP = (void *) (long) (size), \
      DMSG("MPI_Buffer_attach"), \
      MPI_SUCCESS)
#define MPI_Buffer_detach( buffer, size)\
     (MPIUNI_TMP = (void *) (long) (buffer), \
      MPIUNI_TMP = (void *) (long) (size), \
      DMSG("MPI_Buffer_detach"), \
      MPI_SUCCESS)
#define MPI_Ibsend( buf, count,  datatype, dest, tag, comm, request) \
     (MPIUNI_TMP = (void *) (long) (buf), \
      MPIUNI_TMP = (void *) (long) (count), \
      MPIUNI_TMP = (void *) (long) (datatype), \
      MPIUNI_TMP = (void *) (long) (dest), \
      MPIUNI_TMP = (void *) (long) (tag), \
      MPIUNI_TMP = (void *) (long) (comm), \
      MPIUNI_TMP = (void *) (long) (request), \
      DMSG("MPI_Ibsend"), \
      MPI_SUCCESS)
#define MPI_Issend( buf, count,  datatype, dest, tag, comm, request) \
     (MPIUNI_TMP = (void *) (long) (buf), \
      MPIUNI_TMP = (void *) (long) (count), \
      MPIUNI_TMP = (void *) (long) (datatype), \
      MPIUNI_TMP = (void *) (long) (dest), \
      MPIUNI_TMP = (void *) (long) (tag), \
      MPIUNI_TMP = (void *) (long) (comm), \
      MPIUNI_TMP = (void *) (long) (request), \
      DMSG("MPI_Issend"), \
      MPI_SUCCESS)
#define MPI_Irsend( buf, count,  datatype, dest, tag, comm, request) \
     (MPIUNI_TMP = (void *) (long) (buf), \
      MPIUNI_TMP = (void *) (long) (count), \
      MPIUNI_TMP = (void *) (long) (datatype), \
      MPIUNI_TMP = (void *) (long) (dest), \
      MPIUNI_TMP = (void *) (long) (tag), \
      MPIUNI_TMP = (void *) (long) (comm), \
      MPIUNI_TMP = (void *) (long) (request), \
      DMSG("MPI_Irsend"), \
      MPI_SUCCESS)
#define MPI_Irecv( buf, count,  datatype, source, tag, comm, request) \
     (MPIUNI_TMP = (void *) (long) (buf), \
      MPIUNI_TMP = (void *) (long) (count), \
      MPIUNI_TMP = (void *) (long) (datatype), \
      MPIUNI_TMP = (void *) (long) (source), \
      MPIUNI_TMP = (void *) (long) (tag), \
      MPIUNI_TMP = (void *) (long) (comm), \
      MPIUNI_TMP = (void *) (long) (request), \
      MPIUNI_TMP = (void *) (long) (MSG("MPI_Irecv")), \
      MPI_Abort(MPI_COMM_WORLD,0))
#define MPI_Isend( buf, count,  datatype, dest, tag, comm, request) \
     (MPIUNI_TMP = (void *) (long) (buf), \
      MPIUNI_TMP = (void *) (long) (count), \
      MPIUNI_TMP = (void *) (long) (datatype), \
      MPIUNI_TMP = (void *) (long) (dest), \
      MPIUNI_TMP = (void *) (long) (tag), \
      MPIUNI_TMP = (void *) (long) (comm), \
      MPIUNI_TMP = (void *) (long) (request), \
      MPIUNI_TMP = (void *) (long) (MSG("MPI_Isend")), \
      MPI_Abort(MPI_COMM_WORLD,0))
#define MPI_Wait(request, status) \
     (MPIUNI_TMP = (void *) (long) (request), \
      MPIUNI_TMP = (void *) (long) (status), \
      DMSG("MPI_Wait"), \
      MPI_SUCCESS)
#define MPI_Test(request, flag, status) \
     (MPIUNI_TMP = (void *) (long) (request), \
      MPIUNI_TMP = (void *) (long) (status), \
      DMSG("MPI_Test"), \
      (*(flag) = 0), \
      MPI_SUCCESS)
#define MPI_Request_free(request) \
     (MPIUNI_TMP = (void *) (long) (request), \
      DMSG("MPI_Request_free"), \
      MPI_SUCCESS)
#define MPI_Waitany(a, b, c, d) \
     (MPIUNI_TMP = (void *) (long) (a), \
      MPIUNI_TMP = (void *) (long) (b), \
      MPIUNI_TMP = (void *) (long) (c), \
      MPIUNI_TMP = (void *) (long) (d), \
      DMSG("MPI_Waitany"), \
      MPI_SUCCESS)
#define MPI_Testany(a, b, c, d, e) \
     (MPIUNI_TMP = (void *) (long) (a), \
      MPIUNI_TMP = (void *) (long) (b), \
      MPIUNI_TMP = (void *) (long) (c), \
      MPIUNI_TMP = (void *) (long) (d), \
      MPIUNI_TMP = (void *) (long) (e), \
      DMSG("MPI_Testany"), \
      MPI_SUCCESS)
#define MPI_Waitall(count, array_of_requests, array_of_statuses) \
     (MPIUNI_TMP = (void *) (long) (count), \
      MPIUNI_TMP = (void *) (long) (array_of_requests), \
      MPIUNI_TMP = (void *) (long) (array_of_statuses), \
      DMSG("MPI_Waitall"), \
      MPI_SUCCESS)
#define MPI_Testall(count, array_of_requests, flag, array_of_statuses) \
     (MPIUNI_TMP = (void *) (long) (count), \
      MPIUNI_TMP = (void *) (long) (array_of_requests), \
      MPIUNI_TMP = (void *) (long) (flag), \
      MPIUNI_TMP = (void *) (long) (array_of_statuses), \
      DMSG("MPI_Testall"), \
      MPI_SUCCESS)
#define MPI_Waitsome(incount, array_of_requests, outcount, \
                     array_of_indices, array_of_statuses) \
     (MPIUNI_TMP = (void *) (long) (incount), \
      MPIUNI_TMP = (void *) (long) (array_of_requests), \
      MPIUNI_TMP = (void *) (long) (outcount), \
      MPIUNI_TMP = (void *) (long) (array_of_indices), \
      MPIUNI_TMP = (void *) (long) (array_of_statuses), \
      DMSG("MPI_Waitsome"), \
      MPI_SUCCESS)
#define MPI_Comm_group(comm, group) \
     (MPIUNI_TMP = (void *) (long) (comm), \
      MPIUNI_TMP = (void *) (long) (group), \
      DMSG("MPI_Comm_group"), \
      MPI_SUCCESS)
#define MPI_Group_incl(group, n, ranks, newgroup) \
     (MPIUNI_TMP = (void *) (long) (group), \
      MPIUNI_TMP = (void *) (long) (n), \
      MPIUNI_TMP = (void *) (long) (ranks), \
      MPIUNI_TMP = (void *) (long) (newgroup), \
      DMSG("MPI_Group_incl"), \
      MPI_SUCCESS)
#define MPI_Testsome(incount, array_of_requests, outcount, array_of_indices, array_of_statuses) \
      (DMSG("MPI_Testsome"), \
      MPI_SUCCESS)
#define MPI_Iprobe(source, tag, comm, flag, status)  \
      ((*(flag)=0), DMSG("MPI_Iprobe"), MPI_SUCCESS)
#define MPI_Probe(source, tag, comm, status) (DMSG("MPI_Probe"),MPI_SUCCESS)
#define MPI_Cancel(request) \
	(MPIUNI_TMP = (void *) (long) (request), \
    DMSG("MPI_Cancel"), \
    MPI_SUCCESS)
#define MPI_Test_cancelled(status, flag)  \
     (*(flag)=0, MPI_SUCCESS)
#define MPI_Send_init( buf, count,  datatype, dest, tag, comm, request) \
     (MPIUNI_TMP = (void *) (long) (buf), \
     MPIUNI_TMP = (void *) (long) (count), \
     MPIUNI_TMP = (void *) (long) (datatype), \
     MPIUNI_TMP = (void *) (long) (dest), \
     MPIUNI_TMP = (void *) (long) (tag), \
     MPIUNI_TMP = (void *) (long) (comm), \
     MPIUNI_TMP = (void *) (long) (request), \
    DMSG("MPI_Send_init"), \
     MPI_SUCCESS)
#define MPI_Ssend_init( buf, count,  datatype, dest, tag, comm, request) \
     (MPIUNI_TMP = (void *) (long) (buf), \
     MPIUNI_TMP = (void *) (long) (count), \
     MPIUNI_TMP = (void *) (long) (datatype), \
     MPIUNI_TMP = (void *) (long) (dest), \
     MPIUNI_TMP = (void *) (long) (tag), \
     MPIUNI_TMP = (void *) (long) (comm), \
     MPIUNI_TMP = (void *) (long) (request), \
    DMSG("MPI_Ssend_init"), \
     MPI_SUCCESS)
#define MPI_Bsend_init( buf, count,  datatype, dest, tag, comm, request) \
     (MPIUNI_TMP = (void *) (long) (buf), \
     MPIUNI_TMP = (void *) (long) (count), \
     MPIUNI_TMP = (void *) (long) (datatype), \
     MPIUNI_TMP = (void *) (long) (dest), \
     MPIUNI_TMP = (void *) (long) (tag), \
     MPIUNI_TMP = (void *) (long) (comm), \
     MPIUNI_TMP = (void *) (long) (request), \
    DMSG("MPI_Bsend_init"), \
     MPI_SUCCESS)
#define MPI_Rsend_init( buf, count,  datatype, dest, tag, comm, request) \
     (MPIUNI_TMP = (void *) (long) (buf), \
     MPIUNI_TMP = (void *) (long) (count), \
     MPIUNI_TMP = (void *) (long) (datatype), \
     MPIUNI_TMP = (void *) (long) (dest), \
     MPIUNI_TMP = (void *) (long) (tag), \
     MPIUNI_TMP = (void *) (long) (comm), \
     MPIUNI_TMP = (void *) (long) (request), \
    DMSG("MPI_Rsend_init"), \
     MPI_SUCCESS)
#define MPI_Recv_init( buf, count,  datatype, source, tag, comm, request) \
     (MPIUNI_TMP = (void *) (long) (buf), \
     MPIUNI_TMP = (void *) (long) (count), \
     MPIUNI_TMP = (void *) (long) (datatype), \
     MPIUNI_TMP = (void *) (long) (source), \
     MPIUNI_TMP = (void *) (long) (tag), \
     MPIUNI_TMP = (void *) (long) (comm), \
     MPIUNI_TMP = (void *) (long) (request), \
    DMSG("MPI_Recv_init"), \
     MPI_SUCCESS)
#define MPI_Start(request) \
     (MPIUNI_TMP = (void *) (long) (request), \
    DMSG("MPI_Start"), \
     MPI_SUCCESS)
#define MPI_Startall(count, array_of_requests) \
     (MPIUNI_TMP = (void *) (long) (count), \
     MPIUNI_TMP = (void *) (long) (array_of_requests), \
    DMSG("MPI_Startall"), \
     MPI_SUCCESS)
     /* Need to determine sizeof "sendtype" */
#define MPI_Sendrecv(sendbuf, sendcount,  sendtype, \
     dest, sendtag, recvbuf, recvcount, \
     recvtype, source, recvtag, \
     comm, status) \
     MPIUNI_Memcpy( recvbuf, sendbuf, (sendcount) * (sendtype) )
#define MPI_Sendrecv_replace( buf, count,  datatype, dest, sendtag, \
     source, recvtag, comm, status) MPI_SUCCESS
#define MPI_Type_contiguous(count,  oldtype, newtype) \
     ( *(newtype) = (count)*(oldtype), \
    DMSG("MPI_Type_contiguous"), \
	MPI_SUCCESS )
#define MPI_Type_vector(count, blocklength, stride, oldtype,  newtype) \
     (DMSG("MPI_Type_vector"), MPI_SUCCESS)
#define MPI_Type_hvector(count, blocklength, stride, oldtype,  newtype) \
     (DMSG("MPI_Type_hvector"), MPI_SUCCESS)
#define MPI_Type_indexed(count, array_of_blocklengths, \
     array_of_displacements,  oldtype, newtype) \
     (DMSG("MPI_Type_indexed"), MPI_SUCCESS)
#define MPI_Type_hindexed(count, array_of_blocklengths, \
     array_of_displacements,  oldtype, newtype) \
     (DMSG("MPI_Type_hindexed"), MPI_SUCCESS)
#define MPI_Type_struct(count, array_of_blocklengths, \
     array_of_displacements, array_of_types,  newtype) \
     (DMSG("MPI_Type_struct"), MPI_SUCCESS)
#define MPI_Address( location, address) \
     (*(address) = (long)(char *)(location), \
	DMSG("MPI_Address"), \
	MPI_SUCCESS)
#define MPI_Type_extent( datatype, extent) \
     (MSG("MPI_Type_extent"), \
     MPI_Abort(MPI_COMM_WORLD,0))
#define MPI_Type_size( datatype, size) \
     (MSG("MPI_Type_size"), \
     MPI_Abort(MPI_COMM_WORLD,0))
#define MPI_Type_lb( datatype, displacement) \
     (MSG("MPI_Type_lb"), \
     MPI_Abort(MPI_COMM_WORLD,0))
#define MPI_Type_ub( datatype, displacement) \
     (MSG("MPI_Type_ub"), \
     MPI_Abort(MPI_COMM_WORLD,0))
#define MPI_Type_commit( datatype) \
	(DMSG("MPI_Type_commit"), MPI_SUCCESS)
#define MPI_Type_free( datatype) \
	(DMSG("MPI_Type_free"), MPI_SUCCESS)
#define MPI_Get_elements(status,  datatype, count) \
     (MSG("MPI_Get_elements"), \
     MPI_Abort(MPI_COMM_WORLD,0))
#define MPI_Pack( inbuf, incount,  datatype, outbuf, \
     outsize, position,  comm) \
     (MSG("MPI_Pack"), \
     MPI_Abort(MPI_COMM_WORLD,0))
#define MPI_Unpack( inbuf, insize, position, outbuf, \
     outcount,  datatype, comm) \
     (MSG("MPI_Unpack"), \
     MPI_Abort(MPI_COMM_WORLD,0))
#define MPI_Pack_size(incount,  datatype, comm, size) \
     (MSG("MPI_Pack_size"), \
     MPI_Abort(MPI_COMM_WORLD,0))
#define MPI_Barrier(comm ) \
     (MPIUNI_TMP = (void *) (long) (comm), \
	DMSG("MPI_Barrier"), \
     MPI_SUCCESS)
#define MPI_Bcast( buffer, count, datatype, root, comm ) \
     (MPIUNI_TMP = (void *) (long) (buffer), \
     MPIUNI_TMP = (void *) (long) (count), \
     MPIUNI_TMP = (void *) (long) (datatype), \
     MPIUNI_TMP = (void *) (long) (comm), \
	DMSG("MPI_Bcast"), \
     MPI_SUCCESS)
#define MPI_Gather( sendbuf, sendcount,  sendtype, \
     recvbuf, recvcount,  recvtype, \
     root, comm) \
     (MPIUNI_TMP = (void *) (long) (recvcount), \
     MPIUNI_TMP = (void *) (long) (root), \
     MPIUNI_TMP = (void *) (long) (recvtype), \
     MPIUNI_TMP = (void *) (long) (comm), \
     MPIUNI_Memcpy(recvbuf,sendbuf,(sendcount)* (sendtype)), \
     MPI_SUCCESS)
#define MPI_Gatherv( sendbuf, sendcount,  sendtype, \
     recvbuf, recvcounts, displs, \
     recvtype, root, comm) \
     (MPIUNI_TMP = (void *) (long) (recvcounts), \
     MPIUNI_TMP = (void *) (long) (displs), \
     MPIUNI_TMP = (void *) (long) (recvtype), \
     MPIUNI_TMP = (void *) (long) (root), \
     MPIUNI_TMP = (void *) (long) (comm), \
     MPIUNI_Memcpy(recvbuf,sendbuf,(sendcount)* (sendtype)), \
     MPI_SUCCESS)
#define MPI_Scatter( sendbuf, sendcount,  sendtype, \
     recvbuf, recvcount,  recvtype, \
     root, comm) \
     (MPIUNI_TMP = (void *) (long) (sendbuf), \
     MPIUNI_TMP = (void *) (long) (sendcount), \
     MPIUNI_TMP = (void *) (long) (sendtype), \
     MPIUNI_TMP = (void *) (long) (recvbuf), \
     MPIUNI_TMP = (void *) (long) (recvcount), \
     MPIUNI_TMP = (void *) (long) (recvtype), \
     MPIUNI_TMP = (void *) (long) (root), \
     MPIUNI_TMP = (void *) (long) (comm), \
     MSG("MPI_Scatter"), \
     MPI_Abort(MPI_COMM_WORLD,0))
#define MPI_Scatterv( sendbuf, sendcounts, displs, \
     sendtype,  recvbuf, recvcount, \
     recvtype, root, comm) \
     (MPIUNI_TMP = (void *) (long) (sendbuf), \
     MPIUNI_TMP = (void *) (long) (sendcounts), \
     MPIUNI_TMP = (void *) (long) (displs), \
     MPIUNI_TMP = (void *) (long) (sendtype), \
     MPIUNI_TMP = (void *) (long) (recvbuf), \
     MPIUNI_TMP = (void *) (long) (recvcount), \
     MPIUNI_TMP = (void *) (long) (recvtype), \
     MPIUNI_TMP = (void *) (long) (root), \
     MPIUNI_TMP = (void *) (long) (comm), \
     MSG("MPI_Scatterv"), \
     MPI_Abort(MPI_COMM_WORLD,0))
#define MPI_Allgather( sendbuf, sendcount,  sendtype, \
     recvbuf, recvcount,  recvtype, comm) \
     (MPIUNI_TMP = (void *) (long) (recvcount), \
     MPIUNI_TMP = (void *) (long) (recvtype), \
     MPIUNI_TMP = (void *) (long) (comm), \
     MPIUNI_Memcpy(recvbuf,sendbuf,(sendcount)* (sendtype)), \
     MPI_SUCCESS)
#define MPI_Allgatherv( sendbuf, sendcount,  sendtype, \
     recvbuf, recvcounts, displs, recvtype, comm) \
     (MPIUNI_TMP = (void *) (long) (recvcounts), \
     MPIUNI_TMP = (void *) (long) (displs), \
     MPIUNI_TMP = (void *) (long) (recvtype), \
     MPIUNI_TMP = (void *) (long) (comm), \
     MPIUNI_Memcpy(recvbuf,sendbuf,(sendcount)* (sendtype)), \
     MPI_SUCCESS)
#define MPI_Alltoall( sendbuf, sendcount,  sendtype, \
     recvbuf, recvcount,  recvtype, \
     comm) \
     (MSG("MPI_Alltoall"), \
     MPI_Abort(MPI_COMM_WORLD,0))
#define MPI_Alltoallv( sendbuf, sendcounts, sdispls, \
     sendtype,  recvbuf, recvcounts, \
     rdispls,  recvtype, comm) \
     (MSG("MPI_Alltoallv"), \
     MPI_Abort(MPI_COMM_WORLD,0))
#define MPI_Reduce( sendbuf,  recvbuf, count, \
     datatype, op, root, comm) \
     (MPIUNI_Memcpy(recvbuf,sendbuf,(count)*( datatype)), \
     MPIUNI_TMP = (void *) (long) (comm), MPI_SUCCESS)
#define MPI_Op_create(function, commute, op) MPI_SUCCESS
#define MPI_Op_free( op) MPI_SUCCESS
#define MPI_Allreduce( sendbuf,  recvbuf, count, datatype, op, comm) \
     (MPIUNI_Memcpy( recvbuf, sendbuf, (count)*(datatype)), \
     MPIUNI_TMP = (void *) (long) (comm), MPI_SUCCESS)
#define MPI_Scan( sendbuf,  recvbuf, count, datatype, op, comm) \
     (MPIUNI_Memcpy( recvbuf, sendbuf, (count)*(datatype)), \
     MPIUNI_TMP = (void *) (long) (comm), MPI_SUCCESS)
#define MPI_Reduce_scatter( sendbuf,  recvbuf, recvcounts, \
     datatype, op, comm) \
     (MSG("MPI_Reduce_scatter"), \
     MPI_Abort(MPI_COMM_WORLD,0))
#define MPI_Group_size(group, size) (*(size)=1,MPI_SUCCESS)
#define MPI_Group_rank(group, rank) (*(rank)=0,MPI_SUCCESS)
#define MPI_Group_translate_ranks (group1, n, ranks1, \
     group2, ranks2) MPI_Abort(MPI_COMM_WORLD,0)
#define MPI_Group_compare(group1, group2, result) \
     (*(result)=1,DMSG("MPI_Group_compare"),MPI_SUCCESS)
#define MPI_Group_union(group1, group2, newgroup) \
	(DMSG("MPI_Group_union"), MPI_SUCCESS)
#define MPI_Group_intersection(group1, group2, newgroup) \
	(DMSG("MPI_Group_intersection"), MPI_SUCCESS)
#define MPI_Group_difference(group1, group2, newgroup) \
	(DMSG("MPI_Group_difference"), MPI_SUCCESS)
#define MPI_Group_excl(group, n, ranks, newgroup) \
	(DMSG("MPI_Group_excl"), MPI_SUCCESS)
#define MPI_Group_range_incl(group, n, ranges,newgroup) \
	(DMSG("MPI_Group_range_incl"), MPI_SUCCESS)
#define MPI_Group_range_excl(group, n, ranges, newgroup) \
	(DMSG("MPI_Group_range_excl"), MPI_SUCCESS)
#define MPI_Group_free(group) \
     (MPIUNI_TMP = (void *) (long) (group), \
	DMSG("MPI_Group_free"), \
     MPI_SUCCESS)
#define MPI_Comm_size(comm, size) \
     (MPIUNI_TMP = (void *) (long) (comm), \
     *(size)=1, \
	DMSG("MPI_Comm_size = 1"), \
     MPI_SUCCESS)
#define MPI_Comm_rank(comm, rank) \
     (MPIUNI_TMP = (void *) (long) (comm), \
     *(rank)=0, \
	DMSG("MPI_Comm_rank = 0"), \
     MPI_SUCCESS)
#define MPI_Comm_compare(comm1, comm2, result) \
     (MPIUNI_TMP = (void *) (long) (comm1), \
     MPIUNI_TMP = (void *) (long) (comm2), \
     *(result)=MPI_IDENT, \
	DMSG("MPI_Comm_compare"), \
     MPI_SUCCESS )
#define MPI_Comm_create(comm, group, newcomm)  \
     (*(newcomm) =  (comm), \
     MPIUNI_TMP = (void *) (long) (group), \
	DMSG("MPI_Comm_create"), \
     MPI_SUCCESS )
#define MPI_Comm_split(comm, color, key, newcomm) MPI_SUCCESS
#define MPI_Comm_test_inter(comm, flag) (*(flag)=1,MPI_SUCCESS)
#define MPI_Comm_remote_size(comm, size) (*(size)=1,MPI_SUCCESS)
#define MPI_Comm_remote_group(comm, group) MPI_SUCCESS
#define MPI_Intercomm_create(local_comm, local_leader, peer_comm, \
     remote_leader, tag, newintercomm) MPI_SUCCESS
#define MPI_Intercomm_merge(intercomm, high, newintracomm) MPI_SUCCESS

#define MPI_Topo_test(comm, status) MPI_SUCCESS
#define MPI_Cart_create(comm_old, ndims, dims, periods,\
     reorder, comm_cart) MPI_SUCCESS
#define MPI_Dims_create(nnodes, ndims, dims) MPI_Abort(MPI_COMM_WORLD,0)
#define MPI_Graph_create(comm, a, b, c, d, e) MPI_SUCCESS
#define MPI_Graphdims_Get(comm, nnodes, nedges) MPI_Abort(MPI_COMM_WORLD,0)
#define MPI_Graph_get(comm, a, b, c, d) MPI_Abort(MPI_COMM_WORLD,0)
#define MPI_Cartdim_get(comm, ndims) MPI_Abort(MPI_COMM_WORLD,0)
#define MPI_Cart_get(comm, maxdims, dims, periods, coords) \
     MPI_Abort(MPI_COMM_WORLD,0)
#define MPI_Cart_rank(comm, coords, rank) MPI_Abort(MPI_COMM_WORLD,0)
#define MPI_Cart_coords(comm, rank, maxdims, coords) \
     MPI_Abort(MPI_COMM_WORLD,0)
#define MPI_Graph_neighbors_count(comm, rank, nneighbors) \
     MPI_Abort(MPI_COMM_WORLD,0)
#define MPI_Graph_neighbors(comm, rank, maxneighbors,neighbors) \
     MPI_Abort(MPI_COMM_WORLD,0)
#define MPI_Cart_shift(comm, direction, disp, rank_source, rank_dest) \
     MPI_Abort(MPI_COMM_WORLD,0)
#define MPI_Cart_sub(comm, remain_dims, newcomm) MPI_Abort(MPI_COMM_WORLD,0)
#define MPI_Cart_map(comm, ndims, dims, periods, newrank) \
     MPI_Abort(MPI_COMM_WORLD,0)
#define MPI_Graph_map(comm, a, b, c, d) MPI_Abort(MPI_COMM_WORLD,0)
#define MPI_Get_processor_name(name, result_len) \
     (MPIUNI_Memcpy(name,"localhost",9*sizeof(char)), name[10] = 0, *(result_len) = 10)
#define MPI_Errhandler_create(function, errhandler) \
     (MPIUNI_TMP = (void *) (long) (errhandler), \
     MPI_SUCCESS)
#define MPI_Errhandler_set(comm, errhandler) \
     (MPIUNI_TMP = (void *) (long) (comm), \
     MPIUNI_TMP = (void *) (long) (errhandler), \
     MPI_SUCCESS)
#define MPI_Errhandler_get(comm, errhandler) MPI_SUCCESS
#define MPI_Errhandler_free(errhandler) MPI_SUCCESS
#define MPI_Error_string(errorcode, string, result_len) MPI_SUCCESS
#define MPI_Error_class(errorcode, errorclass) MPI_SUCCESS
#define MPI_Wtick() (DMSG("MPI_Wtick bogus"),1.0)
#define MPI_Init(argc, argv) \
	(DMSG("MPI_Init"), \
	MPI_SUCCESS)
#define MPI_Pcontrol(level) MPI_SUCCESS

#define MPI_NULL_COPY_FN   0
#define MPI_NULL_DELETE_FN 0


#if defined(__cplusplus)
}
#endif

#endif

