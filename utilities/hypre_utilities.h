
#ifndef hypre_UTILITIES_HEADER
#define hypre_UTILITIES_HEADER

#ifndef HYPRE_SEQUENTIAL
#include "mpi.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

#ifdef HYPRE_SEQUENTIAL
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

#ifndef hypre_MPISTUBS
#define hypre_MPISTUBS

#ifdef __cplusplus
extern "C" {
#endif

typedef int MPI_Status ;
typedef int MPI_Request ;
typedef int MPI_Op ;
typedef int MPI_Datatype ;
typedef int MPI_Comm ;
typedef int MPI_Group ;
typedef int MPI_Aint;
/* extern MPI_Request MPI_REQUEST_NULL ; */
/* extern MPI_Datatype MPI_DOUBLE ; */
/* extern MPI_Datatype MPI_CHAR; */
/* extern MPI_Datatype MPI_INT ; */
/* extern MPI_Comm MPI_COMM_WORLD ; */
/* extern MPI_Op MPI_SUM  ; */
/* extern MPI_Op MPI_MIN  ; */
/* extern MPI_Op MPI_MAX  ; */
MPI_Request MPI_REQUEST_NULL ;
#define MPI_DOUBLE 0
#define MPI_INT 1
#define MPI_CHAR 2
MPI_Comm MPI_COMM_WORLD ;
MPI_Op MPI_SUM  ;
MPI_Op MPI_MIN  ;
MPI_Op MPI_MAX  ;
#define MPI_UNDEFINED -32766

/* defines for communication */

/* define data types and reduction operation types */
#define COM_DOUBLE 		1
#define COM_CHAR 		2
#define COM_INT 		3
#define COM_SUM 		4
#define COM_MIN 		5
#define COM_MAX 		6

/* define communication types */
#define COM_LAYER1N 		1
#define COM_LAYER1Z 		2
#define COM_LAYER1ZC 		3
#define COM_LAYER2N 		4
#define COM_LAYER2NC 		5
#define COM_LAYER2Z 		6
#define COM_LAYER_ALLN 		7
#define COM_LAYER_ALLZ 		8
#define COM_LAYER_CZ 		9
#define COM_LAYER_CN 	       10
#define COM_ACCUM 	       11
#define COM_SEND_RECV 	       12
#define COM_ALL 	       13
#define COM_ANY 	       14
#define COM_ONE 	       15
#define COM_DIRECT             16
#define COM_DIRECT_ID 	       17

/* define comm routine call flags */
#define COM_RECV 	        1	
#define COM_SEND 	        2	
#define COM_WAIT_RECV 	        3
#define COM_WAIT_SEND 	        4
#define COM_COLLECT   	        5
#define COM_SET_SIZE_DOUBLE     6
#define COM_SET_SIZE_INT        7
#define COM_NCALL_FLGS          7

/* global processor info */
extern int num_procs; /* HH_INIT 1 */

/* define return value for failure to find MPI */
#define NO_MPI 		     -999    /* can't be valid return value in mpi.h */

/*--------------------------------------------------------------------------
 * Prototypes
 *--------------------------------------------------------------------------*/

#ifdef __STDC__
# define        P(s) s
#else
# define P(s) ()
#endif


/* mpistubs.c */
int MPI_Init P((int *argc , char ***argv ));
double MPI_Wtime P((void ));
double MPI_Wtick P((void ));
int MPI_Barrier P((MPI_Comm comm ));
int MPI_Finalize P((void ));
int MPI_Comm_group P((MPI_Comm comm , MPI_Group *group ));
int MPI_Comm_dup P((MPI_Comm comm , MPI_Comm *newcomm ));
int MPI_Group_incl P((MPI_Group group , int n , int *ranks , MPI_Group *newgroup ));
int MPI_Comm_create P((MPI_Comm comm , MPI_Group group , MPI_Comm *newcomm ));
int MPI_Allgather P((void *sendbuf , int sendcount , MPI_Datatype sendtype , void *recvbuf , int recvcount , MPI_Datatype recvtype , MPI_Comm comm ));
int MPI_Allgatherv P((void *sendbuf , int sendcount , MPI_Datatype sendtype , void *recvbuf , int *recvcounts , int *displs , MPI_Datatype recvtype , MPI_Comm comm ));
int MPI_Bcast P((void *buffer , int count , MPI_Datatype datatype , int root , MPI_Comm comm ));
int MPI_Send P((void *buf , int count , MPI_Datatype datatype , int dest , int tag , MPI_Comm comm ));
int MPI_Recv P((void *buf , int count , MPI_Datatype datatype , int source , int tag , MPI_Comm comm , MPI_Status *status ));
int MPI_Isend P((void *buf , int count , MPI_Datatype datatype , int dest , int tag , MPI_Comm comm , MPI_Request *request ));
int MPI_Irecv P((void *buf , int count , MPI_Datatype datatype , int source , int tag , MPI_Comm comm , MPI_Request *request ));
int MPI_Wait P((MPI_Request *request , MPI_Status *status ));
int MPI_Waitall P((int count , MPI_Request *array_of_requests , MPI_Status *array_of_statuses ));
int MPI_Waitany P((int count , MPI_Request *array_of_requests , int *index , MPI_Status *status ));
int MPI_Comm_size P((MPI_Comm comm , int *size ));
int MPI_Comm_rank P((MPI_Comm comm , int *rank ));
int MPI_Allreduce P((void *sendbuf , void *recvbuf , int count , MPI_Datatype datatype , MPI_Op op , MPI_Comm comm ));
int MPI_Type_hvector P((int count , int blocklength , MPI_Aint stride , MPI_Datatype oldtype , MPI_Datatype *newtype ));
int MPI_Type_struct P((int count , int *array_of_blocklengths , MPI_Aint *array_of_displacements , MPI_Datatype *array_of_types , MPI_Datatype *newtype ));
int MPI_Type_free P((MPI_Datatype *datatype ));
int MPI_Type_commit P((MPI_Datatype *datatype ));

#undef P

#ifdef __cplusplus
}
#endif

#endif
#endif
/*BHEADER**********************************************************************
 * (c) 1997   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * Header file for memory management utilities
 *
 *****************************************************************************/

#ifndef hypre_MEMORY_HEADER
#define hypre_MEMORY_HEADER

#include <stdlib.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * Use "Debug Malloc Library", dmalloc
 *--------------------------------------------------------------------------*/

#ifdef HYPRE_MEMORY_DMALLOC

#define hypre_InitMemoryDebug(id)    hypre_InitMemoryDebugDML(id)
#define hypre_FinalizeMemoryDebug()  hypre_FinalizeMemoryDebugDML()

#define hypre_TAlloc(type, count) \
( (type *)hypre_MAllocDML((unsigned int)(sizeof(type) * (count)),\
                          __FILE__, __LINE__) )

#define hypre_CTAlloc(type, count) \
( (type *)hypre_CAllocDML((unsigned int)(count), (unsigned int)sizeof(type),\
                          __FILE__, __LINE__) )

#define hypre_TReAlloc(ptr, type, count) \
( (type *)hypre_ReAllocDML((char *)ptr,\
                           (unsigned int)(sizeof(type) * (count)),\
                           __FILE__, __LINE__) )

#define hypre_TFree(ptr) \
( hypre_FreeDML((char *)ptr, __FILE__, __LINE__), ptr = NULL )

/*--------------------------------------------------------------------------
 * Use standard memory routines
 *--------------------------------------------------------------------------*/

#else

#define hypre_InitMemoryDebug(id)
#define hypre_FinalizeMemoryDebug()  

#define hypre_TAlloc(type, count) \
( (type *)hypre_MAlloc((unsigned int)(sizeof(type) * (count))) )

#define hypre_CTAlloc(type, count) \
( (type *)hypre_CAlloc((unsigned int)(count), (unsigned int)sizeof(type)) )

#define hypre_TReAlloc(ptr, type, count) \
( (type *)hypre_ReAlloc((char *)ptr, (unsigned int)(sizeof(type) * (count))) )

#define hypre_TFree(ptr) \
( hypre_Free((char *)ptr), ptr = NULL )

#endif

/*--------------------------------------------------------------------------
 * Prototypes
 *--------------------------------------------------------------------------*/

#undef P
#ifdef __STDC__
# define	P(s) s
#else
# define P(s) ()
#endif


/* memory.c */
int hypre_InitMemoryDebugDML P((int id ));
int hypre_FinalizeMemoryDebugDML P((void ));
char *hypre_MAllocDML P((int size , char *file , int line ));
char *hypre_CAllocDML P((int count , int elt_size , char *file , int line ));
char *hypre_ReAllocDML P((char *ptr , int size , char *file , int line ));
void hypre_FreeDML P((char *ptr , char *file , int line ));
char *hypre_MAlloc P((int size ));
char *hypre_CAlloc P((int count , int elt_size ));
char *hypre_ReAlloc P((char *ptr , int size ));
void hypre_Free P((char *ptr ));

#undef P

#ifdef __cplusplus
}
#endif

#endif
/*BHEADER**********************************************************************
 * (c) 1997   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

/******************************************************************************
 *
 * Header file for doing timing
 *
 *****************************************************************************/

#ifndef HYPRE_TIMING_HEADER
#define HYPRE_TIMING_HEADER

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#ifndef HYPRE_SEQUENTIAL
#include "mpi.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * Prototypes for low-level timing routines
 *--------------------------------------------------------------------------*/

#ifdef __STDC__
# define        P(s) s
#else
# define P(s) ()
#endif


/* timer.c */
double time_getWallclockSeconds P((void ));
double time_getCPUSeconds P((void ));
double time_get_wallclock_seconds_ P((void ));
double time_get_cpu_seconds_ P((void ));

#undef P

/*--------------------------------------------------------------------------
 * With timing off
 *--------------------------------------------------------------------------*/

#ifndef HYPRE_TIMING

#define hypre_InitializeTiming(name) (int)(name)
#define hypre_IncFLOPCount(inc)
#define hypre_BeginTiming(i)
#define hypre_EndTiming(i)
#define hypre_PrintTiming(heading, comm)
#define hypre_FinalizeTiming(index)

/*--------------------------------------------------------------------------
 * With timing on
 *--------------------------------------------------------------------------*/

#else

/*-------------------------------------------------------
 * Global timing structure
 *-------------------------------------------------------*/

typedef struct
{
   double  *wall_time;
   double  *cpu_time;
   double  *flops;
   char   **name;
   int     *state;     /* boolean flag to allow for recursive timing */
   int     *num_regs;  /* count of how many times a name is registered */

   int      num_names;
   int      size;

   double   wall_count;
   double   CPU_count;
   double   FLOP_count;

} hypre_TimingType;

#ifdef HYPRE_TIMING_GLOBALS
hypre_TimingType *hypre_global_timing = NULL;
#else
extern hypre_TimingType *hypre_global_timing;
#endif

/*-------------------------------------------------------
 * Accessor functions
 *-------------------------------------------------------*/

#define hypre_TimingWallTime(i) (hypre_global_timing -> wall_time[(i)])
#define hypre_TimingCPUTime(i)  (hypre_global_timing -> cpu_time[(i)])
#define hypre_TimingFLOPS(i)    (hypre_global_timing -> flops[(i)])
#define hypre_TimingName(i)     (hypre_global_timing -> name[(i)])
#define hypre_TimingState(i)    (hypre_global_timing -> state[(i)])
#define hypre_TimingNumRegs(i)  (hypre_global_timing -> num_regs[(i)])
#define hypre_TimingWallCount   (hypre_global_timing -> wall_count)
#define hypre_TimingCPUCount    (hypre_global_timing -> CPU_count)
#define hypre_TimingFLOPCount   (hypre_global_timing -> FLOP_count)

/*-------------------------------------------------------
 * Prototypes
 *-------------------------------------------------------*/

#ifdef __STDC__
# define	P(s) s
#else
# define P(s) ()
#endif


/* timing.c */
int hypre_InitializeTiming P((char *name ));
void hypre_FinalizeTiming P((int time_index ));
void hypre_IncFLOPCount P((int inc ));
void hypre_BeginTiming P((int time_index ));
void hypre_EndTiming P((int time_index ));
void hypre_ClearTiming P((void ));
void hypre_PrintTiming P((char *heading , MPI_Comm comm ));

#undef P

#endif

#ifdef __cplusplus
}
#endif

#endif

#ifdef __cplusplus
}
#endif

#endif

