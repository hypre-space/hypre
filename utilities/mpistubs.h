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

#ifdef __cplusplus
}
#endif

#endif
