#ifndef COMMON_DH
#define COMMON_DH

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>


/*-----------------------------------------------------------------------
 * compile-time dependent includes from other libraries.
 * maintainer's note: this is the only place where non-Euclid
 * files are included.
 *-----------------------------------------------------------------------*/

#if defined(HYPRE_MODE)
#include "HYPRE_parcsr_mv.h"
#include "HYPRE_config.h"
#include "mpistubs.h"

#elif defined(PETSC_MODE)
#include "sles.h"
#define EUCLID_ERROR  PETSC_ERR_LIB /* see: ${PETSC_DIR}/include/petscerror.h */
#else
#define EUCLID_ERROR -1
#endif

#if defined(USING_MPI) && !defined(HYPRE_MODE) && !defined(PETSC_MODE)
#include <mpi.h>
#else
/* #include "fake_mpi.h" */
#endif

#if defined(USING_OPENMP) && !defined(HYPRE_MODE)
#include <omp.h>
#endif

/*-----------------------------------------------------------------------
 * Euclid includes
 *-----------------------------------------------------------------------*/

#include "euclid_config.h" /* contains various user-configurable settings;
                              edit this when building an interface with
                              other libraries.  
                            */  

#include "macros_dh.h" /* macros for error checking, etc */

#include "home_dh.h"  /* defines MASTER_OPTIONS_LIST, which is the absolute
                       * filename of the default options read by Parser_dh
                       * objects; home_dh.h is generated during compilation
                       * (see, for example, the src/build script)
                       */

/*----------------------------------------------------------- 
 *  Euclid classes 
 *-----------------------------------------------------------*/
typedef struct _matgenfd*           MatGenFD;
typedef struct _timer_dh*           Timer_dh;
typedef struct _parser_dh*          Parser_dh;
typedef struct _timeLog_dh*         TimeLog_dh;
typedef struct _mem_dh*             Mem_dh;
typedef struct _mat_dh*             Mat_dh;
typedef struct _vec_dh*             Vec_dh;
typedef struct _numbering_dh*       Numbering_dh;
typedef struct _hash_dh*            Hash_dh;
typedef struct _mpi_interface_dh*   Euclid_dh;

/*
typedef struct _localPerm_dh*       LocalPerm_dh;
typedef struct _procGrid_dh*        ProcGrid_dh;
typedef struct _globalPerm_dh*      GlobalPerm_dh;
typedef struct _apply_dh*           Apply_dh;
typedef struct _sortedList_dh*      SortedList_dh;
typedef struct _externalRows_dh*    ExternalRows_dh;
*/

/*---------------------------------------------------------------------
 * misc.
 *---------------------------------------------------------------------*/


#if defined(__cplusplus)
#else
typedef int bool;
#define true   1
#define false  0
#endif

/* ------------------------------------------------------------------
 * Globally scoped variables, error handling functions, etc.
 * These are all defined in /src/globalObjects.c 
 * ------------------------------------------------------------------*/
extern Parser_dh   parser_dh;  /* for setting/getting runtime options */
extern TimeLog_dh  tlog_dh;    /* internal timing  functionality */
extern Mem_dh      mem_dh;     /* memory management */
extern FILE        *logFile;
extern int         np_dh;     /* number of processors and subdomains */
extern int         myid_dh;   /* rank of this processor (and subdomain) */
extern MPI_Comm    comm_dh; 


/* Error and message handling.  These are accessed through
 * macros defined in "macros_dh.h"
 */
extern bool  errFlag_dh;
extern void  setInfo_dh(char *msg, char *function, char *file, int line);
extern void  setError_dh(char *msg, char *function, char *file, int line);
extern void  printErrorMsg(FILE *fp);
#define MSG_BUF_SIZE_DH 1024
extern char  msgBuf_dh[MSG_BUF_SIZE_DH];

/* Each processor (may) open a logfile.
 * The bools are switches for controlling the amount of informational 
 * output, and where it gets written to.  Function trace logging is only 
 * enabled when compiled with the debugging (-g) option.
 */
extern void openLogfile_dh(int argc, char *argv[]);
extern void closeLogfile_dh();
extern bool logInfoToStderr;
extern bool logInfoToFile;
extern bool logFuncsToStderr;
extern bool logFuncsToFile;
extern void Error_dhStartFunc(char *function, char *file, int line);
extern void Error_dhEndFunc(char *function);

#endif
