#ifndef MACROS_DH
#define MACROS_DH

#ifndef FMAX
#define FMAX(a,b)  ((FABS(a)) > (FABS(b)) ? (FABS(a)) : (FABS(b)))
#endif

#ifndef MAX
#define MAX(a,b)   ((a) > (b) ? (a) : (b))
#endif

#ifndef MIN
#define MIN(a,b) ((a)<(b)?(a):(b))
#endif

#ifndef ABS
#define ABS(x) (((x)<0)?(-(x)):(x))
#endif

#ifndef FABS
#define FABS(a)    ((a) < 0 ? -(a) : a)
#endif

/* used in Mat_SEQ_PrintTriples, so matlab won't discard zeros (yuck!) */
#define _MATLAB_ZERO_  1e-100


/*---------------------------------------------------------------------- 
 * Memory management.  These macros work with functions in Mem_dh.c;
 * Change if you want to use some memory management and reporting schemes 
 * other than that supplied with Euclid.  This section for reference
 * only; to configure, make changes to "euclid_config.h"
 *---------------------------------------------------------------------- */

#if 0

#ifdef USE_PETSC_MALLOC
#define MALLOC_DH(s)  PetscMalloc(s)
#define FREE_DH(p)    PetscFree(p)

#else
#define MALLOC_DH(s)  mem_dh->malloc(mem_dh, (s))
#define FREE_DH(p)    mem_dh->free(mem_dh, p)
#endif


  /* The actual calls used by Mem_dh objects to allocate/free memory 
   * from the heap.
   */
#define PRIVATE_MALLOC  malloc
#define PRIVATE_FREE    free

#endif

/*---------------------------------------------------------------------- 
 * top-level error handling.  This is the default macro, for reference
 * only; the configurable macro is in "euclid_config.h"
 *---------------------------------------------------------------------- */

/*

#define ERRCHKA   \
    if (errFlag_dh) {  \
      if (logFile != NULL) {  \
        printErrorMsg(logFile);  \
        closeLogfile_dh();  \
        printErrorMsg(stderr);  \
        exit(-1); \
      } \
    } 

*/

/*---------------------------------------------------------------------- 
 * macros for error handling everyplace except in main.
 *---------------------------------------------------------------------- */
#define SET_V_ERROR(msg)  \
      { setError_dh(msg, __FUNC__, __FILE__, __LINE__); \
        return; \
      }

#define SET_ERROR(retval, msg)  \
      { setError_dh(msg, __FUNC__, __FILE__, __LINE__); \
        return (retval); \
      }

#define CHECK_V_ERROR   \
          if (errFlag_dh) { \
            setError_dh("",  __FUNC__, __FILE__, __LINE__); \
            return; \
          }

#define CHECK_ERROR(retval)  \
          if (errFlag_dh) { \
            setError_dh("",  __FUNC__, __FILE__, __LINE__); \
            return (retval); \
          }


/*---------------------------------------------------------------*/
/* next two are only used immediately after a call to a
   function from the PETSc library. (for interior, not
   top-level use)
*/

#define CHECK_P_ERROR(ierr, msg)  \
          if (ierr) { \
            sprintf(msgBuf_dh, "PETSc error= %i", ierr); \
            setError_dh(msg,  __FUNC__, __FILE__, __LINE__); \
            errFlag_dh = ierr; \
            return (ierr); \
          }

#define CHECK_PV_ERROR(ierr, msg)  \
          if (ierr) { \
            sprintf(msgBuf_dh, "PETSc error= %i", ierr); \
            setError_dh(msg,  __FUNC__, __FILE__, __LINE__); \
            errFlag_dh = ierr; \
            return; \
          }

/*---------------------------------------------------------------*/

#define CHECK_PTR(retval, ptr) \
      { if (ptr == NULL) { \
          setError_dh("malloc failed!", __FUNC__, __FILE__, __LINE__); \
          return (retval); \
        } \
      }

#define CHECK_V_PTR(ptr) \
      { if (ptr == NULL) { \
          setError_dh("malloc failed!", __FUNC__, __FILE__, __LINE__); \
          return; \
        } \
      }

/*---------------------------------------------------------------------- 
 * informational macros
 *---------------------------------------------------------------------- */

#define SET_INFO(msg)  setInfo_dh(msg, __FUNC__, __FILE__, __LINE__);

/*---------------------------------------------------------------------- 
 * macros for tracking the function call stack
 *---------------------------------------------------------------------- */
#ifdef OPTIMIZED_DH

#define START_FUNC_DH   /**/
#define END_FUNC_DH     /**/
#define END_FUNC_VAL(a) return a ;

#else

#define START_FUNC_DH  \
          if (logFuncsToStderr || logFuncsToFile)\
            Error_dhStartFunc(__FUNC__, __FILE__, __LINE__); \
          {

#define END_FUNC_DH   \
          if (logFuncsToStderr || logFuncsToFile) \
            Error_dhEndFunc(__FUNC__); \
          return; \
          } \

#define END_FUNC_VAL(retval) \
          if (logFuncsToStderr || logFuncsToFile) \
            Error_dhEndFunc(__FUNC__); \
          return(retval); \
          } \

#endif 

#endif  /* #ifndef MACROS_DH */
