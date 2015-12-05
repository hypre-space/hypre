/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.6 $
 ***********************************************************************EHEADER*/




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
 * macros for error handling everyplace except in main.
 *---------------------------------------------------------------------- */

/* for future expansion: should check that "ptr" points to
   a valid memory address, if not null.
*/
#define ASSERT_DH(ptr) \
    { \
      if (ptr == NULL) { \
        hypre_sprintf(msgBuf_dh, "%s is NULL", ptr); \
        SET_V_ERROR(msgBuf_dh); \
      } \
    }


#if 0
#define CHECK_MPI_V_ERROR(errCode)  \
      { \
        if (errCode) { \
          HYPRE_Int len; \
          hypre_MPI_Error_string(errCode, msgBuf_dh, &len); \
          setError_dh(msgBuf_dh, __FUNC__, __FILE__, __LINE__); \
          return; \
        } \
      }

#define CHECK_MPI_ERROR(errCode)  \
      { \
        if (errCode) { \
          HYPRE_Int len; \
          hypre_MPI_Error_string(errCode, msgBuf_dh, &len); \
          setError_dh(msgBuf_dh, __FUNC__, __FILE__, __LINE__); \
          return(errCode); \
        } \
      }
#endif

#define CHECK_MPI_V_ERROR(errCode)  \
      { \
        if (errCode) { \
          setError_dh("MPI error!", __FUNC__, __FILE__, __LINE__); \
          return; \
        } \
      }

#define CHECK_MPI_ERROR(errCode)  \
      { \
        if (errCode) { \
          setError_dh("MPI error!", __FUNC__, __FILE__, __LINE__); \
          return(errCode); \
        } \
      }

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

/*---------------------------------------------------------------------- 
 * informational macros
 *---------------------------------------------------------------------- */

#define SET_INFO(msg)  setInfo_dh(msg, __FUNC__, __FILE__, __LINE__);

/*---------------------------------------------------------------------- 
 * macros for tracking the function call stack
 *---------------------------------------------------------------------- */
#ifdef OPTIMIZED_DH

#define START_FUNC_DH   \
          dh_StartFunc(__FUNC__, __FILE__, __LINE__, 1);  \
          {

#define END_FUNC_DH     \
          } \
          dh_EndFunc(__FUNC__, 1);

#define END_FUNC_VAL(a) \
         dh_EndFunc(__FUNC__, 1); \
         return a ; \
         }

#define START_FUNC_DH_2 /**/
#define END_FUNC_DH_2   /**/
#define END_FUNC_VAL_2(a) return a ;

#else

#define START_FUNC_DH  \
          dh_StartFunc(__FUNC__, __FILE__, __LINE__, 1); \
          if (logFuncsToStderr || logFuncsToFile)\
            Error_dhStartFunc(__FUNC__, __FILE__, __LINE__); \
          {

#define END_FUNC_DH   \
          dh_EndFunc(__FUNC__, 1); \
          if (logFuncsToStderr || logFuncsToFile) \
            Error_dhEndFunc(__FUNC__); \
          return; \
          } \

#define START_FUNC_DH_2  \
          dh_StartFunc(__FUNC__, __FILE__, __LINE__, 2); \
          if (logFuncsToStderr || logFuncsToFile)\
            Error_dhStartFunc(__FUNC__, __FILE__, __LINE__); \
          {

#define END_FUNC_DH_2   \
          dh_EndFunc(__FUNC__, 2); \
          if (logFuncsToStderr || logFuncsToFile) \
            Error_dhEndFunc(__FUNC__); \
          return; \
          } \


#define END_FUNC_VAL(retval) \
          dh_EndFunc(__FUNC__, 1); \
          if (logFuncsToStderr || logFuncsToFile) \
            Error_dhEndFunc(__FUNC__); \
          return(retval); \
          } \

#define END_FUNC_VAL_2(retval) \
          dh_EndFunc(__FUNC__, 2); \
          if (logFuncsToStderr || logFuncsToFile) \
            Error_dhEndFunc(__FUNC__); \
          return(retval); \
          } \


#endif 

#endif  /* #ifndef MACROS_DH */
