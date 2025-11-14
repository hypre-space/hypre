/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/
 
 /******************************************************************************
 *
 * hypre utilities mixed-precision interface
 *
 *****************************************************************************/
 
#include "_hypre_utilities.h"
 
#if defined(HYPRE_MIXED_PRECISION)

/*--------------------------------------------------------------------------*
* hypre_RealArrayCopyHost_mp: copy n array contents from x to y.
* Assumes arrays x and y are both on host memory.
*--------------------------------------------------------------------------*/
HYPRE_Int
hypre_RealArrayCopyHost_mp(HYPRE_Precision precision_x, void *x, 
		       HYPRE_Precision precision_y, void *y, HYPRE_Int n)
{
   HYPRE_Int      i;

   /* Mixed-precision copy of data */
   switch (precision_x)
   {
      case HYPRE_REAL_SINGLE:
         switch (precision_y)
         {
            case HYPRE_REAL_DOUBLE:
#ifdef HYPRE_USING_OPENMP
               #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
               for (i = 0; i < n; i++)
               {
                  ((hypre_double *)y)[i] = (hypre_double)((hypre_float *)x)[i];
               }
               break;
            case HYPRE_REAL_LONGDOUBLE:
#ifdef HYPRE_USING_OPENMP
               #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
               for (i = 0; i < n; i++)
               {
                  ((hypre_long_double *)y)[i] = (hypre_long_double)((hypre_float *)x)[i];
               }
               break;
            default:
               break;
         }
         break;
      case HYPRE_REAL_DOUBLE:
         switch (precision_y)
         {
            case HYPRE_REAL_SINGLE:
#ifdef HYPRE_USING_OPENMP
               #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
               for (i = 0; i < n; i++)
               {
                  ((hypre_float *)y)[i] = (hypre_float)((hypre_double *)x)[i];
               }
               break;
            case HYPRE_REAL_LONGDOUBLE:
#ifdef HYPRE_USING_OPENMP
               #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
               for (i = 0; i < n; i++)
               {
                  ((hypre_long_double *)y)[i] = (hypre_long_double)((hypre_double *)x)[i];
               }
               break;
            default:
               break;
         }
         break;
      case HYPRE_REAL_LONGDOUBLE:
         switch (precision_y)
         {
            case HYPRE_REAL_SINGLE:
#ifdef HYPRE_USING_OPENMP
               #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
               for (i = 0; i < n; i++)
               {
                  ((hypre_float *)y)[i] = (hypre_float)((hypre_long_double *)x)[i];
               }
               break;
            case HYPRE_REAL_DOUBLE:
#ifdef HYPRE_USING_OPENMP
               #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
               for (i = 0; i < n; i++)
               {
                  ((hypre_double *)y)[i] = (hypre_double)((hypre_long_double *)x)[i];
               }
               break;
            default:
               break;
         }
         break;
      default:
         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Error: Undefined precision type for array Copy!\n");
         break;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------*
* hypre_RealArrayCopy_mp: copy n array contents from x to y.
* Arrays x and y need not have the same memory location.
*--------------------------------------------------------------------------*/
HYPRE_Int
hypre_RealArrayCopy_mp(HYPRE_Precision precision_x, void *x, HYPRE_MemoryLocation location_x, 
		       HYPRE_Precision precision_y, void *y, HYPRE_MemoryLocation location_y, HYPRE_Int n)
{
#ifdef HYPRE_PROFILE
    hypre_profile_times[HYPRE_TIMER_ID_BLAS1] -= hypre_MPI_Wtime();
#endif

   hypre_GpuProfilingPushRange("RealArrayCopy");

   HYPRE_Int      i;
   HYPRE_Int	  nbytes;
   size_t	  sizeof_x;

   /* tmp pointer for data copy */
   void               *xp = NULL;

   /* get sizeof x data */
   sizeof_x = hypre_GetSizeOfReal_pre(precision_x);
   
   nbytes = n * sizeof_x;
   /* Call standard memory copy if precisions match. */
   if (precision_x == precision_y)
   {
      hypre_Memcpy(y, x, nbytes, location_y, location_x);

      return hypre_error_flag;
   }

   /* Check memory location */
   if(location_x != location_y)
   {
      /* Allocate memory and copy x to y's memory location */
      xp = hypre_CAlloc(n, sizeof_x, location_y);
      hypre_Memcpy(xp, x, nbytes, location_y, location_x);
   }
   else
   {
      xp = x;
   }

#if defined(HYPRE_USING_GPU) || defined(HYPRE_USING_DEVICE_OPENMP)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1(location_y);

   if (exec == HYPRE_EXEC_DEVICE)
   {
      hypre_RealArrayCopyDevice_mp(precision_x, xp, precision_y, y, n);
   }
   else
#endif
   {
      hypre_RealArrayCopyHost_mp(precision_x, xp, precision_y, y, n);
   }
   /* free xp if allocated */
   if(location_x != location_y)
   {
      hypre_TFree(xp, location_y);
   }
#ifdef HYPRE_PROFILE
    hypre_profile_times[HYPRE_TIMER_ID_BLAS1] += hypre_MPI_Wtime();
#endif
    hypre_GpuProfilingPopRange();

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------*
* hypre_RealArrayClone_mp: Clone array x.
*--------------------------------------------------------------------------*/
void *
hypre_RealArrayClone_mp(HYPRE_Precision precision_x, void *x, HYPRE_MemoryLocation location_x, HYPRE_Precision new_precision, HYPRE_MemoryLocation new_location, HYPRE_Int n)
{
   /* cloned data */
   void           *y = NULL;
   size_t	  sizeof_y;

   /* get sizeof new_precision data */
   sizeof_y = hypre_GetSizeOfReal_pre(new_precision); 
   /* Allocate memory for cloned data */
   y = hypre_CAlloc(n, sizeof_y, new_location);

   /* Copy from x to y */
   hypre_RealArrayCopy_mp(precision_x, x, location_x, 
		          new_precision, y, new_location, n);
   return y;
}

/*--------------------------------------------------------------------------*
* hypre_RealArrayAxpynHost_mp: Axpy on n array contents into y.
* Assumes arrays x and y are both on host memory.
*--------------------------------------------------------------------------*/
HYPRE_Int
hypre_RealArrayAxpynHost_mp(HYPRE_Precision precision_x, hypre_long_double alpha, void *x, 
		       HYPRE_Precision precision_y, void *y, HYPRE_Int n)
{
   HYPRE_Int      i;

   /* Mixed-precision copy of data */
   switch (precision_x)
   {
      case HYPRE_REAL_SINGLE:
         switch (precision_y)
         {
            case HYPRE_REAL_DOUBLE:
#ifdef HYPRE_USING_OPENMP
               #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
               for (i = 0; i < n; i++)
               {
                  ((hypre_double *)y)[i] += (hypre_double)((hypre_float)alpha * ((hypre_float *)x)[i]);
               }
               break;
            case HYPRE_REAL_LONGDOUBLE:
#ifdef HYPRE_USING_OPENMP
               #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
               for (i = 0; i < n; i++)
               {
                  ((hypre_long_double *)y)[i] += (hypre_long_double)((hypre_float)alpha * ((hypre_float *)x)[i]);
               }
               break;
            default:
               break;
         }
         break;
      case HYPRE_REAL_DOUBLE:
         switch (precision_y)
         {
            case HYPRE_REAL_SINGLE:
#ifdef HYPRE_USING_OPENMP
               #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
               for (i = 0; i < n; i++)
               {
                  ((hypre_float *)y)[i] += (hypre_float)((hypre_double)alpha * ((hypre_double *)x)[i]);
               }
               break;
            case HYPRE_REAL_LONGDOUBLE:
#ifdef HYPRE_USING_OPENMP
               #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
               for (i = 0; i < n; i++)
               {
                  ((hypre_long_double *)y)[i] += (hypre_long_double)((hypre_double)alpha * ((hypre_double *)x)[i]);
               }
               break;
            default:
               break;
         }
         break;
      case HYPRE_REAL_LONGDOUBLE:
         switch (precision_y)
         {
            case HYPRE_REAL_SINGLE:
#ifdef HYPRE_USING_OPENMP
               #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
               for (i = 0; i < n; i++)
               {
                  ((hypre_float *)y)[i] += (hypre_float)((hypre_long_double)alpha * ((hypre_long_double *)x)[i]);
               }
               break;
            case HYPRE_REAL_DOUBLE:
#ifdef HYPRE_USING_OPENMP
               #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
               for (i = 0; i < n; i++)
               {
                  ((hypre_double *)y)[i] += (hypre_double)((hypre_long_double)alpha * ((hypre_long_double *)x)[i]);
               }
               break;
            default:
               break;
         }
         break;
      default:
         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Error: Undefined precision type for array Axpyn!\n");
         break;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------*
* hypre_RealArrayAxpyn_mp: Axpy on n array contents into y.
* Assumes arrays x and y have the same memory location.
*--------------------------------------------------------------------------*/
HYPRE_Int
hypre_RealArrayAxpyn_mp(HYPRE_Precision precision_x, void *x, HYPRE_Precision precision_y, void *y,
		        HYPRE_MemoryLocation location, HYPRE_Int n, hypre_long_double alpha)
{
#ifdef HYPRE_PROFILE
    hypre_profile_times[HYPRE_TIMER_ID_BLAS1] -= hypre_MPI_Wtime();
#endif

   hypre_GpuProfilingPushRange("RealArrayAxpyn");

   HYPRE_Int      i;

   /* Call standard memory copy if precisions match. */
   if (precision_x == precision_y)
   {
         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Error: Not Implemented!\n");
/*
#if defined(HYPRE_USING_GPU) || defined(HYPRE_USING_DEVICE_OPENMP)
      HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1(location);
      
      hypre_long_double d_alpha = (hypre_long_double)(*alpha);

      if (exec == HYPRE_EXEC_DEVICE)
      {
         hypreDevice_ComplexDeviceArrayAxpyn_pre(precision_y, d_alpha, x, y, n);
      }
      else
#endif
      {
         HYPRE_Int inc = 1;
         hypre_daxpy_pre(precision_y, n, alpha, x, inc, y, inc);

      }
      return hypre_error_flag;
*/
   }

#if defined(HYPRE_USING_GPU) || defined(HYPRE_USING_DEVICE_OPENMP)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1(location);

   if (exec == HYPRE_EXEC_DEVICE)
   {
      hypre_RealArrayAxpynDevice_mp(precision_x, alpha, x, precision_y, y, n);
   }
   else
#endif
   {
      hypre_RealArrayAxpynHost_mp(precision_x, alpha, x, precision_y, y, n);
   }

#ifdef HYPRE_PROFILE
    hypre_profile_times[HYPRE_TIMER_ID_BLAS1] += hypre_MPI_Wtime();
#endif
    hypre_GpuProfilingPopRange();

   return hypre_error_flag;
}


#endif
 
