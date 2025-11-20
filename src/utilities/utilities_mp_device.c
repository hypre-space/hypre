/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
*
* hypre utilities mixed-precision interface on device
*
*****************************************************************************/

#include "_hypre_utilities.h"
#include "_hypre_utilities.hpp"
#include "_hypre_onedpl.hpp"

#if defined(HYPRE_MIXED_PRECISION)

#if defined(HYPRE_USING_GPU) || defined(HYPRE_USING_DEVICE_OPENMP)
/*--------------------------------------------------------------------------*
* hypre_RealArrayCopyDevice_mp: copy n array contents from x to y.
* Assumes arrays x and y are both on device memory.
*
* NOTE: We could use in if-statement for the inner switch statement on precision_y.
        However, inner switch-statement allows for additional future cases - DOK.
*--------------------------------------------------------------------------*/
HYPRE_Int
hypre_RealArrayCopyDevice_mp(HYPRE_Precision precision_x, void *x,
                             HYPRE_Precision precision_y, void *y, HYPRE_Int n)
{

   /* Mixed-precision copy of data.
    * Execute the same code for hypre_long_double and hypre_double
    */
   switch (precision_x)
   {
      case HYPRE_REAL_SINGLE:
         switch (precision_y)
         {
            case HYPRE_REAL_DOUBLE:
            case HYPRE_REAL_LONGDOUBLE:
            {
               hypre_float *xp = (hypre_float *)x;
               hypre_double *yp = (hypre_double *)y;

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
               HYPRE_THRUST_CALL( transform, xp, xp + n, yp,
                                  hypreFunctor_ElementCast<hypre_float, hypre_double>() );
#elif defined(HYPRE_USING_SYCL)
               HYPRE_ONEDPL_CALL( std::transform, xp, xp + n, yp, [](const auto & x) {return static_cast<hypre_double>(x);} );
#elif defined(HYPRE_USING_DEVICE_OPENMP)
               HYPRE_Int i;

               #pragma omp target teams distribute parallel for private(i) is_device_ptr(xp, yp)
               for (i = 0; i < n; i++)
               {
                  yp[i] = static_cast<hypre_double>(xp[i]);
               }
#endif
            }
            break;
            default:
               break;
         }
         break;
      case HYPRE_REAL_DOUBLE:
      case HYPRE_REAL_LONGDOUBLE:
         switch (precision_y)
         {
            case HYPRE_REAL_SINGLE:
            {
               hypre_double *xp = (hypre_double *)x;
               hypre_float *yp = (hypre_float *)y;

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
               HYPRE_THRUST_CALL( transform, xp, xp + n, yp,
                                  hypreFunctor_ElementCast<hypre_double, hypre_float>() );
#elif defined(HYPRE_USING_SYCL)
               HYPRE_ONEDPL_CALL( std::transform, xp, xp + n, yp, [](const auto & x) {return static_cast<hypre_float>(x);});
#elif defined(HYPRE_USING_DEVICE_OPENMP)
               HYPRE_Int i;

               #pragma omp target teams distribute parallel for private(i) is_device_ptr(xp, yp)
               for (i = 0; i < n; i++)
               {
                  yp[i] = static_cast<hypre_float>(xp[i]);
               }
#endif
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
* hypre_RealArrayAxpynDevice_mp: Axpy on n array contents into y.
* Assumes arrays x and y are both on device memory.
*
* NOTE: We could use in if-statement for the inner switch statement on precision_y.
        However, inner switch-statement allows for additional future cases - DOK.
*--------------------------------------------------------------------------*/
HYPRE_Int
hypre_RealArrayAxpynDevice_mp(HYPRE_Precision precision_x, hypre_long_double alpha, void *x,
                              HYPRE_Precision precision_y, void *y, HYPRE_Int n)
{

   /* Mixed-precision copy of data.
    * Execute the same code for hypre_long_double and hypre_double
    */
   switch (precision_x)
   {
      case HYPRE_REAL_SINGLE:
         switch (precision_y)
         {
            case HYPRE_REAL_DOUBLE:
            case HYPRE_REAL_LONGDOUBLE:
            {
               hypre_float *xp = (hypre_float *)x;
               hypre_double *yp = (hypre_double *)y;

#if defined(HYPRE_USING_GPU)
               hypreDevice_Axpyzn_mp(n, xp, yp, yp, (hypre_float)alpha, 1.0);
               hypre_SyncComputeStream();
#elif defined(HYPRE_USING_DEVICE_OPENMP)
               HYPRE_Int i;

               #pragma omp target teams distribute parallel for private(i) is_device_ptr(xp, yp)
               for (i = 0; i < n; i++)
               {
                  yp[i] += static_cast<hypre_double>((hypre_float)alpha * xp[i]);
               }
#endif
            }
            break;
            default:
               break;
         }
         break;
      case HYPRE_REAL_DOUBLE:
      case HYPRE_REAL_LONGDOUBLE:
         switch (precision_y)
         {
            case HYPRE_REAL_SINGLE:
            {
               hypre_double *xp = (hypre_double *)x;
               hypre_float *yp = (hypre_float *)y;

#if defined(HYPRE_USING_GPU)

               hypreDevice_Axpyzn_mp(n, xp, yp, yp, (hypre_double)alpha, 1.0f);
               hypre_SyncComputeStream();
#elif defined(HYPRE_USING_DEVICE_OPENMP)
               HYPRE_Int i;

               #pragma omp target teams distribute parallel for private(i) is_device_ptr(xp, yp)
               for (i = 0; i < n; i++)
               {
                  yp[i] += static_cast<hypre_float>((hypre_double)alpha * xp[i]);
               }
#endif
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

#endif // HYPRE_USING_GPU || HYPRE_USING_DEVICE_OPENMP
#endif // HYPRE_MIXED_PRECISION

