/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef hypre_ParCheby_DATA_HEADER
#define hypre_ParCheby_DATA_HEADER

/*--------------------------------------------------------------------------
 * hypre_ParChebyData
 *--------------------------------------------------------------------------*/

typedef struct hypre_ParChebyData_struct
{
   /* Base solver data structure */
   hypre_Solver          base;

   /* Solver options */
   HYPRE_Int             print_level;
   HYPRE_Int             zero_guess;
   HYPRE_Int             max_iter;
   HYPRE_Int             order;
   HYPRE_Int             variant;
   HYPRE_Int             scale;
   HYPRE_Int             eig_est;
   HYPRE_Int             eig_provided;
   HYPRE_Real            eig_ratio;
   HYPRE_Real            min_eig_est;
   HYPRE_Real            max_eig_est;
   HYPRE_Real            tol;

   /* log info */
   HYPRE_Int             logging;
   HYPRE_Int             num_iterations;
   HYPRE_Real            rel_resid_norm;
   hypre_ParVector      *residual; /* available if logging > 1 */

   /* Work variables */
   HYPRE_Real           *coefs;
   hypre_ParVector      *scaling;
   hypre_ParVector      *Ptemp;
   hypre_ParVector      *Rtemp;
   hypre_ParVector      *Vtemp;
   hypre_ParVector      *Ztemp;
   HYPRE_Int             owns_temp;

   /* Statistics variables */

} hypre_ParChebyData;

/*--------------------------------------------------------------------------
 * Accessor macros
 *--------------------------------------------------------------------------*/

#define hypre_ParChebyDataPrintLevel(data)     ((data) -> print_level)
#define hypre_ParChebyDataZeroGuess(data)      ((data) -> zero_guess)
#define hypre_ParChebyDataMaxIterations(data)  ((data) -> max_iter)
#define hypre_ParChebyDataTol(data)            ((data) -> tol)
#define hypre_ParChebyDataOrder(data)          ((data) -> order)
#define hypre_ParChebyDataVariant(data)        ((data) -> variant)
#define hypre_ParChebyDataScale(data)          ((data) -> scale)
#define hypre_ParChebyDataEigEst(data)         ((data) -> eig_est)
#define hypre_ParChebyDataEigProvided(data)    ((data) -> eig_provided)
#define hypre_ParChebyDataEigRatio(data)       ((data) -> eig_ratio)
#define hypre_ParChebyDataMinEigEst(data)      ((data) -> min_eig_est)
#define hypre_ParChebyDataMaxEigEst(data)      ((data) -> max_eig_est)

#define hypre_ParChebyDataLogging(data)        ((data) -> logging)
#define hypre_ParChebyDataNumIterations(data)  ((data) -> num_iterations)
#define hypre_ParChebyDataRelResidNorm(data)   ((data) -> rel_resid_norm)
#define hypre_ParChebyDataResidual(data)       ((data) -> residual)

#define hypre_ParChebyDataScaling(data)        ((data) -> scaling)
#define hypre_ParChebyDataCoefs(data)          ((data) -> coefs)
#define hypre_ParChebyDataPtemp(data)          ((data) -> Ptemp)
#define hypre_ParChebyDataRtemp(data)          ((data) -> Rtemp)
#define hypre_ParChebyDataVtemp(data)          ((data) -> Vtemp)
#define hypre_ParChebyDataZtemp(data)          ((data) -> Ztemp)
#define hypre_ParChebyDataOwnsTemp(data)       ((data) -> owns_temp)

#endif /* #ifndef hypre_ParCheby_DATA_HEADER */
