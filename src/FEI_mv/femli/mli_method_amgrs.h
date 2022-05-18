/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Header info for the Ruge Stuben AMG data structure
 *
 *****************************************************************************/

#ifndef __MLIMETHODAMGRSH__
#define __MLIMETHODAMGRSH__

#include "_hypre_utilities.h"
#include "_hypre_parcsr_mv.h"
#include "mli.h"
#include "mli_matrix.h"
#include "mli_method.h"

#define MLI_METHOD_AMGRS_CLJP    0
#define MLI_METHOD_AMGRS_RUGE    1
#define MLI_METHOD_AMGRS_FALGOUT 2
#define MLI_METHOD_AMGRS_CR      3

/* ***********************************************************************
 * definition of the classical Ruge Stuben AMG data structure
 * ----------------------------------------------------------------------*/

class MLI_Method_AMGRS : public MLI_Method
{
   int      maxLevels_;              /* the finest level is 0            */
   int      numLevels_;              /* number of levels requested       */
   int      currLevel_;              /* current level being processed    */
   int      outputLevel_;            /* for diagnostics                  */
   int      coarsenScheme_;          /* coarsening scheme                */
   int      measureType_;            /* local or local measure           */
   double   threshold_;              /* strength threshold               */
   double   truncFactor_;            /* truncation factor                */
   int      mxelmtsP_;               /* max no. of elmts per row for P   */
   int      nodeDOF_;                /* equation block size (fixed)      */
   int      minCoarseSize_;          /* tell when to stop coarsening     */
   double   maxRowSum_;              /* used in Boomeramg                */
   int      symmetric_;              /* symmetric or nonsymmetric amg    */
   int      useInjectionForR_;       /* how R is to be constructed       */
   char     smoother_[20];           /* denote which pre-smoother to use */
   int      smootherNSweeps_;        /* number of pre-smoother sweeps    */
   double   *smootherWeights_;       /* number of postsmoother sweeps    */
   int      smootherPrintRNorm_;     /* tell smoother to print rnorm     */
   int      smootherFindOmega_;      /* tell smoother to find omega      */
   char     coarseSolver_[20];       /* denote which coarse solver to use*/
   int      coarseSolverNSweeps_;    /* number of coarse solver sweeps   */
   double   *coarseSolverWeights_;   /* weight used in coarse solver     */
   double   RAPTime_;
   double   totalTime_;

public :

   MLI_Method_AMGRS( MPI_Comm comm );
   ~MLI_Method_AMGRS();
   int    setup( MLI *mli );
   int    setParams(char *name, int argc, char *argv[]);

   int    setOutputLevel( int outputLevel );
   int    setNumLevels( int nlevels );
   int    setCoarsenScheme( int scheme );
   int    setMeasureType( int measure );
   int    setStrengthThreshold( double thresh );
   int    setMinCoarseSize( int minSize );
   int    setNodeDOF( int dof );
   int    setSmoother( char *stype, int num, double *wgt );
   int    setCoarseSolver( char *stype, int num, double *wgt );
   int    print();
   int    printStatistics(MLI *mli);
   MLI_Matrix *performCR(MLI_Matrix *, int *, MLI_Matrix **,int,
                         hypre_ParCSRMatrix *);
   MLI_Matrix *createPmat(int *, MLI_Matrix *, MLI_Matrix *, MLI_Matrix *);
};

#endif

