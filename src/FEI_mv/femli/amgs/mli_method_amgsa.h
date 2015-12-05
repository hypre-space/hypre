/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 1.29 $
 ***********************************************************************EHEADER*/





/******************************************************************************
 *
 * Header info for the smoothed aggregation data structure
 *
 *****************************************************************************/

#ifndef __MLIMETHODAMGSAH__
#define __MLIMETHODAMGSAH__

#include "utilities/_hypre_utilities.h"
#include "parcsr_mv/_hypre_parcsr_mv.h"
#include "base/mli.h"
#include "matrix/mli_matrix.h"
#include "amgs/mli_method.h"

#define MLI_METHOD_AMGSA_LOCAL        0
#define MLI_METHOD_AMGSA_HYBRID       1

/* ********************************************************************* *
 * internal data structure used for domain decomposition
 * --------------------------------------------------------------------- */

typedef struct MLI_AMGSA_DD_Struct
{
   int nSends;
   int nRecvs;
   int *sendLengs;
   int *recvLengs;
   int *sendProcs;
   int *recvProcs;
   int *sendMap;
   int nSendMap;
   int NNodes;
   int *ANodeEqnList;
   int *SNodeEqnList;
   int dofPerNode;
}
MLI_AMGSA_DD;

/* ***********************************************************************
 * definition of the aggregation based AMG data structure
 * ----------------------------------------------------------------------*/

class MLI_Method_AMGSA : public MLI_Method
{
   int      maxLevels_;              /* the finest level is 0            */
   int      numLevels_;              /* number of levels requested       */
   int      currLevel_;              /* current level being processed    */
   int      outputLevel_;            /* for diagnostics                  */
   int      scalar_;                 /* aggregate in scalar manner       */
   int      nodeDofs_;               /* equation block size (fixed)      */
   int      currNodeDofs_ ;          /* current block size (this stage)  */
   double   threshold_;              /* for creating aggregation graph   */
   int      nullspaceDim_;           /* null space information (changes  */
   int      nullspaceLen_;           /* length of nullspace in each dim  */
   double   *nullspaceVec_;          /* as curr_level)                   */
   int      numSmoothVec_;           /* if nonzero, use smooth vectors   */
   int      numSmoothVecSteps_;      /* steps for generating smooth vecs */
   double   Pweight_;                /* weight for prolongator smoother  */
   int      SPLevel_;                /* start level for P smoother       */
   double   dropTolForP_  ;          /* tolerance for sparsifying P      */
   int      *saCounts_;              /* store aggregation information at */
   int      **saData_;               /* each level                       */
   int      **saLabels_;             /* labels for aggregation           */
   int      **saDataAux_;            /* for subdomain aggregates         */
   double   *spectralNorms_;         /* computed matrix norms            */
   int      calcNormScheme_;         /* method to estimate matrix norm   */
   int      minAggrSize_;            /* tell when to stop aggregation    */
   int      minCoarseSize_;          /* tell when to stop aggregation    */
   int      coarsenScheme_;          /* different aggregation schemes    */
   char     preSmoother_[20];        /* denote which pre-smoother to use */
   char     postSmoother_[20];       /* denote which postsmoother to use */
   int      preSmootherNum_;         /* number of pre-smoother sweeps    */
   int      postSmootherNum_;        /* number of postsmoother sweeps    */
   double   *preSmootherWgt_;        /* weight used in pre-smoother      */
   double   *postSmootherWgt_;       /* weight used in postsmoother      */
   int	    smootherPrintRNorm_;     /* tell smoother to print rnorm     */
   int	    smootherFindOmega_;      /* tell smoother to find good omega */
   char     coarseSolver_[20];       /* denote which coarse solver to use*/
   int      coarseSolverNum_;        /* number of coarse solver sweeps   */
   double   *coarseSolverWgt_;       /* weight used in coarse solver     */
   int      calibrationSize_;        /* for calibration AMG method       */
   int      symmetric_;              /* choose between symm/nonsymm      */
   int      useSAMGeFlag_;           /* element based method             */
   int      useSAMGDDFlag_;          /* domain decomposition (NN) method */
   double   RAPTime_;
   double   totalTime_;
   int      ARPACKSuperLUExists_;
   MLI_AMGSA_DD *ddObj_;
   char     paramFile_[100];
   int      printToFile_;
   double   arpackTol_;

public :

   MLI_Method_AMGSA(MPI_Comm);
   ~MLI_Method_AMGSA();
   int    setup(MLI *);
   int    setParams(char *, int, char *argv[]);
   int    getParams(char *, int *, char *argv[]);

   int    setOutputLevel(int);
   int    setNumLevels(int);
   int    setSmoother(int, char *, int, double *);
   int    setCoarseSolver(char *, int, double *);
   int    setCoarsenScheme(int);
   int    setMinAggregateSize(int);
   int    setMinCoarseSize(int);
   int    setStrengthThreshold(double);
   int    setSmoothVec(int);
   int    setSmoothVecSteps(int);
   int    setPweight(double);
   int    setSPLevel(int);
   int    setCalcSpectralNorm();
   int    setAggregateInfo(int, int, int, int *);
   int    setNullSpace(int, int, double *, int);
   int    resetNullSpaceComponents(int, int, int *);
   int    adjustNullSpace(double *);
   int    setNodalCoordinates(int, int, int, double *, int, double *);
   int    setCalibrationSize(int);
   int    setupCalibration(MLI *);
   int    setupFEDataBasedNullSpaces(MLI *);
   int    setupFEDataBasedAggregates(MLI *);
   int    setupFEDataBasedSuperLUSmoother(MLI *, int);
   int    setupSFEIBasedNullSpaces(MLI *);
   int    setupSFEIBasedAggregates(MLI *);
   int    setupExtendedDomainDecomp(MLI *);
   int    setupExtendedDomainDecomp2(MLI *);
   int    setupSFEIBasedSuperLUSmoother(MLI *, int);
   int    print();
   int    printStatistics(MLI *);
   int    getNullSpace(int &, int &, double *&, int &);
   int    copy(MLI_Method *);
   int    relaxNullSpaces(MLI_Matrix *);

private :

   double genP( MLI_Matrix *, MLI_Matrix **, int, int * );
   double genPGlobal(hypre_ParCSRMatrix *, MLI_Matrix **, int, int *);
   int    formSmoothVec(MLI_Matrix *);
   int    formSmoothVecLanczos(MLI_Matrix *);
   int    smoothTwice(MLI_Matrix *mli_Amat);
   int    formLocalGraph( hypre_ParCSRMatrix *, hypre_ParCSRMatrix **, int *);
   int    formGlobalGraph(hypre_ParCSRMatrix *, hypre_ParCSRMatrix **);
   int    coarsenLocal( hypre_ParCSRMatrix *, int *, int **);
   int    coarsenGlobal(hypre_ParCSRMatrix *, int *, int **);
   double genP_DD(MLI_Matrix *, MLI_Matrix **, int **, int **);
   double genP_Selective(MLI_Matrix *, MLI_Matrix **, int, int *);
   int    coarsenGraded(hypre_ParCSRMatrix *graph, int *, int **, int **);
   int    coarsenSelective(hypre_ParCSRMatrix *graph, int *, int **, int *);
   double genP_AExt(MLI_Matrix *, MLI_Matrix **, int);
   int    coarsenAExt(hypre_ParCSRMatrix *graph, int *, int **, int);
};

#endif

