/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.25 $
 ***********************************************************************EHEADER*/




/****************************************************************************/ 
/* HYPRE_LSI_MLI interface                                                  */
/*--------------------------------------------------------------------------*/
/*  local functions :
 * 
 *        HYPRE_LSI_MLICreate
 *        HYPRE_LSI_MLIDestroy
 *        HYPRE_LSI_MLISetup
 *        HYPRE_LSI_MLISolve
 *        HYPRE_LSI_MLISetParams
 *--------------------------------------------------------------------------
 *        HYPRE_LSI_MLICreateNodeEqnMap
 *        HYPRE_LSI_MLIAdjustNodeEqnMap
 *        HYPRE_LSI_MLIAdjustNullSpace
 *        HYPRE_LSI_MLISetFEData
 *        HYPRE_LSI_MLILoadNodalCoordinates
 *        HYPRE_LSI_MLILoadMatrixScalings
 *        HYPRE_LSI_MLILoadMaterialLabels
 *--------------------------------------------------------------------------
 *        HYPRE_LSI_MLIFEDataCreate
 *        HYPRE_LSI_MLIFEDataDestroy
 *        HYPRE_LSI_MLIFEDataInitFields
 *        HYPRE_LSI_MLIFEDataInitElemBlock
 *        HYPRE_LSI_MLIFEDataInitElemNodeList
 *        HYPRE_LSI_MLIFEDataInitSharedNodes
 *        HYPRE_LSI_MLIFEDataInitComplete
 *        HYPRE_LSI_MLIFEDataLoadElemMatrix
 *        HYPRE_LSI_MLIFEDataWriteToFile
 *--------------------------------------------------------------------------
 *        HYPRE_LSI_MLISFEICreate
 *        HYPRE_LSI_MLISFEIDestroy
 *        HYPRE_LSI_MLISFEILoadElemMatrices
 *        HYPRE_LSI_MLISFEIAddNumElems
 ****************************************************************************/

/****************************************************************************/ 
/* system include files                                                     */
/*--------------------------------------------------------------------------*/

#include "HYPRE_utilities.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

#ifdef WIN32
#define strcmp _stricmp
#endif

#define HAVE_MLI 

/****************************************************************************/ 
/* MLI include files                                                        */
/*--------------------------------------------------------------------------*/

#ifdef HAVE_MLI
#include "base/mli.h"
#include "util/mli_utils.h"
#include "amgs/mli_method.h"
#endif
#include "HYPRE_LSI_mli.h"

/****************************************************************************/ 
/* HYPRE_LSI_MLI data structure                                             */
/*--------------------------------------------------------------------------*/

typedef struct HYPRE_LSI_MLI_Struct
{
#ifdef HAVE_MLI
   MLI        *mli_;
   MLI_FEData *feData_;           /* holds FE information */
   MLI_SFEI   *sfei_;             /* holds FE information */
   MLI_Mapper *mapper_;           /* holds mapping information */
#endif
   MPI_Comm mpiComm_;
   int      outputLevel_;         /* for diagnostics */
   int      nLevels_;             /* max number of levels */
   int      cycleType_;           /* 1 for V and 2 for W */
   int      maxIterations_;       /* default - 1 iteration */
   char     method_[20];          /* default - smoothed aggregation */
   char     coarsenScheme_[20];   /* default - default in MLI */
   char     preSmoother_[20];     /* default - symmetric Gauss Seidel */
   char     postSmoother_ [20];   /* default - symmetric Gauss Seidel */
   int      preNSweeps_;          /* default - 2 smoothing steps */
   int      postNSweeps_;         /* default - 2 smoothing steps */
   double   *preSmootherWts_;     /* relaxation weights */
   double   *postSmootherWts_;    /* relaxation weights */
   int      smootherPrintRNorm_;  /* for smoother diagnostics */
   int      smootherFindOmega_;   /* for SGS smoother */
   double   strengthThreshold_;   /* strength threshold */
   char     coarseSolver_[20];    /* default = SuperLU */
   int      coarseSolverNSweeps_; /* number of sweeps (if iterative used) */
   double   *coarseSolverWts_;    /* relaxation weight (if Jacobi used) */
   int      minCoarseSize_;       /* minimum coarse grid size */
   int      scalar_;              /* aggregate in a scalar manner */
   int      nodeDOF_;             /* node degree of freedom */
   int      spaceDim_;            /* 2D or 3D */
   int      nSpaceDim_;           /* number of null vectors */
   int      localNEqns_;          /* number of equations locally */
   int      nCoordAccept_;        /* flag to accept nodal coordinate or not */ 
   double   *nCoordinates_;       /* for storing nodal coordinates */
   double   *nullScales_;         /* scaling vector for null space */
   int      calibrationSize_;     /* for calibration smoothed aggregation */
   double   Pweight_;
   int      SPLevel_;
   char     paramFile_[50];
   int      adjustNullSpace_;
   int      numResetNull_;
   int      *resetNullIndices_;
   int      numMatLabels_;        /* for controlling aggregation */
   int      *matLabels_;
   int      printNullSpace_;
   int      symmetric_;
   int      injectionForR_;
   HYPRE_ParCSRMatrix correctionMatrix_; /* for nullspace correction */
   int      numSmoothVecs_;
   int      smoothVecSteps_;
   double   arpackTol_;
} 
HYPRE_LSI_MLI;

/****************************************************************************/ 
/* HYPRE_MLI_FEData data structure                                          */
/*--------------------------------------------------------------------------*/

typedef struct HYPRE_MLI_FEData_Struct
{
#ifdef HAVE_MLI
   MPI_Comm   comm_;              /* MPI communicator */
   MLI_FEData *fedata_;           /* holds FE information */
   int        fedataOwn_;         /* flag to indicate ownership */
   int        computeNull_;       /* flag - compute null space or not */
   int        nullDim_;           /* number of null space vectors */
#endif
}
HYPRE_MLI_FEData;

/****************************************************************************/ 
/* HYPRE_MLI_SFEI data structure                                            */
/*--------------------------------------------------------------------------*/

typedef struct HYPRE_MLI_SFEI_Struct
{
#ifdef HAVE_MLI
   MPI_Comm   comm_;            /* MPI communicator */
   MLI_SFEI   *sfei_;           /* holds FE information */
   int        sfeiOwn_;         /* flag to indicate ownership */
#endif
}
HYPRE_MLI_SFEI;

/****************************************************************************/
/* HYPRE_LSI_MLICreate                                                      */
/*--------------------------------------------------------------------------*/

extern "C"
int HYPRE_LSI_MLICreate( MPI_Comm comm, HYPRE_Solver *solver )
{
   HYPRE_LSI_MLI *mli_object = (HYPRE_LSI_MLI *) malloc(sizeof(HYPRE_LSI_MLI));
   *solver = (HYPRE_Solver) mli_object;
   mli_object->mpiComm_             = comm;
   mli_object->outputLevel_         = 0;
   mli_object->nLevels_             = 0;
   mli_object->maxIterations_       = 1;
   mli_object->cycleType_           = 1;
   strcpy(mli_object->method_ , "AMGSA");
   strcpy(mli_object->coarsenScheme_ , "default"); /* default in MLI */
   strcpy(mli_object->preSmoother_, "default");
   strcpy(mli_object->postSmoother_, "default");
   mli_object->preNSweeps_          = 1;
   mli_object->postNSweeps_         = 1;
   mli_object->preSmootherWts_      = NULL;
   mli_object->postSmootherWts_     = NULL;
   mli_object->smootherPrintRNorm_  = 0;
   mli_object->smootherFindOmega_   = 0;
   mli_object->strengthThreshold_   = 0.0;
   strcpy(mli_object->coarseSolver_, "default"); /* default in MLI */
   mli_object->coarseSolverNSweeps_ = 0;
   mli_object->coarseSolverWts_     = NULL;
   mli_object->minCoarseSize_       = 0;
   mli_object->scalar_              = 0;
   mli_object->nodeDOF_             = 1;
   mli_object->spaceDim_            = 1;
   mli_object->nSpaceDim_           = 1;
   mli_object->localNEqns_          = 0;
   mli_object->nCoordinates_        = NULL;
   mli_object->nCoordAccept_        = 0;
   mli_object->nullScales_          = NULL;
   mli_object->calibrationSize_     = 0;
   mli_object->Pweight_             = -1.0;      /* default in MLI */
   mli_object->SPLevel_             = 0;         /* default in MLI */
   mli_object->adjustNullSpace_     = 0;
   mli_object->numResetNull_        = 0;
   mli_object->resetNullIndices_    = NULL;
   mli_object->correctionMatrix_    = NULL;
   strcpy(mli_object->paramFile_, "empty");
   mli_object->numMatLabels_        = 0;
   mli_object->matLabels_           = NULL;
   mli_object->printNullSpace_      = 0;
   mli_object->symmetric_           = 1;
   mli_object->injectionForR_       = 0;
   mli_object->numSmoothVecs_       = 0;
   mli_object->smoothVecSteps_      = 0;
   mli_object->arpackTol_           = 0.0;
#ifdef HAVE_MLI
   mli_object->mli_                 = NULL;
   mli_object->sfei_                = NULL;
   mli_object->feData_              = NULL;
   mli_object->mapper_              = NULL;
   return 0;
#else
   printf("MLI not available.\n");
   exit(1);
   return -1;
#endif
}

/****************************************************************************/
/* HYPRE_LSI_MLIDestroy                                                     */
/*--------------------------------------------------------------------------*/

extern "C"
int HYPRE_LSI_MLIDestroy( HYPRE_Solver solver )
{
   HYPRE_LSI_MLI *mli_object = (HYPRE_LSI_MLI *) solver;

   if ( mli_object->preSmootherWts_ != NULL ) 
      delete [] mli_object->preSmootherWts_;
   if ( mli_object->postSmootherWts_ != NULL ) 
      delete [] mli_object->postSmootherWts_;
   if ( mli_object->coarseSolverWts_ != NULL ) 
      delete [] mli_object->coarseSolverWts_;
   if ( mli_object->nCoordinates_ != NULL ) 
      delete [] mli_object->nCoordinates_;
   if ( mli_object->nullScales_ != NULL ) 
      delete [] mli_object->nullScales_;
   if ( mli_object->resetNullIndices_ != NULL ) 
      delete [] mli_object->resetNullIndices_;
   if ( mli_object->correctionMatrix_ != NULL ) 
      HYPRE_ParCSRMatrixDestroy(mli_object->correctionMatrix_); 
   if ( mli_object->matLabels_ != NULL ) delete [] mli_object->matLabels_; 
#ifdef HAVE_MLI
   if ( mli_object->feData_ != NULL ) delete mli_object->feData_;
   if ( mli_object->mli_ != NULL ) delete mli_object->mli_;
   free( mli_object );
   return 0;
#else
   free( mli_object );
   printf("MLI not available.\n");
   return -1;
#endif
}

/****************************************************************************/
/* HYPRE_LSI_MLISetup                                                       */
/*--------------------------------------------------------------------------*/

extern "C"
int HYPRE_LSI_MLISetup( HYPRE_Solver solver, HYPRE_ParCSRMatrix A,
                        HYPRE_ParVector b,   HYPRE_ParVector x )
{
#ifdef HAVE_MLI
   int           targc, nNodes, iZero=0;
   double        tol=1.0e-8;
   char          *targv[6], paramString[100];;
   HYPRE_LSI_MLI *mli_object;
   MLI_Matrix    *mli_mat;   
   MLI_Method    *method;   
   MPI_Comm      mpiComm;
   MLI           *mli;

   /* -------------------------------------------------------- */ 
   /* create object                                            */
   /* -------------------------------------------------------- */ 

   mli_object = (HYPRE_LSI_MLI *) solver;
   mpiComm    = mli_object->mpiComm_;
   mli        = new MLI( mpiComm );
   if ( mli_object->mli_ != NULL ) delete mli_object->mli_;
   mli_object->mli_ = mli;

   /* -------------------------------------------------------- */ 
   /* set general parameters                                   */
   /* -------------------------------------------------------- */ 

   if (!strcmp(mli_object->method_,"AMGSADD") ||
       !strcmp(mli_object->method_,"AMGSADDe")) mli_object->nLevels_ = 2;
   
   mli->setNumLevels( mli_object->nLevels_ );
   mli->setTolerance( tol );

   /* -------------------------------------------------------- */ 
   /* set method specific parameters                           */
   /* -------------------------------------------------------- */ 

   method = MLI_Method_CreateFromName(mli_object->method_,mpiComm);
   if ( mli_object->outputLevel_ > 0 )
   {
      sprintf(paramString, "setOutputLevel %d", mli_object->outputLevel_);
      method->setParams( paramString, 0, NULL );
   }
   if ( mli_object->nLevels_ > 0 )
   {
      sprintf(paramString, "setNumLevels %d", mli_object->nLevels_);
      method->setParams( paramString, 0, NULL );
   }
   if ( mli_object->strengthThreshold_ > 0.0 )
   {
      sprintf(paramString, "setStrengthThreshold %f",
           mli_object->strengthThreshold_);
      method->setParams( paramString, 0, NULL );
   }
   if ( mli_object->scalar_ == 1 )
   {
      strcpy(paramString, "scalar");
      method->setParams( paramString, 0, NULL );
   }
   if ( mli_object->symmetric_ == 0 )
   {
      strcpy(paramString, "nonsymmetric");
      method->setParams( paramString, 0, NULL );
   }
   if ( mli_object->injectionForR_ == 1 )
   {
      strcpy(paramString, "useInjectionForR");
      method->setParams( paramString, 0, NULL );
   }
   if ( mli_object->smootherPrintRNorm_ == 1 )
   {
      strcpy(paramString, "setSmootherPrintRNorm");
      method->setParams( paramString, 0, NULL );
   }
   if ( mli_object->smootherFindOmega_ == 1 )
   {
      strcpy(paramString, "setSmootherFindOmega");
      method->setParams( paramString, 0, NULL );
   }
   if ( mli_object->numSmoothVecs_ > 0 )
   {
      sprintf(paramString, "setSmoothVec %d", mli_object->numSmoothVecs_);
      method->setParams( paramString, 0, NULL );
      if ( mli_object->smoothVecSteps_ > 0 )
         sprintf(paramString, "setSmoothVecSteps %d", 
                 mli_object->smoothVecSteps_);
      else
         sprintf(paramString, "setSmoothVecSteps 5"); 
      method->setParams( paramString, 0, NULL );
   }
   if ( mli_object->arpackTol_ > 0.0 )
   {
      sprintf(paramString, "arpackTol %e", mli_object->arpackTol_);
      method->setParams( paramString, 0, NULL );
   }

   /* -------------------------------------------------------- */ 
   /* set up presmoother                                       */
   /* -------------------------------------------------------- */ 

   if ( strcmp(mli_object->preSmoother_, "default") )
   {
      targc    = 2;
      targv[0] = (char *) &mli_object->preNSweeps_;
      targv[1] = (char *) mli_object->preSmootherWts_;
      sprintf(paramString, "setPreSmoother %s", mli_object->preSmoother_);
      method->setParams( paramString, targc, targv );
   }

   /* -------------------------------------------------------- */ 
   /* set up postsmoother                                      */
   /* -------------------------------------------------------- */ 

   if ( strcmp(mli_object->preSmoother_, "default") )
   {
      targc    = 2;
      targv[0] = (char *) &mli_object->postNSweeps_;
      targv[1] = (char *) mli_object->postSmootherWts_;
      sprintf(paramString, "setPostSmoother %s", mli_object->postSmoother_);
      method->setParams( paramString, targc, targv );
   }

   /* -------------------------------------------------------- */ 
   /* set up coarse solver                                     */
   /* -------------------------------------------------------- */ 

   if ( strcmp(mli_object->coarseSolver_, "default") )
   {
      targc    = 2;
      targv[0] = (char *) &(mli_object->coarseSolverNSweeps_);
      targv[1] = (char *) mli_object->coarseSolverWts_;
      sprintf(paramString, "setCoarseSolver %s", mli_object->coarseSolver_);
      method->setParams( paramString, targc, targv );
   }

   /* -------------------------------------------------------- */ 
   /* load minimum coarse grid size and others                 */
   /* -------------------------------------------------------- */ 

   if ( mli_object->minCoarseSize_ != 0 )
   {
      sprintf(paramString, "setMinCoarseSize %d",mli_object->minCoarseSize_);
      method->setParams( paramString, 0, NULL );
   }
   if ( mli_object->Pweight_ >= 0.0 )
   {
      sprintf( paramString, "setPweight %e", mli_object->Pweight_ );
      method->setParams( paramString, 0, NULL );
      if ( mli_object->SPLevel_ > 0 )
      { 
         sprintf( paramString, "setSPLevel %d", mli_object->SPLevel_ );
         method->setParams( paramString, 0, NULL );
      }
   }
   if ( strcmp(mli_object->coarsenScheme_, "default") )
   {
      sprintf(paramString, "setCoarsenScheme %s",mli_object->coarsenScheme_);
      method->setParams( paramString, 0, NULL );
   }

   /* -------------------------------------------------------- */ 
   /* load calibration size                                    */
   /* -------------------------------------------------------- */ 

   if ( mli_object->calibrationSize_ > 0 )
   {
      sprintf(paramString,"setCalibrationSize %d",mli_object->calibrationSize_);
      method->setParams( paramString, 0, NULL );
   }

   /* -------------------------------------------------------- */ 
   /* load FEData, if there is any                             */
   /* -------------------------------------------------------- */ 

   if ( mli_object->feData_ != NULL )
      mli->setFEData( 0, mli_object->feData_, mli_object->mapper_ );
   if ( mli_object->sfei_ != NULL ) mli->setSFEI(0, mli_object->sfei_);
   //mli_object->mapper_ = NULL;
   //mli_object->feData_ = NULL;

   /* -------------------------------------------------------- */ 
   /* load null space, if there is any                         */
   /* -------------------------------------------------------- */ 

   if ((mli_object->printNullSpace_ & 1) == 1 )
   {
      strcpy( paramString, "printNullSpace" );
      method->setParams( paramString, 0, NULL );
   }
   if ((mli_object->printNullSpace_ & 2) == 2 )
   {
      strcpy( paramString, "printElemNodeList" );
      method->setParams( paramString, 0, NULL );
   }
   if ((mli_object->printNullSpace_ & 4) == 4 )
   {
      strcpy( paramString, "printNodalCoord" );
      method->setParams( paramString, 0, NULL );
   }
   if ( mli_object->nCoordinates_ != NULL )
   {
      nNodes = mli_object->localNEqns_ / mli_object->nodeDOF_;
      targc    = 6;
      targv[0] = (char *) &nNodes;
      targv[1] = (char *) &(mli_object->nodeDOF_);
      targv[2] = (char *) &(mli_object->spaceDim_);
      targv[3] = (char *) mli_object->nCoordinates_;
      targv[4] = (char *) &(mli_object->nSpaceDim_);
      targv[5] = (char *) mli_object->nullScales_;
      strcpy( paramString, "setNodalCoord" );
      method->setParams( paramString, targc, targv );
#if 0
      if ( mli_object->adjustNullSpace_ == 1 &&
           mli_object->correctionMatrix_ != NULL )
      {   
         int                iV, irow, numNS, length, *partition, mypid;
         double             *NSpace, *vecInData, *vecOutData, *nullCorrection;
         HYPRE_IJVector     IJVecIn, IJVecOut;
         HYPRE_ParCSRMatrix hypreA;
         hypre_ParVector    *hypreVecIn, *hypreVecOut;
         HYPRE_Solver       solver;

         strcpy( paramString, "getNullSpace" );
         method->getParams( paramString, &targc, targv );
         numNS  = *(int *)   targv[1];
         NSpace = (double *) targv[2]; 
         length = *(int *)   targv[3]; 
         hypreA = mli_object->correctionMatrix_;
         HYPRE_ParCSRMatrixGetRowPartitioning( hypreA, &partition );
         MPI_Comm_rank( mpiComm , &mypid );
         HYPRE_IJVectorCreate(mpiComm, partition[mypid], 
                              partition[mypid+1]-1, &IJVecIn);
         HYPRE_IJVectorSetObjectType(IJVecIn, HYPRE_PARCSR);
         HYPRE_IJVectorInitialize(IJVecIn);
         HYPRE_IJVectorAssemble(IJVecIn);
         HYPRE_IJVectorCreate(mpiComm, partition[mypid], 
                              partition[mypid+1]-1, &IJVecOut);
         HYPRE_IJVectorSetObjectType(IJVecOut, HYPRE_PARCSR);
         HYPRE_IJVectorInitialize(IJVecOut);
         HYPRE_IJVectorAssemble(IJVecOut);
         HYPRE_IJVectorGetObject(IJVecIn, (void **) &hypreVecIn);
         HYPRE_IJVectorGetObject(IJVecOut, (void **) &hypreVecOut);
         vecInData = hypre_VectorData(hypre_ParVectorLocalVector(hypreVecIn));
         vecOutData = hypre_VectorData(hypre_ParVectorLocalVector(hypreVecOut));
         nullCorrection = new double[numNS*length];
         HYPRE_ParCSRGMRESCreate(mpiComm, &solver);
         HYPRE_ParCSRGMRESSetKDim(solver, 100);
         HYPRE_ParCSRGMRESSetMaxIter(solver, 100);
         HYPRE_ParCSRGMRESSetTol(solver, 1.0e-8);
         HYPRE_ParCSRGMRESSetPrecond(solver, HYPRE_ParCSRDiagScale,
                                     HYPRE_ParCSRDiagScaleSetup,NULL);
         HYPRE_ParCSRGMRESSetLogging(solver, mli_object->outputLevel_);
         HYPRE_ParCSRGMRESSetup(solver, hypreA, (HYPRE_ParVector) hypreVecIn, 
                                (HYPRE_ParVector) hypreVecOut);

         for ( iV = 0; iV < numNS; iV++ )
         {
            for ( irow = 0; irow < length; irow++ )
               vecOutData[irow] = NSpace[length*iV+irow];
            HYPRE_ParCSRMatrixMatvec(1.0,hypreA,(HYPRE_ParVector) hypreVecOut, 
                                     0.0, (HYPRE_ParVector) hypreVecIn);
            for ( irow = 0; irow < length; irow++ ) vecOutData[irow] = 0.0;
            HYPRE_ParCSRGMRESSolve(solver,A,(HYPRE_ParVector) hypreVecIn, 
                                   (HYPRE_ParVector) hypreVecOut);
            for ( irow = 0; irow < length; irow++ )
               nullCorrection[length*iV+irow] = - vecOutData[irow];
         }
         strcpy( paramString, "adjustNullSpace" );
         targc = 1;
         targv[0] = (char *) nullCorrection;
         method->setParams( paramString, targc, targv );
         HYPRE_ParCSRGMRESDestroy(solver);
         HYPRE_IJVectorDestroy( IJVecIn );
         HYPRE_IJVectorDestroy( IJVecOut );
         delete [] nullCorrection;
         free( partition );
      }
      if ( mli_object->numResetNull_ != 0 &&
           mli_object->resetNullIndices_ != NULL )
      {
         int *rowPartition, my_id;
         HYPRE_ParCSRMatrixGetRowPartitioning( A, &rowPartition );
         MPI_Comm_rank( mpiComm , &my_id );
         targc = 3;
         targv[0] = (char *) &(mli_object->numResetNull_);
         targv[1] = (char *) &(rowPartition[my_id]);
         targv[2] = (char *) (mli_object->resetNullIndices_);
         strcpy( paramString, "resetNullSpaceComponents" );
         method->setParams( paramString, targc, targv );
         free( rowPartition );
      }
#endif
   }
   else
   {
      targc    = 4;
      targv[0] = (char *) &(mli_object->nodeDOF_);
      //if ( mli_object->nSpaceDim_ > mli_object->nodeDOF_ ) 
      //   mli_object->nSpaceDim_ = mli_object->nodeDOF_; 
      targv[1] = (char *) &(mli_object->nSpaceDim_);
      targv[2] = (char *) NULL;
      targv[3] = (char *) &iZero;
      strcpy( paramString, "setNullSpace" );
      method->setParams( paramString, targc, targv );
   }
   if ( mli_object->correctionMatrix_ != NULL )
   {
      HYPRE_ParCSRMatrixDestroy( mli_object->correctionMatrix_ );
      mli_object->correctionMatrix_ = NULL;
   }
   if (!strcmp(mli_object->method_,"AMGRS") )
   {
      sprintf( paramString, "setNodeDOF %d", mli_object->nodeDOF_ );
      method->setParams( paramString, 0, NULL );
   }

   /* -------------------------------------------------------- */ 
   /* load material labels, if there is any                    */
   /* -------------------------------------------------------- */ 
   
   if ( mli_object->matLabels_ != NULL )
   {
      strcpy( paramString, "setLabels" );
      targc = 3;
      targv[0] = (char *) &(mli_object->numMatLabels_);
      targv[1] = (char *) &iZero;
      targv[2] = (char *) mli_object->matLabels_;
      method->setParams( paramString, targc, targv );
   }

   /* -------------------------------------------------------- */ 
   /* set parameter file                                       */
   /* -------------------------------------------------------- */ 
   
   if ( strcmp(mli_object->paramFile_, "empty") )
   {
      targc    = 1;
      targv[0] = (char *) mli_object->paramFile_;
      strcpy( paramString, "setParamFile" );
      method->setParams( paramString, targc, targv );
   }
   if ( mli_object->outputLevel_ >= 1 )
   {
      strcpy( paramString, "print" );
      method->setParams( paramString, 0, NULL );
   }

   /* -------------------------------------------------------- */ 
   /* finally, set up                                          */
   /* -------------------------------------------------------- */ 

   strcpy( paramString, "HYPRE_ParCSR" );
   mli_mat = new MLI_Matrix((void*) A, paramString, NULL);
   mli->setMethod( method );
   mli->setSystemMatrix( 0, mli_mat );
   mli->setOutputLevel( mli_object->outputLevel_ );
   mli->setup();
   mli->setMaxIterations( mli_object->maxIterations_ );
   mli->setCyclesAtLevel( -1, mli_object->cycleType_ );
   return 0;
#else
   printf("MLI not available.\n");
   return -1;
#endif
}

/****************************************************************************/
/* HYPRE_LSI_MLISolve                                                       */
/*--------------------------------------------------------------------------*/

extern "C"
int HYPRE_LSI_MLISolve( HYPRE_Solver solver, HYPRE_ParCSRMatrix A,
                        HYPRE_ParVector b, HYPRE_ParVector x )
{
#ifdef HAVE_MLI
   HYPRE_LSI_MLI *mli_object;
   MLI_Vector    *sol, *rhs;
   char          paramString[100];

   strcpy(paramString, "HYPRE_ParVector");
   sol = new MLI_Vector( (void *) x, paramString, NULL);
   rhs = new MLI_Vector( (void *) b, paramString, NULL);

   mli_object = (HYPRE_LSI_MLI *) solver;
   if ( mli_object->mli_ == NULL )
   {
      printf("HYPRE_LSI_MLISolve ERROR : mli not instantiated.\n");
      exit(1);
   }
   mli_object->mli_->solve( sol, rhs);

   return 0;
#else
   printf("MLI not available.\n");
   return -1;
#endif
}

/****************************************************************************/
/* HYPRE_LSI_MLISetParams                                                   */
/*--------------------------------------------------------------------------*/

extern "C"
int HYPRE_LSI_MLISetParams( HYPRE_Solver solver, char *paramString )
{
   int           i, mypid;
   double        weight;
   HYPRE_LSI_MLI *mli_object;
   char          param1[256], param2[256], param3[256];

   mli_object = (HYPRE_LSI_MLI *) solver;
   sscanf(paramString,"%s", param1);
   if ( strcmp(param1, "MLI") )
   {
      printf("HYPRE_LSI_MLI::parameters not for me.\n");
      return 1;
   }
   MPI_Comm_rank( mli_object->mpiComm_, &mypid );
   sscanf(paramString,"%s %s", param1, param2);
   if ( !strcmp(param2, "help") )
   {
      if ( mypid == 0 )
      {
         printf("%4d : Available options for MLI are : \n", mypid);
         printf("\t      outputLevel <d> \n");
         printf("\t      numLevels <d> \n");
         printf("\t      maxIterations <d> \n");
         printf("\t      cycleType <'V','W'> \n");
         printf("\t      strengthThreshold <f> \n");
         printf("\t      method <AMGSA, AMGSAe> \n");
         printf("\t      coarsenScheme <local, hybrid, cljp, falgout> \n");
         printf("\t      smoother <Jacobi,GS,...> \n");
         printf("\t      coarseSolver <Jacobi,GS,...> \n");
         printf("\t      numSweeps <d> \n");
         printf("\t      smootherWeight <f> \n");
         printf("\t      smootherPrintRNorm \n");
         printf("\t      smootherFindOmega \n");
         printf("\t      minCoarseSize <d> \n");
         printf("\t      Pweight <f> \n");
         printf("\t      SPLevel <d> \n");
         printf("\t      scalar\n");
         printf("\t      nodeDOF <d> \n");
         printf("\t      nullSpaceDim <d> \n");
         printf("\t      useNodalCoord <on,off> \n");
         printf("\t      saAMGCalibrationSize <d> \n");
         printf("\t      rsAMGSymmetric <d> \n");
         printf("\t      rsAMGInjectionForR\n");
         printf("\t      printNullSpace\n");
         printf("\t      printElemNodeList \n");
         printf("\t      printNodalCoord \n");
         printf("\t      paramFile <s> \n");
         printf("\t      numSmoothVecs <d> \n");
         printf("\t      smoothVecSteps <d> \n");
         printf("\t      arpackTol <f> \n");
      }
   }
   else if ( !strcmp(param2, "outputLevel") )
   {
      sscanf(paramString,"%s %s %d",param1,param2,&(mli_object->outputLevel_));
   }
   else if ( !strcmp(param2, "numLevels") )
   {
      sscanf(paramString,"%s %s %d", param1, param2, &(mli_object->nLevels_));
      if ( mli_object->nLevels_ <= 0 ) mli_object->nLevels_ = 1;
   }
   else if ( !strcmp(param2, "maxIterations") )
   {
      sscanf(paramString,"%s %s %d", param1, param2,
             &(mli_object->maxIterations_));
      if ( mli_object->maxIterations_ <= 0 ) mli_object->maxIterations_ = 1;
   }
   else if ( !strcmp(param2, "cycleType") )
   {
      sscanf(paramString,"%s %s %s", param1, param2, param3);
      if ( ! strcmp( param3, "V" ) )      mli_object->cycleType_ = 1;
      else if ( ! strcmp( param3, "W" ) ) mli_object->cycleType_ = 2;
   }
   else if ( !strcmp(param2, "strengthThreshold") )
   {
      sscanf(paramString,"%s %s %lg", param1, param2,
             &(mli_object->strengthThreshold_));
      if ( mli_object->strengthThreshold_ < 0.0 )
         mli_object->strengthThreshold_ = 0.0;
   }
   else if ( !strcmp(param2, "method") )
   {
      sscanf(paramString,"%s %s %s", param1, param2, param3);
      strcpy( mli_object->method_, param3 );
   }
   else if ( !strcmp(param2, "coarsenScheme") )
   {
      sscanf(paramString,"%s %s %s", param1, param2, param3);
      strcpy( mli_object->coarsenScheme_, param3 );
   }
   else if ( !strcmp(param2, "smoother") )
   {
      sscanf(paramString,"%s %s %s", param1, param2, param3);
      strcpy( mli_object->preSmoother_, param3 );
      strcpy( mli_object->postSmoother_, param3 );
   }
   else if ( !strcmp(param2, "coarseSolver") )
   {
      sscanf(paramString,"%s %s %s", param1, param2, param3);
      strcpy( mli_object->coarseSolver_, param3 );
   }
   else if ( !strcmp(param2, "coarseSolverNumSweeps") )
   {
      sscanf(paramString,"%s %s %d", param1, param2,
             &(mli_object->coarseSolverNSweeps_));
      if ( mli_object->coarseSolverNSweeps_ < 1 )
         mli_object->coarseSolverNSweeps_ = 1;
   }
   else if ( !strcmp(param2, "numSweeps") )
   {
      sscanf(paramString,"%s %s %d",param1,param2,&(mli_object->preNSweeps_));
      if ( mli_object->preNSweeps_ <= 0 ) mli_object->preNSweeps_ = 1;
      mli_object->postNSweeps_ = mli_object->preNSweeps_; 
      if ( mli_object->preSmootherWts_ != NULL )
      {
         weight = mli_object->preSmootherWts_[0];
         delete [] mli_object->preSmootherWts_;
         mli_object->preSmootherWts_ = new double[mli_object->preNSweeps_];
         for ( i = 0; i < mli_object->preNSweeps_; i++ )
            mli_object->preSmootherWts_[i] = weight;
      }
      if ( mli_object->postSmootherWts_ != NULL )
      {
         weight = mli_object->postSmootherWts_[0];
         delete [] mli_object->postSmootherWts_;
         mli_object->postSmootherWts_ = new double[mli_object->postNSweeps_];
         for ( i = 0; i < mli_object->postNSweeps_; i++ )
            mli_object->postSmootherWts_[i] = weight;
      }
   }
   else if ( !strcmp(param2, "smootherWeight") )
   {
      sscanf(paramString,"%s %s %lg",param1,param2,&weight);
      if ( weight < 0.0 || weight > 2.0 ) weight = 1.0;
      if ( mli_object->preNSweeps_ > 0 )
      {
         if ( mli_object->preSmootherWts_ != NULL )
            delete [] mli_object->preSmootherWts_;
         mli_object->preSmootherWts_ = new double[mli_object->preNSweeps_];
         for ( i = 0; i < mli_object->preNSweeps_; i++ )
            mli_object->preSmootherWts_[i] = weight;
         mli_object->postNSweeps_ = mli_object->preNSweeps_;
         if ( mli_object->postSmootherWts_ != NULL )
            delete [] mli_object->postSmootherWts_;
         mli_object->postSmootherWts_ = new double[mli_object->preNSweeps_];
         for ( i = 0; i < mli_object->preNSweeps_; i++ )
            mli_object->postSmootherWts_[i] = weight;
      }
   }
   else if ( !strcmp(param2, "smootherPrintRNorm") )
   {
      mli_object->smootherPrintRNorm_ = 1;
   }
   else if ( !strcmp(param2, "smootherFindOmega") )
   {
      mli_object->smootherFindOmega_ = 1;
   }
   else if ( !strcmp(param2, "minCoarseSize") )
   {
      sscanf(paramString,"%s %s %d",param1,param2,
             &(mli_object->minCoarseSize_));
      if ( mli_object->minCoarseSize_ <= 0 ) mli_object->minCoarseSize_ = 20;
   }
   else if ( !strcmp(param2, "Pweight") )
   {
      sscanf(paramString,"%s %s %lg",param1,param2, &(mli_object->Pweight_));
      if ( mli_object->Pweight_ < 0. ) mli_object->Pweight_ = 1.333;
   }
   else if ( !strcmp(param2, "SPLevel") )
   {
      sscanf(paramString,"%s %s %d",param1,param2, &(mli_object->SPLevel_));
      if ( mli_object->SPLevel_ < 0 ) mli_object->SPLevel_ = 0;
   }
   else if ( !strcmp(param2, "scalar") )
   {
      mli_object->scalar_ = 1;
   }
   else if ( !strcmp(param2, "nodeDOF") )
   {
      sscanf(paramString,"%s %s %d",param1,param2, &(mli_object->nodeDOF_));
      if ( mli_object->nodeDOF_ <= 0 ) mli_object->nodeDOF_ = 1;
   }
   else if ( !strcmp(param2, "nullSpaceDim") )
   {
      sscanf(paramString,"%s %s %d",param1,param2, &(mli_object->nSpaceDim_));
      if ( mli_object->nSpaceDim_ <= 0 ) mli_object->nSpaceDim_ = 1;
   }
   else if ( !strcmp(param2, "useNodalCoord") )
   {
      sscanf(paramString,"%s %s %s",param1,param2,param3);
      if ( !strcmp(param3, "on") ) mli_object->nCoordAccept_ = 1;
      else                             mli_object->nCoordAccept_ = 0;
   }
   else if ( !strcmp(param2, "saAMGCalibrationSize") )
   {
      sscanf(paramString,"%s %s %d",param1,param2,
             &(mli_object->calibrationSize_));
      if (mli_object->calibrationSize_ < 0) mli_object->calibrationSize_ = 0; 
   }
   else if ( !strcmp(param2, "rsAMGSymmetric") )
   {
      sscanf(paramString,"%s %s %d",param1,param2, &(mli_object->symmetric_));
      if ( mli_object->symmetric_ < 0 ) mli_object->symmetric_ = 0; 
      if ( mli_object->symmetric_ > 1 ) mli_object->symmetric_ = 1; 
   }
   else if ( !strcmp(param2, "rsAMGInjectionForR") )
   {
      mli_object->injectionForR_ = 1;
   }
   else if ( !strcmp(param2, "printNullSpace") )
   {
      mli_object->printNullSpace_ |= 1;
   }
   else if ( !strcmp(param2, "printElemNodeList") )
   {
      mli_object->printNullSpace_ |= 2;
   }
   else if ( !strcmp(param2, "printNodalCoord") )
   {
      mli_object->printNullSpace_ |= 4;
   }
   else if ( !strcmp(param2, "paramFile") )
   {
      sscanf(paramString,"%s %s %s",param1,param2,mli_object->paramFile_);
   }
   else if ( !strcmp(param2, "numSmoothVecs") )
   {
      sscanf(paramString,"%s %s %d",param1,param2,
             &(mli_object->numSmoothVecs_));
      if ( mli_object->numSmoothVecs_ < 0 ) mli_object->numSmoothVecs_ = 0; 
   }
   else if ( !strcmp(param2, "smoothVecSteps") )
   {
      sscanf(paramString,"%s %s %d",param1,param2,
             &(mli_object->smoothVecSteps_));
      if ( mli_object->smoothVecSteps_ < 0 ) mli_object->smoothVecSteps_ = 0; 
   }
   else if ( !strcmp(param2, "arpackTol") )
   {
      sscanf(paramString,"%s %s %lg",param1,param2,
             &(mli_object->arpackTol_));
      if ( mli_object->arpackTol_ <= 0.0 ) mli_object->arpackTol_ = 0.0; 
   }
   else if ( !strcmp(param2, "incrNullSpaceDim") )
   {
      sscanf(paramString,"%s %s %d",param1,param2, &i);
      mli_object->nSpaceDim_ += i;
   }
   else 
   {
      if ( mypid == 0 )
      {
         printf("%4d : HYPRE_LSI_MLISetParams ERROR : unrecognized request.\n",
                mypid);
         printf("\t    offending request = %s.\n", paramString);
         printf("\tAvailable options for MLI are : \n");
         printf("\t      outputLevel <d> \n");
         printf("\t      numLevels <d> \n");
         printf("\t      maxIterations <d> \n");
         printf("\t      cycleType <'V','W'> \n");
         printf("\t      strengthThreshold <f> \n");
         printf("\t      method <AMGSA, AMGSAe> \n");
         printf("\t      smoother <Jacobi,GS,...> \n");
         printf("\t      coarseSolver <Jacobi,GS,...> \n");
         printf("\t      numSweeps <d> \n");
         printf("\t      smootherWeight <f> \n");
         printf("\t      smootherPrintRNorm\n");
         printf("\t      smootherFindOmega\n");
         printf("\t      minCoarseSize <d> \n");
         printf("\t      Pweight <f> \n");
         printf("\t      SPLevel <d> \n");
         printf("\t      nodeDOF <d> \n");
         printf("\t      nullSpaceDim <d> \n");
         printf("\t      useNodalCoord <on,off> \n");
         printf("\t      saAMGCalibrationSize <d> \n"); 
         printf("\t      rsAMGSymmetric <d> \n"); 
         printf("\t      rsAMGInjectionForR\n"); 
         printf("\t      printNullSpace\n");
         printf("\t      printElemNodeList\n");
         printf("\t      printNodalCoord\n");
         printf("\t      paramFile <s> \n");
         printf("\t      numSmoothVecs <d> \n");
         printf("\t      smoothVecSteps <d> \n");
         printf("\t      arpackTol <f> \n");
         exit(1);
      }
   }
   return 0;
}

/****************************************************************************/
/* HYPRE_LSI_MLICreateNodeEqnMap                                            */
/*--------------------------------------------------------------------------*/

extern "C"
int HYPRE_LSI_MLICreateNodeEqnMap(HYPRE_Solver solver, int nNodes, 
                                  int *nodeNumbers, int *eqnNumbers,
                                  int *procNRows)
{
#ifdef HAVE_MLI
   int           iN, iP, mypid, nprocs, *procMapArray, *iTempArray;
   int           iS, nSends, *sendLengs, *sendProcs, **iSendBufs;
   int           iR, nRecvs, *recvLengs, *recvProcs, **iRecvBufs, *procList;
   int           newNumNodes, *newNodeNumbers, *newEqnNumbers, procIndex;
   MPI_Comm      mpiComm;
   MPI_Request   *mpiRequests;
   MPI_Status    mpiStatus;
   MLI_Mapper    *mapper;
   HYPRE_LSI_MLI *mli_object = (HYPRE_LSI_MLI *) solver;

   /* -------------------------------------------------------- */ 
   /* fetch processor information                              */
   /* -------------------------------------------------------- */ 

   if ( mli_object == NULL ) return 1;
   if ( mli_object->mapper_ != NULL ) delete mli_object->mapper_;
   mpiComm = mli_object->mpiComm_;
   MPI_Comm_rank( mpiComm, &mypid );
   MPI_Comm_size( mpiComm, &nprocs );

   /* -------------------------------------------------------- */ 
   /* construct node to processor map array                    */
   /* -------------------------------------------------------- */ 

   procMapArray = new int[nNodes]; 
   for ( iN = 0; iN < nNodes; iN++ )
   {
      procMapArray[iN] = -1;
      if ( eqnNumbers[iN] < procNRows[mypid] || 
           eqnNumbers[iN] >= procNRows[mypid+1] )
      {
         for ( iP = 0; iP < nprocs; iP++ )
            if ( eqnNumbers[iN] < procNRows[iP] ) break;
         procMapArray[iN] = iP - 1;
      }
   } 

   /* -------------------------------------------------------- */ 
   /* construct send information                               */
   /* -------------------------------------------------------- */ 

   procList = new int[nprocs];
   for ( iP = 0; iP < nprocs; iP++ ) procList[iP] = 0;
   for ( iN = 0; iN < nNodes; iN++ )
      if ( procMapArray[iN] >= 0 ) procList[procMapArray[iN]]++;
   nSends = 0;
   for ( iP = 0; iP < nprocs; iP++ ) if ( procList[iP] > 0 ) nSends++;
   if ( nSends > 0 )
   {
      sendProcs = new int[nSends]; 
      sendLengs = new int[nSends]; 
      iSendBufs = new int*[nSends]; 
   }
   nSends = 0;
   for ( iP = 0; iP < nprocs; iP++ ) 
   {
      if ( procList[iP] > 0 ) 
      {
         sendLengs[nSends] = procList[iP];
         sendProcs[nSends++] = iP;
      }
   }

   /* -------------------------------------------------------- */ 
   /* construct recv information                               */
   /* -------------------------------------------------------- */ 

   for ( iP = 0; iP < nprocs; iP++ ) procList[iP] = 0;
   for ( iP = 0; iP < nSends; iP++ ) procList[sendProcs[iP]]++; 
   iTempArray = new int[nprocs]; 
   MPI_Allreduce(procList,iTempArray,nprocs,MPI_INT,MPI_SUM,mpiComm);
   nRecvs = iTempArray[mypid];
   delete [] procList;
   delete [] iTempArray;
   if ( nRecvs > 0 )
   {
      recvLengs = new int[nRecvs];
      recvProcs = new int[nRecvs];
      iRecvBufs = new int*[nRecvs];
      mpiRequests = new MPI_Request[nRecvs];
   }
   for ( iP = 0; iP < nRecvs; iP++ ) 
      MPI_Irecv(&(recvLengs[iP]), 1, MPI_INT, MPI_ANY_SOURCE, 29421, 
                mpiComm, &(mpiRequests[iP]));
   for ( iP = 0; iP < nSends; iP++ ) 
      MPI_Send(&(sendLengs[iP]), 1, MPI_INT, sendProcs[iP], 29421, mpiComm);
   for ( iP = 0; iP < nRecvs; iP++ ) 
   {
      MPI_Wait(&(mpiRequests[iP]), &mpiStatus);
      recvProcs[iP] = mpiStatus.MPI_SOURCE;
   }

   /* -------------------------------------------------------- */ 
   /* communicate node and equation information                */
   /* -------------------------------------------------------- */ 

   for ( iP = 0; iP < nRecvs; iP++ ) 
   {
      iRecvBufs[iP] = new int[recvLengs[iP]*2];
      MPI_Irecv(iRecvBufs[iP], recvLengs[iP]*2, MPI_INT, recvProcs[iP], 
                29422, mpiComm, &(mpiRequests[iP]));
   }
   for ( iP = 0; iP < nSends; iP++ ) 
   {
      iSendBufs[iP] = new int[sendLengs[iP]*2];
      sendLengs[iP] = 0;
   }
   for ( iN = 0; iN < nNodes; iN++ )
   {
      if ( procMapArray[iN] >= 0 ) 
      {
         procIndex = procMapArray[iN];
         for ( iP = 0; iP < nSends; iP++ ) 
            if ( sendProcs[iP] == procIndex ) break;
         iSendBufs[iP][sendLengs[iP]++] = nodeNumbers[iN];
         iSendBufs[iP][sendLengs[iP]++] = eqnNumbers[iN];
      }
   }
   for ( iP = 0; iP < nSends; iP++ ) 
   {
      sendLengs[iP] /= 2;
      MPI_Send(iSendBufs[iP], sendLengs[iP]*2, MPI_INT, sendProcs[iP], 
               29422, mpiComm);
   }
   for ( iP = 0; iP < nRecvs; iP++ ) MPI_Wait(&(mpiRequests[iP]), &mpiStatus);

   /* -------------------------------------------------------- */ 
   /* form node and equation map                               */
   /* -------------------------------------------------------- */ 

   newNumNodes = nNodes;
   for ( iP = 0; iP < nRecvs; iP++ ) newNumNodes += recvLengs[iP];
   newNodeNumbers = new int[newNumNodes];
   newEqnNumbers  = new int[newNumNodes];
   newNumNodes = 0;
   for (iN = 0; iN < nNodes; iN++) 
   {
      newNodeNumbers[newNumNodes]  = nodeNumbers[iN];
      newEqnNumbers[newNumNodes++] = eqnNumbers[iN];
   }
   for ( iP = 0; iP < nRecvs; iP++ ) 
   {
      for ( iR = 0; iR < recvLengs[iP]; iR++ ) 
      {
         newNodeNumbers[newNumNodes]  = iRecvBufs[iP][iR*2];
         newEqnNumbers[newNumNodes++] = iRecvBufs[iP][iR*2+1];
      }
   }
   mapper = new MLI_Mapper();
   mapper->setMap( newNumNodes, newNodeNumbers, newEqnNumbers );
   mli_object->mapper_ = mapper;

   /* -------------------------------------------------------- */ 
   /* clean up and return                                      */
   /* -------------------------------------------------------- */ 

   delete [] procMapArray; 
   if ( nSends > 0 )
   {
      delete [] sendProcs;
      delete [] sendLengs;
      for ( iS = 0; iS < nSends; iS++ ) delete [] iSendBufs[iS];
      delete [] iSendBufs;
   } 
   if ( nRecvs > 0 )
   {
      delete [] recvProcs;
      delete [] recvLengs;
      for ( iR = 0; iR < nRecvs; iR++ ) delete [] iRecvBufs[iR];
      delete [] iRecvBufs;
      delete [] mpiRequests;
   } 
   delete [] newNodeNumbers;
   delete [] newEqnNumbers;
   return 0;
#else
   (void) solver;
   (void) nNodes;
   (void) nodeNumbers;
   (void) eqnNumbers;
   (void) procNRows;
   return 1;
#endif
}

/****************************************************************************/
/* HYPRE_LSI_MLIAdjustNodeEqnMap                                            */
/*--------------------------------------------------------------------------*/

extern "C"
int HYPRE_LSI_MLIAdjustNodeEqnMap(HYPRE_Solver solver, int *procNRows,
                                  int *procOffsets)
{
#ifdef HAVE_MLI
   HYPRE_LSI_MLI *mli_object = (HYPRE_LSI_MLI *) solver;
   if ( mli_object == NULL ) return 1;
   if ( mli_object->mapper_ == NULL ) return 1;
   mli_object->mapper_->adjustMapOffset( mli_object->mpiComm_, procNRows, 
                                         procOffsets );
   return 0;
#else
   (void) solver;
   (void) procNRows;
   (void) procOffsets;
   return 1;
#endif
}

/****************************************************************************/
/* HYPRE_LSI_MLIAdjustNullSpace                                             */
/*--------------------------------------------------------------------------*/

extern "C"
int HYPRE_LSI_MLIAdjustNullSpace(HYPRE_Solver solver, int nConstraints,
                                 int *slaveIndices, 
                                 HYPRE_ParCSRMatrix hypreA)
{
#ifdef HAVE_MLI
   HYPRE_LSI_MLI *mli_object = (HYPRE_LSI_MLI *) solver;
   if ( mli_object == NULL ) return 1;
   mli_object->adjustNullSpace_ = 1;
   mli_object->numResetNull_    = nConstraints;
   if ( nConstraints > 0 )
      mli_object->resetNullIndices_ = new int[nConstraints];
   for ( int i = 0; i < nConstraints; i++ ) 
      mli_object->resetNullIndices_[i] = slaveIndices[i];
   mli_object->correctionMatrix_ = hypreA;
   return 0;
#else
   (void) solver;
   (void) nConstraints;
   (void) slaveIndices;
   (void) hypreA;
   return 1;
#endif
}

/****************************************************************************/
/* HYPRE_LSI_MLISetFEData                                                   */
/*--------------------------------------------------------------------------*/

extern "C"
int HYPRE_LSI_MLISetFEData(HYPRE_Solver solver, void *object)
{
#ifdef HAVE_MLI
   HYPRE_LSI_MLI *mli_object = (HYPRE_LSI_MLI *) solver;
   HYPRE_MLI_FEData *hypre_fedata = (HYPRE_MLI_FEData *) object;
   mli_object->feData_      = hypre_fedata->fedata_; 
   hypre_fedata->fedata_    = NULL; 
   hypre_fedata->fedataOwn_ = 0;
   return 0;
#else
   (void) solver;
   (void) object;
   return 1;
#endif
}

/****************************************************************************/
/* HYPRE_LSI_MLISetSFEI                                                   */
/*--------------------------------------------------------------------------*/

extern "C"
int HYPRE_LSI_MLISetSFEI(HYPRE_Solver solver, void *object)
{
#ifdef HAVE_MLI
   HYPRE_LSI_MLI *mli_object = (HYPRE_LSI_MLI *) solver;
   HYPRE_MLI_SFEI *hypre_sfei = (HYPRE_MLI_SFEI *) object;
   mli_object->sfei_    = hypre_sfei->sfei_; 
   hypre_sfei->sfei_    = NULL; 
   hypre_sfei->sfeiOwn_ = 0;
   return 0;
#else
   (void) solver;
   (void) object;
   return 1;
#endif
}

/****************************************************************************/
/* HYPRE_LSI_MLILoadNodalCoordinates                                        */
/* (The nodal coordinates loaded in here conforms to the nodal labeling in  */
/* FEI, so the lookup object can be used to find the equation number.       */
/* In addition, node numbers and coordinates need to be shuffled between    */
/* processors)                                                              */
/*--------------------------------------------------------------------------*/

/*
#define FEI_250
*/

extern "C"
int HYPRE_LSI_MLILoadNodalCoordinates(HYPRE_Solver solver, int nNodes,
              int nodeDOF, int *eqnNumbers, int nDim, double *coords,
              int localNRows)
{
   int           iN, iD, eqnInd, mypid, *flags, arrayLeng;
   double        *nCoords;
   MPI_Comm      mpiComm;
#ifdef FEI_250
   int           iMax, iMin, offFlag;
#else
   int           iP, nprocs, *nodeProcMap, *iTempArray;
   int           iS, nSends, *sendLengs, *sendProcs, **iSendBufs, procIndex;
   int           iR, nRecvs, *recvLengs, *recvProcs, **iRecvBufs, *procList;
   int           *procNRows, numNodes, coordLength;
   double        **dSendBufs, **dRecvBufs;
   MPI_Request   *mpiRequests;
   MPI_Status    mpiStatus;
#endif
   HYPRE_LSI_MLI *mli_object = (HYPRE_LSI_MLI *) solver;

   /* -------------------------------------------------------- */ 
   /* if nodal coordinates flag off, do not take coordinates   */
   /* -------------------------------------------------------- */ 

   if ( ! mli_object->nCoordAccept_ ) return 1;
   mpiComm = mli_object->mpiComm_;
   MPI_Comm_rank( mpiComm, &mypid );

   /* -------------------------------------------------------- */ 
   /* clean up previously allocated space                      */
   /* -------------------------------------------------------- */ 

   if ( mli_object->nCoordinates_ != NULL )
      delete [] mli_object->nCoordinates_;
   if ( mli_object->nullScales_ != NULL )
      delete [] mli_object->nullScales_; 
   mli_object->nCoordinates_ = NULL;
   mli_object->nullScales_   = NULL;

   /* -------------------------------------------------------- */ 
   /* This code is used in place of the 'else' block in view   */
   /* of the changes made to FEI 2.5.0                         */
   /* -------------------------------------------------------- */ 

#ifdef FEI_250
   mli_object->spaceDim_ = nDim;
   mli_object->nodeDOF_  = nodeDOF;
   iMin = 1000000000; iMax = 0;
   for ( iN = 0; iN < nNodes; iN++ )  
   {
      iMin = ( eqnNumbers[iN] < iMin ) ? eqnNumbers[iN] : iMin;
      iMax = ( eqnNumbers[iN] > iMax ) ? eqnNumbers[iN] : iMax;
   }
   mli_object->localNEqns_ = ( iMax/nDim - iMin/nDim + 1 ) * nDim;
   offFlag = 0;
   if ( mli_object->localNEqns_ != nNodes*nDim ) offFlag = 1;
   iN = offFlag;
   MPI_Allreduce(&iN,&offFlag,1,MPI_INT,MPI_SUM,mpiComm);
   if ( offFlag != 0 )
   {
      if ( mypid == 0 )
         printf("HYPRE_LSI_MLILoadNodalCoordinates - turned off.\n");
      mli_object->nCoordAccept_ = 0;
      mli_object->localNEqns_ = 0;
      return 1;
   }
   mli_object->nCoordinates_ = new double[nNodes*nDim];
   nCoords                   = mli_object->nCoordinates_;
   for ( iN = 0; iN < nNodes; iN++ )  
   {
      eqnInd = (eqnNumbers[iN] - iMin) / nodeDOF;
      for ( iD = 0; iD < nDim; iD++ )  
         nCoords[eqnInd*nDim+iD] = coords[iN*nDim+iD]; 
   }

#else
   /* -------------------------------------------------------- */ 
   /* fetch machine information                                */
   /* -------------------------------------------------------- */ 

   MPI_Comm_size( mpiComm, &nprocs );

   /* -------------------------------------------------------- */
   /* construct procNRows array                                */
   /* -------------------------------------------------------- */

   procNRows = new int[nprocs+1];
   iTempArray = new int[nprocs];
   for ( iP = 0; iP <= nprocs; iP++ ) procNRows[iP] = 0;
   procNRows[mypid] = localNRows;
   MPI_Allreduce(procNRows,iTempArray,nprocs,MPI_INT,MPI_SUM,mpiComm);
   procNRows[0] = 0;
   for ( iP = 1; iP <= nprocs; iP++ ) 
      procNRows[iP] = procNRows[iP-1] + iTempArray[iP-1];

   /* -------------------------------------------------------- */
   /* construct node to processor map                          */
   /* -------------------------------------------------------- */

   nodeProcMap = new int[nNodes];
   for ( iN = 0; iN < nNodes; iN++ )
   {
      nodeProcMap[iN] = -1;
      if ( eqnNumbers[iN] < procNRows[mypid] ||
           eqnNumbers[iN] >= procNRows[mypid+1] )
      {
         for ( iP = 0; iP < nprocs; iP++ )
            if ( eqnNumbers[iN] < procNRows[iP] ) break;
         nodeProcMap[iN] = iP - 1;
      }
   }

   /* -------------------------------------------------------- */
   /* construct send information                               */
   /* -------------------------------------------------------- */

   procList = new int[nprocs];
   for ( iP = 0; iP < nprocs; iP++ ) procList[iP] = 0;
   for ( iN = 0; iN < nNodes; iN++ )
      if ( nodeProcMap[iN] >= 0 ) procList[nodeProcMap[iN]]++;
   nSends = 0;
   for ( iP = 0; iP < nprocs; iP++ ) if ( procList[iP] > 0 ) nSends++;
   if ( nSends > 0 )
   {
      sendProcs = new int[nSends];
      sendLengs = new int[nSends];
      iSendBufs = new int*[nSends];
      dSendBufs = new double*[nSends];
   }
   nSends = 0;
   for ( iP = 0; iP < nprocs; iP++ )
   {
      if ( procList[iP] > 0 )
      {
         sendLengs[nSends] = procList[iP];
         sendProcs[nSends++] = iP;
      }
   }

   /* -------------------------------------------------------- */
   /* construct recv information                               */
   /* -------------------------------------------------------- */

   for ( iP = 0; iP < nprocs; iP++ ) procList[iP] = 0;
   for ( iP = 0; iP < nSends; iP++ ) procList[sendProcs[iP]]++;
   MPI_Allreduce(procList,iTempArray,nprocs,MPI_INT,MPI_SUM,mpiComm);
   nRecvs = iTempArray[mypid];
   if ( nRecvs > 0 )
   {
      recvLengs = new int[nRecvs];
      recvProcs = new int[nRecvs];
      iRecvBufs = new int*[nRecvs];
      dRecvBufs = new double*[nRecvs];
      mpiRequests = new MPI_Request[nRecvs];
   }
   for ( iP = 0; iP < nRecvs; iP++ )
      MPI_Irecv(&(recvLengs[iP]), 1, MPI_INT, MPI_ANY_SOURCE, 29421,
                mpiComm, &(mpiRequests[iP]));
   for ( iP = 0; iP < nSends; iP++ )
      MPI_Send(&(sendLengs[iP]), 1, MPI_INT, sendProcs[iP], 29421, mpiComm);
   for ( iP = 0; iP < nRecvs; iP++ )
   {
      MPI_Wait(&(mpiRequests[iP]), &mpiStatus);
      recvProcs[iP] = mpiStatus.MPI_SOURCE;
   }

   /* -------------------------------------------------------- */
   /* communicate equation numbers information                */
   /* -------------------------------------------------------- */

   for ( iP = 0; iP < nRecvs; iP++ )
   {
      iRecvBufs[iP] = new int[recvLengs[iP]];
      MPI_Irecv(iRecvBufs[iP], recvLengs[iP], MPI_INT, recvProcs[iP],
                29422, mpiComm, &(mpiRequests[iP]));
   }
   for ( iP = 0; iP < nSends; iP++ )
   {
      iSendBufs[iP] = new int[sendLengs[iP]];
      sendLengs[iP] = 0;
   }
   for ( iN = 0; iN < nNodes; iN++ )
   {
      if ( nodeProcMap[iN] >= 0 )
      {
         procIndex = nodeProcMap[iN];
         for ( iP = 0; iP < nSends; iP++ )
            if ( procIndex == sendProcs[iP] ) break;
         iSendBufs[iP][sendLengs[iP]++] = eqnNumbers[iN];
      }
   }
   for ( iP = 0; iP < nSends; iP++ )
   {
      MPI_Send(iSendBufs[iP], sendLengs[iP], MPI_INT, sendProcs[iP],
               29422, mpiComm);
   }
   for ( iP = 0; iP < nRecvs; iP++ ) MPI_Wait(&(mpiRequests[iP]), &mpiStatus);

   /* -------------------------------------------------------- */
   /* communicate coordinate information                       */
   /* -------------------------------------------------------- */

   for ( iP = 0; iP < nRecvs; iP++ )
   {
      dRecvBufs[iP] = new double[recvLengs[iP]*nDim];
      MPI_Irecv(dRecvBufs[iP], recvLengs[iP]*nDim, MPI_DOUBLE, 
                recvProcs[iP], 29425, mpiComm, &(mpiRequests[iP]));
   }
   for ( iP = 0; iP < nSends; iP++ )
   {
      dSendBufs[iP] = new double[sendLengs[iP]*nDim];
      sendLengs[iP] = 0;
   }
   for ( iN = 0; iN < nNodes; iN++ )
   {
      if ( nodeProcMap[iN] >= 0 )
      {
         procIndex = nodeProcMap[iN];
         for ( iP = 0; iP < nSends; iP++ )
            if ( procIndex == sendProcs[iP] ) break;
         for ( iD = 0; iD < nDim; iD++ )
            dSendBufs[iP][sendLengs[iP]++]=coords[iN*nDim+iD];
      }
   }
   for ( iP = 0; iP < nSends; iP++ )
   {
      sendLengs[iP] /= nDim;
      MPI_Send(dSendBufs[iP], sendLengs[iP]*nDim, MPI_DOUBLE, 
               sendProcs[iP], 29425, mpiComm);
   }
   for ( iP = 0; iP < nRecvs; iP++ ) MPI_Wait(&(mpiRequests[iP]), &mpiStatus);

   /* -------------------------------------------------------- */
   /* check any duplicate coordinate information               */
   /* -------------------------------------------------------- */

   arrayLeng = nNodes;
   for ( iP = 0; iP < nRecvs; iP++ ) arrayLeng += recvLengs[iP];
   flags = new int[arrayLeng];
   for ( iN = 0; iN < arrayLeng; iN++ ) flags[iN] = 0;
   for ( iN = 0; iN < nNodes; iN++ )  
   {
      if ( nodeProcMap[iN] < 0 ) 
      {
         eqnInd = (eqnNumbers[iN] - procNRows[mypid]) / nodeDOF;
         if ( eqnInd >= arrayLeng )
         {
            printf("%d : HYPRE_LSI_MLILoadNodalCoordinates - ERROR(1).\n",
                   mypid);
            exit(1);
         }
         flags[eqnInd] = 1;
      }
   }
   for ( iP = 0; iP < nRecvs; iP++ )
   {
      for ( iR = 0; iR < recvLengs[iP]; iR++ )
      {
         eqnInd = (iRecvBufs[iP][iR] - procNRows[mypid]) / nodeDOF;
         if ( eqnInd >= arrayLeng )
         {
            printf("%d : HYPRE_LSI_MLILoadNodalCoordinates - ERROR(2).\n",
                   mypid);
            exit(1);
         }
         flags[eqnInd] = 1;
      }
   }
   numNodes = 0;
   for ( iN = 0; iN < arrayLeng; iN++ ) 
      if ( flags[iN] == 0 ) break;
      else                  numNodes++;
   delete [] flags;

   /* -------------------------------------------------------- */
   /* set up nodal coordinate information in correct order     */
   /* -------------------------------------------------------- */

   mli_object->spaceDim_     = nDim;
   mli_object->nodeDOF_      = nodeDOF;
   mli_object->localNEqns_   = numNodes * nodeDOF;
   coordLength               = numNodes * nodeDOF * nDim;
   mli_object->nCoordinates_ = new double[coordLength];
   nCoords                   = mli_object->nCoordinates_;

   arrayLeng = numNodes * nodeDOF;
   for ( iN = 0; iN < nNodes; iN++ )  
   {
      if ( nodeProcMap[iN] < 0 ) 
      {
         eqnInd = (eqnNumbers[iN] - procNRows[mypid]) / nodeDOF;
         if ( eqnInd >= 0 && eqnInd < arrayLeng ) 
            for ( iD = 0; iD < nDim; iD++ )  
               nCoords[eqnInd*nDim+iD] = coords[iN*nDim+iD]; 
      }
   }
   for ( iP = 0; iP < nRecvs; iP++ )
   {
      for ( iR = 0; iR < recvLengs[iP]; iR++ )
      {
         eqnInd = (iRecvBufs[iP][iR] - procNRows[mypid]) / nodeDOF;
         if ( eqnInd >= 0 && eqnInd < arrayLeng ) 
            for ( iD = 0; iD < nDim; iD++ )  
               nCoords[eqnInd*nDim+iD] = dRecvBufs[iP][iR*nDim+iD];
      }
   }
   for ( iN = 0; iN < numNodes*nodeDOF; iN++ )  
      if (nCoords[iN] == -99999.0) printf("%d : LSI_mli error %d\n",mypid,iN);

   /* -------------------------------------------------------- */
   /* clean up                                                 */
   /* -------------------------------------------------------- */

   delete [] procList;
   delete [] iTempArray;
   delete [] nodeProcMap;
   delete [] procNRows;
   if ( nSends > 0 )
   {
      delete [] sendProcs;
      delete [] sendLengs;
      for ( iS = 0; iS < nSends; iS++ ) delete [] iSendBufs[iS];
      for ( iS = 0; iS < nSends; iS++ ) delete [] dSendBufs[iS];
      delete [] dSendBufs;
      delete [] iSendBufs;
   }
   if ( nRecvs > 0 )
   {
      delete [] recvProcs;
      delete [] recvLengs;
      for ( iR = 0; iR < nRecvs; iR++ ) delete [] iRecvBufs[iR];
      for ( iR = 0; iR < nRecvs; iR++ ) delete [] dRecvBufs[iR];
      delete [] iRecvBufs;
      delete [] dRecvBufs;
      delete [] mpiRequests;
   }
#endif
   return 0;
} 

/****************************************************************************/
/* HYPRE_LSI_MLILoadMatrixScalings                                          */
/*--------------------------------------------------------------------------*/

extern "C"
int HYPRE_LSI_MLILoadMatrixScalings(HYPRE_Solver solver, int nEqns,
                                    double *scalings)
{
   HYPRE_LSI_MLI *mli_object = (HYPRE_LSI_MLI *) solver;
   if ( scalings != NULL )
   {
      mli_object->nullScales_ = new double[nEqns];
      for ( int i = 0; i < nEqns; i++ )  
         mli_object->nullScales_[i] = scalings[i];
   }
   return 0;
}

/****************************************************************************/
/* HYPRE_LSI_MLILoadMaterialLabels                                          */
/*--------------------------------------------------------------------------*/

extern "C"
int HYPRE_LSI_MLILoadMaterialLabels(HYPRE_Solver solver, int nLabels,
                                    int *labels)
{
   HYPRE_LSI_MLI *mli_object = (HYPRE_LSI_MLI *) solver;
   if ( labels != NULL )
   {
      mli_object->matLabels_ = new int[nLabels];
      for ( int i = 0; i < nLabels; i++ )  
         mli_object->matLabels_[i] = labels[i];
      mli_object->numMatLabels_ = nLabels;
   }
   return 0;
}

/****************************************************************************/
/* HYPRE_LSI_MLIFEDataCreate                                                */
/*--------------------------------------------------------------------------*/

extern "C"
void *HYPRE_LSI_MLIFEDataCreate( MPI_Comm mpi_comm )
{
#ifdef HAVE_MLI
   HYPRE_MLI_FEData *hypre_fedata;
   hypre_fedata = (HYPRE_MLI_FEData *) malloc( sizeof(HYPRE_MLI_FEData) );  
   hypre_fedata->comm_          = mpi_comm;
   hypre_fedata->fedata_        = NULL;
   hypre_fedata->fedataOwn_     = 0;
   hypre_fedata->computeNull_   = 0;
   hypre_fedata->nullDim_       = 1;
   return ((void *) hypre_fedata);
#else
   return NULL;
#endif 
}

/****************************************************************************/
/* HYPRE_LSI_MLIFEDataDestroy                                               */
/*--------------------------------------------------------------------------*/

extern "C"
int HYPRE_LSI_MLIFEDataDestroy( void *object )
{
#ifdef HAVE_MLI
   HYPRE_MLI_FEData *hypre_fedata = (HYPRE_MLI_FEData *) object;
   if ( hypre_fedata == NULL ) return 1;
   if ( hypre_fedata->fedataOwn_ && hypre_fedata->fedata_ != NULL ) 
      delete hypre_fedata->fedata_;
   hypre_fedata->fedata_        = NULL;
   free( hypre_fedata );
   return 0;
#else
   return 1;
#endif 
}

/****************************************************************************/
/* HYPRE_LSI_MLIFEDataInitFields                                            */
/*--------------------------------------------------------------------------*/

extern "C"
int HYPRE_LSI_MLIFEDataInitFields( void *object, int nFields, int *fieldSizes,
                                   int *fieldIDs )
{
#ifdef HAVE_MLI
   HYPRE_MLI_FEData *hypre_fedata = (HYPRE_MLI_FEData *) object;
   MLI_FEData       *fedata;

   if ( hypre_fedata == NULL ) return 1;
   if ( hypre_fedata->fedata_ != NULL ) delete hypre_fedata->fedata_;
   hypre_fedata->fedata_    = new MLI_FEData(hypre_fedata->comm_);
   hypre_fedata->fedataOwn_ = 1;
   fedata = (MLI_FEData *) hypre_fedata->fedata_;
   fedata->initFields( nFields, fieldSizes, fieldIDs );
   return 0;
#else
   (void) object;
   (void) nFields;
   (void) fieldSizes;
   (void) fieldIDs;
   return 1;
#endif
}

/****************************************************************************/
/* HYPRE_LSI_MLIFEDataInitElemBlock                                         */
/*--------------------------------------------------------------------------*/

extern "C"
int HYPRE_LSI_MLIFEDataInitElemBlock(void *object, int nElems, 
                                     int nNodesPerElem, int numNodeFields,
                                     int *nodeFieldIDs) 
{
#ifdef HAVE_MLI
   HYPRE_MLI_FEData *hypre_fedata = (HYPRE_MLI_FEData *) object;
   MLI_FEData       *fedata;

   if ( hypre_fedata == NULL ) return 1;
   fedata = (MLI_FEData *) hypre_fedata->fedata_;
   if ( fedata == NULL ) return 1;
   if ( numNodeFields != 1 ) return 1;
   hypre_fedata->fedata_->initElemBlock(nElems,nNodesPerElem,numNodeFields,
                                        nodeFieldIDs,0,NULL);
   return 0;
#else
   (void) object;
   (void) nElems;
   (void) nNodesPerElem;
   (void) numNodeFields;
   (void) nodeFieldIDs;
   return 1;
#endif
}

/****************************************************************************/
/* HYPRE_LSI_MLIFEDataInitElemNodeList                                      */
/*--------------------------------------------------------------------------*/

extern "C"
int HYPRE_LSI_MLIFEDataInitElemNodeList( void *object, int elemID, 
                       int nNodesPerElem, int *elemNodeList)
{
#ifdef HAVE_MLI
   HYPRE_MLI_FEData *hypre_fedata = (HYPRE_MLI_FEData *) object;
   MLI_FEData       *fedata;

   if ( hypre_fedata == NULL ) return 1;
   fedata = (MLI_FEData *) hypre_fedata->fedata_;
   if ( fedata == NULL ) return 1;
   fedata->initElemNodeList(elemID,nNodesPerElem,elemNodeList,3,NULL);
   return 0;
#else
   return 1;
#endif
}

/****************************************************************************/
/* HYPRE_LSI_MLIFEDataInitSharedNodes                                       */
/*--------------------------------------------------------------------------*/

extern "C"
int HYPRE_LSI_MLIFEDataInitSharedNodes(void *object, int nSharedNodes,
                  int *sharedNodeIDs, int *sharedProcLengs,
                  int **sharedProcIDs) 
{
#ifdef HAVE_MLI
   HYPRE_MLI_FEData *hypre_fedata = (HYPRE_MLI_FEData *) object;
   MLI_FEData       *fedata;

   if ( hypre_fedata == NULL ) return 1;
   fedata = (MLI_FEData *) hypre_fedata->fedata_;
   if ( fedata == NULL ) return 1;
   if ( nSharedNodes > 0 )
      fedata->initSharedNodes(nSharedNodes, sharedNodeIDs, sharedProcLengs, 
                              sharedProcIDs);
   return 0;
#else
   (void) object;
   (void) nSharedNodes;
   (void) sharedNodeIDs;
   (void) sharedProcLengs;
   (void) sharedProcIDs;
   return 1;
#endif
}

/****************************************************************************/
/* HYPRE_LSI_MLIFEDataInitComplete                                          */
/*--------------------------------------------------------------------------*/

extern "C"
int HYPRE_LSI_MLIFEDataInitComplete( void *object )
{
#ifdef HAVE_MLI
   HYPRE_MLI_FEData *hypre_fedata = (HYPRE_MLI_FEData *) object;
   MLI_FEData       *fedata;

   if ( hypre_fedata == NULL ) return 1;
   fedata = (MLI_FEData *) hypre_fedata->fedata_;
   if ( fedata == NULL ) return 1;
   fedata->initComplete();
   return 0;
#else
   (void) object;
   return 1;
#endif
}

/****************************************************************************/
/* HYPRE_LSI_MLIFEDataLoadElemMatrix                                        */
/*--------------------------------------------------------------------------*/

extern "C"
int HYPRE_LSI_MLIFEDataLoadElemMatrix(void *object, int elemID, int nNodes,
                            int *nodeList, int matDim, double **inMat)
{
   (void) nNodes;
   (void) nodeList;
#ifdef HAVE_MLI
   int              i, j;
   double           *elemMat;
   HYPRE_MLI_FEData *hypre_fedata = (HYPRE_MLI_FEData *) object;
   MLI_FEData       *fedata;

   /* -------------------------------------------------------- */ 
   /* error checking                                           */
   /* -------------------------------------------------------- */ 

   if ( hypre_fedata == NULL ) return 1;
   fedata = (MLI_FEData *) hypre_fedata->fedata_;
   if ( fedata == NULL ) return 1;

   /* -------------------------------------------------------- */ 
   /* load the element matrix                                  */
   /* -------------------------------------------------------- */ 

   elemMat = new double[matDim*matDim];
   for ( i = 0; i < matDim; i++ )
      for ( j = 0; j < matDim; j++ )
         elemMat[i+j*matDim] = inMat[i][j];
   fedata->loadElemMatrix(elemID, matDim, elemMat);
   delete [] elemMat;
   return 0;

#else
   (void) object;
   (void) elemID;
   (void) matDim;
   (void) inMat;
   return 1;
#endif
}

/****************************************************************************/
/* HYPRE_LSI_MLIFEDataWriteToFile                                           */
/*--------------------------------------------------------------------------*/

extern "C"
int HYPRE_LSI_MLIFEDataWriteToFile( void *object, char *filename )
{
#ifdef HAVE_MLI
   MLI_FEData       *fedata;
   HYPRE_MLI_FEData *hypre_fedata = (HYPRE_MLI_FEData *) object;

   /* -------------------------------------------------------- */ 
   /* error checking                                           */
   /* -------------------------------------------------------- */ 

   if ( hypre_fedata == NULL ) return 1;
   fedata = (MLI_FEData *) hypre_fedata->fedata_;
   if ( fedata == NULL ) return 1;

   /* -------------------------------------------------------- */ 
   /* write to file                                            */
   /* -------------------------------------------------------- */ 

   fedata->writeToFile( filename );
   return 0;
#else
   (void) object;
   (void) filename;
   return 1;
#endif
}

/****************************************************************************/
/* HYPRE_LSI_MLISFEICreate                                                  */
/*--------------------------------------------------------------------------*/

extern "C"
void *HYPRE_LSI_MLISFEICreate( MPI_Comm mpiComm )
{
#ifdef HAVE_MLI
   HYPRE_MLI_SFEI *hypre_sfei;
   hypre_sfei = (HYPRE_MLI_SFEI *) malloc( sizeof(HYPRE_MLI_SFEI) );  
   hypre_sfei->comm_    = mpiComm;
   hypre_sfei->sfei_    = new MLI_SFEI(mpiComm);;
   hypre_sfei->sfeiOwn_ = 1;
   return ((void *) hypre_sfei);
#else
   return NULL;
#endif 
}

/****************************************************************************/
/* HYPRE_LSI_MLISFEIDestroy                                                 */
/*--------------------------------------------------------------------------*/

extern "C"
int HYPRE_LSI_MLISFEIDestroy( void *object )
{
#ifdef HAVE_MLI
   HYPRE_MLI_SFEI *hypre_sfei = (HYPRE_MLI_SFEI *) object;
   if ( hypre_sfei == NULL ) return 1;
   if ( hypre_sfei->sfeiOwn_ && hypre_sfei->sfei_ != NULL ) 
      delete hypre_sfei->sfei_;
   hypre_sfei->sfei_ = NULL;
   free( hypre_sfei );
   return 0;
#else
   return 1;
#endif 
}

/****************************************************************************/
/* HYPRE_LSI_MLISFEILoadElemMatrices                                      */
/*--------------------------------------------------------------------------*/

extern "C"
int HYPRE_LSI_MLISFEILoadElemMatrices(void *object, int elemBlk, int nElems,
              int *elemIDs, double ***inMat, int elemNNodes, int **nodeLists)
{
#ifdef HAVE_MLI
   HYPRE_MLI_SFEI *hypre_sfei = (HYPRE_MLI_SFEI *) object;
   MLI_SFEI       *sfei;

   /* -------------------------------------------------------- */ 
   /* error checking                                           */
   /* -------------------------------------------------------- */ 

   if ( hypre_sfei == NULL ) return 1;
   sfei = (MLI_SFEI *) hypre_sfei->sfei_;
   if ( sfei == NULL ) return 1;

   /* -------------------------------------------------------- */ 
   /* load the element matrix                                  */
   /* -------------------------------------------------------- */ 

   sfei->loadElemBlock(elemBlk,nElems,elemIDs,inMat,elemNNodes,nodeLists);
   return 0;
#else
   (void) object;
   (void) elemBlk;
   (void) nElems;
   (void) elemIDs;
   (void) inMat;
   (void) elemNNodes;
   (void) nodeLists;
   return 1;
#endif
}

/****************************************************************************/
/* HYPRE_LSI_MLISFEIAddNumElems                                             */
/*--------------------------------------------------------------------------*/

extern "C"
int HYPRE_LSI_MLISFEIAddNumElems(void *object, int elemBlk, int nElems,
                                 int elemNNodes)
{
#ifdef HAVE_MLI
   HYPRE_MLI_SFEI *hypre_sfei = (HYPRE_MLI_SFEI *) object;
   MLI_SFEI       *sfei;

   /* -------------------------------------------------------- */ 
   /* error checking                                           */
   /* -------------------------------------------------------- */ 

   if ( hypre_sfei == NULL ) return 1;
   sfei = (MLI_SFEI *) hypre_sfei->sfei_;
   if ( sfei == NULL ) return 1;

   /* -------------------------------------------------------- */ 
   /* send information to sfei object                          */
   /* -------------------------------------------------------- */ 

   sfei->addNumElems(elemBlk,nElems,elemNNodes);
   return 0;
#else
   (void) object;
   (void) elemBlk;
   (void) nElems;
   (void) elemNNodes;
   return 1;
#endif
}

