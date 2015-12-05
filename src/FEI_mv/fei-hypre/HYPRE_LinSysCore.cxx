/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.80 $
 ***********************************************************************EHEADER*/


//***************************************************************************
// system includes
//---------------------------------------------------------------------------

#include "utilities/_hypre_utilities.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

#ifdef WIN32
#define strcmp _stricmp
#endif

//***************************************************************************
// HYPRE includes
//---------------------------------------------------------------------------

#include "HYPRE.h"
#include "IJ_mv/HYPRE_IJ_mv.h"
#include "parcsr_mv/HYPRE_parcsr_mv.h"
#include "parcsr_ls/HYPRE_parcsr_ls.h"
#include "HYPRE_parcsr_bicgstabl.h"
#include "HYPRE_parcsr_TFQmr.h"
#include "HYPRE_parcsr_bicgs.h"
#include "HYPRE_parcsr_symqmr.h"
#include "HYPRE_parcsr_fgmres.h"
#include "HYPRE_parcsr_lsicg.h"
#include "HYPRE_LinSysCore.h"
#include "parcsr_mv/_hypre_parcsr_mv.h"
#include "HYPRE_LSI_schwarz.h"
#include "HYPRE_LSI_ddilut.h"
#include "HYPRE_LSI_ddict.h"
#include "HYPRE_LSI_poly.h"
#include "HYPRE_LSI_block.h"
#include "HYPRE_LSI_Uzawa_c.h"
#include "HYPRE_LSI_Dsuperlu.h"
#include "HYPRE_MLMaxwell.h"
#include "HYPRE_SlideReduction.h"

//#define HAVE_SYSPDE
//#define HAVE_DSUPERLU
#include "dsuperlu_include.h"

//***************************************************************************
// timers 
//---------------------------------------------------------------------------

#ifdef HYPRE_SEQUENTIAL
#include <time.h>
extern "C"
{
   double LSC_Wtime()
   {
      clock_t ticks;
      double  seconds; 
      ticks   = clock() ;
      seconds = (double) ticks / (double) CLOCKS_PER_SEC;
      return seconds;
   }
}
#else
   double LSC_Wtime()
   {
      return (MPI_Wtime());
   }
#endif
extern "C" {
   int HYPRE_LSI_qsort1a( int *, int *, int, int );
   int HYPRE_LSI_PartitionMatrix(int, int, int*, int**, double**, int*, int**);
   int HYPRE_LSI_GetMatrixDiagonal(int, int, int *, int **, double **, int *, 
                                   int *, double *);
}

//***************************************************************************
// These are external functions needed internally here
//---------------------------------------------------------------------------

#include "HYPRE_LSI_mli.h"

extern "C" {

#ifdef HAVE_ML
   int   HYPRE_LSI_MLCreate( MPI_Comm, HYPRE_Solver *);
   int   HYPRE_LSI_MLDestroy( HYPRE_Solver );
#endif

#ifdef HAVE_MLMAXWELL
   int   HYPRE_LSI_MLMaxwellCreate(MPI_Comm, HYPRE_Solver *);
   int   HYPRE_LSI_MLMaxwellDestroy(HYPRE_Solver );
#endif

#ifdef HAVE_AMGE
   int   HYPRE_LSI_AMGeCreate();
   int   HYPRE_LSI_AMGeDestroy();
   int   HYPRE_LSI_AMGeSetNNodes(int);
   int   HYPRE_LSI_AMGeSetNElements(int);
   int   HYPRE_LSI_AMGeSetSystemSize(int);
   int   HYPRE_LSI_AMGePutRow(int,int,double*,int*);
   int   HYPRE_LSI_AMGeSolve( double *rhs, double *sol ); 
   int   HYPRE_LSI_AMGeSetBoundary( int leng, int *colInd );
   int   HYPRE_LSI_AMGeWriteToFile();
#endif

   void  qsort0(int *, int, int);
   void  qsort1(int *, double *, int, int);
}

#define HAVE_MLI
#define habs(x)  ( ( (x) > 0 ) ? x : -(x))

//***************************************************************************
// constructor
//---------------------------------------------------------------------------

HYPRE_LinSysCore::HYPRE_LinSysCore(MPI_Comm comm) : 
                  comm_(comm),
                  HYOutputLevel_(0),
                  memOptimizerFlag_(0),
                  mapFromSolnFlag_(0),
                  mapFromSolnLeng_(0),
                  mapFromSolnLengMax_(0),
                  mapFromSolnList_(NULL),
                  mapFromSolnList2_(NULL),
                  HYA_(NULL),
                  HYnormalA_(NULL),
                  HYb_(NULL),
                  HYnormalB_(NULL),
                  HYbs_(NULL),
                  HYx_(NULL),
                  HYr_(NULL),
                  HYpxs_(NULL),
                  HYpbs_(NULL),
                  numGlobalRows_(0),
                  localStartRow_(0),
                  localEndRow_(-1),
                  localStartCol_(-1),
                  localEndCol_(-1),
                  rowLengths_(NULL),
                  colIndices_(NULL),
                  colValues_(NULL),
                  reducedA_(NULL),
                  reducedB_(NULL),
                  reducedX_(NULL),
                  reducedR_(NULL),
                  HYA21_(NULL),
                  HYA12_(NULL),
                  A21NRows_(0),
                  A21NCols_(0),
                  HYinvA22_(NULL),
                  currA_(NULL),
                  currB_(NULL),
                  currX_(NULL),
                  currR_(NULL),
                  currentRHS_(0),
                  numRHSs_(1),
                  nStored_(0),
                  storedIndices_(NULL),
                  auxStoredIndices_(NULL),
                  mRHSFlag_(0),
                  mRHSNumGEqns_(0),
                  mRHSGEqnIDs_(NULL),
                  mRHSNEntries_(NULL),
                  mRHSBCType_(NULL),
                  mRHSRowInds_(NULL),
                  mRHSRowVals_(NULL),
                  matrixVectorsCreated_(0),
                  systemAssembled_(0),
                  slideReduction_(0),
                  slideReductionMinNorm_(-1.0),
                  slideReductionScaleMatrix_(0),
                  schurReduction_(0),
                  schurReductionCreated_(0),
                  projectionScheme_(0),
                  projectSize_(0),
                  projectCurrSize_(0),
                  projectionMatrix_(NULL),
                  normalEqnFlag_(0),
                  slideObj_(NULL),
                  selectedList_(NULL),
                  selectedListAux_(NULL),
                  nConstraints_(0),
                  constrList_(NULL),
                  matrixPartition_(0),
                  HYSolver_(NULL), 
                  maxIterations_(1000),
                  tolerance_(1.0e-6),
                  normAbsRel_(0),
                  pcgRecomputeRes_(0),
                  HYPrecon_(NULL), 
                  HYPreconReuse_(0), 
                  HYPreconSetup_(0),
                  lookup_(NULL),
                  haveLookup_(0)
{
   //-------------------------------------------------------------------
   // find my processor ID 
   //-------------------------------------------------------------------

   MPI_Comm_rank(comm, &mypid_);
   MPI_Comm_size(comm, &numProcs_);

   //-------------------------------------------------------------------
   // default method = gmres
   //-------------------------------------------------------------------

   HYSolverName_ = new char[64];
   strcpy(HYSolverName_,"gmres");
   HYSolverID_ = HYGMRES;

   //-------------------------------------------------------------------
   // default preconditioner = identity
   //-------------------------------------------------------------------

   HYPreconName_ = new char[64];
   strcpy(HYPreconName_,"diagonal");
   HYPreconID_ = HYDIAGONAL;

   //-------------------------------------------------------------------
   // parameters for controlling amg, pilut, SuperLU, etc.
   //-------------------------------------------------------------------

   gmresDim_           = 100;  // restart size in GMRES
   fgmresUpdateTol_    = 0;

   amgMaxLevels_       = 30;   // default max number of levels
   amgCoarsenType_     = 0;    // default coarsening
   amgMeasureType_     = 0;    // local measure
   amgSystemSize_      = 1;    // system size
   amgMaxIter_         = 1;    // number of iterations
   amgNumSweeps_[0]    = 1;    // no. of sweeps for fine grid
   amgNumSweeps_[1]    = 1;    // no. of presmoothing sweeps 
   amgNumSweeps_[2]    = 1;    // no. of postsmoothing sweeps 
   amgNumSweeps_[3]    = 1;    // no. of sweeps for coarsest grid
   amgRelaxType_[0]    = 3;    // hybrid for the fine grid
   amgRelaxType_[1]    = 3;    // hybrid for presmoothing 
   amgRelaxType_[2]    = 3;    // hybrid for postsmoothing
   amgRelaxType_[3]    = 9;    // direct for the coarsest level
   amgGridRlxType_     = 0;    // smoothes all points
   amgStrongThreshold_ = 0.25;
   amgSmoothType_      = 0;    // default non point smoother, none
   amgSmoothNumLevels_ = 0;    // no. of levels for non point smoothers
   amgSmoothNumSweeps_ = 1;    // no. of sweeps for non point smoothers
   amgCGSmoothNumSweeps_ = 0;  // no. of sweeps for preconditioned CG smoother
   amgSchwarzRelaxWt_  = 1.0;  // relaxation weight for Schwarz smoother
   amgSchwarzVariant_  = 0;    // hybrid multiplicative Schwarz with
                               // no overlap across processor boundaries
   amgSchwarzOverlap_  = 1;    // minimal overlap
   amgSchwarzDomainType_ = 2;  // domain through agglomeration
   amgUseGSMG_         = 0;
   amgGSMGNSamples_    = 0;
   amgAggLevels_       = 0;
   amgInterpType_      = 0;
   amgPmax_            = 0;

   for (int i = 0; i < 25; i++) amgRelaxWeight_[i] = 1.0; 
   for (int j = 0; j < 25; j++) amgRelaxOmega_[j] = 1.0; 

   pilutFillin_        = 0;    // how many nonzeros to keep in L and U
   pilutDropTol_       = 0.0;
   pilutMaxNnzPerRow_  = 0;    // register the max NNZ/row in matrix A

   ddilutFillin_       = 1.0;  // additional fillin other than A
   ddilutDropTol_      = 1.0e-8;
   ddilutOverlap_      = 0;
   ddilutReorder_      = 0;

   ddictFillin_        = 1.0;  // additional fillin other than A
   ddictDropTol_       = 1.0e-8;

   schwarzFillin_      = 1.0;  // additional fillin other than A
   schwarzNblocks_     = 1;
   schwarzBlksize_     = 0;

   polyOrder_          = 8;    // order of polynomial preconditioner

   parasailsSym_       = 0;    // default is nonsymmetric
   parasailsThreshold_ = 0.1;
   parasailsNlevels_   = 1;
   parasailsFilter_    = 0.05;
   parasailsLoadbal_   = 0.0;
   parasailsReuse_     = 0;    // reuse pattern if nonzero

   euclidargc_         = 2;    // parameters information for Euclid
   euclidargv_         = new char*[euclidargc_*2];
   for (int k = 0; k < euclidargc_*2; k++) euclidargv_[k] = new char[50];
   strcpy(euclidargv_[0], "-level");   
   strcpy(euclidargv_[1], "0");   
   strcpy(euclidargv_[2], "-sparseA");   
   strcpy(euclidargv_[3], "0.0");   

   superluOrdering_    = 0;    // natural ordering in SuperLU
   superluScale_[0]    = 'N';  // no scaling in SuperLUX

   mlMethod_           = 1;
   mlNumPreSweeps_     = 1;
   mlNumPostSweeps_    = 1;
   mlPresmootherType_  = 1;    // default Gauss-Seidel
   mlPostsmootherType_ = 1;    // default Gauss-Seidel
   mlRelaxWeight_      = 0.5;
   mlStrongThreshold_  = 0.08; // one suggested by Vanek/Brezina/Mandel
   mlCoarseSolver_     = 0;    // default coarse solver = SuperLU
   mlCoarsenScheme_    = 1;    // default coarsening scheme = uncoupled
   mlNumPDEs_          = 3;    // default block size 

   truncThresh_        = 0.0;
   rnorm_              = 0.0;
   rhsIDs_             = new int[1];
   rhsIDs_[0]          = 0;
   feData_             = NULL;
   haveFEData_         = 0;
   feData_             = NULL;
   MLI_NumNodes_       = 0;
   MLI_FieldSize_      = 0;
   MLI_NodalCoord_     = NULL;
   MLI_EqnNumbers_     = NULL;
   MLI_Hybrid_GSA_     = 0;
   MLI_Hybrid_NSIncr_   = 2;
   MLI_Hybrid_MaxIter_  = 100;
   MLI_Hybrid_ConvRate_ = 0.95;
   MLI_Hybrid_NTrials_  = 5;
   AMSData_.numNodes_      = 0;
   AMSData_.numLocalNodes_ = 0;
   AMSData_.EdgeNodeList_  = NULL;
   AMSData_.NodeNumbers_   = NULL;
   AMSData_.NodalCoord_    = NULL;
   amsX_ = NULL;
   amsY_ = NULL;
   amsZ_ = NULL;
   amsNumPDEs_ = 3;
   amsMaxIter_ = 1;
   amsTol_     = 0.0;
   amsCycleType_ = 1;
   amsRelaxType_ = 2;
   amsRelaxTimes_ = 1;
   amsRelaxWt_    = 1.0;
   amsRelaxOmega_ = 1.0;
   amsBetaPoisson_ = NULL;
   amsPrintLevel_ = 0;
   amsAlphaCoarsenType_ = 10;
   amsAlphaAggLevels_ = 1;
   amsAlphaRelaxType_ = 6;
   amsAlphaStrengthThresh_ = 0.25;
   amsBetaCoarsenType_ = 10;
   amsBetaAggLevels_ = 1;
   amsBetaRelaxType_ = 6;
   amsBetaStrengthThresh_ = 0.25;
   FEI_mixedDiagFlag_ = 0;
   FEI_mixedDiag_ = NULL;
   sysPDEMethod_ = -1;
   sysPDEFormat_ = -1;
   sysPDETol_ = 0.0;
   sysPDEMaxIter_ = -1;
   sysPDENumPre_ = -1;
   sysPDENumPost_ = -1;
   sysPDENVars_ = 3;

   //-------------------------------------------------------------------
   // parameters ML Maxwell solver
   //-------------------------------------------------------------------

   maxwellANN_ = NULL;
   maxwellGEN_ = NULL;
}

//***************************************************************************
// destructor
//---------------------------------------------------------------------------

HYPRE_LinSysCore::~HYPRE_LinSysCore() 
{
   int i;
   //-------------------------------------------------------------------
   // diagnostic message
   //-------------------------------------------------------------------

   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
      printf("%4d : HYPRE_LSC::entering destructor.\n",mypid_);

   //-------------------------------------------------------------------
   // clean up the allocated matrix and vectors
   //-------------------------------------------------------------------

   if ( HYA_ != NULL ) {HYPRE_IJMatrixDestroy(HYA_); HYA_ = NULL;}
   if ( HYx_ != NULL ) {HYPRE_IJVectorDestroy(HYx_); HYx_ = NULL;}
   if ( HYr_ != NULL ) {HYPRE_IJVectorDestroy(HYr_); HYr_ = NULL;}
   if ( HYbs_ != NULL ) 
   {
      for ( i = 0; i < numRHSs_; i++ ) 
         if ( HYbs_[i] != NULL ) HYPRE_IJVectorDestroy(HYbs_[i]);
      delete [] HYbs_;
      HYbs_ = NULL;
   }
   if ( HYpbs_ != NULL ) 
   {
      for ( i = 0; i <= projectSize_; i++ ) 
         if ( HYpbs_[i] != NULL ) HYPRE_IJVectorDestroy(HYpbs_[i]);
      delete [] HYpbs_;
      HYpbs_ = NULL;
   }
   if ( HYpxs_ != NULL ) 
   {
      for ( i = 0; i <= projectSize_; i++ ) 
         if ( HYpxs_[i] != NULL ) HYPRE_IJVectorDestroy(HYpxs_[i]);
      delete [] HYpxs_;
      HYpxs_ = NULL;
   }
   if (HYnormalA_!= NULL) {HYPRE_IJMatrixDestroy(HYnormalA_);HYnormalA_ = NULL;}
   if (HYnormalB_!= NULL) {HYPRE_IJVectorDestroy(HYnormalB_);HYnormalB_ = NULL;}
   if (reducedA_ != NULL) {HYPRE_IJMatrixDestroy(reducedA_); reducedA_ = NULL;}
   if (reducedB_ != NULL) {HYPRE_IJVectorDestroy(reducedB_); reducedB_ = NULL;}
   if (reducedX_ != NULL) {HYPRE_IJVectorDestroy(reducedX_); reducedX_ = NULL;}
   if (reducedR_ != NULL) {HYPRE_IJVectorDestroy(reducedR_); reducedR_ = NULL;}
   if (HYA21_    != NULL) {HYPRE_IJMatrixDestroy(HYA21_);    HYA21_    = NULL;}
   if (HYA12_    != NULL) {HYPRE_IJMatrixDestroy(HYA12_);    HYA12_    = NULL;}
   if (HYinvA22_ != NULL) {HYPRE_IJMatrixDestroy(HYinvA22_); HYinvA22_ = NULL;}

   matrixVectorsCreated_ = 0;
   systemAssembled_ = 0;
   projectCurrSize_ = 0;

   if ( colIndices_ != NULL )
   {
      for ( i = 0; i < localEndRow_-localStartRow_+1; i++ )
         if ( colIndices_[i] != NULL ) delete [] colIndices_[i];
      delete [] colIndices_;
      colIndices_ = NULL;
   }
   if ( colValues_ != NULL )
   {
      for ( i = 0; i < localEndRow_-localStartRow_+1; i++ )
         if ( colValues_[i] != NULL ) delete [] colValues_[i];
      delete [] colValues_;
      colValues_ = NULL;
   }
   if ( rowLengths_ != NULL ) 
   {
      delete [] rowLengths_;
      rowLengths_ = NULL;
   }
   if ( rhsIDs_ != NULL ) delete [] rhsIDs_;
   if ( storedIndices_ != NULL ) delete [] storedIndices_;
   if ( auxStoredIndices_ != NULL ) delete [] auxStoredIndices_;
   if ( mRHSNumGEqns_ > 0)
   {
      if (mRHSGEqnIDs_  != NULL) delete [] mRHSGEqnIDs_;
      if (mRHSNEntries_ != NULL) delete [] mRHSNEntries_;
      if (mRHSBCType_   != NULL) delete [] mRHSBCType_ ;
      if (mRHSRowInds_ != NULL)
      {
         for (i = 0; i < mRHSNumGEqns_; i++) delete [] mRHSRowInds_[i];
         delete [] mRHSRowInds_;
      }
      if (mRHSRowVals_ != NULL)
      {
         for (i = 0; i < mRHSNumGEqns_; i++) delete [] mRHSRowVals_[i];
         delete [] mRHSRowVals_;
      }
      mRHSNumGEqns_ = 0;
      mRHSGEqnIDs_ = NULL;
      mRHSNEntries_ = NULL;
      mRHSBCType_ = NULL;
      mRHSRowInds_ = NULL;
      mRHSRowVals_ = NULL;
   }

   //-------------------------------------------------------------------
   // clean up direct matrix access variables
   //-------------------------------------------------------------------

   if ( mapFromSolnList_ != NULL ) 
   {
      delete [] mapFromSolnList_;
      mapFromSolnList_ = NULL;
   }
   if ( mapFromSolnList2_ != NULL ) 
   {
      delete [] mapFromSolnList2_;
      mapFromSolnList2_ = NULL;
   }

   //-------------------------------------------------------------------
   // call solver destructors
   //-------------------------------------------------------------------

   if ( HYSolver_ != NULL )
   {
      if (HYSolverID_ == HYPCG)     HYPRE_ParCSRPCGDestroy(HYSolver_);
      if (HYSolverID_ == HYGMRES)   HYPRE_ParCSRGMRESDestroy(HYSolver_);
      if (HYSolverID_ == HYCGSTAB)  HYPRE_ParCSRBiCGSTABDestroy(HYSolver_);
      if (HYSolverID_ == HYCGSTABL) HYPRE_ParCSRBiCGSTABLDestroy(HYSolver_);
      if (HYSolverID_ == HYAMG)     HYPRE_BoomerAMGDestroy(HYSolver_);
      if (HYSolverID_ == HYTFQMR)   HYPRE_ParCSRTFQmrDestroy(HYSolver_);
      HYSolver_ = NULL;
   }
   delete [] HYSolverName_;
   HYSolverName_ = NULL;
#ifdef HAVE_AMGE
   if ( HYSolverID_ == HYAMGE ) HYPRE_LSI_AMGeDestroy();
#endif

   //-------------------------------------------------------------------
   // call preconditioner destructors
   //-------------------------------------------------------------------

   if ( HYPrecon_ != NULL )
   {
      if ( HYPreconID_ == HYPILUT )
         HYPRE_ParCSRPilutDestroy( HYPrecon_ );

      else if ( HYPreconID_ == HYPARASAILS )
         HYPRE_ParCSRParaSailsDestroy( HYPrecon_ );

      else if ( HYPreconID_ == HYBOOMERAMG )
         HYPRE_BoomerAMGDestroy( HYPrecon_ );

      else if ( HYPreconID_ == HYDDILUT )
         HYPRE_LSI_DDIlutDestroy( HYPrecon_ );

      else if ( HYPreconID_ == HYSCHWARZ )
         HYPRE_LSI_SchwarzDestroy( HYPrecon_ );

      else if ( HYPreconID_ == HYPOLY )
         HYPRE_LSI_PolyDestroy( HYPrecon_ );

      else if ( HYPreconID_ == HYEUCLID )
         HYPRE_EuclidDestroy( HYPrecon_ );

      else if ( HYPreconID_ == HYBLOCK )
         HYPRE_LSI_BlockPrecondDestroy( HYPrecon_ );

#ifdef HAVE_ML
      else if ( HYPreconID_ == HYML )
         HYPRE_LSI_MLDestroy( HYPrecon_ );
#endif

#ifdef HAVE_MLMAXWELL
      else if ( HYPreconID_ == HYMLMAXWELL )
         HYPRE_LSI_MLMaxwellDestroy( HYPrecon_ );
#endif
      else if ( HYPreconID_ == HYMLI )
         HYPRE_LSI_MLIDestroy( HYPrecon_ );

      else if ( HYPreconID_ == HYAMS )
      {
 	 // Destroy G and coordinate vectors
         HYPRE_AMSFEIDestroy( HYPrecon_ );
         HYPRE_AMSDestroy( HYPrecon_ );
      }
#ifdef HAVE_SYSPDE
      else if ( HYPreconID_ == HYSYSPDE )
         HYPRE_ParCSRSysPDEDestroy( HYPrecon_ );
#endif
#ifdef HAVE_DSUPERLU
      else if ( HYPreconID_ == HYDSLU )
         HYPRE_LSI_DSuperLUDestroy(HYPrecon_);
#endif

      HYPrecon_ = NULL;
   }
   delete [] HYPreconName_;
   HYPreconName_ = NULL;

   for (i = 0; i < euclidargc_*2; i++) delete [] euclidargv_[i];
   delete [] euclidargv_;
   euclidargv_ = NULL;

   //-------------------------------------------------------------------
   // clean up variable for various reduction
   //-------------------------------------------------------------------

   if ( constrList_ != NULL ) 
   {
      delete [] constrList_; 
      constrList_ = NULL;
   }
   if (selectedList_ != NULL) 
   {
      delete [] selectedList_; 
      selectedList_ = NULL;
   }
   if (selectedListAux_ != NULL) 
   {
      delete [] selectedListAux_; 
      selectedListAux_ = NULL;
   }
    
   //-------------------------------------------------------------------
   // deallocate local storage for MLI
   //-------------------------------------------------------------------

#ifdef HAVE_MLI
   if ( feData_ != NULL ) 
   {
      if      (haveFEData_ == 1) HYPRE_LSI_MLIFEDataDestroy(feData_);
      else if (haveFEData_ == 2) HYPRE_LSI_MLISFEIDestroy(feData_);
      feData_ = NULL;
   }
   if ( MLI_NodalCoord_ != NULL ) delete [] MLI_NodalCoord_;
   if ( MLI_EqnNumbers_ != NULL ) delete [] MLI_EqnNumbers_;
#endif

   if (maxwellANN_ != NULL)
   {
      HYPRE_ParCSRMatrixDestroy(maxwellANN_);
      maxwellANN_ = NULL;
   }
   if (amsX_ != NULL) HYPRE_IJVectorDestroy(amsX_);
   if (amsY_ != NULL) HYPRE_IJVectorDestroy(amsY_);
   if (amsZ_ != NULL) HYPRE_IJVectorDestroy(amsZ_);
   // Users who copy this matrix in should be responsible for
   // destroying this
   //if (maxwellGEN_ != NULL)
   //{
   //   HYPRE_ParCSRMatrixDestroy(maxwellGEN_);
   //   maxwellGEN_ = NULL;
   //}
   if (AMSData_.EdgeNodeList_ != NULL) delete [] AMSData_.EdgeNodeList_;
   if (AMSData_.NodeNumbers_  != NULL) delete [] AMSData_.NodeNumbers_;
   if (AMSData_.NodalCoord_   != NULL) delete [] AMSData_.NodalCoord_;
   if (FEI_mixedDiag_ != NULL) delete [] FEI_mixedDiag_;

   //-------------------------------------------------------------------
   // diagnostic message
   //-------------------------------------------------------------------

   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
      printf("%4d : HYPRE_LSC::leaving  destructor.\n",mypid_);
}

//***************************************************************************
// clone a copy of HYPRE_LinSysCore
//---------------------------------------------------------------------------

#ifndef NOFEI
LinearSystemCore* HYPRE_LinSysCore::clone() 
{
   return(new HYPRE_LinSysCore(comm_));
}
#endif

//***************************************************************************
// passing a lookup table to this object
// (note : this is called in FEI_initComplete)
//---------------------------------------------------------------------------

int HYPRE_LinSysCore::setLookup(Lookup& lookup)
{
   //-------------------------------------------------------------------
   // diagnostic message
   //-------------------------------------------------------------------

   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
      printf("%4d : HYPRE_LSC::entering setLookup.\n",mypid_);

   //-------------------------------------------------------------------
   // set the lookup object and initialize the MLI_FEData object
   //-------------------------------------------------------------------

   if (&lookup == NULL) return (0);
   lookup_ = &lookup;
   haveLookup_ = 1;

   //-------------------------------------------------------------------
   // diagnostic message
   //-------------------------------------------------------------------

   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
      printf("%4d : HYPRE_LSC::leaving  setLookup.\n",mypid_);
   return (0);
}

//***************************************************************************
//This function is where we establish the structures/objects associated
//with the linear algebra library. i.e., do initial allocations, etc.
// Rows and columns are 1-based.
//---------------------------------------------------------------------------

int HYPRE_LinSysCore::createMatricesAndVectors(int numGlobalEqns,
                       int firstLocalEqn, int numLocalEqns) 
{
   int i, ierr;

   //-------------------------------------------------------------------
   // diagnostic message
   //-------------------------------------------------------------------

   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
   {
      printf("%4d : HYPRE_LSC::entering createMatricesAndVectors.\n",mypid_);
      printf("%4d : HYPRE_LSC::startrow, endrow = %d %d\n",mypid_,
                    firstLocalEqn, firstLocalEqn+numLocalEqns-1);
   }

   //-------------------------------------------------------------------
   // clean up previously allocated matrix
   //-------------------------------------------------------------------

   if ( rowLengths_ != NULL ) delete [] rowLengths_;
   if ( colIndices_ != NULL )
   {
      int nrows = localEndRow_ - localStartRow_ + 1;
      for ( i = 0; i < nrows; i++ )
         if ( colIndices_[i] != NULL ) delete [] colIndices_[i];
      delete [] colIndices_;
   }
   if ( colValues_ != NULL )
   {
      int nrows = localEndRow_ - localStartRow_ + 1;
      for ( i = 0; i < nrows; i++ )
         if ( colValues_[i] != NULL ) delete [] colValues_[i];
      delete [] colValues_;
   }
   rowLengths_ = NULL;
   colIndices_ = NULL;
   colValues_  = NULL;

   if ( mRHSNumGEqns_ > 0)
   {
      if (mRHSGEqnIDs_  != NULL) delete [] mRHSGEqnIDs_;
      if (mRHSNEntries_ != NULL) delete [] mRHSNEntries_;
      if (mRHSBCType_   != NULL) delete [] mRHSBCType_;
      if (mRHSRowInds_ != NULL)
      {
         for (i = 0; i < mRHSNumGEqns_; i++) delete [] mRHSRowInds_[i];
         delete [] mRHSRowInds_;
      }
      if (mRHSRowVals_ != NULL)
      {
         for (i = 0; i < mRHSNumGEqns_; i++) delete [] mRHSRowVals_[i];
         delete [] mRHSRowVals_;
      }
      mRHSNumGEqns_ = 0;
      mRHSGEqnIDs_ = NULL;
      mRHSNEntries_ = NULL;
      mRHSBCType_ = NULL;
      mRHSRowInds_ = NULL;
      mRHSRowVals_ = NULL;
   }

   //-------------------------------------------------------------------
   // error checking
   //-------------------------------------------------------------------

   if ( ( firstLocalEqn <= 0 ) || 
        ( firstLocalEqn+numLocalEqns-1) > numGlobalEqns)
   {
      printf("%4d : createMatricesVectors: invalid local equation nos.\n",
             mypid_);
      exit(1);
   }

   localStartRow_ = firstLocalEqn;
   localEndRow_   = firstLocalEqn + numLocalEqns - 1;
   numGlobalRows_ = numGlobalEqns;

   //-------------------------------------------------------------------
   // first clean up previous allocations
   //-------------------------------------------------------------------

   if ( matrixVectorsCreated_ )
   {
      if ( HYA_ != NULL ) {HYPRE_IJMatrixDestroy(HYA_); HYA_ = NULL;}
      if ( HYx_ != NULL ) {HYPRE_IJVectorDestroy(HYx_); HYx_ = NULL;}
      if ( HYr_ != NULL ) {HYPRE_IJVectorDestroy(HYr_); HYr_ = NULL;}
      if ( HYbs_ != NULL ) 
      {
         for ( i = 0; i < numRHSs_; i++ ) 
            if ( HYbs_[i] != NULL ) HYPRE_IJVectorDestroy(HYbs_[i]);
         delete [] HYbs_;
         HYbs_ = NULL;
      }
      if (reducedA_ != NULL) HYPRE_IJMatrixDestroy(reducedA_); 
      if (reducedB_ != NULL) HYPRE_IJVectorDestroy(reducedB_);
      if (reducedX_ != NULL) HYPRE_IJVectorDestroy(reducedX_);
      if (reducedR_ != NULL) HYPRE_IJVectorDestroy(reducedR_);
      if (HYA21_    != NULL) HYPRE_IJMatrixDestroy(HYA21_);
      if (HYA12_    != NULL) HYPRE_IJMatrixDestroy(HYA12_);
      if (HYinvA22_ != NULL) HYPRE_IJMatrixDestroy(HYinvA22_);
      reducedA_ = NULL;
      reducedB_ = NULL;
      reducedX_ = NULL;
      reducedR_ = NULL;
      HYA21_    = NULL;
      HYA12_    = NULL;
      HYinvA22_ = NULL;
      A21NRows_ = A21NCols_ = reducedAStartRow_ = 0;
   }

   //-------------------------------------------------------------------
   // instantiate the matrix (can also handle rectangular matrix)
   //-------------------------------------------------------------------
   
   if (localStartCol_ == -1)
      ierr = HYPRE_IJMatrixCreate(comm_, localStartRow_-1,localEndRow_-1,
                                  localStartRow_-1,localEndRow_-1, &HYA_);
   else
      ierr = HYPRE_IJMatrixCreate(comm_, localStartRow_-1,localEndRow_-1,
                                  localStartCol_,localEndCol_, &HYA_);
   ierr = HYPRE_IJMatrixSetObjectType(HYA_, HYPRE_PARCSR);
   //assert(!ierr);

   //-------------------------------------------------------------------
   // instantiate the right hand vectors
   //-------------------------------------------------------------------

   HYbs_ = new HYPRE_IJVector[numRHSs_];
   for ( i = 0; i < numRHSs_; i++ )
   {
      ierr = HYPRE_IJVectorCreate(comm_, localStartRow_-1, localEndRow_-1,
                                  &(HYbs_[i]));
      ierr = HYPRE_IJVectorSetObjectType(HYbs_[i], HYPRE_PARCSR);
      ierr = HYPRE_IJVectorInitialize(HYbs_[i]);
      ierr = HYPRE_IJVectorAssemble(HYbs_[i]);
      //assert(!ierr);
   }
   HYb_ = HYbs_[0];

   //-------------------------------------------------------------------
   // instantiate the solution vector
   //-------------------------------------------------------------------

   if (localStartCol_ == -1)
      ierr = HYPRE_IJVectorCreate(comm_,localStartRow_-1,localEndRow_-1,&HYx_);
   else
      ierr = HYPRE_IJVectorCreate(comm_,localStartCol_,localEndCol_,&HYx_);
   ierr = HYPRE_IJVectorSetObjectType(HYx_, HYPRE_PARCSR);
   ierr = HYPRE_IJVectorInitialize(HYx_);
   ierr = HYPRE_IJVectorAssemble(HYx_);
   //assert(!ierr);

   //-------------------------------------------------------------------
   // reset fedata
   //-------------------------------------------------------------------

#ifdef HAVE_MLI
   if ( feData_ != NULL ) 
   {
      if      (haveFEData_ == 1) HYPRE_LSI_MLIFEDataDestroy(feData_);
      else if (haveFEData_ == 2) HYPRE_LSI_MLISFEIDestroy(feData_);
      feData_ = NULL;
      if ( MLI_NodalCoord_ != NULL ) delete [] MLI_NodalCoord_;
      if ( MLI_EqnNumbers_ != NULL ) delete [] MLI_EqnNumbers_;
      MLI_NodalCoord_ = NULL;
      MLI_EqnNumbers_ = NULL;
      MLI_NumNodes_ = 0;
   }
#endif

   //-------------------------------------------------------------------
   // for amge
   //-------------------------------------------------------------------

#ifdef HAVE_AMGE
   HYPRE_LSI_AMGeCreate();
   HYPRE_LSI_AMGeSetNNodes(numGlobalRows_);
   HYPRE_LSI_AMGeSetNElements(numGlobalRows_);
   HYPRE_LSI_AMGeSetSystemSize(1);
#endif 

   //-------------------------------------------------------------------
   // instantiate the residual vector
   //-------------------------------------------------------------------

   ierr = HYPRE_IJVectorCreate(comm_, localStartRow_-1, localEndRow_-1, &HYr_);
   ierr = HYPRE_IJVectorSetObjectType(HYr_, HYPRE_PARCSR);
   ierr = HYPRE_IJVectorInitialize(HYr_);
   ierr = HYPRE_IJVectorAssemble(HYr_);
   //assert(!ierr);
   matrixVectorsCreated_ = 1;
   schurReductionCreated_ = 0;
   systemAssembled_ = 0;
   normalEqnFlag_ &= 1;
   if ( HYnormalA_ != NULL ) 
   {
      HYPRE_IJMatrixDestroy(HYnormalA_); 
      HYnormalA_ = NULL;
   }
   if ( HYnormalB_ != NULL ) 
   {
      HYPRE_IJVectorDestroy(HYnormalB_); 
      HYnormalB_ = NULL;
   }

   //-------------------------------------------------------------------
   // diagnostic message
   //-------------------------------------------------------------------

   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
      printf("%4d : HYPRE_LSC::leaving  createMatricesAndVectors.\n",mypid_);
   return (0);
}

//***************************************************************************
// set global and local number of equations
// (This is called in FEI_initComplete after setLookup)
//---------------------------------------------------------------------------

int HYPRE_LinSysCore::setGlobalOffsets(int leng, int* nodeOffsets,
                       int* eqnOffsets, int* blkEqnOffsets)
{
   //-------------------------------------------------------------------
   // diagnostic message
   //-------------------------------------------------------------------

   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
      printf("%4d : HYPRE_LSC::entering setGlobalOffsets.\n",mypid_);

   //-------------------------------------------------------------------
   // set local range (incoming 0-based, locally 1-based)
   //-------------------------------------------------------------------

   (void) leng;
   (void) nodeOffsets;
   (void) blkEqnOffsets;
   int firstLocalEqn = eqnOffsets[mypid_] + 1;
   int numLocalEqns  = eqnOffsets[mypid_+1] - firstLocalEqn + 1;
   int numGlobalEqns = eqnOffsets[numProcs_];
   createMatricesAndVectors(numGlobalEqns, firstLocalEqn, numLocalEqns); 

   //-------------------------------------------------------------------
   // diagnostic message
   //-------------------------------------------------------------------

   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
   {
      printf("%4d : HYPRE_LSC::startrow, endrow = %d %d\n",mypid_,
                    localStartRow_, localEndRow_);
      printf("%4d : HYPRE_LSC::leaving  setGlobalOffsets.\n",mypid_);
   }
   return (0);
}

//***************************************************************************
// Grid related function : element node connectivities
//---------------------------------------------------------------------------

int HYPRE_LinSysCore::setConnectivities(GlobalID elemBlk, int nElems,
                       int nNodesPerElem, const GlobalID* elemIDs,
                       const int* const* connNodes)
{
#ifdef HAVE_MLI
   (void) elemIDs;
   (void) connNodes;
   if ( HYPreconID_ == HYMLI && haveFEData_ == 2 )
   {
      if (feData_ == NULL) feData_ = (void *) HYPRE_LSI_MLISFEICreate(comm_);
      HYPRE_LSI_MLISFEIAddNumElems(feData_,elemBlk,nElems,nNodesPerElem);
   }
#else
   (void) elemBlk;
   (void) nElems;
   (void) nNodesPerElem;
   (void) elemIDs;
   (void) connNodes;
   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) > 3 )
      printf("%4d : HYPRE_LSC::setConnectivities not implemented.\n",mypid_);
#endif
   return (0);
}

//***************************************************************************
// Grid related function : element stiffness matrix loading
//---------------------------------------------------------------------------

int HYPRE_LinSysCore::setStiffnessMatrices(GlobalID elemBlk, int nElems,
              const GlobalID* elemIDs,const double *const *const *stiff,
              int nEqnsPerElem, const int *const * eqnIndices)
{
#ifdef HAVE_MLI
   if ( HYPreconID_ == HYMLI && feData_ != NULL )
   {
      HYPRE_LSI_MLISFEILoadElemMatrices(feData_,elemBlk,nElems,(int*)elemIDs,
                           (double***)stiff,nEqnsPerElem,(int**)eqnIndices);
   }
#else
   (void) elemBlk;
   (void) nElems;
   (void) elemIDs;
   (void) stiff;
   (void) nEqnsPerElem;
   (void) eqnIndices;
   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) > 3 )
      printf("%4d : HYPRE_LSC::setStiffnessMatrices not implemented.\n",
             mypid_);
#endif
   return (0);
}

//***************************************************************************
// Grid related function : element load vector loading
//---------------------------------------------------------------------------

int HYPRE_LinSysCore::setLoadVectors(GlobalID elemBlock, int numElems,
                       const GlobalID* elemIDs, const double *const *load,
                       int numEqnsPerElem, const int *const * eqnIndices)
{
   (void) elemBlock;
   (void) numElems;
   (void) elemIDs;
   (void) load;
   (void) numEqnsPerElem;
   (void) eqnIndices;

   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) > 3 )
      printf("%4d : HYPRE_LSC::setLoadVectors not implemented.\n", mypid_);
   return (0);
}

//***************************************************************************
// Set the number of rows in the diagonal part and off diagonal part
// of the matrix, using the structure of the matrix, stored in rows.
// rows is an array that is 0-based. localStartRow and localEndRow are 1-based.
//---------------------------------------------------------------------------

int HYPRE_LinSysCore::allocateMatrix(int **colIndices, int *rowLengths)
{
   int i, j, nsize, rowLeng, maxSize, minSize, searchFlag, *indPtr, *indPtr2;
   double *vals;

   //-------------------------------------------------------------------
   // diagnoistic message and error checking 
   //-------------------------------------------------------------------

   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
      printf("%4d : HYPRE_LSC::entering allocateMatrix.\n", mypid_);
   if ( localEndRow_ < localStartRow_ ) 
   {
      printf("allocateMatrix ERROR : createMatrixAndVectors should be\n");
      printf("                       called before allocateMatrix.\n");
      exit(1);
   }

   //-------------------------------------------------------------------
   // clean up previously allocated matrix
   //-------------------------------------------------------------------

   if ( rowLengths_ != NULL ) delete [] rowLengths_;
   rowLengths_ = NULL;
   if ( colIndices_ != NULL )
   {
      for ( i = 0; i < localEndRow_-localStartRow_+1; i++ )
         if ( colIndices_[i] != NULL ) delete [] colIndices_[i];
      delete [] colIndices_;
      colIndices_ = NULL;
   }
   if ( colValues_ != NULL )
   {
      for ( i = 0; i < localEndRow_-localStartRow_+1; i++ )
         if ( colValues_[i] != NULL ) delete [] colValues_[i];
      delete [] colValues_;
      colValues_ = NULL;
   }

   //-------------------------------------------------------------------
   // allocate and store the column index information
   //-------------------------------------------------------------------

   nsize       = localEndRow_ - localStartRow_ + 1;
   rowLengths_ = new int[nsize];
   colIndices_ = new int*[nsize];
   colValues_  = new double*[nsize];
   maxSize     = 0;
   minSize     = 1000000;
   for ( i = 0; i < nsize; i++ )
   {
      rowLeng = rowLengths_[i] = rowLengths[i];
      if ( rowLeng > 0 ) 
      {
         colIndices_[i] = new int[rowLeng];
         assert( colIndices_[i] != NULL );
      }
      else colIndices_[i] = NULL;
      indPtr  = colIndices_[i];
      indPtr2 = colIndices[i];
      for ( j = 0; j < rowLeng; j++ ) indPtr[j] = indPtr2[j];
      searchFlag = 0;
      for ( j = 1; j < rowLeng; j++ )
         if ( indPtr[j] < indPtr[j-1]) {searchFlag = 1; break;}
      if ( searchFlag ) qsort0( indPtr, 0, rowLeng-1);
      maxSize = ( rowLeng > maxSize ) ? rowLeng : maxSize;
      minSize = ( rowLeng < minSize ) ? rowLeng : minSize;
      if ( rowLeng > 0 ) 
      {
         colValues_[i] = new double[rowLeng];
         assert( colValues_[i] != NULL );
      }
      vals = colValues_[i];
      for ( j = 0; j < rowLeng; j++ ) vals[j] = 0.0;
   }
   MPI_Allreduce(&maxSize, &pilutMaxNnzPerRow_,1,MPI_INT,MPI_MAX,comm_);

   //-------------------------------------------------------------------
   // diagnoistic message 
   //-------------------------------------------------------------------

   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
   {
      printf("%4d : allocateMatrix : max/min nnz/row = %d %d\n", mypid_, 
                    maxSize, minSize);
      printf("%4d : HYPRE_LSC::leaving  allocateMatrix.\n", mypid_);
   }
   return (0);
}

//***************************************************************************
// to establish the structures/objects associated with the linear algebra 
// library. i.e., do initial allocations, etc.
//---------------------------------------------------------------------------

int HYPRE_LinSysCore::setMatrixStructure(int** ptColIndices, int* ptRowLengths,
                           int** blkColIndices, int* blkRowLengths,
                           int* ptRowsPerBlkRow)
{
   int i, j;

   //-------------------------------------------------------------------
   // diagnoistic message 
   //-------------------------------------------------------------------

   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
   {
      printf("%4d : HYPRE_LSC::entering setMatrixStructure.\n",mypid_);
      if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 6 )
      {
         int nRows = localEndRow_ - localStartRow_ + 1;
         for (i = 0; i < nRows; i++)
            for (j = 0; j < ptRowLengths[i]; j++) 
               printf("  %4d : row, col = %d %d\n",mypid_,
                      localStartRow_+i, ptColIndices[i][j]+1);
      }
   }
   (void) blkColIndices;
   (void) blkRowLengths;
   (void) ptRowsPerBlkRow;

   //-------------------------------------------------------------------
   // allocate local space for matrix
   //-------------------------------------------------------------------

   int numLocalRows = localEndRow_ - localStartRow_ + 1;
   for ( i = 0; i < numLocalRows; i++ )
      for ( j = 0; j < ptRowLengths[i]; j++ ) ptColIndices[i][j]++;

   allocateMatrix(ptColIndices, ptRowLengths);

   for ( i = 0; i < numLocalRows; i++ )
      for ( j = 0; j < ptRowLengths[i]; j++ ) ptColIndices[i][j]--;

   //-------------------------------------------------------------------
   // diagnoistic message 
   //-------------------------------------------------------------------

   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
      printf("%4d : HYPRE_LSC::leaving  setMatrixStructure.\n",mypid_);
   return (0);
}

//***************************************************************************
// set Lagrange multiplier equations
//---------------------------------------------------------------------------

int HYPRE_LinSysCore::setMultCREqns(int multCRSetID, int numCRs, 
                       int numNodesPerCR, int** nodeNumbers, int** eqnNumbers,
                       int* fieldIDs, int* multiplierEqnNumbers)
{
   (void) multCRSetID;
   (void) numCRs;
   (void) numNodesPerCR;
   (void) nodeNumbers;
   (void) eqnNumbers;
   (void) fieldIDs;
   (void) multiplierEqnNumbers;

   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) > 3 )
      printf("%4d : HYPRE_LSC::setMultCREqns not implemented.\n",mypid_);
   return (0);
}

//***************************************************************************
// set penalty constraint equations
//---------------------------------------------------------------------------

int HYPRE_LinSysCore::setPenCREqns(int penCRSetID, int numCRs, 
                       int numNodesPerCR, int** nodeNumbers, int** eqnNumbers,
                       int* fieldIDs)
{
   (void) penCRSetID;
   (void) numCRs;
   (void) numNodesPerCR;
   (void) nodeNumbers;
   (void) eqnNumbers;
   (void) fieldIDs;

   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) > 3 )
      printf("%4d : HYPRE_LSC::setPenCREqns not implemented.\n",mypid_);
   return (0);
}

//***************************************************************************
// This function is needed in order to construct a new problem with the
// same sparsity pattern.
//---------------------------------------------------------------------------

int HYPRE_LinSysCore::resetMatrixAndVector(double setValue)
{
   int    i, j, ierr, localNRows, *cols;
   double *vals;

   //-------------------------------------------------------------------
   // diagnoistic message 
   //-------------------------------------------------------------------

   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
      printf("%4d : HYPRE_LSC::entering resetMatrixAndVector.\n",mypid_);
   if ( setValue != 0.0 && mypid_ == 0 )
   {
      printf("resetMatrixAndVector ERROR : cannot take nonzeros.\n");
      exit(1);
   }

   //-------------------------------------------------------------------
   // reset vector values
   //-------------------------------------------------------------------

   localNRows = localEndRow_ - localStartRow_ + 1;
   cols  = new int[localNRows];
   vals  = new double[localNRows];
   for (i = 0; i < localNRows; i++)
   {
      cols[i] = localStartRow_ + i - 1;
      vals[i] = 0.0;
   }    
   for (i = 0; i < numRHSs_; i++) 
      HYPRE_IJVectorSetValues(HYbs_[i], localNRows, (const int *) cols,
                              (const double *) vals);

   delete [] cols;
   delete [] vals;

   systemAssembled_ = 0;
   schurReductionCreated_ = 0;
   projectCurrSize_ = 0;
   normalEqnFlag_ &= 1;
   if ( HYnormalA_ != NULL ) 
   {
      HYPRE_IJMatrixDestroy(HYnormalA_); 
      HYnormalA_ = NULL;
   }
   if ( HYnormalB_ != NULL ) 
   {
      HYPRE_IJVectorDestroy(HYnormalB_); 
      HYnormalB_ = NULL;
   }

   //-------------------------------------------------------------------
   // for now, since HYPRE does not yet support
   // re-initializing the matrix, restart the whole thing
   //-------------------------------------------------------------------

   if ( HYA_ != NULL ) HYPRE_IJMatrixDestroy(HYA_);
   ierr = HYPRE_IJMatrixCreate(comm_, localStartRow_-1, localEndRow_-1,
                               localStartRow_-1, localEndRow_-1, &HYA_);
   ierr = HYPRE_IJMatrixSetObjectType(HYA_, HYPRE_PARCSR);
   //assert(!ierr);

   //-------------------------------------------------------------------
   // clean the reduction stuff
   //-------------------------------------------------------------------

   if (reducedA_ != NULL) {HYPRE_IJMatrixDestroy(reducedA_); reducedA_ = NULL;}
   if (reducedB_ != NULL) {HYPRE_IJVectorDestroy(reducedB_); reducedB_ = NULL;}
   if (reducedX_ != NULL) {HYPRE_IJVectorDestroy(reducedX_); reducedX_ = NULL;}
   if (reducedR_ != NULL) {HYPRE_IJVectorDestroy(reducedR_); reducedR_ = NULL;}
   if (HYA21_    != NULL) {HYPRE_IJMatrixDestroy(HYA21_);    HYA21_    = NULL;}
   if (HYA12_    != NULL) {HYPRE_IJMatrixDestroy(HYA12_);    HYA12_    = NULL;}
   if (HYinvA22_ != NULL) {HYPRE_IJMatrixDestroy(HYinvA22_); HYinvA22_ = NULL;}
   A21NRows_ = A21NCols_ = reducedAStartRow_ = 0;

   //-------------------------------------------------------------------
   // allocate space for storing the matrix coefficient
   //-------------------------------------------------------------------

   if ( colValues_ != NULL )
   {
      int nrows = localEndRow_ - localStartRow_ + 1;
      for ( i = 0; i < nrows; i++ )
         if ( colValues_[i] != NULL ) delete [] colValues_[i];
      delete [] colValues_;
   }
   colValues_  = NULL;

   colValues_ = new double*[localNRows];
   for ( i = 0; i < localNRows; i++ )
   {
      if ( rowLengths_[i] > 0 ) colValues_[i] = new double[rowLengths_[i]];
      for ( j = 0; j < rowLengths_[i]; j++ ) colValues_[i][j] = 0.0;
   }

   //-------------------------------------------------------------------
   // reset fedata
   //-------------------------------------------------------------------

#ifdef HAVE_MLI
   if ( feData_ != NULL ) 
   {
      if      (haveFEData_ == 1) HYPRE_LSI_MLIFEDataDestroy(feData_);
      else if (haveFEData_ == 2) HYPRE_LSI_MLISFEIDestroy(feData_);
      feData_ = NULL;
      if ( MLI_NodalCoord_ != NULL ) delete [] MLI_NodalCoord_;
      if ( MLI_EqnNumbers_ != NULL ) delete [] MLI_EqnNumbers_;
      MLI_NodalCoord_ = NULL;
      MLI_EqnNumbers_ = NULL;
      MLI_NumNodes_ = 0;
   }
#endif

   //-------------------------------------------------------------------
   // diagnostic message
   //-------------------------------------------------------------------

   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
      printf("%4d : HYPRE_LSC::leaving  resetMatrixAndVector.\n", mypid_);
   return (0);
}

//***************************************************************************
// new function to reset matrix independently
//---------------------------------------------------------------------------

int HYPRE_LinSysCore::resetMatrix(double setValue) 
{
   int  i, j, ierr, size;

   //-------------------------------------------------------------------
   // diagnostic message
   //-------------------------------------------------------------------

   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
      printf("%4d : HYPRE_LSC::entering resetMatrix.\n",mypid_);
   if ( setValue != 0.0 && mypid_ == 0 )
   {
      printf("resetMatrix ERROR : cannot take nonzeros.\n");
      exit(1);
   }

   //-------------------------------------------------------------------
   // clean the reduction stuff
   //-------------------------------------------------------------------

   if (reducedA_ != NULL) {HYPRE_IJMatrixDestroy(reducedA_); reducedA_ = NULL;}
   if (reducedB_ != NULL) {HYPRE_IJVectorDestroy(reducedB_); reducedB_ = NULL;}
   if (reducedX_ != NULL) {HYPRE_IJVectorDestroy(reducedX_); reducedX_ = NULL;}
   if (reducedR_ != NULL) {HYPRE_IJVectorDestroy(reducedR_); reducedR_ = NULL;}
   if (HYA21_    != NULL) {HYPRE_IJMatrixDestroy(HYA21_);    HYA21_    = NULL;}
   if (HYA12_    != NULL) {HYPRE_IJMatrixDestroy(HYA12_);    HYA12_    = NULL;}
   if (HYinvA22_ != NULL) {HYPRE_IJMatrixDestroy(HYinvA22_); HYinvA22_ = NULL;}
   A21NRows_ = A21NCols_ = reducedAStartRow_ = 0;

   //-------------------------------------------------------------------
   // for now, since HYPRE does not yet support
   // re-initializing the matrix, restart the whole thing
   //-------------------------------------------------------------------

   if ( HYA_ != NULL ) HYPRE_IJMatrixDestroy(HYA_);
   size = localEndRow_ - localStartRow_ + 1;
   if (localStartCol_ == -1)
      ierr = HYPRE_IJMatrixCreate(comm_, localStartRow_-1, localEndRow_-1,
                                  localStartRow_-1, localEndRow_-1, &HYA_);
   else
      ierr = HYPRE_IJMatrixCreate(comm_, localStartRow_-1, localEndRow_-1,
                                  localStartCol_, localEndCol_, &HYA_);
   ierr = HYPRE_IJMatrixSetObjectType(HYA_, HYPRE_PARCSR);
   //assert(!ierr);

   //-------------------------------------------------------------------
   // allocate space for storing the matrix coefficient
   //-------------------------------------------------------------------

   if ( colValues_ != NULL )
   {
      int nrows = localEndRow_ - localStartRow_ + 1;
      for ( i = 0; i < nrows; i++ )
         if ( colValues_[i] != NULL ) delete [] colValues_[i];
      delete [] colValues_;
   }
   colValues_  = NULL;

   colValues_ = new double*[size];
   for ( i = 0; i < size; i++ )
   {
      if ( rowLengths_[i] > 0 ) colValues_[i] = new double[rowLengths_[i]];
      for ( j = 0; j < rowLengths_[i]; j++ ) colValues_[i][j] = 0.0;
   }

   //-------------------------------------------------------------------
   // reset system flags
   //-------------------------------------------------------------------

   systemAssembled_ = 0;
   schurReductionCreated_ = 0;
   projectCurrSize_ = 0;
   normalEqnFlag_ &= 5;
   if ( HYnormalA_ != NULL ) 
   {
      HYPRE_IJMatrixDestroy(HYnormalA_); 
      HYnormalA_ = NULL;
   }

   //-------------------------------------------------------------------
   // reset fedata
   //-------------------------------------------------------------------

#ifdef HAVE_MLI
   if ( feData_ != NULL ) 
   {
      if      (haveFEData_ == 1) HYPRE_LSI_MLIFEDataDestroy(feData_);
      else if (haveFEData_ == 2) HYPRE_LSI_MLISFEIDestroy(feData_);
      feData_ = NULL;
      if ( MLI_NodalCoord_ != NULL ) delete [] MLI_NodalCoord_;
      if ( MLI_EqnNumbers_ != NULL ) delete [] MLI_EqnNumbers_;
      MLI_NodalCoord_ = NULL;
      MLI_EqnNumbers_ = NULL;
      MLI_NumNodes_ = 0;
   }
#endif

   //-------------------------------------------------------------------
   // diagnostic message
   //-------------------------------------------------------------------

   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
      printf("%4d : HYPRE_LSC::leaving  resetMatrix.\n", mypid_);
   return (0);
}

//***************************************************************************
// new function to reset vectors independently
//---------------------------------------------------------------------------

int HYPRE_LinSysCore::resetRHSVector(double setValue) 
{
   int    i, localNRows, *cols;
   double *vals;

   //-------------------------------------------------------------------
   // diagnostic message
   //-------------------------------------------------------------------

   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
      printf("%4d : HYPRE_LSC::entering resetRHSVector.\n",mypid_);

   //-------------------------------------------------------------------
   // reset right hand side vectors
   //-------------------------------------------------------------------

   if ( HYbs_ != NULL )
   {
      localNRows = localEndRow_ - localStartRow_ + 1;
      cols       = new int[localNRows];
      vals       = new double[localNRows];
      for (i = 0; i < localNRows; i++)
      {
         cols[i] = localStartRow_ + i - 1;
         vals[i] = setValue;
      }    
      for (i = 0; i < numRHSs_; i++) 
         if ( HYbs_[i] != NULL ) 
            HYPRE_IJVectorSetValues(HYbs_[i], localNRows, (const int *) cols,
                                    (const double *) vals);
      delete [] cols;
      delete [] vals;
   }
   normalEqnFlag_ &= 3;
   if ( HYnormalB_ != NULL ) 
   {
      HYPRE_IJVectorDestroy(HYnormalB_); 
      HYnormalB_ = NULL;
   }

   //-------------------------------------------------------------------
   // diagnostic message
   //-------------------------------------------------------------------

   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
      printf("%4d : HYPRE_LSC::leaving  resetRHSVector.\n",mypid_);
   return (0);
}

//***************************************************************************
// add nonzero entries into the matrix data structure (not in LSC but needed)
//---------------------------------------------------------------------------

int HYPRE_LinSysCore::sumIntoSystemMatrix(int row, int numValues,
                  const double* values, const int* scatterIndices)
{
   int i, j, index, colIndex, localRow;

   //-------------------------------------------------------------------
   // diagnostic message for high output level only
   //-------------------------------------------------------------------

   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 5 )
   {
      printf("%4d : HYPRE_LSC::entering sumIntoSystemMatrix.\n",mypid_);
      printf("%4d : row number = %d.\n", mypid_, row);
      if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 6 )
         for ( i = 0; i < numValues; i++ )
            printf("  %4d : row,col = %d %d, data = %e\n", mypid_, 
                   row+1, scatterIndices[i]+1, values[i]);
   }

   //-------------------------------------------------------------------
   // error checking
   //-------------------------------------------------------------------

   if ( systemAssembled_ == 1 )
   {
      printf("%4d : sumIntoSystemMatrix ERROR : matrix already assembled\n",
             mypid_);
      exit(1);
   }
   if ( row < localStartRow_ || row > localEndRow_ )
   {
      printf("%4d : sumIntoSystemMatrix ERROR : invalid row number %d.\n",
             mypid_,row);
      exit(1);
   }
   localRow = row - localStartRow_;
   if ( numValues > rowLengths_[localRow] )
   {
      printf("%4d : sumIntoSystemMatrix ERROR : row size too large.\n",mypid_);
      exit(1);
   }

   //-------------------------------------------------------------------
   // load the local matrix
   //-------------------------------------------------------------------

   for ( i = 0; i < numValues; i++ ) 
   {
      colIndex = scatterIndices[i];
      index    = hypre_BinarySearch(colIndices_[localRow], colIndex, 
                                    rowLengths_[localRow]);
      if ( index < 0 )
      {
         printf("%4d : sumIntoSystemMatrix ERROR - loading column",mypid_);
         printf("      that has not been declared before - %d.\n",colIndex);
         for ( j = 0; j < rowLengths_[localRow]; j++ ) 
            printf("       available column index = %d\n",
                   colIndices_[localRow][j]);
         exit(1);
      }
      colValues_[localRow][index] += values[i];
   }

#ifdef HAVE_AMGE
   HYPRE_LSI_AMGePutRow(row,numValues,(double*) values,(int*)scatterIndices);
#endif

   //-------------------------------------------------------------------
   // diagnostic message 
   //-------------------------------------------------------------------

   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 5 )
      printf("%4d : HYPRE_LSC::leaving  sumIntoSystemMatrix.\n",mypid_);
   return (0);
}

//***************************************************************************
// add nonzero entries into the matrix data structure
//---------------------------------------------------------------------------

int HYPRE_LinSysCore::sumIntoSystemMatrix(int numPtRows, const int* ptRows,
                            int numPtCols, const int* ptCols,
                            const double* const* values)
{
   int    i, j, k, index, colIndex, localRow, orderFlag=0; 
   int    *indptr, rowLeng, useOld;
   double *valptr, *auxValues;

   //-------------------------------------------------------------------
   // diagnostic message for high output level only
   //-------------------------------------------------------------------

   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 5 )
   {
      printf("%4d : HYPRE_LSC::entering sumIntoSystemMatrix(2).\n",mypid_);
      if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 6 )
      {
         for ( i = 0; i < numPtRows; i++ )
         {
            localRow = ptRows[i] - localStartRow_ + 1;
            for ( j = 0; j < numPtCols; j++ )
               printf("  %4d : row,col,val = %8d %8d %e\n",mypid_,
                      ptRows[i]+1, ptCols[j]+1, values[i][j]); 
         }
      }
   }

   //-------------------------------------------------------------------
   // error checking
   //-------------------------------------------------------------------

   if ( systemAssembled_ == 1 )
   {
      printf("sumIntoSystemMatrix ERROR : matrix already assembled\n");
      exit(1);
   }
   if (FEI_mixedDiagFlag_ && FEI_mixedDiag_ == NULL)
   {
      FEI_mixedDiag_ = new double[localEndRow_-localStartRow_+1];
      for ( i = 0; i < localEndRow_-localStartRow_+1; i++ )
         FEI_mixedDiag_[i] = 0.0;
   }

   //-------------------------------------------------------------------
   // load the local matrix
   //-------------------------------------------------------------------

   useOld = orderFlag = 0;
   if ( numPtCols == nStored_ && storedIndices_ != NULL )
   {
      for ( i = 0; i < numPtCols; i++ ) 
         if ( storedIndices_[i] != ptCols[i] ) break;
      if ( i == numPtCols ) useOld = 1;
   }
   if ( ! useOld ) 
   {
      for ( i = 1; i < numPtCols; i++ ) 
         if ( ptCols[i] < ptCols[i-1] ) { orderFlag = 1; break; }
      if ( orderFlag == 1 )
      { 
         if ( numPtCols != nStored_ )
         {
            if ( storedIndices_    != NULL ) delete [] storedIndices_;
            if ( auxStoredIndices_ != NULL ) delete [] auxStoredIndices_;
            storedIndices_ = new int[numPtCols];
            auxStoredIndices_ = new int[numPtCols];
            nStored_ = numPtCols;
         }
         for ( i = 0; i < numPtCols; i++ ) 
         {
            storedIndices_[i] = ptCols[i];
            auxStoredIndices_[i] = i; 
         }
         HYPRE_LSI_qsort1a(storedIndices_,auxStoredIndices_,0,numPtCols-1);
         for ( i = 0; i < numPtCols; i++ ) storedIndices_[i] = ptCols[i];
      }
      else
      {
         if ( storedIndices_    != NULL ) delete [] storedIndices_;
         if ( auxStoredIndices_ != NULL ) delete [] auxStoredIndices_;
         storedIndices_ = NULL;
         auxStoredIndices_ = NULL;
         nStored_ = 0;
      }
   }
   for ( i = 0; i < numPtRows; i++ ) 
   {
      localRow  = ptRows[i] - localStartRow_ + 1;
      indptr    = colIndices_[localRow];
      valptr    = colValues_[localRow];
      rowLeng   = rowLengths_[localRow];
      auxValues = (double *) values[i];
      index     = 0;
      for ( j = 0; j < numPtCols; j++ ) 
      {
         if ( storedIndices_ )
            colIndex = storedIndices_[auxStoredIndices_[j]] + 1;
         else
            colIndex = ptCols[j] + 1;

         if (FEI_mixedDiag_ && ptRows[i] == ptCols[j] && numPtRows > 1)
            FEI_mixedDiag_[ptCols[numPtCols-1]-localStartRow_+1] += auxValues[j]; 

         while ( index < rowLeng && indptr[index] < colIndex ) index++; 
         if ( index >= rowLeng )
         {
            printf("%4d : sumIntoSystemMatrix ERROR - loading column",mypid_);
            printf(" that has not been declared before - %d (row=%d).\n",
                   colIndex, ptRows[i]+1);
            for ( k = 0; k < rowLeng; k++ ) 
               printf("       available column index = %d\n", indptr[k]);
            exit(1);
         }
         if ( auxStoredIndices_ )
            valptr[index] += auxValues[auxStoredIndices_[j]];
         else
            valptr[index] += auxValues[j];
      }
   }
#ifdef HAVE_AMGE
   HYPRE_LSI_AMGePutRow(localRow,numPtCols,(double*) values[i],(int*)ptCols);
#endif

   //-------------------------------------------------------------------
   // diagnostic message 
   //-------------------------------------------------------------------

   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 5 )
      printf("%4d : HYPRE_LSC::leaving  sumIntoSystemMatrix(2).\n",mypid_);
   return (0);
}

//***************************************************************************
// add nonzero entries into the matrix data structure 
//---------------------------------------------------------------------------

int HYPRE_LinSysCore::sumIntoSystemMatrix(int numPtRows, const int* ptRows,
                       int numPtCols, const int* ptCols, int numBlkRows, 
                       const int* blkRows, int numBlkCols, const int* blkCols,
                       const double* const* values)
{
   (void) numBlkRows;
   (void) blkRows;
   (void) numBlkCols;
   (void) blkCols;

   return(sumIntoSystemMatrix(numPtRows, ptRows, numPtCols, ptCols, values));
}

//***************************************************************************
// put nonzero entries into the matrix data structure
//---------------------------------------------------------------------------

int HYPRE_LinSysCore::putIntoSystemMatrix(int numPtRows, const int* ptRows,
                            int numPtCols, const int* ptCols,
                            const double* const* values)
{
   int    i, j, localRow, newLeng, *tempInd, colIndex, index, localNRows;
   int    sortFlag;
   double *tempVal;

   //-------------------------------------------------------------------
   // diagnostic message for high output level only
   //-------------------------------------------------------------------

   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 5 )
      printf("%4d : HYPRE_LSC::entering putIntoSystemMatrix.\n",mypid_);

   //-------------------------------------------------------------------
   // error checking
   //-------------------------------------------------------------------

   if ( systemAssembled_ == 1 )
   {
      printf("putIntoSystemMatrix ERROR : matrix already assembled\n");
      exit(1);
   }
   if ( numPtRows <= 0 || numPtCols <= 0 )
   {
      printf("%4d : putIntoSystemMatrix ERROR : invalid numPt.\n",mypid_);
      return (-1);
   }

   //-------------------------------------------------------------------
   // in case the matrix is loaded from scratch
   //-------------------------------------------------------------------

   if ( rowLengths_ == NULL && colIndices_ == NULL )
   {
      localNRows = localEndRow_ - localStartRow_ + 1;
      if ( localNRows > 0 ) 
      {
         rowLengths_ = new int[localNRows];
         colIndices_ = new int*[localNRows];
         colValues_  = new double*[localNRows];
      }
      for ( i = 0; i < localNRows; i++ ) 
      {
         rowLengths_[i] = 0;
         colIndices_[i] = NULL;
         colValues_[i]  = NULL;
      }
   }

   //-------------------------------------------------------------------
   // first adjust memory allocation (conservative)
   //-------------------------------------------------------------------

   for ( i = 0; i < numPtRows; i++ ) 
   {
      localRow = ptRows[i] - localStartRow_ + 1;
      if ( rowLengths_[localRow] > 0 )
      {
         newLeng  = rowLengths_[localRow] + numPtCols;
         tempInd  = new int[newLeng];
         tempVal  = new double[newLeng];
         for ( j = 0; j < rowLengths_[localRow]; j++ ) 
         {
            tempVal[j] = colValues_[localRow][j];
            tempInd[j] = colIndices_[localRow][j];
         }
         delete [] colValues_[localRow];
         delete [] colIndices_[localRow];
         colValues_[localRow] = tempVal;
         colIndices_[localRow] = tempInd;
      }
      else
      {
         if ( colIndices_[localRow] != NULL ) delete [] colIndices_[localRow];
         if ( colValues_[localRow]  != NULL ) delete [] colValues_[localRow];
         colIndices_[localRow] = new int[numPtCols];
         colValues_[localRow] = new double[numPtCols];
      }
   }

   //-------------------------------------------------------------------
   // load the local matrix
   //-------------------------------------------------------------------

   for ( i = 0; i < numPtRows; i++ ) 
   {
      localRow = ptRows[i] - localStartRow_ + 1;
      if ( rowLengths_[localRow] > 0 )
      {
         newLeng  = rowLengths_[localRow];
         tempInd  = colIndices_[localRow];
         tempVal  = colValues_[localRow];
         for ( j = 0; j < numPtCols; j++ ) 
         {
            colIndex = ptCols[j] + 1;
            index    = hypre_BinarySearch(tempInd, colIndex, newLeng);
            if ( index < 0 )
            {
               tempInd[rowLengths_[localRow]]   = colIndex;
               tempVal[rowLengths_[localRow]++] = values[i][j];
            }
            else tempVal[index] = values[i][j];
         }
         newLeng  = rowLengths_[localRow];
         qsort1( tempInd, tempVal, 0, newLeng-1 );
      }
      else
      {
         tempInd = colIndices_[localRow];
         tempVal = colValues_[localRow];
         for ( j = 0; j < numPtCols; j++ ) 
         {
            colIndex = ptCols[j] + 1;
            tempInd[j] = colIndex;
            tempVal[j] = values[i][j];
         }
         sortFlag = 0;
         for ( j = 1; j < numPtCols; j++ ) 
            if ( tempInd[j] < tempInd[j-1] ) sortFlag = 1;
         rowLengths_[localRow] = numPtCols;
         if ( sortFlag == 1 ) qsort1( tempInd, tempVal, 0, numPtCols-1 );
      }
   }

   //-------------------------------------------------------------------
   // diagnostic message 
   //-------------------------------------------------------------------

   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 5 )
      printf("%4d : HYPRE_LSC::leaving  putIntoSystemMatrix.\n",mypid_);

   return(0);
}

//***************************************************************************
// get the length of the local row
//---------------------------------------------------------------------------

int HYPRE_LinSysCore::getMatrixRowLength(int row, int& length)
{
   int    *colInd, rowLeng;
   double *colVal;
   HYPRE_ParCSRMatrix A_csr;

   if ((row+1) < localStartRow_ || (row+1) > localEndRow_) return (-1);
   if ( systemAssembled_ == 0 )
   {
      if ( rowLengths_ == NULL ) return (-1);
      length = rowLengths_[row+1];
   }
   else
   {
      HYPRE_IJMatrixGetObject(currA_, (void **) &A_csr);
      HYPRE_ParCSRMatrixGetRow(A_csr,row,&rowLeng,&colInd,&colVal);
      length = rowLeng;
      HYPRE_ParCSRMatrixRestoreRow(A_csr,row,&rowLeng,&colInd,&colVal);
   }
   return(0);
}

//***************************************************************************
// get the data of a local row
//---------------------------------------------------------------------------

int HYPRE_LinSysCore::getMatrixRow(int row, double* coefs, int* indices,
				  int len, int& rowLength)
{
   int    i, rowIndex, rowLeng, *colInd, minLeng;
   double *colVal;
   HYPRE_ParCSRMatrix A_csr;

   if ( systemAssembled_ == 0 )
   {
      rowIndex = row + 1;
      if (rowIndex < localStartRow_ || rowIndex > localEndRow_) return (-1);
      if ( rowLengths_ == NULL || colIndices_ == NULL ) return (-1);
      rowLeng = rowLengths_[rowIndex];
      colInd  = colIndices_[rowIndex];
      colVal  = colValues_[rowIndex];
      minLeng = len;
      if ( minLeng > rowLeng ) minLeng = rowLeng;
      for( i = 0; i < minLeng; i++ ) 
      {
         coefs[i] = colVal[i];
         indices[i] = colInd[i];
      }
      rowLength = rowLeng;
   }
   else
   {
      HYPRE_IJMatrixGetObject(currA_, (void **) &A_csr);
      rowIndex = row + 1;
      if (rowIndex < localStartRow_ || rowIndex > localEndRow_) return (-1);
      rowIndex--;
      HYPRE_ParCSRMatrixGetRow(A_csr,rowIndex,&rowLeng,&colInd,&colVal);
      minLeng = len;
      if ( minLeng > rowLeng ) minLeng = rowLeng;
      for( i = 0; i < minLeng; i++ ) 
      {
         coefs[i] = colVal[i];
         indices[i] = colInd[i];
      }
      HYPRE_ParCSRMatrixRestoreRow(A_csr,rowIndex,&rowLeng,&colInd,&colVal);
      rowLength = rowLeng;
   }
   return(0);
}

//***************************************************************************
// input is 1-based, but HYPRE vectors are 0-based
//---------------------------------------------------------------------------

int HYPRE_LinSysCore::sumIntoRHSVector(int num, const double* values,
                       const int* indices)
{
   int    i, ierr, *localInds;

   //-------------------------------------------------------------------
   // diagnostic message
   //-------------------------------------------------------------------

   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 5 )
   {
      printf("%4d : HYPRE_LSC::entering sumIntoRHSVector.\n", mypid_);
      if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 6 )
      {
         for ( i = 0; i < num; i++ )
            printf("%d : sumIntoRHSVector - %d = %e.\n", mypid_, indices[i], 
                         values[i]);
      }
   }

   //-------------------------------------------------------------------
   // change the incoming indices to 0-based before loading
   //-------------------------------------------------------------------

   localInds = new int[num];
   for ( i = 0; i < num; i++ ) // change to 0-based
   {
#if defined(NOFEI)
      if ( indices[i] >= localStartRow_  && indices[i] <= localEndRow_ )
         localInds[i] = indices[i] - 1;
#else
      if ((indices[i]+1) >= localStartRow_  && (indices[i]+1) <= localEndRow_)
         localInds[i] = indices[i];
#endif
      else
      {
         printf("%d : sumIntoRHSVector ERROR - index %d out of range.\n",
                      mypid_, indices[i]);
         exit(1);
      }
   }

   ierr = HYPRE_IJVectorAddToValues(HYb_, num, (const int *) localInds, 
                                    (const double *) values);
   //assert(!ierr);

   delete [] localInds;

   //-------------------------------------------------------------------
   // diagnostic message
   //-------------------------------------------------------------------

   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 5 )
      printf("%4d : HYPRE_LSC::leaving  sumIntoRHSVector.\n", mypid_);
   return (0);
}

//***************************************************************************
// This function scatters (puts) values into the linear-system's
// currently selected RHS vector.
// num is how many values are being passed,
// indices holds the global 0-based 'row-numbers' into which the values go,
// and values holds the actual coefficients to be scattered.
//---------------------------------------------------------------------------

int HYPRE_LinSysCore::putIntoRHSVector(int num, const double* values,
                                       const int* indices)
{
   int    i, index;

   if ((numRHSs_ == 0) && (HYb_ == NULL)) return(0);

   for ( i = 0; i < num; i++ )
   {
      index = indices[i];
      if (index < localStartRow_-1 || index >= localEndRow_) continue;
      HYPRE_IJVectorSetValues(HYb_, 1, (const int *) &index,
                              (const double *) &(values[i]));
   }
   return(0);
}

//***************************************************************************
// This function gets values from the linear-system's
// currently selected RHS vector.
// num is how many values are being requested,
// indices holds the requested global 0-based 'row-numbers',
// and values will hold the returned coefficients.
//---------------------------------------------------------------------------

int HYPRE_LinSysCore::getFromRHSVector(int num, double* values,
                                       const int* indices)
{
   int    i, index;

   if ((numRHSs_ == 0) && (HYb_ == NULL)) return(0);

   for ( i = 0; i < num; i++ )
   {
      index = indices[i];
      if (index < localStartRow_-1 || index >= localEndRow_) continue;
      HYPRE_IJVectorGetValues(HYb_,1,&index,&(values[i]));
   }
   return(0);
}

//***************************************************************************
// start assembling the matrix into its internal format
//---------------------------------------------------------------------------

int HYPRE_LinSysCore::matrixLoadComplete()
{
   int    i, j, ierr, numLocalEqns, leng, eqnNum, nnz, *newColInd=NULL;
   int    maxRowLeng, newLeng, rowSize, *colInd, nrows;
   double *newColVal=NULL, *colVal, value;
   char   fname[40];
   FILE   *fp;
   HYPRE_ParCSRMatrix A_csr;
   HYPRE_ParVector    b_csr;
   HYPRE_SlideReduction *slideObj;

   //-------------------------------------------------------------------
   // diagnostic message
   //-------------------------------------------------------------------

   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
      printf("%4d : HYPRE_LSC::entering matrixLoadComplete.\n",mypid_);

   //-------------------------------------------------------------------
   // Write MLI FEData information to a file 
   //-------------------------------------------------------------------

#ifdef HAVE_MLI
   if ( haveFEData_ && feData_ != NULL )
   {
      char filename[100];
      if ( (HYOutputLevel_ & HYFEI_PRINTFEINFO) )
      {
         strcpy( filename, "fedata" );
         HYPRE_LSI_MLIFEDataWriteToFile( feData_, filename );
      }
   }
#endif

   if ( matrixPartition_ == 2 ) matrixPartition_ = 1;

   //-------------------------------------------------------------------
   // if the matrix has not been assembled or it has been reset
   //-------------------------------------------------------------------

   if ( systemAssembled_ != 1 )
   {
      //----------------------------------------------------------------
      // set system matrix initialization parameters 
      //----------------------------------------------------------------

      ierr = HYPRE_IJMatrixSetRowSizes(HYA_, rowLengths_);
      ierr = HYPRE_IJMatrixInitialize(HYA_);
      //assert(!ierr);

      //----------------------------------------------------------------
      // load the matrix stored locally to a HYPRE matrix
      //----------------------------------------------------------------

      numLocalEqns = localEndRow_ - localStartRow_ + 1;
      if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 4 )
      {
         printf("%4d : HYPRE_LSC::matrixLoadComplete - NEqns = %d.\n",
                 mypid_, numLocalEqns);
      }
      maxRowLeng = 0;
      for ( i = 0; i < numLocalEqns; i++ )
      {
         leng   = rowLengths_[i];
         if ( leng > maxRowLeng ) maxRowLeng = leng;
      }
      if ( maxRowLeng > 0 )
      {
         newColInd = new int[maxRowLeng];
         newColVal = new double[maxRowLeng];
      }
      nnz = 0;
      for ( i = 0; i < numLocalEqns; i++ )
      {
         eqnNum  = localStartRow_ - 1 + i;
         leng    = rowLengths_[i];
         newLeng = 0;
         for ( j = 0; j < leng; j++ ) 
         {
            if ( habs(colValues_[i][j]) >= truncThresh_ )
            {
               newColInd[newLeng]   = colIndices_[i][j] - 1;
               newColVal[newLeng++] = colValues_[i][j];
            }
         }
         HYPRE_IJMatrixSetValues(HYA_, 1, &newLeng,(const int *) &eqnNum,
                       (const int *) newColInd, (const double *) newColVal);
         delete [] colValues_[i];
         if ( memOptimizerFlag_ != 0 ) delete [] colIndices_[i];
         nnz += newLeng;
      }
      if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
      {
         printf("%4d : HYPRE_LSC::matrixLoadComplete - nnz = %d.\n",
                 mypid_, nnz);
      }
      delete [] colValues_;
      colValues_ = NULL;
      if ( memOptimizerFlag_ != 0 ) 
      {
         delete [] colIndices_;
         colIndices_ = NULL;
      }
      if ( maxRowLeng > 0 )
      {
         delete [] newColInd;
         delete [] newColVal;
      }
      HYPRE_IJMatrixAssemble(HYA_);
      systemAssembled_ = 1;
      projectCurrSize_ = 0;
      currA_ = HYA_;
      currB_ = HYb_;
      currX_ = HYx_;
      currR_ = HYr_;
      if (slideObj_ != NULL)
      {
         slideObj = (HYPRE_SlideReduction *) slideObj_; 
         delete slideObj;
      }
      slideObj_ = NULL;
   }

   //-------------------------------------------------------------------
   // diagnostics : print the matrix and rhs to files 
   //-------------------------------------------------------------------

   if ( (HYOutputLevel_ & HYFEI_PRINTMAT) &&
        (!(HYOutputLevel_ & HYFEI_PRINTREDMAT) ) )
   {
      if ( HYOutputLevel_ & HYFEI_PRINTPARCSRMAT )
      {
         printf("%4d : HYPRE_LSC::print the matrix/rhs to files(1)\n",mypid_);
         HYPRE_IJMatrixGetObject(HYA_, (void **) &A_csr);
         sprintf(fname, "HYPRE_Mat");
         HYPRE_ParCSRMatrixPrint( A_csr, fname );
         HYPRE_IJVectorGetObject(HYb_, (void **) &b_csr);
         sprintf(fname, "HYPRE_RHS");
         HYPRE_ParVectorPrint( b_csr, fname );
      }
      else
      {
         printf("%4d : HYPRE_LSC::print the matrix/rhs to files(2)\n",mypid_);
         HYPRE_IJMatrixGetObject(HYA_, (void **) &A_csr);
         sprintf(fname, "hypre_mat.out.%d",mypid_);
         fp = fopen(fname,"w");
         nrows = localEndRow_ - localStartRow_ + 1;
         nnz = 0;
         for ( i = localStartRow_-1; i <= localEndRow_-1; i++ )
         {
            HYPRE_ParCSRMatrixGetRow(A_csr,i,&rowSize,&colInd,&colVal);
            for ( j = 0; j < rowSize; j++ ) if ( colVal[j] != 0.0 ) nnz++;
            HYPRE_ParCSRMatrixRestoreRow(A_csr,i,&rowSize,&colInd,&colVal);
         }
         fprintf(fp, "%6d  %7d \n", nrows, nnz);
         for ( i = localStartRow_-1; i <= localEndRow_-1; i++ )
         {
            HYPRE_ParCSRMatrixGetRow(A_csr,i,&rowSize,&colInd,&colVal);
            for (j = 0; j < rowSize; j++)
            {
               if ( colVal[j] != 0.0 )
                  fprintf(fp, "%6d  %6d  %25.16e \n",i+1,colInd[j]+1,colVal[j]);
            }
             HYPRE_ParCSRMatrixRestoreRow(A_csr,i,&rowSize,&colInd,&colVal);
         }
         fclose(fp);
         sprintf(fname, "hypre_rhs.out.%d",mypid_);
         fp = fopen(fname,"w");
         fprintf(fp, "%6d \n", nrows);
         for ( i = localStartRow_-1; i <= localEndRow_-1; i++ )
         {
            HYPRE_IJVectorGetValues(HYb_, 1, &i, &value);
            fprintf(fp, "%6d  %25.16e \n", i+1, value);
         }
         fclose(fp);
         MPI_Barrier(comm_);
      }
      if ( HYOutputLevel_ & HYFEI_STOPAFTERPRINT ) exit(1);
   }
   if (FEI_mixedDiagFlag_)
   {
      for ( i = 0; i < localEndRow_-localStartRow_+1; i++ )
      {
         FEI_mixedDiag_[i] *= 0.125;
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
            printf("Mixed diag %5d = %e\n", i, FEI_mixedDiag_[i]);
      }
   }

   //-------------------------------------------------------------------
   // diagnostic message
   //-------------------------------------------------------------------

   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
      printf("%4d : HYPRE_LSC::leaving  matrixLoadComplete.\n",mypid_);
   return (0);
}

//***************************************************************************
// put nodal information
//---------------------------------------------------------------------------

int HYPRE_LinSysCore::putNodalFieldData(int fieldID, int fieldSize,
                       int* nodeNumbers, int numNodes, const double* data)
{
   int    i, j, **nodeFieldIDs, nodeFieldID, *procNRows, nRows, errCnt;
   int    blockID, *blockIDs, *eqnNumbers, *iArray, newNumEdges;
   //int   checkFieldSize;
   int    *aleNodeNumbers, index, newNumNodes, numEdges;
   double *newData;

   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
   {
      printf("%4d : HYPRE_LSC::entering putNodalFieldData.\n",mypid_);
      if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 4 && mypid_ == 0 )
      {
         printf("      putNodalFieldData : fieldSize = %d\n", fieldSize);
         printf("      putNodalFieldData : fieldID   = %d\n", fieldID);
         printf("      putNodalFieldData : numNodes  = %d\n", numNodes);
      }
   }

   //-------------------------------------------------------------------
   // This part is for loading the nodal coordinate information.
   // The node IDs in nodeNumbers are the one used in FEI (and thus
   // corresponds to the ones in the system matrix using lookup)
   //-------------------------------------------------------------------

   if ( fieldID == -3 || fieldID == -25333 )
   {
      if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 5 )
      {
         for ( i = 0; i < numNodes; i++ )
            for ( j = 0; j < fieldSize; j++ )
               printf("putNodalFieldData : %4d %2d = %e\n",i,j,
                      data[i*fieldSize+j]);
      }    
      if ( HYPreconID_ == HYMLI && lookup_ != NULL )
      {
         blockIDs       = (int *) lookup_->getElemBlockIDs();
         blockID        = blockIDs[0];
         nodeFieldIDs   = (int **) lookup_->getFieldIDsTable(blockID);
         nodeFieldID    = nodeFieldIDs[0][0];
         //checkFieldSize = lookup_->getFieldSize(nodeFieldID);
         //assert( checkFieldSize == fieldSize );
         eqnNumbers  = new int[numNodes];
         newData     = new double[numNodes*fieldSize];
         newNumNodes = 0;
         for ( i = 0; i < numNodes*fieldSize; i++ ) newData[i] = -99999.9;
         for ( i = 0; i < numNodes; i++ )
         { 
            index = lookup_->getEqnNumber(nodeNumbers[i],nodeFieldID);

/* ======
This should ultimately be taken out even for newer ale3d implementation
   =====*/
            if ( index >= localStartRow_-1 && index < localEndRow_)
            {
               if ( newData[newNumNodes*fieldSize] == -99999.9 )
               { 
                  for ( j = 0; j < fieldSize; j++ ) 
                     newData[newNumNodes*fieldSize+j] = data[i*fieldSize+j];
                  eqnNumbers[newNumNodes++] = index;
               }
            }
         }
         nRows = localEndRow_ - localStartRow_ + 1;
         if ( MLI_NodalCoord_ == NULL )
         {
            MLI_EqnNumbers_ = new int[nRows/fieldSize];
            for (i=0; i<nRows/fieldSize; i++) 
               MLI_EqnNumbers_[i] = localStartRow_ - 1 + i * fieldSize;
            MLI_NodalCoord_ = new double[localEndRow_-localStartRow_+1];
            for (i=0; i<nRows; i++) MLI_NodalCoord_[i] = -99999.0;
            MLI_FieldSize_  = fieldSize;
            MLI_NumNodes_   = nRows / fieldSize;
         }
         for ( i = 0; i < newNumNodes; i++ )
         {
            index = eqnNumbers[i] - localStartRow_ + 1;
            for ( j = 0; j < fieldSize; j++ ) 
               MLI_NodalCoord_[index+j] = newData[i*fieldSize+j];
         }
         delete [] eqnNumbers;
         delete [] newData;
         errCnt = 0;
         for (i = 0; i < nRows; i++)
            if (MLI_NodalCoord_[i] == -99999.0) errCnt++;
         if (errCnt > 0)
            printf("putNodalFieldData ERROR:incomplete nodal coordinates (%d %d).\n",
                   errCnt, nRows);
      }
      else
      {
         if (nodeNumbers != NULL && numNodes != 0)
         {
            printf("putNodalFieldData WARNING : \n");
            printf("    set nodeNumbers = NULL, set numNodes = 0.\n");
         }
         MLI_NodalCoord_ = new double[localEndRow_-localStartRow_+1];
         for (i=0; i<nRows; i++) MLI_NodalCoord_[i] = data[i];
      }
   }

   //-------------------------------------------------------------------
   // This part is for loading the edge to (hypre-compatible) node list 
   // for AMS (the node list is ordered with the edge equation numbers
   // and the node numbers are true node equation numbers passed in by
   // the application which obtains the true node eqn numbers via the
   // nodal FEI) ===> EdgeNodeList  
   //-------------------------------------------------------------------

   if (fieldID == -4)
   {
      if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 5)
      {
         for (i = 0; i < numNodes; i++)
            for (j = 0; j < fieldSize; j++)
               printf("putNodalFieldData : %4d %2d = %e\n",i,j,
                      data[i*fieldSize+j]);
      }    
      if (lookup_ != NULL && fieldSize == 2 &&
           numNodes > 0)
      {
         blockIDs       = (int *) lookup_->getElemBlockIDs();
         blockID        = blockIDs[0];
         nodeFieldIDs   = (int **) lookup_->getFieldIDsTable(blockID);
         nodeFieldID    = nodeFieldIDs[0][0];
         numEdges    = numNodes;
         eqnNumbers  = new int[numEdges];
         iArray      = new int[numEdges*fieldSize];
         newNumEdges = 0;
         for (i = 0; i < numEdges; i++)
         { 
            index = lookup_->getEqnNumber(nodeNumbers[i],nodeFieldID);
            if (index >= localStartRow_-1 && index < localEndRow_)
            {
               for (j = 0; j < fieldSize; j++) 
                  iArray[newNumEdges*fieldSize+j] = (int) data[i*fieldSize+j];
               eqnNumbers[newNumEdges++] = index;
            }
         }
         nRows = localEndRow_ - localStartRow_ + 1;
         if (AMSData_.EdgeNodeList_ != NULL) delete [] AMSData_.EdgeNodeList_;
         AMSData_.EdgeNodeList_ = NULL;
         if (newNumEdges > 0)
         {
            AMSData_.numEdges_ = nRows;
            AMSData_.EdgeNodeList_ = new int[nRows*fieldSize];
            for (i = 0; i < nRows*fieldSize; i++)
               AMSData_.EdgeNodeList_[i] = -99999;
            for (i = 0; i < newNumEdges; i++)
            {
               index = eqnNumbers[i] - localStartRow_ + 1;
               for (j = 0; j < fieldSize; j++ ) 
                  AMSData_.EdgeNodeList_[index*fieldSize+j] = iArray[i*fieldSize+j];
            }
            errCnt = 0;
            for (i = 0; i < nRows*fieldSize; i++)
               if (AMSData_.EdgeNodeList_[i] == -99999) errCnt++;
            if (errCnt > 0)
               printf("putNodalFieldData ERROR:incomplete AMS edge vertex list\n");
         }
         delete [] eqnNumbers;
         delete [] iArray;
      }
   }

   //-------------------------------------------------------------------
   // This part is for converting node numbers to equations as well as
   // for loading the nodal coordinate information
   // (stored in NodeNumbers, NodalCoord, numNodes, numLocalNodes)
   //-------------------------------------------------------------------

   if (fieldID == -5)
   {
      if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 5)
      {
         for (i = 0; i < numNodes; i++)
            for (j = 0; j < fieldSize; j++)
               printf("putNodalFieldData : %4d %2d = %e\n",i,j,
                      data[i*fieldSize+j]);
      }    
      if (lookup_ != NULL && fieldSize == 3)
      {
         blockIDs       = (int *) lookup_->getElemBlockIDs();
         blockID        = blockIDs[0];
         nodeFieldIDs   = (int **) lookup_->getFieldIDsTable(blockID);
         nodeFieldID    = nodeFieldIDs[0][0];
         if (AMSData_.NodeNumbers_ != NULL) delete [] AMSData_.NodeNumbers_;
         if (AMSData_.NodalCoord_  != NULL) delete [] AMSData_.NodalCoord_;
         AMSData_.NodeNumbers_ = NULL;
         AMSData_.NodalCoord_  = NULL;
         AMSData_.numNodes_ = 0;
         if (numNodes > 0)
         {
            AMSData_.numNodes_ = numNodes;
            AMSData_.numLocalNodes_ = localEndRow_ - localStartRow_ + 1;
            AMSData_.NodeNumbers_ = new int[numNodes];
            AMSData_.NodalCoord_  = new double[fieldSize*numNodes];
            for (i = 0; i < numNodes; i++)
            {
               index = lookup_->getEqnNumber(nodeNumbers[i],nodeFieldID);
               AMSData_.NodeNumbers_[i] = index;
               for (j = 0; j < fieldSize; j++)
                  AMSData_.NodalCoord_[i*fieldSize+j] = data[i*fieldSize+j];
            }
         }
      }
   }

   //-------------------------------------------------------------------
   // this is needed to set up the correct node equation map
   // (the FEI remaps the node IDs in the incoming nodeNumbers array.
   //  to revert to the original ALE3D node numbers, it is passed in
   //  here as data)
   //-------------------------------------------------------------------

   else if ( fieldID == -49773 )
   {
      if ( HYPreconID_ == HYMLI && lookup_ != NULL )
      {
         blockIDs       = (int *) lookup_->getElemBlockIDs();
         blockID        = blockIDs[0];
         nodeFieldIDs   = (int **) lookup_->getFieldIDsTable(blockID);
         nodeFieldID    = nodeFieldIDs[0][0];
         //checkFieldSize = lookup_->getFieldSize(nodeFieldID);
         assert( fieldSize == 1 );
         aleNodeNumbers = new int[numNodes];
         eqnNumbers     = new int[numNodes];
         for ( i = 0; i < numNodes; i++ )
         {
            aleNodeNumbers[i] = (int) data[i];
            eqnNumbers[i] = lookup_->getEqnNumber(nodeNumbers[i],nodeFieldID);
         }
         procNRows = new int[numProcs_];
         for ( i = 0; i < numProcs_; i++ ) procNRows[i] = 0;
         procNRows[mypid_] = localEndRow_;
         iArray = procNRows;
         procNRows  = new int[numProcs_+1];
         for ( i = 0; i <= numProcs_; i++ ) procNRows[i] = 0;
         MPI_Allreduce(iArray,&(procNRows[1]),numProcs_,MPI_INT,MPI_SUM,comm_);
         delete [] iArray;
         HYPRE_LSI_MLICreateNodeEqnMap(HYPrecon_, numNodes, aleNodeNumbers,
                                       eqnNumbers, procNRows);
         delete [] procNRows;
         delete [] eqnNumbers;
         delete [] aleNodeNumbers;
      } 
   } 
   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
      printf("%4d : HYPRE_LSC::leaving  putNodalFieldData.\n",mypid_);
   return (0);
}

#if 0
/* ------------------------------------------------------ */
/* This section has been replaced due to changes in ale3d */
/* ------------------------------------------------------ */
int HYPRE_LinSysCore::putNodalFieldData(int fieldID, int fieldSize,
                       int* nodeNumbers, int numNodes, const double* data)
{
   int    i, **nodeFieldIDs, nodeFieldID, numFields, *procNRows;
   int    blockID, *blockIDs, *eqnNumbers, *iArray, checkFieldSize;
   int    *aleNodeNumbers;

   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
   {
      printf("%4d : HYPRE_LSC::entering putNodalFieldData.\n",mypid_);
      if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
      {
         printf("      putNodalFieldData : fieldSize = %d\n", fieldSize);
         printf("      putNodalFieldData : fieldID   = %d\n", fieldID);
         printf("      putNodalFieldData : numNodes  = %d\n", numNodes);
      }
   }

   //-------------------------------------------------------------------
   // This part is for loading the nodal coordinate information.
   // The node IDs in nodeNumbers are the one used in FEI (and thus
   // corresponds to the ones in the system matrix using lookup)
   //-------------------------------------------------------------------

   if ( fieldID == -3 || fieldID == -25333 )
   {
      if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 4 )
      {
         for ( int i = 0; i < numNodes; i++ )
            for ( int j = 0; j < fieldSize; j++ )
               printf("putNodalFieldData : %4d %2d = %e\n",i,j,
                      data[i*fieldSize+j]);
      }    
      if ( HYPreconID_ == HYMLI && lookup_ != NULL )
      {
         blockIDs       = (int *) lookup_->getElemBlockIDs();
         blockID        = blockIDs[0];
         nodeFieldIDs   = (int **) lookup_->getFieldIDsTable(blockID);
         nodeFieldID    = nodeFieldIDs[0][0];
         checkFieldSize = lookup_->getFieldSize(nodeFieldID);
         //assert( checkFieldSize == fieldSize );
         eqnNumbers = new int[numNodes];
         for ( i = 0; i < numNodes; i++ )
            eqnNumbers[i] = lookup_->getEqnNumber(nodeNumbers[i],nodeFieldID);
         HYPRE_LSI_MLILoadNodalCoordinates(HYPrecon_, numNodes, checkFieldSize,
                       eqnNumbers, fieldSize, (double *) data);
         delete [] eqnNumbers;
      }    
   }    

   //-------------------------------------------------------------------
   // this is needed to set up the correct node equation map
   // (the FEI remaps the node IDs in the incoming nodeNumbers array.
   //  to revert to the original ALE3D node numbers, it is passed in
   //  here as data)
   //-------------------------------------------------------------------

   else if ( fieldID == -49773 )
   {
      if ( HYPreconID_ == HYMLI && lookup_ != NULL )
      {
         blockIDs       = (int *) lookup_->getElemBlockIDs();
         blockID        = blockIDs[0];
         nodeFieldIDs   = (int **) lookup_->getFieldIDsTable(blockID);
         nodeFieldID    = nodeFieldIDs[0][0];
         checkFieldSize = lookup_->getFieldSize(nodeFieldID);
         assert( fieldSize == 1 );
         aleNodeNumbers = new int[numNodes];
         eqnNumbers     = new int[numNodes];
         for ( i = 0; i < numNodes; i++ )
         {
            aleNodeNumbers[i] = (int) data[i];
            eqnNumbers[i] = lookup_->getEqnNumber(nodeNumbers[i],nodeFieldID);
         }
         procNRows = new int[numProcs_];
         for ( i = 0; i < numProcs_; i++ ) procNRows[i] = 0;
         procNRows[mypid_] = localEndRow_;
         iArray = procNRows;
         procNRows  = new int[numProcs_+1];
         for ( i = 0; i <= numProcs_; i++ ) procNRows[i] = 0;
         MPI_Allreduce(iArray,&(procNRows[1]),numProcs_,MPI_INT,MPI_SUM,
                       comm_);
         delete [] iArray;
         HYPRE_LSI_MLICreateNodeEqnMap(HYPrecon_, numNodes, aleNodeNumbers,
                                       eqnNumbers, procNRows);
         delete [] procNRows;
         delete [] eqnNumbers;
         delete [] aleNodeNumbers;
      } 
   } 
   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
      printf("%4d : HYPRE_LSC::leaving  putNodalFieldData.\n",mypid_);
   return (0);
}
#endif

//***************************************************************************
// This function must enforce an essential boundary condition on each local
// equation in 'globalEqn'. This means, that the following modification
// should be made to A and b, for each globalEqn:
//
// for(each local equation i){
//    for(each column j in row i) {
//       if (i==j) b[i] = gamma/alpha;
//       else b[j] -= (gamma/alpha)*A[j,i];
//    }
// }
// all of row 'globalEqn' and column 'globalEqn' in A should be zeroed,
// except for 1.0 on the diagonal.
//---------------------------------------------------------------------------

int HYPRE_LinSysCore::enforceEssentialBC(int* globalEqn, double* alpha,
                                          double* gamma1, int leng)
{
   int    i, j, k, localEqnNum, colIndex, rowSize, *colInd, *iarray;
   int    numLocalRows, eqnNum, rowSize2, *colInd2, numLabels, *labels;
   int    **i2array, count;
   double rhs_term, val, *colVal2, *colVal, **d2array;

   //-------------------------------------------------------------------
   // diagnostic message and error checking
   // (this function should be called before matrixLoadComplete)
   //-------------------------------------------------------------------

   if ( (HYOutputLevel_ & HYFEI_IMPOSENOBC) != 0 ) return 0;
   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 5 )
      printf("%4d : HYPRE_LSC::entering enforceEssentialBC.\n",mypid_);
   if ( systemAssembled_ )
   {
      printf("enforceEssentialBC ERROR : system assembled already.\n");
      exit(1);
   }

   //-------------------------------------------------------------------
   // if matrix partitioning is requested, do it.
   //-------------------------------------------------------------------

   numLocalRows = localEndRow_ - localStartRow_ + 1;
   if ( matrixPartition_ == 1 && HYPreconID_ == HYMLI )
   {
      HYPRE_LSI_PartitionMatrix(numLocalRows,localStartRow_,rowLengths_, 
                                colIndices_,colValues_, &numLabels, &labels); 
      HYPRE_LSI_MLILoadMaterialLabels(HYPrecon_, numLabels, labels);
      free( labels );
      matrixPartition_ = 2;
   }
      
   //-------------------------------------------------------------------
   // examine each row individually
   //-------------------------------------------------------------------

   //**/================================================================
   //**/ The following is for multiple right hand side (Mar 2009)
   if (mRHSFlag_ == 1 && currentRHS_ != 0 & mRHSNumGEqns_ > 0)
   {
      for( i = 0; i < leng; i++ )
      {
         for ( j = 0; j < mRHSNumGEqns_; j++ )
            if (mRHSGEqnIDs_[j] == globalEqn[i] && mRHSBCType_[j] == 1) break;
         if (j == mRHSNumGEqns_)
         {
            printf("%4d : HYPRE_LSC::enforceEssentialBC ERROR (1).\n",mypid_);
            return -1;
         }
         k = j;
         localEqnNum = globalEqn[i] + 1 - localStartRow_;
         if ( localEqnNum >= 0 && localEqnNum < numLocalRows )
         {
            for ( j = 0; j < mRHSNEntries_[k]; j++ )
            {
               rhs_term = gamma1[i] / alpha[i] * mRHSRowVals_[k][j];
               eqnNum = mRHSRowInds_[k][j] - 1;
               HYPRE_IJVectorGetValues(HYb_,1, &eqnNum, &val);
               val -= rhs_term;
               HYPRE_IJVectorSetValues(HYb_, 1, (const int *) &eqnNum,
                                       (const double *) &rhs_term);
            }
         }
         // Set rhs for boundary point
         rhs_term = gamma1[i] / alpha[i];
         eqnNum = globalEqn[i];
         HYPRE_IJVectorSetValues(HYb_, 1, (const int *) &eqnNum,
                                 (const double *) &rhs_term);
      }
   }
   //**/================================================================
   else
   {
      //**/=============================================================
      //**/ save the BC information
      if (mRHSFlag_ == 1)
      {
         if (mRHSNumGEqns_ == 0)
         {
            mRHSGEqnIDs_ = new int[leng];
            mRHSNEntries_ = new int[leng];
            mRHSBCType_  = new int[leng];
            mRHSRowInds_ = new int*[leng];
            mRHSRowVals_ = new double*[leng];
            for (i = 0; i < leng; i++) mRHSRowInds_[i] = NULL;
            for (i = 0; i < leng; i++) mRHSRowVals_[i] = NULL;
         }
         else
         {
            iarray = mRHSGEqnIDs_;
            mRHSGEqnIDs_ = new int[mRHSNumGEqns_+leng];
            for (i = 0; i < mRHSNumGEqns_; i++) mRHSGEqnIDs_[i] = iarray[i];
            iarray = mRHSNEntries_;
            mRHSNEntries_ = new int[mRHSNumGEqns_+leng];
            for (i = 0; i < mRHSNumGEqns_; i++) mRHSNEntries_[i] = iarray[i];
            iarray = mRHSBCType_;
            mRHSBCType_ = new int[mRHSNumGEqns_+leng];
            for (i = 0; i < mRHSNumGEqns_; i++) mRHSBCType_[i] = iarray[i];
            i2array = mRHSRowInds_;
            mRHSRowInds_ = new int*[mRHSNumGEqns_+leng];
            for (i = 0; i < mRHSNumGEqns_; i++) mRHSRowInds_[i] = i2array[i];
            d2array = mRHSRowVals_;
            for (i = 0; i < mRHSNumGEqns_; i++) mRHSRowInds_[i] = i2array[i];
            for (i = mRHSNumGEqns_; i < mRHSNumGEqns_+leng; i++)
                mRHSRowInds_[i] = NULL;
            mRHSRowVals_ = new double*[mRHSNumGEqns_+leng];
            for (i = 0; i < mRHSNumGEqns_; i++) mRHSRowVals_[i] = d2array[i];
            for (i = mRHSNumGEqns_; i < mRHSNumGEqns_+leng; i++)
                mRHSRowVals_[i] = NULL;
         }
      }
      //**/=============================================================
      for( i = 0; i < leng; i++ ) 
      {
         localEqnNum = globalEqn[i] + 1 - localStartRow_;
         if ( localEqnNum >= 0 && localEqnNum < numLocalRows )
         {
            rowSize = rowLengths_[localEqnNum];
            colInd  = colIndices_[localEqnNum];
            colVal  = colValues_[localEqnNum];

            //===================================================
            // store the information for multiple right hand side
            if (mRHSFlag_ == 1)
            {
               count = 0;
               for ( j = 0; j < rowSize; j++ ) 
               {
                  colIndex = colInd[j];
                  if (colIndex >= localStartRow_ && colIndex <= localEndRow_) 
                  {
                     if ( (colIndex-1) != globalEqn[i]) 
                     {
                        rowSize2 = rowLengths_[colIndex-localStartRow_];
                        colInd2  = colIndices_[colIndex-localStartRow_];
                        colVal2  = colValues_ [colIndex-localStartRow_];
                        for( k = 0; k < rowSize2; k++ ) 
                        {
                           if (colInd2[k]-1 == globalEqn[i]) 
                              count++;
                           break;
                        }
                     }
                  }
               }
               if (count > 0)
               {
                  mRHSBCType_[mRHSNumGEqns_] = 1;
                  mRHSGEqnIDs_[mRHSNumGEqns_] = globalEqn[i];
                  mRHSNEntries_[mRHSNumGEqns_] = count;
                  mRHSRowInds_[mRHSNumGEqns_] = new int[count];
                  mRHSRowVals_[mRHSNumGEqns_] = new double[count];
               }
               count = 0;
               for ( j = 0; j < rowSize; j++ ) 
               {
                  colIndex = colInd[j];
                  if (colIndex >= localStartRow_ && colIndex <= localEndRow_) 
                  {
                     if ( (colIndex-1) != globalEqn[i]) 
                     {
                        rowSize2 = rowLengths_[colIndex-localStartRow_];
                        colInd2  = colIndices_[colIndex-localStartRow_];
                        colVal2  = colValues_ [colIndex-localStartRow_];
                        for( k = 0; k < rowSize2; k++ ) 
                        {
                           if ( colInd2[k]-1 == globalEqn[i] ) 
                           {
                              mRHSRowVals_[mRHSNumGEqns_][count] = colVal2[k];
                              mRHSRowInds_[mRHSNumGEqns_][count] = colIndex;
                              count++;
                              break;
                           }
                        }
                     }
                  }
               }
               mRHSNumGEqns_++;
            }
            //===================================================

            for ( j = 0; j < rowSize; j++ ) 
            {
               colIndex = colInd[j];
               if ( colIndex-1 == globalEqn[i] ) colVal[j] = 1.0;
               else                              colVal[j] = 0.0;
               if ( colIndex >= localStartRow_ && colIndex <= localEndRow_) 
               {
                  if ( (colIndex-1) != globalEqn[i]) 
                  {
                     rowSize2 = rowLengths_[colIndex-localStartRow_];
                     colInd2  = colIndices_[colIndex-localStartRow_];
                     colVal2  = colValues_ [colIndex-localStartRow_];

                     for( k = 0; k < rowSize2; k++ ) 
                     {
                        if ( colInd2[k]-1 == globalEqn[i] ) 
                        {
                           rhs_term = gamma1[i] / alpha[i] * colVal2[k];
                           eqnNum = colIndex - 1;
                           HYPRE_IJVectorGetValues(HYb_,1,&eqnNum, &val);
                           val -= rhs_term;
                           HYPRE_IJVectorSetValues(HYb_,1,(const int *) &eqnNum,
                                                   (const double *) &val);
                           colVal2[k] = 0.0;
                           break;
                        }
                     }
                  }
               }
            }// end for(j<rowSize) loop

            // Set rhs for boundary point
            rhs_term = gamma1[i] / alpha[i];
            eqnNum = globalEqn[i];
            HYPRE_IJVectorSetValues(HYb_, 1, (const int *) &eqnNum,
                                    (const double *) &rhs_term);

         }
      }
   }

   //-------------------------------------------------------------------
   // set up the AMGe Dirichlet boundary conditions
   //-------------------------------------------------------------------

#ifdef HAVE_AMGE
   colInd = new int[leng];
   for( i = 0; i < leng; i++ ) colInd[i] = globalEqn[i];
   HYPRE_LSI_AMGeSetBoundary( leng, colInd );
   delete [] colInd;
#endif

   //-------------------------------------------------------------------
   // diagnostic message 
   //-------------------------------------------------------------------

   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 5 )
      printf("%4d : HYPRE_LSC::leaving  enforceEssentialBC.\n",mypid_);
   return (0);
}

//***************************************************************************
// put in globalEqns should hold eqns that are owned locally, but which contain
// column indices (the ones in colIndices) which are from remote equations
// on which essential boundary-conditions need to be enforced.
// This function will only make the modification if the above conditions
// hold -- i.e., the equation is a locally-owned equation, and the column
// index is NOT a locally owned equation.
//---------------------------------------------------------------------------

int HYPRE_LinSysCore::enforceRemoteEssBCs(int numEqns, int* globalEqns,
                                          int** colIndices, int* colIndLen,
                                          double** coefs) 
{
   int    i, j, k, numLocalRows, localEqnNum, rowLen, *colInd, eqnNum;
   int    *iarray, **i2array, count;
   double bval, *colVal, rhs_term, **d2array;

   //-------------------------------------------------------------------
   // diagnostic message and error checking
   // (this function should be called before matrixLoadComplete)
   //-------------------------------------------------------------------

   if ( (HYOutputLevel_ & HYFEI_IMPOSENOBC) != 0 ) return 0;
   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 5 )
      printf("%4d : HYPRE_LSC::entering enforceRemoteEssBC.\n",mypid_);
   if ( systemAssembled_ )
   {
      printf("enforceRemoteEssBC ERROR : system assembled already.\n");
      exit(1);
   }

   //-------------------------------------------------------------------
   // examine each row individually
   //-------------------------------------------------------------------

   numLocalRows = localEndRow_ - localStartRow_ + 1;

   //============================================================
   //**/ The following is for multiple right hand side (Mar 2009)

   if (mRHSFlag_ == 1 && currentRHS_ != 0 & mRHSNumGEqns_ > 0)
   {
      for( i = 0; i < numEqns; i++ )
      {
         for ( j = 0; j < mRHSNumGEqns_; j++ )
            if (mRHSGEqnIDs_[j] == globalEqns[i] && mRHSBCType_[j] == 2) break;
         if (j == mRHSNumGEqns_)
         {
            printf("%4d : HYPRE_LSC::enforceRemoteEssBCs ERROR (1).\n",mypid_);
            return -1;
         }
         k = j;
         localEqnNum = globalEqns[i] + 1 - localStartRow_;
         if ( localEqnNum < 0 || localEqnNum >= numLocalRows )
         {
            continue;
         }
         eqnNum = globalEqns[i];
         rowLen = mRHSNEntries_[k];
         colInd = mRHSRowInds_[k];
         colVal = mRHSRowVals_[k];
         for ( j = 0; j < colIndLen[i]; j++) 
         {
            for ( k = 0; k < rowLen; k++ ) 
            {
               if (colInd[k]-1 == colIndices[i][j]) 
               {
                  rhs_term = colVal[k] * coefs[i][j];
                  HYPRE_IJVectorGetValues(HYb_,1,&eqnNum,&bval);
                  bval -= rhs_term;
                  HYPRE_IJVectorSetValues(HYb_, 1, (const int *) &eqnNum,
                                          (const double *) &bval);
               }
            }
         }
      }
   }
   //**/================================================================
   else
   {
      //**/=============================================================
      //**/ save the BC information
      if (mRHSFlag_ == 1)
      {
         if (mRHSNumGEqns_ == 0)
         {
            mRHSGEqnIDs_ = new int[numEqns];
            mRHSNEntries_ = new int[numEqns];
            mRHSBCType_  = new int[numEqns];
            mRHSRowInds_ = new int*[numEqns];
            mRHSRowVals_ = new double*[numEqns];
            for (i = 0; i < numEqns; i++) mRHSRowInds_[i] = NULL;
            for (i = 0; i < numEqns; i++) mRHSRowVals_[i] = NULL;
         }
         else
         {
            iarray = mRHSGEqnIDs_;
            mRHSGEqnIDs_ = new int[mRHSNumGEqns_+numEqns];
            for (i = 0; i < mRHSNumGEqns_; i++) mRHSGEqnIDs_[i] = iarray[i];
            iarray = mRHSNEntries_;
            mRHSNEntries_ = new int[mRHSNumGEqns_+numEqns];
            for (i = 0; i < mRHSNumGEqns_; i++) mRHSNEntries_[i] = iarray[i];
            iarray = mRHSBCType_;
            mRHSBCType_ = new int[mRHSNumGEqns_+numEqns];
            for (i = 0; i < mRHSNumGEqns_; i++) mRHSBCType_[i] = iarray[i];
            i2array = mRHSRowInds_;
            mRHSRowInds_ = new int*[mRHSNumGEqns_+numEqns];
            for (i = 0; i < mRHSNumGEqns_; i++) mRHSRowInds_[i] = i2array[i];
            d2array = mRHSRowVals_;
            for (i = 0; i < mRHSNumGEqns_; i++) mRHSRowInds_[i] = i2array[i];
            for (i = mRHSNumGEqns_; i < mRHSNumGEqns_+numEqns; i++)
                mRHSRowInds_[i] = NULL;
            mRHSRowVals_ = new double*[mRHSNumGEqns_+numEqns];
            for (i = 0; i < mRHSNumGEqns_; i++) mRHSRowVals_[i] = d2array[i];
            for (i = mRHSNumGEqns_; i < mRHSNumGEqns_+numEqns; i++)
                mRHSRowVals_[i] = NULL;
         }
      }
      //**/=============================================================

      for( i = 0; i < numEqns; i++ ) 
      {
         localEqnNum = globalEqns[i] + 1 - localStartRow_;
         if ( localEqnNum < 0 || localEqnNum >= numLocalRows )
         {
            continue;
         }

         rowLen = rowLengths_[localEqnNum];
         colInd = colIndices_[localEqnNum];
         colVal = colValues_[localEqnNum];
         eqnNum = globalEqns[i];

         //===================================================
         // store the information for multiple right hand side
         if (mRHSFlag_ == 1)
         {
            count = 0;
            for ( j = 0; j < colIndLen[i]; j++) 
            {
               for ( k = 0; k < rowLen; k++ ) 
                  if (colInd[k]-1 == colIndices[i][j]) count++;
            }
            if (count > 0)
            {
               mRHSGEqnIDs_[mRHSNumGEqns_] = globalEqns[i];
               mRHSBCType_[mRHSNumGEqns_] = 2;
               mRHSNEntries_[mRHSNumGEqns_] = count;
               mRHSRowInds_[mRHSNumGEqns_] = new int[count];
               mRHSRowVals_[mRHSNumGEqns_] = new double[count];
            }
            count = 0;
            for ( j = 0; j < colIndLen[i]; j++) 
            {
               for ( k = 0; k < rowLen; k++ ) 
               {
                  if (colInd[k]-1 == colIndices[i][j]) 
                  {
                     mRHSRowVals_[k][count] = colVal[k];
                     mRHSRowInds_[k][count] = colInd[k];
                  }
               }
            }
         }
         //===================================================

         for ( j = 0; j < colIndLen[i]; j++) 
         {
            for ( k = 0; k < rowLen; k++ ) 
            {
               if (colInd[k]-1 == colIndices[i][j]) 
               {
                  rhs_term = colVal[k] * coefs[i][j];
                  HYPRE_IJVectorGetValues(HYb_,1,&eqnNum,&bval);
                  bval -= rhs_term;
                  HYPRE_IJVectorSetValues(HYb_, 1, (const int *) &eqnNum,
                                          (const double *) &bval);
                  colVal[k] = 0.0;
               }
            }
         }
      }
   } 

   //-------------------------------------------------------------------
   // diagnostic message 
   //-------------------------------------------------------------------

   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 5 )
      printf("%4d : HYPRE_LSC::leaving  enforceRemoteEssBC.\n",mypid_);
   return (0);
}

//***************************************************************************
//This function must enforce a natural or mixed boundary condition on the
//equations in 'globalEqn'. This means that the following modification should
//be made to A and b:
//
//A[globalEqn,globalEqn] += alpha/beta;
//b[globalEqn] += gamma/beta;
//---------------------------------------------------------------------------

int HYPRE_LinSysCore::enforceOtherBC(int* globalEqn, double* alpha, 
                                     double* beta, double* gamma1, int leng)
{
   int    i, j, numLocalRows, localEqnNum, *colInd, rowSize, eqnNum;
   double val, *colVal, rhs_term;

   //-------------------------------------------------------------------
   // diagnostic message and error checking
   // (this function should be called before matrixLoadComplete)
   //-------------------------------------------------------------------

   if ( (HYOutputLevel_ & HYFEI_IMPOSENOBC) != 0 ) return 0;
   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 5 )
      printf("%4d : HYPRE_LSC::entering enforceOtherBC.\n",mypid_);
   if ( systemAssembled_ )
   {
      printf("enforceOtherBC ERROR : system assembled already.\n");
      exit(1);
   }

   //-------------------------------------------------------------------
   // examine each row individually
   //-------------------------------------------------------------------

   numLocalRows = localEndRow_ - localStartRow_ + 1;

   //============================================================
   //**/ The following is for multiple right hand side (Mar 2009)
   if (mRHSFlag_ == 1 && currentRHS_ != 0)
   {
      for( i = 0; i < leng; i++ ) 
      {
         localEqnNum = globalEqn[i] + 1 - localStartRow_;
         if ( localEqnNum < 0 || localEqnNum >= numLocalRows )
         {
            continue;
         }
         eqnNum = globalEqn[i];
         rhs_term = gamma1[i] / beta[i];
         HYPRE_IJVectorGetValues(HYb_,1,&eqnNum,&val);
         val += rhs_term;
         HYPRE_IJVectorSetValues(HYb_, 1, (const int *) &eqnNum,
                                 (const double *) &val);
      }
   }
   //============================================================
   else
   {
      for( i = 0; i < leng; i++ ) 
      {
         localEqnNum = globalEqn[i] + 1 - localStartRow_;
         if ( localEqnNum < 0 || localEqnNum >= numLocalRows )
         {
            continue;
         }

         rowSize = rowLengths_[localEqnNum];
         colVal  = colValues_[localEqnNum];
         colInd  = colIndices_[localEqnNum];

         for ( j = 0; j < rowSize; j++) 
         {
            if ((colInd[j]-1) == globalEqn[i]) 
            {
               colVal[j] += alpha[i]/beta[i];
               break;
            }
         }

         //now make the rhs modification.
         // need to fetch matrix and put it back before assembled

         eqnNum = globalEqn[i];
         rhs_term = gamma1[i] / beta[i];
         HYPRE_IJVectorGetValues(HYb_,1,&eqnNum,&val);
         val += rhs_term;
         HYPRE_IJVectorSetValues(HYb_, 1, (const int *) &eqnNum,
                                 (const double *) &val);
      }
   }

   //-------------------------------------------------------------------
   // diagnostic message
   //-------------------------------------------------------------------

   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 5 )
      printf("%4d : HYPRE_LSC::leaving  enforceOtherBC.\n",mypid_);
   return (0);
}

//***************************************************************************
// put the pointer to the A matrix into the Data object
//---------------------------------------------------------------------------

#ifndef NOFEI
int HYPRE_LinSysCore::getMatrixPtr(Data& data) 
{
   (void) data;
   printf("%4d : HYPRE_LSC::getMatrixPtr ERROR - not implemented.\n",mypid_);
   exit(1);
   return (0);
}
#endif

//***************************************************************************
// Overwrites the current internal matrix with a scaled copy of the
// input argument.
//---------------------------------------------------------------------------

#ifndef NOFEI
int HYPRE_LinSysCore::copyInMatrix(double scalar, const Data& data) 
{
   int  i;
   char *name;
   HYPRE_FEI_AMSData *auxAMSData;

   (void) scalar;

   name  = data.getTypeName();
   if (!strcmp(name, "ANN"))
   {
      maxwellANN_ = (HYPRE_ParCSRMatrix) data.getDataPtr();
   }
   else if (!strcmp(name, "GEN"))
   {
      maxwellGEN_ = (HYPRE_ParCSRMatrix) data.getDataPtr();
   }
   else if (!strcmp(name, "AMSBMATRIX"))
   {
      amsBetaPoisson_ = (HYPRE_ParCSRMatrix) data.getDataPtr();
   }
   else if (!strcmp(name, "AMSData"))
   {
      auxAMSData = (HYPRE_FEI_AMSData *) data.getDataPtr();
      if (AMSData_.NodeNumbers_ != NULL) delete [] AMSData_.NodeNumbers_;
      if (AMSData_.NodalCoord_  != NULL) delete [] AMSData_.NodalCoord_;
      AMSData_.NodeNumbers_ = NULL;
      AMSData_.NodalCoord_  = NULL;
      AMSData_.numNodes_ = auxAMSData->numNodes_;
      AMSData_.numLocalNodes_ = auxAMSData->numLocalNodes_;
      if (AMSData_.numNodes_ > 0)
      {
         AMSData_.NodeNumbers_ = new int[AMSData_.numNodes_];
         AMSData_.NodalCoord_  = new double[AMSData_.numNodes_*mlNumPDEs_];
         for (i = 0; i < AMSData_.numNodes_; i++)
            AMSData_.NodeNumbers_[i] = auxAMSData->NodeNumbers_[i];
         for (i = 0; i < AMSData_.numNodes_*mlNumPDEs_; i++)
            AMSData_.NodalCoord_[i] = auxAMSData->NodalCoord_[i];
      }
   }
   else
   {
      printf("%4d : HYPRE_LSC::copyInMatrix ERROR - invalid data.\n",mypid_);
      exit(1);
   }
   return (0);
}
#endif

//***************************************************************************
//Passes out a scaled copy of the current internal matrix.
//---------------------------------------------------------------------------

#ifndef NOFEI
int HYPRE_LinSysCore::copyOutMatrix(double scalar, Data& data) 
{
   char *name;

   (void) scalar;

   name = data.getTypeName();

   if (!strcmp(name, "A"))
   {
      data.setDataPtr((void *) HYA_);
   }
   else if (!strcmp(name, "AMSData"))
   {
      data.setDataPtr((void *) &AMSData_);
   }
   else
   {
      printf("HYPRE_LSC::copyOutMatrix ERROR - invalid command.\n");
      exit(1);
   }
   return (0);
}
#endif

//***************************************************************************
// add nonzero entries into the matrix data structure
//---------------------------------------------------------------------------

#ifndef NOFEI
int HYPRE_LinSysCore::sumInMatrix(double scalar, const Data& data) 
{
   (void) scalar;
   (void) data;
   printf("%4d : HYPRE_LSC::sumInMatrix ERROR - not implemented.\n",mypid_);
   exit(1);
   return (0);
}
#endif

//***************************************************************************
// get the data pointer for the right hand side
//---------------------------------------------------------------------------

#ifndef NOFEI
int HYPRE_LinSysCore::getRHSVectorPtr(Data& data) 
{
   //-------------------------------------------------------------------
   // diagnostic message
   //-------------------------------------------------------------------

   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
      printf("%4d : HYPRE_LSC::entering getRHSVectorPtr.\n",mypid_);

   //-------------------------------------------------------------------
   // get the right hand side vector pointer
   //-------------------------------------------------------------------

   data.setTypeName("IJ_Vector");
   data.setDataPtr((void*) HYb_);

   //-------------------------------------------------------------------
   // diagnostic message
   //-------------------------------------------------------------------

   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
      printf("%4d : HYPRE_LSC::leaving  getRHSVectorPtr.\n",mypid_);
   return (0);
}
#endif

//***************************************************************************
// copy the content of the incoming vector to the right hand side vector
//---------------------------------------------------------------------------

#ifndef NOFEI
int HYPRE_LinSysCore::copyInRHSVector(double scalar, const Data& data) 
{
   //-------------------------------------------------------------------
   // diagnostic message
   //-------------------------------------------------------------------

   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
      printf("%4d : HYPRE_LSC::entering copyInRHSVector.\n",mypid_);
   if (strcmp("IJ_Vector", data.getTypeName()) &&
       strcmp("Sol_Vector", data.getTypeName()))
   {
      printf("copyInRHSVector: data's type string not compatible.\n");
      exit(1);
   }

   //-------------------------------------------------------------------
   // copy the incoming vector to the internal right hand side
   //-------------------------------------------------------------------

   HYPRE_IJVector inVec = (HYPRE_IJVector) data.getDataPtr();
   HYPRE_ParVector srcVec;
   HYPRE_ParVector destVec;
   HYPRE_IJVectorGetObject(inVec, (void **) &srcVec);
   if (!strcmp("Sol_Vector", data.getTypeName()))
      HYPRE_IJVectorGetObject(HYb_, (void **) &destVec);
   else
      HYPRE_IJVectorGetObject(HYx_, (void **) &destVec);
 
   HYPRE_ParVectorCopy(srcVec, destVec);
 
   if ( scalar != 1.0 ) HYPRE_ParVectorScale( scalar, destVec);
   // do not destroy the incoming vector
   //HYPRE_IJVectorDestroy(inVec);

   //-------------------------------------------------------------------
   // diagnostic message
   //-------------------------------------------------------------------

   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
      printf("%4d : HYPRE_LSC::leaving  copyInRHSVector.\n",mypid_);
   return (0);
}
#endif

//***************************************************************************
// create an ParVector and copy the right hand side to it (scaled)
//---------------------------------------------------------------------------

#ifndef NOFEI
int HYPRE_LinSysCore::copyOutRHSVector(double scalar, Data& data) 
{
   int ierr;

   //-------------------------------------------------------------------
   // diagnostic message
   //-------------------------------------------------------------------

   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
      printf("%4d : HYPRE_LSC::entering copyOutRHSVector.\n",mypid_);

   //-------------------------------------------------------------------
   // extract the right hand side vector
   //-------------------------------------------------------------------

   HYPRE_IJVector newVector;
   ierr = HYPRE_IJVectorCreate(comm_, localStartRow_-1, localEndRow_-1, 
                               &newVector);
   ierr = HYPRE_IJVectorSetObjectType(newVector, HYPRE_PARCSR);
   ierr = HYPRE_IJVectorInitialize(newVector);
   ierr = HYPRE_IJVectorAssemble(newVector);
   //assert(!ierr);

   HYPRE_ParVector Vec1;
   HYPRE_ParVector Vec2;
   HYPRE_IJVectorGetObject(HYb_, (void **) &Vec1);
   HYPRE_IJVectorGetObject(newVector, (void **) &Vec2);

   HYPRE_ParVectorCopy( Vec1, Vec2);
   if ( scalar != 1.0 ) HYPRE_ParVectorScale( scalar, Vec2);

   data.setTypeName("IJ_Vector");
   data.setDataPtr((void*) Vec2);

   //-------------------------------------------------------------------
   // diagnostic message
   //-------------------------------------------------------------------

   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
      printf("%4d : HYPRE_LSC::leaving  copyOutRHSVector.\n",mypid_);
   return (0);
}
#endif 

//***************************************************************************
// add the incoming ParCSR vector to the current right hand side (scaled)
//---------------------------------------------------------------------------

#ifndef NOFEI
int HYPRE_LinSysCore::sumInRHSVector(double scalar, const Data& data) 
{
   //-------------------------------------------------------------------
   // diagnostic message
   //-------------------------------------------------------------------

   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
      printf("%4d : HYPRE_LSC::entering sumInRHSVector.\n",mypid_);
   if (strcmp("IJ_Vector", data.getTypeName()))
   {
      printf("sumInRHSVector ERROR : data's type string not 'IJ_Vector'.\n");
      exit(1);
   }

   //-------------------------------------------------------------------
   // add the incoming vector to the right hand side
   //-------------------------------------------------------------------

   HYPRE_IJVector inVec = (HYPRE_IJVector) data.getDataPtr();
   HYPRE_ParVector xVec;
   HYPRE_ParVector yVec;
   HYPRE_IJVectorGetObject(inVec, (void **) &xVec);
   HYPRE_IJVectorGetObject(HYb_, (void **) &yVec);
   hypre_ParVectorAxpy(scalar,(hypre_ParVector*)xVec,(hypre_ParVector*)yVec);

   //-------------------------------------------------------------------
   // diagnostic message
   //-------------------------------------------------------------------

   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
      printf("%4d : HYPRE_LSC::leaving  sumInRHSVector.\n",mypid_);
   return (0);
}
#endif 

//***************************************************************************
// deallocate an incoming IJ matrix
//---------------------------------------------------------------------------

#ifndef NOFEI
int HYPRE_LinSysCore::destroyMatrixData(Data& data) 
{
   //-------------------------------------------------------------------
   // diagnostic message
   //-------------------------------------------------------------------

   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
      printf("%4d : HYPRE_LSC::entering destroyMatrixData.\n",mypid_);
   if (strcmp("IJ_Matrix", data.getTypeName()))
   {
      printf("destroyMatrixData ERROR : data doesn't contain a IJ_Matrix.\n");
      exit(1);
   }

   //-------------------------------------------------------------------
   // destroy the incoming matrix data
   //-------------------------------------------------------------------

   HYPRE_IJMatrix mat = (HYPRE_IJMatrix) data.getDataPtr();
   HYPRE_IJMatrixDestroy(mat);

   //-------------------------------------------------------------------
   // diagnostic message
   //-------------------------------------------------------------------

   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
      printf("%4d : HYPRE_LSC::leaving  destroyMatrixData.\n",mypid_);
   return (0);
}
#endif 

//***************************************************************************
// deallocate an incoming IJ vector
//---------------------------------------------------------------------------

#ifndef NOFEI
int HYPRE_LinSysCore::destroyVectorData(Data& data) 
{
   //-------------------------------------------------------------------
   // diagnostic message
   //-------------------------------------------------------------------

   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
      printf("%4d : HYPRE_LSC::entering destroyVectorData.\n",mypid_);
   if (strcmp("IJ_Vector", data.getTypeName()))
   {
      printf("destroyVectorData ERROR : data doesn't contain a IJ_Vector.");
      exit(1);
   }

   //-------------------------------------------------------------------
   // destroy the incoming vector data
   //-------------------------------------------------------------------

   HYPRE_IJVector vec = (HYPRE_IJVector) data.getDataPtr();
   if ( vec != NULL ) HYPRE_IJVectorDestroy(vec);

   //-------------------------------------------------------------------
   // diagnostic message
   //-------------------------------------------------------------------

   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
      printf("%4d : HYPRE_LSC::leaving  destroyVectorData.\n",mypid_);
   return (0);
}
#endif 

//***************************************************************************
// set number of right hand side vectors
//---------------------------------------------------------------------------

int HYPRE_LinSysCore::setNumRHSVectors(int numRHSs, const int* rhsIDs) 
{
   int ierr = 0;
   //-------------------------------------------------------------------
   // diagnostic message
   //-------------------------------------------------------------------

   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
   {
      printf("%4d : HYPRE_LSC::entering setNumRHSVectors.\n",mypid_);
      printf("%4d : HYPRE_LSC::incoming numRHSs = %d\n",mypid_,numRHSs);
      for ( int i = 0; i < numRHSs_; i++ ) 
         printf("%4d : HYPRE_LSC::incoming RHSIDs  = %d\n",mypid_,rhsIDs[i]);
   }
   if (numRHSs < 0)
   {
      printf("setNumRHSVectors ERROR : numRHSs < 0.\n");
      exit(1);
   }

   //-------------------------------------------------------------------
   // first destroy the existing right hand side vectors
   //-------------------------------------------------------------------

   if ( matrixVectorsCreated_ )
   {
      if ( HYbs_ != NULL ) 
      {
         for ( int i = 0; i < numRHSs_; i++ ) 
            if ( HYbs_[i] != NULL ) HYPRE_IJVectorDestroy(HYbs_[i]);
         delete [] HYbs_;
         HYbs_ = NULL;
      }
   }
   if (numRHSs == 0) return (0);

   //-------------------------------------------------------------------
   // instantiate the right hand vectors
   //-------------------------------------------------------------------

   if ( matrixVectorsCreated_ )
   {
      HYbs_ = new HYPRE_IJVector[numRHSs_];
      for ( int i = 0; i < numRHSs_; i++ )
      {
         ierr = HYPRE_IJVectorCreate(comm_, localStartRow_-1, localEndRow_-1,
                                   &(HYbs_[i]));
         ierr = HYPRE_IJVectorSetObjectType(HYbs_[i], HYPRE_PARCSR);
         ierr = HYPRE_IJVectorInitialize(HYbs_[i]);
         ierr = HYPRE_IJVectorAssemble(HYbs_[i]);
         //assert(!ierr);
      }
      HYb_ = HYbs_[0];
   }

   //-------------------------------------------------------------------
   // copy in the right hand side IDs
   //-------------------------------------------------------------------

   delete [] rhsIDs_;
   numRHSs_ = numRHSs;
   rhsIDs_ = new int[numRHSs_];
 
   for ( int i = 0; i < numRHSs; i++ ) rhsIDs_[i] = rhsIDs[i];

   //-------------------------------------------------------------------
   // diagnostic message
   //-------------------------------------------------------------------

   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
      printf("%4d : HYPRE_LSC::leaving  setNumRHSVectors.\n",mypid_);
   return (ierr);
}

//***************************************************************************
// select a right hand side vector
//---------------------------------------------------------------------------

int HYPRE_LinSysCore::setRHSID(int rhsID) 
{
   //-------------------------------------------------------------------
   // diagnostic message
   //-------------------------------------------------------------------

   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
      printf("%4d : HYPRE_LSC::setRHSID = %d.\n",mypid_,rhsID);

   //-------------------------------------------------------------------
   // set current right hand side vector ID
   //-------------------------------------------------------------------

   for( int i = 0; i < numRHSs_; i++ )
   {
      if (rhsIDs_[i] == rhsID)
      {
         currentRHS_ = i;
         HYb_ = HYbs_[currentRHS_];
         currB_ = HYb_;
         return (0);
      }
   }
   printf("setRHSID ERROR : rhsID %d not found.\n", rhsID);
   exit(1);
   return (0);
}

//***************************************************************************
// used for initializing the initial guess
//---------------------------------------------------------------------------

int HYPRE_LinSysCore::putInitialGuess(const int* eqnNumbers,
                                       const double* values, int leng) 
{
   int i, ierr, *localInds, *iarray, *iarray2;

   //-------------------------------------------------------------------
   // diagnostic message
   //-------------------------------------------------------------------

   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
      printf("%4d : HYPRE_LSC::entering putInitalGuess.\n",mypid_);

   //-------------------------------------------------------------------
   // this is to create a FEI to HYPRE equation node map
   //-------------------------------------------------------------------

   if ( mapFromSolnFlag_ == 1 )
   {
      if ( (mapFromSolnLeng_+leng) >= mapFromSolnLengMax_ )
      {
         iarray  = mapFromSolnList_;
         iarray2 = mapFromSolnList2_;
         mapFromSolnLengMax_ = mapFromSolnLengMax_ + 2 * leng;
         mapFromSolnList_  = new int[mapFromSolnLengMax_];
         mapFromSolnList2_ = new int[mapFromSolnLengMax_];
         for ( i = 0; i < mapFromSolnLeng_; i++ ) 
         {
            mapFromSolnList_[i] = iarray[i];
            mapFromSolnList2_[i] = iarray2[i];
         }
         if ( iarray  != NULL ) delete [] iarray;
         if ( iarray2 != NULL ) delete [] iarray2;
      }
   }

   localInds = new int[leng];
   for ( i = 0; i < leng; i++ ) // change to 0-based
   {
      if ((eqnNumbers[i]+1) >= localStartRow_ && 
          (eqnNumbers[i]+1) <= localEndRow_) localInds[i] = eqnNumbers[i];
      else
      {
         printf("%d : putInitialGuess ERROR - index %d out of range\n",
                      mypid_, eqnNumbers[i]);
         exit(1);
      }
      if ( mapFromSolnFlag_ == 1 )
      {
         mapFromSolnList_[mapFromSolnLeng_] = eqnNumbers[i];
         mapFromSolnList2_[mapFromSolnLeng_++] = (int) values[i];
      }
   }
   ierr = HYPRE_IJVectorSetValues(HYx_, leng, (const int *) localInds,
                                  (const double *) values);
   //assert(!ierr);

   delete [] localInds;

   //-------------------------------------------------------------------
   // inject the initial guess into reduced systems, if any
   //-------------------------------------------------------------------

   if ( schurReduction_ == 1 ) buildSchurInitialGuess();

   //-------------------------------------------------------------------
   // diagnostic message
   //-------------------------------------------------------------------

   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
      printf("%4d : HYPRE_LSC::leaving  putInitalGuess.\n",mypid_);
   return (0);
}

//***************************************************************************
// This is a modified function for version 1.5
// used for getting the solution out of the solver, and into the application
//---------------------------------------------------------------------------

int HYPRE_LinSysCore::getSolution(double* answers,int leng) 
{
   int    i, ierr, *equations;

   //-------------------------------------------------------------------
   // diagnostic message
   //-------------------------------------------------------------------

   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 4 )
      printf("%4d : HYPRE_LSC::entering getSolution.\n",mypid_);
   if (localStartCol_ == -1 && leng != (localEndRow_-localStartRow_+1))
   {
      printf("%4d : HYPRE_LSC ERROR : getSolution: leng != numLocalRows.\n",
             mypid_);
      exit(1);
   }

   //-------------------------------------------------------------------
   // get the whole solution vector
   //-------------------------------------------------------------------

   equations = new int[leng];
   if (localStartCol_ == -1)
      for ( i = 0; i < leng; i++ ) equations[i] = localStartRow_ + i - 1;
   else
      for ( i = 0; i < leng; i++ ) equations[i] = localStartCol_ + i;

   ierr = HYPRE_IJVectorGetValues(HYx_,leng,equations,answers);
   //assert(!ierr);

   delete [] equations;

   //-------------------------------------------------------------------
   // diagnostic message
   //-------------------------------------------------------------------

   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 4 )
      printf("%4d : HYPRE_LSC::leaving  getSolution.\n",mypid_);
   return (0);
}

//***************************************************************************
// used for getting the solution out of the solver, and into the application
//---------------------------------------------------------------------------

int HYPRE_LinSysCore::getSolnEntry(int eqnNumber, double& answer) 
{
   double val;
   int    ierr, equation;

   //-------------------------------------------------------------------
   // diagnostic message
   //-------------------------------------------------------------------

   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 5 )
      printf("%4d : HYPRE_LSC::entering getSolnEntry.\n",mypid_);

   //-------------------------------------------------------------------
   // get a single solution entry
   //-------------------------------------------------------------------

   equation = eqnNumber; // incoming 0-based index

   if (localStartCol_ == -1 &&  equation < localStartRow_-1 && 
       equation > localEndRow_ )
   {
      printf("%d : getSolnEntry ERROR - index out of range = %d.\n", mypid_, 
                   eqnNumber);
      exit(1);
   }

   ierr = HYPRE_IJVectorGetValues(HYx_,1,&equation,&val);
   //assert(!ierr);
   answer = val;

   //-------------------------------------------------------------------
   // diagnostic message
   //-------------------------------------------------------------------

   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 5 )
      printf("%4d : HYPRE_LSC::leaving  getSolnEntry.\n",mypid_);
   return (0);
}

//***************************************************************************
// select which Krylov solver to use
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::selectSolver(char* name) 
{
   //-------------------------------------------------------------------
   // diagnostic message
   //-------------------------------------------------------------------

   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
   {
      printf("%4d : HYPRE_LSC::entering selectSolver.\n",mypid_);
      printf("%4d : HYPRE_LSC::solver name = %s.\n",mypid_,name);
   }

   //-------------------------------------------------------------------
   // if already been allocated, destroy it first
   //-------------------------------------------------------------------

   if ( HYSolver_ != NULL )
   {
      if ( HYSolverID_ == HYPCG )    HYPRE_ParCSRPCGDestroy(HYSolver_);
      if ( HYSolverID_ == HYLSICG )  HYPRE_ParCSRLSICGDestroy(HYSolver_);
      if ( HYSolverID_ == HYHYBRID ) HYPRE_ParCSRHybridDestroy(HYSolver_);
      if ( HYSolverID_ == HYGMRES )  HYPRE_ParCSRGMRESDestroy(HYSolver_);
      if ( HYSolverID_ == HYFGMRES)  HYPRE_ParCSRFGMRESDestroy(HYSolver_);
      if ( HYSolverID_ == HYCGSTAB)  HYPRE_ParCSRBiCGSTABDestroy(HYSolver_);
      if ( HYSolverID_ == HYCGSTABL) HYPRE_ParCSRBiCGSTABLDestroy(HYSolver_);
      if ( HYSolverID_ == HYAMG)     HYPRE_BoomerAMGDestroy(HYSolver_);
      if ( HYSolverID_ == HYTFQMR)   HYPRE_ParCSRTFQmrDestroy(HYSolver_);
      if ( HYSolverID_ == HYBICGS)   HYPRE_ParCSRBiCGSDestroy(HYSolver_);
      if ( HYSolverID_ == HYSYMQMR)  HYPRE_ParCSRSymQMRDestroy(HYSolver_);
   }

   //-------------------------------------------------------------------
   // check for the validity of the solver name
   //-------------------------------------------------------------------

   if ( !strcmp(name, "cg" ) )
   {
      strcpy( HYSolverName_, name );
      HYSolverID_ = HYPCG;
   }
   else if ( !strcmp(name, "lsicg" ) )
   {
      strcpy( HYSolverName_, name );
      HYSolverID_ = HYLSICG;
   }
   else if ( !strcmp(name, "hybrid") )
   {
      strcpy( HYSolverName_, name );
      HYSolverID_ = HYHYBRID;
   }
   else if ( !strcmp(name, "gmres") )
   {
      strcpy( HYSolverName_, name );
      HYSolverID_ = HYGMRES;
   }
   else if ( !strcmp(name, "fgmres") )
   {
      strcpy( HYSolverName_, name );
      HYSolverID_ = HYFGMRES;
   }
   else if ( !strcmp(name, "bicgstab") )
   {
      strcpy( HYSolverName_, name );
      HYSolverID_ = HYCGSTAB;
   }
   else if ( !strcmp(name, "bicgstabl") )
   {
      strcpy( HYSolverName_, name );
      HYSolverID_ = HYCGSTABL;
   }
   else if ( !strcmp(name, "tfqmr") )
   {
      strcpy( HYSolverName_, name );
      HYSolverID_ = HYTFQMR;
   }
   else if ( !strcmp(name, "bicgs") )
   {
      strcpy( HYSolverName_, name );
      HYSolverID_ = HYBICGS;
   }
   else if ( !strcmp(name, "symqmr") )
   {
      strcpy( HYSolverName_, name );
      HYSolverID_ = HYSYMQMR;
   }
   else if ( !strcmp(name, "boomeramg") )
   {
      strcpy( HYSolverName_, name );
      HYSolverID_ = HYAMG;
   }
   else if ( !strcmp(name, "superlu") )
   {
      strcpy( HYSolverName_, name );
      HYSolverID_ = HYSUPERLU;
   }
   else if ( !strcmp(name, "superlux") )
   {
      strcpy( HYSolverName_, name );
      HYSolverID_ = HYSUPERLUX;
   }
   else if ( !strcmp(name, "dsuperlu") )
   {
      strcpy( HYSolverName_, name );
#ifdef HAVE_DSUPERLU
      HYSolverID_ = HYDSUPERLU;
#else
      printf("HYPRE_LinSysCore:: DSuperLU not available.\n");
      printf("                   default solver to be GMRES.\n");
      HYSolverID_ = HYGMRES;
#endif
   }
   else if ( !strcmp(name, "y12m") )
   {
      strcpy( HYSolverName_, name );
      HYSolverID_ = HYY12M;
   }
   else if ( !strcmp(name, "amge") )
   {
      strcpy( HYSolverName_, name );
      HYSolverID_ = HYAMGE;
   }
   else
   {
      if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
         printf("HYPRE_LSC selectSolver : use default = gmres.\n");
      strcpy( HYSolverName_, "gmres" );
      HYSolverID_ = HYGMRES;
   }

   //-------------------------------------------------------------------
   // instantiate solver
   //-------------------------------------------------------------------

   switch ( HYSolverID_ )
   {
      case HYPCG :
           HYPRE_ParCSRPCGCreate(comm_, &HYSolver_);
           break;
      case HYLSICG :
           HYPRE_ParCSRLSICGCreate(comm_, &HYSolver_);
           break;
      case HYHYBRID :
           HYPRE_ParCSRHybridCreate(&HYSolver_);
           break;
      case HYGMRES :
           HYPRE_ParCSRGMRESCreate(comm_, &HYSolver_);
           break;
      case HYFGMRES :
           HYPRE_ParCSRFGMRESCreate(comm_, &HYSolver_);
           break;
      case HYCGSTAB :
           HYPRE_ParCSRBiCGSTABCreate(comm_, &HYSolver_);
           break;
      case HYCGSTABL :
           HYPRE_ParCSRBiCGSTABLCreate(comm_, &HYSolver_);
           break;
      case HYTFQMR :
           HYPRE_ParCSRTFQmrCreate(comm_, &HYSolver_);
           break;
      case HYBICGS :
           HYPRE_ParCSRBiCGSCreate(comm_, &HYSolver_);
           break;
      case HYSYMQMR :
           HYPRE_ParCSRSymQMRCreate(comm_, &HYSolver_);
           break;
      case HYAMG :
           HYPRE_BoomerAMGCreate( &HYSolver_);
           HYPRE_BoomerAMGSetCycleType(HYSolver_, 1);
           HYPRE_BoomerAMGSetMaxLevels(HYSolver_, 25);
           break;
      default:
           break;
   }

   //-------------------------------------------------------------------
   // diagnostic message
   //-------------------------------------------------------------------

   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
      printf("%4d : HYPRE_LSC::leaving  selectSolver.\n",mypid_);
   return;
}

//***************************************************************************
// select which preconditioner to use
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::selectPreconditioner(char *name)
{
   int ierr;

   //-------------------------------------------------------------------
   // diagnostic message
   //-------------------------------------------------------------------

   if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3)
      printf("%4d : HYPRE_LSC::entering selectPreconditioner = %s.\n",
             mypid_, name);

   //-------------------------------------------------------------------
   // if already been allocated, destroy it first
   //-------------------------------------------------------------------

   HYPreconSetup_  = 0;
   parasailsReuse_ = 0;
   if ( HYPrecon_ != NULL )
   {
      if (HYPreconID_ == HYPILUT)
         HYPRE_ParCSRPilutDestroy(HYPrecon_);

      else if (HYPreconID_ == HYPARASAILS)
         HYPRE_ParCSRParaSailsDestroy(HYPrecon_);

      else if (HYPreconID_ == HYBOOMERAMG)
         HYPRE_BoomerAMGDestroy(HYPrecon_);

      else if (HYPreconID_ == HYDDILUT)
         HYPRE_LSI_DDIlutDestroy(HYPrecon_);

      else if (HYPreconID_ == HYSCHWARZ)
         HYPRE_LSI_SchwarzDestroy(HYPrecon_);

      else if (HYPreconID_ == HYDDICT)
         HYPRE_LSI_DDICTDestroy(HYPrecon_);

      else if (HYPreconID_ == HYPOLY)
         HYPRE_LSI_PolyDestroy(HYPrecon_);

      else if (HYPreconID_ == HYEUCLID)
         HYPRE_EuclidDestroy(HYPrecon_);

      else if (HYPreconID_ == HYBLOCK)
         HYPRE_LSI_BlockPrecondDestroy(HYPrecon_);

#ifdef HAVE_ML
      else if (HYPreconID_ == HYML)
         HYPRE_LSI_MLDestroy(HYPrecon_);
#endif
#ifdef HAVE_MLI
      else if (HYPreconID_ == HYMLI)
         HYPRE_LSI_MLIDestroy(HYPrecon_);
#endif
      else if (HYPreconID_ == HYUZAWA)   
         HYPRE_LSI_UzawaDestroy(HYPrecon_);
#ifdef HAVE_SYSPDE
      else if (HYPreconID_ == HYSYSPDE)
         HYPRE_ParCSRSysPDEDestroy(HYPrecon_);
#endif
#ifdef HAVE_DSUPERLU
      else if (HYPreconID_ == HYDSLU)
         HYPRE_LSI_DSuperLUDestroy(HYPrecon_);
#endif
   }

   //-------------------------------------------------------------------
   // check for the validity of the preconditioner name
   //-------------------------------------------------------------------

   if (!strcmp(name, "identity"))
   {
      strcpy(HYPreconName_, name);
      HYPreconID_ = HYIDENTITY;
   }
   else if (!strcmp(name, "diagonal"))
   {
      strcpy(HYPreconName_, name);
      HYPreconID_ = HYDIAGONAL;
   }
   else if (!strcmp(name, "pilut"))
   {
      strcpy(HYPreconName_, name);
      HYPreconID_ = HYPILUT;
   }
   else if (!strcmp(name, "parasails"))
   {
      strcpy(HYPreconName_, name);
      HYPreconID_ = HYPARASAILS;
   }
   else if (!strcmp(name, "boomeramg"))
   {
      strcpy(HYPreconName_, name);
      HYPreconID_ = HYBOOMERAMG;
   }
   else if (!strcmp(name, "ddilut"))
   {
      strcpy(HYPreconName_, name);
      HYPreconID_ = HYDDILUT;
   }
   else if (!strcmp(name, "schwarz"))
   {
      strcpy(HYPreconName_, name);
      HYPreconID_ = HYSCHWARZ;
   }
   else if (!strcmp(name, "ddict"))
   {
      strcpy(HYPreconName_, name);
      HYPreconID_ = HYDDICT;
   }
   else if (!strcmp(name, "poly"))
   {
      strcpy(HYPreconName_, name);
      HYPreconID_ = HYPOLY;
   }
   else if (!strcmp(name, "euclid"))
   {
      strcpy(HYPreconName_, name);
      HYPreconID_ = HYEUCLID;
   }
   else if (!strcmp(name, "blockP"))
   {
      strcpy(HYPreconName_, name);
      HYPreconID_ = HYBLOCK;
   }
   else if (!strcmp(name, "ml"))
   {
#ifdef HAVE_ML
      strcpy(HYPreconName_, name);
      HYPreconID_ = HYML;
#else
      if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3)
      {
         printf("selectPreconditioner - ML not available.\n");
         printf("                       set default to diagonal.\n");
      }
      strcpy(HYPreconName_, "diagonal");
      HYPreconID_ = HYDIAGONAL;
#endif
   }
   else if (!strcmp(name, "mlmaxwell"))
   {
#ifdef HAVE_MLMAXWELL
      strcpy(HYPreconName_, name);
      HYPreconID_ = HYMLMAXWELL;
#else
      if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3)
      {
         printf("selectPreconditioner - MLMaxwell not available.\n");
         printf("                       set default to diagonal.\n");
      }
      strcpy(HYPreconName_, "diagonal");
      HYPreconID_ = HYDIAGONAL;
#endif
   }
   else if (!strcmp(name, "mli"))
   {
#ifdef HAVE_MLI
      strcpy(HYPreconName_, name);
      HYPreconID_ = HYMLI;
#else
      if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3)
      {
         printf("selectPreconditioner - MLI not available.\n");
         printf("                       set default to diagonal.\n");
      }
      strcpy(HYPreconName_, "diagonal");
      HYPreconID_ = HYDIAGONAL;
#endif
   }
   else if (!strcmp(name, "ams"))
   {
      strcpy(HYPreconName_, name);
      HYPreconID_ = HYAMS;
   }
   else if (!strcmp(name, "uzawa"))
   {
      strcpy(HYPreconName_, name);
      HYPreconID_ = HYUZAWA;
   }
#ifdef HAVE_SYSPDE
   else if (!strcmp(name, "syspde"))
   {
      strcpy(HYPreconName_, name);
      HYPreconID_ = HYSYSPDE;
   }
#endif
#ifdef HAVE_DSUPERLU
   else if (!strcmp(name, "dsuperlu"))
   {
      strcpy(HYPreconName_, name);
      HYPreconID_ = HYDSLU;
   }
#endif
   else
   {
      if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3)
      {
         printf("selectPreconditioner error : invalid option.\n");
         printf("                     use default = diagonal.\n");
      }
      strcpy(HYPreconName_, "diagonal");
      HYPreconID_ = HYDIAGONAL;
   }

   //-------------------------------------------------------------------
   // instantiate preconditioner
   //-------------------------------------------------------------------

   switch (HYPreconID_)
   {
      case HYIDENTITY :
           HYPrecon_ = NULL;
           break;

      case HYDIAGONAL :
           HYPrecon_ = NULL;
           break;

      case HYPILUT :
           ierr = HYPRE_ParCSRPilutCreate(comm_, &HYPrecon_);
           //assert(!ierr);
           HYPRE_ParCSRPilutSetMaxIter(HYPrecon_, 1);
           break;

      case HYPARASAILS :
           ierr = HYPRE_ParCSRParaSailsCreate(comm_, &HYPrecon_);
           //assert(!ierr);
           break;

      case HYBOOMERAMG :
           HYPRE_BoomerAMGCreate(&HYPrecon_);
           HYPRE_BoomerAMGSetMaxIter(HYPrecon_, 1);
           HYPRE_BoomerAMGSetCycleType(HYPrecon_, 1);
           HYPRE_BoomerAMGSetMaxLevels(HYPrecon_, 25);
           HYPRE_BoomerAMGSetMeasureType(HYPrecon_, 0);
           break;

      case HYDDILUT :
           ierr = HYPRE_LSI_DDIlutCreate(comm_, &HYPrecon_);
           //assert( !ierr );
           break;

      case HYSCHWARZ :
           ierr = HYPRE_LSI_SchwarzCreate(comm_, &HYPrecon_);
           //assert( !ierr );
           break;

      case HYDDICT :
           ierr = HYPRE_LSI_DDICTCreate(comm_, &HYPrecon_);
           //assert( !ierr );
           break;

      case HYPOLY :
           ierr = HYPRE_LSI_PolyCreate(comm_, &HYPrecon_);
           //assert( !ierr );
           break;

      case HYEUCLID :
           ierr = HYPRE_EuclidCreate(comm_, &HYPrecon_);
           //assert( !ierr );
           break;

      case HYBLOCK :
           ierr = HYPRE_LSI_BlockPrecondCreate(comm_, &HYPrecon_);
           //assert( !ierr );
           break;

      case HYML :
#ifdef HAVE_ML
           ierr = HYPRE_LSI_MLCreate(comm_, &HYPrecon_);
#else
           printf("HYPRE_LSC::selectPreconditioner - ML not supported.\n");
#endif
           break;
      case HYMLI :
#ifdef HAVE_MLI
           ierr = HYPRE_LSI_MLICreate(comm_, &HYPrecon_);
#else
           printf("HYPRE_LSC::selectPreconditioner - MLI not supported.\n");
#endif
           break;
      case HYMLMAXWELL :
#ifdef HAVE_MLMAXWELL
           ierr = HYPRE_LSI_MLMaxwellCreate(comm_, &HYPrecon_);
#else
           printf("HYPRE_LSC::selectPreconditioner-MLMaxwell unsupported.\n");
#endif
           break;
      case HYAMS :
           ierr = HYPRE_AMSCreate(&HYPrecon_);
           break;
      case HYUZAWA :
           HYPRE_LSI_UzawaCreate(comm_, &HYPrecon_);
           break;
      case HYSYSPDE :
#ifdef HAVE_SYSPDE
           ierr = HYPRE_ParCSRSysPDECreate(comm_, sysPDENVars_, &HYPrecon_);
#else
           printf("HYPRE_LSC::selectPreconditioner-SYSPDE unsupported.\n");
#endif
           break;
      case HYDSLU :
#ifdef HAVE_DSUPERLU
           ierr = HYPRE_LSI_DSuperLUCreate(comm_, &HYPrecon_);
#else
           printf("HYPRE_LSC::selectPreconditioner-DSUPERLU unsupported.\n");
#endif
           break;
   }

   //-------------------------------------------------------------------
   // diagnostic message
   //-------------------------------------------------------------------

   if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3)
      printf("%4d : HYPRE_LSC::leaving  selectPreconditioner.\n",mypid_);
}

//***************************************************************************
// form the residual vector
//---------------------------------------------------------------------------

int HYPRE_LinSysCore::formResidual(double* values, int leng)
{
   int                i, index, nrows;
   HYPRE_ParCSRMatrix A_csr;
   HYPRE_ParVector    x_csr;
   HYPRE_ParVector    b_csr;
   HYPRE_ParVector    r_csr;

   //-------------------------------------------------------------------
   // diagnostic message and error checking
   //-------------------------------------------------------------------

   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
      printf("%4d : HYPRE_LSC::entering formResidual.\n", mypid_);

   nrows = localEndRow_ - localStartRow_ + 1;
   if (leng != nrows)
   {
      printf("%4d : HYPRE_LSC::formResidual ERROR - inleng != numLocalRows",
             mypid_);
      printf("                 numLocalRows, inleng = %d %d",nrows,leng);
      return (0);
   }
   if ( ! systemAssembled_ )
   {
      printf("%4d : HYPRE_LSC formResidual ERROR : system not assembled.\n",
             mypid_);
      exit(1);
   }

   //-------------------------------------------------------------------
   // fetch matrix and vector pointers
   //-------------------------------------------------------------------

   HYPRE_IJMatrixGetObject(HYA_, (void **) &A_csr);
   HYPRE_IJVectorGetObject(HYx_, (void **) &x_csr);
   HYPRE_IJVectorGetObject(HYb_, (void **) &b_csr);
   HYPRE_IJVectorGetObject(HYr_, (void **) &r_csr);

   //-------------------------------------------------------------------
   // form residual vector
   //-------------------------------------------------------------------

   HYPRE_ParVectorCopy( b_csr, r_csr );
   HYPRE_ParCSRMatrixMatvec( -1.0, A_csr, x_csr, 1.0, r_csr );

   //-------------------------------------------------------------------
   // fetch residual vector
   //-------------------------------------------------------------------

   for ( i = localStartRow_-1; i < localEndRow_; i++ )
   {
      index = i - localStartRow_ + 1;
      HYPRE_IJVectorGetValues(HYr_, 1, &i, &values[index]);
   }

   //-------------------------------------------------------------------
   // diagnostic message 
   //-------------------------------------------------------------------

   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
      printf("%4d : HYPRE_LSC::leaving  formResidual.\n", mypid_);
   return (0);
}

//***************************************************************************
// solve the linear system
//---------------------------------------------------------------------------

int HYPRE_LinSysCore::launchSolver(int& solveStatus, int &iterations)
{
   int                i, j, numIterations=0, status, ierr, localNRows;
   int                startRow, *procNRows, rowSize, *colInd, nnz, nrows;
   int                slideCheck[2];
#ifdef HAVE_MLI
   int                *constrMap, *constrEqns, ncount, *iArray;
   double             *tempNodalCoord; 
#endif
   int                *numSweeps, *relaxType, reduceAFlag;
   int                *matSizes, *rowInd, retFlag, tempIter, nTrials;
   double             rnorm=0.0, ddata, *colVal, *relaxWt, *diagVals;
   double             stime, etime, ptime, rtime1, rtime2, newnorm;
   double             rnorm0, rnorm1, convRate, rateThresh; 
   char               fname[40], paramString[100];
   FILE               *fp;
   HYPRE_IJMatrix     TempA, IJI;
   HYPRE_IJVector     TempX, TempB, TempR;
#ifdef HAVE_MLI
   HYPRE_ParCSRMatrix perturb_csr;
#endif
   HYPRE_ParCSRMatrix A_csr, I_csr, normalA_csr;
   HYPRE_ParVector    x_csr, b_csr, r_csr;
   HYPRE_SlideReduction *slideObj = (HYPRE_SlideReduction *) slideObj_;

   //-------------------------------------------------------------------
   // diagnostic message 
   //-------------------------------------------------------------------

   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
      printf("%4d : HYPRE_LSC::entering launchSolver.\n", mypid_);

   //-------------------------------------------------------------------
   // see if Schur or slide reduction is to be performed
   //-------------------------------------------------------------------

   rnorm_ = 0.0;
   MPI_Barrier(comm_);
   rtime1  = LSC_Wtime();
   if ( schurReduction_ == 1 && schurReductionCreated_ == 0 )
   {
      buildSchurReducedSystem();
      schurReductionCreated_ = 1;
   }
   else if ( schurReduction_ == 1 ) buildSchurReducedRHS();

   if ( schurReduction_ == 0 && slideReduction_ != 0 ) 
   {
      if ( constrList_ != NULL ) delete [] constrList_;
      constrList_ = NULL;
      if      ( slideReduction_ == 1 ) buildSlideReducedSystem();
      else if ( slideReduction_ == 2 ) buildSlideReducedSystem2();
      else if ( slideReduction_ == 3 || slideReduction_ == 4 ) 
      {
         if (slideObj == NULL)
         {
            slideObj = new HYPRE_SlideReduction(comm_);
            slideObj_ = (void *) slideObj;
         }
         TempA = currA_;
         TempX = currX_;
         TempB = currB_;
         TempR = currR_;
         HYPRE_IJVectorGetLocalRange(HYb_,&slideCheck[0],&slideCheck[1]);
         // check to see if it has been reduced before
         // if so, need to redo B and X
         reduceAFlag = 1;
         if (currA_ != HYA_)
         {
            HYPRE_IJVectorDestroy(currB_);
            HYPRE_IJVectorDestroy(currX_);
            HYPRE_IJVectorDestroy(currR_);
            currB_ = HYb_;
            currX_ = HYx_;
            currR_ = HYr_;
            reduceAFlag = 0;
         }

         if ( HYOutputLevel_ & HYFEI_SLIDEREDUCE1 )
            slideObj->setOutputLevel(1);
         if ( HYOutputLevel_ & HYFEI_SLIDEREDUCE2 )
            slideObj->setOutputLevel(2);
         if ( HYOutputLevel_ & HYFEI_SLIDEREDUCE3 )
            slideObj->setOutputLevel(3);
         if ( slideReductionMinNorm_ >= 0.0 )
            slideObj->setBlockMinNorm( slideReductionMinNorm_ );
         if ( slideReductionScaleMatrix_ == 1 )
            slideObj->setScaleMatrix();
         slideObj->setTruncationThreshold( truncThresh_ );
         if ( slideReduction_ == 4 ) slideObj->setUseSimpleScheme();
         slideObj->setup(currA_, currX_, currB_);
         if ( slideReductionScaleMatrix_ == 1 && HYPreconID_ == HYMLI )
         {
            diagVals = slideObj->getMatrixDiagonal();
            nrows    = slideObj->getMatrixNumRows();
            HYPRE_LSI_MLILoadMatrixScalings(HYPrecon_, nrows, diagVals);
         }
#ifdef HAVE_MLI
         if ( HYPreconID_ == HYMLI )
         {
            HYPRE_IJMatrixGetObject(currA_, (void **) &A_csr);
            HYPRE_ParCSRMatrixGetRowPartitioning( A_csr, &procNRows );
            slideObj->getProcConstraintMap(&constrMap);
            HYPRE_LSI_MLIAdjustNodeEqnMap(HYPrecon_, procNRows, constrMap);
            j = constrMap[mypid_+1] - constrMap[mypid_];
            free(procNRows);
            slideObj->getSlaveEqnList(&constrEqns);
            slideObj->getPerturbationMatrix(&perturb_csr);
            HYPRE_LSI_MLIAdjustNullSpace(HYPrecon_,j,constrEqns,perturb_csr);
         }
#endif
         if (reduceAFlag == 1)
         {
            slideObj->getReducedMatrix(&currA_);
            slideObj->getReducedAuxVector(&currR_);
         }
         slideObj->getReducedSolnVector(&currX_);
         slideObj->getReducedRHSVector(&currB_);
         if ( currA_ == NULL )
         {
            currA_ = TempA;
            currX_ = TempX;
            currB_ = TempB;
            currR_ = TempR;
         }
      }
   }

   MPI_Barrier(comm_);
   rtime2  = LSC_Wtime();
   
   //-------------------------------------------------------------------
   // if normal equation requested
   //-------------------------------------------------------------------

   if ( (normalEqnFlag_ & 1) != 0 )
   {
      if ( (normalEqnFlag_ & 2) == 0 )
      {
         if ( HYnormalA_ != NULL ) HYPRE_IJMatrixDestroy(HYnormalA_);
         HYPRE_IJMatrixGetObject(HYA_, (void **) &A_csr);
         ierr = HYPRE_IJMatrixCreate(comm_, localStartRow_-1,
		 localEndRow_-1, localStartRow_-1, localEndRow_-1,&IJI);
         ierr += HYPRE_IJMatrixSetObjectType(IJI, HYPRE_PARCSR);
         //assert(!ierr);
         localNRows = localEndRow_ - localStartRow_ + 1;
         matSizes = new int[localNRows];
         rowInd   = new int[localNRows];
         colInd   = new int[localNRows];
         colVal   = new double[localNRows];
         for ( i = 0; i < localNRows; i++ ) 
         {
            matSizes[i] = 1;
            rowInd[i] = localStartRow_ - 1 + i;
            colInd[i] = rowInd[i];
            colVal[i] = 1.0;
         }
         ierr  = HYPRE_IJMatrixSetRowSizes(IJI, matSizes);
         ierr += HYPRE_IJMatrixInitialize(IJI);
         ierr += HYPRE_IJMatrixSetValues(IJI, localNRows, matSizes,
                   (const int *) rowInd, (const int *) colInd,
                   (const double *) colVal);
         //assert(!ierr);
         delete [] rowInd;
         delete [] colInd;
         delete [] colVal;
         HYPRE_IJMatrixAssemble(IJI);
         HYPRE_IJMatrixGetObject(IJI, (void **) &I_csr);
         hypre_BoomerAMGBuildCoarseOperator((hypre_ParCSRMatrix*) A_csr,
             (hypre_ParCSRMatrix*) I_csr, (hypre_ParCSRMatrix*) A_csr, 
             (hypre_ParCSRMatrix**) &normalA_csr);
         HYPRE_IJMatrixDestroy( IJI );
         ierr = HYPRE_IJMatrixCreate(comm_, localStartRow_-1,
		 localEndRow_-1, localStartRow_-1, localEndRow_-1,&HYnormalA_);
         ierr += HYPRE_IJMatrixSetObjectType(HYnormalA_, HYPRE_PARCSR);
         //assert(!ierr);
         for ( i = localStartRow_-1; i < localEndRow_; i++ )
         {
            HYPRE_ParCSRMatrixGetRow(normalA_csr,i,&rowSize,NULL,NULL);
            matSizes[i-localStartRow_+1] = rowSize;
            HYPRE_ParCSRMatrixRestoreRow(normalA_csr,i,&rowSize,NULL,NULL);
         }
         ierr  = HYPRE_IJMatrixSetRowSizes(HYnormalA_, matSizes);
         ierr += HYPRE_IJMatrixInitialize(HYnormalA_);
         for ( i = localStartRow_-1; i < localEndRow_; i++ )
         {
            HYPRE_ParCSRMatrixGetRow(normalA_csr,i,&rowSize,&colInd,&colVal);
            ierr += HYPRE_IJMatrixSetValues(HYnormalA_, 1, &rowSize,
                      (const int *) &i, (const int *) colInd,
                      (const double *) colVal);
            HYPRE_ParCSRMatrixRestoreRow(normalA_csr,i,&rowSize,&colInd,&colVal);
         }
         HYPRE_IJMatrixAssemble(HYnormalA_);
         delete [] matSizes;
         normalEqnFlag_ |= 2;
      }
      if ( (normalEqnFlag_ & 4) == 0 )
      {
         if ( HYnormalB_ != NULL ) HYPRE_IJVectorDestroy(HYnormalB_);
         HYPRE_IJVectorCreate(comm_, localStartRow_-1, localEndRow_-1,
                              &HYnormalB_);
         HYPRE_IJVectorSetObjectType(HYnormalB_, HYPRE_PARCSR);
         HYPRE_IJVectorInitialize(HYnormalB_);
         HYPRE_IJVectorAssemble(HYnormalB_);
         HYPRE_IJMatrixGetObject(HYA_, (void **) &A_csr);
         HYPRE_IJVectorGetObject(HYb_, (void **) &b_csr);
         HYPRE_IJVectorGetObject(HYnormalB_, (void **) &r_csr);
         HYPRE_ParCSRMatrixMatvecT( 1.0, A_csr, b_csr, 0.0, r_csr );
         normalEqnFlag_ |= 4;
      }
   }

   //-------------------------------------------------------------------
   // fetch matrix and vector pointers
   //-------------------------------------------------------------------

   HYPRE_IJMatrixGetObject(currA_, (void **) &A_csr);
   HYPRE_IJVectorGetObject(currX_, (void **) &x_csr);
   HYPRE_IJVectorGetObject(currB_, (void **) &b_csr);
   HYPRE_IJVectorGetObject(currR_, (void **) &r_csr);
   if ( A_csr == NULL || x_csr == NULL || b_csr == NULL || r_csr == NULL )
   {
      printf("%4d : HYPRE_LSC::launchSolver ERROR.\n",mypid_);
      printf("             csr pointers null \n");
      printf("             Did you forget to call matrixLoadComplete?\n");
      exit(1);
   }
   if ( (normalEqnFlag_ & 7) == 7 ) 
   {
      HYPRE_IJMatrixGetObject(HYnormalA_, (void **) &A_csr);
      HYPRE_IJVectorGetObject(HYnormalB_, (void **) &b_csr);
   }

   //-------------------------------------------------------------------
   // diagnostics (print the reduced matrix to a file)
   //-------------------------------------------------------------------

   if ( HYOutputLevel_ & HYFEI_PRINTREDMAT )
   {
      if ( HYOutputLevel_ & HYFEI_PRINTPARCSRMAT )
      {
         printf("%4d : HYPRE_LSC::print matrix/rhs to files(A)\n",mypid_);
         sprintf(fname, "HYPRE_Mat");
         HYPRE_ParCSRMatrixPrint( A_csr, fname);
         sprintf(fname, "HYPRE_RHS");
         HYPRE_ParVectorPrint( b_csr, fname);
      }
      else
      {
         printf("%4d : HYPRE_LSC::print matrix/rhs to files(B)\n",mypid_);
         HYPRE_ParCSRMatrixGetRowPartitioning( A_csr, &procNRows );
         startRow = procNRows[mypid_];
         nrows = procNRows[mypid_+1] - startRow;
         free( procNRows );

         sprintf(fname, "hypre_mat.out.%d", mypid_);
         fp = fopen( fname, "w");
         nnz = 0;
         for ( i = startRow; i < startRow+nrows; i++ )
         {
            HYPRE_ParCSRMatrixGetRow(A_csr,i,&rowSize,&colInd,&colVal);
            for ( j = 0; j < rowSize; j++ ) if ( colVal[j] != 0.0 ) nnz++;
            HYPRE_ParCSRMatrixRestoreRow(A_csr,i,&rowSize,&colInd,&colVal);
         }
         fprintf(fp, "%6d  %7d \n", nrows, nnz);
         for ( i = startRow; i < startRow+nrows; i++ )
         {
            HYPRE_ParCSRMatrixGetRow(A_csr,i,&rowSize,&colInd,&colVal);
            for ( j = 0; j < rowSize; j++ )
            {
               if ( colVal[j] != 0.0 )
                  fprintf(fp, "%6d  %6d  %25.8e\n",i+1,colInd[j]+1,colVal[j]);
            }
            HYPRE_ParCSRMatrixRestoreRow(A_csr,i,&rowSize,&colInd,&colVal);
         }
         fclose(fp);

         sprintf(fname, "hypre_rhs.out.%d", mypid_);
         fp = fopen( fname, "w");
         fprintf(fp, "%6d \n", nrows);
         for ( i = startRow; i < startRow+nrows; i++ )
         {
            HYPRE_IJVectorGetValues(currB_, 1, &i, &ddata);
            fprintf(fp, "%6d  %25.8e \n", i+1, ddata);
         }
         fclose(fp);
         MPI_Barrier(comm_);
      }
      if ( MLI_NumNodes_ > 0 )
      {
         fp = fopen("rbm","w");
         for (i = 0; i < MLI_NumNodes_; i++)
            for (j = 0; j < MLI_FieldSize_; j++)
               fprintf(fp,"%8d %25.16e\n", MLI_EqnNumbers_[i]+j+1, 
                       MLI_NodalCoord_[i*3+j]);
         fclose(fp);
      }
      if ( HYOutputLevel_ & HYFEI_STOPAFTERPRINT ) exit(1);
   }

#ifdef HAVE_AMGE
   if ( HYOutputLevel_ & HYFEI_PRINTFEINFO )
   {
      HYPRE_LSI_AMGeWriteToFile();
   }
#endif

   //-------------------------------------------------------------------
   // choose PCG, GMRES, ... or direct solver
   //-------------------------------------------------------------------

   MPI_Barrier(comm_);
   status = 1;
   stime  = LSC_Wtime();
   ptime  = stime;

   if ( projectionScheme_ == 1 )
   {
      computeAConjProjection(A_csr, x_csr, b_csr);
   } 
   else if ( projectionScheme_ == 2 )
   {
      computeMinResProjection(A_csr, x_csr, b_csr);
   } 
   
#ifdef HAVE_MLI
   if ( HYPreconID_ == HYMLI && feData_ != NULL )
   {
      if (haveFEData_ == 1) HYPRE_LSI_MLISetFEData( HYPrecon_, feData_ );
      if (haveFEData_ == 2) HYPRE_LSI_MLISetSFEI( HYPrecon_, feData_ );
   }
   if ( HYPreconID_ == HYMLI && MLI_EqnNumbers_ != NULL )
   {
      iArray = new int[MLI_NumNodes_];
      for (i = 0; i < MLI_NumNodes_; i++) iArray[i] = i;
      HYPRE_LSI_qsort1a(MLI_EqnNumbers_, iArray, 0, MLI_NumNodes_-1);
      tempNodalCoord = MLI_NodalCoord_; 
      ncount = 1;
      for (i = 1; i < MLI_NumNodes_; i++) 
         if (MLI_EqnNumbers_[i] != MLI_EqnNumbers_[ncount-1]) ncount++;
      MLI_NodalCoord_ = new double[ncount*MLI_FieldSize_];
      for (j = 0; j < MLI_FieldSize_; j++) 
         MLI_NodalCoord_[j] = tempNodalCoord[iArray[0]*MLI_FieldSize_+j];
      ncount = 1;
      for (i = 1; i < MLI_NumNodes_; i++) 
      {
         if (MLI_EqnNumbers_[i] != MLI_EqnNumbers_[ncount-1]) 
         {
            MLI_EqnNumbers_[ncount] = MLI_EqnNumbers_[i];
            for (j = 0; j < MLI_FieldSize_; j++) 
               MLI_NodalCoord_[ncount*MLI_FieldSize_+j] =
                  tempNodalCoord[iArray[i]*MLI_FieldSize_+j];
            ncount++;
         }
      }
      MLI_NumNodes_ = ncount;
      //assert((MLI_NumNodes_*MLI_FieldSize_)==(localEndRow_-localStartRow_+1));
      delete [] tempNodalCoord;
      delete [] iArray;
      for (i = 0; i < MLI_NumNodes_; i++) 
      {
         if (MLI_NodalCoord_[i] == -99999.0) 
            printf("%d : HYPRE launchSolver ERROR - coord %d not filled.\n",
                   mypid_, i);
      }
      HYPRE_LSI_MLILoadNodalCoordinates(HYPrecon_, MLI_NumNodes_, 
               MLI_FieldSize_, MLI_EqnNumbers_, MLI_FieldSize_, 
               MLI_NodalCoord_, localEndRow_-localStartRow_+1);
   }
#endif
#if 0
   // replaced by better scheme, to be deleted later
   if ( HYPreconID_ == HYAMS && MLI_EqnNumbers_ != NULL )
   {
      HYPRE_LSI_BuildNodalCoordinates();
   }
#endif

   switch ( HYSolverID_ )
   {
      //----------------------------------------------------------------
      // choose PCG 
      //----------------------------------------------------------------

      case HYPCG :
           if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0 )
           {
              printf("***************************************************\n");
              printf("* Preconditioned Conjugate Gradient solver \n");
              printf("* maximum no. of iterations = %d\n", maxIterations_);
              printf("* convergence tolerance     = %e\n", tolerance_);
              printf("*--------------------------------------------------\n");
           }
           setupPCGPrecon();
           HYPRE_ParCSRPCGSetMaxIter(HYSolver_, maxIterations_);
           HYPRE_ParCSRPCGSetRelChange(HYSolver_, 0);
           HYPRE_ParCSRPCGSetTwoNorm(HYSolver_, 1);
           HYPRE_PCGSetRecomputeResidual(HYSolver_, pcgRecomputeRes_);
           if ( normAbsRel_ == 0 )
           {
              HYPRE_PCGSetStopCrit(HYSolver_,0);
              HYPRE_PCGSetTol(HYSolver_, tolerance_);
           }
           else
           {
              HYPRE_PCGSetStopCrit(HYSolver_,1);
              HYPRE_PCGSetAbsoluteTol(HYSolver_, tolerance_);
           }
           if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 )
           {
              if ( mypid_ == 0 )
                printf("***************************************************\n");
              HYPRE_ParCSRPCGSetPrintLevel(HYSolver_, 1);
              if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
                 HYPRE_ParCSRPCGSetPrintLevel(HYSolver_, 2);
           }
           retFlag = HYPRE_ParCSRPCGSetup(HYSolver_, A_csr, b_csr, x_csr);
           if ( retFlag != 0 )
           {
              printf("HYPRE_LSC::launchSolver ERROR : in PCG setup.\n");
              return retFlag;
           }
           // if variable mli preconditioner (SA and GSA)
           if ( MLI_Hybrid_GSA_ && HYPreconID_ == HYMLI )
           {
              if ( normAbsRel_ == 0 )
              {
                 HYPRE_ParVectorCopy( b_csr, r_csr );
                 HYPRE_ParCSRMatrixMatvec( -1.0, A_csr, x_csr, 1.0, r_csr );
                 HYPRE_ParVectorInnerProd( r_csr, r_csr, &rnorm0);
                 rnorm0 = sqrt( rnorm0 );
              }
              else rnorm0 = 1.0;
              HYPRE_ParCSRPCGSetMaxIter(HYSolver_, MLI_Hybrid_MaxIter_);
              rateThresh = 1.0;
              for ( i = 0; i < MLI_Hybrid_MaxIter_; i++ )
                 rateThresh *= MLI_Hybrid_ConvRate_;
           }
           MPI_Barrier( comm_ );
           ptime  = LSC_Wtime();
           retFlag = HYPRE_ParCSRPCGSolve(HYSolver_, A_csr, b_csr, x_csr);
           HYPRE_ParCSRPCGGetNumIterations(HYSolver_, &numIterations);
           HYPRE_ParVectorCopy( b_csr, r_csr );
           HYPRE_ParCSRMatrixMatvec( -1.0, A_csr, x_csr, 1.0, r_csr );
           HYPRE_ParVectorInnerProd( r_csr, r_csr, &rnorm);
           rnorm = sqrt( rnorm );
           // if variable mli preconditioner (SA and GSA)
           if ( MLI_Hybrid_GSA_ && HYPreconID_ == HYMLI )
           {
              nTrials = 1;
              if (rnorm/rnorm0 >= tolerance_)
              {
                 HYPRE_ParCSRPCGSolve(HYSolver_, A_csr, b_csr, x_csr);
                 HYPRE_ParCSRPCGGetNumIterations(HYSolver_, &tempIter);
                 numIterations += tempIter;
                 HYPRE_ParVectorCopy( b_csr, r_csr );
                 HYPRE_ParCSRMatrixMatvec( -1.0, A_csr, x_csr, 1.0, r_csr );
                 HYPRE_ParVectorInnerProd( r_csr, r_csr, &rnorm1);
                 rnorm1 = sqrt( rnorm1 );
                 convRate = rnorm1 / rnorm;
                 rnorm = rnorm1;
              }
              while ((rnorm/rnorm0)>=tolerance_ && nTrials<MLI_Hybrid_NTrials_)
              {
                 nTrials++;
                 if ( convRate > rateThresh )
                 {
                    if ( MLI_Hybrid_NSIncr_ > 1 )
                       sprintf(paramString, "MLI incrNullSpaceDim %d", 
                               MLI_Hybrid_NSIncr_);
                    else
                       sprintf(paramString, "MLI incrNullSpaceDim 2");
                    HYPRE_LSI_MLISetParams(HYPrecon_, paramString);
                    HYPRE_ParCSRPCGSetup(HYSolver_, A_csr, b_csr, x_csr);
                 }
                 HYPRE_ParCSRPCGSolve(HYSolver_, A_csr, b_csr, x_csr);
                 HYPRE_ParCSRPCGGetNumIterations(HYSolver_, &tempIter);
                 numIterations += tempIter;
                 HYPRE_ParVectorCopy( b_csr, r_csr );
                 HYPRE_ParCSRMatrixMatvec( -1.0, A_csr, x_csr, 1.0, r_csr );
                 rnorm1 = rnorm;
                 HYPRE_ParVectorInnerProd( r_csr, r_csr, &rnorm);
                 rnorm = sqrt( rnorm );
                 convRate = rnorm / rnorm1;
              }
              if (rnorm/rnorm0 < tolerance_) retFlag = 0;
              else if (numIterations < maxIterations_)
              {
                 HYPRE_ParCSRPCGSetMaxIter(HYSolver_,maxIterations_-numIterations);
                 retFlag = HYPRE_ParCSRPCGSolve(HYSolver_, A_csr, b_csr, x_csr);
                 HYPRE_ParCSRPCGGetNumIterations(HYSolver_, &tempIter);
                 numIterations += tempIter;
              }
              else retFlag = 1;
           }
           if ( retFlag != 0 )
           {
              printf("HYPRE_LSC::launchSolver ERROR : in PCG solve.\n");
              return retFlag;
           }
           switch ( projectionScheme_ )
           {
              case 1 : addToAConjProjectionSpace(currX_,currB_);  break;
              case 2 : addToMinResProjectionSpace(currX_,currB_); break;
           }
           if ( numIterations >= maxIterations_ ) status = 1; else status = 0;
           break;

      //----------------------------------------------------------------
      // choose LSICG 
      //----------------------------------------------------------------

      case HYLSICG :
           if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0 )
           {
              printf("***************************************************\n");
              printf("* Conjugate Gradient solver \n");
              printf("* maximum no. of iterations = %d\n", maxIterations_);
              printf("* convergence tolerance     = %e\n", tolerance_);
              printf("*--------------------------------------------------\n");
           }
           setupLSICGPrecon();
           HYPRE_ParCSRLSICGSetMaxIter(HYSolver_, maxIterations_);
           HYPRE_ParCSRLSICGSetTol(HYSolver_, tolerance_);
           if (normAbsRel_ == 0) HYPRE_ParCSRLSICGSetStopCrit(HYSolver_,0);
           else                  HYPRE_ParCSRLSICGSetStopCrit(HYSolver_,1);
           if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 )
           {
              if ( mypid_ == 0 )
                printf("***************************************************\n");
              HYPRE_ParCSRLSICGSetLogging(HYSolver_, 1);
              if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
                 HYPRE_ParCSRLSICGSetLogging(HYSolver_, 2);
           }
           retFlag = HYPRE_ParCSRLSICGSetup(HYSolver_, A_csr, b_csr, x_csr);
           if ( retFlag != 0 )
           {
              printf("HYPRE_LSC::launchSolver ERROR : in LSICG setup.\n");
              return retFlag;
           }
           MPI_Barrier( comm_ );
           ptime  = LSC_Wtime();
           retFlag = HYPRE_ParCSRLSICGSolve(HYSolver_, A_csr, b_csr, x_csr);
           if ( retFlag != 0 )
           {
              printf("HYPRE_LSC::launchSolver ERROR : in LSICG solve.\n");
              return retFlag;
           }
           HYPRE_ParCSRLSICGGetNumIterations(HYSolver_, &numIterations);
           HYPRE_ParVectorCopy( b_csr, r_csr );
           HYPRE_ParCSRMatrixMatvec( -1.0, A_csr, x_csr, 1.0, r_csr );
           HYPRE_ParVectorInnerProd( r_csr, r_csr, &rnorm);
           rnorm = sqrt( rnorm );
           switch ( projectionScheme_ )
           {
              case 1 : addToAConjProjectionSpace(currX_,currB_);  break;
              case 2 : addToMinResProjectionSpace(currX_,currB_); break;
           }
           if ( numIterations >= maxIterations_ ) status = 1; else status = 0;
           break;

      //----------------------------------------------------------------
      // choose hybrid method : CG with diagonal/BoomerAMG preconditioner 
      //----------------------------------------------------------------

      case HYHYBRID :
           if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0 )
           {
              printf("***************************************************\n");
              printf("* PCG with hybrid diagonal/BoomerAMG preconditioner\n");
              printf("* maximum no. of iterations = %d\n", maxIterations_);
              printf("* convergence tolerance     = %e\n", tolerance_);
              printf("*--------------------------------------------------\n");
           }
           HYPRE_ParCSRHybridSetPCGMaxIter(HYSolver_, maxIterations_);
           HYPRE_ParCSRHybridSetTol(HYSolver_, tolerance_);
           HYPRE_ParCSRHybridSetRelChange(HYSolver_, 0);
           HYPRE_ParCSRHybridSetTwoNorm(HYSolver_, 1);
           HYPRE_ParCSRHybridSetConvergenceTol(HYSolver_, 0.9);
           HYPRE_ParCSRHybridSetDSCGMaxIter(HYSolver_, 20);
           if ( HYOutputLevel_ & HYFEI_AMGDEBUG )
              HYPRE_ParCSRHybridSetPrintLevel(HYSolver_, 32);
           HYPRE_ParCSRHybridSetCoarsenType(HYSolver_, amgCoarsenType_);
           HYPRE_ParCSRHybridSetMeasureType(HYSolver_, amgMeasureType_);
           HYPRE_ParCSRHybridSetStrongThreshold(HYSolver_,amgStrongThreshold_);
           numSweeps = hypre_CTAlloc(int,4);
           for ( i = 0; i < 4; i++ ) numSweeps[i] = amgNumSweeps_[i];
           HYPRE_ParCSRHybridSetNumGridSweeps(HYSolver_, numSweeps);
           relaxType = hypre_CTAlloc(int,4);
           for ( i = 0; i < 4; i++ ) relaxType[i] = amgRelaxType_[i];
           HYPRE_ParCSRHybridSetGridRelaxType(HYSolver_, relaxType);
           relaxWt = hypre_CTAlloc(double,25);
           for ( i = 0; i < 25; i++ ) relaxWt[i] = amgRelaxWeight_[i];
           HYPRE_ParCSRHybridSetRelaxWeight(HYSolver_, relaxWt);
           if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 )
           {
              if ( mypid_ == 0 )
                printf("***************************************************\n");
              HYPRE_ParCSRHybridSetLogging(HYSolver_, 1);
              if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
                 HYPRE_ParCSRHybridSetLogging(HYSolver_, 2);
           }
           retFlag = HYPRE_ParCSRHybridSetup(HYSolver_, A_csr, b_csr, x_csr);
           if ( retFlag != 0 )
           {
              printf("HYPRE_LSC::launchSolver ERROR : in Hybrid setup.\n");
              return retFlag;
           }
           MPI_Barrier( comm_ );
           ptime  = LSC_Wtime();
           retFlag = HYPRE_ParCSRHybridSolve(HYSolver_, A_csr, b_csr, x_csr);
           if ( retFlag != 0 )
           {
              printf("HYPRE_LSC::launchSolver ERROR : in Hybrid solve.\n");
              return retFlag;
           }
           HYPRE_ParCSRHybridGetNumIterations(HYSolver_, &numIterations);
           HYPRE_ParVectorCopy( b_csr, r_csr );
           HYPRE_ParCSRMatrixMatvec( -1.0, A_csr, x_csr, 1.0, r_csr );
           HYPRE_ParVectorInnerProd( r_csr, r_csr, &rnorm);
           rnorm = sqrt( rnorm );
           if ( numIterations >= maxIterations_ ) status = 1; else status = 0;
           break;

      //----------------------------------------------------------------
      // choose GMRES 
      //----------------------------------------------------------------

      case HYGMRES :
           if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0 )
           {
              printf("***************************************************\n");
              printf("* Generalized Minimal Residual (GMRES) solver \n");
              printf("* restart size              = %d\n", gmresDim_);
              printf("* maximum no. of iterations = %d\n", maxIterations_);
              printf("* convergence tolerance     = %e\n", tolerance_);
              printf("*--------------------------------------------------\n");
           }
           setupGMRESPrecon();
           HYPRE_ParCSRGMRESSetKDim(HYSolver_, gmresDim_);
           HYPRE_ParCSRGMRESSetMaxIter(HYSolver_, maxIterations_);
           if ( normAbsRel_ == 0 )
           {
              HYPRE_GMRESSetStopCrit(HYSolver_,0);
              HYPRE_GMRESSetTol(HYSolver_, tolerance_);
           }
           else
           {
              HYPRE_GMRESSetStopCrit(HYSolver_,1);
              HYPRE_GMRESSetAbsoluteTol(HYSolver_, tolerance_);
           }
           if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 )
           {
              HYPRE_ParCSRGMRESSetPrintLevel(HYSolver_, 1);
              if ( mypid_ == 0 )
                printf("***************************************************\n");
              if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
                 HYPRE_ParCSRGMRESSetPrintLevel(HYSolver_, 2);
           }
           retFlag = HYPRE_ParCSRGMRESSetup(HYSolver_, A_csr, b_csr, x_csr);
           if ( retFlag != 0 )
           {
              printf("HYPRE_LSC::launchSolver ERROR : in GMRES setup.\n");
              return retFlag;
           }
           // if variable mli preconditioner (SA and GSA)
           if ( MLI_Hybrid_GSA_ && HYPreconID_ == HYMLI )
           {
              if ( normAbsRel_ == 0 )
              {
                 HYPRE_ParVectorCopy( b_csr, r_csr );
                 HYPRE_ParCSRMatrixMatvec( -1.0, A_csr, x_csr, 1.0, r_csr );
                 HYPRE_ParVectorInnerProd( r_csr, r_csr, &rnorm0);
                 rnorm0 = sqrt( rnorm0 );
              }
              else rnorm0 = 1.0;
              HYPRE_ParCSRGMRESSetMaxIter(HYSolver_, MLI_Hybrid_MaxIter_);
              rateThresh = 1.0;
              for ( i = 0; i < MLI_Hybrid_MaxIter_; i++ )
                 rateThresh *= MLI_Hybrid_ConvRate_;
           }
           MPI_Barrier( comm_ );
           ptime  = LSC_Wtime();
           retFlag = HYPRE_ParCSRGMRESSolve(HYSolver_, A_csr, b_csr, x_csr);
           HYPRE_ParCSRGMRESGetNumIterations(HYSolver_, &numIterations);
           HYPRE_ParVectorCopy( b_csr, r_csr );
           HYPRE_ParCSRMatrixMatvec( -1.0, A_csr, x_csr, 1.0, r_csr );
           HYPRE_ParVectorInnerProd( r_csr, r_csr, &rnorm);
           rnorm = sqrt( rnorm );
           // if variable mli preconditioner (SA and GSA)
           if ( MLI_Hybrid_GSA_ && HYPreconID_ == HYMLI )
           {
              nTrials = 0;
              //if (rnorm/rnorm0 >= tolerance_)
              //{
              //   HYPRE_ParCSRGMRESSolve(HYSolver_, A_csr, b_csr, x_csr);
              //   HYPRE_ParCSRGMRESGetNumIterations(HYSolver_, &tempIter);
              //   numIterations += tempIter;
              //   HYPRE_ParVectorCopy( b_csr, r_csr );
              //   HYPRE_ParCSRMatrixMatvec( -1.0, A_csr, x_csr, 1.0, r_csr );
              //   HYPRE_ParVectorInnerProd( r_csr, r_csr, &rnorm1);
              //   rnorm1 = sqrt( rnorm1 );
              //   convRate = rnorm1 / rnorm;
              //   rnorm = rnorm1;
              //}
              convRate = rnorm / rnorm0;
              while ((rnorm/rnorm0)>=tolerance_ && nTrials<MLI_Hybrid_NTrials_)
              {
                 nTrials++;
                 if ( convRate > rateThresh )
                 {
                    if ( MLI_Hybrid_NSIncr_ > 1 )
                       sprintf(paramString, "MLI incrNullSpaceDim %d", 
                               MLI_Hybrid_NSIncr_);
                    else
                       sprintf(paramString, "MLI incrNullSpaceDim 2");
                    HYPRE_LSI_MLISetParams(HYPrecon_, paramString);
                    HYPRE_ParCSRGMRESSetup(HYSolver_, A_csr, b_csr, x_csr);
                 }
                 HYPRE_ParCSRGMRESSolve(HYSolver_, A_csr, b_csr, x_csr);
                 HYPRE_ParCSRGMRESGetNumIterations(HYSolver_, &tempIter);
                 numIterations += tempIter;
                 HYPRE_ParVectorCopy( b_csr, r_csr );
                 HYPRE_ParCSRMatrixMatvec( -1.0, A_csr, x_csr, 1.0, r_csr );
                 rnorm1 = rnorm;
                 HYPRE_ParVectorInnerProd( r_csr, r_csr, &rnorm);
                 rnorm = sqrt( rnorm );
                 convRate = rnorm / rnorm1;
              }
              if (rnorm/rnorm0 < tolerance_) retFlag = 0;
              else if (numIterations < maxIterations_)
              {
                 HYPRE_ParCSRGMRESSetMaxIter(HYSolver_,
                                             maxIterations_-numIterations);
                 retFlag = HYPRE_ParCSRGMRESSolve(HYSolver_,A_csr,b_csr,x_csr);
                 HYPRE_ParCSRGMRESGetNumIterations(HYSolver_, &tempIter);
                 numIterations += tempIter;
              }
              else retFlag = 1;
           }
           if ( retFlag != 0 )
           {
              printf("HYPRE_LSC::launchSolver ERROR : in GMRES solve.\n");
              return retFlag;
           }
           switch ( projectionScheme_ )
           {
              case 1 : addToAConjProjectionSpace(currX_,currB_);  break;
              case 2 : addToMinResProjectionSpace(currX_,currB_); break;
           }
           if ( numIterations >= maxIterations_ ) status = 1; else status = 0;
           break;

      //----------------------------------------------------------------
      // choose flexible GMRES 
      //----------------------------------------------------------------

      case HYFGMRES :
           if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0 )
           {
              printf("***************************************************\n");
              printf("* Flexible GMRES solver \n");
              printf("* restart size              = %d\n", gmresDim_);
              printf("* maximum no. of iterations = %d\n", maxIterations_);
              printf("* convergence tolerance     = %e\n", tolerance_);
              printf("*--------------------------------------------------\n");
           }
           setupFGMRESPrecon();
           HYPRE_ParCSRFGMRESSetKDim(HYSolver_, gmresDim_);
           HYPRE_ParCSRFGMRESSetMaxIter(HYSolver_, maxIterations_);
           HYPRE_ParCSRFGMRESSetTol(HYSolver_, tolerance_);
           if ( normAbsRel_ == 0 ) HYPRE_ParCSRFGMRESSetStopCrit(HYSolver_,0);
           else                    HYPRE_ParCSRFGMRESSetStopCrit(HYSolver_,1);
           if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 )
           {
              if ( mypid_ == 0 )
                printf("***************************************************\n");
              HYPRE_ParCSRFGMRESSetLogging(HYSolver_, 1);
           }

           retFlag = HYPRE_ParCSRFGMRESSetup(HYSolver_, A_csr, b_csr, x_csr);
           if ( retFlag != 0 )
           {
              printf("HYPRE_LSC::launchSolver ERROR : in FGMRES setup.\n");
              return retFlag;
           }

           if ( fgmresUpdateTol_ && HYPreconID_ == HYBLOCK )
              HYPRE_ParCSRFGMRESUpdatePrecondTolerance(HYSolver_, 
                      HYPRE_LSI_BlockPrecondSetA11Tolerance);

           MPI_Barrier( comm_ );
           ptime  = LSC_Wtime();
           retFlag = HYPRE_ParCSRFGMRESSolve(HYSolver_, A_csr, b_csr, x_csr);
           if ( retFlag != 0 )
           {
              printf("HYPRE_LSC::launchSolver ERROR : in FGMRES solve.\n");
              return retFlag;
           }
           HYPRE_ParCSRFGMRESGetNumIterations(HYSolver_, &numIterations);
           HYPRE_ParVectorCopy( b_csr, r_csr );
           HYPRE_ParCSRMatrixMatvec( -1.0, A_csr, x_csr, 1.0, r_csr );
           HYPRE_ParVectorInnerProd( r_csr, r_csr, &rnorm);
           rnorm = sqrt( rnorm );
           switch ( projectionScheme_ )
           {
              case 1 : addToAConjProjectionSpace(currX_,currB_);  break;
              case 2 : addToMinResProjectionSpace(currX_,currB_); break;
           }
           if ( numIterations >= maxIterations_ ) status = 1; else status = 0;
           break;

      //----------------------------------------------------------------
      // choose BiCGSTAB 
      //----------------------------------------------------------------

      case HYCGSTAB :
           if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0 )
           {
              printf("***************************************************\n");
              printf("* BiCGSTAB solver \n");
              printf("* maximum no. of iterations = %d\n", maxIterations_);
              printf("* convergence tolerance     = %e\n", tolerance_);
              printf("*--------------------------------------------------\n");
           }
           setupBiCGSTABPrecon();
           HYPRE_ParCSRBiCGSTABSetMaxIter(HYSolver_, maxIterations_);
           HYPRE_ParCSRBiCGSTABSetTol(HYSolver_, tolerance_);
           if ( normAbsRel_ == 0 ) HYPRE_ParCSRBiCGSTABSetStopCrit(HYSolver_,0);
           else                    HYPRE_ParCSRBiCGSTABSetStopCrit(HYSolver_,1);
           if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 )
           {
              HYPRE_ParCSRBiCGSTABSetPrintLevel(HYSolver_, 1);
              if ( mypid_ == 0 )
                printf("***************************************************\n");
              if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
                 HYPRE_ParCSRBiCGSTABSetPrintLevel(HYSolver_, 2);
           }
           retFlag = HYPRE_ParCSRBiCGSTABSetup(HYSolver_, A_csr, b_csr, x_csr);
           if ( retFlag != 0 )
           {
              printf("HYPRE_LSC::launchSolver ERROR : in BiCGSTAB setup.\n");
              return retFlag;
           }
           MPI_Barrier( comm_ );
           ptime  = LSC_Wtime();
           retFlag = HYPRE_ParCSRBiCGSTABSolve(HYSolver_, A_csr, b_csr, x_csr);
           if ( retFlag != 0 )
           {
              printf("HYPRE_LSC::launchSolver ERROR : in BiCGSTAB solve.\n");
              return retFlag;
           }
           HYPRE_ParCSRBiCGSTABGetNumIterations(HYSolver_, &numIterations);
           HYPRE_ParVectorCopy( b_csr, r_csr );
           HYPRE_ParCSRMatrixMatvec( -1.0, A_csr, x_csr, 1.0, r_csr );
           HYPRE_ParVectorInnerProd( r_csr, r_csr, &rnorm);
           rnorm = sqrt( rnorm );
           switch ( projectionScheme_ )
           {
              case 1 : addToAConjProjectionSpace(currX_,currB_);  break;
              case 2 : addToMinResProjectionSpace(currX_,currB_); break;
           }
           if ( numIterations >= maxIterations_ ) status = 1; else status = 0;
           break;

      //----------------------------------------------------------------
      // choose BiCGSTABL 
      //----------------------------------------------------------------

      case HYCGSTABL :
           if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0 )
           {
              printf("***************************************************\n");
              printf("* BiCGSTAB(2) solver \n");
              printf("* maximum no. of iterations = %d\n", maxIterations_);
              printf("* convergence tolerance     = %e\n", tolerance_);
              printf("*--------------------------------------------------\n");
           }
           setupBiCGSTABLPrecon();
           HYPRE_ParCSRBiCGSTABLSetMaxIter(HYSolver_, maxIterations_);
           HYPRE_ParCSRBiCGSTABLSetTol(HYSolver_, tolerance_);
           if (normAbsRel_ == 0) HYPRE_ParCSRBiCGSTABLSetStopCrit(HYSolver_,0);
           else                  HYPRE_ParCSRBiCGSTABLSetStopCrit(HYSolver_,1);
           if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 )
           {
              if ( mypid_ == 0 )
                printf("***************************************************\n");
              HYPRE_ParCSRBiCGSTABLSetLogging(HYSolver_, 1);
           }
           retFlag = HYPRE_ParCSRBiCGSTABLSetup(HYSolver_, A_csr, b_csr, x_csr);
           if ( retFlag != 0 )
           {
              printf("HYPRE_LSC::launchSolver ERROR : in BiCGSTABL setup.\n");
              return retFlag;
           }
           MPI_Barrier( comm_ );
           ptime  = LSC_Wtime();
           retFlag = HYPRE_ParCSRBiCGSTABLSolve(HYSolver_, A_csr, b_csr, x_csr);
           if ( retFlag != 0 )
           {
              printf("HYPRE_LSC::launchSolver ERROR : in BiCGSTABL solve.\n");
              return retFlag;
           }
           HYPRE_ParCSRBiCGSTABLGetNumIterations(HYSolver_, &numIterations);
           HYPRE_ParVectorCopy( b_csr, r_csr );
           HYPRE_ParCSRMatrixMatvec( -1.0, A_csr, x_csr, 1.0, r_csr );
           HYPRE_ParVectorInnerProd( r_csr, r_csr, &rnorm);
           rnorm = sqrt( rnorm );
           switch ( projectionScheme_ )
           {
              case 1 : addToAConjProjectionSpace(currX_,currB_);  break;
              case 2 : addToMinResProjectionSpace(currX_,currB_); break;
           }
           if ( numIterations >= maxIterations_ ) status = 1; else status = 0;
           break;

      //----------------------------------------------------------------
      // choose TFQMR 
      //----------------------------------------------------------------

      case HYTFQMR :
           if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0 )
           {
              printf("***************************************************\n");
              printf("* TFQMR solver \n");
              printf("* maximum no. of iterations = %d\n", maxIterations_);
              printf("* convergence tolerance     = %e\n", tolerance_);
              printf("*--------------------------------------------------\n");
           }
           setupTFQmrPrecon();
           HYPRE_ParCSRTFQmrSetMaxIter(HYSolver_, maxIterations_);
           HYPRE_ParCSRTFQmrSetTol(HYSolver_, tolerance_);
           if ( normAbsRel_ == 0 ) HYPRE_ParCSRTFQmrSetStopCrit(HYSolver_,0);
           else                    HYPRE_ParCSRTFQmrSetStopCrit(HYSolver_,1);
           if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 )
           {
              if ( mypid_ == 0 )
                printf("***************************************************\n");
              HYPRE_ParCSRTFQmrSetLogging(HYSolver_, 1);
           }
           retFlag = HYPRE_ParCSRTFQmrSetup(HYSolver_, A_csr, b_csr, x_csr);
           if ( retFlag != 0 )
           {
              printf("HYPRE_LSC::launchSolver ERROR : in TFQMR setup.\n");
              return retFlag;
           }
           MPI_Barrier( comm_ );
           ptime  = LSC_Wtime();
           retFlag = HYPRE_ParCSRTFQmrSolve(HYSolver_, A_csr, b_csr, x_csr);
           if ( retFlag != 0 )
           {
              printf("HYPRE_LSC::launchSolver ERROR : in TFQMR solve.\n");
              return retFlag;
           }
           HYPRE_ParCSRTFQmrGetNumIterations(HYSolver_, &numIterations);
           HYPRE_ParVectorCopy( b_csr, r_csr );
           HYPRE_ParCSRMatrixMatvec( -1.0, A_csr, x_csr, 1.0, r_csr );
           HYPRE_ParVectorInnerProd( r_csr, r_csr, &rnorm);
           rnorm = sqrt( rnorm );
           switch ( projectionScheme_ )
           {
              case 1 : addToAConjProjectionSpace(currX_,currB_);  break;
              case 2 : addToMinResProjectionSpace(currX_,currB_); break;
           }
           if ( numIterations >= maxIterations_ ) status = 1; else status = 0;
           break;

      //----------------------------------------------------------------
      // choose BiCGS 
      //----------------------------------------------------------------

      case HYBICGS :
           if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0 )
           {
              printf("***************************************************\n");
              printf("* BiCGS solver \n");
              printf("* maximum no. of iterations = %d\n", maxIterations_);
              printf("* convergence tolerance     = %e\n", tolerance_);
              printf("*--------------------------------------------------\n");
           }
           setupBiCGSPrecon();
           HYPRE_ParCSRBiCGSSetMaxIter(HYSolver_, maxIterations_);
           HYPRE_ParCSRBiCGSSetTol(HYSolver_, tolerance_);
           if ( normAbsRel_ == 0 ) HYPRE_ParCSRBiCGSSetStopCrit(HYSolver_,0);
           else                    HYPRE_ParCSRBiCGSSetStopCrit(HYSolver_,1);
           if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 )
           {
              if ( mypid_ == 0 )
                printf("***************************************************\n");
              HYPRE_ParCSRBiCGSSetLogging(HYSolver_, 1);
           }
           retFlag = HYPRE_ParCSRBiCGSSetup(HYSolver_, A_csr, b_csr, x_csr);
           if ( retFlag != 0 )
           {
              printf("HYPRE_LSC::launchSolver ERROR : in CGS setup.\n");
              return retFlag;
           }
           MPI_Barrier( comm_ );
           ptime  = LSC_Wtime();
           retFlag = HYPRE_ParCSRBiCGSSolve(HYSolver_, A_csr, b_csr, x_csr);
           if ( retFlag != 0 )
           {
              printf("HYPRE_LSC::launchSolver ERROR : in CGS solve.\n");
              return retFlag;
           }
           HYPRE_ParCSRBiCGSGetNumIterations(HYSolver_, &numIterations);
           HYPRE_ParVectorCopy( b_csr, r_csr );
           HYPRE_ParCSRMatrixMatvec( -1.0, A_csr, x_csr, 1.0, r_csr );
           HYPRE_ParVectorInnerProd( r_csr, r_csr, &rnorm);
           rnorm = sqrt( rnorm );
           switch ( projectionScheme_ )
           {
              case 1 : addToAConjProjectionSpace(currX_,currB_);  break;
              case 2 : addToMinResProjectionSpace(currX_,currB_); break;
           }
           if ( numIterations >= maxIterations_ ) status = 1; else status = 0;
           break;

      //----------------------------------------------------------------
      // choose Symmetric QMR 
      //----------------------------------------------------------------

      case HYSYMQMR :
           if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0 )
           {
              printf("***************************************************\n");
              printf("* SymQMR solver (for symmetric matrices and precond) \n");
              printf("* maximum no. of iterations = %d\n", maxIterations_);
              printf("* convergence tolerance     = %e\n", tolerance_);
              printf("*--------------------------------------------------\n");
           }
           setupSymQMRPrecon();
           HYPRE_ParCSRSymQMRSetMaxIter(HYSolver_, maxIterations_);
           HYPRE_ParCSRSymQMRSetTol(HYSolver_, tolerance_);
           if ( normAbsRel_ == 0 ) HYPRE_ParCSRSymQMRSetStopCrit(HYSolver_,0);
           else                    HYPRE_ParCSRSymQMRSetStopCrit(HYSolver_,1);
           if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 )
           {
              if ( mypid_ == 0 )
                printf("***************************************************\n");
              HYPRE_ParCSRSymQMRSetLogging(HYSolver_, 1);
           }
           retFlag = HYPRE_ParCSRSymQMRSetup(HYSolver_, A_csr, b_csr, x_csr);
           if ( retFlag != 0 )
           {
              printf("HYPRE_LSC::launchSolver ERROR : in SymQMR setup.\n");
              return retFlag;
           }
           MPI_Barrier( comm_ );
           ptime  = LSC_Wtime();
           retFlag = HYPRE_ParCSRSymQMRSolve(HYSolver_, A_csr, b_csr, x_csr);
           if ( retFlag != 0 )
           {
              printf("HYPRE_LSC::launchSolver ERROR : in SymQMR solve.\n");
              return retFlag;
           }
           HYPRE_ParCSRSymQMRGetNumIterations(HYSolver_, &numIterations);
           HYPRE_ParVectorCopy( b_csr, r_csr );
           HYPRE_ParCSRMatrixMatvec( -1.0, A_csr, x_csr, 1.0, r_csr );
           HYPRE_ParVectorInnerProd( r_csr, r_csr, &rnorm);
           rnorm = sqrt( rnorm );
           switch ( projectionScheme_ )
           {
              case 1 : addToAConjProjectionSpace(currX_,currB_);  break;
              case 2 : addToMinResProjectionSpace(currX_,currB_); break;
           }
           if ( numIterations >= maxIterations_ ) status = 1; else status = 0;
           break;

      //----------------------------------------------------------------
      // choose Boomeramg  
      //----------------------------------------------------------------

      case HYAMG :
           solveUsingBoomeramg(status);
           HYPRE_BoomerAMGGetNumIterations(HYSolver_, &numIterations);
           HYPRE_ParVectorCopy( b_csr, r_csr );
           HYPRE_ParCSRMatrixMatvec( -1.0, A_csr, x_csr, 1.0, r_csr );
           HYPRE_ParVectorInnerProd( r_csr, r_csr, &rnorm);
           rnorm = sqrt( rnorm );
           if ( numIterations >= maxIterations_ ) status = 1; else status = 0;
           ptime  = stime;
           //printf("Boomeramg solver - return status = %d\n",status);
           break;

      //----------------------------------------------------------------
      // choose SuperLU (single processor) 
      //----------------------------------------------------------------

      case HYSUPERLU :
           if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0 )
           {
              printf("***************************************************\n");
              printf("* SuperLU (sequential) solver \n");
              printf("*--------------------------------------------------\n");
           }
           rnorm = solveUsingSuperLU(status);
#ifndef NOFEI
           if ( status == 1 ) status = 0; 
#endif      
           numIterations = 1;
           ptime  = stime;
           //printf("SuperLU solver - return status = %d\n",status);
           break;

      //----------------------------------------------------------------
      // choose SuperLU (single processor) 
      //----------------------------------------------------------------

      case HYSUPERLUX :
           if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0 )
           {
              printf("***************************************************\n");
              printf("* SuperLU (sequential) solver with refinement \n");
              printf("*--------------------------------------------------\n");
           }
           rnorm = solveUsingSuperLUX(status);
#ifndef NOFEI
           if ( status == 1 ) status = 0; 
#endif      
           numIterations = 1;
           //printf("SuperLUX solver - return status = %d\n",status);
           break;

      //----------------------------------------------------------------
      // choose distributed SuperLU 
      //----------------------------------------------------------------

      case HYDSUPERLU :
#ifdef HAVE_DSUPERLU
           if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0 )
           {
              printf("***************************************************\n");
              printf("* distributed SuperLU solver \n");
              printf("*--------------------------------------------------\n");
           }
           rnorm = solveUsingDSuperLU(status);
#ifndef NOFEI
           if ( status == 1 ) status = 0; 
#endif      
           numIterations = 1;
#else
           printf("distributed SuperLU not available.\n");
           exit(1);
#endif
           break;

      //----------------------------------------------------------------
      // choose Y12M (single processor) 
      //----------------------------------------------------------------

      case HYY12M :
#ifdef Y12M
           if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0 )
           {
              printf("***************************************************\n");
              printf("* Y12M (sequential) solver\n");
              printf("*--------------------------------------------------\n");
           }
           solveUsingY12M(status);
#ifndef NOFEI
           if ( status == 1 ) status = 0; 
#endif      
           numIterations = 1;
           ptime  = stime;
           //printf("Y12M solver - return status = %d\n",status);
           break;
#else
           printf("HYPRE_LSC : Y12M not available. \n");
           exit(1);
           break; 
#endif

      //----------------------------------------------------------------
      // choose AMGE (single processor) 
      //----------------------------------------------------------------

      case HYAMGE :
#ifdef HAVE_AMGE
           if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0 )
           {
              printf("***************************************************\n");
              printf("* AMGe (sequential) solver\n");
              printf("*--------------------------------------------------\n");
           }
           solveUsingAMGe(numIterations);
           if ( numIterations >= maxIterations_ ) status = 1;
           ptime  = stime;
#else
           printf("AMGe not supported.\n");
#endif
           break;
   }

   //-------------------------------------------------------------------
   // recover solution for reduced system
   //-------------------------------------------------------------------

   if ( slideReduction_ == 1 )
   {
      newnorm = rnorm;
      rnorm   = buildSlideReducedSoln();
   }
   else if ( slideReduction_ == 2 )
   {
      newnorm = rnorm;
      rnorm   = buildSlideReducedSoln2();
   }
   else if ( slideReduction_ == 3 || slideReduction_ == 4 )
   {
      newnorm = rnorm;
      currA_ = TempA;
      currX_ = TempX;
      currB_ = TempB;
      currR_ = TempR;
      HYPRE_IJMatrixGetObject(currA_, (void **) &A_csr);
      HYPRE_IJVectorGetObject(currX_, (void **) &x_csr);
      HYPRE_IJVectorGetObject(currB_, (void **) &b_csr);
      HYPRE_IJVectorGetObject(currR_, (void **) &r_csr);
      if ( slideReduction_ == 3 )
           slideObj->buildReducedSolnVector(currX_, currB_);
      else slideObj->buildModifiedSolnVector(currX_);
      HYPRE_ParVectorCopy( b_csr, r_csr );
      HYPRE_ParCSRMatrixMatvec( -1.0, A_csr, x_csr, 1.0, r_csr );
      HYPRE_ParVectorInnerProd( r_csr, r_csr, &rnorm);
      rnorm = sqrt( rnorm );
   }
   else if ( schurReduction_ == 1 )
   {
      newnorm = rnorm;
      rnorm   = buildSchurReducedSoln();
   }
   if ( (normalEqnFlag_ & 7) == 7 )
   {
      HYPRE_IJMatrixGetObject(currA_, (void **) &A_csr);
      HYPRE_IJVectorGetObject(currX_,  (void **) &x_csr);
      HYPRE_IJVectorGetObject(currB_, (void **) &b_csr);
      HYPRE_IJVectorGetObject(currR_,  (void **) &r_csr);
      HYPRE_ParVectorCopy( b_csr, r_csr );
      HYPRE_ParCSRMatrixMatvec( -1.0, A_csr, x_csr, 1.0, r_csr );
      HYPRE_ParVectorInnerProd( r_csr, r_csr, &rnorm);
      rnorm = sqrt( rnorm );
   }

   //-------------------------------------------------------------------
   // register solver return information and print timing information
   //-------------------------------------------------------------------

   solveStatus = status;
   iterations = numIterations;
   rnorm_ = rnorm;

   MPI_Barrier(comm_);
   etime = LSC_Wtime();
   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0 )
   {
      printf("***************************************************\n");
      printf("*                Solver Statistics                *\n");
      printf("*-------------------------------------------------*\n");
      if ( slideReduction_ || schurReduction_ )
         printf("** HYPRE matrix reduction time     = %e\n",rtime2-rtime1);
      printf("** HYPRE preconditioner setup time = %e\n", ptime - stime);
      printf("** HYPRE solution time             = %e\n", etime - ptime);
      printf("** HYPRE total time                = %e\n", etime - stime);
      printf("** HYPRE number of iterations      = %d\n", numIterations);
      if ( slideReduction_ || schurReduction_ )
         printf("** HYPRE reduced residual norm     = %e\n", newnorm);
      printf("** HYPRE final residual norm       = %e\n", rnorm);
      printf("***************************************************\n");
   }

   //-------------------------------------------------------------------
   // write solution to an output file
   //-------------------------------------------------------------------

   if ( HYOutputLevel_ & HYFEI_PRINTSOL )
   {
      nrows = localEndRow_ - localStartRow_ + 1;
      startRow = localStartRow_ - 1;
      sprintf(fname, "hypre_sol.out.%d", mypid_);
      fp = fopen( fname, "w");
      fprintf(fp, "%6d \n", nrows);
      for ( i = startRow; i < startRow+nrows; i++ )
      {
         HYPRE_IJVectorGetValues(currX_, 1, &i, &ddata);
         fprintf(fp, "%6d  %25.16e \n", i+1, ddata);
      }
      fclose(fp);
      MPI_Barrier(comm_);
   }

   //-------------------------------------------------------------------
   // diagnostic message 
   //-------------------------------------------------------------------

   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
      printf("%4d : HYPRE_LSC::leaving  launchSolver.\n", mypid_);
   return (0);
}

//***************************************************************************
// this function extracts the matrix in a CSR format
//---------------------------------------------------------------------------

int HYPRE_LinSysCore::writeSystem(const char *name)
{
   (void) name;
   printf("HYPRE_LinsysCore : writeSystem not implemented.\n");
   return (0);
}

//***************************************************************************
// this function computes matrix vector product
//---------------------------------------------------------------------------

int HYPRE_LinSysCore::HYPRE_LSC_Matvec(void *x, void *y)
{
   HYPRE_ParCSRMatrix A_csr;
   HYPRE_ParVector    x_csr = (HYPRE_ParVector)    x;
   HYPRE_ParVector    y_csr = (HYPRE_ParVector)    y;
   HYPRE_IJMatrixGetObject(currA_, (void **) &A_csr);
   HYPRE_ParCSRMatrixMatvec(1.0, A_csr, x_csr, 0.0, y_csr);
   return (0);
}

//***************************************************************************
// this function computes vector multiply and add
//---------------------------------------------------------------------------

int HYPRE_LinSysCore::HYPRE_LSC_Axpby(double a, void *x, double b, void *y)
{
   HYPRE_ParVector x_csr = (HYPRE_ParVector) x;
   HYPRE_ParVector y_csr = (HYPRE_ParVector) y;
   if ( b != 1.0 ) HYPRE_ParVectorScale( b, y_csr);
   hypre_ParVectorAxpy(a, (hypre_ParVector*) x_csr, (hypre_ParVector*) y_csr);
   return (0);
}

//***************************************************************************
// this function fetches the right hand side vector
//---------------------------------------------------------------------------

void *HYPRE_LinSysCore::HYPRE_LSC_GetRHSVector()
{
   HYPRE_ParVector b_csr;
   HYPRE_IJVectorGetObject(HYb_, (void **) &b_csr);
   return (void *) b_csr;
}

//***************************************************************************
// this function fetches the solution vector
//---------------------------------------------------------------------------

void *HYPRE_LinSysCore::HYPRE_LSC_GetSolVector()
{
   HYPRE_ParVector x_csr;
   HYPRE_IJVectorGetObject(HYx_, (void **) &x_csr);
   return (void *) x_csr;
}

//***************************************************************************
// this function fetches the matrix 
//---------------------------------------------------------------------------

void *HYPRE_LinSysCore::HYPRE_LSC_GetMatrix()
{
   HYPRE_ParCSRMatrix A_csr;
   HYPRE_IJMatrixGetObject(currA_, (void **) &A_csr);
   return (void *) A_csr;
}

//***************************************************************************
// this function set column ranges
//---------------------------------------------------------------------------

void *HYPRE_LinSysCore::HYPRE_LSC_SetColMap(int start, int end)
{
   localStartCol_ = start;
   localEndCol_ = end;
   return (void *) NULL;
}

//***************************************************************************
// this function set column ranges
//---------------------------------------------------------------------------

void *HYPRE_LinSysCore::HYPRE_LSC_MatMatMult(void *inMat)
{
   HYPRE_ParCSRMatrix A_csr;
   hypre_ParCSRMatrix *B_csr, *C_csr;
   HYPRE_IJMatrixGetObject(currA_, (void **) &A_csr);
   B_csr = (hypre_ParCSRMatrix *) inMat;
   C_csr = hypre_ParMatmul((hypre_ParCSRMatrix *)A_csr,B_csr);
   return (void *) C_csr;
}

//***************************************************************************
// this function returns the residual norm
//---------------------------------------------------------------------------

double HYPRE_LinSysCore::HYPRE_LSC_GetRNorm()
{
   return rnorm_;
}

