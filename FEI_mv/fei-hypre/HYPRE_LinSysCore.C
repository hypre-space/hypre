/*BHEADER**********************************************************************
 * (c) 2001   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

#include "fei_defs.h"
#include "Data.h"
#include "basicTypes.h"
#include "utilities/utilities.h"

#ifndef NOFEI
#include "Lookup.h"
#include "LinearSystemCore.h"
#endif

#include "HYPRE.h"
#include "HYPRE_config.h"
#include "../../IJ_mv/HYPRE_IJ_mv.h"
#include "../../parcsr_mv/HYPRE_parcsr_mv.h"
#include "../../parcsr_ls/HYPRE_parcsr_ls.h"
#include "HYPRE_parcsr_bicgstabl.h"
#include "HYPRE_parcsr_TFQmr.h"
#include "HYPRE_parcsr_bicgs.h"
#include "HYPRE_parcsr_symqmr.h"
#include "HYPRE_LinSysCore.h"
#include "fegridinfo.h"

//---------------------------------------------------------------------------
// parcsr_mv.h is put here instead of in HYPRE_LinSysCore.h 
// because it gives warning when compiling cfei.cc
//---------------------------------------------------------------------------

#include "parcsr_mv/parcsr_mv.h"
#include "HYPRE_LSI_schwarz.h"
#include "HYPRE_LSI_ddilut.h"
#include "HYPRE_LSI_ddict.h"
#include "HYPRE_LSI_poly.h"
#include "HYPRE_LSI_block.h"

#ifdef SUPERLU
#include "dsp_defs.h"
#include "util.h"
#endif

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

//---------------------------------------------------------------------------
// These are external functions needed internally here
//---------------------------------------------------------------------------

extern "C" {

#ifdef MLPACK
   int HYPRE_ParCSRMLCreate( MPI_Comm, HYPRE_Solver *);
   int HYPRE_ParCSRMLDestroy( HYPRE_Solver );
   int HYPRE_ParCSRMLSetup( HYPRE_Solver, HYPRE_ParCSRMatrix,
                            HYPRE_ParVector, HYPRE_ParVector );
   int HYPRE_ParCSRMLSolve( HYPRE_Solver, HYPRE_ParCSRMatrix,
                            HYPRE_ParVector, HYPRE_ParVector );
   int HYPRE_ParCSRMLSetStrongThreshold( HYPRE_Solver, double );
   int HYPRE_ParCSRMLSetNumPreSmoothings( HYPRE_Solver, int );
   int HYPRE_ParCSRMLSetNumPostSmoothings( HYPRE_Solver, int );
   int HYPRE_ParCSRMLSetPreSmoother( HYPRE_Solver, int );
   int HYPRE_ParCSRMLSetPostSmoother( HYPRE_Solver, int );
   int HYPRE_ParCSRMLSetDampingFactor( HYPRE_Solver, double );
   int HYPRE_ParCSRMLSetMethod( HYPRE_Solver, int );
   int HYPRE_ParCSRMLSetCoarsenScheme( HYPRE_Solver , int );
   int HYPRE_ParCSRMLSetCoarseSolver( HYPRE_Solver, int );
#endif

   void qsort0(int *, int, int);
   void qsort1(int *, double *, int, int);

#ifdef HAVE_AMGE
    int HYPRE_LSI_AMGeCreate();
    int HYPRE_LSI_AMGeDestroy();
    int HYPRE_LSI_AMGeSetNNodes(int);
    int HYPRE_LSI_AMGeSetNElements(int);
    int HYPRE_LSI_AMGeSetSystemSize(int);
    int HYPRE_LSI_AMGePutRow(int,int,double*,int*);
    int HYPRE_LSI_AMGeSolve( double *rhs, double *sol ); 
    int HYPRE_LSI_AMGeSetBoundary( int leng, int *colInd );
    int HYPRE_LSI_AMGeWriteToFile();
#endif
}

//***************************************************************************
// constructor
//---------------------------------------------------------------------------

HYPRE_LinSysCore::HYPRE_LinSysCore(MPI_Comm comm) : 
                  comm_(comm),
                  HYA_(NULL),
                  HYA21_(NULL),
                  HYA12_(NULL),
                  HYinvA22_(NULL),
                  HYb_(NULL),
                  HYbs_(NULL),
                  HYx_(NULL),
                  HYr_(NULL),
                  currA_(NULL),
                  currB_(NULL),
                  currX_(NULL),
                  currR_(NULL),
                  reducedA_(NULL),
                  reducedB_(NULL),
                  reducedX_(NULL),
                  reducedR_(NULL),
                  matrixVectorsCreated_(0),
                  numRHSs_(1),
                  currentRHS_(0),
                  HYSolver_(NULL), 
                  HYPrecon_(NULL), 
                  HYPreconReuse_(0), 
                  numGlobalRows_(0),
                  localStartRow_(0),
                  localEndRow_(-1),
                  nConstraints_(0),
                  constrList_(NULL),
                  maxIterations_(1000),
                  tolerance_(1.0e-6),
                  normAbsRel_(0),
                  systemAssembled_(0),
                  HYPreconSetup_(0),
                  slideReduction_(0),
                  schurReduction_(0),
                  schurReductionCreated_(0),
                  A21NRows_(0),
                  A21NCols_(0),
                  finalResNorm_(0.0),
                  rowLengths_(NULL),
                  colIndices_(NULL),
                  colValues_(NULL),
                  selectedList_(NULL),
                  selectedListAux_(NULL),
                  HYOutputLevel_(0),
                  lookup_(NULL),
                  haveLookup_(0),
                  projectionScheme_(0),
                  projectSize_(0),
                  projectCurrSize_(0),
                  HYpxs_(NULL),
                  projectionMatrix_(NULL),
                  mapFromSolnFlag_(0),
                  mapFromSolnLeng_(0),
                  mapFromSolnLengMax_(0),
                  mapFromSolnList_(NULL),
                  mapFromSolnList2_(NULL),
                  HYpbs_(NULL)
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
    amgStrongThreshold_ = 0.25;
    for (int i = 0; i < 25; i++) amgRelaxWeight_[i] = 0.0; 

    pilutFillin_        = 0;    // how many nonzeros to keep in L and U
    pilutDropTol_       = 0.0;
    pilutMaxNnzPerRow_  = 0;    // register the max NNZ/row in matrix A

    ddilutFillin_       = 1.0;  // additional fillin other than A
    ddilutDropTol_      = 1.0e-8;

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
    for (int j = 0; j < euclidargc_*2; j++) euclidargv_[j] = new char[50];
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

    rhsIDs_             = new int[1];
    rhsIDs_[0]          = 0;

    FEGridInfo *gridinfo = new FEGridInfo(mypid_);
    fegrid               = (void *) gridinfo;
}

//***************************************************************************
// destructor
//---------------------------------------------------------------------------

HYPRE_LinSysCore::~HYPRE_LinSysCore() 
{
    int  i;

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
    {
       printf("%4d : HYPRE_LSC::entering destructor.\n",mypid_);
    }

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
       for ( int i = 0; i < localEndRow_-localStartRow_+1; i++ )
          if ( colIndices_[i] != NULL ) delete [] colIndices_[i];
       delete [] colIndices_;
       colIndices_ = NULL;
    }
    if ( colValues_ != NULL )
    {
       for ( int j = 0; j < localEndRow_-localStartRow_+1; j++ )
          if ( colValues_[j] != NULL ) delete [] colValues_[j];
       delete [] colValues_;
       colValues_ = NULL;
    }
    if ( rowLengths_ != NULL ) 
    {
       delete [] rowLengths_;
       rowLengths_ = NULL;
    }
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

#ifdef MLPACK
       else if ( HYPreconID_ == HYML )
          HYPRE_ParCSRMLDestroy( HYPrecon_ );
#endif
       HYPrecon_ = NULL;
    }
    delete [] HYPreconName_;
    HYPreconName_ = NULL;

    if ( rhsIDs_ != NULL ) delete [] rhsIDs_;

    //-------------------------------------------------------------------
    // deallocate the local store for the constraint indices
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
    for (i = 0; i < euclidargc_*2; i++) delete [] euclidargv_[i];
    delete [] euclidargv_;
    euclidargv_ = NULL;
    
    FEGridInfo *gridinfo = (FEGridInfo *) fegrid;
    delete gridinfo;
    fegrid = NULL;
    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
    {
       printf("%4d : HYPRE_LSC::leaving  destructor.\n",mypid_);
    }
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
//---------------------------------------------------------------------------

int HYPRE_LinSysCore::setLookup(Lookup& lookup)
{
   if (&lookup == NULL) return (0);

   lookup_ = &lookup;
   haveLookup_ = 1;

   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 5 )
   {
      //let's test the shared-node lookup functions.

#ifndef NOFEI
      int numSharedNodes = lookup_->getNumSharedNodes();
      printf("setLookup: numSharedNodes: %d\n", numSharedNodes);

      const int* shNodeNums = lookup_->getSharedNodeNumbers();

      for(int i=0; i<numSharedNodes; i++) {
         printf("    shNodeNums[%d]: %d\n", i, shNodeNums[i]);

         const int* shNodeProcs =
               lookup_->getSharedNodeProcs(shNodeNums[i]);
         int numSharingProcs = lookup_->getNumSharingProcs(shNodeNums[i]);

         if (shNodeProcs == NULL) {
            printf("ERROR, couldn't get sharing procs\n");
            continue;
         }

         printf("    sharing procs: ");
         for(int j=0; j<numSharingProcs; j++) printf("%d ", shNodeProcs[j]);
         printf("\n");
      }
      printf("\n");
      fflush(stdout);
#endif
   }
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

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
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

    //-------------------------------------------------------------------
    // error checking
    //-------------------------------------------------------------------

    if ( ( firstLocalEqn <= 0 ) || 
         ( firstLocalEqn+numLocalEqns-1) > numGlobalEqns)
    {
       printf("%4d : createMatricesVectors: invalid local equation nos.\n");
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
    // instantiate the matrix
    //-------------------------------------------------------------------
    
    //--old_IJ-----------------------------------------------------------
    // ierr = HYPRE_IJMatrixCreate(comm_,&HYA_,numGlobalRows_,numGlobalRows_);
    // ierr = HYPRE_IJMatrixSetLocalStorageType(HYA_, HYPRE_PARCSR);
    // ierr = HYPRE_IJMatrixSetLocalSize(HYA_, numLocalEqns, numLocalEqns);
    //--new_IJ-----------------------------------------------------------
    ierr = HYPRE_IJMatrixCreate(comm_, localStartRow_-1,localEndRow_-1,
				localStartRow_-1,localEndRow_-1, &HYA_);
    ierr = HYPRE_IJMatrixSetObjectType(HYA_, HYPRE_PARCSR);
    //-------------------------------------------------------------------
    assert(!ierr);

    //-------------------------------------------------------------------
    // instantiate the right hand vectors
    //-------------------------------------------------------------------

    HYbs_ = new HYPRE_IJVector[numRHSs_];
    for ( i = 0; i < numRHSs_; i++ )
    {
       //--old_IJ--------------------------------------------------------
       // ierr = HYPRE_IJVectorCreate(comm_, &(HYbs_[i]), numGlobalRows_);
       // ierr = HYPRE_IJVectorSetLocalStorageType(HYbs_[i], HYPRE_PARCSR);
       // ierr = HYPRE_IJVectorSetLocalPartitioning(HYbs_[i],localStartRow_-1,
       //                                           localEndRow_);
       // ierr = HYPRE_IJVectorAssemble(HYbs_[i]);
       // ierr = HYPRE_IJVectorInitialize(HYbs_[i]);
       // ierr = HYPRE_IJVectorZeroLocalComponents(HYbs_[i]);
       //--new_IJ--------------------------------------------------------
       ierr = HYPRE_IJVectorCreate(comm_, localStartRow_-1, localEndRow_-1,
				&(HYbs_[i]));
       ierr = HYPRE_IJVectorSetObjectType(HYbs_[i], HYPRE_PARCSR);
       ierr = HYPRE_IJVectorInitialize(HYbs_[i]);
       ierr = HYPRE_IJVectorAssemble(HYbs_[i]);
       //----------------------------------------------------------------
       assert(!ierr);
    }
    HYb_ = HYbs_[0];

    //-------------------------------------------------------------------
    // instantiate the solution vector
    //-------------------------------------------------------------------

    //--old_IJ-----------------------------------------------------------
    // ierr = HYPRE_IJVectorCreate(comm_, &HYx_, numGlobalRows_);
    // ierr = HYPRE_IJVectorSetLocalStorageType(HYx_, HYPRE_PARCSR);
    // ierr = HYPRE_IJVectorSetLocalPartitioning(HYx_,localStartRow_-1,
    //                                          localEndRow_);
    // ierr = HYPRE_IJVectorAssemble(HYx_);
    // ierr = HYPRE_IJVectorInitialize(HYx_);
    // ierr = HYPRE_IJVectorZeroLocalComponents(HYx_);
    //--new_IJ-----------------------------------------------------------
    ierr = HYPRE_IJVectorCreate(comm_, localStartRow_-1, localEndRow_-1, &HYx_);
    ierr = HYPRE_IJVectorSetObjectType(HYx_, HYPRE_PARCSR);
    ierr = HYPRE_IJVectorInitialize(HYx_);
    ierr = HYPRE_IJVectorAssemble(HYx_);
    //-------------------------------------------------------------------
    assert(!ierr);

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

    //--old_IJ-----------------------------------------------------------
    // ierr = HYPRE_IJVectorCreate(comm_, &HYr_, numGlobalRows_);
    // ierr = HYPRE_IJVectorSetLocalStorageType(HYr_, HYPRE_PARCSR);
    // ierr = HYPRE_IJVectorSetLocalPartitioning(HYr_,localStartRow_-1,
    //                                           localEndRow_);
    // ierr = HYPRE_IJVectorAssemble(HYr_);
    // ierr = HYPRE_IJVectorInitialize(HYr_);
    // ierr = HYPRE_IJVectorZeroLocalComponents(HYr_);
    //--new_IJ-----------------------------------------------------------
    ierr = HYPRE_IJVectorCreate(comm_, localStartRow_-1, localEndRow_-1, &HYr_);
    ierr = HYPRE_IJVectorSetObjectType(HYr_, HYPRE_PARCSR);
    ierr = HYPRE_IJVectorInitialize(HYr_);
    ierr = HYPRE_IJVectorAssemble(HYr_);
    //-------------------------------------------------------------------
    assert(!ierr);
    matrixVectorsCreated_ = 1;
    schurReductionCreated_ = 0;
    systemAssembled_ = 0;

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
    {
       printf("%4d : HYPRE_LSC::leaving  createMatricesAndVectors.\n",mypid_);
    }
    return (0);
}

//***************************************************************************
// similar to createMatrixVectors (FEI 1.5 compatible)
//---------------------------------------------------------------------------

int HYPRE_LinSysCore::setGlobalOffsets(int leng, int* nodeOffsets,
                       int* eqnOffsets, int* blkEqnOffsets)
{
    //-------------------------------------------------------------------
    // diagnostic message
    //-------------------------------------------------------------------

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
    {
       printf("%4d : HYPRE_LSC::entering setGlobalOffsets.\n",mypid_);
    }

    //-------------------------------------------------------------------
    // set local range (incoming 0-based, locally 1-based)
    //-------------------------------------------------------------------

    int firstLocalEqn = eqnOffsets[mypid_] + 1;
    int numLocalEqns  = eqnOffsets[mypid_+1] - firstLocalEqn + 1;
    int numGlobalEqns = eqnOffsets[numProcs_];
    createMatricesAndVectors(numGlobalEqns, firstLocalEqn, numLocalEqns); 

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
    {
       printf("%4d : HYPRE_LSC::startrow, endrow = %d %d\n",mypid_,
                     localStartRow_, localEndRow_);
    }
    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
    {
       printf("%4d : HYPRE_LSC::leaving  setGlobalOffsets.\n",mypid_);
    }
    return (0);
}

//***************************************************************************
// new functions in FEI 1.5 
//---------------------------------------------------------------------------

int HYPRE_LinSysCore::setConnectivities(GlobalID elemBlock, int numElements,
                       int numNodesPerElem, const GlobalID* elemIDs,
                       const int* const* connNodes)
{
    (void) elemBlock;
    (void) numElements;
    (void) numNodesPerElem;
    (void) elemIDs;
    (void) connNodes;

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) > 2 )
       printf("%4d : HYPRE_LSC::setConnectivities not implemented.\n",mypid_);
    return (0);
}

//---------------------------------------------------------------------------

int HYPRE_LinSysCore::setStiffnessMatrices(GlobalID elemBlock, int numElems,
                     const GlobalID* elemIDs,const double *const *const *stiff,
                     int numEqnsPerElem, const int *const * eqnIndices)
{
    (void) elemBlock;
    (void) numElems;
    (void) elemIDs;
    (void) stiff;
    (void) numEqnsPerElem;
    (void) eqnIndices;

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) > 2 )
       printf("%4d : HYPRE_LSC::setStiffnessMatrices not implemented.\n",
              mypid_);
    return (0);
}

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

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) > 2 )
       printf("%4d : HYPRE_LSC::setLoadVectors not implemented.\n", mypid_);
    return (0);
}

//***************************************************************************
// Set the number of rows in the diagonal part and off diagonal part
// of the matrix, using the structure of the matrix, stored in rows.
// rows is an array that is 0-based.  localStartRow and localEndRow are 1-based.
//---------------------------------------------------------------------------

int HYPRE_LinSysCore::allocateMatrix(int **colIndices, int *rowLengths)
{
    int i, j, ierr, nsize, *indices, maxSize, minSize;

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
    {
       printf("%4d : HYPRE_LSC::entering allocateMatrix.\n", mypid_);
    }

    //-------------------------------------------------------------------
    // error checking
    //-------------------------------------------------------------------

    if ( localEndRow_ < localStartRow_ ) 
    {
       printf("allocateMatrix ERROR : createMatrixAndVectors should be\n");
       printf("                       called before allocateMatrix.\n");
       exit(1);
    }

    nsize       = localEndRow_ - localStartRow_ + 1;
    rowLengths_ = new int[nsize];
    colIndices_ = new int*[nsize];
    colValues_  = new double*[nsize];

    //-------------------------------------------------------------------
    // store the column index information
    //-------------------------------------------------------------------

    maxSize = 0;
    minSize = 1000000;
    for ( i = 0; i < nsize; i++ )
    {
       rowLengths_[i] = rowLengths[i];
       if ( rowLengths[i] > 0 ) colIndices_[i] = new int[rowLengths[i]];
       else                     colIndices_[i] = NULL;
       for ( j = 0; j < rowLengths[i]; j++ )
       {
          colIndices_[i][j] = colIndices[i][j];
       }
       qsort0( colIndices_[i], 0, rowLengths[i]-1);
       maxSize = ( rowLengths[i] > maxSize ) ? rowLengths[i] : maxSize;
       minSize = ( rowLengths[i] < minSize ) ? rowLengths[i] : minSize;
       if ( rowLengths[i] > 0 ) colValues_[i] = new double[rowLengths[i]];
       for ( j = 0; j < rowLengths[i]; j++ ) colValues_[i][j] = 0.0;
    }

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
    {
       printf("%4d : allocateMatrix : max/min nnz/row = %d %d\n", mypid_, 
                     maxSize, minSize);
    }

    MPI_Allreduce(&maxSize, &pilutMaxNnzPerRow_,1,MPI_INT,MPI_MAX,comm_);

    //ierr = HYPRE_IJMatrixSetRowSizes(HYA_, rowLengths_);
    //ierr = HYPRE_IJMatrixInitialize(HYA_);
    //assert(!ierr);

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
    {
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

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
    {
       printf("%4d : HYPRE_LSC::entering setMatrixStructure.\n",mypid_);
    }

    (void) blkColIndices;
    (void) blkRowLengths;
    (void) ptRowsPerBlkRow;

    int numLocalRows = localEndRow_ - localStartRow_ + 1;
    for ( i = 0; i < numLocalRows; i++ )
    {
       for ( int j = 0; j < ptRowLengths[i]; j++ ) ptColIndices[i][j]++;
    }
    allocateMatrix(ptColIndices, ptRowLengths);
    for ( i = 0; i < numLocalRows; i++ )
    {
       for ( j = 0; j < ptRowLengths[i]; j++ ) ptColIndices[i][j]--;
    }

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
    {
       printf("%4d : HYPRE_LSC::leaving  setMatrixStructure.\n",mypid_);
    }
    return (0);
}

//***************************************************************************
// set Lagrange multiplier and penalty constraint equations
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

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) > 2 )
       printf("%4d : HYPRE_LSC::setMultCREqns not implemented.\n",mypid_);
    return (0);
}

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

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) > 2 )
       printf("%4d : HYPRE_LSC::setPenCREqns not implemented.\n",mypid_);
    return (0);
}

//***************************************************************************
// This function is needed in order to construct a new problem with the
// same sparsity pattern.
//---------------------------------------------------------------------------

int HYPRE_LinSysCore::resetMatrixAndVector(double s)
{
    int    i, j, ierr, size, *indices0;
    double *values0;

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
    {
       printf("%4d : HYPRE_LSC::entering resetMatrixAndVector.\n",mypid_);
    }

    if ( s != 0.0 && mypid_ == 0 )
    {
       printf("resetMatrixAndVector ERROR : cannot take nonzeros.\n");
       exit(1);
    }

    //--old_IJ-----------------------------------------------------------
    // for (i = 0; i < numRHSs_; i++) 
    //	  HYPRE_IJVectorZeroLocalComponents(HYbs_[i]);
    //--new_IJ-----------------------------------------------------------
    size     = localEndRow_ - localStartRow_ + 1;
    indices0 = new int[size];
    values0  = new double[size];
    for (i = 0; i < size; i++)
    {
       indices0[i] = localStartRow_ + i - 1;
       values0[i] = 0.0;
    }    
    for (i = 0; i < numRHSs_; i++) 
       HYPRE_IJVectorSetValues(HYbs_[i], size, (const int *) indices0,
				(const double *) values0);

    delete [] indices0;
    delete [] values0;
    //-------------------------------------------------------------------

    systemAssembled_ = 0;
    schurReductionCreated_ = 0;
    projectCurrSize_ = 0;

    //-------------------------------------------------------------------
    // for now, since HYPRE does not yet support
    // re-initializing the matrix, restart the whole thing
    //-------------------------------------------------------------------

    //--old_IJ-----------------------------------------------------------
    // if ( HYA_ != NULL ) HYPRE_IJMatrixDestroy(HYA_);
    // ierr = HYPRE_IJMatrixCreate(comm_,&HYA_,numGlobalRows_,numGlobalRows_);
    // ierr = HYPRE_IJMatrixSetLocalStorageType(HYA_, HYPRE_PARCSR);
    // size = localEndRow_ - localStartRow_ + 1;
    // ierr = HYPRE_IJMatrixSetLocalSize(HYA_, size, size);
    //--new_IJ-----------------------------------------------------------
    if ( HYA_ != NULL ) HYPRE_IJMatrixDestroy(HYA_);
    ierr = HYPRE_IJMatrixCreate(comm_, localStartRow_-1, localEndRow_-1,
				localStartRow_-1, localEndRow_-1, &HYA_);
    ierr = HYPRE_IJMatrixSetObjectType(HYA_, HYPRE_PARCSR);
    //-------------------------------------------------------------------
    assert(!ierr);

    //-------------------------------------------------------------------
    // clean the reducetion stuff
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

    colValues_ = new double*[size];
    for ( i = 0; i < size; i++ )
    {
       if ( rowLengths_[i] > 0 ) colValues_[i] = new double[rowLengths_[i]];
       for ( j = 0; j < rowLengths_[i]; j++ ) colValues_[i][j] = 0.0;
    }

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
    {
       printf("%4d : HYPRE_LSC::leaving  resetMatrixAndVector.\n", mypid_);
    }
    return (0);
}

//***************************************************************************
// new function to reset matrix independently
//---------------------------------------------------------------------------

int HYPRE_LinSysCore::resetMatrix(double s) 
{
    int  i, j, ierr, size;

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
    {
       printf("%4d : HYPRE_LSC::entering resetMatrix.\n",mypid_);
    }

    if ( s != 0.0 && mypid_ == 0 )
    {
       printf("resetMatrix ERROR : cannot take nonzeros.\n");
       exit(1);
    }

    systemAssembled_ = 0;
    schurReductionCreated_ = 0;
    projectCurrSize_ = 0;

    //-------------------------------------------------------------------
    // clean the reducetion stuff
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

    //--old_IJ-----------------------------------------------------------
    // if ( HYA_ != NULL ) HYPRE_IJMatrixDestroy(HYA_);
    // ierr = HYPRE_IJMatrixCreate(comm_,&HYA_,numGlobalRows_,numGlobalRows_);
    // ierr = HYPRE_IJMatrixSetLocalStorageType(HYA_, HYPRE_PARCSR);
    // size = localEndRow_ - localStartRow_ + 1;
    // ierr = HYPRE_IJMatrixSetLocalSize(HYA_, size, size);
    //--new_IJ-----------------------------------------------------------
    if ( HYA_ != NULL ) HYPRE_IJMatrixDestroy(HYA_);
    size = localEndRow_ - localStartRow_ + 1;
    ierr = HYPRE_IJMatrixCreate(comm_, localStartRow_-1, localEndRow_-1,
				localStartRow_-1, localEndRow_-1, &HYA_);
    ierr = HYPRE_IJMatrixSetObjectType(HYA_, HYPRE_PARCSR);
    //-------------------------------------------------------------------
    assert(!ierr);

    //-------------------------------------------------------------------
    // allocate space for storing the matrix coefficient
    //-------------------------------------------------------------------

    colValues_ = new double*[size];
    for ( i = 0; i < size; i++ )
    {
       if ( rowLengths_[i] > 0 ) colValues_[i] = new double[rowLengths_[i]];
       for ( j = 0; j < rowLengths_[i]; j++ ) colValues_[i][j] = 0.0;
    }

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
    {
       printf("%4d : HYPRE_LSC::leaving  resetMatrix.\n", mypid_);
    }
    return (0);
}

//***************************************************************************
// new function to reset vectors independently
//---------------------------------------------------------------------------

int HYPRE_LinSysCore::resetRHSVector(double s) 
{
    int    i, j, ierr, size, *indices0;
    double *values0;

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
    {
       printf("%4d : HYPRE_LSC::entering resetRHSVector.\n",mypid_);
    }

    if ( s != 0.0 && mypid_ == 0 )
    {
       printf("resetRHSVector ERROR : cannot take nonzeros.\n");
       exit(1);
    }

    if ( HYbs_ != NULL )
    {
       //--old_IJ--------------------------------------------------------
       // for (i = 0; i < numRHSs_; i++) 
       // if ( HYbs_[i] != NULL ) HYPRE_IJVectorZeroLocalComponents(HYbs_[i]);
       //--new_IJ--------------------------------------------------------
       size     = localEndRow_ - localStartRow_ + 1;
       indices0 = new int[size];
       values0  = new double[size];
       for (i = 0; i < size; i++)
       {
          indices0[i] = localStartRow_ + i - 1;
          values0[i] = 0.0;
       }    
       for (i = 0; i < numRHSs_; i++) 
          if ( HYbs_[i] != NULL ) 
             HYPRE_IJVectorSetValues(HYbs_[i], size, (const int *) indices0,
 				(const double *) values0);
       delete [] indices0;
       delete [] values0;
       //----------------------------------------------------------------
    }

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
    {
       printf("%4d : HYPRE_LSC::leaving  resetRHSVector.\n",mypid_);
    }
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

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
    {
       printf("%4d : HYPRE_LSC::entering sumIntoSystemMatrix.\n",mypid_);
       printf("%4d : row number = %d.\n", mypid_, row);
       if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 4 )
       {
          for ( i = 0; i < numValues; i++ )
             printf("  %4d : col = %d, data = %e\n", mypid_, scatterIndices[i], 
                     values[i]);
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
    if ( row < localStartRow_ || row > localEndRow_ )
    {
       printf("sumIntoSystemMatrix ERROR : invalid row number %d.\n",row);
       exit(1);
    }
    localRow = row - localStartRow_;
    if ( numValues > rowLengths_[localRow] )
    {
       printf("sumIntoSystemMatrix ERROR : row size too large.\n");
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
          printf("%4d : sumIntoSystemMatrix ERROR - loading column");
          printf("      that has not been declared before - %d.\n",colIndex);
          for ( j = 0; j < rowLengths_[localRow]; j++ ) 
             printf("       available column index = %d\n",
                    colIndices_[localRow][j]);
          exit(1);
       }
       colValues_[localRow][index] += values[i];
    }

#ifdef HAVE_AMGE
    HYPRE_LSI_AMGePutRow(row, numValues, (double*) values, (int*)scatterIndices);
#endif

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
    {
       printf("%4d : HYPRE_LSC::leaving  sumIntoSystemMatrix.\n",mypid_);
    }
    return (0);
}

//***************************************************************************
// add nonzero entries into the matrix data structure
//---------------------------------------------------------------------------

int HYPRE_LinSysCore::sumIntoSystemMatrix(int numPtRows, const int* ptRows,
                            int numPtCols, const int* ptCols,
                            const double* const* values)
{
    int i, j, k, index, colIndex, localRow;

    //-------------------------------------------------------------------
    // diagnostic message for high output level only
    //-------------------------------------------------------------------

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
    {
       printf("%4d : HYPRE_LSC::entering sumIntoSystemMatrix(2).\n",mypid_);
       if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 4 )
       {
          for ( i = 0; i < numPtRows; i++ )
          {
             localRow = ptRows[i] - localStartRow_ + 1;
             for ( j = 0; j < numPtCols; j++ )
                printf("  %4d : row,col,val = %8d %8d %e\n",mypid_,
                       localRow, ptCols[j], values[i][j]); 
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

    //-------------------------------------------------------------------
    // load the local matrix
    //-------------------------------------------------------------------

    for ( i = 0; i < numPtRows; i++ ) 
    {
       localRow = ptRows[i] - localStartRow_ + 1;
       for ( j = 0; j < numPtCols; j++ ) 
       { 
          colIndex = ptCols[j] + 1;
          index    = hypre_BinarySearch(colIndices_[localRow], colIndex, 
                                        rowLengths_[localRow]);
          if ( index < 0 )
          {
             printf("%4d : sumIntoSystemMatrix ERROR - loading column");
             printf("      that has not been declared before - %d.\n",colIndex);
             for ( k = 0; k < rowLengths_[localRow]; k++ ) 
                printf("       available column index = %d\n",
                        colIndices_[localRow][k]);
             exit(1);
          }
          colValues_[localRow][index] += values[i][j];
       }
#ifdef HAVE_AMGE
    HYPRE_LSI_AMGePutRow(localRow,numPtCols,(double*) values[i],(int*)ptCols);
#endif
    }

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
    {
       printf("%4d : HYPRE_LSC::leaving  sumIntoSystemMatrix.\n",mypid_);
    }
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
    int    i, j, localRow, newLeng, *tempInd, colIndex, index;
    double *tempVal;

    //-------------------------------------------------------------------
    // diagnostic message for high output level only
    //-------------------------------------------------------------------

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
    {
       printf("%4d : HYPRE_LSC::entering putIntoSystemMatrix(2).\n",mypid_);
    }

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
    // first adjust memory allocation (conservative)
    //-------------------------------------------------------------------

    for ( i = 0; i < numPtRows; i++ ) 
    {
       localRow = ptRows[i] - localStartRow_ + 1;
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

    //-------------------------------------------------------------------
    // load the local matrix
    //-------------------------------------------------------------------

    for ( i = 0; i < numPtRows; i++ ) 
    {
       localRow = ptRows[i] - localStartRow_ + 1;
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
      //--old_IJ---------------------------------------------------------
      // A_csr  = (HYPRE_ParCSRMatrix) HYPRE_IJMatrixGetLocalStorage(currA_);
      //--new_IJ---------------------------------------------------------
      HYPRE_IJMatrixGetObject(currA_, (void **) &A_csr);
      //-----------------------------------------------------------------
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
       //---old_IJ-------------------------------------------------------	
       // A_csr  = (HYPRE_ParCSRMatrix) HYPRE_IJMatrixGetLocalStorage(currA_);
       //---new_IJ-------------------------------------------------------	
       HYPRE_IJMatrixGetObject(currA_, (void **) &A_csr);
       //----------------------------------------------------------------	
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
    int    i, ierr, *local_ind;

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
    {
       printf("%4d : HYPRE_LSC::entering sumIntoRHSVector.\n", mypid_);
       if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 4 )
       {
          for ( i = 0; i < num; i++ )
             printf("%d : sumIntoRHSVector - %d = %e.\n", mypid_, indices[i], 
                          values[i]);
       }
    }

    //-------------------------------------------------------------------
    // change the incoming indices to 0-based before loading
    //-------------------------------------------------------------------

    local_ind = new int[num];
    for ( i = 0; i < num; i++ ) // change to 0-based
    {
#if defined(NOFEI)
       if ( indices[i] >= localStartRow_  && indices[i] <= localEndRow_ )
          local_ind[i] = indices[i] - 1;
#else
       if ((indices[i]+1) >= localStartRow_  && (indices[i]+1) <= localEndRow_)
          local_ind[i] = indices[i];
#endif
       else
       {
          printf("%d : sumIntoRHSVector ERROR - index %d out of range.\n",
                       mypid_, indices[i]);
          exit(1);
       }
    }

    //---old_IJ------------------------------------------------------------	
    //ierr = HYPRE_IJVectorAddToLocalComponents(HYb_,num,local_ind,NULL,values);
    //---new_IJ------------------------------------------------------------	
    ierr = HYPRE_IJVectorAddToValues(HYb_, num, (const int *) local_ind, 
			(const double *) values);
    //-------------------------------------------------------------------
    assert(!ierr);

    delete [] local_ind;

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
    {
       printf("%4d : HYPRE_LSC::leaving  sumIntoRHSVector.\n", mypid_);
    }
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

    for( i = 0; i < num; i++ )
    {
       index = indices[i];
       if (index < localStartRow_-1 || index >= localEndRow_) continue;
       //---old_IJ-------------------------------------------------------	
       // HYPRE_IJVectorSetLocalComponents(HYb_,1,&index,NULL,&(values[i]));
       //---new_IJ-------------------------------------------------------	
       HYPRE_IJVectorSetValues(HYb_, 1, (const int *) &index,
                    	       (const double *) &(values[i]));
       //----------------------------------------------------------------
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

    for( i = 0; i < num; i++ )
    {
       index = indices[i];
       if (index < localStartRow_-1 || index >= localEndRow_) continue;
       //---old_IJ-------------------------------------------------------	
       // HYPRE_IJVectorGetLocalComponents(HYb_,1,&index,NULL,&(values[i]));
       //---new_IJ-------------------------------------------------------	
       HYPRE_IJVectorGetValues(HYb_,1,&index,&(values[i]));
       //----------------------------------------------------------------	
    }
    return(0);
}

//***************************************************************************
// start assembling the matrix into its internal format
//---------------------------------------------------------------------------

int HYPRE_LinSysCore::matrixLoadComplete()
{
    int    i, j, ierr, numLocalEqns, leng, eqnNum, nnz, *newColInd=NULL;
    int    maxRowLeng, newLeng;
    double *newColVal=NULL;

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
    {
       printf("%4d : HYPRE_LSC::entering matrixLoadComplete.\n",mypid_);
    }

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
       assert(!ierr);

       //----------------------------------------------------------------
       // load the matrix stored locally to a HYPRE matrix
       //----------------------------------------------------------------

       numLocalEqns = localEndRow_ - localStartRow_ + 1;
       if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
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
             if ( colValues_[i][j] != 0.0 )
             {
                newColInd[newLeng]   = colIndices_[i][j] - 1;
                newColVal[newLeng++] = colValues_[i][j];
             }
          }
          //---old_IJ-------------------------------------------------	
          // HYPRE_IJMatrixInsertRow(HYA_,newLeng,eqnNum,newColInd,newColVal);
          //---new_IJ-------------------------------------------------	
          HYPRE_IJMatrixSetValues(HYA_, 1, &newLeng,(const int *) &eqnNum,
		(const int *) newColInd, (const double *) newColVal);
          //----------------------------------------------------------	
          delete [] colValues_[i];
          nnz += newLeng;
       }
       if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
       {
          printf("%4d : HYPRE_LSC::matrixLoadComplete - nnz = %d.\n",
                  mypid_, nnz);
       }
       delete [] colValues_;
       colValues_ = NULL;
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
    }

    //-------------------------------------------------------------------
    // diagnostics : print the matrix and rhs to files 
    //-------------------------------------------------------------------

    if ( HYOutputLevel_ & HYFEI_PRINTMAT )
    {
       int    rowSize, *colInd, nnz, nrows;
       double *colVal, value;
       char   fname[40];
       FILE   *fp;
       HYPRE_ParCSRMatrix A_csr;

       printf("%4d : HYPRE_LSC::print the matrix and rhs to files.\n",mypid_);
       //---old_IJ-------------------------------------------------------	
       // A_csr  = (HYPRE_ParCSRMatrix) HYPRE_IJMatrixGetLocalStorage(HYA_);
       //---new_IJ-------------------------------------------------------	
       HYPRE_IJMatrixGetObject(HYA_, (void **) &A_csr);
       //----------------------------------------------------------------	
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
                fprintf(fp, "%6d  %6d  %25.16e \n", i+1, colInd[j]+1, colVal[j]);
          }
          HYPRE_ParCSRMatrixRestoreRow(A_csr,i,&rowSize,&colInd,&colVal);
       }
       fclose(fp);
       sprintf(fname, "hypre_rhs.out.%d",mypid_);
       fp = fopen(fname,"w");
       fprintf(fp, "%6d \n", nrows);
       for ( i = localStartRow_-1; i <= localEndRow_-1; i++ )
       {
          //---old_IJ-------------------------------------------------	
          // HYPRE_IJVectorGetLocalComponents(HYb_, 1, &i, NULL, &value);
          //---new_IJ-------------------------------------------------	
          HYPRE_IJVectorGetValues(HYb_, 1, &i, &value);
          //----------------------------------------------------------	
          fprintf(fp, "%6d  %25.16e \n", i+1, value);
       }
       fclose(fp);
       MPI_Barrier(comm_);
    }

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
    {
       printf("%4d : HYPRE_LSC::leaving  matrixLoadComplete.\n",mypid_);
    }
    return (0);
}

//***************************************************************************
// has not been implemented yet
//---------------------------------------------------------------------------

int HYPRE_LinSysCore::putNodalFieldData(int fieldID, int fieldSize,
                       int* nodeNumbers, int numNodes, const double* data)
{
    (void) fieldID;
    (void) fieldSize;
    (void) nodeNumbers;
    (void) numNodes;
    (void) data;

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) > 2 )
       printf("%4d : HYPRE_LSC::putNodalFieldData not implemented.\n",mypid_);
    return (0);
}

//***************************************************************************
//This function must enforce an essential boundary condition on each local
//equation in 'globalEqn'. This means, that the following modification
//should be made to A and b, for each globalEqn:
//
//for(each local equation i){
//   for(each column j in row i) {
//      if (i==j) b[i] = gamma/alpha;
//      else b[j] -= (gamma/alpha)*A[j,i];
//   }
//}
//all of row 'globalEqn' and column 'globalEqn' in A should be zeroed,
//except for 1.0 on the diagonal.
//---------------------------------------------------------------------------

int HYPRE_LinSysCore::enforceEssentialBC(int* globalEqn, double* alpha,
                                          double* gamma, int leng)
{
    int    i, j, k, localEqnNum, colIndex, rowSize, *colInd;
    int    numLocalRows, eqnNum, rowSize2, *colInd2;
    double rhs_term, val, *colVal2, *colVal;

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
    {
       printf("%4d : HYPRE_LSC::entering enforceEssentialBC.\n",mypid_);
    }

    //-------------------------------------------------------------------
    // this function should be called before matrixLoadComplete
    //-------------------------------------------------------------------

    if ( systemAssembled_ )
    {
       printf("enforceEssentialBC ERROR : system assembled already.\n");
       exit(1);
    }

    //-------------------------------------------------------------------
    // examine each row individually
    //-------------------------------------------------------------------

    numLocalRows = localEndRow_ - localStartRow_ + 1;

    for( i = 0; i < leng; i++ ) 
    {
       localEqnNum = globalEqn[i] + 1 - localStartRow_;
       if ( localEqnNum >= 0 && localEqnNum < numLocalRows )
       {
          rowSize = rowLengths_[localEqnNum];
          colInd  = colIndices_[localEqnNum];
          colVal  = colValues_[localEqnNum];

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
                         rhs_term = gamma[i] / alpha[i] * colVal2[k];
                         eqnNum = colIndex - 1;
                         //---old_IJ------------------------------------	
                         // HYPRE_IJVectorGetLocalComponents(HYb_,1,&eqnNum, 
                         //                                 NULL, &val);
                         //---new_IJ------------------------------------	
                         HYPRE_IJVectorGetValues(HYb_,1,&eqnNum, &val);
                         //---------------------------------------------	
                         val -= rhs_term;
                         //---old_IJ------------------------------------	
                         // HYPRE_IJVectorSetLocalComponents(HYb_,1,&eqnNum,
                         //                                 NULL, &val);
                         //---new_IJ------------------------------------	
                         HYPRE_IJVectorSetValues(HYb_, 1, (const int *) &eqnNum,
                                                     (const double *) &val);
                         //---------------------------------------------	
                         colVal2[k] = 0.0;
                         break;
                      }
                   }
                }
             }
          }// end for(j<rowSize) loop

          // Set rhs for boundary point
          rhs_term = gamma[i] / alpha[i];
          eqnNum = globalEqn[i];
          //---old_IJ-------------------------------------------------	
          // HYPRE_IJVectorSetLocalComponents(HYb_,1,&eqnNum,NULL,&rhs_term);
          //---new_IJ-------------------------------------------------	
          HYPRE_IJVectorSetValues(HYb_, 1, (const int *) &eqnNum,
			(const double *) &rhs_term);
          //----------------------------------------------------------	
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

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
    {
       printf("%4d : HYPRE_LSC::leaving  enforceEssentialBC.\n",mypid_);
    }
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
    double bval, *colVal;

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
    {
       printf("%4d : HYPRE_LSC::entering enforceRemoteEssBC.\n",mypid_);
    }

    //-------------------------------------------------------------------
    // this function should be called before matrixLoadComplete
    //-------------------------------------------------------------------

    if ( systemAssembled_ )
    {
       printf("enforceRemoteEssBC ERROR : system assembled already.\n");
       exit(1);
    }

    //-------------------------------------------------------------------
    // examine each row individually
    //-------------------------------------------------------------------

    numLocalRows = localEndRow_ - localStartRow_ + 1;

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
       for ( j = 0; j < colIndLen[i]; j++) 
       {
          for ( k = 0; k < rowLen; k++ ) 
          {
             if (colInd[k]-1 == colIndices[i][j]) 
             {
                //---old_IJ----------------------------------------------	
                // HYPRE_IJVectorGetLocalComponents(HYb_,1,&eqnNum,NULL,&bval);
                //---new_IJ----------------------------------------------	
                HYPRE_IJVectorGetValues(HYb_,1,&eqnNum,&bval);
                //-------------------------------------------------------	
                bval -= ( colVal[k] * coefs[i][j] );
                colVal[k] = 0.0;
                //---old_IJ----------------------------------------------	
                // HYPRE_IJVectorSetLocalComponents(HYb_,1,&eqnNum,NULL,&bval);
                //---new_IJ----------------------------------------------	
                HYPRE_IJVectorSetValues(HYb_, 1, (const int *) &eqnNum,
				(const double *) &bval);
                //-------------------------------------------------------	
             }
          }
       }
    }
    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
    {
       printf("%4d : HYPRE_LSC::leaving  enforceRemoteEssBC.\n",mypid_);
    }
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
                                     double* beta, double* gamma, int leng)
{
    int    i, j, numLocalRows, localEqnNum, *colInd, rowSize, eqnNum;
    double val, *colVal;

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
    {
       printf("%4d : HYPRE_LSC::entering enforceOtherBC.\n",mypid_);
    }

    //-------------------------------------------------------------------
    // this function should be called before matrixLoadComplete
    //-------------------------------------------------------------------

    if ( systemAssembled_ )
    {
       printf("enforceOtherBC ERROR : system assembled already.\n");
       exit(1);
    }

    //-------------------------------------------------------------------
    // examine each row individually
    //-------------------------------------------------------------------

    numLocalRows = localEndRow_ - localStartRow_ + 1;

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
       //---old_IJ-------------------------------------------------------	
       // HYPRE_IJVectorGetLocalComponents(HYb_,1,&eqnNum,NULL,&val);
       //---new_IJ-------------------------------------------------------	
       HYPRE_IJVectorGetValues(HYb_,1,&eqnNum,&val);
       //----------------------------------------------------------------	
       val += ( gamma[i] / beta[i] );
       //---old_IJ-------------------------------------------------------	
       // HYPRE_IJVectorSetLocalComponents(HYb_,1,&eqnNum,NULL,&val);
       //---new_IJ-------------------------------------------------------	
       HYPRE_IJVectorSetValues(HYb_, 1, (const int *) &eqnNum,
                               (const double *) &val);
       //----------------------------------------------------------------	
    }
    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
    {
       printf("%4d : HYPRE_LSC::leaving  enforceOtherBC.\n",mypid_);
    }
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
//Overwrites the current internal matrix with a scaled copy of the
//input argument.
//---------------------------------------------------------------------------

#ifndef NOFEI
int HYPRE_LinSysCore::copyInMatrix(double scalar, const Data& data) 
{
    (void) scalar;
    (void) data;
    printf("%4d : HYPRE_LSC::copyInMatrix ERROR - not implemented.\n",mypid_);
    exit(1);
    return (0);
}
#endif

//***************************************************************************
//Passes out a scaled copy of the current internal matrix.
//---------------------------------------------------------------------------

#ifndef NOFEI
int HYPRE_LinSysCore::copyOutMatrix(double scalar, Data& data) 
{
    (void) scalar;
    (void) data;
    printf("%4d : HYPRE_LSC::copyOutMatrix ERROR - not implemented.\n",mypid_);
    exit(1);
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
    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
    {
       printf("%4d : HYPRE_LSC::entering getRHSVectorPtr.\n",mypid_);
    }

    data.setTypeName("IJ_Vector");
    data.setDataPtr((void*) HYb_);

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
    {
       printf("%4d : HYPRE_LSC::leaving  getRHSVectorPtr.\n",mypid_);
    }
    return (0);
}
#endif

//***************************************************************************
// copy the content of the incoming vector to the right hand side vector
//---------------------------------------------------------------------------

#ifndef NOFEI
int HYPRE_LinSysCore::copyInRHSVector(double scalar, const Data& data) 
{
    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
    {
       printf("%4d : HYPRE_LSC::entering copyInRHSVector.\n",mypid_);
    }

    if (strcmp("IJ_Vector", data.getTypeName()))
    {
       printf("copyInRHSVector: data's type string not 'IJ_Vector'.\n");
       exit(1);
    }

    HYPRE_IJVector inVec = (HYPRE_IJVector) data.getDataPtr();

    //---old_IJ-------------------------------------------------------	
    // HYPRE_ParVector srcVec = 
    //      (HYPRE_ParVector) HYPRE_IJVectorGetLocalStorage(inVec);
    // HYPRE_ParVector destVec = 
    //      (HYPRE_ParVector) HYPRE_IJVectorGetLocalStorage(HYb_);
    //---new_IJ-------------------------------------------------------	
    HYPRE_ParVector srcVec;
    HYPRE_ParVector destVec;
    HYPRE_IJVectorGetObject(inVec, (void **) &srcVec);
    HYPRE_IJVectorGetObject(HYb_, (void **) &destVec);
    //----------------------------------------------------------------	
 
    HYPRE_ParVectorCopy( srcVec, destVec);
 
    if ( scalar != 1.0 ) HYPRE_ParVectorScale( scalar, destVec);
    HYPRE_IJVectorDestroy(inVec);

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
    {
       printf("%4d : HYPRE_LSC::leaving  copyInRHSVector.\n",mypid_);
    }
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

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
    {
       printf("%4d : HYPRE_LSC::entering copyOutRHSVector.\n",mypid_);
    }

    HYPRE_IJVector newVector;
    //---old_IJ---------------------------------------------------------
    // ierr = HYPRE_IJVectorCreate(comm_, &newVector, numGlobalRows_);
    // ierr = HYPRE_IJVectorSetLocalStorageType(newVector, HYPRE_PARCSR);
    // ierr = HYPRE_IJVectorSetLocalPartitioning(newVector,localStartRow_-1,
    //                                          localEndRow_);
    // ierr = HYPRE_IJVectorAssemble(newVector);
    // ierr = HYPRE_IJVectorInitialize(newVector);
    // ierr = HYPRE_IJVectorZeroLocalComponents(newVector);
    //---new_IJ---------------------------------------------------------
    ierr = HYPRE_IJVectorCreate(comm_, localStartRow_-1, localEndRow_-1, 
		&newVector);
    ierr = HYPRE_IJVectorSetObjectType(newVector, HYPRE_PARCSR);
    ierr = HYPRE_IJVectorInitialize(newVector);
    ierr = HYPRE_IJVectorAssemble(newVector);
    //------------------------------------------------------------------
    assert(!ierr);

    //---old_IJ---------------------------------------------------------
    // HYPRE_ParVector Vec1 = 
    //      (HYPRE_ParVector) HYPRE_IJVectorGetLocalStorage(HYb_);
    // HYPRE_ParVector Vec2 = 
    //      (HYPRE_ParVector) HYPRE_IJVectorGetLocalStorage(newVector);
    //---new_IJ---------------------------------------------------------
    HYPRE_ParVector Vec1;
    HYPRE_ParVector Vec2;
    HYPRE_IJVectorGetObject(HYb_, (void **) &Vec1);
    HYPRE_IJVectorGetObject(newVector, (void **) &Vec2);
    //------------------------------------------------------------------

    HYPRE_ParVectorCopy( Vec1, Vec2);
    if ( scalar != 1.0 ) HYPRE_ParVectorScale( scalar, Vec2);

    data.setTypeName("IJ_Vector");
    data.setDataPtr((void*) Vec2);

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
    {
       printf("%4d : HYPRE_LSC::leaving  copyOutRHSVector.\n",mypid_);
    }
    return (0);
}
#endif 

//***************************************************************************
// add the incoming ParCSR vector to the current right hand side (scaled)
//---------------------------------------------------------------------------

#ifndef NOFEI
int HYPRE_LinSysCore::sumInRHSVector(double scalar, const Data& data) 
{
    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
    {
       printf("%4d : HYPRE_LSC::entering sumInRHSVector.\n",mypid_);
    }

    if (strcmp("IJ_Vector", data.getTypeName()))
    {
       printf("sumInRHSVector ERROR : data's type string not 'IJ_Vector'.\n");
       exit(1);
    }

    HYPRE_IJVector inVec = (HYPRE_IJVector) data.getDataPtr();
    //---old_IJ---------------------------------------------------------
    // HYPRE_ParVector xVec = 
    //      (HYPRE_ParVector) HYPRE_IJVectorGetLocalStorage(inVec);
    // HYPRE_ParVector yVec = 
    //      (HYPRE_ParVector) HYPRE_IJVectorGetLocalStorage(HYb_);
    //---new_IJ---------------------------------------------------------
    HYPRE_ParVector xVec;
    HYPRE_ParVector yVec;
    HYPRE_IJVectorGetObject(inVec, (void **) &xVec);
    HYPRE_IJVectorGetObject(HYb_, (void **) &yVec);
    //------------------------------------------------------------------
 
    hypre_ParVectorAxpy(scalar,(hypre_ParVector*)xVec,(hypre_ParVector*)yVec);
 
    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
    {
       printf("%4d : HYPRE_LSC::leaving  sumInRHSVector.\n",mypid_);
    }
    return (0);
}
#endif 

//***************************************************************************
// deallocate an incoming IJ matrix
//---------------------------------------------------------------------------

#ifndef NOFEI
int HYPRE_LinSysCore::destroyMatrixData(Data& data) 
{
    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
    {
       printf("%4d : HYPRE_LSC::entering destroyMatrixData.\n",mypid_);
    }

    if (strcmp("IJ_Matrix", data.getTypeName()))
    {
       printf("destroyMatrixData ERROR : data doesn't contain a IJ_Matrix.\n");
       exit(1);
    }
    HYPRE_IJMatrix mat = (HYPRE_IJMatrix) data.getDataPtr();
    HYPRE_IJMatrixDestroy(mat);

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
    {
       printf("%4d : HYPRE_LSC::leaving  destroyMatrixData.\n",mypid_);
    }
    return (0);
}
#endif 

//***************************************************************************
// deallocate an incoming IJ vector
//---------------------------------------------------------------------------

#ifndef NOFEI
int HYPRE_LinSysCore::destroyVectorData(Data& data) 
{
    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
    {
       printf("%4d : HYPRE_LSC::entering destroyVectorData.\n",mypid_);
    }

    if (strcmp("IJ_Vector", data.getTypeName()))
    {
       printf("destroyVectorData ERROR : data doesn't contain a IJ_Vector.");
       exit(1);
    }

    HYPRE_IJVector vec = (HYPRE_IJVector) data.getDataPtr();
    if ( vec != NULL ) HYPRE_IJVectorDestroy(vec);

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
    {
       printf("%4d : HYPRE_LSC::leaving  destroyVectorData.\n",mypid_);
    }
    return (0);
}
#endif 

//***************************************************************************
// set number of right hand side vectors
//---------------------------------------------------------------------------

int HYPRE_LinSysCore::setNumRHSVectors(int numRHSs, const int* rhsIDs) 
{
    int  i;

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
    {
       printf("%4d : HYPRE_LSC::entering setNumRHSVectors.\n",mypid_);
       printf("%4d : HYPRE_LSC::incoming numRHSs = %d\n",mypid_,numRHSs);
       for ( i = 0; i < numRHSs_; i++ ) 
          printf("%4d : HYPRE_LSC::incoming RHSIDs  = %d\n",mypid_,rhsIDs[i]);
    }

    if ( matrixVectorsCreated_ )
    {
       if ( HYbs_ != NULL ) 
       {
          for ( i = 0; i < numRHSs_; i++ ) 
             if ( HYbs_[i] != NULL ) HYPRE_IJVectorDestroy(HYbs_[i]);
          delete [] HYbs_;
          HYbs_ = NULL;
       }
    }

    if (numRHSs < 0)
    {
       printf("setNumRHSVectors ERROR : numRHSs < 0.\n");
       exit(1);
    }

    if (numRHSs == 0) return (0);

    delete [] rhsIDs_;
    numRHSs_ = numRHSs;
    rhsIDs_ = new int[numRHSs_];
 
    for ( i = 0; i < numRHSs; i++ ) rhsIDs_[i] = rhsIDs[i];

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
    {
       printf("%4d : HYPRE_LSC::leaving  setNumRHSVectors.\n",mypid_);
    }
    return (0);
}

//***************************************************************************
// select a right hand side vector
//---------------------------------------------------------------------------

int HYPRE_LinSysCore::setRHSID(int rhsID) 
{
    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
    {
       printf("%4d : HYPRE_LSC::entering setRHSID.\n",mypid_);
    }

    for( int i = 0; i < numRHSs_; i++ )
    {
       if (rhsIDs_[i] == rhsID)
       {
          currentRHS_ = i;
          HYb_ = HYbs_[currentRHS_];
          return (0);
       }
    }

    printf("setRHSID ERROR : rhsID %d not found.\n", rhsID);
    exit(1);

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
    {
       printf("%4d : HYPRE_LSC::leaving  setRHSID.\n",mypid_);
    }
    return (0);
}

//***************************************************************************
// used for initializing the initial guess
//---------------------------------------------------------------------------

int HYPRE_LinSysCore::putInitialGuess(const int* eqnNumbers,
                                       const double* values, int leng) 
{
    int i, ierr, *local_ind, *iarray, *iarray2;

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 0 )
    {
       printf("%4d : HYPRE_LSC::entering putInitalGuess.\n",mypid_);
    }

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
 
    local_ind = new int[leng];
    for ( i = 0; i < leng; i++ ) // change to 0-based
    {
       if ((eqnNumbers[i]+1) >= localStartRow_ && 
           (eqnNumbers[i]+1) <= localEndRow_) local_ind[i] = eqnNumbers[i];
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
    //--old_IJ-----------------------------------------------------------
    // ierr = HYPRE_IJVectorSetLocalComponents(HYx_,leng,local_ind,NULL,values);
    //--new_IJ-----------------------------------------------------------
    ierr = HYPRE_IJVectorSetValues(HYx_, leng, (const int *) local_ind,
                                   (const double *) values);
    //-------------------------------------------------------------------
    assert(!ierr);

    delete [] local_ind;

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 0 )
    {
       printf("%4d : HYPRE_LSC::leaving  putInitalGuess.\n",mypid_);
    }
    return (0);
}

//***************************************************************************
// This is a modified function for version 1.5
// used for getting the solution out of the solver, and into the application
//---------------------------------------------------------------------------

int HYPRE_LinSysCore::getSolution(double* answers,int leng) 
{
    int    i, ierr, *equations;

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
    {
       printf("%4d : HYPRE_LSC::entering getSolution.\n",mypid_);
    }
    if (leng != (localEndRow_-localStartRow_+1))
    {
       printf("%4d : HYPRE_LSC ERROR : getSolution: leng != numLocalRows.\n",
              mypid_);
       exit(1);
    }
    equations = new int[leng];
    for ( i = 0; i < leng; i++ ) equations[i] = i;

    //--old_IJ-----------------------------------------------------------
    //ierr = HYPRE_IJVectorGetLocalComponents(HYx_,leng,equations,NULL,answers);
    //--new_IJ-----------------------------------------------------------
    ierr = HYPRE_IJVectorGetValues(HYx_,leng,equations,answers);
    //-------------------------------------------------------------------
    assert(!ierr);

    delete [] equations;

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
    {
       printf("%4d : HYPRE_LSC::leaving  getSolution.\n",mypid_);
    }
    return (0);
}

//***************************************************************************
// used for getting the solution out of the solver, and into the application
//---------------------------------------------------------------------------

int HYPRE_LinSysCore::getSolnEntry(int eqnNumber, double& answer) 
{
    double val;
    int    ierr, equation;

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
    {
       printf("%4d : HYPRE_LSC::entering getSolnEntry.\n",mypid_);
    }
    equation = eqnNumber; // incoming 0-based index

    if ( equation < localStartRow_-1 && equation > localEndRow_ )
    {
       printf("%d : getSolnEntry ERROR - index out of range = %d.\n", mypid_, 
                    eqnNumber);
       exit(1);
    }

    //--old_IJ-----------------------------------------------------------
    // ierr = HYPRE_IJVectorGetLocalComponents(HYx_,1,&equation,NULL,&val);
    //--new_IJ-----------------------------------------------------------
    ierr = HYPRE_IJVectorGetValues(HYx_,1,&equation,&val);
    //-------------------------------------------------------------------
    assert(!ierr);
    answer = val;

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
    {
       printf("%4d : HYPRE_LSC::leaving  getSolnEntry.\n",mypid_);
    }
    return (0);
}

//***************************************************************************
// select which Krylov solver to use
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::selectSolver(char* name) 
{
    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
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
       if ( HYSolverID_ == HYGMRES )  HYPRE_ParCSRGMRESDestroy(HYSolver_);
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
    else if ( !strcmp(name, "gmres") )
    {
       strcpy( HYSolverName_, name );
       HYSolverID_ = HYGMRES;
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
       if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
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

       case HYGMRES :
            HYPRE_ParCSRGMRESCreate(comm_, &HYSolver_);
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
    }

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
    {
       printf("%4d : HYPRE_LSC::leaving  selectSolver.\n",mypid_);
    }
    return;
}

//***************************************************************************
// select which preconditioner to use
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::selectPreconditioner(char *name)
{
    int ierr;

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
    {
       printf("%4d : HYPRE_LSC::entering selectPreconditioner = %s.\n",
              mypid_, name);
    }
    HYPreconSetup_ = 0;
    parasailsReuse_ = 0;

    //-------------------------------------------------------------------
    // if already been allocated, destroy it first
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

       else if ( HYPreconID_ == HYDDICT )
          HYPRE_LSI_DDICTDestroy( HYPrecon_ );

       else if ( HYPreconID_ == HYPOLY )
          HYPRE_LSI_PolyDestroy( HYPrecon_ );

       else if ( HYPreconID_ == HYEUCLID )
          HYPRE_EuclidDestroy( HYPrecon_ );

       else if ( HYPreconID_ == HYBLOCK )
          HYPRE_LSI_BlockPrecondDestroy( HYPrecon_ );

#ifdef MLPACK
       else if ( HYPreconID_ == HYML )
          HYPRE_ParCSRMLDestroy( HYPrecon_ );
#endif
    }

    //-------------------------------------------------------------------
    // check for the validity of the preconditioner name
    //-------------------------------------------------------------------

    if ( !strcmp(name, "identity"  ) )
    {
       strcpy( HYPreconName_, name );
       HYPreconID_ = HYIDENTITY;
    }
    else if ( !strcmp(name, "diagonal"  ) )
    {
       strcpy( HYPreconName_, name );
       HYPreconID_ = HYDIAGONAL;
    }
    else if ( !strcmp(name, "pilut") )
    {
       strcpy( HYPreconName_, name );
       HYPreconID_ = HYPILUT;
    }
    else if ( !strcmp(name, "parasails") )
    {
       strcpy( HYPreconName_, name );
       HYPreconID_ = HYPARASAILS;
    }
    else if ( !strcmp(name, "boomeramg") )
    {
       strcpy( HYPreconName_, name );
       HYPreconID_ = HYBOOMERAMG;
    }
    else if ( !strcmp(name, "ddilut") )
    {
       strcpy( HYPreconName_, name );
       HYPreconID_ = HYDDILUT;
    }
    else if ( !strcmp(name, "schwarz") )
    {
       strcpy( HYPreconName_, name );
       HYPreconID_ = HYSCHWARZ;
    }
    else if ( !strcmp(name, "ddict") )
    {
       strcpy( HYPreconName_, name );
       HYPreconID_ = HYDDICT;
    }
    else if ( !strcmp(name, "poly") )
    {
       strcpy( HYPreconName_, name );
       HYPreconID_ = HYPOLY;
    }
    else if ( !strcmp(name, "euclid") )
    {
       strcpy( HYPreconName_, name );
       HYPreconID_ = HYEUCLID;
    }
    else if ( !strcmp(name, "block") )
    {
       strcpy( HYPreconName_, name );
       HYPreconID_ = HYBLOCK;
    }
    else if ( !strcmp(name, "ml") )
    {
#ifdef MLPACK
       strcpy( HYPreconName_, name );
       HYPreconID_ = HYML;
#else
       if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
       {
          printf("selectPreconditioner - MLPACK not declared.\n");
          printf("                       set default to diagonal.\n");
       }
       strcpy( HYPreconName_, "diagonal" );
       HYPreconID_ = HYDIAGONAL;
#endif
    }
    else
    {
       if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
       {
          printf("selectPreconditioner error : invalid option.\n");
          printf("                     use default = diagonal.\n");
       }
       strcpy( HYPreconName_, "diagonal" );
       HYPreconID_ = HYDIAGONAL;
    }

    //-------------------------------------------------------------------
    // instantiate preconditioner
    //-------------------------------------------------------------------

    switch ( HYPreconID_ )
    {
       case HYIDENTITY :
            HYPrecon_ = NULL;
            break;

       case HYDIAGONAL :
            HYPrecon_ = NULL;
            break;

       case HYPILUT :
            ierr = HYPRE_ParCSRPilutCreate( comm_, &HYPrecon_ );
            assert( !ierr );
            HYPRE_ParCSRPilutSetMaxIter( HYPrecon_, 1 );
            break;

       case HYPARASAILS :
            ierr = HYPRE_ParCSRParaSailsCreate( comm_, &HYPrecon_ );
            assert( !ierr );
            break;

       case HYBOOMERAMG :
            HYPRE_BoomerAMGCreate(&HYPrecon_);
            HYPRE_BoomerAMGSetMaxIter(HYPrecon_, 1);
            HYPRE_BoomerAMGSetCycleType(HYPrecon_, 1);
            HYPRE_BoomerAMGSetMaxLevels(HYPrecon_, 25);
            HYPRE_BoomerAMGSetMeasureType(HYPrecon_, 0);
            break;

       case HYDDILUT :
            ierr = HYPRE_LSI_DDIlutCreate( comm_, &HYPrecon_ );
            assert( !ierr );
            break;

       case HYSCHWARZ :
            ierr = HYPRE_LSI_SchwarzCreate( comm_, &HYPrecon_ );
            assert( !ierr );
            break;

       case HYDDICT :
            ierr = HYPRE_LSI_DDICTCreate( comm_, &HYPrecon_ );
            assert( !ierr );
            break;

       case HYPOLY :
            ierr = HYPRE_LSI_PolyCreate( comm_, &HYPrecon_ );
            assert( !ierr );
            break;

       case HYEUCLID :
            ierr = HYPRE_EuclidCreate( comm_, &HYPrecon_ );
            assert( !ierr );
            break;

       case HYBLOCK :
            ierr = HYPRE_LSI_BlockPrecondCreate( comm_, &HYPrecon_ );
            assert( !ierr );
            break;

#ifdef MLPACK
       case HYML :
            ierr = HYPRE_ParCSRMLCreate( comm_, &HYPrecon_ );
            break;
#endif
    }

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
    {
       printf("%4d : HYPRE_LSC::leaving  selectPreconditioner.\n",mypid_);
    }
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

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
    {
       printf("%4d : HYPRE_LSC::entering formResidual.\n", mypid_);
    }

    //*******************************************************************
    // error checking
    //-------------------------------------------------------------------

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
       printf("%4d : HYPRE_LSC formResidual ERROR : system not assembled.\n");
       exit(1);
    }

    //*******************************************************************
    // fetch matrix and vector pointers
    //-------------------------------------------------------------------

    //--old_IJ-----------------------------------------------------------
    // A_csr  = (HYPRE_ParCSRMatrix) HYPRE_IJMatrixGetLocalStorage(HYA_);
    // x_csr  = (HYPRE_ParVector)    HYPRE_IJVectorGetLocalStorage(HYx_);
    // b_csr  = (HYPRE_ParVector)    HYPRE_IJVectorGetLocalStorage(HYb_);
    // r_csr  = (HYPRE_ParVector)    HYPRE_IJVectorGetLocalStorage(HYr_);
    //--new_IJ-----------------------------------------------------------
    HYPRE_IJMatrixGetObject(HYA_, (void **) &A_csr);
    HYPRE_IJVectorGetObject(HYx_, (void **) &x_csr);
    HYPRE_IJVectorGetObject(HYb_, (void **) &b_csr);
    HYPRE_IJVectorGetObject(HYr_, (void **) &r_csr);
    //-------------------------------------------------------------------

    //*******************************************************************
    // form residual vector
    //-------------------------------------------------------------------

    HYPRE_ParVectorCopy( b_csr, r_csr );
    HYPRE_ParCSRMatrixMatvec( -1.0, A_csr, x_csr, 1.0, r_csr );

    //*******************************************************************
    // fetch residual vector
    //-------------------------------------------------------------------

    for ( i = localStartRow_-1; i < localEndRow_; i++ )
    {
       index = i - localStartRow_ + 1;
       //--old_IJ--------------------------------------------------------
       // HYPRE_IJVectorGetLocalComponents(HYr_, 1, &i, NULL, &values[index]);
       //--new_IJ--------------------------------------------------------
       HYPRE_IJVectorGetValues(HYr_, 1, &i, &values[index]);
       //----------------------------------------------------------------
    }

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
    {
       printf("%4d : HYPRE_LSC::leaving  formResidual.\n", mypid_);
    }
    return (0);
}

//***************************************************************************
// solve the linear system
//---------------------------------------------------------------------------

int HYPRE_LinSysCore::launchSolver(int& solveStatus, int &iterations)
{
    int                i, j, num_iterations=0, status, *num_sweeps, *relax_type;
    int                ierr, localNRows, rowNum, index, x2NRows;
    int                startRow, *int_array, *gint_array, startRow2;
    int                rowSize, *colInd, nnz, nrows;
    double             rnorm=0.0, *relax_wt, ddata, *colVal, value;
    double             stime, etime, ptime, rtime1, rtime2, newnorm;
    char               fname[40];
    FILE               *fp;
    HYPRE_ParCSRMatrix A_csr;
    HYPRE_ParVector    x_csr;
    HYPRE_ParVector    b_csr;
    HYPRE_ParVector    r_csr;

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
    {
       printf("%4d : HYPRE_LSC::entering launchSolver.\n", mypid_);
    }

    //*******************************************************************
    // see if Schur or slide reduction is to be performed
    //-------------------------------------------------------------------

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
    }

    MPI_Barrier(comm_);
    rtime2  = LSC_Wtime();
    
    //*******************************************************************
    // fetch matrix and vector pointers
    //-------------------------------------------------------------------

    //--old_IJ-----------------------------------------------------------
    // A_csr  = (HYPRE_ParCSRMatrix) HYPRE_IJMatrixGetLocalStorage(currA_);
    // x_csr  = (HYPRE_ParVector)    HYPRE_IJVectorGetLocalStorage(currX_);
    // b_csr  = (HYPRE_ParVector)    HYPRE_IJVectorGetLocalStorage(currB_);
    // r_csr  = (HYPRE_ParVector)    HYPRE_IJVectorGetLocalStorage(currR_);
    //--new_IJ-----------------------------------------------------------
    HYPRE_IJMatrixGetObject(currA_, (void **) &A_csr);
    HYPRE_IJVectorGetObject(currX_, (void **) &x_csr);
    HYPRE_IJVectorGetObject(currB_, (void **) &b_csr);
    HYPRE_IJVectorGetObject(currR_, (void **) &r_csr);
    //-------------------------------------------------------------------

    //*******************************************************************
    // diagnostics (print the reduced matrix to a file)
    //-------------------------------------------------------------------

    if ( HYOutputLevel_ & HYFEI_PRINTREDMAT )
    {
       if ( schurReduction_ == 1 )
       {
          x2NRows = localEndRow_ - localStartRow_ + 1 - A21NRows_;
          int_array = new int[numProcs_];
          gint_array = new int[numProcs_];
          for ( i = 0; i < numProcs_; i++ ) int_array[i] = 0;
          int_array[mypid_] = x2NRows;
          MPI_Allreduce(int_array,gint_array,numProcs_,MPI_INT,MPI_SUM,comm_);
          rowNum = 0;
          for ( i = 0; i < mypid_; i++ ) rowNum += gint_array[i];
          startRow = rowNum;
          delete [] int_array;
          delete [] gint_array;
          nrows = x2NRows;
       }
       else if ( slideReduction_ == 1 )
       {
          int_array = new int[numProcs_];
          gint_array = new int[numProcs_];
          for ( i = 0; i < numProcs_; i++ ) int_array[i] = 0;
          int_array[mypid_] = 2 * nConstraints_;
          MPI_Allreduce(int_array,gint_array,numProcs_,MPI_INT,MPI_SUM,comm_);
          rowNum = 0;
          for ( i = 0; i < mypid_; i++ ) rowNum += gint_array[i];
          startRow = localStartRow_ - 1 - rowNum;
          delete [] int_array;
          delete [] gint_array;
          nrows = localEndRow_ - localStartRow_ + 1 - 2 * nConstraints_;
       }
       else if ( slideReduction_ == 2 )
       {
          int_array = new int[numProcs_];
          gint_array = new int[numProcs_];
          for ( i = 0; i < numProcs_; i++ ) int_array[i] = 0;
          int_array[mypid_] = nConstraints_;
          MPI_Allreduce(int_array,gint_array,numProcs_,MPI_INT,MPI_SUM,comm_);
          rowNum = 0;
          for ( i = 0; i < mypid_; i++ ) rowNum += gint_array[i];
          startRow = localStartRow_ - 1 - rowNum;
          delete [] int_array;
          delete [] gint_array;
          nrows = localEndRow_ - localStartRow_ + 1 - nConstraints_;
       }
       else
       {
          nrows = localEndRow_ - localStartRow_ + 1;
          startRow = localStartRow_ - 1;
       }

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
                fprintf(fp, "%6d  %6d  %25.16e \n", i+1, colInd[j]+1, colVal[j]);
          }
          HYPRE_ParCSRMatrixRestoreRow(A_csr,i,&rowSize,&colInd,&colVal);
       }
       fclose(fp);
       sprintf(fname, "hypre_rhs.out.%d", mypid_);
       fp = fopen( fname, "w");
       fprintf(fp, "%6d \n", nrows);
       for ( i = startRow; i < startRow+nrows; i++ )
       {
          //--old_IJ-----------------------------------------------------
          // HYPRE_IJVectorGetLocalComponents(currB_, 1, &i, NULL, &ddata);
          //--new_IJ-----------------------------------------------------
          HYPRE_IJVectorGetValues(currB_, 1, &i, &ddata);
          //-------------------------------------------------------------
          fprintf(fp, "%6d  %25.16e \n", i+1, ddata);
       }
       fclose(fp);
       MPI_Barrier(comm_);
    }
#ifdef HAVE_AMGE
    if ( HYOutputLevel_ & HYFEI_PRINTFEINFO )
    {
       HYPRE_LSI_AMGeWriteToFile();
    }
#endif

    //*******************************************************************
    // choose PCG, GMRES or direct solver
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
            HYPRE_ParCSRPCGSetTol(HYSolver_, tolerance_);
            HYPRE_ParCSRPCGSetRelChange(HYSolver_, 0);
            HYPRE_ParCSRPCGSetTwoNorm(HYSolver_, 1);
            if ( normAbsRel_ == 0 ) HYPRE_ParCSRPCGSetStopCrit(HYSolver_,0);
            else                    HYPRE_ParCSRPCGSetStopCrit(HYSolver_,1);
            if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 )
            {
               if ( mypid_ == 0 )
                 printf("***************************************************\n");
               HYPRE_ParCSRPCGSetLogging(HYSolver_, 1);
            }
            HYPRE_ParCSRPCGSetup(HYSolver_, A_csr, b_csr, x_csr);
            MPI_Barrier( comm_ );
            ptime  = LSC_Wtime();
            HYPRE_ParCSRPCGSolve(HYSolver_, A_csr, b_csr, x_csr);
            HYPRE_ParCSRPCGGetNumIterations(HYSolver_, &num_iterations);
            HYPRE_ParVectorCopy( b_csr, r_csr );
            HYPRE_ParCSRMatrixMatvec( -1.0, A_csr, x_csr, 1.0, r_csr );
            HYPRE_ParVectorInnerProd( r_csr, r_csr, &rnorm);
            rnorm = sqrt( rnorm );
            switch ( projectionScheme_ )
            {
               case 1 : addToAConjProjectionSpace(currX_,currB_);  break;
               case 2 : addToMinResProjectionSpace(currX_,currB_); break;
            }
            if ( num_iterations >= maxIterations_ ) status = 1; else status = 0;
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
            HYPRE_ParCSRGMRESSetTol(HYSolver_, tolerance_);
            if ( normAbsRel_ == 0 ) HYPRE_ParCSRGMRESSetStopCrit(HYSolver_,0);
            else                    HYPRE_ParCSRGMRESSetStopCrit(HYSolver_,1);
            if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 )
            {
               if ( mypid_ == 0 )
                 printf("***************************************************\n");
               HYPRE_ParCSRGMRESSetLogging(HYSolver_, 1);
            }
            HYPRE_ParCSRGMRESSetup(HYSolver_, A_csr, b_csr, x_csr);
            MPI_Barrier( comm_ );
            ptime  = LSC_Wtime();
            HYPRE_ParCSRGMRESSolve(HYSolver_, A_csr, b_csr, x_csr);
            HYPRE_ParCSRGMRESGetNumIterations(HYSolver_, &num_iterations);
            HYPRE_ParVectorCopy( b_csr, r_csr );
            HYPRE_ParCSRMatrixMatvec( -1.0, A_csr, x_csr, 1.0, r_csr );
            HYPRE_ParVectorInnerProd( r_csr, r_csr, &rnorm);
            rnorm = sqrt( rnorm );
            switch ( projectionScheme_ )
            {
               case 1 : addToAConjProjectionSpace(currX_,currB_);  break;
               case 2 : addToMinResProjectionSpace(currX_,currB_); break;
            }
            if ( num_iterations >= maxIterations_ ) status = 1; else status = 0;
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
               if ( mypid_ == 0 )
                 printf("***************************************************\n");
               HYPRE_ParCSRBiCGSTABSetLogging(HYSolver_, 1);
            }
            HYPRE_ParCSRBiCGSTABSetup(HYSolver_, A_csr, b_csr, x_csr);
            MPI_Barrier( comm_ );
            ptime  = LSC_Wtime();
            HYPRE_ParCSRBiCGSTABSolve(HYSolver_, A_csr, b_csr, x_csr);
            HYPRE_ParCSRBiCGSTABGetNumIterations(HYSolver_, &num_iterations);
            HYPRE_ParVectorCopy( b_csr, r_csr );
            HYPRE_ParCSRMatrixMatvec( -1.0, A_csr, x_csr, 1.0, r_csr );
            HYPRE_ParVectorInnerProd( r_csr, r_csr, &rnorm);
            rnorm = sqrt( rnorm );
            switch ( projectionScheme_ )
            {
               case 1 : addToAConjProjectionSpace(currX_,currB_);  break;
               case 2 : addToMinResProjectionSpace(currX_,currB_); break;
            }
            if ( num_iterations >= maxIterations_ ) status = 1; else status = 0;
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
            HYPRE_ParCSRBiCGSTABLSetup(HYSolver_, A_csr, b_csr, x_csr);
            MPI_Barrier( comm_ );
            ptime  = LSC_Wtime();
            HYPRE_ParCSRBiCGSTABLSolve(HYSolver_, A_csr, b_csr, x_csr);
            HYPRE_ParCSRBiCGSTABLGetNumIterations(HYSolver_, &num_iterations);
            HYPRE_ParVectorCopy( b_csr, r_csr );
            HYPRE_ParCSRMatrixMatvec( -1.0, A_csr, x_csr, 1.0, r_csr );
            HYPRE_ParVectorInnerProd( r_csr, r_csr, &rnorm);
            rnorm = sqrt( rnorm );
            switch ( projectionScheme_ )
            {
               case 1 : addToAConjProjectionSpace(currX_,currB_);  break;
               case 2 : addToMinResProjectionSpace(currX_,currB_); break;
            }
            if ( num_iterations >= maxIterations_ ) status = 1; else status = 0;
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
            HYPRE_ParCSRTFQmrSetup(HYSolver_, A_csr, b_csr, x_csr);
            MPI_Barrier( comm_ );
            ptime  = LSC_Wtime();
            HYPRE_ParCSRTFQmrSolve(HYSolver_, A_csr, b_csr, x_csr);
            HYPRE_ParCSRTFQmrGetNumIterations(HYSolver_, &num_iterations);
            HYPRE_ParVectorCopy( b_csr, r_csr );
            HYPRE_ParCSRMatrixMatvec( -1.0, A_csr, x_csr, 1.0, r_csr );
            HYPRE_ParVectorInnerProd( r_csr, r_csr, &rnorm);
            rnorm = sqrt( rnorm );
            switch ( projectionScheme_ )
            {
               case 1 : addToAConjProjectionSpace(currX_,currB_);  break;
               case 2 : addToMinResProjectionSpace(currX_,currB_); break;
            }
            if ( num_iterations >= maxIterations_ ) status = 1; else status = 0;
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
            HYPRE_ParCSRBiCGSSetup(HYSolver_, A_csr, b_csr, x_csr);
            MPI_Barrier( comm_ );
            ptime  = LSC_Wtime();
            HYPRE_ParCSRBiCGSSolve(HYSolver_, A_csr, b_csr, x_csr);
            HYPRE_ParCSRBiCGSGetNumIterations(HYSolver_, &num_iterations);
            HYPRE_ParVectorCopy( b_csr, r_csr );
            HYPRE_ParCSRMatrixMatvec( -1.0, A_csr, x_csr, 1.0, r_csr );
            HYPRE_ParVectorInnerProd( r_csr, r_csr, &rnorm);
            rnorm = sqrt( rnorm );
            switch ( projectionScheme_ )
            {
               case 1 : addToAConjProjectionSpace(currX_,currB_);  break;
               case 2 : addToMinResProjectionSpace(currX_,currB_); break;
            }
            if ( num_iterations >= maxIterations_ ) status = 1; else status = 0;
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
            HYPRE_ParCSRSymQMRSetup(HYSolver_, A_csr, b_csr, x_csr);
            MPI_Barrier( comm_ );
            ptime  = LSC_Wtime();
            HYPRE_ParCSRSymQMRSolve(HYSolver_, A_csr, b_csr, x_csr);
            HYPRE_ParCSRSymQMRGetNumIterations(HYSolver_, &num_iterations);
            HYPRE_ParVectorCopy( b_csr, r_csr );
            HYPRE_ParCSRMatrixMatvec( -1.0, A_csr, x_csr, 1.0, r_csr );
            HYPRE_ParVectorInnerProd( r_csr, r_csr, &rnorm);
            rnorm = sqrt( rnorm );
            switch ( projectionScheme_ )
            {
               case 1 : addToAConjProjectionSpace(currX_,currB_);  break;
               case 2 : addToMinResProjectionSpace(currX_,currB_); break;
            }
            if ( num_iterations >= maxIterations_ ) status = 1; else status = 0;
            break;

       //----------------------------------------------------------------
       // choose Boomeramg  
       //----------------------------------------------------------------

       case HYAMG :
            solveUsingBoomeramg(status);
            HYPRE_BoomerAMGGetNumIterations(HYSolver_, &num_iterations);
            HYPRE_ParVectorCopy( b_csr, r_csr );
            HYPRE_ParCSRMatrixMatvec( -1.0, A_csr, x_csr, 1.0, r_csr );
            HYPRE_ParVectorInnerProd( r_csr, r_csr, &rnorm);
            rnorm = sqrt( rnorm );
            if ( num_iterations >= maxIterations_ ) status = 1; else status = 0;
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
            solveUsingSuperLU(status);
#ifndef NOFEI
            if ( status == 1 ) status = 0; 
#endif      
            num_iterations = 1;
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
            solveUsingSuperLUX(status);
#ifndef NOFEI
            if ( status == 1 ) status = 0; 
#endif      
            num_iterations = 1;
            //printf("SuperLUX solver - return status = %d\n",status);
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
            num_iterations = 1;
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

#ifdef HAVE_AMGE
       case HYAMGE :
            if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0 )
            {
               printf("***************************************************\n");
               printf("* AMGe (sequential) solver\n");
               printf("*--------------------------------------------------\n");
            }
            solveUsingAMGe(num_iterations);
            if ( num_iterations >= maxIterations_ ) status = 1;
            ptime  = stime;
            break;
#endif
    }

    //*******************************************************************
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
    else if ( schurReduction_ == 1 )
    {
       newnorm = rnorm;
       rnorm   = buildSchurReducedSoln();
    }

    //*******************************************************************
    // register solver return information and print timing information
    //-------------------------------------------------------------------

    solveStatus = status;
    iterations = num_iterations;

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
       printf("** HYPRE number of iterations      = %d\n", num_iterations);
       if ( slideReduction_ || schurReduction_ )
          printf("** HYPRE reduced residual norm     = %e\n", newnorm);
       printf("** HYPRE final residual norm       = %e\n", rnorm);
       printf("***************************************************\n");
    }

    //*******************************************************************
    // diagnostic information
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
          //--old_IJ-----------------------------------------------------
          // HYPRE_IJVectorGetLocalComponents(currX_, 1, &i, NULL, &ddata);
          //--new_IJ-----------------------------------------------------
          HYPRE_IJVectorGetValues(currX_, 1, &i, &ddata);
          //-------------------------------------------------------------
          fprintf(fp, "%6d  %25.16e \n", i+1, ddata);
       }
       fclose(fp);
       MPI_Barrier(comm_);
    }

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
    {
       printf("%4d : HYPRE_LSC::leaving  launchSolver.\n", mypid_);
    }
    return (0);
}

//***************************************************************************
// this function extracts the matrix in a CSR format
//---------------------------------------------------------------------------

int HYPRE_LinSysCore::writeSystem(const char *name)
{
    printf("HYPRE_LinsysCore : writeSystem not implemented.\n");
    return (0);
}

