/*BHEADER**********************************************************************
 * (c) 1999   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

#include <stdlib.h>
#include <string.h>
#include <iostream.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

#include "utilities/utilities.h"
#ifndef NOFEI
#include "Data.h"
#include "basicTypes.h"
#if defined(FEI_V13) 
#include "LinearSystemCore.1.3.h"
#elseif defined(FEI_V14)
#include "LinearSystemCore.1.4.h"
#else
#include "Lookup.h"
#include "LinearSystemCore.h"
#include "LSC.h"
#endif
#endif
#include "HYPRE.h"
#include "../../IJ_mv/HYPRE_IJ_mv.h"
#include "../../parcsr_mv/HYPRE_parcsr_mv.h"
#include "../../parcsr_ls/HYPRE_parcsr_ls.h"
#include "HYPRE_parcsr_bicgstabl.h"
#include "HYPRE_parcsr_TFQmr.h"
#include "HYPRE_parcsr_bicgs.h"
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

#ifdef SUPERLU
#include "dsp_defs.h"
#include "util.h"
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
   int HYPRE_ParCSRMLSetCoarsenScheme( HYPRE_Solver, int );
   int HYPRE_ParCSRMLSetCoarseSolver( HYPRE_Solver, int );
#endif

   int hypre_BoomerAMGBuildCoarseOperator(hypre_ParCSRMatrix*,
                   hypre_ParCSRMatrix*, hypre_ParCSRMatrix*,
                   hypre_ParCSRMatrix**);
   void qsort0(int *, int, int);
   void qsort1(int *, double *, int, int);
   int  HYPRE_DummyFunction(HYPRE_Solver, HYPRE_ParCSRMatrix,
                            HYPRE_ParVector, HYPRE_ParVector) {return 0;}

   int   getMatrixCSR(HYPRE_IJMatrix,int nrows,int nnz,int*,int*,double*);
   int   HYPRE_LSI_Search(int*, int, int);
   int   HYPRE_LSI_Sort(int*, int, int *, double *);
   void  HYPRE_LSI_Get_IJAMatrixFromFile(double **val, int **ia, int **ja, 
                  int *N, double **rhs, char *matfile, char *rhsfile);

#ifdef Y12M
   void y12maf_(int*,int*,double*,int*,int*,int*,int*,double*,
                int*,int*, double*,int*,double*,int*);
#endif
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
                  minResProjection_(0),
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

    return;
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
       for ( i = 0; i < projectSize_; i++ ) 
          if ( HYpxs_[i] != NULL ) HYPRE_IJVectorDestroy(HYpxs_[i]);
       delete [] HYpxs_;
       HYpxs_ = NULL;
    }
    if ( projectionMatrix_ != NULL ) 
    {
       for ( i = 0; i < projectSize_; i++ ) 
          if ( projectionMatrix_[i] != NULL ) delete [] projectionMatrix_[i];
       delete [] projectionMatrix_;
       projectionMatrix_ = NULL;
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
// this function takes parameters for setting internal things like solver
// and preconditioner choice, etc.
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::parameters(int numParams, char **params)
{
    int    i, k, nsweeps, rtype, olevel;
    double weight;
    char   param[256], param1[256], param2[80];

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
    {
       printf("%4d : HYPRE_LSC::entering parameters function.\n",mypid_);
       if ( mypid_ == 0 )
       {
          printf("HYPRE_LSC::parameters - numParams = %d\n", numParams);
          for ( i = 0; i < numParams; i++ )
          {
             printf("           param %d = %s \n", i, params[i]);
          }
       }
    }

    if ( numParams <= 0 ) return;

    //-------------------------------------------------------------------
    // parse all parameters
    //-------------------------------------------------------------------

    for ( i = 0; i < numParams; i++ )
    {

       sscanf(params[i],"%s", param1);
       
       //----------------------------------------------------------------
       // output level
       //----------------------------------------------------------------

       if ( !strcmp(param1, "outputLevel") )
       {
          sscanf(params[i],"%s %d", param, &olevel);
          if ( olevel < 0 ) olevel = 0;
          if ( olevel > 4 ) olevel = 4;
          HYOutputLevel_ = ( HYOutputLevel_ & HYFEI_HIGHMASK ) + olevel;
          if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
          {
             printf("       HYPRE_LSC::parameters outputLevel = %d\n",
                    HYOutputLevel_);
          }
       }

       //----------------------------------------------------------------
       // special output level
       //----------------------------------------------------------------

       else if ( !strcmp(param1, "setDebug") )
       {
          sscanf(params[i],"%s %s", param, param2);
          if (!strcmp(param2, "slideReduction1")) 
             HYOutputLevel_ |= HYFEI_SLIDEREDUCE1;
          if (!strcmp(param2, "slideReduction2")) 
             HYOutputLevel_ |= HYFEI_SLIDEREDUCE2;
          if (!strcmp(param2, "slideReduction3")) 
             HYOutputLevel_ |= HYFEI_SLIDEREDUCE3;
          if (!strcmp(param2, "schurReduction1")) 
             HYOutputLevel_ |= HYFEI_SCHURREDUCE1;
          if (!strcmp(param2, "schurReduction2")) 
             HYOutputLevel_ |= HYFEI_SCHURREDUCE2;
          if (!strcmp(param2, "schurReduction3")) 
             HYOutputLevel_ |= HYFEI_SCHURREDUCE3;
          if (!strcmp(param2, "printMat")) HYOutputLevel_ |= HYFEI_PRINTMAT;
          if (!strcmp(param2, "printSol")) HYOutputLevel_ |= HYFEI_PRINTSOL;
          if (!strcmp(param2, "printReducedMat")) 
             HYOutputLevel_ |= HYFEI_PRINTREDMAT;
          if (!strcmp(param2, "printFEInfo")) HYOutputLevel_ |= HYFEI_PRINTFEINFO;
          if (!strcmp(param2, "ddilut")) HYOutputLevel_ |= HYFEI_DDILUT;
          if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
          {
             printf("       HYPRE_LSC::parameters setDebug.\n");
          }
       }

       //----------------------------------------------------------------
       // perform Schur complement reduction
       //----------------------------------------------------------------

       else if ( !strcmp(param1, "schurReduction") )
       {
          schurReduction_ = 1;
          if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
          {
             printf("       HYPRE_LSC::parameters - schur reduction.\n");
          }
       }

       //----------------------------------------------------------------
       // perform slide reduction 
       //----------------------------------------------------------------

       else if ( !strcmp(param1, "slideReduction") )
       {
          slideReduction_ = 1;
          if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
          {
             printf("       HYPRE_LSC::parameters - slide reduction.\n");
          }
       }
       else if ( !strcmp(param1, "slideReduction2") )
       {
          slideReduction_ = 2;
          if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
          {
             printf("       HYPRE_LSC::parameters - slide reduction.\n");
          }
       }

       //----------------------------------------------------------------
       // perform minimal residual projection 
       //----------------------------------------------------------------

       else if ( !strcmp(param1, "minResProjection") )
       {
          if ( HYpbs_ != NULL ) 
          {
             for ( k = 0; k <= projectSize_; k++ ) 
                if ( HYpbs_[k] != NULL ) HYPRE_IJVectorDestroy(HYpbs_[k]);
             delete [] HYpbs_;
             HYpbs_ = NULL;
          }
          if ( HYpxs_ != NULL ) 
          {
             for ( k = 0; k < projectSize_; k++ ) 
                if ( HYpxs_[k] != NULL ) HYPRE_IJVectorDestroy(HYpxs_[k]);
             delete [] HYpxs_;
             HYpxs_ = NULL;
          }
          if ( projectionMatrix_ != NULL ) 
          {
             for ( k = 0; k < projectSize_; k++ ) 
                if (projectionMatrix_[k] != NULL) delete [] projectionMatrix_[k];
             delete [] projectionMatrix_;
             projectionMatrix_ = NULL;
          }
          sscanf(params[i],"%s %d", param, &k);
          if ( k > 0 && k < 100 ) projectSize_ = k; else projectSize_ = 10;
          if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
          {
             printf("       HYPRE_LSC::parameters minResProjection = %d\n",
                    projectSize_);
          }
          minResProjection_ = 1;
       }

       //----------------------------------------------------------------
       // which solver to pick : cg, gmres, superlu, superlux, y12m
       //----------------------------------------------------------------

       else if ( !strcmp(param1, "solver") )
       {
          sscanf(params[i],"%s %s", param, HYSolverName_);
          selectSolver(HYSolverName_);
          if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
          {
             printf("       HYPRE_LSC::parameters solver = %s\n",
                    HYSolverName_);
          }
       }

       //----------------------------------------------------------------
       // for GMRES, the restart size
       //----------------------------------------------------------------

       else if ( !strcmp(param1, "gmresDim") )
       {
          sscanf(params[i],"%s %d", param, &gmresDim_);
          if ( gmresDim_ < 1 ) gmresDim_ = 100;
          if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
          {
             printf("       HYPRE_LSC::parameters gmresDim = %d\n",
                    gmresDim_);
          }
       }

       //----------------------------------------------------------------
       // for GMRES, the convergence criterion 
       //----------------------------------------------------------------

       else if ( !strcmp(param1, "gmresStopCrit") )
       {
          sscanf(params[i],"%s %s", param, param2);
          if      ( !strcmp(param2, "absolute" ) ) normAbsRel_ = 1;
          else if ( !strcmp(param2, "relative" ) ) normAbsRel_ = 0;
          else                                     normAbsRel_ = 0;   
          
          if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
          {
             printf("       HYPRE_LSC::parameters gmresStopCrit = %s\n",
                    param2);
          }
       }

       else if ( !strcmp(param1, "stopCrit") )
       {
          sscanf(params[i],"%s %s", param, param2);
          if      ( !strcmp(param2, "absolute" ) ) normAbsRel_ = 1;
          else if ( !strcmp(param2, "relative" ) ) normAbsRel_ = 0;
          else                                     normAbsRel_ = 0;   
          
          if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
          {
             printf("       HYPRE_LSC::parameters gmresStopCrit = %s\n",
                    param2);
          }
       }

       //----------------------------------------------------------------
       // preconditioner reuse
       //----------------------------------------------------------------

       else if ( !strcmp(param1, "precond_reuse") )
       {
          sscanf(params[i],"%s %s", param, param2);
          if      ( !strcmp(param2, "on" ) )  HYPreconReuse_ = 1;
          else                                HYPreconReuse_ = 0;
          if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
          {
             printf("       HYPRE_LSC::parameters precond_reuse = %s\n",
                    param2);
          }
       }

       //----------------------------------------------------------------
       // which preconditioner : diagonal, pilut, boomeramg, parasails
       //----------------------------------------------------------------

       else if ( !strcmp(param1, "preconditioner") )
       {
          sscanf(params[i],"%s %s", param, param2);
          if      (!strcmp(param2, "reuse" )) HYPreconReuse_ = 1;
          else if (!strcmp(param2, "parasails_reuse")) parasailsReuse_ = 1;
          else
          {
             sscanf(params[i],"%s %s", param, HYPreconName_);
             selectPreconditioner(HYPreconName_);
          }
          if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
          {
             printf("       HYPRE_LSC::parameters preconditioner = %s\n",
                    HYPreconName_);
          }
       }

       //----------------------------------------------------------------
       // maximum number of iterations for pcg or gmres
       //----------------------------------------------------------------

       else if ( !strcmp(param1, "maxIterations") )
       {
          sscanf(params[i],"%s %d", param, &maxIterations_);
          if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
          {
             printf("       HYPRE_LSC::parameters maxIterations = %d\n",
                    maxIterations_);
          }
       }

       //----------------------------------------------------------------
       // tolerance as termination criterion
       //----------------------------------------------------------------

       else if ( !strcmp(param1, "tolerance") )
       {
          sscanf(params[i],"%s %lg", param, &tolerance_);
          if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
          {
             printf("       HYPRE_LSC::parameters tolerance = %e\n",
                    tolerance_);
          }
       }

       //----------------------------------------------------------------
       // pilut preconditioner : max number of nonzeros to keep per row
       //----------------------------------------------------------------

       else if ( !strcmp(param1, "pilutFillin") )
       {
          sscanf(params[i],"%s %d", param, &pilutFillin_);
          if ( pilutFillin_ < 1 ) pilutFillin_ = 50;
          if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
          {
             printf("       HYPRE_LSC::parameters pilutFillin_ = %d\n",
                    pilutFillin_);
          }
       }

       //----------------------------------------------------------------
       // pilut preconditioner : threshold to drop small nonzeros
       //----------------------------------------------------------------

       else if ( !strcmp(param1, "pilutDropTol") )
       {
          sscanf(params[i],"%s %lg", param, &pilutDropTol_);
          if (pilutDropTol_<0.0 || pilutDropTol_ >=1.0) pilutDropTol_ = 0.0;
          if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
          {
             printf("       HYPRE_LSC::parameters pilutDropTol = %e\n",
                    pilutDropTol_);
          }
       }

       //----------------------------------------------------------------
       // DDILUT preconditioner : amount of fillin (0 == same as A)
       //----------------------------------------------------------------

       else if ( !strcmp(param1, "ddilutFillin") )
       {
          sscanf(params[i],"%s %lg", param, &ddilutFillin_);
          if ( ddilutFillin_ < 0.0 ) ddilutFillin_ = 0.0;
          if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
          {
             printf("       HYPRE_LSC::parameters ddilutFillin = %d\n",
                    ddilutFillin_);
          }
       }

       //----------------------------------------------------------------
       // DDILUT preconditioner : threshold to drop small nonzeros
       //----------------------------------------------------------------

       else if ( !strcmp(param1, "ddilutDropTol") )
       {
          sscanf(params[i],"%s %lg", param, &ddilutDropTol_);
          if (ddilutDropTol_<0.0 || ddilutDropTol_ >=1.0) ddilutDropTol_ = 0.0;
          if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
          {
             printf("       HYPRE_LSC::parameters ddilutDropTol = %e\n",
                    ddilutDropTol_);
          }
       }

       //----------------------------------------------------------------
       // DDICT preconditioner : amount of fillin (0 == same as A)
       //----------------------------------------------------------------

       else if ( !strcmp(param1, "ddictFillin") )
       {
          sscanf(params[i],"%s %lg", param, &ddictFillin_);
          if ( ddictFillin_ < 0.0 ) ddictFillin_ = 0.0;
          if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
          {
             printf("       HYPRE_LSC::parameters ddictFillin = %d\n",
                    ddictFillin_);
          }
       }

       //----------------------------------------------------------------
       // DDICT preconditioner : threshold to drop small nonzeros
       //----------------------------------------------------------------

       else if ( !strcmp(param1, "ddictDropTol") )
       {
          sscanf(params[i],"%s %lg", param, &ddictDropTol_);
          if (ddictDropTol_<0.0 || ddictDropTol_ >=1.0) ddictDropTol_ = 0.0;
          if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
          {
             printf("       HYPRE_LSC::parameters ddictDropTol = %e\n",
                    ddictDropTol_);
          }
       }

       //----------------------------------------------------------------
       // Schwarz preconditioner : Fillin 
       //----------------------------------------------------------------

       else if ( !strcmp(param1, "schwarzFillin") )
       {
          sscanf(params[i],"%s %lg", param, &schwarzFillin_);
          if ( schwarzFillin_ < 0.0 ) schwarzFillin_ = 0.0;
          if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
          {
             printf("       HYPRE_LSC::parameters schwarzFillin = %e\n",
                    schwarzFillin_);
          }
       }

       //----------------------------------------------------------------
       // Schwarz preconditioner : block size 
       //----------------------------------------------------------------

       else if ( !strcmp(param1, "schwarzNBlocks") )
       {
          sscanf(params[i],"%s %d", param, &schwarzNblocks_);
          if ( schwarzNblocks_ <= 0 ) schwarzNblocks_ = 1;
          if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
          {
             printf("       HYPRE_LSC::parameters schwarzNblocks = %d\n",
                    schwarzNblocks_);
          }
       }

       //----------------------------------------------------------------
       // Schwarz preconditioner : block size 
       //----------------------------------------------------------------

       else if ( !strcmp(param1, "schwarzBlockSize") )
       {
          sscanf(params[i],"%s %d", param, &schwarzBlksize_);
          if ( schwarzBlksize_ <= 0 ) schwarzBlksize_ = 1000;
          if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
          {
             printf("       HYPRE_LSC::parameters schwarzBlockSize = %d\n",
                    schwarzBlksize_);
          }
       }

       //----------------------------------------------------------------
       // Polynomial preconditioner : order
       //----------------------------------------------------------------

       else if ( !strcmp(param1, "polyOrder") )
       {
          sscanf(params[i],"%s %d", param, &polyOrder_);
          if ( polyOrder_ < 0 ) polyOrder_ = 0;
          if ( polyOrder_ > 8 ) polyOrder_ = 8;
          if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
          {
             printf("       HYPRE_LSC::parameters polyOrder = %d\n",
                    polyOrder_);
          }
       }

       //----------------------------------------------------------------
       // superlu : ordering to use (natural, mmd)
       //----------------------------------------------------------------

       else if ( !strcmp(param1, "superluOrdering") )
       {
          sscanf(params[i],"%s %s", param, param2);
          if      ( !strcmp(param2, "natural" ) ) superluOrdering_ = 0;
          else if ( !strcmp(param2, "mmd") )      superluOrdering_ = 2;
          else                                    superluOrdering_ = 0;
          if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
          {
             printf("       HYPRE_LSC::parameters superluOrdering = %s\n",
                    param2);
          }
       }

       //----------------------------------------------------------------
       // superlu : scaling none ('N') or both col/row ('B')
       //----------------------------------------------------------------

       else if ( !strcmp(param1, "superluScale") )
       {
          sscanf(params[i],"%s %s", param, param2);
          if   ( !strcmp(param2, "y" ) ) superluScale_[0] = 'B';
          else                           superluScale_[0] = 'N';
          if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
          {
             printf("       HYPRE_LSC::parameters superluScale = %s\n",
                    params);
          }
       }

       //----------------------------------------------------------------
       // amg preconditoner : coarsening type 
       //----------------------------------------------------------------

       else if ( !strcmp(param1, "amgCoarsenType") )
       {
          sscanf(params[i],"%s %s", param, param2);
          if      ( !strcmp(param2, "cljp" ) )    amgCoarsenType_ = 0;
          else if ( !strcmp(param2, "ruge" ) )    amgCoarsenType_ = 1;
          else if ( !strcmp(param2, "ruge3c" ) )  amgCoarsenType_ = 4;
          else if ( !strcmp(param2, "falgout" ) ) amgCoarsenType_ = 6;
          else                                    amgCoarsenType_ = 0;
          if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
          {
             printf("       HYPRE_LSC::parameters amgCoarsenType = %s\n",
                    param2);
          }
       }

       //----------------------------------------------------------------
       // amg preconditoner : measure 
       //----------------------------------------------------------------

       else if ( !strcmp(param1, "amgMeasureType") )
       {
          sscanf(params[i],"%s %s", param, param2);
          if      ( !strcmp(param2, "local" ) )   amgMeasureType_ = 0;
          else if ( !strcmp(param2, "global" ) )  amgMeasureType_ = 1;
          else                                    amgMeasureType_ = 0;
          if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
          {
             printf("       HYPRE_LSC::parameters amgCoarsenType = %s\n",
                    param2);
          }
       }

       //----------------------------------------------------------------
       // amg preconditoner : no of relaxation sweeps per level
       //----------------------------------------------------------------

       else if ( !strcmp(param1, "amgNumSweeps") )
       {
          sscanf(params[i],"%s %d", param, &nsweeps);
          if ( nsweeps < 1 ) nsweeps = 1;
          for ( k = 0; k < 3; k++ ) amgNumSweeps_[k] = nsweeps;
          if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
          {
             printf("       HYPRE_LSC::parameters amgNumSweeps = %d\n",
                    nsweeps);
          }
       }

       //---------------------------------------------------------------
       // amg preconditoner : which smoother to use
       //----------------------------------------------------------------

       else if ( !strcmp(param1, "amgRelaxType") )
       {
          sscanf(params[i],"%s %s", param, param2);
          if      ( !strcmp(param2, "jacobi" ) ) rtype = 0;
          else if ( !strcmp(param2, "gsSlow") )  rtype = 1;
          else if ( !strcmp(param2, "gsFast") )  rtype = 4;
          else if ( !strcmp(param2, "hybrid" ) ) rtype = 3;
          else if ( !strcmp(param2, "hybridsym" ) ) rtype = 6;
          else                                   rtype = 4;
          for ( k = 0; k < 3; k++ ) amgRelaxType_[k] = rtype;
          if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
          {
             printf("       HYPRE_LSC::parameters amgRelaxType = %s\n",
                    params);
          }
       }

       //---------------------------------------------------------------
       // amg preconditoner : damping factor for Jacobi smoother
       //---------------------------------------------------------------

       else if ( !strcmp(param1, "amgRelaxWeight") )
       {
          sscanf(params[i],"%s %lg", param, &weight);
          if ( weight < 0.0 || weight > 1.0 ) weight = 0.5;
          for ( k = 0; k < 25; k++ ) amgRelaxWeight_[k] = weight;
          if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
          {
             printf("       HYPRE_LSC::parameters amgRelaxWeight = %e\n",
                    weight);
          }
       }

       //---------------------------------------------------------------
       // amg preconditoner : threshold to determine strong coupling
       //---------------------------------------------------------------

       else if ( !strcmp(param1, "amgStrongThreshold") )
       {
          sscanf(params[i],"%s %lg", param, &amgStrongThreshold_);
          if ( amgStrongThreshold_ < 0.0 || amgStrongThreshold_ > 1.0 )
             amgStrongThreshold_ = 0.25;
          if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
          {
             printf("       HYPRE_LSC::parameters amgStrongThreshold = %e\n",
                    amgStrongThreshold_);
          }
       }

       //---------------------------------------------------------------
       // parasails preconditoner : threshold ( >= 0.0 )
       //---------------------------------------------------------------

       else if ( !strcmp(param1, "parasailsThreshold") )
       {
          sscanf(params[i],"%s %lg", param, &parasailsThreshold_);
          if ( parasailsThreshold_ < 0.0 ) parasailsThreshold_ = 0.1;
          if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
          {
             printf("       HYPRE_LSC::parameters parasailsThreshold = %e\n",
                    parasailsThreshold_);
          }
       }

       //---------------------------------------------------------------
       // parasails preconditoner : nlevels ( >= 0)
       //---------------------------------------------------------------

       else if ( !strcmp(param1, "parasailsNlevels") )
       {
          sscanf(params[i],"%s %d", param, &parasailsNlevels_);
          if ( parasailsNlevels_ < 0 ) parasailsNlevels_ = 1;
          if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
          {
             printf("       HYPRE_LSC::parameters parasailsNlevels = %d\n",
                    parasailsNlevels_);
          }
       }

       //---------------------------------------------------------------
       // parasails preconditoner : filter
       //---------------------------------------------------------------

       else if ( !strcmp(param1, "parasailsFilter") )
       {
          sscanf(params[i],"%s %lg", param, &parasailsFilter_);

          if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
          {
             printf("       HYPRE_LSC::parameters parasailsFilter = %e\n",
                    parasailsFilter_);
          }
       }

       //---------------------------------------------------------------
       // parasails preconditoner : loadbal
       //---------------------------------------------------------------

       else if ( !strcmp(param1, "parasailsLoadbal") )
       {
          sscanf(params[i],"%s %lg", param, &parasailsLoadbal_);

          if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
          {
             printf("       HYPRE_LSC::parameters parasailsLoadbal = %e\n",
                    parasailsLoadbal_);
          }
       }

       //---------------------------------------------------------------
       // parasails preconditoner : symmetry flag (1 - symm, 0 - nonsym) 
       //---------------------------------------------------------------

       else if ( !strcmp(param1, "parasailsSymmetric") )
       {
          parasailsSym_ = 1;
          if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
          {
             printf("       HYPRE_LSC::parameters parasailsSym = %d\n",
                    parasailsSym_);
          }
       }
       else if ( !strcmp(param1, "parasailsUnSymmetric") )
       {
          parasailsSym_ = 0;
          if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
          {
             printf("       HYPRE_LSC::parameters parasailsSym = %d\n",
                    parasailsSym_);
          }
       }

       //---------------------------------------------------------------
       // parasails preconditoner : reuse flag
       //---------------------------------------------------------------

       else if ( !strcmp(param1, "parasailsReuse") )
       {
          sscanf(params[i],"%s %d", param, &parasailsReuse_);
          if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
          {
             printf("       HYPRE_LSC::parameters parasailsReuse = %d\n",
                    parasailsReuse_);
          }
       }

       //---------------------------------------------------------------
       // mlpack preconditoner : no of relaxation sweeps per level
       //---------------------------------------------------------------

       else if ( !strcmp(param1, "mlNumPresweeps") )
       {
          sscanf(params[i],"%s %d", param, &nsweeps);
          if ( nsweeps < 1 ) nsweeps = 1;
          mlNumPreSweeps_ = nsweeps;
          if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
          {
             printf("       HYPRE_LSC::parameters mlNumPresweeps = %d\n",
                    nsweeps);
          }
       }
       else if ( !strcmp(param1, "mlNumPostsweeps") )
       {
          sscanf(params[i],"%s %d", param, &nsweeps);
          if ( nsweeps < 1 ) nsweeps = 1;
          mlNumPostSweeps_ = nsweeps;
          if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
          {
             printf("       HYPRE_LSC::parameters mlNumPostsweeps = %d\n",
                    nsweeps);
          }
       }
       else if ( !strcmp(param1, "mlNumSweeps") )
       {
          sscanf(params[i],"%s %d", param, &nsweeps);
          if ( nsweeps < 1 ) nsweeps = 1;
          mlNumPreSweeps_  = nsweeps;
          mlNumPostSweeps_ = nsweeps;
          if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
          {
             printf("       HYPRE_LSC::parameters mlNumSweeps = %d\n",
                    nsweeps);
          }
       }

       //---------------------------------------------------------------
       // mlpack preconditoner : which smoother to use
       //---------------------------------------------------------------

       else if ( !strcmp(param1, "mlPresmootherType") )
       {
          sscanf(params[i],"%s %s", param, param2);
          rtype = 1;
          if      ( !strcmp(param2, "jacobi" ) )  rtype = 0;
          else if ( !strcmp(param2, "sgs") )      rtype = 1;
          else if ( !strcmp(param2, "sgsseq") )   rtype = 2;
          else if ( !strcmp(param2, "vbjacobi"))  rtype = 3;
          else if ( !strcmp(param2, "vbsgs") )    rtype = 4;
          else if ( !strcmp(param2, "vbsgsseq"))  rtype = 5;
          else if ( !strcmp(param2, "ilut") )     rtype = 6;
          else if ( !strcmp(param2, "aSchwarz") ) rtype = 7;
          else if ( !strcmp(param2, "mSchwarz") ) rtype = 8;
          mlPresmootherType_  = rtype;
          if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
          {
             printf("       HYPRE_LSC::parameters mlPresmootherType = %s\n",
                    param2);
          }
       }
       else if ( !strcmp(param1, "mlPostsmootherType") )
       {
          sscanf(params[i],"%s %s", param, param2);
          rtype = 1;
          if      ( !strcmp(param2, "jacobi" ) ) rtype = 0;
          else if ( !strcmp(param2, "sgs") )     rtype = 1;
          else if ( !strcmp(param2, "sgsseq") )  rtype = 2;
          else if ( !strcmp(param2, "vbjacobi")) rtype = 3;
          else if ( !strcmp(param2, "vbsgs") )   rtype = 4;
          else if ( !strcmp(param2, "vbsgsseq")) rtype = 5;
          mlPostsmootherType_  = rtype;
          if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
          {
             printf("       HYPRE_LSC::parameters mlPostsmootherType = %s\n",
                    param2);
          }
       }
       else if ( !strcmp(param1, "mlRelaxType") )
       {
          sscanf(params[i],"%s %s", param, param2);
          rtype = 1;
          if      ( !strcmp(param2, "jacobi" ) ) rtype = 0;
          else if ( !strcmp(param2, "sgs") )     rtype = 1;
          else if ( !strcmp(param2, "sgsseq") )  rtype = 2;
          else if ( !strcmp(param2, "vbjacobi")) rtype = 3;
          else if ( !strcmp(param2, "vbsgs") )   rtype = 4;
          else if ( !strcmp(param2, "vbsgsseq")) rtype = 5;
          mlPresmootherType_  = rtype;
          mlPostsmootherType_ = rtype;
          if ( rtype == 6 ) mlPostsmootherType_ = 1;
          if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
          {
             printf("       HYPRE_LSC::parameters mlRelaxType = %s\n",
                    param2);
          }
       }

       //---------------------------------------------------------------
       // mlpack preconditoner : damping factor for Jacobi smoother
       //---------------------------------------------------------------

       else if ( !strcmp(param1, "mlRelaxWeight") )
       {
          sscanf(params[i],"%s %lg", param, &weight);
          if ( weight < 0.0 || weight > 1.0 ) weight = 0.5;
          mlRelaxWeight_ = weight;
          if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
          {
             printf("       HYPRE_LSC::parameters mlRelaxWeight = %e\n",
                    weight);
          }
       }

       //---------------------------------------------------------------
       // mlpack preconditoner : threshold to determine strong coupling
       //---------------------------------------------------------------

       else if ( !strcmp(param1, "mlStrongThreshold") )
       {
          sscanf(params[i],"%s %lg", param, &mlStrongThreshold_);
          if ( mlStrongThreshold_ < 0.0 || mlStrongThreshold_ > 1.0 )
             mlStrongThreshold_ = 0.08;
          if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
          {
             printf("       HYPRE_LSC::parameters mlStrongThreshold = %e\n",
                    mlStrongThreshold_);
          }
       }

       //---------------------------------------------------------------
       // mlpack preconditoner : method to use
       //---------------------------------------------------------------

       else if ( !strcmp(param1, "mlMethod") )
       {
          sscanf(params[i],"%s %s", param, param2);
          if      ( !strcmp(param2, "amg" ) ) mlMethod_ = 0;
          else                                mlMethod_ = 1;
          if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
          {
             printf("       HYPRE_LSC::parameters mlMethod = %d\n",mlMethod_);
          }
       }

       //---------------------------------------------------------------
       // mlpack preconditoner : coarse solver to use
       //---------------------------------------------------------------

       else if ( !strcmp(param1, "mlCoarseSolver") )
       {
          sscanf(params[i],"%s %s", param, param2);
          if      ( !strcmp(param2, "superlu" ) )     mlCoarseSolver_ = 0;
          else if ( !strcmp(param2, "aggregation" ) ) mlCoarseSolver_ = 1;
          else if ( !strcmp(param2, "GS" ) )          mlCoarseSolver_ = 2;
          else                                        mlCoarseSolver_ = 1;
          if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
          {
             printf("       HYPRE_LSC::parameters mlCoarseSolver = %d\n",
                    mlCoarseSolver_);
          }
       }

       //---------------------------------------------------------------
       // mlpack preconditoner : coarsening scheme to use
       //---------------------------------------------------------------

       else if ( !strcmp(param1, "mlCoarsenScheme") )
       {
          sscanf(params[i],"%s %s", param, param2);
          if      ( !strcmp(param2, "uncoupled" ) ) mlCoarsenScheme_ = 1;
          else if ( !strcmp(param2, "coupled" ) )   mlCoarsenScheme_ = 2;
          else if ( !strcmp(param2, "mis" ) )       mlCoarsenScheme_ = 3;
          else if ( !strcmp(param2, "hybridum" ) )  mlCoarsenScheme_ = 5;
          else if ( !strcmp(param2, "hybriduc" ) )  mlCoarsenScheme_ = 6;
          else                                      mlCoarsenScheme_ = 1;
          if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
          {
             printf("       HYPRE_LSC::parameters mlCoarsenScheme = %d\n",
                    mlCoarsenScheme_);
          }
       }

       //---------------------------------------------------------------
       // error 
       //---------------------------------------------------------------

       else
       {
          if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 && mypid_ == 0 )
          {
             printf("HYPRE_LSC::parameters WARNING : %s not recognized\n",
                    params[i]);
          }
       }
    }

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
    {
       printf("%4d : HYPRE_LSC::leaving  parameters function.\n",mypid_);
    }

    return;
}

//***************************************************************************
// passing a lookup table to this object
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::setLookup(Lookup& lookup)
{
   if (&lookup == NULL) return;

   lookup_ = &lookup;
   haveLookup_ = 1;

   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 5 )
   {
      //let's test the shared-node lookup functions.

#ifndef NOFEI
#if (!defined(FEI_V14)) && (!defined(FEI_V13))
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
#else
      printf("HYPRE_LSC::setLookup : function not implemented.\n");
#endif
#endif
   }
}

//***************************************************************************
//This function is where we establish the structures/objects associated
//with the linear algebra library. i.e., do initial allocations, etc.
// Rows and columns are 1-based.
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::createMatricesAndVectors(int numGlobalEqns,
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
       if (reducedA_ != NULL) {HYPRE_IJMatrixDestroy(reducedA_); reducedA_ = NULL;}
       if (reducedB_ != NULL) {HYPRE_IJVectorDestroy(reducedB_); reducedB_ = NULL;}
       if (reducedX_ != NULL) {HYPRE_IJVectorDestroy(reducedX_); reducedX_ = NULL;}
       if (reducedR_ != NULL) {HYPRE_IJVectorDestroy(reducedR_); reducedR_ = NULL;}
       if (HYA21_    != NULL) {HYPRE_IJMatrixDestroy(HYA21_);    HYA21_    = NULL;}
       if (HYA12_    != NULL) {HYPRE_IJMatrixDestroy(HYA12_);    HYA12_    = NULL;}
       if (HYinvA22_ != NULL) {HYPRE_IJMatrixDestroy(HYinvA22_); HYinvA22_ = NULL;}
       A21NRows_ = A21NCols_ = reducedAStartRow_ = 0;
    }

    //-------------------------------------------------------------------
    // instantiate the matrix
    //-------------------------------------------------------------------

    ierr = HYPRE_IJMatrixCreate(comm_,&HYA_,numGlobalRows_,numGlobalRows_);
    ierr = HYPRE_IJMatrixSetLocalStorageType(HYA_, HYPRE_PARCSR);
    ierr = HYPRE_IJMatrixSetLocalSize(HYA_, numLocalEqns, numLocalEqns);
    assert(!ierr);

    //-------------------------------------------------------------------
    // instantiate the right hand vectors
    //-------------------------------------------------------------------

    HYbs_ = new HYPRE_IJVector[numRHSs_];
    for ( i = 0; i < numRHSs_; i++ )
    {
       ierr = HYPRE_IJVectorCreate(comm_, &(HYbs_[i]), numGlobalRows_);
       ierr = HYPRE_IJVectorSetLocalStorageType(HYbs_[i], HYPRE_PARCSR);
       ierr = HYPRE_IJVectorSetLocalPartitioning(HYbs_[i],localStartRow_-1,
                                                 localEndRow_);
       ierr = HYPRE_IJVectorAssemble(HYbs_[i]);
       ierr = HYPRE_IJVectorInitialize(HYbs_[i]);
       ierr = HYPRE_IJVectorZeroLocalComponents(HYbs_[i]);
       assert(!ierr);
    }
    HYb_ = HYbs_[0];

    //-------------------------------------------------------------------
    // instantiate the solution vector
    //-------------------------------------------------------------------

    ierr = HYPRE_IJVectorCreate(comm_, &HYx_, numGlobalRows_);
    ierr = HYPRE_IJVectorSetLocalStorageType(HYx_, HYPRE_PARCSR);
    ierr = HYPRE_IJVectorSetLocalPartitioning(HYx_,localStartRow_-1,
                                              localEndRow_);
    ierr = HYPRE_IJVectorAssemble(HYx_);
    ierr = HYPRE_IJVectorInitialize(HYx_);
    ierr = HYPRE_IJVectorZeroLocalComponents(HYx_);
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

    ierr = HYPRE_IJVectorCreate(comm_, &HYr_, numGlobalRows_);
    ierr = HYPRE_IJVectorSetLocalStorageType(HYr_, HYPRE_PARCSR);
    ierr = HYPRE_IJVectorSetLocalPartitioning(HYr_,localStartRow_-1,
                                              localEndRow_);
    ierr = HYPRE_IJVectorAssemble(HYr_);
    ierr = HYPRE_IJVectorInitialize(HYr_);
    ierr = HYPRE_IJVectorZeroLocalComponents(HYr_);
    assert(!ierr);
    matrixVectorsCreated_ = 1;
    schurReductionCreated_ = 0;
    systemAssembled_ = 0;

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
    {
       printf("%4d : HYPRE_LSC::leaving  createMatricesAndVectors.\n",mypid_);
    }
}

//***************************************************************************
// similar to createMatrixVectors (FEI 1.5 compatible)
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::setGlobalOffsets(int leng, int* nodeOffsets,
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
}

//***************************************************************************
// new functions in FEI 1.5 
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::setConnectivities(GlobalID elemBlock, int numElements,
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
}

void HYPRE_LinSysCore::setStiffnessMatrices(GlobalID elemBlock, int numElems,
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
}

void HYPRE_LinSysCore::setLoadVectors(GlobalID elemBlock, int numElems,
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
}

//***************************************************************************
// Set the number of rows in the diagonal part and off diagonal part
// of the matrix, using the structure of the matrix, stored in rows.
// rows is an array that is 0-based.  localStartRow and localEndRow are 1-based.
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::allocateMatrix(int **colIndices, int *rowLengths)
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
}

//***************************************************************************
// to establish the structures/objects associated with the linear algebra 
// library. i.e., do initial allocations, etc.
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::setMatrixStructure(int** ptColIndices, int* ptRowLengths,
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
}

//***************************************************************************
// new functions in FEI 1.5
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::setMultCREqns(int multCRSetID, int numCRs, 
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
}

void HYPRE_LinSysCore::setPenCREqns(int penCRSetID, int numCRs, 
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
}

//***************************************************************************
// This function is needed in order to construct a new problem with the
// same sparsity pattern.
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::resetMatrixAndVector(double s)
{
    int  i, j, ierr, size;

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
    {
       printf("%4d : HYPRE_LSC::entering resetMatrixAndVector.\n",mypid_);
    }

    if ( s != 0.0 && mypid_ == 0 )
    {
       printf("resetMatrixAndVector ERROR : cannot take nonzeros.\n");
       exit(1);
    }

    for (i = 0; i < numRHSs_; i++) HYPRE_IJVectorZeroLocalComponents(HYbs_[i]);
    systemAssembled_ = 0;
    schurReductionCreated_ = 0;
    projectCurrSize_ = 0;

    //-------------------------------------------------------------------
    // for now, since HYPRE does not yet support
    // re-initializing the matrix, restart the whole thing
    //-------------------------------------------------------------------

    if ( HYA_ != NULL ) HYPRE_IJMatrixDestroy(HYA_);
    ierr = HYPRE_IJMatrixCreate(comm_,&HYA_,numGlobalRows_,numGlobalRows_);
    ierr = HYPRE_IJMatrixSetLocalStorageType(HYA_, HYPRE_PARCSR);
    size = localEndRow_ - localStartRow_ + 1;
    ierr = HYPRE_IJMatrixSetLocalSize(HYA_, size, size);
    //ierr = HYPRE_IJMatrixSetRowSizes(HYA_, rowLengths_);
    //ierr = HYPRE_IJMatrixInitialize(HYA_);
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
}

//***************************************************************************
// new function to reset matrix independently
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::resetMatrix(double s) 
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

    if ( HYA_ != NULL ) HYPRE_IJMatrixDestroy(HYA_);
    ierr = HYPRE_IJMatrixCreate(comm_,&HYA_,numGlobalRows_,numGlobalRows_);
    ierr = HYPRE_IJMatrixSetLocalStorageType(HYA_, HYPRE_PARCSR);
    size = localEndRow_ - localStartRow_ + 1;
    ierr = HYPRE_IJMatrixSetLocalSize(HYA_, size, size);
    //ierr = HYPRE_IJMatrixSetRowSizes(HYA_, rowLengths_);
    //ierr = HYPRE_IJMatrixInitialize(HYA_);
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
}

//***************************************************************************
// new function to reset vectors independently
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::resetRHSVector(double s) 
{
    int  i, j, ierr, size;

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
       for (i = 0; i < numRHSs_; i++) 
          if ( HYbs_[i] != NULL ) HYPRE_IJVectorZeroLocalComponents(HYbs_[i]);
    }

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
    {
       printf("%4d : HYPRE_LSC::leaving  resetRHSVector.\n",mypid_);
    }
}

//***************************************************************************
// add nonzero entries into the matrix data structure
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::sumIntoSystemMatrix(int row, int numValues,
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
             printf("       available column index = %d\n",colIndices_[localRow][j]);
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
}

//***************************************************************************
// add nonzero entries into the matrix data structure (FEI 1.5)
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::sumIntoSystemMatrix(int numPtRows, const int* ptRows,
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
}

//***************************************************************************
// add nonzero entries into the matrix data structure (FEI 1.5)
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::sumIntoSystemMatrix(int numPtRows, const int* ptRows,
                       int numPtCols, const int* ptCols, int numBlkRows, 
                       const int* blkRows, int numBlkCols, const int* blkCols,
                       const double* const* values)
{
    (void) numBlkRows;
    (void) blkRows;
    (void) numBlkCols;
    (void) blkCols;

    sumIntoSystemMatrix(numPtRows, ptRows, numPtCols, ptCols, values);
}

//***************************************************************************
// input is 1-based, but HYPRE vectors are 0-based
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::sumIntoRHSVector(int num, const double* values,
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
#if defined(FEI_V13) || defined(FEI_V14) || defined(NOFEI)
       if ( indices[i] >= localStartRow_  && indices[i] <= localEndRow_ )
          local_ind[i] = indices[i] - 1;
#else
       if ( (indices[i]+1) >= localStartRow_  && (indices[i]+1) <= localEndRow_ )
          local_ind[i] = indices[i];
#endif
       else
       {
          printf("%d : sumIntoRHSVector ERROR - index %d out of range.\n",
                       mypid_, indices[i]);
          exit(1);
       }
    }

    ierr = HYPRE_IJVectorAddToLocalComponents(HYb_,num,local_ind,NULL,values);
    assert(!ierr);

    delete [] local_ind;

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
    {
       printf("%4d : HYPRE_LSC::leaving  sumIntoRHSVector.\n", mypid_);
    }
}

//***************************************************************************
// start assembling the matrix into its internal format
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::matrixLoadComplete()
{
    int i, j, ierr, numLocalEqns, leng, eqnNum, nnz;

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
    {
       printf("%4d : HYPRE_LSC::entering matrixLoadComplete.\n",mypid_);
    }

    //-------------------------------------------------------------------
    // if matrix has not been assembled or it has been reset
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
       nnz = 0;
       for ( i = 0; i < numLocalEqns; i++ )
       {
          eqnNum = localStartRow_ - 1 + i;
          leng   = rowLengths_[i];
          nnz   += leng;
          for ( j = 0; j < leng; j++ ) colIndices_[i][j]--;
          HYPRE_IJMatrixInsertRow(HYA_,leng,eqnNum,colIndices_[i],colValues_[i]);
          for ( j = 0; j < leng; j++ ) colIndices_[i][j]++;
          delete [] colValues_[i];
       }
       if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
       {
          printf("%4d : HYPRE_LSC::matrixLoadComplete - nnz = %d.\n",
                  mypid_, nnz);
       }
       delete [] colValues_;
       colValues_ = NULL;

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

       printf("%4d : HYPRE_LSC::print matrix and rhs to files.\n",mypid_);
       A_csr  = (HYPRE_ParCSRMatrix) HYPRE_IJMatrixGetLocalStorage(HYA_);
       sprintf(fname, "hypre_mat.out.%d",mypid_);
       fp = fopen(fname,"w");
       nrows = localEndRow_ - localStartRow_ + 1;
       nnz = 0;
       for ( i = localStartRow_-1; i <= localEndRow_-1; i++ )
       {
          HYPRE_ParCSRMatrixGetRow(A_csr,i,&rowSize,&colInd,&colVal);
          nnz += rowSize;
          HYPRE_ParCSRMatrixRestoreRow(A_csr,i,&rowSize,&colInd,&colVal);
       }
       fprintf(fp, "%6d  %7d \n", nrows, nnz);
       for ( i = localStartRow_-1; i <= localEndRow_-1; i++ )
       {
          HYPRE_ParCSRMatrixGetRow(A_csr,i,&rowSize,&colInd,&colVal);
          for (j = 0; j < rowSize; j++)
             fprintf(fp, "%6d  %6d  %e \n", i+1, colInd[j]+1, colVal[j]);
          HYPRE_ParCSRMatrixRestoreRow(A_csr,i,&rowSize,&colInd,&colVal);
       }
       fclose(fp);
       sprintf(fname, "hypre_rhs.out.%d",mypid_);
       fp = fopen(fname,"w");
       fprintf(fp, "%6d \n", nrows);
       for ( i = localStartRow_-1; i <= localEndRow_-1; i++ )
       {
          HYPRE_IJVectorGetLocalComponents(currB_, 1, &i, NULL, &value);
          fprintf(fp, "%6d  %e \n", i+1, value);
       }
       fclose(fp);
       MPI_Barrier(comm_);
    }

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
    {
       printf("%4d : HYPRE_LSC::leaving  matrixLoadComplete.\n",mypid_);
    }
}

//***************************************************************************
// new function in 1.5
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::putNodalFieldData(int fieldID, int fieldSize,
                       int* nodeNumbers, int numNodes, const double* data)
{
    (void) fieldID;
    (void) fieldSize;
    (void) nodeNumbers;
    (void) numNodes;
    (void) data;

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) > 2 )
       printf("%4d : HYPRE_LSC::putNodalFieldData not implemented.\n",mypid_);
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

void HYPRE_LinSysCore::enforceEssentialBC(int* globalEqn, double* alpha,
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
#if defined(FEI_V13) || defined(FEI_V14)
       localEqnNum = globalEqn[i] - localStartRow_;
#else
       localEqnNum = globalEqn[i] + 1 - localStartRow_;
#endif
       if ( localEqnNum >= 0 && localEqnNum < numLocalRows )
       {
          rowSize = rowLengths_[localEqnNum];
          colInd  = colIndices_[localEqnNum];
          colVal  = colValues_[localEqnNum];

          for ( j = 0; j < rowSize; j++ ) 
          {
             colIndex = colInd[j];
#if defined(FEI_V13) || defined(FEI_V14)
             if ( colIndex == globalEqn[i] ) colVal[j] = 1.0;
             else                            colVal[j] = 0.0;
#else
             if ( colIndex-1 == globalEqn[i] ) colVal[j] = 1.0;
             else                              colVal[j] = 0.0;
#endif

             if ( colIndex >= localStartRow_ && colIndex <= localEndRow_) 
             {
#if defined(FEI_V13) || defined(FEI_V14)
                if ( colIndex != globalEqn[i]) 
#else
                if ( (colIndex-1) != globalEqn[i]) 
#endif
                {
                   rowSize2 = rowLengths_[colIndex-localStartRow_];
                   colInd2  = colIndices_[colIndex-localStartRow_];
                   colVal2  = colValues_ [colIndex-localStartRow_];

                   for( k = 0; k < rowSize2; k++ ) 
                   {
#if defined(FEI_V13) || defined(FEI_V14)
                      if ( colInd2[k] == globalEqn[i] ) 
#else
                      if ( colInd2[k]-1 == globalEqn[i] ) 
#endif
                      {
                         rhs_term = gamma[i] / alpha[i] * colVal2[k];
                         eqnNum = colIndex - 1;
                         HYPRE_IJVectorGetLocalComponents(HYb_,1,&eqnNum, 
                                                          NULL, &val);
                         val -= rhs_term;
                         HYPRE_IJVectorSetLocalComponents(HYb_,1,&eqnNum,
                                                          NULL, &val);
                         colVal2[k] = 0.0;
                         break;
                      }
                   }
                }
             }
          }// end for(j<rowSize) loop

          // Set rhs for boundary point
          rhs_term = gamma[i] / alpha[i];
#if defined(FEI_V13) || defined(FEI_V14)
          eqnNum = globalEqn[i] - 1;
#else
          eqnNum = globalEqn[i];
#endif
          HYPRE_IJVectorSetLocalComponents(HYb_,1,&eqnNum,NULL,&rhs_term);
       }
    }

    //-------------------------------------------------------------------
    // set up the AMGe Dirichlet boundary conditions
    //-------------------------------------------------------------------

#ifdef HAVE_AMGE
    colInd = new int[leng];
#if defined(FEI_V13) || defined(FEI_V14)
    for( i = 0; i < leng; i++ ) colInd[i] = globalEqn[i] - 1;
#else
    for( i = 0; i < leng; i++ ) colInd[i] = globalEqn[i];
#endif
    HYPRE_LSI_AMGeSetBoundary( leng, colInd );
    delete [] colInd;
#endif

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
    {
       printf("%4d : HYPRE_LSC::leaving  enforceEssentialBC.\n",mypid_);
    }
}

//***************************************************************************
// new function 
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::enforceRemoteEssBCs(int numEqns, int* globalEqns,
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
#if defined(FEI_V13) || defined(FEI_V14)
       localEqnNum = globalEqns[i] - localStartRow_;
#else
       localEqnNum = globalEqns[i] + 1 - localStartRow_;
#endif
       if ( localEqnNum < 0 || localEqnNum >= numLocalRows )
       {
          continue;
       }

       rowLen = rowLengths_[localEqnNum];
       colInd = colIndices_[localEqnNum];
       colVal = colValues_[localEqnNum];

#if defined(FEI_V13) || defined(FEI_V14)
       eqnNum = globalEqns[i] - 1;
#else
       eqnNum = globalEqns[i];
#endif

       for ( j = 0; j < colIndLen[i]; j++) 
       {
          for ( k = 0; k < rowLen; k++ ) 
          {
#if defined(FEI_V13) || defined(FEI_V14)
             if (colInd[k] == colIndices[i][j]) 
#else
             if (colInd[k]-1 == colIndices[i][j]) 
#endif
             {
                HYPRE_IJVectorGetLocalComponents(HYb_,1,&eqnNum,NULL,&bval);
                bval -= ( colVal[k] * coefs[i][j] );
                colVal[k] = 0.0;
                HYPRE_IJVectorSetLocalComponents(HYb_,1,&eqnNum,NULL,&bval);
             }
          }
       }
    }

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
    {
       printf("%4d : HYPRE_LSC::leaving  enforceRemoteEssBC.\n",mypid_);
    }
}

//***************************************************************************
//This function must enforce a natural or mixed boundary condition on the
//equations in 'globalEqn'. This means that the following modification should
//be made to A and b:
//
//A[globalEqn,globalEqn] += alpha/beta;
//b[globalEqn] += gamma/beta;
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::enforceOtherBC(int* globalEqn, double* alpha, 
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
#if defined(FEI_V13) || defined(FEI_V14)
       localEqnNum = globalEqn[i] - localStartRow_;
#else
       localEqnNum = globalEqn[i] + 1 - localStartRow_;
#endif
       if ( localEqnNum < 0 || localEqnNum >= numLocalRows )
       {
          continue;
       }

       rowSize = rowLengths_[localEqnNum];
       colVal  = colValues_[localEqnNum];
       colInd  = colIndices_[localEqnNum];

       for ( j = 0; j < rowSize; j++) 
       {
#if defined(FEI_V13) || defined(FEI_V14)
          if ( colInd[j] == globalEqn[i]) 
#else
          if ((colInd[j]-1) == globalEqn[i]) 
#endif
          {
             colVal[j] += alpha[i]/beta[i];
             break;
          }
       }

       //now make the rhs modification.
       // need to fetch matrix and put it back before assembled

#if defined(FEI_V13) || defined(FEI_V14)
       eqnNum = globalEqn[i] - 1;
#else
       eqnNum = globalEqn[i];
#endif

       HYPRE_IJVectorGetLocalComponents(HYb_,1,&eqnNum,NULL,&val);
       val += ( gamma[i] / beta[i] );
       HYPRE_IJVectorSetLocalComponents(HYb_,1,&eqnNum,NULL,&val);
    }

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
    {
       printf("%4d : HYPRE_LSC::leaving  enforceOtherBC.\n",mypid_);
    }
}

//***************************************************************************
// put the pointer to the A matrix into the Data object
//---------------------------------------------------------------------------

#ifndef NOFEI
void HYPRE_LinSysCore::getMatrixPtr(Data& data) 
{
   (void) data;
   printf("%4d : HYPRE_LSC::getmatrixPtr ERROR - not implemented.\n",mypid_);
   exit(1);
}
#endif

//***************************************************************************
//Overwrites the current internal matrix with a scaled copy of the
//input argument.
//---------------------------------------------------------------------------

#ifndef NOFEI
void HYPRE_LinSysCore::copyInMatrix(double scalar, const Data& data) 
{
    (void) scalar;
    (void) data;
    printf("%4d : HYPRE_LSC::copyInMatrix ERROR - not implemented.\n",mypid_);
    exit(1);
}
#endif

//***************************************************************************
//Passes out a scaled copy of the current internal matrix.
//---------------------------------------------------------------------------

#ifndef NOFEI
void HYPRE_LinSysCore::copyOutMatrix(double scalar, Data& data) 
{
    (void) scalar;
    (void) data;
    printf("%4d : HYPRE_LSC::copyOutMatrix ERROR - not implemented.\n",mypid_);
    exit(1);
}
#endif

//***************************************************************************
// add nonzero entries into the matrix data structure
//---------------------------------------------------------------------------

#ifndef NOFEI
void HYPRE_LinSysCore::sumInMatrix(double scalar, const Data& data) 
{
    (void) scalar;
    (void) data;
    printf("%4d : HYPRE_LSC::sumInMatrix ERROR - not implemented.\n",mypid_);
    exit(1);
}
#endif

//***************************************************************************
// get the data pointer for the right hand side
//---------------------------------------------------------------------------

#ifndef NOFEI
void HYPRE_LinSysCore::getRHSVectorPtr(Data& data) 
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
}
#endif

//***************************************************************************

#ifndef NOFEI
void HYPRE_LinSysCore::copyInRHSVector(double scalar, const Data& data) 
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

    HYPRE_ParVector srcVec = 
          (HYPRE_ParVector) HYPRE_IJVectorGetLocalStorage(inVec);
    HYPRE_ParVector destVec = 
          (HYPRE_ParVector) HYPRE_IJVectorGetLocalStorage(HYb_);
 
    HYPRE_ParVectorCopy( srcVec, destVec);
 
    if ( scalar != 1.0 ) HYPRE_ParVectorScale( scalar, destVec);
    HYPRE_IJVectorDestroy(inVec);

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
    {
       printf("%4d : HYPRE_LSC::leaving  copyInRHSVector.\n",mypid_);
    }
}
#endif

//***************************************************************************

#ifndef NOFEI
void HYPRE_LinSysCore::copyOutRHSVector(double scalar, Data& data) 
{
    int ierr;

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
    {
       printf("%4d : HYPRE_LSC::entering copyOutRHSVector.\n",mypid_);
    }

    HYPRE_IJVector newVector;
    ierr = HYPRE_IJVectorCreate(comm_, &newVector, numGlobalRows_);
    ierr = HYPRE_IJVectorSetLocalStorageType(newVector, HYPRE_PARCSR);
    ierr = HYPRE_IJVectorSetLocalPartitioning(newVector,localStartRow_-1,
                                              localEndRow_);
    ierr = HYPRE_IJVectorAssemble(newVector);
    ierr = HYPRE_IJVectorInitialize(newVector);
    ierr = HYPRE_IJVectorZeroLocalComponents(newVector);
    assert(!ierr);

    HYPRE_ParVector Vec1 = 
          (HYPRE_ParVector) HYPRE_IJVectorGetLocalStorage(HYb_);
    HYPRE_ParVector Vec2 = 
          (HYPRE_ParVector) HYPRE_IJVectorGetLocalStorage(newVector);
    HYPRE_ParVectorCopy( Vec1, Vec2);
    if ( scalar != 1.0 ) HYPRE_ParVectorScale( scalar, Vec2);

    data.setTypeName("IJ_Vector");
    data.setDataPtr((void*) Vec2);

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
    {
       printf("%4d : HYPRE_LSC::leaving  copyOutRHSVector.\n",mypid_);
    }
}
#endif 

//***************************************************************************

#ifndef NOFEI
void HYPRE_LinSysCore::sumInRHSVector(double scalar, const Data& data) 
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
    HYPRE_ParVector xVec = 
          (HYPRE_ParVector) HYPRE_IJVectorGetLocalStorage(inVec);
    HYPRE_ParVector yVec = 
          (HYPRE_ParVector) HYPRE_IJVectorGetLocalStorage(HYb_);
 
    hypre_ParVectorAxpy(scalar,(hypre_ParVector*)xVec,(hypre_ParVector*)yVec);
 
    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
    {
       printf("%4d : HYPRE_LSC::leaving  sumInRHSVector.\n",mypid_);
    }
}
#endif 

//***************************************************************************

#ifndef NOFEI
void HYPRE_LinSysCore::destroyMatrixData(Data& data) 
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
}
#endif 

//***************************************************************************

#ifndef NOFEI
void HYPRE_LinSysCore::destroyVectorData(Data& data) 
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
}
#endif 

//***************************************************************************

void HYPRE_LinSysCore::setNumRHSVectors(int numRHSs, const int* rhsIDs) 
{
    int  i;

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
    {
       printf("%4d : HYPRE_LSC::entering setNumRHSVectors.\n",mypid_);
       printf("%4d : HYPRE_LSC::incoming numRHSs = %d\n",mypid_,numRHSs);
       printf("%4d : setNumRHSVectors - hardwired to 1 rhs.\n",mypid_);
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

    if (numRHSs == 0) return;

    delete [] rhsIDs_;
    numRHSs_ = numRHSs;
    rhsIDs_ = new int[numRHSs_];
 
    for ( i = 0; i < numRHSs; i++ ) rhsIDs_[i] = rhsIDs[i];

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
    {
       printf("%4d : HYPRE_LSC::leaving  setNumRHSVectors.\n",mypid_);
    }
}

//***************************************************************************

void HYPRE_LinSysCore::setRHSID(int rhsID) 
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
          return;
       }
    }

    printf("setRHSID ERROR : rhsID not found.\n");
    exit(1);

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
    {
       printf("%4d : HYPRE_LSC::leaving  setRHSID.\n",mypid_);
    }
}

//***************************************************************************
// used for initializing the initial guess
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::putInitialGuess(const int* eqnNumbers,
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

#if defined(FEI_V13) || defined(FEI_V14)
       if (eqnNumbers[i] >= localStartRow_ && eqnNumbers[i] <= localEndRow_)
          local_ind[i] = eqnNumbers[i] - 1;
#else
       if ((eqnNumbers[i]+1) >= localStartRow_ && (eqnNumbers[i]+1) <= localEndRow_)
          local_ind[i] = eqnNumbers[i];
#endif
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

    ierr = HYPRE_IJVectorSetLocalComponents(HYx_,leng,local_ind,NULL,values);
    assert(!ierr);

    delete [] local_ind;

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 0 )
    {
       printf("%4d : HYPRE_LSC::leaving  putInitalGuess.\n",mypid_);
    }
}

//***************************************************************************
// used for getting the solution out of the solver, and into the application
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::getSolution(int* eqnNumbers, double* answers,int leng) 
{
    int    i, ierr, *equations;

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
    {
       printf("%4d : HYPRE_LSC::entering getSolution.\n",mypid_);
    }

    equations = new int[leng];

    for ( i = 0; i < leng; i++ )
    {
       equations[i] = eqnNumbers[i] - 1; // construct 0-based index
       if ( equations[i] < localStartRow_-1 || equations[i] > localEndRow_ )
       {
          printf("%d : getSolution ERROR - index out of range = %d.\n",
                       mypid_, eqnNumbers[i]);
          exit(1);
       }
    }
    ierr = HYPRE_IJVectorGetLocalComponents(HYx_,leng,equations,NULL,answers);
    assert(!ierr);
    delete [] equations;

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
    {
       printf("%4d : HYPRE_LSC::leaving  getSolution.\n",mypid_);
    }
}

//***************************************************************************
// This is a modified function for version 1.5
// used for getting the solution out of the solver, and into the application
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::getSolution(double* answers,int leng) 
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

    ierr = HYPRE_IJVectorGetLocalComponents(HYx_,leng,equations,NULL,answers);
    assert(!ierr);

    delete [] equations;

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
    {
       printf("%4d : HYPRE_LSC::leaving  getSolution.\n",mypid_);
    }
}

//***************************************************************************
// used for getting the solution out of the solver, and into the application
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::getSolnEntry(int eqnNumber, double& answer) 
{
    double val;
    int    ierr, equation;

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
    {
       printf("%4d : HYPRE_LSC::entering getSolnEntry.\n",mypid_);
    }

#if defined(FEI_V13) || defined(FEI_V14)
    equation = eqnNumber - 1; // construct 0-based index
#else
    equation = eqnNumber; // incoming 0-based index
#endif

    if ( equation < localStartRow_-1 && equation > localEndRow_ )
    {
       printf("%d : getSolnEntry ERROR - index out of range = %d.\n", mypid_, 
                    eqnNumber);
       exit(1);
    }

    ierr = HYPRE_IJVectorGetLocalComponents(HYx_,1,&equation,NULL,&val);
    assert(!ierr);
    answer = val;

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
    {
       printf("%4d : HYPRE_LSC::leaving  getSolnEntry.\n",mypid_);
    }
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
    HYPreconReuse_ = 0;
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

#ifdef MLPACK
       else if ( HYPreconID_ == HYML )
          HYPRE_ParCSRMLDestroy( HYPrecon_ );
#endif
    }

    //-------------------------------------------------------------------
    // check for the validity of the preconditioner name
    //-------------------------------------------------------------------

    if ( !strcmp(name, "diagonal"  ) )
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
// solve the linear system
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::formResidual(int* eqnNumbers, double* values, int leng)
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
       printf("                        numLocalRows, inleng = %d %d",nrows,leng);
       return;
    }
    if ( ! systemAssembled_ )
    {
       printf("formResidual ERROR : system not yet assembled.\n");
       exit(1);
    }

    //*******************************************************************
    // fetch matrix and vector pointers
    //-------------------------------------------------------------------

    A_csr  = (HYPRE_ParCSRMatrix) HYPRE_IJMatrixGetLocalStorage(HYA_);
    x_csr  = (HYPRE_ParVector)    HYPRE_IJVectorGetLocalStorage(HYx_);
    b_csr  = (HYPRE_ParVector)    HYPRE_IJVectorGetLocalStorage(HYb_);
    r_csr  = (HYPRE_ParVector)    HYPRE_IJVectorGetLocalStorage(HYr_);

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
       HYPRE_IJVectorGetLocalComponents(HYr_, 1, &i, NULL, &values[index]);
       eqnNumbers[index] = i + 1;
    }

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
    {
       printf("%4d : HYPRE_LSC::leaving  formResidual.\n", mypid_);
    }
}

//***************************************************************************
// This is a new interface for formResidual for version 1.5
//===========================================================================
// form the residual vector
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::formResidual(double* values, int leng)
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
       return;
    }
    if ( ! systemAssembled_ )
    {
       printf("%4d : HYPRE_LSC formResidual ERROR : system not assembled.\n");
       exit(1);
    }

    //*******************************************************************
    // fetch matrix and vector pointers
    //-------------------------------------------------------------------

    A_csr  = (HYPRE_ParCSRMatrix) HYPRE_IJMatrixGetLocalStorage(HYA_);
    x_csr  = (HYPRE_ParVector)    HYPRE_IJVectorGetLocalStorage(HYx_);
    b_csr  = (HYPRE_ParVector)    HYPRE_IJVectorGetLocalStorage(HYb_);
    r_csr  = (HYPRE_ParVector)    HYPRE_IJVectorGetLocalStorage(HYr_);

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
       HYPRE_IJVectorGetLocalComponents(HYr_, 1, &i, NULL, &values[index]);
    }

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
    {
       printf("%4d : HYPRE_LSC::leaving  formResidual.\n", mypid_);
    }
}

//***************************************************************************
// solve the linear system
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::launchSolver(int& solveStatus, int &iterations)
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
    // temporary kludge before FEI adds functions to address this
    //-------------------------------------------------------------------

    MPI_Barrier(comm_);
    rtime1  = MPI_Wtime();

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
    rtime2  = MPI_Wtime();
    
    //*******************************************************************
    // fetch matrix and vector pointers
    //-------------------------------------------------------------------

    A_csr  = (HYPRE_ParCSRMatrix) HYPRE_IJMatrixGetLocalStorage(currA_);
    x_csr  = (HYPRE_ParVector)    HYPRE_IJVectorGetLocalStorage(currX_);
    b_csr  = (HYPRE_ParVector)    HYPRE_IJVectorGetLocalStorage(currB_);
    r_csr  = (HYPRE_ParVector)    HYPRE_IJVectorGetLocalStorage(currR_);

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
          nnz += rowSize;
          HYPRE_ParCSRMatrixRestoreRow(A_csr,i,&rowSize,&colInd,&colVal);
       }
       fprintf(fp, "%6d  %7d \n", nrows, nnz);
       for ( i = startRow; i < startRow+nrows; i++ )
       {
          HYPRE_ParCSRMatrixGetRow(A_csr,i,&rowSize,&colInd,&colVal);
          for (j = 0; j < rowSize; j++)
             if ( colVal[j] != 0.0 )
                fprintf(fp, "%6d  %6d  %e \n", i+1, colInd[j]+1, colVal[j]);
          HYPRE_ParCSRMatrixRestoreRow(A_csr,i,&rowSize,&colInd,&colVal);
       }
       fclose(fp);
       sprintf(fname, "hypre_rhs.out.%d", mypid_);
       fp = fopen( fname, "w");
       fprintf(fp, "%6d \n", nrows);
       for ( i = startRow; i < startRow+nrows; i++ )
       {
          HYPRE_IJVectorGetLocalComponents(currB_, 1, &i, NULL, &ddata);
          fprintf(fp, "%6d  %e \n", i+1, ddata);
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
#if defined(FEI_V13) || defined(FEI_V14)
    status = 0;
#else
    status = 1;
#endif
    stime  = MPI_Wtime();
    ptime  = stime;

    if ( minResProjection_ == 1 )
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

          switch ( HYPreconID_ )
          {
             case HYDIAGONAL :
                  if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
                     printf("Diagonal preconditioning \n");
                  if ( HYPreconReuse_ == 1 )
                  {
                     HYPRE_ParCSRPCGSetPrecond(HYSolver_,
                                    HYPRE_ParCSRDiagScale,
                                    HYPRE_DummyFunction,HYPrecon_);
                  }
                  else
                  {
                     HYPRE_ParCSRPCGSetPrecond(HYSolver_,
                                    HYPRE_ParCSRDiagScale,
                                    HYPRE_ParCSRDiagScaleSetup,HYPrecon_);
                  }
                  break;

             case HYPILUT :
                  if ( mypid_ == 0 )
                     printf("HYPRE_LSI : CG does not work with pilut.\n");
                  exit(1);
                  break;

             case HYDDILUT :
                  if ( mypid_ == 0 )
                     printf("HYPRE_LSI : CG does not work with ddilut.\n");
                  exit(1);
                  break;

             case HYDDICT :
                  if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && 
                        mypid_ == 0 )
                  {
                     printf("DDICT - fillin   = %e\n", ddictFillin_);
                     printf("DDICT - drop tol = %e\n", ddictDropTol_);
                  }
                  HYPRE_LSI_DDICTSetFillin(HYPrecon_,ddictFillin_);
                  HYPRE_LSI_DDICTSetDropTolerance(HYPrecon_,ddictDropTol_);
                  if ( HYOutputLevel_ & HYFEI_DDILUT )
                     HYPRE_LSI_DDICTSetOutputLevel(HYPrecon_,2);

                  if ( HYPreconReuse_ == 1 )
                  {
                     HYPRE_ParCSRPCGSetPrecond(HYSolver_,HYPRE_LSI_DDICTSolve,
                                      HYPRE_DummyFunction, HYPrecon_);
                  }
                  else
                  {
                     HYPRE_ParCSRPCGSetPrecond(HYSolver_,HYPRE_LSI_DDICTSolve,
                                      HYPRE_LSI_DDICTSetup, HYPrecon_);
                  }
                  break;

             case HYSCHWARZ :
                  if ( HYOutputLevel_ & HYFEI_DDILUT )
                     HYPRE_LSI_SchwarzSetOutputLevel(HYPrecon_,2);

                  HYPRE_LSI_SchwarzSetILUTFillin(HYPrecon_,schwarzFillin_);
                  HYPRE_LSI_SchwarzSetNBlocks(HYPrecon_, schwarzNblocks_);
                  HYPRE_LSI_SchwarzSetBlockSize(HYPrecon_, schwarzBlksize_);
                  HYPRE_ParCSRPCGSetPrecond(HYSolver_,HYPRE_LSI_SchwarzSolve,
                                            HYPRE_LSI_SchwarzSetup, HYPrecon_);
                  break;

             case HYPOLY :
                  if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
                     printf("Polynomial preconditioning - order = %d\n", polyOrder_);
                  HYPRE_LSI_PolySetOrder(HYPrecon_, polyOrder_);
                  if ( HYPreconReuse_ == 1 )
                  {
                     HYPRE_ParCSRPCGSetPrecond(HYSolver_,HYPRE_LSI_PolySolve,
                                      HYPRE_DummyFunction, HYPrecon_);
                  }
                  else
                  {
                     HYPRE_ParCSRPCGSetPrecond(HYSolver_,HYPRE_LSI_PolySolve,
                                      HYPRE_LSI_PolySetup, HYPrecon_);
                  }
                  break;

             case HYPARASAILS :
                  if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
                  {
                     printf("ParaSails - nlevels   = %d\n",parasailsNlevels_);
                     printf("ParaSails - threshold = %e\n",parasailsThreshold_);
                     printf("ParaSails - filter    = %e\n",parasailsFilter_);
                     printf("ParaSails - sym       = %d\n",parasailsSym_);
                     printf("ParaSails - loadbal   = %e\n",parasailsLoadbal_);
                  }
                  HYPRE_ParCSRParaSailsSetSym(HYPrecon_,parasailsSym_);
                  HYPRE_ParCSRParaSailsSetParams(HYPrecon_,parasailsThreshold_, 
                                                 parasailsNlevels_);
                  HYPRE_ParCSRParaSailsSetFilter(HYPrecon_,parasailsFilter_);
                  HYPRE_ParCSRParaSailsSetLoadbal(HYPrecon_,parasailsLoadbal_);
                  HYPRE_ParCSRParaSailsSetReuse(HYPrecon_,parasailsReuse_);
                  if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1)
                  {
                     HYPRE_ParCSRParaSailsSetLogging(HYPrecon_, 1);
                  }
                  if ( HYPreconReuse_ == 1 )
                  {
                     HYPRE_ParCSRPCGSetPrecond(HYSolver_,
                                    HYPRE_ParCSRParaSailsSolve,
                                    HYPRE_DummyFunction, HYPrecon_);
                  }
                  else
                  {
                     HYPRE_ParCSRPCGSetPrecond(HYSolver_,
                                    HYPRE_ParCSRParaSailsSolve,
                                    HYPRE_ParCSRParaSailsSetup, HYPrecon_);
                  }
                  break;

             case HYBOOMERAMG :
                  HYPRE_BoomerAMGSetCoarsenType(HYPrecon_, amgCoarsenType_);
                  HYPRE_BoomerAMGSetMeasureType(HYPrecon_, amgMeasureType_);
                  HYPRE_BoomerAMGSetStrongThreshold(HYPrecon_,
                                                    amgStrongThreshold_);
                  num_sweeps = hypre_CTAlloc(int,4);
                  for ( i = 0; i < 4; i++ ) num_sweeps[i] = amgNumSweeps_[i];

                  HYPRE_BoomerAMGSetNumGridSweeps(HYPrecon_, num_sweeps);
                  relax_type = hypre_CTAlloc(int,4);
                  for ( i = 0; i < 4; i++ ) relax_type[i] = amgRelaxType_[i];

                  HYPRE_BoomerAMGSetGridRelaxType(HYPrecon_, relax_type);
                  relax_wt = hypre_CTAlloc(double,25);
                  for ( i = 0; i < 25; i++ ) relax_wt[i] = amgRelaxWeight_[i];
                  HYPRE_BoomerAMGSetRelaxWeight(HYPrecon_, relax_wt);
                  if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
                  {
                     printf("AMG coarsen type = %d\n", amgCoarsenType_);
                     printf("AMG measure type = %d\n", amgMeasureType_);
                     printf("AMG threshold    = %e\n", amgStrongThreshold_);
                     printf("AMG numsweeps    = %d\n", amgNumSweeps_[0]);
                     printf("AMG relax type   = %d\n", amgRelaxType_[0]);
                     printf("AMG relax weight = %e\n", amgRelaxWeight_[0]);
                  }
                  if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2)
                  {
                     HYPRE_BoomerAMGSetDebugFlag(HYPrecon_, 0);
                     HYPRE_BoomerAMGSetIOutDat(HYPrecon_, 3);
                  }
                  if ( HYPreconReuse_ == 1 )
                  {
                     HYPRE_ParCSRPCGSetPrecond(HYSolver_, HYPRE_BoomerAMGSolve,
                                    HYPRE_DummyFunction, HYPrecon_);
                  }
                  else
                  {
                     HYPRE_ParCSRPCGSetPrecond(HYSolver_, HYPRE_BoomerAMGSolve,
                                    HYPRE_BoomerAMGSetup, HYPrecon_);
                  }
                  break;

#ifdef MLPACK
             case HYML :

                  HYPRE_ParCSRMLSetMethod(HYPrecon_,mlMethod_);
                  HYPRE_ParCSRMLSetCoarseSolver(HYPrecon_,mlCoarseSolver_);
                  HYPRE_ParCSRMLSetCoarsenScheme(HYPrecon_,mlCoarsenScheme_);
                  HYPRE_ParCSRMLSetStrongThreshold(HYPrecon_,mlStrongThreshold_);
                  HYPRE_ParCSRMLSetNumPreSmoothings(HYPrecon_,mlNumPreSweeps_);
                  HYPRE_ParCSRMLSetNumPostSmoothings(HYPrecon_,mlNumPostSweeps_);
                  HYPRE_ParCSRMLSetPreSmoother(HYPrecon_,mlPresmootherType_);
                  HYPRE_ParCSRMLSetPostSmoother(HYPrecon_,mlPostsmootherType_);
                  HYPRE_ParCSRMLSetDampingFactor(HYPrecon_,mlRelaxWeight_);
                  if ( HYPreconReuse_ == 1 )
                  {
                     HYPRE_ParCSRPCGSetPrecond(HYSolver_, HYPRE_ParCSRMLSolve,
                                    HYPRE_DummyFunction, HYPrecon_);
                  }
                  else
                  {
                     HYPRE_ParCSRPCGSetPrecond(HYSolver_, HYPRE_ParCSRMLSolve,
                                    HYPRE_ParCSRMLSetup, HYPrecon_);
                  }

                  if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
                  {
                     printf("ML strong threshold = %e\n", mlStrongThreshold_);
                     printf("ML numsweeps(pre)   = %d\n", mlNumPreSweeps_);
                     printf("ML numsweeps(post)  = %d\n", mlNumPostSweeps_);
                     printf("ML smoother (pre)   = %d\n", mlPresmootherType_);
                     printf("ML smoother (post)  = %d\n", mlPostsmootherType_);
                     printf("ML relax weight     = %e\n", mlRelaxWeight_);
                  }
                  break;
#endif
          }

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
          ptime  = MPI_Wtime();
          HYPRE_ParCSRPCGSolve(HYSolver_, A_csr, b_csr, x_csr);
          HYPRE_ParCSRPCGGetNumIterations(HYSolver_, &num_iterations);
          HYPRE_ParVectorCopy( b_csr, r_csr );
          HYPRE_ParCSRMatrixMatvec( -1.0, A_csr, x_csr, 1.0, r_csr );
          if (minResProjection_ == 1) addToProjectionSpace( currX_, currB_ );
          HYPRE_ParVectorInnerProd( r_csr, r_csr, &rnorm);
          rnorm = sqrt( rnorm );
#if defined(FEI_V13) || defined(FEI_V14)
          if ( num_iterations >= maxIterations_ ) status = 0; else status = 1;
#else
          if ( num_iterations >= maxIterations_ ) status = 1; else status = 0;
#endif
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

          switch ( HYPreconID_ )
          {
             case HYDIAGONAL :
                  if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
                     printf("Diagonal preconditioning \n");
                  if ( HYPreconReuse_ == 1 )
                  {
                     HYPRE_ParCSRGMRESSetPrecond(HYSolver_,
                                      HYPRE_ParCSRDiagScale,
                                      HYPRE_DummyFunction, HYPrecon_);
                  }
                  else
                  {
                     HYPRE_ParCSRGMRESSetPrecond(HYSolver_,
                                      HYPRE_ParCSRDiagScale,
                                      HYPRE_ParCSRDiagScaleSetup, HYPrecon_);
                  }
                  break;

             case HYPILUT :
                  if (pilutFillin_ == 0) pilutFillin_ = pilutMaxNnzPerRow_;
                  if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
                  {
                     printf("PILUT - row size = %d\n", pilutFillin_);
                     printf("PILUT - drop tol = %e\n", pilutDropTol_);
                  }
                  HYPRE_ParCSRPilutSetFactorRowSize(HYPrecon_,pilutFillin_);
                  HYPRE_ParCSRPilutSetDropTolerance(HYPrecon_,pilutDropTol_);
                  if ( HYPreconReuse_ == 1 )
                  {
                     HYPRE_ParCSRGMRESSetPrecond(HYSolver_,
                                      HYPRE_ParCSRPilutSolve,
                                      HYPRE_DummyFunction, HYPrecon_);
                  }
                  else
                  {
                     HYPRE_ParCSRGMRESSetPrecond(HYSolver_,
                                      HYPRE_ParCSRPilutSolve,
                                      HYPRE_ParCSRPilutSetup, HYPrecon_);
                  }
                  break;

             case HYDDILUT :
                  if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && 
                        mypid_ == 0 )
                  {
                     printf("DDILUT - fillin   = %e\n", ddilutFillin_);
                     printf("DDILUT - drop tol = %e\n", ddilutDropTol_);
                  }
                  HYPRE_LSI_DDIlutSetFillin(HYPrecon_,ddilutFillin_);
                  HYPRE_LSI_DDIlutSetDropTolerance(HYPrecon_,ddilutDropTol_);
                  if ( HYOutputLevel_ & HYFEI_DDILUT )
                     HYPRE_LSI_DDIlutSetOutputLevel(HYPrecon_,2);

                  if ( HYPreconReuse_ == 1 )
                  {
                     HYPRE_ParCSRGMRESSetPrecond(HYSolver_,HYPRE_LSI_DDIlutSolve,
                                      HYPRE_DummyFunction, HYPrecon_);
                  }
                  else
                  {
                     HYPRE_ParCSRGMRESSetPrecond(HYSolver_,HYPRE_LSI_DDIlutSolve,
                                         HYPRE_LSI_DDIlutSetup, HYPrecon_);
                  }
                  break;

             case HYDDICT :
                  if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && 
                        mypid_ == 0 )
                  {
                     printf("DDICT - fillin   = %e\n", ddictFillin_);
                     printf("DDICT - drop tol = %e\n", ddictDropTol_);
                  }
                  HYPRE_LSI_DDICTSetFillin(HYPrecon_,ddictFillin_);
                  HYPRE_LSI_DDICTSetDropTolerance(HYPrecon_,ddictDropTol_);
                  if ( HYOutputLevel_ & HYFEI_DDILUT )
                     HYPRE_LSI_DDICTSetOutputLevel(HYPrecon_,2);

                  if ( HYPreconReuse_ == 1 )
                  {
                     HYPRE_ParCSRGMRESSetPrecond(HYSolver_,HYPRE_LSI_DDICTSolve,
                                      HYPRE_DummyFunction, HYPrecon_);
                  }
                  else
                  {
                     HYPRE_ParCSRGMRESSetPrecond(HYSolver_,HYPRE_LSI_DDICTSolve,
                                      HYPRE_LSI_DDICTSetup, HYPrecon_);
                  }
                  break;

             case HYSCHWARZ :
                  if ( HYOutputLevel_ & HYFEI_DDILUT )
                     HYPRE_LSI_SchwarzSetOutputLevel(HYPrecon_,2);

                  HYPRE_LSI_SchwarzSetILUTFillin(HYPrecon_,schwarzFillin_);
                  HYPRE_LSI_SchwarzSetNBlocks(HYPrecon_, schwarzNblocks_);
                  HYPRE_LSI_SchwarzSetBlockSize(HYPrecon_, schwarzBlksize_);
                  HYPRE_ParCSRGMRESSetPrecond(HYSolver_,HYPRE_LSI_SchwarzSolve,
                                              HYPRE_LSI_SchwarzSetup, HYPrecon_);
                  break;

             case HYPOLY :
                  if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
                     printf("Polynomial preconditioning - order = %d\n", polyOrder_);
                  HYPRE_LSI_PolySetOrder(HYPrecon_, polyOrder_);
                  if ( HYPreconReuse_ == 1 )
                  {
                     HYPRE_ParCSRGMRESSetPrecond(HYSolver_,HYPRE_LSI_PolySolve,
                                      HYPRE_DummyFunction, HYPrecon_);
                  }
                  else
                  {
                     HYPRE_ParCSRGMRESSetPrecond(HYSolver_,HYPRE_LSI_PolySolve,
                                      HYPRE_LSI_PolySetup, HYPrecon_);
                  }
                  break;

             case HYPARASAILS :
                  if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
                  {
                     printf("ParaSails - nlevels   = %d\n",parasailsNlevels_);
                     printf("ParaSails - threshold = %e\n",parasailsThreshold_);
                     printf("ParaSails - filter    = %e\n",parasailsFilter_);
                     printf("ParaSails - sym       = %d\n",parasailsSym_);
                     printf("ParaSails - loadbal   = %e\n",parasailsLoadbal_);
                  }
                  HYPRE_ParCSRParaSailsSetSym(HYPrecon_,parasailsSym_);
                  HYPRE_ParCSRParaSailsSetParams(HYPrecon_,parasailsThreshold_,
                                                 parasailsNlevels_);
                  HYPRE_ParCSRParaSailsSetFilter(HYPrecon_,parasailsFilter_);
                  HYPRE_ParCSRParaSailsSetLoadbal(HYPrecon_,parasailsLoadbal_);
                  HYPRE_ParCSRParaSailsSetReuse(HYPrecon_,parasailsReuse_);
                  if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1)
                  {
                     HYPRE_ParCSRParaSailsSetLogging(HYPrecon_, 1);
                  }

                  if ( HYPreconReuse_ == 1 )
                  {
                     HYPRE_ParCSRGMRESSetPrecond(HYSolver_,
                                      HYPRE_ParCSRParaSailsSolve,
                                      HYPRE_DummyFunction, HYPrecon_);
                  }
                  else
                  {
                     HYPRE_ParCSRGMRESSetPrecond(HYSolver_,
                                      HYPRE_ParCSRParaSailsSolve,
                                      HYPRE_ParCSRParaSailsSetup, HYPrecon_);
                  }
                  break;

             case HYBOOMERAMG :
                  HYPRE_BoomerAMGSetCoarsenType(HYPrecon_, amgCoarsenType_);
                  HYPRE_BoomerAMGSetMeasureType(HYPrecon_, amgMeasureType_);
                  HYPRE_BoomerAMGSetStrongThreshold(HYPrecon_,
                                                    amgStrongThreshold_);
                  num_sweeps = hypre_CTAlloc(int,4);
                  for ( i = 0; i < 4; i++ ) num_sweeps[i] = amgNumSweeps_[i];

                  HYPRE_BoomerAMGSetNumGridSweeps(HYPrecon_, num_sweeps);
                  relax_type = hypre_CTAlloc(int,4);
                  for ( i = 0; i < 4; i++ ) relax_type[i] = amgRelaxType_[i];

                  HYPRE_BoomerAMGSetGridRelaxType(HYPrecon_, relax_type);
                  relax_wt = hypre_CTAlloc(double,25);
                  for ( i = 0; i < 25; i++ ) relax_wt[i] = amgRelaxWeight_[i];
                  HYPRE_BoomerAMGSetRelaxWeight(HYPrecon_, relax_wt);
                  if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
                  {
                     printf("AMG coarsen type = %d\n", amgCoarsenType_);
                     printf("AMG measure type = %d\n", amgMeasureType_);
                     printf("AMG threshold    = %e\n", amgStrongThreshold_);
                     printf("AMG numsweeps    = %d\n", amgNumSweeps_[0]);
                     printf("AMG relax type   = %d\n", amgRelaxType_[0]);
                     printf("AMG relax weight = %e\n", amgRelaxWeight_[0]);
                  }
                  if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2)
                  {
                     HYPRE_BoomerAMGSetDebugFlag(HYPrecon_, 0);
                     HYPRE_BoomerAMGSetIOutDat(HYPrecon_, 3);
                  }
                  if ( HYPreconReuse_ == 1 )
                  {
                     HYPRE_ParCSRGMRESSetPrecond(HYSolver_,HYPRE_BoomerAMGSolve,
                                      HYPRE_DummyFunction, HYPrecon_);
                  }
                  else
                  {
                     HYPRE_ParCSRGMRESSetPrecond(HYSolver_, HYPRE_BoomerAMGSolve,
                                      HYPRE_BoomerAMGSetup, HYPrecon_);
                  }
                  break;

#ifdef MLPACK
             case HYML :

                  HYPRE_ParCSRMLSetMethod(HYPrecon_,mlMethod_);
                  HYPRE_ParCSRMLSetCoarseSolver(HYPrecon_,mlCoarseSolver_);
                  HYPRE_ParCSRMLSetCoarsenScheme(HYPrecon_,mlCoarsenScheme_);
                  HYPRE_ParCSRMLSetStrongThreshold(HYPrecon_,mlStrongThreshold_);
                  HYPRE_ParCSRMLSetNumPreSmoothings(HYPrecon_,mlNumPreSweeps_);
                  HYPRE_ParCSRMLSetNumPostSmoothings(HYPrecon_,mlNumPostSweeps_);
                  HYPRE_ParCSRMLSetPreSmoother(HYPrecon_,mlPresmootherType_);
                  HYPRE_ParCSRMLSetPostSmoother(HYPrecon_,mlPostsmootherType_);
                  HYPRE_ParCSRMLSetDampingFactor(HYPrecon_, mlRelaxWeight_);
                  if ( HYPreconReuse_ == 1 )
                  {
                     HYPRE_ParCSRGMRESSetPrecond(HYSolver_,HYPRE_ParCSRMLSolve,
                                      HYPRE_DummyFunction, HYPrecon_);
                  }
                  else
                  {
                     HYPRE_ParCSRGMRESSetPrecond(HYSolver_,HYPRE_ParCSRMLSolve,
                                      HYPRE_ParCSRMLSetup, HYPrecon_);
                  }

                  if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
                  {
                     printf("ML strong threshold = %e\n", mlStrongThreshold_);
                     printf("ML numsweeps(pre)   = %d\n", mlNumPreSweeps_);
                     printf("ML numsweeps(post)  = %d\n", mlNumPostSweeps_);
                     printf("ML smoother (pre)   = %d\n", mlPresmootherType_);
                     printf("ML smoother (post)  = %d\n", mlPostsmootherType_);
                     printf("ML relax weight     = %e\n", mlRelaxWeight_);
                  }
                  break;
#endif
          }

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
          ptime  = MPI_Wtime();
          HYPRE_ParCSRGMRESSolve(HYSolver_, A_csr, b_csr, x_csr);
          HYPRE_ParCSRGMRESGetNumIterations(HYSolver_, &num_iterations);
          HYPRE_ParVectorCopy( b_csr, r_csr );
          HYPRE_ParCSRMatrixMatvec( -1.0, A_csr, x_csr, 1.0, r_csr );
          if (minResProjection_ == 1) addToProjectionSpace( currX_, currB_ );
          HYPRE_ParVectorInnerProd( r_csr, r_csr, &rnorm);
          rnorm = sqrt( rnorm );
#if defined(FEI_V13) || defined(FEI_V14)
          if ( num_iterations >= maxIterations_ ) status = 0; else status = 1;
#else
          if ( num_iterations >= maxIterations_ ) status = 1; else status = 0;
#endif
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

          switch ( HYPreconID_ )
          {
             case HYDIAGONAL :
                  if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
                     printf("Diagonal preconditioning \n");
                  if ( HYPreconReuse_ == 1 )
                  {
                     HYPRE_ParCSRBiCGSTABSetPrecond(HYSolver_,
                                      HYPRE_ParCSRDiagScale,
                                      HYPRE_DummyFunction, HYPrecon_);
                  }
                  else
                  {
                     HYPRE_ParCSRBiCGSTABSetPrecond(HYSolver_,
                                      HYPRE_ParCSRDiagScale,
                                      HYPRE_ParCSRDiagScaleSetup, HYPrecon_);
                  }
                  break;

             case HYPILUT :
                  if (pilutFillin_ == 0) pilutFillin_ = pilutMaxNnzPerRow_;
                  if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
                  {
                     printf("PILUT - row size = %d\n", pilutFillin_);
                     printf("PILUT - drop tol = %e\n", pilutDropTol_);
                  }
                  HYPRE_ParCSRPilutSetFactorRowSize(HYPrecon_,pilutFillin_);
                  HYPRE_ParCSRPilutSetDropTolerance(HYPrecon_,pilutDropTol_);
                  if ( HYPreconReuse_ == 1 )
                  {
                     HYPRE_ParCSRBiCGSTABSetPrecond(HYSolver_,
                                      HYPRE_ParCSRPilutSolve,
                                      HYPRE_DummyFunction, HYPrecon_);
                  }
                  else
                  {
                     HYPRE_ParCSRBiCGSTABSetPrecond(HYSolver_,
                                      HYPRE_ParCSRPilutSolve,
                                      HYPRE_ParCSRPilutSetup, HYPrecon_);
                  }
                  break;

             case HYDDILUT :
                  if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && 
                        mypid_ == 0 )
                  {
                     printf("DDILUT - fillin   = %e\n", ddilutFillin_);
                     printf("DDILUT - drop tol = %e\n", ddilutDropTol_);
                  }
                  HYPRE_LSI_DDIlutSetFillin(HYPrecon_,ddilutFillin_);
                  HYPRE_LSI_DDIlutSetDropTolerance(HYPrecon_,ddilutDropTol_);
                  if ( HYOutputLevel_ & HYFEI_DDILUT )
                     HYPRE_LSI_DDIlutSetOutputLevel(HYPrecon_,2);

                  if ( HYPreconReuse_ == 1 )
                  {
                     HYPRE_ParCSRBiCGSTABSetPrecond(HYSolver_,
                         HYPRE_LSI_DDIlutSolve,HYPRE_DummyFunction,HYPrecon_);
                  }
                  else
                  {
                     HYPRE_ParCSRBiCGSTABSetPrecond(HYSolver_,
                         HYPRE_LSI_DDIlutSolve,HYPRE_LSI_DDIlutSetup,HYPrecon_);
                  }
                  break;

             case HYDDICT :
                  if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && 
                        mypid_ == 0 )
                  {
                     printf("DDICT - fillin   = %e\n", ddictFillin_);
                     printf("DDICT - drop tol = %e\n", ddictDropTol_);
                  }
                  HYPRE_LSI_DDICTSetFillin(HYPrecon_,ddictFillin_);
                  HYPRE_LSI_DDICTSetDropTolerance(HYPrecon_,ddictDropTol_);
                  if ( HYOutputLevel_ & HYFEI_DDILUT )
                     HYPRE_LSI_DDICTSetOutputLevel(HYPrecon_,2);

                  if ( HYPreconReuse_ == 1 )
                  {
                     HYPRE_ParCSRBiCGSTABSetPrecond(HYSolver_,HYPRE_LSI_DDICTSolve,
                                      HYPRE_DummyFunction, HYPrecon_);
                  }
                  else
                  {
                     HYPRE_ParCSRBiCGSTABSetPrecond(HYSolver_,HYPRE_LSI_DDICTSolve,
                                      HYPRE_LSI_DDICTSetup, HYPrecon_);
                  }
                  break;

             case HYSCHWARZ :
                  if ( HYOutputLevel_ & HYFEI_DDILUT )
                     HYPRE_LSI_SchwarzSetOutputLevel(HYPrecon_,2);

                  HYPRE_LSI_SchwarzSetILUTFillin(HYPrecon_,schwarzFillin_);
                  HYPRE_LSI_SchwarzSetNBlocks(HYPrecon_, schwarzNblocks_);
                  HYPRE_LSI_SchwarzSetBlockSize(HYPrecon_, schwarzBlksize_);
                  HYPRE_ParCSRBiCGSTABSetPrecond(HYSolver_,HYPRE_LSI_SchwarzSolve,
                                              HYPRE_LSI_SchwarzSetup, HYPrecon_);
                  break;

             case HYPOLY :
                  if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
                     printf("Polynomial preconditioning - order = %d\n", polyOrder_);
                  HYPRE_LSI_PolySetOrder(HYPrecon_, polyOrder_);
                  if ( HYPreconReuse_ == 1 )
                  {
                     HYPRE_ParCSRBiCGSTABSetPrecond(HYSolver_,HYPRE_LSI_PolySolve,
                                      HYPRE_DummyFunction, HYPrecon_);
                  }
                  else
                  {
                     HYPRE_ParCSRBiCGSTABSetPrecond(HYSolver_,HYPRE_LSI_PolySolve,
                                      HYPRE_LSI_PolySetup, HYPrecon_);
                  }
                  break;

             case HYPARASAILS :
                  if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
                  {
                     printf("ParaSails - nlevels   = %d\n",parasailsNlevels_);
                     printf("ParaSails - threshold = %e\n",parasailsThreshold_);
                     printf("ParaSails - filter    = %e\n",parasailsFilter_);
                     printf("ParaSails - sym       = %d\n",parasailsSym_);
                     printf("ParaSails - loadbal   = %e\n",parasailsLoadbal_);
                  }
                  HYPRE_ParCSRParaSailsSetSym(HYPrecon_,parasailsSym_);
                  HYPRE_ParCSRParaSailsSetParams(HYPrecon_,parasailsThreshold_,
                                                 parasailsNlevels_);
                  HYPRE_ParCSRParaSailsSetFilter(HYPrecon_,parasailsFilter_);
                  HYPRE_ParCSRParaSailsSetLoadbal(HYPrecon_,parasailsLoadbal_);
                  HYPRE_ParCSRParaSailsSetReuse(HYPrecon_,parasailsReuse_);
                  if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1)
                  {
                     HYPRE_ParCSRParaSailsSetLogging(HYPrecon_, 1);
                  }

                  if ( HYPreconReuse_ == 1 )
                  {
                     HYPRE_ParCSRBiCGSTABSetPrecond(HYSolver_,
                                      HYPRE_ParCSRParaSailsSolve,
                                      HYPRE_DummyFunction, HYPrecon_);
                  }
                  else
                  {
                     HYPRE_ParCSRBiCGSTABSetPrecond(HYSolver_,
                                      HYPRE_ParCSRParaSailsSolve,
                                      HYPRE_ParCSRParaSailsSetup, HYPrecon_);
                  }
                  break;

             case HYBOOMERAMG :
                  HYPRE_BoomerAMGSetCoarsenType(HYPrecon_, amgCoarsenType_);
                  HYPRE_BoomerAMGSetMeasureType(HYPrecon_, amgMeasureType_);
                  HYPRE_BoomerAMGSetStrongThreshold(HYPrecon_,
                                                    amgStrongThreshold_);
                  num_sweeps = hypre_CTAlloc(int,4);
                  for ( i = 0; i < 4; i++ ) num_sweeps[i] = amgNumSweeps_[i];

                  HYPRE_BoomerAMGSetNumGridSweeps(HYPrecon_, num_sweeps);
                  relax_type = hypre_CTAlloc(int,4);
                  for ( i = 0; i < 4; i++ ) relax_type[i] = amgRelaxType_[i];

                  HYPRE_BoomerAMGSetGridRelaxType(HYPrecon_, relax_type);
                  relax_wt = hypre_CTAlloc(double,25);
                  for ( i = 0; i < 25; i++ ) relax_wt[i] = amgRelaxWeight_[i];
                  HYPRE_BoomerAMGSetRelaxWeight(HYPrecon_, relax_wt);
                  if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
                  {
                     printf("AMG coarsen type = %d\n", amgCoarsenType_);
                     printf("AMG measure type = %d\n", amgMeasureType_);
                     printf("AMG threshold    = %e\n", amgStrongThreshold_);
                     printf("AMG numsweeps    = %d\n", amgNumSweeps_[0]);
                     printf("AMG relax type   = %d\n", amgRelaxType_[0]);
                     printf("AMG relax weight = %e\n", amgRelaxWeight_[0]);
                  }
                  if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2)
                     HYPRE_BoomerAMGSetIOutDat(HYPrecon_, 3);

                  if ( HYPreconReuse_ == 1 )
                  {
                     HYPRE_ParCSRBiCGSTABSetPrecond(HYSolver_,
                         HYPRE_BoomerAMGSolve, HYPRE_DummyFunction, HYPrecon_);
                  }
                  else
                  {
                     HYPRE_ParCSRBiCGSTABSetPrecond(HYSolver_, 
                         HYPRE_BoomerAMGSolve, HYPRE_BoomerAMGSetup, HYPrecon_);
                  }
                  break;

#ifdef MLPACK
             case HYML :

                  HYPRE_ParCSRMLSetMethod(HYPrecon_,mlMethod_);
                  HYPRE_ParCSRMLSetCoarseSolver(HYPrecon_,mlCoarseSolver_);
                  HYPRE_ParCSRMLSetCoarsenScheme(HYPrecon_,mlCoarsenScheme_);
                  HYPRE_ParCSRMLSetStrongThreshold(HYPrecon_,mlStrongThreshold_);
                  HYPRE_ParCSRMLSetNumPreSmoothings(HYPrecon_,mlNumPreSweeps_);
                  HYPRE_ParCSRMLSetNumPostSmoothings(HYPrecon_,mlNumPostSweeps_);
                  HYPRE_ParCSRMLSetPreSmoother(HYPrecon_,mlPresmootherType_);
                  HYPRE_ParCSRMLSetPostSmoother(HYPrecon_,mlPostsmootherType_);
                  HYPRE_ParCSRMLSetDampingFactor(HYPrecon_, mlRelaxWeight_);
                  if ( HYPreconReuse_ == 1 )
                  {
                     HYPRE_ParCSRBiCGSTABSetPrecond(HYSolver_,
                         HYPRE_ParCSRMLSolve, HYPRE_DummyFunction, HYPrecon_);
                  }
                  else
                  {
                     HYPRE_ParCSRBiCGSTABSetPrecond(HYSolver_,
                         HYPRE_ParCSRMLSolve, HYPRE_ParCSRMLSetup, HYPrecon_);
                  }

                  if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
                  {
                     printf("ML strong threshold = %e\n", mlStrongThreshold_);
                     printf("ML numsweeps(pre)   = %d\n", mlNumPreSweeps_);
                     printf("ML numsweeps(post)  = %d\n", mlNumPostSweeps_);
                     printf("ML smoother (pre)   = %d\n", mlPresmootherType_);
                     printf("ML smoother (post)  = %d\n", mlPostsmootherType_);
                     printf("ML relax weight     = %e\n", mlRelaxWeight_);
                  }
                  break;
#endif
          }

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
          ptime  = MPI_Wtime();
          HYPRE_ParCSRBiCGSTABSolve(HYSolver_, A_csr, b_csr, x_csr);
          HYPRE_ParCSRBiCGSTABGetNumIterations(HYSolver_, &num_iterations);
          HYPRE_ParVectorCopy( b_csr, r_csr );
          HYPRE_ParCSRMatrixMatvec( -1.0, A_csr, x_csr, 1.0, r_csr );
          if (minResProjection_ == 1) addToProjectionSpace( currX_, currB_ );
          HYPRE_ParVectorInnerProd( r_csr, r_csr, &rnorm);
          rnorm = sqrt( rnorm );
#if defined(FEI_V13) || defined(FEI_V14)
          if ( num_iterations >= maxIterations_ ) status = 0; else status = 1;
#else
          if ( num_iterations >= maxIterations_ ) status = 1; else status = 0;
#endif
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

          switch ( HYPreconID_ )
          {
             case HYDIAGONAL :
                  if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
                     printf("Diagonal preconditioning \n");
                  if ( HYPreconReuse_ == 1 )
                  {
                     HYPRE_ParCSRBiCGSTABLSetPrecond(HYSolver_,
                                      HYPRE_ParCSRDiagScale,
                                      HYPRE_DummyFunction, HYPrecon_);
                  }
                  else
                  {
                     HYPRE_ParCSRBiCGSTABLSetPrecond(HYSolver_,
                                      HYPRE_ParCSRDiagScale,
                                      HYPRE_ParCSRDiagScaleSetup, HYPrecon_);
                  }
                  break;

             case HYPILUT :
                  if (pilutFillin_ == 0) pilutFillin_ = pilutMaxNnzPerRow_;
                  if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
                  {
                     printf("PILUT - row size = %d\n", pilutFillin_);
                     printf("PILUT - drop tol = %e\n", pilutDropTol_);
                  }
                  HYPRE_ParCSRPilutSetFactorRowSize(HYPrecon_,pilutFillin_);
                  HYPRE_ParCSRPilutSetDropTolerance(HYPrecon_,pilutDropTol_);
                  if ( HYPreconReuse_ == 1 )
                  {
                     HYPRE_ParCSRBiCGSTABLSetPrecond(HYSolver_,
                                      HYPRE_ParCSRPilutSolve,
                                      HYPRE_DummyFunction, HYPrecon_);
                  }
                  else
                  {
                     HYPRE_ParCSRBiCGSTABLSetPrecond(HYSolver_,
                                      HYPRE_ParCSRPilutSolve,
                                      HYPRE_ParCSRPilutSetup, HYPrecon_);
                  }
                  break;

             case HYDDILUT :
                  if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && 
                        mypid_ == 0 )
                  {
                     printf("DDILUT - fillin   = %e\n", ddilutFillin_);
                     printf("DDILUT - drop tol = %e\n", ddilutDropTol_);
                  }
                  HYPRE_LSI_DDIlutSetFillin(HYPrecon_,ddilutFillin_);
                  HYPRE_LSI_DDIlutSetDropTolerance(HYPrecon_,ddilutDropTol_);
                  if ( HYOutputLevel_ & HYFEI_DDILUT )
                     HYPRE_LSI_DDIlutSetOutputLevel(HYPrecon_,2);

                  if ( HYPreconReuse_ == 1 )
                  {
                     HYPRE_ParCSRBiCGSTABLSetPrecond(HYSolver_,
                         HYPRE_LSI_DDIlutSolve,HYPRE_DummyFunction,HYPrecon_);
                  }
                  else
                  {
                     HYPRE_ParCSRBiCGSTABLSetPrecond(HYSolver_,
                         HYPRE_LSI_DDIlutSolve,HYPRE_LSI_DDIlutSetup,HYPrecon_);
                  }
                  break;

             case HYDDICT :
                  if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && 
                        mypid_ == 0 )
                  {
                     printf("DDICT - fillin   = %e\n", ddictFillin_);
                     printf("DDICT - drop tol = %e\n", ddictDropTol_);
                  }
                  HYPRE_LSI_DDICTSetFillin(HYPrecon_,ddictFillin_);
                  HYPRE_LSI_DDICTSetDropTolerance(HYPrecon_,ddictDropTol_);
                  if ( HYOutputLevel_ & HYFEI_DDILUT )
                     HYPRE_LSI_DDICTSetOutputLevel(HYPrecon_,2);

                  if ( HYPreconReuse_ == 1 )
                  {
                     HYPRE_ParCSRBiCGSTABLSetPrecond(HYSolver_,HYPRE_LSI_DDICTSolve,
                                      HYPRE_DummyFunction, HYPrecon_);
                  }
                  else
                  {
                     HYPRE_ParCSRBiCGSTABLSetPrecond(HYSolver_,HYPRE_LSI_DDICTSolve,
                                      HYPRE_LSI_DDICTSetup, HYPrecon_);
                  }
                  break;

             case HYSCHWARZ :
                  if ( HYOutputLevel_ & HYFEI_DDILUT )
                     HYPRE_LSI_SchwarzSetOutputLevel(HYPrecon_,2);

                  HYPRE_LSI_SchwarzSetILUTFillin(HYPrecon_,schwarzFillin_);
                  HYPRE_LSI_SchwarzSetNBlocks(HYPrecon_, schwarzNblocks_);
                  HYPRE_LSI_SchwarzSetBlockSize(HYPrecon_, schwarzBlksize_);
                  HYPRE_ParCSRBiCGSTABLSetPrecond(HYSolver_,HYPRE_LSI_SchwarzSolve,
                                              HYPRE_LSI_SchwarzSetup, HYPrecon_);
                  break;

             case HYPOLY :
                  if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
                     printf("Polynomial preconditioning - order = %d\n", polyOrder_);
                  HYPRE_LSI_PolySetOrder(HYPrecon_, polyOrder_);
                  if ( HYPreconReuse_ == 1 )
                  {
                     HYPRE_ParCSRBiCGSTABLSetPrecond(HYSolver_,HYPRE_LSI_PolySolve,
                                      HYPRE_DummyFunction, HYPrecon_);
                  }
                  else
                  {
                     HYPRE_ParCSRBiCGSTABLSetPrecond(HYSolver_,HYPRE_LSI_PolySolve,
                                      HYPRE_LSI_PolySetup, HYPrecon_);
                  }
                  break;

             case HYPARASAILS :
                  if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
                  {
                     printf("ParaSails - nlevels   = %d\n",parasailsNlevels_);
                     printf("ParaSails - threshold = %e\n",parasailsThreshold_);
                     printf("ParaSails - filter    = %e\n",parasailsFilter_);
                     printf("ParaSails - sym       = %d\n",parasailsSym_);
                     printf("ParaSails - loadbal   = %e\n",parasailsLoadbal_);
                  }
                  HYPRE_ParCSRParaSailsSetSym(HYPrecon_,parasailsSym_);
                  HYPRE_ParCSRParaSailsSetParams(HYPrecon_,parasailsThreshold_,
                                                 parasailsNlevels_);
                  HYPRE_ParCSRParaSailsSetFilter(HYPrecon_,parasailsFilter_);
                  HYPRE_ParCSRParaSailsSetLoadbal(HYPrecon_,parasailsLoadbal_);
                  HYPRE_ParCSRParaSailsSetReuse(HYPrecon_,parasailsReuse_);
                  if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1)
                  {
                     HYPRE_ParCSRParaSailsSetLogging(HYPrecon_, 1);
                  }

                  if ( HYPreconReuse_ == 1 )
                  {
                     HYPRE_ParCSRBiCGSTABLSetPrecond(HYSolver_,
                                      HYPRE_ParCSRParaSailsSolve,
                                      HYPRE_DummyFunction, HYPrecon_);
                  }
                  else
                  {
                     HYPRE_ParCSRBiCGSTABLSetPrecond(HYSolver_,
                                      HYPRE_ParCSRParaSailsSolve,
                                      HYPRE_ParCSRParaSailsSetup, HYPrecon_);
                  }
                  break;

             case HYBOOMERAMG :
                  HYPRE_BoomerAMGSetCoarsenType(HYPrecon_, amgCoarsenType_);
                  HYPRE_BoomerAMGSetMeasureType(HYPrecon_, amgMeasureType_);
                  HYPRE_BoomerAMGSetStrongThreshold(HYPrecon_,
                                                    amgStrongThreshold_);
                  num_sweeps = hypre_CTAlloc(int,4);
                  for ( i = 0; i < 4; i++ ) num_sweeps[i] = amgNumSweeps_[i];

                  HYPRE_BoomerAMGSetNumGridSweeps(HYPrecon_, num_sweeps);
                  relax_type = hypre_CTAlloc(int,4);
                  for ( i = 0; i < 4; i++ ) relax_type[i] = amgRelaxType_[i];

                  HYPRE_BoomerAMGSetGridRelaxType(HYPrecon_, relax_type);
                  relax_wt = hypre_CTAlloc(double,25);
                  for ( i = 0; i < 25; i++ ) relax_wt[i] = amgRelaxWeight_[i];
                  HYPRE_BoomerAMGSetRelaxWeight(HYPrecon_, relax_wt);
                  if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
                  {
                     printf("AMG coarsen type = %d\n", amgCoarsenType_);
                     printf("AMG measure type = %d\n", amgMeasureType_);
                     printf("AMG threshold    = %e\n", amgStrongThreshold_);
                     printf("AMG numsweeps    = %d\n", amgNumSweeps_[0]);
                     printf("AMG relax type   = %d\n", amgRelaxType_[0]);
                     printf("AMG relax weight = %e\n", amgRelaxWeight_[0]);
                  }
                  if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2)
                     HYPRE_BoomerAMGSetIOutDat(HYPrecon_, 2);

                  if ( HYPreconReuse_ == 1 )
                  {
                     HYPRE_ParCSRBiCGSTABLSetPrecond(HYSolver_,
                         HYPRE_BoomerAMGSolve, HYPRE_DummyFunction, HYPrecon_);
                  }
                  else
                  {
                     HYPRE_ParCSRBiCGSTABLSetPrecond(HYSolver_, 
                         HYPRE_BoomerAMGSolve, HYPRE_BoomerAMGSetup, HYPrecon_);
                  }
                  break;

#ifdef MLPACK
             case HYML :

                  HYPRE_ParCSRMLSetMethod(HYPrecon_,mlMethod_);
                  HYPRE_ParCSRMLSetCoarseSolver(HYPrecon_,mlCoarseSolver_);
                  HYPRE_ParCSRMLSetCoarsenScheme(HYPrecon_,mlCoarsenScheme_);
                  HYPRE_ParCSRMLSetStrongThreshold(HYPrecon_,mlStrongThreshold_);
                  HYPRE_ParCSRMLSetNumPreSmoothings(HYPrecon_,mlNumPreSweeps_);
                  HYPRE_ParCSRMLSetNumPostSmoothings(HYPrecon_,mlNumPostSweeps_);
                  HYPRE_ParCSRMLSetPreSmoother(HYPrecon_,mlPresmootherType_);
                  HYPRE_ParCSRMLSetPostSmoother(HYPrecon_,mlPostsmootherType_);
                  HYPRE_ParCSRMLSetDampingFactor(HYPrecon_, mlRelaxWeight_);
                  if ( HYPreconReuse_ == 1 )
                  {
                     HYPRE_ParCSRBiCGSTABLSetPrecond(HYSolver_,
                         HYPRE_ParCSRMLSolve, HYPRE_DummyFunction, HYPrecon_);
                  }
                  else
                  {
                     HYPRE_ParCSRBiCGSTABLSetPrecond(HYSolver_,
                         HYPRE_ParCSRMLSolve, HYPRE_ParCSRMLSetup, HYPrecon_);
                  }

                  if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
                  {
                     printf("ML strong threshold = %e\n", mlStrongThreshold_);
                     printf("ML numsweeps(pre)   = %d\n", mlNumPreSweeps_);
                     printf("ML numsweeps(post)  = %d\n", mlNumPostSweeps_);
                     printf("ML smoother (pre)   = %d\n", mlPresmootherType_);
                     printf("ML smoother (post)  = %d\n", mlPostsmootherType_);
                     printf("ML relax weight     = %e\n", mlRelaxWeight_);
                  }
                  break;
#endif
          }

          HYPRE_ParCSRBiCGSTABLSetMaxIter(HYSolver_, maxIterations_);
          HYPRE_ParCSRBiCGSTABLSetTol(HYSolver_, tolerance_);
          if ( normAbsRel_ == 0 ) HYPRE_ParCSRBiCGSTABLSetStopCrit(HYSolver_,0);
          else                    HYPRE_ParCSRBiCGSTABLSetStopCrit(HYSolver_,1);
          if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 )
          {
             if ( mypid_ == 0 )
                printf("***************************************************\n");
             HYPRE_ParCSRBiCGSTABLSetLogging(HYSolver_, 1);
          }
          HYPRE_ParCSRBiCGSTABLSetup(HYSolver_, A_csr, b_csr, x_csr);
          MPI_Barrier( comm_ );
          ptime  = MPI_Wtime();
          HYPRE_ParCSRBiCGSTABLSolve(HYSolver_, A_csr, b_csr, x_csr);
          HYPRE_ParCSRBiCGSTABLGetNumIterations(HYSolver_, &num_iterations);
          HYPRE_ParVectorCopy( b_csr, r_csr );
          HYPRE_ParCSRMatrixMatvec( -1.0, A_csr, x_csr, 1.0, r_csr );
          if (minResProjection_ == 1) addToProjectionSpace( currX_, currB_ );
          HYPRE_ParVectorInnerProd( r_csr, r_csr, &rnorm);
          rnorm = sqrt( rnorm );
#if defined(FEI_V13) || defined(FEI_V14)
          if ( num_iterations >= maxIterations_ ) status = 0; else status = 1;
#else
          if ( num_iterations >= maxIterations_ ) status = 1; else status = 0;
#endif
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

          switch ( HYPreconID_ )
          {
             case HYDIAGONAL :
                  if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
                     printf("Diagonal preconditioning \n");
                  if ( HYPreconReuse_ == 1 )
                  {
                     HYPRE_ParCSRTFQmrSetPrecond(HYSolver_,HYPRE_ParCSRDiagScale,
                                      HYPRE_DummyFunction, HYPrecon_);
                  }
                  else
                  {
                     HYPRE_ParCSRTFQmrSetPrecond(HYSolver_,HYPRE_ParCSRDiagScale,
                                      HYPRE_ParCSRDiagScaleSetup, HYPrecon_);
                  }
                  break;

             case HYPILUT :
                  if (pilutFillin_ == 0) pilutFillin_ = pilutMaxNnzPerRow_;
                  if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
                  {
                     printf("PILUT - row size = %d\n", pilutFillin_);
                     printf("PILUT - drop tol = %e\n", pilutDropTol_);
                  }
                  HYPRE_ParCSRPilutSetFactorRowSize(HYPrecon_,pilutFillin_);
                  HYPRE_ParCSRPilutSetDropTolerance(HYPrecon_,pilutDropTol_);
                  if ( HYPreconReuse_ == 1 )
                  {
                     HYPRE_ParCSRTFQmrSetPrecond(HYSolver_,HYPRE_ParCSRPilutSolve,
                                      HYPRE_DummyFunction, HYPrecon_);
                  }
                  else
                  {
                     HYPRE_ParCSRTFQmrSetPrecond(HYSolver_,HYPRE_ParCSRPilutSolve,
                                      HYPRE_ParCSRPilutSetup, HYPrecon_);
                  }
                  break;

             case HYDDILUT :
                  if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
                  {
                     printf("DDILUT - fillin   = %e\n", ddilutFillin_);
                     printf("DDILUT - drop tol = %e\n", ddilutDropTol_);
                  }
                  HYPRE_LSI_DDIlutSetFillin(HYPrecon_,ddilutFillin_);
                  HYPRE_LSI_DDIlutSetDropTolerance(HYPrecon_,ddilutDropTol_);
                  if ( HYOutputLevel_ & HYFEI_DDILUT )
                     HYPRE_LSI_DDIlutSetOutputLevel(HYPrecon_,2);

                  if ( HYPreconReuse_ == 1 )
                  {
                     HYPRE_ParCSRTFQmrSetPrecond(HYSolver_,HYPRE_LSI_DDIlutSolve,
                                 HYPRE_DummyFunction,HYPrecon_);
                  }
                  else
                  {
                     HYPRE_ParCSRTFQmrSetPrecond(HYSolver_,HYPRE_LSI_DDIlutSolve,
                                 HYPRE_LSI_DDIlutSetup,HYPrecon_);
                  }
                  break;

             case HYDDICT :
                  if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && 
                        mypid_ == 0 )
                  {
                     printf("DDICT - fillin   = %e\n", ddictFillin_);
                     printf("DDICT - drop tol = %e\n", ddictDropTol_);
                  }
                  HYPRE_LSI_DDICTSetFillin(HYPrecon_,ddictFillin_);
                  HYPRE_LSI_DDICTSetDropTolerance(HYPrecon_,ddictDropTol_);
                  if ( HYOutputLevel_ & HYFEI_DDILUT )
                     HYPRE_LSI_DDICTSetOutputLevel(HYPrecon_,2);

                  if ( HYPreconReuse_ == 1 )
                  {
                     HYPRE_ParCSRTFQmrSetPrecond(HYSolver_,HYPRE_LSI_DDICTSolve,
                                      HYPRE_DummyFunction, HYPrecon_);
                  }
                  else
                  {
                     HYPRE_ParCSRTFQmrSetPrecond(HYSolver_,HYPRE_LSI_DDICTSolve,
                                      HYPRE_LSI_DDICTSetup, HYPrecon_);
                  }
                  break;

             case HYSCHWARZ :
                  if ( HYOutputLevel_ & HYFEI_DDILUT )
                     HYPRE_LSI_SchwarzSetOutputLevel(HYPrecon_,2);

                  HYPRE_LSI_SchwarzSetILUTFillin(HYPrecon_,schwarzFillin_);
                  HYPRE_LSI_SchwarzSetNBlocks(HYPrecon_, schwarzNblocks_);
                  HYPRE_LSI_SchwarzSetBlockSize(HYPrecon_, schwarzBlksize_);
                  HYPRE_ParCSRTFQmrSetPrecond(HYSolver_,HYPRE_LSI_SchwarzSolve,
                                              HYPRE_LSI_SchwarzSetup, HYPrecon_);
                  break;

             case HYPOLY :
                  if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
                     printf("Polynomial preconditioning - order = %d\n", polyOrder_);
                  HYPRE_LSI_PolySetOrder(HYPrecon_, polyOrder_);
                  if ( HYPreconReuse_ == 1 )
                  {
                     HYPRE_ParCSRTFQmrSetPrecond(HYSolver_,HYPRE_LSI_PolySolve,
                                      HYPRE_DummyFunction, HYPrecon_);
                  }
                  else
                  {
                     HYPRE_ParCSRTFQmrSetPrecond(HYSolver_,HYPRE_LSI_PolySolve,
                                      HYPRE_LSI_PolySetup, HYPrecon_);
                  }
                  break;

             case HYPARASAILS :
                  if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
                  {
                     printf("ParaSails - nlevels   = %d\n",parasailsNlevels_);
                     printf("ParaSails - threshold = %e\n",parasailsThreshold_);
                     printf("ParaSails - filter    = %e\n",parasailsFilter_);
                     printf("ParaSails - sym       = %d\n",parasailsSym_);
                     printf("ParaSails - loadbal   = %e\n",parasailsLoadbal_);
                  }
                  HYPRE_ParCSRParaSailsSetSym(HYPrecon_,parasailsSym_);
                  HYPRE_ParCSRParaSailsSetParams(HYPrecon_,parasailsThreshold_,
                                                 parasailsNlevels_);
                  HYPRE_ParCSRParaSailsSetFilter(HYPrecon_,parasailsFilter_);
                  HYPRE_ParCSRParaSailsSetLoadbal(HYPrecon_,parasailsLoadbal_);
                  HYPRE_ParCSRParaSailsSetReuse(HYPrecon_,parasailsReuse_);
                  if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1)
                  {
                     HYPRE_ParCSRParaSailsSetLogging(HYPrecon_, 1);
                  }

                  if ( HYPreconReuse_ == 1 )
                  {
                     HYPRE_ParCSRTFQmrSetPrecond(HYSolver_,
                                      HYPRE_ParCSRParaSailsSolve,
                                      HYPRE_DummyFunction, HYPrecon_);
                  }
                  else
                  {
                     HYPRE_ParCSRTFQmrSetPrecond(HYSolver_,
                                      HYPRE_ParCSRParaSailsSolve,
                                      HYPRE_ParCSRParaSailsSetup, HYPrecon_);
                  }
                  break;

             case HYBOOMERAMG :
                  HYPRE_BoomerAMGSetCoarsenType(HYPrecon_, amgCoarsenType_);
                  HYPRE_BoomerAMGSetMeasureType(HYPrecon_, amgMeasureType_);
                  HYPRE_BoomerAMGSetStrongThreshold(HYPrecon_,
                                                    amgStrongThreshold_);
                  num_sweeps = hypre_CTAlloc(int,4);
                  for ( i = 0; i < 4; i++ ) num_sweeps[i] = amgNumSweeps_[i];

                  HYPRE_BoomerAMGSetNumGridSweeps(HYPrecon_, num_sweeps);
                  relax_type = hypre_CTAlloc(int,4);
                  for ( i = 0; i < 4; i++ ) relax_type[i] = amgRelaxType_[i];

                  HYPRE_BoomerAMGSetGridRelaxType(HYPrecon_, relax_type);
                  relax_wt = hypre_CTAlloc(double,25);
                  for ( i = 0; i < 25; i++ ) relax_wt[i] = amgRelaxWeight_[i];
                  HYPRE_BoomerAMGSetRelaxWeight(HYPrecon_, relax_wt);
                  if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
                  {
                     printf("AMG coarsen type = %d\n", amgCoarsenType_);
                     printf("AMG measure type = %d\n", amgMeasureType_);
                     printf("AMG threshold    = %e\n", amgStrongThreshold_);
                     printf("AMG numsweeps    = %d\n", amgNumSweeps_[0]);
                     printf("AMG relax type   = %d\n", amgRelaxType_[0]);
                     printf("AMG relax weight = %e\n", amgRelaxWeight_[0]);
                  }
                  if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2)
                     HYPRE_BoomerAMGSetIOutDat(HYPrecon_, 2);

                  if ( HYPreconReuse_ == 1 )
                  {
                     HYPRE_ParCSRTFQmrSetPrecond(HYSolver_,
                         HYPRE_BoomerAMGSolve, HYPRE_DummyFunction, HYPrecon_);
                  }
                  else
                  {
                     HYPRE_ParCSRTFQmrSetPrecond(HYSolver_, 
                         HYPRE_BoomerAMGSolve, HYPRE_BoomerAMGSetup, HYPrecon_);
                  }
                  break;

#ifdef MLPACK
             case HYML :

                  HYPRE_ParCSRMLSetMethod(HYPrecon_,mlMethod_);
                  HYPRE_ParCSRMLSetCoarseSolver(HYPrecon_,mlCoarseSolver_);
                  HYPRE_ParCSRMLSetCoarsenScheme(HYPrecon_,mlCoarsenScheme_);
                  HYPRE_ParCSRMLSetStrongThreshold(HYPrecon_,mlStrongThreshold_);
                  HYPRE_ParCSRMLSetNumPreSmoothings(HYPrecon_,mlNumPreSweeps_);
                  HYPRE_ParCSRMLSetNumPostSmoothings(HYPrecon_,mlNumPostSweeps_);
                  HYPRE_ParCSRMLSetPreSmoother(HYPrecon_,mlPresmootherType_);
                  HYPRE_ParCSRMLSetPostSmoother(HYPrecon_,mlPostsmootherType_);
                  HYPRE_ParCSRMLSetDampingFactor(HYPrecon_, mlRelaxWeight_);
                  if ( HYPreconReuse_ == 1 )
                  {
                     HYPRE_ParCSRTFQmrSetPrecond(HYSolver_,
                         HYPRE_ParCSRMLSolve, HYPRE_DummyFunction, HYPrecon_);
                  }
                  else
                  {
                     HYPRE_ParCSRTFQmrSetPrecond(HYSolver_,
                         HYPRE_ParCSRMLSolve, HYPRE_ParCSRMLSetup, HYPrecon_);
                  }

                  if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
                  {
                     printf("ML strong threshold = %e\n", mlStrongThreshold_);
                     printf("ML numsweeps(pre)   = %d\n", mlNumPreSweeps_);
                     printf("ML numsweeps(post)  = %d\n", mlNumPostSweeps_);
                     printf("ML smoother (pre)   = %d\n", mlPresmootherType_);
                     printf("ML smoother (post)  = %d\n", mlPostsmootherType_);
                     printf("ML relax weight     = %e\n", mlRelaxWeight_);
                  }
                  break;
#endif
          }

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
          ptime  = MPI_Wtime();
          HYPRE_ParCSRTFQmrSolve(HYSolver_, A_csr, b_csr, x_csr);
          HYPRE_ParCSRTFQmrGetNumIterations(HYSolver_, &num_iterations);
          HYPRE_ParVectorCopy( b_csr, r_csr );
          HYPRE_ParCSRMatrixMatvec( -1.0, A_csr, x_csr, 1.0, r_csr );
          if (minResProjection_ == 1) addToProjectionSpace( currX_, currB_ );
          HYPRE_ParVectorInnerProd( r_csr, r_csr, &rnorm);
          rnorm = sqrt( rnorm );
#if defined(FEI_V13) || defined(FEI_V14)
          if ( num_iterations >= maxIterations_ ) status = 0; else status = 1;
#else
          if ( num_iterations >= maxIterations_ ) status = 1; else status = 0;
#endif
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

          switch ( HYPreconID_ )
          {
             case HYDIAGONAL :
                  if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
                     printf("Diagonal preconditioning \n");
                  if ( HYPreconReuse_ == 1 )
                  {
                     HYPRE_ParCSRBiCGSSetPrecond(HYSolver_,HYPRE_ParCSRDiagScale,
                                      HYPRE_DummyFunction, HYPrecon_);
                  }
                  else
                  {
                     HYPRE_ParCSRBiCGSSetPrecond(HYSolver_,HYPRE_ParCSRDiagScale,
                                      HYPRE_ParCSRDiagScaleSetup, HYPrecon_);
                  }
                  break;

             case HYPILUT :
                  if (pilutFillin_ == 0) pilutFillin_ = pilutMaxNnzPerRow_;
                  if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
                  {
                     printf("PILUT - row size = %d\n", pilutFillin_);
                     printf("PILUT - drop tol = %e\n", pilutDropTol_);
                  }
                  HYPRE_ParCSRPilutSetFactorRowSize(HYPrecon_,pilutFillin_);
                  HYPRE_ParCSRPilutSetDropTolerance(HYPrecon_,pilutDropTol_);
                  if ( HYPreconReuse_ == 1 )
                  {
                     HYPRE_ParCSRBiCGSSetPrecond(HYSolver_,HYPRE_ParCSRPilutSolve,
                                      HYPRE_DummyFunction, HYPrecon_);
                  }
                  else
                  {
                     HYPRE_ParCSRBiCGSSetPrecond(HYSolver_,HYPRE_ParCSRPilutSolve,
                                      HYPRE_ParCSRPilutSetup, HYPrecon_);
                  }
                  break;

             case HYDDILUT :
                  if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
                  {
                     printf("DDILUT - fillin   = %e\n", ddilutFillin_);
                     printf("DDILUT - drop tol = %e\n", ddilutDropTol_);
                  }
                  HYPRE_LSI_DDIlutSetFillin(HYPrecon_,ddilutFillin_);
                  HYPRE_LSI_DDIlutSetDropTolerance(HYPrecon_,ddilutDropTol_);
                  if ( HYOutputLevel_ & HYFEI_DDILUT )
                     HYPRE_LSI_DDIlutSetOutputLevel(HYPrecon_,2);

                  if ( HYPreconReuse_ == 1 )
                  {
                     HYPRE_ParCSRBiCGSSetPrecond(HYSolver_,HYPRE_LSI_DDIlutSolve,
                                 HYPRE_DummyFunction,HYPrecon_);
                  }
                  else
                  {
                     HYPRE_ParCSRBiCGSSetPrecond(HYSolver_,HYPRE_LSI_DDIlutSolve,
                                 HYPRE_LSI_DDIlutSetup,HYPrecon_);
                  }
                  break;

             case HYDDICT :
                  if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && 
                        mypid_ == 0 )
                  {
                     printf("DDICT - fillin   = %e\n", ddictFillin_);
                     printf("DDICT - drop tol = %e\n", ddictDropTol_);
                  }
                  HYPRE_LSI_DDICTSetFillin(HYPrecon_,ddictFillin_);
                  HYPRE_LSI_DDICTSetDropTolerance(HYPrecon_,ddictDropTol_);
                  if ( HYOutputLevel_ & HYFEI_DDILUT )
                     HYPRE_LSI_DDICTSetOutputLevel(HYPrecon_,2);

                  if ( HYPreconReuse_ == 1 )
                  {
                     HYPRE_ParCSRBiCGSSetPrecond(HYSolver_,HYPRE_LSI_DDICTSolve,
                                      HYPRE_DummyFunction, HYPrecon_);
                  }
                  else
                  {
                     HYPRE_ParCSRBiCGSSetPrecond(HYSolver_,HYPRE_LSI_DDICTSolve,
                                      HYPRE_LSI_DDICTSetup, HYPrecon_);
                  }
                  break;

             case HYSCHWARZ :
                  if ( HYOutputLevel_ & HYFEI_DDILUT )
                     HYPRE_LSI_SchwarzSetOutputLevel(HYPrecon_,2);

                  HYPRE_LSI_SchwarzSetILUTFillin(HYPrecon_,schwarzFillin_);
                  HYPRE_LSI_SchwarzSetNBlocks(HYPrecon_, schwarzNblocks_);
                  HYPRE_LSI_SchwarzSetBlockSize(HYPrecon_, schwarzBlksize_);
                  HYPRE_ParCSRBiCGSSetPrecond(HYSolver_,HYPRE_LSI_SchwarzSolve,
                                              HYPRE_LSI_SchwarzSetup, HYPrecon_);
                  break;

             case HYPOLY :
                  if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
                     printf("Polynomial preconditioning - order = %d\n", polyOrder_);
                  HYPRE_LSI_PolySetOrder(HYPrecon_, polyOrder_);
                  if ( HYPreconReuse_ == 1 )
                  {
                     HYPRE_ParCSRBiCGSSetPrecond(HYSolver_,HYPRE_LSI_PolySolve,
                                      HYPRE_DummyFunction, HYPrecon_);
                  }
                  else
                  {
                     HYPRE_ParCSRBiCGSSetPrecond(HYSolver_,HYPRE_LSI_PolySolve,
                                      HYPRE_LSI_PolySetup, HYPrecon_);
                  }
                  break;

             case HYPARASAILS :
                  if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
                  {
                     printf("ParaSails - nlevels   = %d\n",parasailsNlevels_);
                     printf("ParaSails - threshold = %e\n",parasailsThreshold_);
                     printf("ParaSails - filter    = %e\n",parasailsFilter_);
                     printf("ParaSails - sym       = %d\n",parasailsSym_);
                     printf("ParaSails - loadbal   = %e\n",parasailsLoadbal_);
                  }
                  HYPRE_ParCSRParaSailsSetSym(HYPrecon_,parasailsSym_);
                  HYPRE_ParCSRParaSailsSetParams(HYPrecon_,parasailsThreshold_,
                                                 parasailsNlevels_);
                  HYPRE_ParCSRParaSailsSetFilter(HYPrecon_,parasailsFilter_);
                  HYPRE_ParCSRParaSailsSetLoadbal(HYPrecon_,parasailsLoadbal_);
                  HYPRE_ParCSRParaSailsSetReuse(HYPrecon_,parasailsReuse_);
                  if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1)
                  {
                     HYPRE_ParCSRParaSailsSetLogging(HYPrecon_, 1);
                  }

                  if ( HYPreconReuse_ == 1 )
                  {
                     HYPRE_ParCSRBiCGSSetPrecond(HYSolver_,
                                      HYPRE_ParCSRParaSailsSolve,
                                      HYPRE_DummyFunction, HYPrecon_);
                  }
                  else
                  {
                     HYPRE_ParCSRBiCGSSetPrecond(HYSolver_,
                                      HYPRE_ParCSRParaSailsSolve,
                                      HYPRE_ParCSRParaSailsSetup, HYPrecon_);
                  }
                  break;

             case HYBOOMERAMG :
                  HYPRE_BoomerAMGSetCoarsenType(HYPrecon_, amgCoarsenType_);
                  HYPRE_BoomerAMGSetMeasureType(HYPrecon_, amgMeasureType_);
                  HYPRE_BoomerAMGSetStrongThreshold(HYPrecon_,
                                                    amgStrongThreshold_);
                  num_sweeps = hypre_CTAlloc(int,4);
                  for ( i = 0; i < 4; i++ ) num_sweeps[i] = amgNumSweeps_[i];

                  HYPRE_BoomerAMGSetNumGridSweeps(HYPrecon_, num_sweeps);
                  relax_type = hypre_CTAlloc(int,4);
                  for ( i = 0; i < 4; i++ ) relax_type[i] = amgRelaxType_[i];

                  HYPRE_BoomerAMGSetGridRelaxType(HYPrecon_, relax_type);
                  relax_wt = hypre_CTAlloc(double,25);
                  for ( i = 0; i < 25; i++ ) relax_wt[i] = amgRelaxWeight_[i];
                  HYPRE_BoomerAMGSetRelaxWeight(HYPrecon_, relax_wt);
                  if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
                  {
                     printf("AMG coarsen type = %d\n", amgCoarsenType_);
                     printf("AMG measure type = %d\n", amgMeasureType_);
                     printf("AMG threshold    = %e\n", amgStrongThreshold_);
                     printf("AMG numsweeps    = %d\n", amgNumSweeps_[0]);
                     printf("AMG relax type   = %d\n", amgRelaxType_[0]);
                     printf("AMG relax weight = %e\n", amgRelaxWeight_[0]);
                  }
                  if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2)
                     HYPRE_BoomerAMGSetIOutDat(HYPrecon_, 2);

                  if ( HYPreconReuse_ == 1 )
                  {
                     HYPRE_ParCSRBiCGSSetPrecond(HYSolver_,
                         HYPRE_BoomerAMGSolve, HYPRE_DummyFunction, HYPrecon_);
                  }
                  else
                  {
                     HYPRE_ParCSRBiCGSSetPrecond(HYSolver_, 
                         HYPRE_BoomerAMGSolve, HYPRE_BoomerAMGSetup, HYPrecon_);
                  }
                  break;

#ifdef MLPACK
             case HYML :

                  HYPRE_ParCSRMLSetMethod(HYPrecon_,mlMethod_);
                  HYPRE_ParCSRMLSetCoarseSolver(HYPrecon_,mlCoarseSolver_);
                  HYPRE_ParCSRMLSetCoarsenScheme(HYPrecon_,mlCoarsenScheme_);
                  HYPRE_ParCSRMLSetStrongThreshold(HYPrecon_,mlStrongThreshold_);
                  HYPRE_ParCSRMLSetNumPreSmoothings(HYPrecon_,mlNumPreSweeps_);
                  HYPRE_ParCSRMLSetNumPostSmoothings(HYPrecon_,mlNumPostSweeps_);
                  HYPRE_ParCSRMLSetPreSmoother(HYPrecon_,mlPresmootherType_);
                  HYPRE_ParCSRMLSetPostSmoother(HYPrecon_,mlPostsmootherType_);
                  HYPRE_ParCSRMLSetDampingFactor(HYPrecon_, mlRelaxWeight_);
                  if ( HYPreconReuse_ == 1 )
                  {
                     HYPRE_ParCSRBiCGSSetPrecond(HYSolver_,
                         HYPRE_ParCSRMLSolve, HYPRE_DummyFunction, HYPrecon_);
                  }
                  else
                  {
                     HYPRE_ParCSRBiCGSSetPrecond(HYSolver_,
                         HYPRE_ParCSRMLSolve, HYPRE_ParCSRMLSetup, HYPrecon_);
                  }

                  if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
                  {
                     printf("ML strong threshold = %e\n", mlStrongThreshold_);
                     printf("ML numsweeps(pre)   = %d\n", mlNumPreSweeps_);
                     printf("ML numsweeps(post)  = %d\n", mlNumPostSweeps_);
                     printf("ML smoother (pre)   = %d\n", mlPresmootherType_);
                     printf("ML smoother (post)  = %d\n", mlPostsmootherType_);
                     printf("ML relax weight     = %e\n", mlRelaxWeight_);
                  }
                  break;
#endif
          }

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
          ptime  = MPI_Wtime();
          HYPRE_ParCSRBiCGSSolve(HYSolver_, A_csr, b_csr, x_csr);
          HYPRE_ParCSRBiCGSGetNumIterations(HYSolver_, &num_iterations);
          HYPRE_ParVectorCopy( b_csr, r_csr );
          HYPRE_ParCSRMatrixMatvec( -1.0, A_csr, x_csr, 1.0, r_csr );
          if (minResProjection_ == 1) addToProjectionSpace( currX_, currB_ );
          HYPRE_ParVectorInnerProd( r_csr, r_csr, &rnorm);
          rnorm = sqrt( rnorm );
#if defined(FEI_V13) || defined(FEI_V14)
          if ( num_iterations >= maxIterations_ ) status = 0; else status = 1;
#else
          if ( num_iterations >= maxIterations_ ) status = 1; else status = 0;
#endif
          break;

       //----------------------------------------------------------------
       // choose Boomeramg  
       //----------------------------------------------------------------

       case HYAMG :

          if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0 )
             printf("%4d : launchSolver(Boomeramg)\n",mypid_);
          solveUsingBoomeramg(status);
          HYPRE_BoomerAMGGetNumIterations(HYSolver_, &num_iterations);
          HYPRE_ParVectorCopy( b_csr, r_csr );
          HYPRE_ParCSRMatrixMatvec( -1.0, A_csr, x_csr, 1.0, r_csr );
          HYPRE_ParVectorInnerProd( r_csr, r_csr, &rnorm);
          rnorm = sqrt( rnorm );
#if defined(FEI_V13) || defined(FEI_V14)
          if ( num_iterations >= maxIterations_ ) status = 0; else status = 1;
#else
          if ( num_iterations >= maxIterations_ ) status = 1; else status = 0;
#endif
          ptime  = stime;
          //printf("Boomeramg solver - return status = %d\n",status);
          break;

       //----------------------------------------------------------------
       // choose SuperLU (single processor) 
       //----------------------------------------------------------------

       case HYSUPERLU :

          if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0 )
             printf("%4d : launchSolver(SuperLU)\n",mypid_);
          solveUsingSuperLU(status);
#ifndef NOFEI
#if (!defined(FEI_V13)) || (!defined(FEI_V14))
          if ( status == 1 ) status = 0; 
#endif      
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
             printf("%4d : launchSolver(SuperLUX)\n",mypid_);
          solveUsingSuperLUX(status);
#ifndef NOFEI
#if (!defined(FEI_V13)) || (!defined(FEI_V14))
          if ( status == 1 ) status = 0; 
#endif      
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
             printf("%4d : launchSolver(Y12M)\n",mypid_);
          solveUsingY12M(status);
#ifndef NOFEI
#if (!defined(FEI_V13)) || (!defined(FEI_V14))
          if ( status == 1 ) status = 0; 
#endif      
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
             printf("%4d : launchSolver(AMGe)\n",mypid_);
          solveUsingAMGe(num_iterations);
#if defined(FEI_V13) || defined(FEI_V14)
          if ( num_iterations >= maxIterations_ ) status = 0;
#else
          if ( num_iterations >= maxIterations_ ) status = 1;
#endif
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
    etime = MPI_Wtime();
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
          HYPRE_IJVectorGetLocalComponents(currX_, 1, &i, NULL, &ddata);
          fprintf(fp, "%6d  %e \n", i+1, ddata);
       }
       fclose(fp);
       MPI_Barrier(comm_);
       exit(0);
    }

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
    {
       printf("%4d : HYPRE_LSC::leaving  launchSolver.\n", mypid_);
    }
}

//***************************************************************************
// this function solve the incoming linear system using Boomeramg
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::solveUsingBoomeramg(int& status)
{
    int                i, *relax_type, *num_sweeps;
    double             *relax_wt;
    HYPRE_ParCSRMatrix A_csr;
    HYPRE_ParVector    b_csr;
    HYPRE_ParVector    x_csr;

    A_csr = (HYPRE_ParCSRMatrix) HYPRE_IJMatrixGetLocalStorage(currA_);
    b_csr = (HYPRE_ParVector) HYPRE_IJVectorGetLocalStorage(currB_);
    x_csr = (HYPRE_ParVector) HYPRE_IJVectorGetLocalStorage(currX_);

    HYPRE_BoomerAMGSetCoarsenType(HYSolver_, amgCoarsenType_);
    HYPRE_BoomerAMGSetMeasureType(HYSolver_, amgMeasureType_);
    HYPRE_BoomerAMGSetStrongThreshold(HYSolver_, amgStrongThreshold_);

    num_sweeps = hypre_CTAlloc(int,4);
    for ( i = 0; i < 4; i++ ) num_sweeps[i] = amgNumSweeps_[i];
    HYPRE_BoomerAMGSetNumGridSweeps(HYSolver_, num_sweeps);

    relax_type = hypre_CTAlloc(int,4);
    for ( i = 0; i < 4; i++ ) relax_type[i] = amgRelaxType_[i];
    HYPRE_BoomerAMGSetGridRelaxType(HYSolver_, relax_type);

    relax_wt = hypre_CTAlloc(double,25);
    for ( i = 0; i < 25; i++ ) relax_wt[i] = amgRelaxWeight_[i];
    HYPRE_BoomerAMGSetRelaxWeight(HYSolver_, relax_wt);

    if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
    {
       printf("Boomeramg coarsen type = %d\n", amgCoarsenType_);
       printf("Boomeramg measure type = %d\n", amgMeasureType_);
       printf("Boomeramg threshold    = %e\n", amgStrongThreshold_);
       printf("Boomeramg numsweeps    = %d\n", amgNumSweeps_[0]);
       printf("Boomeramg relax type   = %d\n", amgRelaxType_[0]);
       printf("Boomeramg relax weight = %e\n", amgRelaxWeight_[0]);
    }
    if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2)
    {
       HYPRE_BoomerAMGSetIOutDat(HYSolver_, 2);
       HYPRE_BoomerAMGSetDebugFlag(HYSolver_, 0);
    }
    HYPRE_BoomerAMGSetMaxIter(HYSolver_, maxIterations_);
    HYPRE_BoomerAMGSetMeasureType(HYSolver_, 0);
    HYPRE_BoomerAMGSetup( HYSolver_, A_csr, b_csr, x_csr );
    HYPRE_BoomerAMGSolve( HYSolver_, A_csr, b_csr, x_csr );

    status = 0;
}

//***************************************************************************
// this function solve the incoming linear system using SuperLU
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::solveUsingSuperLU(int& status)
{
    int                i, nnz, nrows, ierr;
    int                rowSize, *colInd, *new_ia, *new_ja, *ind_array;
    int                j, nz_ptr, *partition, start_row, end_row;
    double             *colVal, *new_a, rnorm;
    HYPRE_ParCSRMatrix A_csr;
    HYPRE_ParVector    r_csr;
    HYPRE_ParVector    b_csr;
    HYPRE_ParVector    x_csr;

#ifdef SUPERLU
    int                info, panel_size, permc_spec;
    int                *perm_r, *perm_c;
    double             *rhs, *soln;
    mem_usage_t        mem_usage;
    SuperMatrix        A2, B, L, U;
    NRformat           *Astore, *Ustore;
    SCformat           *Lstore;
    DNformat           *Bstore;

    //------------------------------------------------------------------
    // available for sequential processing only for now
    //------------------------------------------------------------------

    if ( numProcs_ > 1 )
    {
       printf("solveUsingSuperLU ERROR - too many processors.\n");
       status = -1;
       return;
    }

    //------------------------------------------------------------------
    // need to construct a CSR matrix, and the column indices should
    // have been stored in colIndices and rowLengths
    //------------------------------------------------------------------
      
    //if ( colIndices_ == NULL || rowLengths_ == NULL )
    //{
    //   printf("solveUsingSuperLU ERROR - allocateMatrix not called.\n");
    //   status = -1;
    //   return;
    //}
    if ( localStartRow_ != 1 )
    {
       printf("solveUsingSuperLU ERROR - row does not start at 1\n");
       status = -1;
       return;
    }

    //------------------------------------------------------------------
    // get information about the current matrix
    //------------------------------------------------------------------

    A_csr  = (HYPRE_ParCSRMatrix) HYPRE_IJMatrixGetLocalStorage(currA_);
    HYPRE_ParCSRMatrixGetRowPartitioning( A_csr, &partition );
    start_row = partition[0];
    end_row   = partition[1] - 1;
    nrows     = end_row - start_row + 1;

    nnz   = 0;
    for ( i = start_row; i <= end_row; i++ )
    {
       HYPRE_ParCSRMatrixGetRow(A_csr,i,&rowSize,&colInd,&colVal);
       nnz += rowSize;
       HYPRE_ParCSRMatrixRestoreRow(A_csr,i,&rowSize,&colInd,&colVal);
    }

    new_ia = new int[nrows+1];
    new_ja = new int[nnz];
    new_a  = new double[nnz];
    nz_ptr = getMatrixCSR(currA_, nrows, nnz, new_ia, new_ja, new_a);
    nnz    = nz_ptr;

    //------------------------------------------------------------------
    // set up SuperLU CSR matrix and the corresponding rhs
    //------------------------------------------------------------------

    dCreate_CompRow_Matrix(&A2,nrows,nrows,nnz,new_a,new_ja,new_ia,NR,_D,GE);
    ind_array = new int[nrows];
    for ( i = 0; i < nrows; i++ ) ind_array[i] = i;
    rhs = new double[nrows];
    ierr = HYPRE_IJVectorGetLocalComponents(currB_, nrows, ind_array, NULL, rhs);
    assert(!ierr);
    dCreate_Dense_Matrix(&B, nrows, 1, rhs, nrows, DN, _D, GE);

    //------------------------------------------------------------------
    // set up the rest and solve (permc_spec=0 : natural ordering)
    //------------------------------------------------------------------
 
    perm_r = new int[nrows];
    perm_c = new int[nrows];
    permc_spec = superluOrdering_;
    get_perm_c(permc_spec, &A2, perm_c);
    panel_size = sp_ienv(1);

    dgssv(&A2, perm_c, perm_r, &L, &U, &B, &info);

    //------------------------------------------------------------------
    // postprocessing of the return status information
    //------------------------------------------------------------------

    if ( info == 0 ) 
    {
        status = 1;
        Lstore = (SCformat *) L.Store;
        Ustore = (NRformat *) U.Store;
        //printf("No of nonzeros in factor L = %d\n", Lstore->nnz);
        //printf("No of nonzeros in factor U = %d\n", Ustore->nnz);
        //printf("SuperLU : NNZ in L+U = %d\n",Lstore->nnz+Ustore->nnz-nrows);

        //dQuerySpace(&L, &U, panel_size, &mem_usage);
        //printf("L\\U MB %.3f\ttotal MB needed %.3f\texpansions %d\n",
        //       mem_usage.for_lu/1e6, mem_usage.total_needed/1e6,
        //       mem_usage.expansions);

    } 
    else 
    {
        status = 0;
        printf("HYPRE_LinSysCore::solveUsingSuperLU - dgssv error = %d\n",info);
        //if ( info <= nrows ) { /* factorization completes */
        //    dQuerySpace(&L, &U, panel_size, &mem_usage);
        //    printf("L\\U MB %.3f\ttotal MB needed %.3f\texpansions %d\n",
        //           mem_usage.for_lu/1e6, mem_usage.total_needed/1e6,
        //           mem_usage.expansions);
        //}
    }

    //------------------------------------------------------------------
    // fetch the solution and find residual norm
    //------------------------------------------------------------------

    if ( info == 0 )
    {
       soln = (double *) ((DNformat *) B.Store)->nzval;
       ierr = HYPRE_IJVectorSetLocalComponents(currX_,nrows,ind_array,NULL,soln);
       assert(!ierr);
       x_csr  = (HYPRE_ParVector) HYPRE_IJVectorGetLocalStorage(currX_);
       b_csr  = (HYPRE_ParVector) HYPRE_IJVectorGetLocalStorage(currB_);
       r_csr  = (HYPRE_ParVector) HYPRE_IJVectorGetLocalStorage(currR_);
       ierr = HYPRE_ParVectorCopy( b_csr, r_csr );
       assert(!ierr);
       ierr = HYPRE_ParCSRMatrixMatvec( -1.0, A_csr, x_csr, 1.0, r_csr );
       assert(!ierr);
       ierr = HYPRE_ParVectorInnerProd( r_csr, r_csr, &rnorm);
       assert(!ierr);
       rnorm = sqrt( rnorm );
       if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 )
          printf("HYPRE_LSC::solveUsingSuperLU - FINAL NORM = %e.\n",rnorm);
    }

    //------------------------------------------------------------------
    // clean up 
    //------------------------------------------------------------------

    delete [] ind_array; 
    delete [] rhs; 
    delete [] perm_c; 
    delete [] perm_r; 
    delete [] new_ia; 
    delete [] new_ja; 
    delete [] new_a; 
    Destroy_SuperMatrix_Store(&B);
    Destroy_SuperNode_Matrix(&L);
    SUPERLU_FREE( A2.Store );
    SUPERLU_FREE( ((NRformat *) U.Store)->colind);
    SUPERLU_FREE( ((NRformat *) U.Store)->rowptr);
    SUPERLU_FREE( ((NRformat *) U.Store)->nzval);
    SUPERLU_FREE( U.Store );
#else
    printf("HYPRE_LSC::solveUsingSuperLU : not available.\n");
#endif
}

//***************************************************************************
// this function solve the incoming linear system using SuperLU
// using expert mode
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::solveUsingSuperLUX(int& status)
{
    int                i, k, nnz, nrows, ierr;
    int                rowSize, *colInd, *new_ia, *new_ja, *ind_array;
    int                j, nz_ptr, *colLengths, count, maxRowSize, rowSize2;
    int                *partition, start_row, end_row;
    double             *colVal, *new_a, rnorm;
    HYPRE_ParCSRMatrix A_csr;
    HYPRE_ParVector    r_csr;
    HYPRE_ParVector    b_csr;
    HYPRE_ParVector    x_csr;

#ifdef SUPERLU
    int                info, panel_size, permc_spec;
    int                *perm_r, *perm_c, *etree, lwork, relax;
    double             *rhs, *soln;
    double             *R, *C;
    double             *ferr, *berr;
    double             rpg, rcond;
    char               fact[1], equed[1], trans[1], refact[1];
    void               *work=NULL;
    mem_usage_t        mem_usage;
    SuperMatrix        A2, B, X, L, U;
    NRformat           *Astore, *Ustore;
    SCformat           *Lstore;
    DNformat           *Bstore;
    factor_param_t     iparam;

    //------------------------------------------------------------------
    // available for sequential processing only for now
    //------------------------------------------------------------------

    if ( numProcs_ > 1 )
    {
       printf("solveUsingSuperLUX ERROR - too many processors.\n");
       status = -1;
       return;
    }

    //------------------------------------------------------------------
    // need to construct a CSR matrix, and the column indices should
    // have been stored in colIndices and rowLengths
    //------------------------------------------------------------------
      
    //if ( colIndices_ == NULL || rowLengths_ == NULL )
    //{
    //   printf("solveUsingSuperLUX ERROR - Configure not called\n");
    //   status = -1;
    //   return;
    //}
    if ( localStartRow_ != 1 )
    {
       printf("solveUsingSuperLUX ERROR - row not start at 1\n");
       status = -1;
       return;
    }

    //------------------------------------------------------------------
    // get information about the current matrix
    //------------------------------------------------------------------

    A_csr  = (HYPRE_ParCSRMatrix) HYPRE_IJMatrixGetLocalStorage(currA_);
    HYPRE_ParCSRMatrixGetRowPartitioning( A_csr, &partition );
    start_row = partition[0];
    end_row   = partition[1] - 1;
    nrows     = end_row - start_row + 1;

    colLengths = new int[nrows];
    for ( i = 0; i < nrows; i++ ) colLengths[i] = 0;
    
    maxRowSize = 0;
    for ( i = 0; i < nrows; i++ )
    {
       HYPRE_ParCSRMatrixGetRow(A_csr,i,&rowSize,&colInd,&colVal);
       maxRowSize = ( rowSize > maxRowSize ) ? rowSize : maxRowSize;
       for ( j = 0; j < rowSize; j++ ) 
          if ( colVal[j] != 0.0 ) colLengths[colInd[j]]++;
       HYPRE_ParCSRMatrixRestoreRow(A_csr,i,&rowSize,&colInd,&colVal);
    }   
    nnz = 0;
    for ( i = 0; i < nrows; i++ ) nnz += colLengths[i];

    new_ia = new int[nrows+1];
    new_ja = new int[nnz];
    new_a  = new double[nnz];
    nz_ptr = getMatrixCSR(currA_, nrows, nnz, new_ia, new_ja, new_a);
    nnz = nz_ptr;

    //------------------------------------------------------------------
    // set up SuperLU CSR matrix and the corresponding rhs
    //------------------------------------------------------------------

    dCreate_CompRow_Matrix(&A2,nrows,nrows,nnz,new_a,new_ja,new_ia,NR,_D,GE);
    ind_array = new int[nrows];
    for ( i = 0; i < nrows; i++ ) ind_array[i] = i;
    rhs = new double[nrows];
    ierr = HYPRE_IJVectorGetLocalComponents(currB_,nrows,ind_array,NULL,rhs);
    assert(!ierr);
    dCreate_Dense_Matrix(&B, nrows, 1, rhs, nrows, DN, _D, GE);
    soln = new double[nrows];
    for ( i = 0; i < nrows; i++ ) soln[i] = 0.0;
    dCreate_Dense_Matrix(&X, nrows, 1, soln, nrows, DN, _D, GE);

    //------------------------------------------------------------------
    // set up the other parameters (permc_spec=0 : natural ordering)
    //------------------------------------------------------------------
 
    perm_r = new int[nrows];
    perm_c = new int[nrows];
    etree  = new int[nrows];
    permc_spec = superluOrdering_;
    get_perm_c(permc_spec, &A2, perm_c);
    panel_size               = sp_ienv(1);
    iparam.panel_size        = panel_size;
    iparam.relax             = sp_ienv(2);
    iparam.diag_pivot_thresh = 1.0;
    iparam.drop_tol          = -1;
    lwork                    = 0;
    *fact                    = 'N';
    *equed                   = 'N';
    *trans                   = 'N';
    *refact                  = 'N';
    R    = (double *) SUPERLU_MALLOC(A2.nrow * sizeof(double));
    C    = (double *) SUPERLU_MALLOC(A2.ncol * sizeof(double));
    ferr = (double *) SUPERLU_MALLOC(sizeof(double));
    berr = (double *) SUPERLU_MALLOC(sizeof(double));

    //------------------------------------------------------------------
    // solve
    //------------------------------------------------------------------

    dgssvx(fact, trans, refact, &A2, &iparam, perm_c, perm_r, etree,
           equed, R, C, &L, &U, work, lwork, &B, &X, &rpg, &rcond,
           ferr, berr, &mem_usage, &info);

    //------------------------------------------------------------------
    // print SuperLU internal information at the first step
    //------------------------------------------------------------------
       
    if ( info == 0 || info == nrows+1 ) 
    {
        status = 1;
        //printf("Recip. pivot growth = %e\n", rpg);
        //printf("%8s%16s%16s\n", "rhs", "FERR", "BERR");
        //printf("%8d%16e%16e\n", 1, ferr[0], berr[0]);
        //if ( rcond != 0.0 )
        //   printf("   SuperLU : condition number = %e\n", 1.0/rcond);
        //else
        //   printf("   SuperLU : Recip. condition number = %e\n", rcond);

        Lstore = (SCformat *) L.Store;
        Ustore = (NRformat *) U.Store;
        //printf("No of nonzeros in factor L = %d\n", Lstore->nnz);
        //printf("No of nonzeros in factor U = %d\n", Ustore->nnz);
        //printf("SuperLU : NNZ in L+U = %d\n", Lstore->nnz+Ustore->nnz-nrows);

        //dQuerySpace(&L, &U, panel_size, &mem_usage);
        //printf("L\\U MB %.3f\ttotal MB needed %.3f\texpansions %d\n",
        //       mem_usage.for_lu/1e6, mem_usage.total_needed/1e6,
        //       mem_usage.expansions);
    } else {
        //printf("solveUsingSuperLUX - dgssvx error code = %d\n",info);
        status = 0;
    }

    //------------------------------------------------------------------
    // fetch the solution and find residual norm
    //------------------------------------------------------------------

    if ( status == 1 )
    {
       ierr = HYPRE_IJVectorSetLocalComponents(currX_,nrows,ind_array,NULL,soln);
       assert(!ierr);
       x_csr  = (HYPRE_ParVector) HYPRE_IJVectorGetLocalStorage(currX_);
       r_csr  = (HYPRE_ParVector) HYPRE_IJVectorGetLocalStorage(currR_);
       b_csr    = (HYPRE_ParVector) HYPRE_IJVectorGetLocalStorage(currB_);
       ierr = HYPRE_ParVectorCopy( b_csr, r_csr );
       assert(!ierr);
       ierr = HYPRE_ParCSRMatrixMatvec( -1.0, A_csr, x_csr, 1.0, r_csr );
       assert(!ierr);
       ierr = HYPRE_ParVectorInnerProd( r_csr, r_csr, &rnorm);
       assert(!ierr);
       rnorm = sqrt( rnorm );
       if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 )
          printf("HYPRE_LSC::solveUsingSuperLUX - FINAL NORM = %e.\n",rnorm);
    }

    //------------------------------------------------------------------
    // clean up 
    //------------------------------------------------------------------

    delete [] ind_array; 
    delete [] perm_c; 
    delete [] perm_r; 
    delete [] etree; 
    delete [] rhs; 
    delete [] new_ia;
    delete [] new_ja;
    delete [] new_a;
    delete [] soln;
    delete [] colLengths;
    Destroy_SuperMatrix_Store(&B);
    Destroy_SuperNode_Matrix(&L);
    SUPERLU_FREE( A2.Store );
    SUPERLU_FREE( ((NRformat *) U.Store)->colind);
    SUPERLU_FREE( ((NRformat *) U.Store)->rowptr);
    SUPERLU_FREE( ((NRformat *) U.Store)->nzval);
    SUPERLU_FREE( U.Store );
    SUPERLU_FREE (R);
    SUPERLU_FREE (C);
    SUPERLU_FREE (ferr);
    SUPERLU_FREE (berr);
#else
    printf("HYPRE_LSC::solveUsingSuperLUX : not available.\n");
#endif

}

//***************************************************************************
// this function solve the incoming linear system using Y12M
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::solveUsingY12M(int& status)
{
    int                i, k, nnz, nrows, ierr;
    int                rowSize, *colInd, *ind_array;
    int                j, nz_ptr, *colLengths, count, maxRowSize;
    double             *colVal, rnorm;
    double             upperSum, lowerSum, *accuSoln, *origRhs;
    HYPRE_ParCSRMatrix A_csr;
    HYPRE_ParVector    r_csr;
    HYPRE_ParVector    b_csr;
    HYPRE_ParVector    x_csr;

    int                n, nn, nn1, *rnr, *snr, *ha, iha, iflag[10], ifail;
    double             *pivot, *val, *rhs, aflag[8];

#ifdef Y12M
    //------------------------------------------------------------------
    // available for sequential processing only for now
    //------------------------------------------------------------------

    if ( numProcs_ > 1 )
    {
       printf("solveUsingY12M ERROR - too many processors.\n");
       status = 0;
       return;
    }

    //------------------------------------------------------------------
    // need to construct a CSR matrix, and the column indices should
    // have been stored in colIndices and rowLengths
    //------------------------------------------------------------------
      
    //if ( colIndices_ == NULL || rowLengths_ == NULL )
    //{
    //   printf("solveUsingY12M ERROR - Configure not called\n");
    //   status = -1;
    //   return;
    //}
    if ( localStartRow_ != 1 )
    {
       printf("solveUsingY12M ERROR - row does not start at 1.\n");
       status = -1;
       return;
    }
    if (slideReduction_  == 1) 
         nrows = localEndRow_ - 2 * nConstraints_;
    else if (slideReduction_  == 2) 
         nrows = localEndRow_ - nConstraints_;
    else if (schurReduction_ == 1) 
         nrows = localEndRow_ - localStartRow_ + 1 - A21NRows_;
    else nrows = localEndRow_;

    colLengths = new int[nrows];
    for ( i = 0; i < nrows; i++ ) colLengths[i] = 0;
    
    maxRowSize = 0;
    A_csr  = (HYPRE_ParCSRMatrix) HYPRE_IJMatrixGetLocalStorage(currA_);
    for ( i = 0; i < nrows; i++ )
    {
       HYPRE_ParCSRMatrixGetRow(A_csr,i,&rowSize,&colInd,&colVal);
       maxRowSize = ( rowSize > maxRowSize ) ? rowSize : maxRowSize;
       for ( j = 0; j < rowSize; j++ ) 
          if ( colVal[j] != 0.0 ) colLengths[colInd[j]]++;
       HYPRE_ParCSRMatrixRestoreRow(A_csr,i,&rowSize,&colInd,&colVal);
    }   
    nnz   = 0;
    for ( i = 0; i < nrows; i++ ) nnz += colLengths[i];

    nn     = 2 * nnz;
    nn1    = 2 * nnz;
    snr    = new int[nn];
    rnr    = new int[nn1];
    val    = new double[nn];
    pivot  = new double[nrows];
    iha    = nrows;
    ha     = new int[iha*11];

    nz_ptr = 0;
    for ( i = 0; i < nrows; i++ )
    {
       HYPRE_ParCSRMatrixGetRow(A_csr,i,&rowSize,&colInd,&colVal);
       for ( j = 0; j < rowSize; j++ )
       {
          if ( colVal[j] != 0.0 )
          {
             rnr[nz_ptr] = i + 1;
             snr[nz_ptr] = colInd[j] + 1;
             val[nz_ptr] = colVal[j];
             nz_ptr++;
          }
       }
       HYPRE_ParCSRMatrixRestoreRow(A_csr,i,&rowSize,&colInd,&colVal);
    }   

    nnz = nz_ptr;

    //------------------------------------------------------------------
    // set up other parameters and the right hand side
    //------------------------------------------------------------------

    aflag[0] = 16.0;
    aflag[1] = 0.0;
    aflag[2] = 1.0e8;
    aflag[3] = 1.0e-12;
    iflag[0] = 1;
    iflag[1] = 3;
    iflag[2] = 1;
    iflag[3] = 0;
    iflag[4] = 2;
    ind_array = new int[nrows];
    for ( i = 0; i < nrows; i++ ) ind_array[i] = i;
    rhs = new double[nrows];
    ierr = HYPRE_IJVectorGetLocalComponents(currB_,nrows,ind_array,NULL,rhs);
    assert(!ierr);

    //------------------------------------------------------------------
    // call Y12M to solve the linear system
    //------------------------------------------------------------------

    y12maf_(&nrows,&nnz,val,snr,&nn,rnr,&nn1,pivot,ha,&iha,aflag,iflag,
            rhs,&ifail);
    if ( ifail != 0 && (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 )
    {
       printf("solveUsingY12M WARNING - ifail = %d\n", ifail);
    }
 
    //------------------------------------------------------------------
    // postprocessing
    //------------------------------------------------------------------

    if ( ifail == 0 )
    {
       ierr = HYPRE_IJVectorSetLocalComponents(currX_,nrows,ind_array,NULL,rhs);
       assert(!ierr);
       x_csr  = (HYPRE_ParVector) HYPRE_IJVectorGetLocalStorage(currX_);
       r_csr  = (HYPRE_ParVector) HYPRE_IJVectorGetLocalStorage(currR_);
       b_csr  = (HYPRE_ParVector) HYPRE_IJVectorGetLocalStorage(currB_);
       ierr = HYPRE_ParVectorCopy( b_csr, r_csr );
       assert(!ierr);
       ierr = HYPRE_ParCSRMatrixMatvec( -1.0, A_csr, x_csr, 1.0, r_csr );
       assert(!ierr);
       ierr = HYPRE_ParVectorInnerProd( r_csr, r_csr, &rnorm);
       assert(!ierr);
       rnorm = sqrt( rnorm );
       if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 )
          printf("HYPRE_LSC::solveUsingY12M - final norm = %e.\n", rnorm);
    }

    //------------------------------------------------------------------
    // clean up 
    //------------------------------------------------------------------

    delete [] ind_array; 
    delete [] rhs; 
    delete [] val; 
    delete [] snr; 
    delete [] rnr; 
    delete [] ha; 
    delete [] pivot; 
#else
    printf("HYPRE_LSC::solveUsingY12M - not available.\n");
#endif

}

//***************************************************************************
// this function solve the incoming linear system using Y12M
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::solveUsingAMGe(int &iterations)
{
    int                i, nrows, ierr, *ind_array, status;
    double             rnorm, *rhs, *sol;
    HYPRE_ParCSRMatrix A_csr;
    HYPRE_ParVector    r_csr;
    HYPRE_ParVector    b_csr;
    HYPRE_ParVector    x_csr;

#ifdef HAVE_AMGE
    //------------------------------------------------------------------
    // available for sequential processing only for now
    //------------------------------------------------------------------

    if ( numProcs_ > 1 )
    {
       printf("solveUsingAMGE ERROR - too many processors.\n");
       iterations = 0;
       return;
    }

    //------------------------------------------------------------------
    // need to construct a CSR matrix, and the column indices should
    // have been stored in colIndices and rowLengths
    //------------------------------------------------------------------
      
    if ( localStartRow_ != 1 )
    {
       printf("solveUsingAMGe ERROR - row does not start at 1.\n");
       status = -1;
       return;
    }
    if (slideReduction_  == 1) 
         nrows = localEndRow_ - 2 * nConstraints_;
    else if (slideReduction_  == 2) 
         nrows = localEndRow_ - nConstraints_;
    else if (schurReduction_ == 1) 
         nrows = localEndRow_ - localStartRow_ + 1 - A21NRows_;
    else nrows = localEndRow_;

    //------------------------------------------------------------------
    // set up the right hand side
    //------------------------------------------------------------------

    ind_array = new int[nrows];
    for ( i = 0; i < nrows; i++ ) ind_array[i] = i;
    rhs = new double[nrows];
    ierr = HYPRE_IJVectorGetLocalComponents(currB_,nrows,ind_array,NULL,rhs);
    assert(!ierr);

    //------------------------------------------------------------------
    // call Y12M to solve the linear system
    //------------------------------------------------------------------

    sol = new double[nrows];
    status = HYPRE_LSI_AMGeSolve( rhs, sol ); 
 
    //------------------------------------------------------------------
    // postprocessing
    //------------------------------------------------------------------

    ierr = HYPRE_IJVectorSetLocalComponents(currX_,nrows,ind_array,NULL,sol);
    assert(!ierr);
    x_csr  = (HYPRE_ParVector) HYPRE_IJVectorGetLocalStorage(currX_);
    r_csr  = (HYPRE_ParVector) HYPRE_IJVectorGetLocalStorage(currR_);
    b_csr  = (HYPRE_ParVector) HYPRE_IJVectorGetLocalStorage(currB_);
    ierr = HYPRE_ParVectorCopy( b_csr, r_csr );
    assert(!ierr);
    ierr = HYPRE_ParCSRMatrixMatvec( -1.0, A_csr, x_csr, 1.0, r_csr );
    assert(!ierr);
    ierr = HYPRE_ParVectorInnerProd( r_csr, r_csr, &rnorm);
    assert(!ierr);
    rnorm = sqrt( rnorm );
    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 )
       printf("HYPRE_LSC::solveUsingAMGe - final norm = %e.\n", rnorm);

    //------------------------------------------------------------------
    // clean up 
    //------------------------------------------------------------------

    delete [] ind_array; 
    delete [] rhs; 
    delete [] sol; 
#else
    printf("HYPRE_LSC::solveUsingAMGe - not available.\n");
#endif

}

//***************************************************************************
// this function loads in the constraint numbers for reduction
// (to activate automatic slave search, constrList should be NULL)
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::loadConstraintNumbers(int nConstr, int *constrList)
{
    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
    {
       printf("%4d : HYPRE_LSC::loadConstraintNumbers - size = %d\n", 
                     mypid_, nConstr);
       //if ( mypid_ == 0 )
       //   printf("%4d loadConstraintNumbers : DOF=3, NODE#=EQN# assumed.\n",
       //          mypid_); 
    }

    nConstraints_ = nConstr;
    //if ( nConstr > 0 )
    //{
    //   if ( constrList != NULL ) 
    //   {
    //      constrList_ = new int[3 * nConstr];
    //      for (int i = 0; i < nConstr; i++) 
    //      {
    //         constrList_[3*i] = constrList[i] * 3;
    //         constrList_[3*i+1] = constrList[i] * 3 + 1;
    //         constrList_[3*i+2] = constrList[i] * 3 + 2;
    //      }
    //      if ( HYOutputLevel_ > 2 )
    //      {
    //         for (int j = 0; j < 3 * nConstraints_; j++) 
    //            printf("Constraint %5d(%5d) = %d\n",j,nConstraints_,
    //                                                constrList_[j]);
    //      }
    //   }
    //}
    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
    {
       printf("%4d : HYPRE_LSC::leaving  loadConstraintNumbers\n", mypid_);
    }
}

//***************************************************************************
// this function extracts the matrix in a CSR format
//---------------------------------------------------------------------------

#ifdef FEI_V13
void HYPRE_LinSysCore::writeSystem(char *name)
#else
void HYPRE_LinSysCore::writeSystem(const char *name)
#endif
{
    printf("HYPRE_LinsysCore : writeSystem not implemented.\n");
    return;
}

//***************************************************************************
// this function extracts the the version number from HYPRE
//---------------------------------------------------------------------------

char *HYPRE_LinSysCore::getVersion()
{
    char hypre[200], hypre_version[50], version[100], ctmp[50];

    sprintf(hypre, "%s", HYPRE_Version());
    sscanf("%s %s", ctmp, hypre_version);
    sprintf(version, "%s-%s", HYPRE_FEI_Version(), hypre_version);
    return version;
}

//***************************************************************************
// create a node to equation map from the solution vector
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::beginCreateMapFromSoln()
{
   mapFromSolnFlag_    = 1;
   mapFromSolnLengMax_ = 10;
   mapFromSolnLeng_    = 0;
   mapFromSolnList_    = new int[mapFromSolnLengMax_];
   mapFromSolnList2_   = new int[mapFromSolnLengMax_];
   return;
}

//***************************************************************************
// create a node to equation map from the solution vector
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::endCreateMapFromSoln()
{
    int    i, ierr, *equations, local_nrows, *iarray;
    double *darray, *answers;

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
    {
       printf("%4d : HYPRE_LSC::entering endCreateMapFromSoln.\n",mypid_);
    }

    mapFromSolnFlag_ = 0;
    if ( mapFromSolnLeng_ > 0 )
       darray = new double[mapFromSolnLeng_];
    for ( i = 0; i < mapFromSolnLeng_; i++ )
       darray[i] = (double) mapFromSolnList_[i];

    qsort1(mapFromSolnList2_, darray, 0, mapFromSolnLeng_-1);
    iarray = mapFromSolnList2_;
    mapFromSolnList2_ = mapFromSolnList_;
    mapFromSolnList_ = iarray;
    for ( i = 0; i < mapFromSolnLeng_; i++ )
       mapFromSolnList2_[i] = (int) darray[i];
    delete [] darray;

    for ( i = 0; i < mapFromSolnLeng_; i++ )
       printf("HYPRE_LSC::mapFromSoln %d = %d\n",mapFromSolnList_[i],
              mapFromSolnList2_[i]);

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
    {
       printf("%4d : HYPRE_LSC::leaving  endCreateMapFromSoln.\n",mypid_);
    }
}

//***************************************************************************
// add extra nonzero entries into the matrix data structure
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::putIntoMappedMatrix(int row, int numValues,
                  const double* values, const int* scatterIndices)
{
    int    i, index, colIndex, localRow, mappedRow, mappedCol, newLeng;
    int    *tempInd, ind2;
    double *tempVal;

    //-------------------------------------------------------------------
    // error checking
    //-------------------------------------------------------------------

    if ( systemAssembled_ == 1 )
    {
       printf("putIntoMappedMatrix ERROR : matrix already assembled\n");
       exit(1);
    }
    if ( (row+1) < localStartRow_ || (row+1) > localEndRow_ )
    {
       printf("putIntoMappedMatrix ERROR : invalid row number %d.\n",row);
       exit(1);
    }
    index = HYPRE_LSI_Search(mapFromSolnList_, row, mapFromSolnLeng_);

    if ( index >= 0 ) mappedRow = mapFromSolnList2_[index];
    else              mappedRow = row;
    localRow = mappedRow - localStartRow_ + 1;

    //-------------------------------------------------------------------
    // load the local matrix
    //-------------------------------------------------------------------

    newLeng = rowLengths_[localRow] + numValues;
    tempInd = new int[newLeng];
    tempVal = new double[newLeng];
    for ( i = 0; i < rowLengths_[localRow]; i++ ) 
    {
       tempVal[i] = colValues_[localRow][i];
       tempInd[i] = colIndices_[localRow][i];
    }
    delete [] colValues_[localRow];
    delete [] colIndices_[localRow];
    colValues_[localRow] = tempVal;
    colIndices_[localRow] = tempInd;

    index = rowLengths_[localRow];

    for ( i = 0; i < numValues; i++ ) 
    {
       colIndex = scatterIndices[i];

       ind2 = HYPRE_LSI_Search(mapFromSolnList_,colIndex,mapFromSolnLeng_);
       if ( mapFromSolnList_ != NULL ) mappedCol = mapFromSolnList2_[ind2];
       else                            mappedCol = colIndex;

       ind2 = HYPRE_LSI_Search(colIndices_[localRow],mappedCol+1,index);
       if ( ind2 >= 0 ) 
       {
          newLeng--;
          colValues_[localRow][ind2] = values[i];
          if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 0 )
             printf("%4d : putIntoMappedMatrix (add) : row, col = %8d %8d %e \n",
                    mypid_, localRow, colIndices_[localRow][ind2]-1,
                    colValues_[localRow][ind2]);
       }
       else
       {
          ind2 = index;
          colIndices_[localRow][index] = mappedCol + 1;
          colValues_[localRow][index++] = values[i];
          if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 0 )
             printf("%4d : putIntoMappedMatrix : row, col = %8d %8d %e \n",
                    mypid_, localRow, colIndices_[localRow][ind2]-1,
                    colValues_[localRow][ind2]);
          HYPRE_LSI_Sort(colIndices_[localRow],index,NULL,colValues_[localRow]);
       }
    }
    rowLengths_[localRow] = newLeng;
}

//***************************************************************************
// project the initial guess into the previous solutions (x + X inv(R) Q^T b)
// Given r and B (a collection of right hand vectors such that A X = B)
//
//          min   || r - B v ||
//           v
//
// = min (trans(r) r - trans(r) B v-trans(v) trans(B) r+trans(v) trans(B) B v)
//
// ==> trans(B) r = trans(B) B v ==> v = inv(trans(B) B) trans(B) r
//
// Use QR decomposition B = QR
//
//   ==>  v = inv( trans(R) R ) trans(R) trans(Q) r = inv(R) trans(Q) r
//
// Once v is computed, x = x + X v
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::computeMinResProjection(HYPRE_ParCSRMatrix A_csr,
                              HYPRE_ParVector x_csr, HYPRE_ParVector b_csr)
{
    int             i, j;
    double          alpha, *darray, *darray2;
    HYPRE_ParVector r_csr, v_csr;

    //-----------------------------------------------------------------------
    // diagnostic message
    //-----------------------------------------------------------------------

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
    {
       printf("%4d : HYPRE_LSC::entering computeMinResProjection %d\n",mypid_,
              projectCurrSize_);
    }
    if ( projectCurrSize_ == 0 ) return;

    //-----------------------------------------------------------------------
    // compute r = b - A x
    //-----------------------------------------------------------------------

    darray  = (double *) malloc( projectSize_ * sizeof(double) );
    darray2 = (double *) malloc( projectSize_ * sizeof(double) );
    for ( i = 0; i < projectCurrSize_; i++ ) darray[i] = darray2[i] = 0.0;
    r_csr = (HYPRE_ParVector) HYPRE_IJVectorGetLocalStorage(currR_);
    HYPRE_ParVectorCopy( b_csr, r_csr );
    HYPRE_ParCSRMatrixMatvec( -1.0, A_csr, x_csr, 1.0, r_csr );

    //-----------------------------------------------------------------------
    // compute rtil = trans(Q) r
    //-----------------------------------------------------------------------

    for ( i = 0; i < projectCurrSize_; i++ )
    {
       v_csr = (HYPRE_ParVector) HYPRE_IJVectorGetLocalStorage(HYpbs_[i]);
       HYPRE_ParVectorInnerProd( r_csr,  v_csr, &alpha);
       darray[i] = alpha;
    }

    //-----------------------------------------------------------------------
    // compute v = inv(R) * rtil
    //-----------------------------------------------------------------------

    for ( i = projectCurrSize_-1; i >= 0; i-- )
    {
       darray2[i] = darray[i];
       for ( j = i+1; j < projectCurrSize_; j++ )
          darray2[i] -= ( darray2[j] * projectionMatrix_[i][j] );
       darray2[i] /= projectionMatrix_[i][i];
    }

    //-----------------------------------------------------------------------
    // compute x + X v
    //-----------------------------------------------------------------------

    for ( i = 0; i < projectCurrSize_; i++ )
    {
       v_csr = (HYPRE_ParVector) HYPRE_IJVectorGetLocalStorage(HYpxs_[i]);
       alpha = darray2[i];
       hypre_ParVectorAxpy(alpha,(hypre_ParVector*)v_csr,(hypre_ParVector*)x_csr);
    }
    free( darray );
    free( darray2 );

    //-----------------------------------------------------------------------
    // diagnostic message
    //-----------------------------------------------------------------------

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
    {
       printf("%4d : HYPRE_LSC:: leaving computeMinResProjection n", mypid_);
    }
    return;
}

//***************************************************************************
// add a new pair of (x,b) vectors to the projection space
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::addToProjectionSpace(HYPRE_IJVector xvec,
                                            HYPRE_IJVector bvec)
{
    int                i, k, ierr, nrows, numGlobalRows;
    int                *partition, start_row, end_row;
    double             *darray, alpha;
    HYPRE_ParVector    v_csr, x_csr, b_csr;
    HYPRE_IJVector     tmpxvec, tmpbvec;
    HYPRE_ParCSRMatrix A_csr;

    //-----------------------------------------------------------------------
    // diagnostic message
    //-----------------------------------------------------------------------

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
    {
       printf("%4d : HYPRE_LSC::addToProjectionSpace %d\n",mypid_,
              projectCurrSize_);
    }

    //-----------------------------------------------------------------------
    // initially, allocate space for B's and X's and R
    //-----------------------------------------------------------------------

    if ( projectCurrSize_ == 0 && HYpbs_ == NULL )
    {
       A_csr  = (HYPRE_ParCSRMatrix) HYPRE_IJMatrixGetLocalStorage(currA_);
       HYPRE_ParCSRMatrixGetRowPartitioning( A_csr, &partition );
       start_row     = partition[mypid_];
       end_row       = partition[mypid_+1] - 1;
       nrows         = end_row - start_row + 1;
       numGlobalRows = partition[numProcs_];

       HYpxs_ = new HYPRE_IJVector[projectSize_];
       HYpbs_ = new HYPRE_IJVector[projectSize_+1];
       for ( i = 0; i <= projectSize_; i++ )
       {
          HYPRE_IJVectorCreate(comm_, &(HYpbs_[i]), numGlobalRows_);
          HYPRE_IJVectorSetLocalStorageType(HYpbs_[i], HYPRE_PARCSR);
          HYPRE_IJVectorSetLocalPartitioning(HYpbs_[i],start_row,end_row+1);
          ierr = HYPRE_IJVectorAssemble(HYpbs_[i]);
          ierr = HYPRE_IJVectorInitialize(HYpbs_[i]);
          ierr = HYPRE_IJVectorZeroLocalComponents(HYpbs_[i]);
          assert( !ierr );
       }
       for ( i = 0; i < projectSize_; i++ )
       {
          HYPRE_IJVectorCreate(comm_, &(HYpxs_[i]), numGlobalRows_);
          HYPRE_IJVectorSetLocalStorageType(HYpxs_[i], HYPRE_PARCSR);
          HYPRE_IJVectorSetLocalPartitioning(HYpxs_[i],start_row,end_row+1);
          ierr = HYPRE_IJVectorAssemble(HYpxs_[i]);
          ierr = HYPRE_IJVectorInitialize(HYpxs_[i]);
          ierr = HYPRE_IJVectorZeroLocalComponents(HYpxs_[i]);
          assert(!ierr);
       }
       projectCurrSize_ = 0;
       projectionMatrix_ = new double*[projectSize_];
       for ( i = 0; i < projectSize_; i++ )
       {
          projectionMatrix_[i] = new double[projectSize_];
          for ( k = 0; k < projectSize_; k++ )
             projectionMatrix_[i][k] = 0.0;
       }
    }

    //-----------------------------------------------------------------------
    // if buffer has been filled, move things up
    //-----------------------------------------------------------------------

    if ( projectCurrSize_ >= projectSize_ )
    {
       projectCurrSize_--;
       darray  = projectionMatrix_[0];
       tmpxvec = HYpxs_[0];
       tmpbvec = HYpbs_[0];
       for ( i = 0; i < projectCurrSize_; i++ )
       {
          HYpbs_[i] = HYpbs_[i+1];
          HYpxs_[i] = HYpxs_[i+1];
          projectionMatrix_[i] = projectionMatrix_[i+1];
          for ( k = 0; k < projectCurrSize_; k++ )
             projectionMatrix_[i][k] = projectionMatrix_[i][k+1];
          projectionMatrix_[i][projectSize_-1] = 0.0;
       }
       for ( i = 0; i < projectSize_; i++ ) darray[i] = 0.0;
       projectionMatrix_[projectSize_-1] = darray;
       HYpxs_[projectCurrSize_] = tmpxvec;
       HYpbs_[projectCurrSize_] = tmpbvec;
    }

    //-----------------------------------------------------------------------
    // copy incoming vectors to buffer
    //-----------------------------------------------------------------------

    x_csr = (HYPRE_ParVector) HYPRE_IJVectorGetLocalStorage(xvec);
    v_csr = (HYPRE_ParVector)
            HYPRE_IJVectorGetLocalStorage(HYpxs_[projectCurrSize_]);
    HYPRE_ParVectorCopy( x_csr, v_csr );
    b_csr = (HYPRE_ParVector) HYPRE_IJVectorGetLocalStorage(bvec);
    v_csr = (HYPRE_ParVector)
            HYPRE_IJVectorGetLocalStorage(HYpbs_[projectCurrSize_]);
    HYPRE_ParVectorCopy( b_csr, v_csr );

    //-----------------------------------------------------------------------
    // compute QR decomposition
    //-----------------------------------------------------------------------

    b_csr = v_csr;
    for ( i = 0; i < projectCurrSize_; i++ )
    {
       v_csr = (HYPRE_ParVector) HYPRE_IJVectorGetLocalStorage(HYpbs_[i]);
       HYPRE_ParVectorInnerProd( b_csr,  v_csr, &alpha);
       projectionMatrix_[i][projectCurrSize_] = alpha;
       alpha = - alpha;
       hypre_ParVectorAxpy(alpha,(hypre_ParVector*)v_csr,(hypre_ParVector*)b_csr);
    }
    HYPRE_ParVectorInnerProd( b_csr,  b_csr, &alpha);
    alpha = sqrt( alpha );
    if ( alpha < 1.0e-8 ) return;
    projectionMatrix_[projectCurrSize_][projectCurrSize_] = alpha;
    alpha = 1.0 / alpha;
    hypre_ParVectorScale(alpha,(hypre_ParVector*)b_csr);

    projectCurrSize_++;

    //-----------------------------------------------------------------------
    // diagnostic message
    //-----------------------------------------------------------------------

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
    {
       printf("%4d : HYPRE_LSC::leaving addToProjectionSpace %d\n",mypid_,
               projectCurrSize_);
    }
}

