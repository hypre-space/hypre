/*BHEADER**********************************************************************
 * (c) 1999   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

#include <stdlib.h>
#include <string.h>
#include <iostream.h>
#include <stdio.h>
#include <assert.h>

#include "utilities/utilities.h"
#include "src/Data.h"
#include "other/basicTypes.h"
#include "src/Utils.h"
#include "src/LinearSystemCore.h"
#include "HYPRE_LinSysCore.h"

#define abs(x) (((x) > 0.0) ? x : -(x))

//---------------------------------------------------------------------------
// parcsr_matrix_vector.h is put here instead of in HYPRE_LinSysCore.h 
// because it gives warning when compiling cfei.cc
//---------------------------------------------------------------------------

#include "parcsr_matrix_vector/parcsr_matrix_vector.h"

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
   int HYPRE_ParCSRMLSetup( HYPRE_Solver, 
                            HYPRE_ParCSRMatrix,
                            HYPRE_ParVector, 
                            HYPRE_ParVector );
   int HYPRE_ParCSRMLSolve( HYPRE_Solver, 
                            HYPRE_ParCSRMatrix,
                            HYPRE_ParVector,   
                            HYPRE_ParVector );
   int HYPRE_ParCSRMLSetStrongThreshold( HYPRE_Solver, double );
   int HYPRE_ParCSRMLSetNumPreSmoothings( HYPRE_Solver, int );
   int HYPRE_ParCSRMLSetNumPostSmoothings( HYPRE_Solver, int );
   int HYPRE_ParCSRMLSetPreSmoother( HYPRE_Solver, int );
   int HYPRE_ParCSRMLSetPostSmoother( HYPRE_Solver, int );
   int HYPRE_ParCSRMLSetDampingFactor( HYPRE_Solver, double );
#endif

   int hypre_ParAMGBuildCoarseOperator(hypre_ParCSRMatrix*,
                                       hypre_ParCSRMatrix*,
                                       hypre_ParCSRMatrix*,
                                       hypre_ParCSRMatrix**);
   void qsort0(int *, int, int);
   void qsort1(int *, double *, int, int);

#ifdef Y12M
   void y12maf_(int*,int*,double*,int*,int*,int*,int*,double*,
                int*,int*, double*,int*,double*,int*);
#endif
}

//***************************************************************************
// constructor
//---------------------------------------------------------------------------

HYPRE_LinSysCore::HYPRE_LinSysCore(MPI_Comm comm) : 
                  LinearSystemCore(comm),
                  comm_(comm),
                  HYA_(NULL),
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
                  numGlobalRows_(0),
                  localStartRow_(0),
                  localEndRow_(-1),
                  nConstraints_(0),
                  constrList_(NULL),
                  maxIterations_(1000),
                  tolerance_(1.0e-12),
                  normAbsRel_(0),
                  systemAssembled_(0),
                  systemReduced_(0),
                  finalResNorm_(0.0),
                  rowLengths_(NULL),
                  colIndices_(NULL),
                  colValues_(NULL),
                  selectedList_(NULL),
                  selectedListAux_(NULL),
                  HYOutputLevel_(0)
{
    //-------------------------------------------------------------------
    // find my processor ID 
    //-------------------------------------------------------------------

    MPI_Comm_rank(comm, &mypid_);
    MPI_Comm_size(comm, &numProcs_);

#ifdef DEBUG
    printf("%4d : HYPRE_LinSysCore::entering constructor.\n",mypid_);
#endif

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
    strcpy(HYPreconName_,"identity");
    HYPreconID_ = HYNONE;

    //-------------------------------------------------------------------
    // parameters for controlling amg, pilut, and SuperLU
    //-------------------------------------------------------------------

    amgCoarsenType_     = 0;    // default coarsening
    amgNumSweeps_[0]    = 2;    // no. of sweeps for fine grid
    amgNumSweeps_[1]    = 2;    // no. of presmoothing sweeps 
    amgNumSweeps_[2]    = 2;    // no. of postsmoothing sweeps 
    amgNumSweeps_[3]    = 2;    // no. of sweeps for coarsest grid
    amgRelaxType_[0]    = 3;    // hybrid for the fine grid
    amgRelaxType_[1]    = 3;    // hybrid for presmoothing 
    amgRelaxType_[2]    = 3;    // hybrid for postsmoothing
    amgRelaxType_[3]    = 9;    // direct for the coarsest level
    amgStrongThreshold_ = 0.25;
    for (int i = 0; i < 25; i++) amgRelaxWeight_[i] = 0.0; 

    pilutRowSize_       = 0;    // how many nonzeros to keep in L and U
    pilutDropTol_       = 0.0;
    pilutMaxNnzPerRow_  = 0;    // register the max NNZ/per in matrix A

    parasailsNlevels_   = 1;
    parasailsThreshold_ = 0.0;
    parasailsFilter_    = 0.01;

    superluOrdering_    = 0;    // natural ordering in SuperLU
    superluScale_[0]    = 'N';  // no scaling in SuperLUX
    gmresDim_           = 200;  // restart size in GMRES
    mlNumPreSweeps_     = 3;
    mlNumPostSweeps_    = 3;
    mlPresmootherType_  = 2;    // default symmetric Gauss-Seidel
    mlPostsmootherType_ = 2;    // default symmetric Gauss-Seidel
    mlRelaxWeight_      = 0.5;
    mlStrongThreshold_  = 0.25; // one suggested by Vanek/Brezina/Mandel

    rhsIDs_             = new int[1];
    rhsIDs_[0]          = 0;

#ifdef DEBUG
    printf("%4d : HYPRE_LinSysCore::leaving  constructor.\n",mypid_);
#endif
    return;
}

//***************************************************************************
// destructor
//---------------------------------------------------------------------------

HYPRE_LinSysCore::~HYPRE_LinSysCore() 
{
#ifdef DEBUG
    printf("%4d : HYPRE_LinSysCore::entering destructor.\n",mypid_);
#endif
    //-------------------------------------------------------------------
    // clean up the allocated matrix and vectors
    //-------------------------------------------------------------------

    if ( HYA_ != NULL ) {HYPRE_IJMatrixDestroy(HYA_); HYA_ = NULL;}
    if ( HYx_ != NULL ) {HYPRE_IJVectorDestroy(HYx_); HYx_ = NULL;}
    if ( HYr_ != NULL ) {HYPRE_IJVectorDestroy(HYr_); HYr_ = NULL;}
    if ( HYbs_ != NULL ) 
    {
       for ( int i = 0; i < numRHSs_; i++ ) 
          if ( HYbs_[i] != NULL ) HYPRE_IJVectorDestroy(HYbs_[i]);
       delete [] HYbs_;
       HYbs_ = NULL;
    }
    if (reducedA_ != NULL) {HYPRE_IJMatrixDestroy(HYA_); reducedA_ = NULL;}
    if (reducedB_ != NULL) {HYPRE_IJVectorDestroy(HYx_); reducedX_ = NULL;}
    if (reducedX_ != NULL) {HYPRE_IJVectorDestroy(HYr_); reducedX_ = NULL;}
    if (reducedR_ != NULL) {HYPRE_IJVectorDestroy(HYr_); reducedR_ = NULL;}
    if (HYA21_    != NULL) {HYPRE_IJMatrixDestroy(HYA_); HYA21_    = NULL;}
    if (HYinvA22_ != NULL) {HYPRE_IJMatrixDestroy(HYA_); HYinvA22_ = NULL;}

    matrixVectorsCreated_ = 0;
    systemAssembled_ = 0;

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

    //-------------------------------------------------------------------
    // call solver destructors
    //-------------------------------------------------------------------

    if ( HYSolver_ != NULL )
    {
       if (HYSolverID_ == HYPCG)   HYPRE_ParCSRPCGDestroy(HYSolver_);
       if (HYSolverID_ == HYGMRES) HYPRE_ParCSRGMRESDestroy(HYSolver_);
       HYSolver_ = NULL;
    }
    delete [] HYSolverName_;
    HYSolverName_ = NULL;

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
          HYPRE_ParAMGDestroy( HYPrecon_ );

#ifdef MLPACK
       else if ( HYPreconID_ == HYML )
          HYPRE_ParCSRMLDestroy( HYPrecon_ );
#endif
       HYPrecon_ = NULL;
    }
    delete [] HYPreconName_;
    HYPreconName_ = NULL;

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
#ifdef DEBUG
    printf("%4d : HYPRE_LinSysCore::leaving  destructor - BYEBYE.\n",mypid_);
#endif
}

//***************************************************************************
// clone a copy of HYPRE_LinSysCore
//---------------------------------------------------------------------------

LinearSystemCore* HYPRE_LinSysCore::clone() 
{
    return(new HYPRE_LinSysCore(comm_));
}

//***************************************************************************
// this function takes parameters for setting internal things like solver
// and preconditioner choice, etc.
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::parameters(int numParams, char **params)
{
    int    i, nsweeps, rtype, nParamsFound;
    double weight;
    char   param[256], param2[80];

#ifdef DEBUG
    printf("%4d : HYPRE_LinSysCore::entering parameters function.\n",mypid_);
    if ( mypid_ == 0 )
    {
       printf("HYPRE_LinSysCore::parameters - numParams = %d\n", numParams);
       for ( int i = 0; i < numParams; i++ )
       {
          printf("           param %d = %s \n", i, params[i]);
       }
    }
#endif

    if ( numParams <= 0 ) return;
    nParamsFound = 0;

    //-------------------------------------------------------------------
    // output level
    //-------------------------------------------------------------------

    if ( Utils::getParam("outputLevel",numParams,params,param) == 1)
    {
       sscanf(param,"%d", &HYOutputLevel_);
       nParamsFound++;
#ifdef DEBUG
       if ( mypid_ == 0 )
       {
          printf("HYPRE_LinSysCore::parameters outputLevel = %d\n",
                 HYOutputLevel_);
       }
#endif
    }

    //-------------------------------------------------------------------
    // which solver to pick : cg, gmres, superlu, superlux, y12m
    //-------------------------------------------------------------------

    if ( Utils::getParam("solver",numParams,params,param) == 1)
    {
       sscanf(param,"%s",HYSolverName_);
       selectSolver(HYSolverName_);
       nParamsFound++;
#ifdef DEBUG
       if ( mypid_ == 0 )
       {
          printf("HYPRE_LinSysCore::parameters solver = %s\n",HYSolverName_);
       }
#endif
    }

    //-------------------------------------------------------------------
    // for GMRES, the restart size
    //-------------------------------------------------------------------

    if ( Utils::getParam("gmresDim",numParams,params,param) == 1)
    {
       sscanf(param,"%d", &gmresDim_);
       if ( gmresDim_ < 1 ) gmresDim_ = 200;
       nParamsFound++;
#ifdef DEBUG
       if ( mypid_ == 0 )
       {
          printf("HYPRE_LinSysCore::parameters gmresDim = %d\n",gmresDim_);
       }
#endif
    }

    //-------------------------------------------------------------------
    // which preconditioner : diagonal, pilut, boomeramg, parasails
    //-------------------------------------------------------------------

    if ( Utils::getParam("preconditioner",numParams,params,param) == 1)
    {
       sscanf(param,"%s",HYPreconName_);
       selectPreconditioner(HYPreconName_);
       nParamsFound++;
#ifdef DEBUG
       if ( mypid_ == 0 )
       {
          printf("HYPRE_LinSysCore::parameters preconditioner = %s\n",
                 HYPreconName_);
       }
#endif
    }

    //-------------------------------------------------------------------
    // maximum number of iterations for pcg or gmres
    //-------------------------------------------------------------------

    if (Utils::getParam("maxIterations",numParams,params,param) == 1)
    {
       sscanf(param,"%d", &maxIterations_);
       nParamsFound++;
#ifdef DEBUG
       if ( mypid_ == 0 )
       {
          printf("HYPRE_LinSysCore::parameters maxIterations = %d\n",
                 maxIterations_);
       }
#endif
    }

    //-------------------------------------------------------------------
    // tolerance as termination criterion
    //-------------------------------------------------------------------

    if (Utils::getParam("tolerance",numParams,params,param) == 1)
    {
       sscanf(param,"%lg", &tolerance_);
       nParamsFound++;
#ifdef DEBUG
       if ( mypid_ == 0 )
       {
          printf("HYPRE_LinSysCore::parameters tolerance = %e\n",
                 tolerance_);
       }
#endif
    }

    //-------------------------------------------------------------------
    // relative norm as termination criterion
    //-------------------------------------------------------------------

    if (Utils::getParam("relativeNorm",numParams,params,param) == 1)
    {
       normAbsRel_ = 0;
       nParamsFound++;
#ifdef DEBUG
       if ( mypid_ == 0 )
       {
          printf("HYPRE_LinSysCore::parameters relativeNorm \n");
       }
#endif
    }

    //-------------------------------------------------------------------
    // absolute norm as termination criterion
    //-------------------------------------------------------------------

    if (Utils::getParam("absoluteNorm",numParams,params,param) == 1)
    {
       normAbsRel_ = 1;
       nParamsFound++;
#ifdef DEBUG
       if ( mypid_ == 0 )
       {
          printf("HYPRE_LinSysCore::parameters absoluteNorm \n");
       }
#endif
    }

    //-------------------------------------------------------------------
    // pilut preconditioner : max number of nonzeros to keep per row
    //-------------------------------------------------------------------

    if (Utils::getParam("pilutRowSize",numParams,params,param) == 1)
    {
       sscanf(param,"%d", &pilutRowSize_);
       if ( pilutRowSize_ < 1 ) pilutRowSize_ = 50;
       nParamsFound++;
#ifdef DEBUG
       if ( mypid_ == 0 )
       {
          printf("HYPRE_LinSysCore::parameters pilutRowSize = %d\n",
                 pilutRowSize_);
       }
#endif
    }

    //-------------------------------------------------------------------
    // pilut preconditioner : threshold to drop small nonzeros
    //-------------------------------------------------------------------

    if (Utils::getParam("pilutDropTol",numParams,params,param) == 1)
    {
       sscanf(param,"%lg", &pilutDropTol_);
       if (pilutDropTol_<0.0 || pilutDropTol_ >=1.0) pilutDropTol_ = 0.0;
       nParamsFound++;
#ifdef DEBUG
       if ( mypid_ == 0 )
       {
          printf("HYPRE_LinSysCore::parameters pilutDropTol = %e\n",
                 pilutDropTol_);
       }
#endif
    }

    //-------------------------------------------------------------------
    // superlu : ordering to use (natural, mmd)
    //-------------------------------------------------------------------

    if (Utils::getParam("superluOrdering",numParams,params,param) == 1)
    {
       sscanf(param,"%s", &param2);
       if      ( !strcmp(param2, "natural" ) ) superluOrdering_ = 0;
       else if ( !strcmp(param2, "mmd") )      superluOrdering_ = 2;
       else                                    superluOrdering_ = 0;
       nParamsFound++;
#ifdef DEBUG
       if ( mypid_ == 0 )
       {
          printf("HYPRE_LinSysCore::parameters superluOrdering = %s\n",
                 param2);
       }
#endif
    }

    //-------------------------------------------------------------------
    // superlu : scaling none ('N') or both col/row ('B')
    //-------------------------------------------------------------------

    if (Utils::getParam("superluScale",numParams,params,param) == 1)
    {
       sscanf(param,"%s", &param2);
       if      ( !strcmp(param2, "y" ) ) superluScale_[0] = 'B';
       else                              superluScale_[0] = 'N';
       nParamsFound++;
#ifdef DEBUG
       if ( mypid_ == 0 )
       {
          printf("HYPRE_LinSysCore::parameters superluScale = %s\n",
                 params);
       }
#endif
    }

    //-------------------------------------------------------------------
    // amg preconditoner : coarsening type 
    //-------------------------------------------------------------------

    if (Utils::getParam("amgCoarsenType",numParams,params,param) == 1)
    {
       sscanf(param,"%s", param2);
       if      ( !strcmp(param2, "ruge" ) )    amgCoarsenType_ = 1;
       else if ( !strcmp(param2, "falgout" ) ) amgCoarsenType_ = 6;
       else if ( !strcmp(param2, "default" ) ) amgCoarsenType_ = 0;
       nParamsFound++;
#ifdef DEBUG
       if ( mypid_ == 0 )
       {
          printf("HYPRE_LinSysCore::parameters amgCoarsenType = %s\n",
                 param2);
       }
#endif
    }

    //-------------------------------------------------------------------
    // amg preconditoner : no of relaxation sweeps per level
    //-------------------------------------------------------------------

    if (Utils::getParam("amgNumSweeps",numParams,params,param) == 1)
    {
       sscanf(param,"%d", &nsweeps);
       if ( nsweeps < 1 ) nsweeps = 1;
       for ( i = 0; i < 3; i++ ) amgNumSweeps_[i] = nsweeps;
       nParamsFound++;
#ifdef DEBUG
       if ( mypid_ == 0 )
       {
          printf("HYPRE_LinSysCore::parameters amgNumSweeps = %d\n",
                 nsweeps);
       }
#endif
    }

    //-------------------------------------------------------------------
    // amg preconditoner : which smoother to use
    //-------------------------------------------------------------------

    if (Utils::getParam("amgRelaxType",numParams,params,param) == 1)
    {
       sscanf(param,"%s", param2);
       if      ( !strcmp(param2, "jacobi" ) ) rtype = 2;
       else if ( !strcmp(param2, "gsSlow") )  rtype = 1;
       else if ( !strcmp(param2, "gsFast") )  rtype = 4;
       else if ( !strcmp(param2, "hybrid" ) ) rtype = 3;
       else                                   rtype = 4;
       for ( i = 0; i < 3; i++ ) amgRelaxType_[i] = rtype;
       nParamsFound++;
#ifdef DEBUG
       if ( mypid_ == 0 )
       {
          printf("HYPRE_LinSysCore::parameters amgRelaxType = %s\n",
                 params);
       }
#endif
    }

    //-------------------------------------------------------------------
    // amg preconditoner : damping factor for Jacobi smoother
    //-------------------------------------------------------------------

    if (Utils::getParam("amgRelaxWeight",numParams,params,param) == 1)
    {
       sscanf(param,"%lg", &weight);
       if ( weight < 0.0 || weight > 1.0 ) weight = 1.0;
       for ( i = 0; i < 25; i++ ) amgRelaxWeight_[i] = weight;
       nParamsFound++;
#ifdef DEBUG
       if ( mypid_ == 0 )
       {
          printf("HYPRE_LinSysCore::parameters amgRelaxWeight = %e\n",
                 weight);
       }
#endif
    }

    //-------------------------------------------------------------------
    // amg preconditoner : threshold to determine strong coupling
    //-------------------------------------------------------------------

    if (Utils::getParam("amgStrongThreshold",numParams,params,param) == 1)
    {
       sscanf(param,"%lg", &amgStrongThreshold_);
       if ( amgStrongThreshold_ < 0.0 || amgStrongThreshold_ > 1.0 )
          amgStrongThreshold_ = 0.25;
       nParamsFound++;
#ifdef DEBUG
       if ( mypid_ == 0 )
       {
          printf("HYPRE_LinSysCore::parameters amgStrongThreshold = %e\n",
                 amgStrongThreshold_);
       }
#endif
    }

    //-------------------------------------------------------------------
    // parasails preconditoner : threshold ( >= 0.0 )
    //-------------------------------------------------------------------

    if (Utils::getParam("parasailsThreshold",numParams,params,param) == 1)
    {
       sscanf(param,"%lg", &parasailsThreshold_);
       if ( parasailsThreshold_ < 0.0 ) parasailsThreshold_ = 0.0;
       nParamsFound++;
#ifdef DEBUG
       if ( mypid_ == 0 )
       {
          printf("HYPRE_LinSysCore::parameters parasailsThreshold = %e\n",
                 parasailsThreshold_);
       }
#endif
    }

    //-------------------------------------------------------------------
    // parasails preconditoner : nlevels ( >= 1)
    //-------------------------------------------------------------------

    if (Utils::getParam("parasailsNlevels",numParams,params,param) == 1)
    {
       sscanf(param,"%d", &parasailsNlevels_);
       if ( parasailsNlevels_ < 1 ) parasailsNlevels_ = 1;
       nParamsFound++;
#ifdef DEBUG
       if ( mypid_ == 0 )
       {
          printf("HYPRE_LinSysCore::parameters parasailsNlevels = %d\n",
                 parasailsNlevels_);
       }
#endif
    }

    //-------------------------------------------------------------------
    // parasails preconditoner : filter (0.004-0.05)
    //-------------------------------------------------------------------

    if (Utils::getParam("parasailsFilter",numParams,params,param) == 1)
    {
       sscanf(param,"%lg", &parasailsFilter_);
       if ( parasailsFilter_ < 0.0 ) parasailsFilter_ = 0.0;
       nParamsFound++;
#ifdef DEBUG
       if ( mypid_ == 0 )
       {
          printf("HYPRE_LinSysCore::parameters parasailsFilter = %e\n",
                 parasailsFilter_);
       }
#endif
    }

    //-------------------------------------------------------------------
    // mlpack preconditoner : no of relaxation sweeps per level
    //-------------------------------------------------------------------

    if (Utils::getParam("mlNumPresweeps",numParams,params,param) == 1)
    {
       sscanf(param,"%d", &nsweeps);
       if ( nsweeps < 1 ) nsweeps = 1;
       mlNumPreSweeps_ = nsweeps;
       nParamsFound++;
#ifdef DEBUG
       if ( mypid_ == 0 )
       {
          printf("HYPRE_LinSysCore::parameters mlNumPresweeps = %d\n",
                 nsweeps);
       }
#endif
    }
    if (Utils::getParam("mlNumPostsweeps",numParams,params,param) == 1)
    {
       sscanf(param,"%d", &nsweeps);
       if ( nsweeps < 1 ) nsweeps = 1;
       mlNumPostSweeps_ = nsweeps;
       nParamsFound++;
#ifdef DEBUG
       if ( mypid_ == 0 )
       {
          printf("HYPRE_LinSysCore::parameters mlNumPostsweeps = %d\n",
                 nsweeps);
       }
#endif
    }
    if (Utils::getParam("mlNumSweeps",numParams,params,param) == 1)
    {
       sscanf(param,"%d", &nsweeps);
       if ( nsweeps < 1 ) nsweeps = 1;
       mlNumPreSweeps_  = nsweeps;
       mlNumPostSweeps_ = nsweeps;
       nParamsFound++;
#ifdef DEBUG
       if ( mypid_ == 0 )
       {
          printf("HYPRE_LinSysCore::parameters mlNumSweeps = %d\n",
                 nsweeps);
       }
#endif
    }

    //-------------------------------------------------------------------
    // mlpack preconditoner : which smoother to use
    //-------------------------------------------------------------------

    if (Utils::getParam("mlPresmootherType",numParams,params,param) == 1)
    {
       sscanf(param,"%s", param2);
       rtype = 1;
       if      ( !strcmp(param2, "jacobi" ) ) rtype = 0;
       else if ( !strcmp(param2, "gs") )      rtype = 1;
       else if ( !strcmp(param2, "sgs") )     rtype = 2;
       else if ( !strcmp(param2, "bgs") )     rtype = 3;
       else if ( !strcmp(param2, "bjacobi") ) rtype = 4;
       mlPresmootherType_  = rtype;
       nParamsFound++;
#ifdef DEBUG
       if ( mypid_ == 0 )
       {
          printf("HYPRE_LinSysCore::parameters mlPresmootherType = %s\n",
                 param2);
       }
#endif
    }
    if (Utils::getParam("mlPostsmootherType",numParams,params,param) == 1)
    {
       sscanf(param,"%s", param2);
       rtype = 1;
       if      ( !strcmp(param2, "jacobi" ) ) rtype = 0;
       else if ( !strcmp(param2, "gs") )      rtype = 1;
       else if ( !strcmp(param2, "sgs") )     rtype = 2;
       else if ( !strcmp(param2, "bgs") )     rtype = 3;
       else if ( !strcmp(param2, "bjacobi") ) rtype = 4;
       mlPostsmootherType_  = rtype;
       nParamsFound++;
#ifdef DEBUG
       if ( mypid_ == 0 )
       {
          printf("HYPRE_LinSysCore::parameters mlPostsmootherType = %s\n",
                 param2);
       }
#endif
    }
    if (Utils::getParam("mlRelaxType",numParams,params,param) == 1)
    {
       sscanf(param,"%s", param2);
       rtype = 1;
       if      ( !strcmp(param2, "jacobi" ) ) rtype = 0;
       else if ( !strcmp(param2, "gs") )      rtype = 1;
       else if ( !strcmp(param2, "sgs") )     rtype = 2;
       else if ( !strcmp(param2, "bgs") )     rtype = 3;
       else if ( !strcmp(param2, "bjacobi") ) rtype = 4;
       mlPresmootherType_  = rtype;
       mlPostsmootherType_ = rtype;
       nParamsFound++;
#ifdef DEBUG
       if ( mypid_ == 0 )
       {
          printf("HYPRE_LinSysCore::parameters mlRelaxType = %s\n",
                 param2);
       }
#endif
    }

    //-------------------------------------------------------------------
    // mlpack preconditoner : damping factor for Jacobi smoother
    //-------------------------------------------------------------------

    if (Utils::getParam("mlRelaxWeight",numParams,params,param) == 1)
    {
       sscanf(param,"%lg", &weight);
       if ( weight < 0.0 || weight > 1.0 ) weight = 1.0;
       mlRelaxWeight_ = weight;
       nParamsFound++;
#ifdef DEBUG
       if ( mypid_ == 0 )
       {
          printf("HYPRE_LinSysCore::parameters mlRelaxWeight = %e\n",
                 weight);
       }
#endif
    }

    //-------------------------------------------------------------------
    // mlpack preconditoner : threshold to determine strong coupling
    //-------------------------------------------------------------------

    if (Utils::getParam("mlStrongThreshold",numParams,params,param) == 1)
    {
       sscanf(param,"%lg", &mlStrongThreshold_);
       if ( mlStrongThreshold_ < 0.0 || mlStrongThreshold_ > 1.0 )
          mlStrongThreshold_ = 0.0;
       nParamsFound++;
#ifdef DEBUG
       if ( mypid_ == 0 )
       {
          printf("HYPRE_LinSysCore::parameters mlStrongThreshold = %e\n",
                 mlStrongThreshold_);
       }
#endif
    }

    //-------------------------------------------------------------------
    // error checking
    //-------------------------------------------------------------------

    if ( nParamsFound != numParams && mypid_ == 0 )
    {
       printf("HYPRE_LinSysCore::parameters WARNING - some param invalid.\n");
    }

#ifdef DEBUG
    printf("%4d : HYPRE_LinSysCore::leaving  parameters function.\n",mypid_);
#endif

    return;
}

//***************************************************************************
//This function is where we establish the structures/objects associated
//with the linear algebra library. i.e., do initial allocations, etc.
// Rows and columns are 1-based.
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::createMatricesAndVectors(int numGlobalEqns,
                                                int firstLocalEqn,
                                                int numLocalEqns) 
{
    int i, ierr;

    //-------------------------------------------------------------------
    // diagnostic message
    //-------------------------------------------------------------------

    if ( HYOutputLevel_ > 0 )
    {
       printf("%4d : HYPRE_LinSysCore::entering createMatricesAndVectors.\n", 
                     mypid_);
       printf("%4d : HYPRE_LinSysCore::startrow, endrow = %d %d\n",mypid_,
                     firstLocalEqn, firstLocalEqn+numLocalEqns-1);
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

    if ( HYOutputLevel_ > 0 )
    {
       printf("%4d : HYPRE_LinSysCore::leaving  createMatricesAndVectors.\n", 
                     mypid_);
    }
}

//***************************************************************************
// Set the number of rows in the diagonal part and off diagonal part
// of the matrix, using the structure of the matrix, stored in rows.
// rows is an array that is 0-based.  localStartRow and localEndRow are 1-based.
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::allocateMatrix(int **colIndices, int *rowLengths)
{
    int i, j, ierr, nsize, *indices, maxSize, minSize;

    if ( HYOutputLevel_ > 0 )
    {
       printf("%4d : HYPRE_LinSysCore::entering allocateMatrix.\n", mypid_);
    }

    //-------------------------------------------------------------------
    // error checking
    //-------------------------------------------------------------------

    if ( localEndRow_ < localStartRow_ ) 
    {
       printf("allocateMatrix : createMatrixAndVectors should be called\n");
       printf("                 before allocateMatrix.\n");
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

    if ( HYOutputLevel_ > 0 )
    {
       printf("%4d : allocateMatrix : max/min nnz/row = %d %d\n", mypid_, 
                     maxSize, minSize);
    }

    MPI_Allreduce(&maxSize, &pilutMaxNnzPerRow_,1,MPI_INT,MPI_MAX,comm_);

    ierr = HYPRE_IJMatrixSetRowSizes(HYA_, rowLengths_);
    ierr = HYPRE_IJMatrixInitialize(HYA_);
    assert(!ierr);

    if ( HYOutputLevel_ > 0 )
    {
       printf("%4d : HYPRE_LinSysCore::leaving  allocateMatrix.\n", mypid_);
    }
}

//***************************************************************************
// This function is needed in order to construct a new problem with the
// same sparsity pattern.
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::resetMatrixAndVector(double s)
{
    int  i, j, ierr, size;

    if ( HYOutputLevel_ > 0 )
    {
       printf("%4d : HYPRE_LinSysCore::entering resetMatrixAndVector.\n",mypid_);
    }

    if ( s != 0.0 && mypid_ == 0 )
    {
       printf("resetMatrixAndVector : cannot take nonzeros.\n");
       exit(1);
    }

    for (i = 0; i < numRHSs_; i++) HYPRE_IJVectorZeroLocalComponents(HYbs_[i]);
    systemAssembled_ = 0;

    //-------------------------------------------------------------------
    // for now, since HYPRE does not yet support
    // re-initializing the matrix, restart the whole thing
    //-------------------------------------------------------------------

    if ( HYA_ != NULL ) HYPRE_IJMatrixDestroy(HYA_);
    ierr = HYPRE_IJMatrixCreate(comm_,&HYA_,numGlobalRows_,numGlobalRows_);
    ierr = HYPRE_IJMatrixSetLocalStorageType(HYA_, HYPRE_PARCSR);
    size = localEndRow_ - localStartRow_ + 1;
    ierr = HYPRE_IJMatrixSetLocalSize(HYA_, size, size);
    ierr = HYPRE_IJMatrixSetRowSizes(HYA_, rowLengths_);
    ierr = HYPRE_IJMatrixInitialize(HYA_);
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

    if ( HYOutputLevel_ > 0 )
    {
       printf("%4d : HYPRE_LinSysCore::leaving  resetMatrixAndVector.\n",mypid_);
    }
}

//***************************************************************************
// add nonzero entries into the matrix data structure
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::sumIntoSystemMatrix(int row, int numValues,
                  const double* values, const int* scatterIndices)
{
    int i, index, colIndex, localRow;

    //-------------------------------------------------------------------
    // diagnostic message for high output level only
    //-------------------------------------------------------------------

#ifdef DEBUG
    if ( HYOutputLevel_ > 3 )
    {
       printf("%4d : HYPRE_LinSysCore::entering sumIntoSystemMatrix.\n",mypid_);
       printf("%4d : row number = %d.\n", mypid_, row);
       for ( i = 0; i < numValues; i++ )
          printf("  %4d : col = %d, data = %e\n", mypid_, scatterIndices[i], 
                  values[i]);
    }
#endif

    //-------------------------------------------------------------------
    // error checking
    //-------------------------------------------------------------------

    if ( systemAssembled_ == 1 )
    {
       printf("sumIntoSystemMatrix error : matrix already assembled\n");
       exit(1);
    }
    if ( row < localStartRow_ || row > localEndRow_ )
    {
       printf("sumIntoSystemMatrix error : invalid row number %d.\n",row);
       exit(1);
    }
    localRow = row - localStartRow_;
    if ( numValues > rowLengths_[localRow] )
    {
       printf("sumIntoSystemMatrix error : row size too large.\n");
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
          printf("%4d : sumIntoSystemMatrix warning - loading column");
          printf(" that has not been declared before - %d.\n",colIndex);
          exit(1);
       }
       colValues_[localRow][index] += values[i];
    }

#ifdef DEBUG
    if ( HYOutputLevel_ > 3 )
    {
       printf("%4d : HYPRE_LinSysCore::leaving  sumIntoSystemMatrix.\n",mypid_);
    }
#endif
}

//***************************************************************************
// input is 1-based, but HYPRE vectors are 0-based
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::sumIntoRHSVector(int num, const double* values,
                       const int* indices)
{
    int    i, ierr, *local_ind;

#ifdef DEBUG
    if ( HYOutputLevel_ > 3 )
    {
       printf("%d : HYPRE_LinSysCore::entering sumIntoRHSVector.\n", mypid_);
       for ( i = 0; i < num; i++ )
       {
          printf("%d : sumIntoRHSVector - %d = %e.\n", mypid_, indices[i], 
                       values[i]);
       }
    }
#endif

    //-------------------------------------------------------------------
    // change the incoming indices to 0-based before loading
    //-------------------------------------------------------------------

    local_ind = new int[num];
    for ( i = 0; i < num; i++ ) // change to 0-based
    {
       if ( indices[i] >= localStartRow_  && indices[i] <= localEndRow_ )
          local_ind[i] = indices[i] - 1;
       else
       {
          printf("%d : sumIntoRHSVector - index %d out of range.\n", mypid_, 
                       indices[i]);
          exit(1);
       }
    }

    ierr = HYPRE_IJVectorAddToLocalComponents(HYb_,num,local_ind,NULL,values);
    assert(!ierr);

    delete [] local_ind;

#ifdef DEBUG
    if ( HYOutputLevel_ > 3 )
    {
       printf("%d : HYPRE_LinSysCore::leaving  sumIntoRHSVector.\n", mypid_);
    }
#endif
}

//***************************************************************************
// start assembling the matrix into its internal format
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::matrixLoadComplete()
{
    int i, j, numLocalEqns, leng, eqnNum;

    if ( HYOutputLevel_ > 0 )
    {
       printf("%4d : HYPRE_LinSysCore::entering matrixLoadComplete.\n",mypid_);
    }

    //-------------------------------------------------------------------
    // load the matrix stored locally to a HYPRE matrix
    //-------------------------------------------------------------------

    numLocalEqns = localEndRow_ - localStartRow_ + 1;
    for ( i = 0; i < numLocalEqns; i++ )
    {
       eqnNum = localStartRow_ - 1 + i;
       leng   = rowLengths_[i];
       for ( j = 0; j < leng; j++ ) colIndices_[i][j]--;
       HYPRE_IJMatrixInsertRow(HYA_,leng,eqnNum,colIndices_[i],colValues_[i]);
       for ( j = 0; j < leng; j++ ) colIndices_[i][j]++;
       delete [] colValues_[i];
    }
    delete [] colValues_;
    colValues_ = NULL;

    HYPRE_IJMatrixAssemble(HYA_);
    systemAssembled_ = 1;
    currA_ = HYA_;
    currB_ = HYb_;
    currX_ = HYx_;
    currR_ = HYr_;

#ifdef PRINTMAT
    //HYPRE_ParCSRMatrix a = (HYPRE_ParCSRMatrix)
    //    HYPRE_IJMatrixGetLocalStorage(HYA_);
    //HYPRE_ParCSRMatrixPrint(a, "driver.out.a");

    HYPRE_ParCSRMatrix A_csr;
    A_csr  = (HYPRE_ParCSRMatrix) HYPRE_IJMatrixGetLocalStorage(currA_);

    int    rowSize, *colInd, nnz, nrows;
    double *colVal, value;
    char   fname[40];
    FILE   *fp;

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
    MPI_Barrier(MPI_COMM_WORLD);
    exit(1);
#endif

    if ( HYOutputLevel_ > 0 )
    {
       printf("%4d : HYPRE_LinSysCore::leaving  matrixLoadComplete.\n",mypid_);
    }
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

#ifdef DEBUG
    if ( HYOutputLevel_ > 0 )
    {
       printf("%4d : HYPRE_LinSysCore::entering enforceEssentialBC.\n",mypid_);
    }
#endif

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
       localEqnNum = globalEqn[i] - localStartRow_;
       if ( localEqnNum >= 0 && localEqnNum < numLocalRows )
       {
          rowSize = rowLengths_[localEqnNum];
          colInd  = colIndices_[localEqnNum];
          colVal  = colValues_[localEqnNum];

          for ( j = 0; j < rowSize; j++ ) 
          {
             colIndex = colInd[j];
             if ( colIndex == globalEqn[i] ) colVal[j] = 1.0;
             else                            colVal[j] = 0.0;

             if ( colIndex >= localStartRow_ && colIndex <= localEndRow_) 
             {
                if ( colIndex != globalEqn[i]) 
                {
                   rowSize2 = rowLengths_[colIndex-localStartRow_];
                   colInd2  = colIndices_[colIndex-localStartRow_];
                   colVal2  = colValues_ [colIndex-localStartRow_];

                   for( k = 0; k < rowSize2; k++ ) 
                   {
                      if ( colInd2[k] == globalEqn[i] ) 
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
          eqnNum = globalEqn[i] - 1;
          HYPRE_IJVectorSetLocalComponents(HYb_,1,&eqnNum,NULL,&rhs_term);
       }
    }
#ifdef DEBUG
    if ( HYOutputLevel_ > 0 )
    {
       printf("%4d : HYPRE_LinSysCore::leaving  enforceEssentialBC.\n",mypid_);
    }
#endif
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

#ifdef DEBUG
    if ( HYOutputLevel_ > 0 )
    {
       printf("%4d : HYPRE_LinSysCore::entering enforceRemoteEssBC.\n",mypid_);
    }
#endif

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
       localEqnNum = globalEqns[i] - localStartRow_;
       if ( localEqnNum < 0 || localEqnNum >= numLocalRows )
       {
          continue;
       }

       rowLen = rowLengths_[localEqnNum];
       colInd = colIndices_[localEqnNum];
       colVal = colValues_[localEqnNum];
       eqnNum = globalEqns[i] - 1;

       for ( j = 0; j < colIndLen[i]; j++) 
       {
          for ( k = 0; k < rowLen; k++ ) 
          {
             if (colInd[k] == colIndices[i][j]) 
             {
                HYPRE_IJVectorGetLocalComponents(HYb_,1,&eqnNum,NULL,&bval);
                bval -= ( colVal[k] * coefs[i][j] );
                colVal[k] = 0.0;
                HYPRE_IJVectorSetLocalComponents(HYb_,1,&eqnNum,NULL,&bval);
             }
          }
       }
    }

#ifdef DEBUG
    if ( HYOutputLevel_ > 0 )
    {
       printf("%4d : HYPRE_LinSysCore::leaving  enforceRemoteEssBC.\n",mypid_);
    }
#endif
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

#ifdef DEBUG
    if ( HYOutputLevel_ > 0 )
    {
       printf("%4d : HYPRE_LinSysCore::entering enforceOtherBC.\n",mypid_);
    }
#endif

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
       localEqnNum = globalEqn[i] - localStartRow_;
       if ( localEqnNum < 0 || localEqnNum >= numLocalRows )
       {
          continue;
       }

       rowSize = rowLengths_[localEqnNum];
       colVal  = colValues_[localEqnNum];
       colInd  = colIndices_[localEqnNum];

       for ( j = 0; j < rowSize; j++) 
       {
          if (colInd[j] == globalEqn[i]) 
          {
             colVal[j] += alpha[i]/beta[i];
             break;
          }
       }

       //now make the rhs modification.
       // need to fetch matrix and put it back before assembled

       eqnNum = globalEqn[i] - 1;
       HYPRE_IJVectorGetLocalComponents(HYb_,1,&eqnNum,NULL,&val);
       val += ( gamma[i] / beta[i] );
       HYPRE_IJVectorSetLocalComponents(HYb_,1,&eqnNum,NULL,&val);
   }

#ifdef DEBUG
    if ( HYOutputLevel_ > 0 )
    {
       printf("%4d : HYPRE_LinSysCore::leaving  enforceOtherBC.\n",mypid_);
    }
#endif
}

//***************************************************************************
// put the pointer to the A matrix into the Data object
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::getMatrixPtr(Data& data) 
{
   (void) data;
   printf("HYPRE_LinSysCore::getmatrixPtr - not implemented yet.\n");
   exit(1);
}

//***************************************************************************
//Overwrites the current internal matrix with a scaled copy of the
//input argument.
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::copyInMatrix(double scalar, const Data& data) 
{
   (void) scalar;
   (void) data;
   printf("HYPRE_LinSysCore::copyInMatrix - not implemented yet.\n");
   exit(1);
}

//***************************************************************************
//Passes out a scaled copy of the current internal matrix.
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::copyOutMatrix(double scalar, Data& data) 
{
   (void) scalar;
   (void) data;
   printf("HYPRE_LinSysCore::copyOutMatrix - not implemented yet.\n");
   exit(1);
}

//***************************************************************************
// add nonzero entries into the matrix data structure
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::sumInMatrix(double scalar, const Data& data) 
{
   (void) scalar;
   (void) data;
   printf("HYPRE_LinSysCore::sumInMatrix - not implemented yet.\n");
   exit(1);
}

//***************************************************************************
// get the data pointer for the right hand side
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::getRHSVectorPtr(Data& data) 
{
#ifdef DEBUG
    printf("%4d : HYPRE_LinSysCore::entering getRHSVectorPtr.\n",mypid_);
#endif

   data.setTypeName("IJ_Vector");
   data.setDataPtr((void*) HYb_);

#ifdef DEBUG
    printf("%4d : HYPRE_LinSysCore::leaving  getRHSVectorPtr.\n",mypid_);
#endif
}

//***************************************************************************

void HYPRE_LinSysCore::copyInRHSVector(double scalar, const Data& data) 
{
    if ( HYOutputLevel_ > 0 )
    {
       printf("%4d : HYPRE_LinSysCore::entering copyInRHSVector.\n",mypid_);
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

    if ( HYOutputLevel_ > 0 )
    {
       printf("%4d : HYPRE_LinSysCore::leaving  copyInRHSVector.\n",mypid_);
    }
}

//***************************************************************************

void HYPRE_LinSysCore::copyOutRHSVector(double scalar, Data& data) 
{
    int ierr;

    if ( HYOutputLevel_ > 0 )
    {
       printf("%4d : HYPRE_LinSysCore::entering copyOutRHSVector.\n",mypid_);
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

    if ( HYOutputLevel_ > 0 )
    {
       printf("%4d : HYPRE_LinSysCore::leaving  copyOutRHSVector.\n",mypid_);
    }
}

//***************************************************************************

void HYPRE_LinSysCore::sumInRHSVector(double scalar, const Data& data) 
{
    if ( HYOutputLevel_ > 0 )
    {
       printf("%4d : HYPRE_LinSysCore::entering sumInRHSVector.\n",mypid_);
    }

    if (strcmp("IJ_Vector", data.getTypeName()))
    {
       printf("sumInRHSVector: data's type string not 'IJ_Vector'.\n");
       exit(1);
    }

    HYPRE_IJVector inVec = (HYPRE_IJVector) data.getDataPtr();
    HYPRE_ParVector xVec = 
          (HYPRE_ParVector) HYPRE_IJVectorGetLocalStorage(inVec);
    HYPRE_ParVector yVec = 
          (HYPRE_ParVector) HYPRE_IJVectorGetLocalStorage(HYb_);
 
    hypre_ParVectorAxpy(scalar,(hypre_ParVector*)xVec,(hypre_ParVector*)yVec);
 
    if ( HYOutputLevel_ > 0 )
    {
       printf("%4d : HYPRE_LinSysCore::leaving  sumInRHSVector.\n",mypid_);
    }
}

//***************************************************************************

void HYPRE_LinSysCore::destroyMatrixData(Data& data) 
{
    if ( HYOutputLevel_ > 0 )
    {
       printf("%4d : HYPRE_LinSysCore::entering destroyMatrixData.\n",mypid_);
    }

    if (strcmp("IJ_Matrix", data.getTypeName()))
    {
       printf("destroyMatrixData: data doesn't contain a IJ_Matrix.\n");
       exit(1);
    }
    HYPRE_IJMatrix mat = (HYPRE_IJMatrix) data.getDataPtr();
    HYPRE_IJMatrixDestroy(mat);

    if ( HYOutputLevel_ > 0 )
    {
       printf("%4d : HYPRE_LinSysCore::leaving  destroyMatrixData.\n",mypid_);
    }
}

//***************************************************************************

void HYPRE_LinSysCore::destroyVectorData(Data& data) 
{
    if ( HYOutputLevel_ > 0 )
    {
       printf("%4d : HYPRE_LinSysCore::entering destroyVectorData.\n",mypid_);
    }

    if (strcmp("IJ_Vector", data.getTypeName()))
    {
       printf("destroyVectorData: data doesn't contain a IJ_Vector.");
       exit(1);
    }

    HYPRE_IJVector vec = (HYPRE_IJVector) data.getDataPtr();
    if ( vec != NULL ) HYPRE_IJVectorDestroy(vec);

    if ( HYOutputLevel_ > 0 )
    {
       printf("%4d : HYPRE_LinSysCore::leaving  destroyVectorData.\n",mypid_);
    }
}

//***************************************************************************

void HYPRE_LinSysCore::setNumRHSVectors(int numRHSs, const int* rhsIDs) 
{
    if ( HYOutputLevel_ > 0 )
    {
       printf("%4d : HYPRE_LinSysCore::entering setNumRHSVectors.\n",mypid_);
       printf("%4d : HYPRE_LinSysCore::incoming numRHSs = %d\n",mypid_,numRHSs);
       printf("%4d : setNumRHSVectors - hardwired to 1 rhs.\n",mypid_);
    }

    if ( matrixVectorsCreated_ )
    {
       printf("setNumRHSVectors ERROR : createMatrixAndVectors called.\n");
       exit(1);
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
 
    for ( int i = 0; i < numRHSs; i++ ) rhsIDs_[i] = rhsIDs[i];

    if ( HYOutputLevel_ > 0 )
    {
       printf("%4d : HYPRE_LinSysCore::leaving  setNumRHSVectors.\n",mypid_);
    }
}

//***************************************************************************

void HYPRE_LinSysCore::setRHSID(int rhsID) 
{
    if ( HYOutputLevel_ > 0 )
    {
       printf("%4d : HYPRE_LinSysCore::entering setRHSID.\n",mypid_);
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

    if ( HYOutputLevel_ > 0 )
    {
       printf("%4d : HYPRE_LinSysCore::leaving  setRHSID.\n",mypid_);
    }
}

//***************************************************************************
// used for initializing the initial guess
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::putInitialGuess(const int* eqnNumbers,
                                       const double* values, int leng) 
{
    int i, ierr, *local_ind;

#ifdef DEBUG
    if ( HYOutputLevel_ > 3 )
    {
       printf("%4d : HYPRE_LinSysCore::entering putInitalGuess.\n",mypid_);
    }
#endif 

    local_ind = new int[leng];
    for ( i = 0; i < leng; i++ ) // change to 0-based
    {
       if (eqnNumbers[i] >= localStartRow_ && eqnNumbers[i] <= localEndRow_)
          local_ind[i] = eqnNumbers[i] - 1;
       else
       {
          printf("%d : putInitialGuess - index %d out of range\n", mypid_,
                       eqnNumbers[i]);
          exit(1);
       }
    }

    ierr = HYPRE_IJVectorSetLocalComponents(HYx_,leng,local_ind,NULL,values);
    assert(!ierr);

    delete [] local_ind;

#ifdef DEBUG
    if ( HYOutputLevel_ > 3 )
    {
       printf("%4d : HYPRE_LinSysCore::leaving  putInitalGuess.\n",mypid_);
    }
#endif 
}

//***************************************************************************
// used for getting the solution out of the solver, and into the application
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::getSolution(int* eqnNumbers, double* answers,int leng) 
{
    int    i, ierr, *equations;

#ifdef DEBUG
    if ( HYOutputLevel_ > 3 )
    {
       printf("%4d : HYPRE_LinSysCore::entering getSolution.\n",mypid_);
    }
#endif 

    equations = new int[leng];

    for ( i = 0; i < leng; i++ )
    {
       equations[i] = eqnNumbers[i] - 1; // construct 0-based index
       if ( equations[i] < localStartRow_ || equations[i] > localEndRow_ )
       {
          printf("%d : getSolution - index out of range = %d.\n", mypid_, 
                       eqnNumbers[i]);
          exit(1);
       }
    }
    ierr = HYPRE_IJVectorGetLocalComponents(HYx_,leng,equations,NULL,answers);
    assert(!ierr);
    delete [] equations;

#ifdef DEBUG
    if ( HYOutputLevel_ > 3 )
    {
       printf("%4d : HYPRE_LinSysCore::leaving  getSolution.\n",mypid_);
    }
#endif 
}

//***************************************************************************
// used for getting the solution out of the solver, and into the application
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::getSolnEntry(int eqnNumber, double& answer) 
{
    double val;
    int    ierr, equation;

#ifdef DEBUG
    if ( HYOutputLevel_ > 3 )
    {
       printf("%4d : HYPRE_LinSysCore::entering getSolnEntry.\n",mypid_);
    }
#endif 

    equation = eqnNumber - 1; // construct 0-based index
    if ( equation < localStartRow_ && equation > localEndRow_ )
    {
       printf("%d : getSolnEntry - index out of range = %d.\n", mypid_, 
                    eqnNumber);
       exit(1);
    }

    ierr = HYPRE_IJVectorGetLocalComponents(HYx_,1,&equation,NULL,&val);
    assert(!ierr);
    answer = val;

#ifdef DEBUG
    if ( HYOutputLevel_ > 3 )
    {
       printf("%4d : HYPRE_LinSysCore::leaving  getSolnEntry.\n",mypid_);
    }
#endif 
}

//***************************************************************************
// select which Krylov solver to use
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::selectSolver(char* name) 
{
    if ( HYOutputLevel_ > 0 )
    {
       printf("%4d : HYPRE_LinSysCore::entering selectSolver.\n",mypid_);
       printf("%4d : HYPRE_LinSysCore::solver name = %s.\n",mypid_,name);
    }

    //-------------------------------------------------------------------
    // if already been allocated, destroy it first
    //-------------------------------------------------------------------

    if ( HYSolver_ != NULL )
    {
       if ( HYSolverID_ == HYPCG )   HYPRE_ParCSRPCGDestroy(HYSolver_);
       if ( HYSolverID_ == HYGMRES ) HYPRE_ParCSRGMRESDestroy(HYSolver_);
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
    else
    {
       printf("HYPRE_LinSysCore selectSolver : use default = gmres.\n");
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
            //HYPRE_ParCSRPCGSetTwoNorm(HYSolver_, 1);
            //HYPRE_ParCSRPCGSetRelChange(HYSolver_, 0);
            //HYPRE_ParCSRPCGSetLogging(HYSolver_, 1);
            break;

       case HYGMRES :
            HYPRE_ParCSRGMRESCreate(comm_, &HYSolver_);
            //HYPRE_ParCSRGMRESSetLogging(HYSolver_, 1);
            break;
    }

    if ( HYOutputLevel_ > 0 )
    {
       printf("%4d : HYPRE_LinSysCore::leaving  selectSolver.\n",mypid_);
    }
    return;
}

//***************************************************************************
// select which preconditioner to use
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::selectPreconditioner(char *name)
{
    int ierr;

    if ( HYOutputLevel_ > 0 )
    {
       printf("%4d : HYPRE_LinSysCore::entering selectPreconditioner = %s.\n",
              mypid_, name);
    }

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
          HYPRE_ParAMGDestroy( HYPrecon_ );

#ifdef MLPACK
       else if ( HYPreconID_ == HYML )
          HYPRE_ParCSRMLDestroy( HYPrecon_ );
#endif
    }

    //-------------------------------------------------------------------
    // check for the validity of the preconditioner name
    //-------------------------------------------------------------------

    if ( !strcmp(name, "identity" ) )
    {
       strcpy( HYPreconName_, name );
       HYPreconID_ = HYNONE;
    }
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
    else if ( !strcmp(name, "ml") )
    {
#ifdef MLPACK
       strcpy( HYPreconName_, name );
       HYPreconID_ = HYML;
#else
       printf("selectPreconditioner - MLPACK not declared.\n");
       printf("                       set default to identity.\n");
       strcpy( HYPreconName_, "identity" );
       HYPreconID_ = HYNONE;
#endif
    }
    else
    {
       printf("selectPreconditioner error : invalid option.\n");
       printf("                     use default = identity.\n");
       strcpy( HYPreconName_, "identity" );
       HYPreconID_ = HYNONE;
    }
    if ( HYSolverID_ != HYPCG && HYSolverID_ != HYGMRES ) 
    {
       strcpy( HYPreconName_, "identity" );
       HYPreconID_ = HYNONE;
    }

    //-------------------------------------------------------------------
    // instantiate preconditioner
    //-------------------------------------------------------------------

    switch ( HYPreconID_ )
    {
       case HYNONE :
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
            HYPRE_ParAMGCreate(&HYPrecon_);
            HYPRE_ParAMGSetMaxIter(HYPrecon_, 1);
            HYPRE_ParAMGSetCycleType(HYPrecon_, 1);
            HYPRE_ParAMGSetMaxLevels(HYPrecon_, 25);
            HYPRE_ParAMGSetMeasureType(HYPrecon_, 0);
            break;

#ifdef MLPACK
       case HYML :
            ierr = HYPRE_ParCSRMLCreate( comm_, &HYPrecon_ );
            break;
#endif
    }

    if ( HYOutputLevel_ > 0 )
    {
       printf("%4d : HYPRE_LinSysCore::leaving  selectPreconditioner.\n",mypid_);
    }
}

//***************************************************************************
// solve the linear system
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::launchSolver(int& solveStatus, int &iterations)
{
    int                i, j, num_iterations, status, *num_sweeps, *relax_type;
    int                ierr, localNRows, rowNum, index, x2NRows, x2GlobalNRows;
    int                startRow, *int_array, *gint_array, startRow2;
    int                globalNConstrs, rowSize, *colInd, nnz, nrows;
    double             rnorm, *relax_wt, ddata, *colVal, value;
    char               fname[40];
    FILE               *fp;
    HYPRE_ParCSRMatrix A_csr;
    HYPRE_ParVector    x_csr;
    HYPRE_ParVector    b_csr;
    HYPRE_ParVector    r_csr;

    if ( HYOutputLevel_ > 0 )
    {
       printf("%4d : HYPRE_LinSysCore::entering launchSolver.\n", mypid_);
    }

    //*******************************************************************
    // temporary kludge before FEI adds functions to address this
    //-------------------------------------------------------------------

#ifdef DEBUG
    
    printf("%4d : HYPRE_LinSysCore::launchSolver - currently force", mypid_);
    printf(" reduction if loadConstraintNumbers is used.\n");

#endif

    MPI_Allreduce(&nConstraints_, &globalNConstrs,1,MPI_INT,MPI_SUM,comm_);
    
    if ( globalNConstrs != 0 )
    {
       if ( constrList_ != NULL ) delete [] constrList_;
       constrList_ = NULL;
       buildReducedSystem();
    }
    
    //*******************************************************************
    // fetch matrix and vector pointers
    //-------------------------------------------------------------------

    A_csr  = (HYPRE_ParCSRMatrix) HYPRE_IJMatrixGetLocalStorage(currA_);
    x_csr  = (HYPRE_ParVector)    HYPRE_IJVectorGetLocalStorage(currX_);
    b_csr  = (HYPRE_ParVector)    HYPRE_IJVectorGetLocalStorage(currB_);
    r_csr  = (HYPRE_ParVector)    HYPRE_IJVectorGetLocalStorage(currR_);

    //*******************************************************************
    // program segment for diagnostics
    //-------------------------------------------------------------------

#ifdef PRINTMAT

    if ( systemReduced_ == 1 )
    {
       x2NRows = 2 * nConstraints_;
       MPI_Allreduce(&x2NRows, &x2GlobalNRows,1,MPI_INT,MPI_SUM,comm_);
       int_array = new int[numProcs_];
       gint_array = new int[numProcs_];
       for ( i = 0; i < numProcs_; i++ ) int_array[i] = 0;
       int_array[mypid_] = 2 * nConstraints_;
       MPI_Allreduce(int_array, gint_array, numProcs_,MPI_INT, MPI_SUM, comm_);
       rowNum = 0;
       for ( i = 0; i < mypid_; i++ ) rowNum += gint_array[i];
       startRow = localStartRow_ - 1 - rowNum;
       delete [] int_array;
       delete [] gint_array;
       nrows = localEndRow_ - localStartRow_ + 1 - 2 * nConstraints_;
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
    MPI_Barrier(MPI_COMM_WORLD);
    exit(0);

#endif

    //*******************************************************************
    // choose PCG, GMRES or direct solver
    //-------------------------------------------------------------------

    status = 1;

    switch ( HYSolverID_ )
    {

       //----------------------------------------------------------------
       // choose PCG 
       //----------------------------------------------------------------

       case HYPCG :

          if (HYOutputLevel_ > 0) printf("%4d : lauchSolver(PCG) \n", mypid_);
          switch ( HYPreconID_ )
          {
             case HYDIAGONAL :
                  HYPRE_ParCSRPCGSetPrecond(HYSolver_,
                                 HYPRE_ParCSRDiagScale,
                                 HYPRE_ParCSRDiagScaleSetup,
                                 HYPrecon_);
                  break;

             case HYPILUT :
                  if ( pilutRowSize_ == 0 )
                  {
                     pilutRowSize_ = (int) (1.2 * pilutMaxNnzPerRow_);
                     if ( HYOutputLevel_ > 0 )
                     {
                        printf("PILUT - row size = %d\n", pilutRowSize_);
                        printf("PILUT - drop tol = %e\n", pilutDropTol_);
                     }
                  }
                  HYPRE_ParCSRPilutSetFactorRowSize(HYPrecon_,pilutRowSize_);
                  HYPRE_ParCSRPilutSetDropTolerance(HYPrecon_,pilutDropTol_);
                  HYPRE_ParCSRPCGSetPrecond(HYSolver_,
                                            HYPRE_ParCSRPilutSolve,
                                            HYPRE_ParCSRPilutSetup,
                                            HYPrecon_);
                  break;

             case HYPARASAILS :
                  if ( HYOutputLevel_ > 0 && mypid_ == 0 )
                  {
                     printf("ParaSails - nlevels   = %d\n", parasailsNlevels_);
                     printf("ParaSails - threshold = %e\n", parasailsThreshold_);
                     printf("ParaSails - filter    = %e\n", parasailsFilter_);
                  }
                  HYPRE_ParCSRParaSailsSetParams(HYPrecon_,
                                                 parasailsThreshold_,
                                                 parasailsNlevels_);
                  if ( parasailsThreshold_ == 0.0 ) 
                  {
                     HYPRE_ParCSRParaSailsSelectThresh(HYPrecon_, 0.9);
                  }
                  if ( parasailsFilter_ > 0.0 ) 
                  {
                     HYPRE_ParCSRParaSailsSetFilter(HYPrecon_,parasailsFilter_);
                  }
                  HYPRE_ParCSRPCGSetPrecond(HYSolver_,
                                            HYPRE_ParCSRParaSailsSolve,
                                            HYPRE_ParCSRParaSailsSetup,
                                            HYPrecon_);
                  break;

             case HYBOOMERAMG :
                  HYPRE_ParAMGSetCoarsenType(HYPrecon_, amgCoarsenType_);
                  HYPRE_ParAMGSetStrongThreshold(HYPrecon_,
                                                 amgStrongThreshold_);
                  num_sweeps = hypre_CTAlloc(int,4);
                  for ( i = 0; i < 4; i++ ) num_sweeps[i] = amgNumSweeps_[i];

                  HYPRE_ParAMGSetNumGridSweeps(HYPrecon_, num_sweeps);
                  relax_type = hypre_CTAlloc(int,4);
                  for ( i = 0; i < 4; i++ ) relax_type[i] = amgRelaxType_[i];

                  HYPRE_ParAMGSetGridRelaxType(HYPrecon_, relax_type);
                  relax_wt = hypre_CTAlloc(double,25);
                  for ( i = 0; i < 25; i++ ) relax_wt[i] = amgRelaxWeight_[i];
                  HYPRE_ParAMGSetRelaxWeight(HYPrecon_, relax_wt);
                  if ( HYOutputLevel_ > 0 && mypid_ == 0 )
                  {
                     printf("AMG coarsen type = %d\n", amgCoarsenType_);
                     printf("AMG threshold    = %e\n", amgStrongThreshold_);
                     printf("AMG numsweeps    = %d\n", amgNumSweeps_[0]);
                     printf("AMG relax type   = %d\n", amgRelaxType_[0]);
                     printf("AMG relax weight = %e\n", amgRelaxWeight_[0]);
                  }
                  if ( HYOutputLevel_ > 2 && mypid_ == 0 )
                     HYPRE_ParAMGSetIOutDat(HYPrecon_, 2);
                  HYPRE_ParCSRPCGSetPrecond(HYSolver_, HYPRE_ParAMGSolve,
                                   HYPRE_ParAMGSetup, HYPrecon_);
                  break;

#ifdef MLPACK
             case HYML :

                  HYPRE_ParCSRMLSetStrongThreshold(HYPrecon_,
                                                   mlStrongThreshold_);
                  HYPRE_ParCSRMLSetNumPreSmoothings(HYPrecon_,
                                                    mlNumPreSweeps_);
                  HYPRE_ParCSRMLSetNumPostSmoothings(HYPrecon_,
                                                     mlNumPostSweeps_);
                  HYPRE_ParCSRMLSetPreSmoother(HYPrecon_,
                                               mlPresmootherType_);
                  HYPRE_ParCSRMLSetPostSmoother(HYPrecon_,
                                                mlPostsmootherType_);
                  HYPRE_ParCSRMLSetDampingFactor(HYPrecon_, mlRelaxWeight_);
                  HYPRE_ParCSRPCGSetPrecond(HYSolver_,
                                            HYPRE_ParCSRMLSolve,
                                            HYPRE_ParCSRMLSetup,
                                            HYPrecon_);
                  if ( HYOutputLevel_ > 0 && mypid_ == 0 )
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
          if ( normAbsRel_ == 0 ) HYPRE_ParCSRPCGSetRelChange(HYSolver_, 0);
          else                    HYPRE_ParCSRPCGSetTwoNorm(HYSolver_, 1);
          HYPRE_ParCSRPCGSetup(HYSolver_, A_csr, b_csr, x_csr);
          HYPRE_ParCSRPCGSolve(HYSolver_, A_csr, b_csr, x_csr);
          HYPRE_ParCSRPCGGetNumIterations(HYSolver_, &num_iterations);
          HYPRE_ParVectorCopy( b_csr, r_csr );
          HYPRE_ParCSRMatrixMatvec( -1.0, A_csr, x_csr, 1.0, r_csr );
          HYPRE_ParVectorInnerProd( r_csr, r_csr, &rnorm);
          rnorm = sqrt( rnorm );
          iterations = num_iterations;
          if ( mypid_ == 0 )
          {
             printf("launchSolver(PCG) - NO. ITERATION = %d\n",num_iterations);
             printf("launchSolver(PCG) - FINAL NORM    = %e.\n", rnorm);
          }
          if ( num_iterations >= maxIterations_ ) status = 0;
          break;

       //----------------------------------------------------------------
       // choose GMRES 
       //----------------------------------------------------------------

       case HYGMRES :

          if (HYOutputLevel_ > 0) printf("%4d : lauchSolver(GMRES) \n",mypid_);
          switch ( HYPreconID_ )
          {
             case HYDIAGONAL :
                  HYPRE_ParCSRGMRESSetPrecond(HYSolver_,
                                              HYPRE_ParCSRDiagScale,
                                              HYPRE_ParCSRDiagScaleSetup,
                                              HYPrecon_);
                  break;

             case HYPILUT :
                  if ( pilutRowSize_ == 0 )
                  {
                     pilutRowSize_ = (int) (1.2 * pilutMaxNnzPerRow_);
                  }
                  if ( HYOutputLevel_ > 0 && mypid_ == 0 )
                  {
                     printf("PILUT - row size = %d\n", pilutRowSize_);
                     printf("PILUT - drop tol = %e\n", pilutDropTol_);
                  }
                  HYPRE_ParCSRPilutSetFactorRowSize(HYPrecon_,pilutRowSize_);
                  HYPRE_ParCSRPilutSetDropTolerance(HYPrecon_,pilutDropTol_);
                  HYPRE_ParCSRGMRESSetPrecond(HYSolver_,
                                            HYPRE_ParCSRPilutSolve,
                                            HYPRE_ParCSRPilutSetup,
                                            HYPrecon_);
                  break;

             case HYPARASAILS :
                  if ( HYOutputLevel_ > 0 && mypid_ == 0 )
                  {
                     printf("ParaSails - nlevels   = %d\n", parasailsNlevels_);
                     printf("ParaSails - threshold = %e\n", parasailsThreshold_);
                     printf("ParaSails - filter    = %e\n", parasailsFilter_);
                  }
                  HYPRE_ParCSRParaSailsSetSym(HYPrecon_,0);
                  HYPRE_ParCSRParaSailsSetParams(HYPrecon_,
                                                 parasailsThreshold_,
                                                 parasailsNlevels_);
                  if ( parasailsThreshold_ == 0.0 ) 
                  {
                     HYPRE_ParCSRParaSailsSelectThresh(HYPrecon_, 0.9);
                  }
                  if ( parasailsFilter_ > 0.0 ) 
                  {
                     HYPRE_ParCSRParaSailsSetFilter(HYPrecon_,parasailsFilter_);
                  }
                  HYPRE_ParCSRGMRESSetPrecond(HYSolver_,
                                            HYPRE_ParCSRParaSailsSolve,
                                            HYPRE_ParCSRParaSailsSetup,
                                            HYPrecon_);
                  break;

             case HYBOOMERAMG :
                  HYPRE_ParAMGSetCoarsenType(HYPrecon_, amgCoarsenType_);
                  HYPRE_ParAMGSetStrongThreshold(HYPrecon_,
                                                 amgStrongThreshold_);
                  num_sweeps = hypre_CTAlloc(int,4);
                  for ( i = 0; i < 4; i++ ) num_sweeps[i] = amgNumSweeps_[i];

                  HYPRE_ParAMGSetNumGridSweeps(HYPrecon_, num_sweeps);
                  relax_type = hypre_CTAlloc(int,4);
                  for ( i = 0; i < 4; i++ ) relax_type[i] = amgRelaxType_[i];

                  HYPRE_ParAMGSetGridRelaxType(HYPrecon_, relax_type);
                  relax_wt = hypre_CTAlloc(double,25);
                  for ( i = 0; i < 25; i++ ) relax_wt[i] = amgRelaxWeight_[i];
                  HYPRE_ParAMGSetRelaxWeight(HYPrecon_, relax_wt);
                  if ( HYOutputLevel_ > 0 && mypid_ == 0 )
                  {
                     printf("AMG coarsen type = %d\n", amgCoarsenType_);
                     printf("AMG threshold    = %e\n", amgStrongThreshold_);
                     printf("AMG numsweeps    = %d\n", amgNumSweeps_[0]);
                     printf("AMG relax type   = %d\n", amgRelaxType_[0]);
                     printf("AMG relax weight = %e\n", amgRelaxWeight_[0]);
                  }
                  if ( HYOutputLevel_ > 2 && mypid_ == 0 )
                     HYPRE_ParAMGSetIOutDat(HYPrecon_, 2);
                  HYPRE_ParCSRGMRESSetPrecond(HYSolver_, HYPRE_ParAMGSolve,
                                   HYPRE_ParAMGSetup, HYPrecon_);
                  break;

#ifdef MLPACK
             case HYML :

                  HYPRE_ParCSRMLSetStrongThreshold(HYPrecon_,
                                                   mlStrongThreshold_);
                  HYPRE_ParCSRMLSetNumPreSmoothings(HYPrecon_,
                                                    mlNumPreSweeps_);
                  HYPRE_ParCSRMLSetNumPostSmoothings(HYPrecon_,
                                                     mlNumPostSweeps_);
                  HYPRE_ParCSRMLSetPreSmoother(HYPrecon_,
                                               mlPresmootherType_);
                  HYPRE_ParCSRMLSetPostSmoother(HYPrecon_,
                                                mlPostsmootherType_);
                  HYPRE_ParCSRMLSetDampingFactor(HYPrecon_, mlRelaxWeight_);
                  HYPRE_ParCSRGMRESSetPrecond(HYSolver_,
                                            HYPRE_ParCSRMLSolve,
                                            HYPRE_ParCSRMLSetup,
                                            HYPrecon_);
                  if ( HYOutputLevel_ > 0 && mypid_ == 0 )
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
          //if ( normAbsRel_ == 0 ) HYPRE_ParCSRGMRESSetStopCrit(HYsolver_,0);
          //else                    HYPRE_ParCSRGMRESSetStopCrit(HYsolver_,1);
          HYPRE_ParCSRGMRESSetup(HYSolver_, A_csr, b_csr, x_csr);
          HYPRE_ParCSRGMRESSolve(HYSolver_, A_csr, b_csr, x_csr);
          HYPRE_ParCSRGMRESGetNumIterations(HYSolver_, &num_iterations);
          HYPRE_ParVectorCopy( b_csr, r_csr );
          HYPRE_ParCSRMatrixMatvec( -1.0, A_csr, x_csr, 1.0, r_csr );
          HYPRE_ParVectorInnerProd( r_csr, r_csr, &rnorm);
          iterations = num_iterations;
          rnorm = sqrt( rnorm );
          if ( mypid_ == 0 )
          {
             printf("launchSolver(GMRES) - NO. ITERATION = %d\n",
                                 num_iterations);
             printf("launchSolver(GMRES) - FINAL NORM    = %e\n", rnorm);
          }
          if ( num_iterations >= maxIterations_ ) status = 0;
          break;

       //----------------------------------------------------------------
       // choose SuperLU (single processor) 
       //----------------------------------------------------------------

       case HYSUPERLU :

          if (HYOutputLevel_ > 0) printf("%4d : launchSolver(SuperLU)\n",mypid_);
          solveUsingSuperLU(status);
          iterations = 1;
          //printf("SuperLU solver - return status = %d\n",status);
          break;

       //----------------------------------------------------------------
       // choose SuperLU (single processor) 
       //----------------------------------------------------------------

       case HYSUPERLUX :

          if (HYOutputLevel_ > 0) printf("%4d : launchSolver(SuperLUX)\n",mypid_);
          solveUsingSuperLUX(status);
          iterations = 1;
          //printf("SuperLUX solver - return status = %d\n",status);
          break;

       //----------------------------------------------------------------
       // choose Y12M (single processor) 
       //----------------------------------------------------------------

       case HYY12M :

#ifdef Y12M
          if (HYOutputLevel_ > 0) printf("%4d : launchSolver(Y12M)\n",mypid_);
          solveUsingY12M(status);
          iterations = 1;
          //printf("Y12M solver - return status = %d\n",status);
          break;

#else
          printf("HYPRE_LinSysCore : Y12M not available. \n");
          break; 
#endif
    }

    //*******************************************************************
    // register solver return information
    //-------------------------------------------------------------------

    solveStatus = status;
    iterations = num_iterations;

    //*******************************************************************
    // recover solution for reduced system
    //-------------------------------------------------------------------

    HYPRE_ParCSRMatrix A21_csr, A22_csr;
    HYPRE_IJVector     R1, x2;
    HYPRE_ParVector    x2_csr;

    if ( systemReduced_ == 1 )
    {
       if ( HYA21_ == NULL || HYinvA22_ == NULL )
       {
          printf("launchSolver ERROR : A21 or A22 absent.\n");
          exit(1);
       }
       else
       {
          //-------------------------------------------------------------
          // compute A21 * sol
          //-------------------------------------------------------------

          int_array = new int[numProcs_];
          gint_array = new int[numProcs_];
          x2NRows = 2 * nConstraints_;
          for ( i = 0; i < numProcs_; i++ ) int_array[i] = 0;
          int_array[mypid_] = x2NRows;
          MPI_Allreduce(int_array,gint_array,numProcs_,MPI_INT,MPI_SUM,comm_);
          x2GlobalNRows = 0;
          for ( i = 0; i < numProcs_; i++ ) x2GlobalNRows += gint_array[i];
          rowNum = 0;
          for ( i = 0; i < mypid_; i++ ) rowNum += gint_array[i];
          startRow = rowNum;
          startRow2 = localStartRow_ - 1 - rowNum;
          delete [] int_array;
          delete [] gint_array;
  
          ierr = HYPRE_IJVectorCreate(comm_, &R1, x2GlobalNRows);
          ierr = HYPRE_IJVectorSetLocalStorageType(R1, HYPRE_PARCSR);
          HYPRE_IJVectorSetLocalPartitioning(R1,startRow,startRow+x2NRows);
          ierr = HYPRE_IJVectorAssemble(R1);
          ierr = HYPRE_IJVectorInitialize(R1);
          ierr = HYPRE_IJVectorZeroLocalComponents(R1);
          assert(!ierr);

          A21_csr = (HYPRE_ParCSRMatrix) HYPRE_IJMatrixGetLocalStorage(HYA21_);
          x_csr   = (HYPRE_ParVector)    HYPRE_IJVectorGetLocalStorage(currX_);
          r_csr   = (HYPRE_ParVector)    HYPRE_IJVectorGetLocalStorage(R1);

          HYPRE_ParCSRMatrixMatvec( -1.0, A21_csr, x_csr, 0.0, r_csr );

          //-------------------------------------------------------------
          // f2 - A21 * sol
          //-------------------------------------------------------------

          for ( i = 0; i < nConstraints_; i++ )
          {
             for ( j = 0; j < nConstraints_; j++ ) 
             {
                if ( selectedListAux_[j] == i ) 
                {
                   index = selectedList_[j]; 
                   break;
                }
             }
             HYPRE_IJVectorGetLocalComponents(HYb_, 1, &index, NULL, &ddata);
             HYPRE_IJVectorAddToLocalComponents(R1,1,&rowNum,NULL,&ddata);
             rowNum++;
          }
          for ( i = localEndRow_-nConstraints_; i < localEndRow_; i++ )
          {
             HYPRE_IJVectorGetLocalComponents(HYb_, 1, &i, NULL, &ddata);
             HYPRE_IJVectorAddToLocalComponents(R1,1,&rowNum,NULL,&ddata);
             rowNum++;
          } 

          //-------------------------------------------------------------
          // inv(A22) * (f2 - A21 * sol)
          //-------------------------------------------------------------

          ierr = HYPRE_IJVectorCreate(comm_, &x2, x2GlobalNRows);
          ierr = HYPRE_IJVectorSetLocalStorageType(x2, HYPRE_PARCSR);
          HYPRE_IJVectorSetLocalPartitioning(x2,startRow,startRow+x2NRows);
          ierr = HYPRE_IJVectorAssemble(x2);
          ierr = HYPRE_IJVectorInitialize(x2);
          ierr = HYPRE_IJVectorZeroLocalComponents(x2);
          assert(!ierr);
          A22_csr = (HYPRE_ParCSRMatrix) HYPRE_IJMatrixGetLocalStorage(HYinvA22_);
          r_csr   = (HYPRE_ParVector)    HYPRE_IJVectorGetLocalStorage(R1);
          x2_csr  = (HYPRE_ParVector)    HYPRE_IJVectorGetLocalStorage(x2);
          HYPRE_ParCSRMatrixMatvec( 1.0, A22_csr, r_csr, 0.0, x2_csr );

          //-------------------------------------------------------------
          // inject final solution to the solution vector
          //-------------------------------------------------------------

          localNRows = localEndRow_ - localStartRow_ + 1 - 2 * nConstraints_;
          rowNum = localStartRow_ - 1;
          for ( i = startRow2; i < startRow2+localNRows; i++ )
          {
             HYPRE_IJVectorGetLocalComponents(reducedX_, 1, &i, NULL, &ddata);
             while (HYFEI_BinarySearch(selectedList_,rowNum,nConstraints_)>=0)
                rowNum++;
             HYPRE_IJVectorSetLocalComponents(HYx_,1,&rowNum,NULL,&ddata);
             rowNum++;
          }
          for ( i = 0; i < nConstraints_; i++ )
          {
             for ( j = 0; j < nConstraints_; j++ ) 
             {
                if ( selectedListAux_[j] == i ) 
                {
                   index = selectedList_[j]; 
                   break;
                }
             }
             j = i + startRow; 
             HYPRE_IJVectorGetLocalComponents(x2, 1, &j, NULL, &ddata);
             HYPRE_IJVectorSetLocalComponents(HYx_,1,&index,NULL,&ddata);
          }
          for ( i = nConstraints_; i < 2*nConstraints_; i++ )
          {
             j = startRow + i;
             HYPRE_IJVectorGetLocalComponents(x2, 1, &j, NULL, &ddata);
             index = localEndRow_ - 2 * nConstraints_ + i;
             HYPRE_IJVectorSetLocalComponents(HYx_,1,&index,NULL,&ddata);
          } 

          //-------------------------------------------------------------
          // residual norm check 
          //-------------------------------------------------------------

          A_csr  = (HYPRE_ParCSRMatrix) HYPRE_IJMatrixGetLocalStorage(HYA_);
          x_csr  = (HYPRE_ParVector)    HYPRE_IJVectorGetLocalStorage(HYx_);
          b_csr  = (HYPRE_ParVector)    HYPRE_IJVectorGetLocalStorage(HYb_);
          r_csr  = (HYPRE_ParVector)    HYPRE_IJVectorGetLocalStorage(HYr_);
          HYPRE_ParVectorCopy( b_csr, r_csr );
          HYPRE_ParCSRMatrixMatvec( -1.0, A_csr, x_csr, 1.0, r_csr );
          HYPRE_ParVectorInnerProd( r_csr, r_csr, &rnorm);
          rnorm = sqrt( rnorm );
          if ( mypid_ == 0 )
             printf("launchSolver::reduced sytem final norm = %e\n", rnorm);
       } 
       currX_ = HYx_;

       //****************************************************************
       // clean up
       //----------------------------------------------------------------

       HYPRE_IJMatrixDestroy(HYA21_); 
       HYA21_ = NULL;
       HYPRE_IJMatrixDestroy(HYinvA22_); 
       HYinvA22_ = NULL;
       HYPRE_IJVectorDestroy(R1); 
       HYPRE_IJVectorDestroy(x2); 
    }

    //*******************************************************************
    // diagnostic information
    //-------------------------------------------------------------------

#ifdef PRINT_SOL

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
    MPI_Barrier(MPI_COMM_WORLD);
    exit(0);

#endif 

    if ( HYOutputLevel_ > 0 )
    {
       printf("%4d : HYPRE_LinSysCore::leaving  launchSolver.\n", mypid_);
    }
}

//***************************************************************************
// this function solve the incoming linear system using SuperLU
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::solveUsingSuperLU(int& status)
{
    int                i, nnz, nrows, ierr;
    int                rowSize, *colInd, *new_ia, *new_ja, *ind_array;
    int                j, nz_ptr;
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
      
    if ( colIndices_ == NULL || rowLengths_ == NULL )
    {
       printf("solveUsingSuperLU error - allocateMatrix not called.\n");
       status = -1;
       return;
    }
    if ( localStartRow_ != 1 )
    {
       printf("solveUsingSuperLU ERROR - row does not start at 1\n");
       status = -1;
       return;
    }
    nrows = localEndRow_;
    nnz   = 0;
    for ( i = 0; i < nrows; i++ ) nnz += rowLengths_[i];

    new_ia = new int[nrows+1];
    new_ja = new int[nnz];
    new_a  = new double[nnz];
    A_csr  = (HYPRE_ParCSRMatrix) HYPRE_IJMatrixGetLocalStorage(HYA_);

    nz_ptr = getMatrixCSR(nrows, nnz, new_ia, new_ja, new_a);

    nnz = nz_ptr;

    //------------------------------------------------------------------
    // set up SuperLU CSR matrix and the corresponding rhs
    //------------------------------------------------------------------

    dCreate_CompRow_Matrix(&A2,nrows,nrows,nnz,new_a,new_ja,new_ia,NR,_D,GE);
    ind_array = new int[nrows];
    for ( i = 0; i < nrows; i++ ) ind_array[i] = i;
    rhs = new double[nrows];
    ierr = HYPRE_IJVectorGetLocalComponents(HYb_, nrows, ind_array, NULL, rhs);
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
        printf("SuperLU : NNZ in L+U = %d\n",Lstore->nnz+Ustore->nnz-nrows);

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
       ierr = HYPRE_IJVectorSetLocalComponents(HYx_,nrows,ind_array,NULL,soln);
       assert(!ierr);
       x_csr  = (HYPRE_ParVector) HYPRE_IJVectorGetLocalStorage(HYx_);
       b_csr  = (HYPRE_ParVector) HYPRE_IJVectorGetLocalStorage(HYb_);
       r_csr  = (HYPRE_ParVector) HYPRE_IJVectorGetLocalStorage(HYr_);
       ierr = HYPRE_ParVectorCopy( b_csr, r_csr );
       assert(!ierr);
       ierr = HYPRE_ParCSRMatrixMatvec( -1.0, A_csr, x_csr, 1.0, r_csr );
       assert(!ierr);
       ierr = HYPRE_ParVectorInnerProd( r_csr, r_csr, &rnorm);
       assert(!ierr);
       rnorm = sqrt( rnorm );
       printf("HYPRE_LinSysCore::solveUsingSuperLU - FINAL NORM = %e.\n",rnorm);
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
    SUPERLU_FREE( ((NRformat *) A2.Store)->colind);
    SUPERLU_FREE( ((NRformat *) A2.Store)->rowptr);
    SUPERLU_FREE( ((NRformat *) A2.Store)->nzval);
    SUPERLU_FREE( A2.Store );
    SUPERLU_FREE( ((NRformat *) U.Store)->colind);
    SUPERLU_FREE( ((NRformat *) U.Store)->rowptr);
    SUPERLU_FREE( ((NRformat *) U.Store)->nzval);
    SUPERLU_FREE( U.Store );
#else
    printf("HYPRE_LinSysCore::solveUsingSuperLU : not available.\n");
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
      
    if ( colIndices_ == NULL || rowLengths_ == NULL )
    {
       printf("solveUsingSuperLUX ERROR - Configure not called\n");
       status = -1;
       return;
    }
    if ( localStartRow_ != 1 )
    {
       printf("solveUsingSuperLUX ERROR - row not start at 1\n");
       status = -1;
       return;
    }
    nrows = localEndRow_;
    colLengths = new int[nrows];
    for ( i = 0; i < nrows; i++ ) colLengths[i] = 0;
    
    maxRowSize = 0;
    A_csr  = (HYPRE_ParCSRMatrix) HYPRE_IJMatrixGetLocalStorage(HYA_);
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

    new_ia = new int[nrows+1];
    new_ja = new int[nnz];
    new_a  = new double[nnz];

    nz_ptr = getMatrixCSR(nrows, nnz, new_ia, new_ja, new_a);

    nnz = nz_ptr;

    //------------------------------------------------------------------
    // set up SuperLU CSR matrix and the corresponding rhs
    //------------------------------------------------------------------

    dCreate_CompRow_Matrix(&A2,nrows,nrows,nnz,new_a,new_ja,new_ia,NR,_D,GE);
    ind_array = new int[nrows];
    for ( i = 0; i < nrows; i++ ) ind_array[i] = i;
    rhs = new double[nrows];
    ierr = HYPRE_IJVectorGetLocalComponents(HYb_,nrows,ind_array,NULL,rhs);
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
        if ( rcond != 0.0 )
           printf("   SuperLU : condition number = %e\n", 1.0/rcond);
        else
           printf("   SuperLU : Recip. condition number = %e\n", rcond);

        Lstore = (SCformat *) L.Store;
        Ustore = (NRformat *) U.Store;
        //printf("No of nonzeros in factor L = %d\n", Lstore->nnz);
        //printf("No of nonzeros in factor U = %d\n", Ustore->nnz);
        printf("SuperLU : NNZ in L+U = %d\n", Lstore->nnz+Ustore->nnz-nrows);

        //dQuerySpace(&L, &U, panel_size, &mem_usage);
        //printf("L\\U MB %.3f\ttotal MB needed %.3f\texpansions %d\n",
        //       mem_usage.for_lu/1e6, mem_usage.total_needed/1e6,
        //       mem_usage.expansions);
    } else {
        printf("solveUsingSuperLUX - dgssvx error code = %d\n",info);
        status = 0;
    }

    //------------------------------------------------------------------
    // fetch the solution and find residual norm
    //------------------------------------------------------------------

    if ( status == 1 )
    {
       ierr = HYPRE_IJVectorSetLocalComponents(HYx_,nrows,ind_array,NULL,soln);
       assert(!ierr);
       x_csr  = (HYPRE_ParVector) HYPRE_IJVectorGetLocalStorage(HYx_);
       r_csr  = (HYPRE_ParVector) HYPRE_IJVectorGetLocalStorage(HYr_);
       b_csr    = (HYPRE_ParVector) HYPRE_IJVectorGetLocalStorage(HYb_);
       ierr = HYPRE_ParVectorCopy( b_csr, r_csr );
       assert(!ierr);
       ierr = HYPRE_ParCSRMatrixMatvec( -1.0, A_csr, x_csr, 1.0, r_csr );
       assert(!ierr);
       ierr = HYPRE_ParVectorInnerProd( r_csr, r_csr, &rnorm);
       assert(!ierr);
       rnorm = sqrt( rnorm );
       printf("HYPRE_LinSysCore::solveUsingSuperLUX - FINAL NORM = %e.\n",rnorm);
    }

    //------------------------------------------------------------------
    // clean up 
    //------------------------------------------------------------------

    delete [] ind_array; 
    delete [] rhs; 
    delete [] perm_c; 
    delete [] perm_r; 
    delete [] etree; 
    delete [] new_ia; 
    delete [] new_ja; 
    delete [] new_a; 
    delete [] soln;
    delete [] colLengths;
    Destroy_SuperMatrix_Store(&B);
    Destroy_SuperNode_Matrix(&L);
    SUPERLU_FREE( ((NRformat *) A2.Store)->colind);
    SUPERLU_FREE( ((NRformat *) A2.Store)->rowptr);
    SUPERLU_FREE( ((NRformat *) A2.Store)->nzval);
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
    printf("HYPRE_LinSysCore::solveUsingSuperLUX : not available.\n");
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
      
    if ( colIndices_ == NULL || rowLengths_ == NULL )
    {
       printf("solveUsingY12M ERROR - Configure not called\n");
       status = -1;
       return;
    }
    if ( localStartRow_ != 1 )
    {
       printf("solveUsingY12M ERROR - row does not start at 1.\n");
       status = -1;
       return;
    }
    nrows = localEndRow_;
    colLengths = new int[nrows];
    for ( i = 0; i < nrows; i++ ) colLengths[i] = 0;
    
    maxRowSize = 0;
    A_csr  = (HYPRE_ParCSRMatrix) HYPRE_IJMatrixGetLocalStorage(HYA_);
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
    ierr = HYPRE_IJVectorGetLocalComponents(HYb_,nrows,ind_array,NULL,rhs);
    assert(!ierr);

    //------------------------------------------------------------------
    // call Y12M to solve the linear system
    //------------------------------------------------------------------

    y12maf_(&nrows,&nnz,val,snr,&nn,rnr,&nn1,pivot,ha,&iha,aflag,iflag,
            rhs,&ifail);
    if ( ifail != 0 )
    {
       printf("solveUsingY12M warning - ifail = %d\n", ifail);
    }
 
    //------------------------------------------------------------------
    // postprocessing
    //------------------------------------------------------------------

    if ( ifail == 0 )
    {
       ierr = HYPRE_IJVectorSetLocalComponents(HYx_,nrows,ind_array,NULL,rhs);
       assert(!ierr);
       x_csr  = (HYPRE_ParVector) HYPRE_IJVectorGetLocalStorage(HYx_);
       r_csr  = (HYPRE_ParVector) HYPRE_IJVectorGetLocalStorage(HYr_);
       b_csr  = (HYPRE_ParVector) HYPRE_IJVectorGetLocalStorage(HYb_);
       ierr = HYPRE_ParVectorCopy( b_csr, r_csr );
       assert(!ierr);
       ierr = HYPRE_ParCSRMatrixMatvec( -1.0, A_csr, x_csr, 1.0, r_csr );
       assert(!ierr);
       ierr = HYPRE_ParVectorInnerProd( r_csr, r_csr, &rnorm);
       assert(!ierr);
       rnorm = sqrt( rnorm );
       printf("HYPRE_LinSysCore::solveUsingY12M - final norm = %e.\n", rnorm);
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
    printf("HYPRE_LinSysCore::solveUsingY12M - not available.\n");
#endif

}

//***************************************************************************
// this function loads in the constraint numbers for reduction
// (to activate automatic slave search, constrList should be NULL)
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::loadConstraintNumbers(int nConstr, int *constrList)
{
    if ( HYOutputLevel_ > 0 )
    {
       printf("%4d : HYPRE_LinSysCore::loadSConstraintNumbers - size = %d\n", 
                     mypid_, nConstr);
       if ( mypid_ == 0 )
          printf("%4d loadSConstraintNumbers : DOF=3, NODE#=EQN# assumed.\n",
                 mypid_); 
    }

    nConstraints_ = nConstr;
    if ( nConstr > 0 )
    {
       if ( constrList != NULL ) 
       {
          constrList_ = new int[3 * nConstr];
          for (int i = 0; i < nConstr; i++) 
          {
             constrList_[3*i] = constrList[i] * 3;
             constrList_[3*i+1] = constrList[i] * 3 + 1;
             constrList_[3*i+2] = constrList[i] * 3 + 2;
          }
          if ( HYOutputLevel_ > 2 )
          {
             for (int j = 0; j < 3 * nConstraints_; j++) 
                printf("Constraint %5d(%5d) = %d\n",j,nConstraints_,
                                                    constrList_[j]);
          }
       }
    }
    if ( HYOutputLevel_ > 0 )
    {
       printf("%4d : HYPRE_LinSysCore::leaving  loadSConstraintNumbers\n",
              mypid_);
    }
}

//***************************************************************************
// this function extracts the matrix in a CSR format
//---------------------------------------------------------------------------

int HYPRE_LinSysCore::getMatrixCSR(int nrows, int nnz, int *ia_ptr, 
                                   int *ja_ptr, double *a_ptr) 
{
    int                nz, i, j, ierr, rowSize, *colInd, nz_ptr, *colInd2;
    int                firstNnz;
    double             *colVal, *colVal2;
    HYPRE_ParCSRMatrix A_csr;

    nz        = 0;
    nz_ptr    = 0;
    ia_ptr[0] = nz_ptr;
    A_csr  = (HYPRE_ParCSRMatrix) HYPRE_IJMatrixGetLocalStorage(HYA_);
    for ( i = 0; i < nrows; i++ )
    {
       ierr = HYPRE_ParCSRMatrixGetRow(A_csr,i,&rowSize,&colInd,&colVal);
       assert(!ierr);
       colInd2 = new int[rowSize];
       colVal2 = new double[rowSize];
       for ( j = 0; j < rowSize; j++ )
       {
          colInd2[j] = colInd[j];
          colVal2[j] = colVal[j];
       }
       if ( rowSize > rowLengths_[i] )
          printf("getMatrixCSR warning at row %d - %d %d\n", i,rowSize,
                   rowLengths_[i]);
       qsort1(colInd2, colVal2, 0, rowSize-1);
       for ( j = 0; j < rowSize-1; j++ )
          if ( colInd2[j] == colInd2[j+1] )
             printf("getMatrixCSR - duplicate colind at row %d \n",i);

       firstNnz = 0;
       for ( j = 0; j < rowSize; j++ )
       {
          if ( colVal2[j] != 0.0 )
          {
             if (nz_ptr > 0 && firstNnz > 0 && colInd2[j] == ja_ptr[nz_ptr-1]) 
             {
                a_ptr[nz_ptr-1] += colVal2[j];
                printf("getMatrixCSR :: repeated col in row %d\n", i);
             }
             else
             { 
                ja_ptr[nz_ptr] = colInd2[j];
                a_ptr[nz_ptr++]  = colVal2[j];
                if ( nz_ptr > nnz )
                {
                   printf("getMatrixCSR error (1) - %d %d.\n",i, nrows);
                   exit(1);
                }
                firstNnz++;
             }
          } else nz++;
       }
       delete [] colInd2;
       delete [] colVal2;
       ia_ptr[i+1] = nz_ptr;
       ierr = HYPRE_ParCSRMatrixRestoreRow(A_csr,i,&rowSize,&colInd,&colVal);
       assert(!ierr);
    }   
    if ( nnz != nz_ptr )
    {
       printf("getMatrixCSR note : matrix sparsity has been changed since\n");
       printf("             matConfigure - %d > %d ?\n", nnz, nz_ptr);
       printf("             number of zeros            = %d \n", nz );
    }
    return nz_ptr;
}

//***************************************************************************
// this function extracts the matrix in a CSR format
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::writeSystem(char *name)
{
    printf("HYPRE_LinsysCore : writeSystem not implemented.\n");
    return;
}

//***************************************************************************
// HYFEI_BinarySearch - this is a modification of hypre_BinarySearch
//---------------------------------------------------------------------------

int HYPRE_LinSysCore::HYFEI_BinarySearch(int *list,int value,int list_length)
{
   int low, high, m;
   int not_found = 1;

   low = 0;
   high = list_length-1;
   while (not_found && low <= high)
   {
      m = (low + high) / 2;
      if (value < list[m])
      {
         high = m - 1;
      }
      else if (value > list[m])
      {
        low = m + 1;
      }
      else
      {
        not_found = 0;
        return m;
      }
   }
   return -(low+1);
}

//******************************************************************************
// Given the matrix (A) within the object, compute the reduced system and put
// it in place.  Additional information given are :
//
// Additional assumptions are :
//
//    - a given slave equation and the corresponding constraint equation
//      reside in the same processor
//    - constraint equations are given at the end of the local matrix
//      (hence given by EndRow_-nConstr to EndRow_)
//    - each processor gets a contiguous block of equations, and processor
//      i+1 has equation numbers higher than those of processor i
//------------------------------------------------------------------------------
// This first draft does not try to reduce the number of columns for
// A21 so that the triple matrix product operations will result in a
// NxN matrix.  This can be inefficient and needs to be studied further.
// However, this approach simplifies the searching and remapping. 
//------------------------------------------------------------------------------

void HYPRE_LinSysCore::buildReducedSystem()
{
    int    j, k, nRows, globalNRows, colIndex, nSlaves;
    int    globalNConstr, globalNSelected, *globalSelectedList;
    int    *globalSelectedListAux, *selectedListAux;
    int    nSelected, *tempList, i, reducedAStartRow;
    int    searchIndex, procIndex, A21StartRow, A12StartRow, A12NRows;
    int    rowSize, *colInd, A21NRows, A21GlobalNRows;
    int    A21NCols, A21GlobalNCols, rowCount, maxRowSize, newEndRow;
    int    A12NCols, A12GlobalNCols, *constrListAux;
    int    *A21MatSize, rowIndex, *A12MatSize, A12GlobalNRows;
    int    *newColInd, diagCount, newRowSize, ierr;
    int    invA22NRows, invA22GlobalNRows, invA22NCols, invA22GlobalNCols;
    int    *invA22MatSize, newNRows, newGlobalNRows;
    int    *colInd2, *selectedList, ncnt, ubound;
    int    rowSize2, *recvCntArray, *displArray, ncnt2;
    int    StartRow, EndRow, *reducedAMatSize;
    int    *ProcNRows, *ProcNConstr;

    double searchValue, *colVal, *colVal2, *newColVal, *diagonal;
    double *extDiagonal, *dble_array, ddata;

    HYPRE_IJMatrix     A12, A21, invA22, reducedA;
    HYPRE_ParCSRMatrix A_csr, A12_csr, A21_csr, invA22_csr, RAP_csr;
    HYPRE_ParCSRMatrix reducedA_csr;
    HYPRE_IJVector     f2, f2hat;
    HYPRE_ParVector    f2_csr, f2hat_csr, reducedB_csr;

    //******************************************************************
    // initial set up 
    //------------------------------------------------------------------

    if ( mypid_ == 0 ) printf("%4d buildReducedSystem activated.\n",mypid_);
    StartRow = localStartRow_ - 1;
    EndRow   = localEndRow_ - 1;
    if ( HYOutputLevel_ > 0 )
    {
       printf("%4d buildReducedSystem : StartRow/EndRow = %d %d\n",mypid_,
                                        StartRow,EndRow);
    }

    //******************************************************************
    // construct local and global information about where the constraints
    // are (this is given by user or searched within this code)
    //------------------------------------------------------------------

    //------------------------------------------------------------------
    // get the CSR matrix for A
    //------------------------------------------------------------------

    A_csr = (HYPRE_ParCSRMatrix) HYPRE_IJMatrixGetLocalStorage(HYA_);

    //------------------------------------------------------------------
    // search the entire local matrix to find where the constraint
    // equations are, if not already given
    //------------------------------------------------------------------
    
    MPI_Allreduce(&nConstraints_,&globalNConstr,1,MPI_INT,MPI_SUM,comm_);
    if ( globalNConstr == 0 )
    {
       for ( i = EndRow; i >= StartRow; i-- ) 
       {
          ierr = HYPRE_ParCSRMatrixGetRow(A_csr,i,&rowSize,&colInd,&colVal);
          assert(!ierr);
          for (j = 0; j < rowSize; j++) 
          {
             if ( colInd[j] == i && colVal[j] != 0.0 ) break;
          }
          ierr = HYPRE_ParCSRMatrixRestoreRow(A_csr,i,&rowSize,&colInd,&colVal);
          assert(!ierr);
          if ( j < rowSize ) nConstraints_++;
          else               break;
       }
    }
    if ( HYOutputLevel_ > 0 )
    {
       printf("%4d buildReducedSystem : no. constr = %d\n",mypid_,nConstraints_);
    }

    MPI_Allreduce(&nConstraints_, &globalNConstr, 1, MPI_INT, MPI_SUM, comm_);
    if ( globalNConstr == 0 ) return;

    //------------------------------------------------------------------
    // get information about nRows from all processors
    //------------------------------------------------------------------
 
    nRows       = localEndRow_ - localStartRow_ + 1;
    ProcNRows   = new int[numProcs_];
    tempList    = new int[numProcs_];
    for ( i = 0; i < numProcs_; i++ ) tempList[i] = 0;
    tempList[mypid_] = nRows;
    MPI_Allreduce(tempList, ProcNRows, numProcs_, MPI_INT, MPI_SUM, comm_);
    delete [] tempList;
    if ( HYOutputLevel_ > 0 )
    {
       printf("%4d buildReducedSystem : localNRows = %d\n", mypid_, nRows);
    }

    //------------------------------------------------------------------
    // compute the base NRows on each processor
    // (This is needed later on for column index conversion)
    //------------------------------------------------------------------

    globalNRows = 0;
    ncnt = 0;
    for ( i = 0; i < numProcs_; i++ ) 
    {
       globalNRows   += ProcNRows[i];
       ncnt2          = ProcNRows[i];
       ProcNRows[i]   = ncnt;
       ncnt          += ncnt2;
    }

    //------------------------------------------------------------------
    // compose a global array marking where the constraint equations are
    //------------------------------------------------------------------
    
    globalNConstr = 0;
    tempList    = new int[numProcs_];
    ProcNConstr = new int[numProcs_];
    for ( i = 0; i < numProcs_; i++ ) tempList[i] = 0;
    tempList[mypid_] = nConstraints_;
    MPI_Allreduce(tempList,ProcNConstr,numProcs_,MPI_INT,MPI_SUM,comm_);
    delete [] tempList;

    //------------------------------------------------------------------
    // compute the base nConstraints on each processor
    // (This is needed later on for column index conversion)
    //------------------------------------------------------------------

    ncnt = 0;
    for ( i = 0; i < numProcs_; i++ ) 
    {
       globalNConstr += ProcNConstr[i];
       ncnt2          = ProcNConstr[i];
       ProcNConstr[i] = ncnt;
       ncnt          += ncnt2;
    }
   
    //******************************************************************
    // compose the local and global selected node lists
    //------------------------------------------------------------------

    //------------------------------------------------------------------
    // allocate array for storing indices of selected nodes
    //------------------------------------------------------------------

    globalNSelected = globalNConstr;
    if (globalNSelected > 0) 
    {
       globalSelectedList = new int[globalNSelected];
       globalSelectedListAux = new int[globalNSelected];
    }
    else
    {
       globalSelectedList = NULL;
       globalSelectedListAux = NULL;
    }
    nSelected = nConstraints_;
    if ( nConstraints_ > 0 ) 
    {
       selectedList = new int[nConstraints_];
       selectedListAux = new int[nConstraints_];
    }
    else 
    {
       selectedList = NULL;
       selectedListAux = NULL;
    }
   
    //------------------------------------------------------------------
    // compose candidate slave list (if not given already)
    //------------------------------------------------------------------

    if ( nConstraints_ > 0 && constrList_ == NULL )
    {
       constrList_   = new int[EndRow-nConstraints_-StartRow+1];
       constrListAux = new int[EndRow-nConstraints_-StartRow+1];
       nSlaves = 0;

       //------------------------------------------------------------------
       // candidates are those with 1 link to the constraint list
       //------------------------------------------------------------------

       for ( i = StartRow; i <= EndRow-nConstraints_; i++ ) 
       {
          ierr = HYPRE_ParCSRMatrixGetRow(A_csr,i,&rowSize,&colInd,&colVal);
          assert(!ierr);
          ncnt = 0;
          for (j = 0; j < rowSize; j++) 
          {
             colIndex = colInd[j];
             for (procIndex=0; procIndex < numProcs_; procIndex++ )
                if ( colIndex < ProcNRows[procIndex] ) break;
             if ( procIndex == numProcs_ ) 
                ubound = globalNRows - 
                         (globalNConstr-ProcNConstr[procIndex-1]);
             else                          
                ubound = ProcNRows[procIndex] - (ProcNConstr[procIndex] - 
                                                 ProcNConstr[procIndex-1]); 

             //Note : include structural zeros by not checking for nonzero
             //if ( colIndex >= ubound && colVal[j] != 0.0 ) 
             if ( colIndex >= ubound ) 
             {
                ncnt++;
                searchIndex = colIndex;
             }
             if ( ncnt > 1 ) break;
          }
          HYPRE_ParCSRMatrixRestoreRow(A_csr,i,&rowSize,&colInd,&colVal);
          if ( j == rowSize && ncnt == 1 ) 
          {
             constrListAux[nSlaves] = searchIndex;
             constrList_[nSlaves++] = i;
          }
          if ( HYOutputLevel_ > 2 )
          {
             if ( j == rowSize && ncnt == 1 ) 
                printf("%d buildReducedSystem : slave candidate %d = %d(%d)\n", 
                        mypid_, nSlaves-1, i, constrListAux[nSlaves-1]);
          }
       }
       if ( HYOutputLevel_ > 1 )
       {
          printf("%d buildReducedSystem : nSlave Candidate, nConstr = %d %d\n",
                 mypid_,nSlaves, nConstraints_);
       }
    }
    else
    {
       if ( mypid_ == 0 )
          printf("%4d buildReducedSystem WARNING : HARDWIRED TO 3 DOF/NODE.\n",
                  mypid_);
       constrListAux = new int[EndRow-nConstraints_-StartRow+1];
       nSlaves = 3 * nConstraints_;;
       for ( i = 0; i < 3*nConstraints_; i++ ) 
       {
          rowIndex = constrList_[i]; 
          if ( rowIndex < localStartRow_-1 || rowIndex >= localEndRow_)
          {
             printf("%4d buildReducedSystem : slave %d not on my proc\n",
                    mypid_, rowIndex, localStartRow_-1, localEndRow_);
             exit(1);
          }
          ierr = HYPRE_ParCSRMatrixGetRow(A_csr,rowIndex,&rowSize,&colInd,&colVal);
          assert(!ierr);
          ncnt = 0;
          for (j = 0; j < rowSize; j++) 
          {
             colIndex = colInd[j];
             for (procIndex=0; procIndex < numProcs_; procIndex++ )
                if ( colIndex < ProcNRows[procIndex] ) break;
             if ( procIndex == numProcs_ ) 
                ubound = globalNRows - 
                         (globalNConstr-ProcNConstr[procIndex-1]);
             else                          
                ubound = ProcNRows[procIndex] - (ProcNConstr[procIndex] - 
                                                 ProcNConstr[procIndex-1]); 
             if ( colIndex >= ubound && colVal[j] != 0.0 ) 
             {
                ncnt++;
                searchIndex = colIndex;
             }
             if ( ncnt > 1 ) break;
          }
          HYPRE_ParCSRMatrixRestoreRow(A_csr,rowIndex,&rowSize,&colInd,&colVal);
          if ( j == rowSize && ncnt == 1 ) constrListAux[i] = searchIndex;
          else                             constrListAux[i] = -1;
          if ( HYOutputLevel_ > 1 )
          {
             if ( j == rowSize && ncnt == 1 ) 
                printf("%4d buildReducedSystem : slave,constr pair = %d %d\n",
                        mypid_, constrList_[i], constrListAux[i]);
          }
       }
    }   
    if ( HYOutputLevel_ > 1 )
    {
       printf("%d buildReducedSystem : nSlave Candidate, nConstr = %d %d\n",
              mypid_,nSlaves, nConstraints_);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    //---------------------------------------------------------------------
    // search the constraint equations for the selected nodes
    // (search for candidates column index with maximum magnitude)
    //---------------------------------------------------------------------
    
    nSelected = 0;
    rowIndex = -1;

    for ( i = EndRow-nConstraints_+1; i <= EndRow; i++ ) 
    {
       ierr = HYPRE_ParCSRMatrixGetRow(A_csr,i,&rowSize,&colInd,&colVal);
       assert(!ierr);
       searchIndex = -1;
       searchValue = -1.0E10;
       for (j = 0; j < rowSize; j++) 
       {
          if (colVal[j] != 0.0 && colInd[j] >= StartRow 
                               && colInd[j] <= (EndRow-nConstraints_)) 
          {
             colIndex = hypre_BinarySearch(constrList_,colInd[j],nSlaves);
             if ( colIndex >= 0 && constrListAux[colIndex] != -1) 
             {
                 if ( abs(colVal[j]) > searchValue )
                 {
                    if (i != constrListAux[colIndex]) 
                    {
                       printf("%4d buildReducedSystem WARNING : slave %d",
                               mypid_, colInd[j]);
                       printf(" candidate does not have constr %d\n", i);
                    }
                    searchValue = abs(colVal[j]);
                    searchIndex = colInd[j];
                 }
             }
          }
       } 
       if ( searchIndex >= 0 )
       {
          selectedList[nSelected++] = searchIndex;
          if ( HYOutputLevel_ > 1 )
          {
             printf("%4d buildReducedSystem : constraint %4d <=> slave %d\n",
                    mypid_,i,searchIndex);
          }
       } else 
       {
          // get ready for error processing

          colInd2 = new int[rowSize];
          colVal2 = new double[rowSize];
          for ( j = 0; j < rowSize; j++ )
          {    
             colInd2[j] = colInd[j];
             colVal2[j] = colVal[j];
          }
          rowIndex = i;
          HYPRE_ParCSRMatrixRestoreRow(A_csr,i,&rowSize,&colInd,&colVal);
          colInd = colInd2;
          colVal = colVal2;
          break;
       }
       HYPRE_ParCSRMatrixRestoreRow(A_csr,i,&rowSize,&colInd,&colVal);
    }

    //---------------------------------------------------------------------
    // error processing
    //---------------------------------------------------------------------

    if ( searchIndex < 0 ) searchIndex = 1; else searchIndex = 0;
    MPI_Allreduce(&searchIndex, &ncnt,1,MPI_INT,MPI_MAX,comm_);

    if ( ncnt > 0 )
    {
       ncnt2 = 0;
       while ( ncnt2 < numProcs_ )
       { 
          if ( ncnt2 == mypid_ && rowIndex >= 0 )
          {
             printf("%4d buildReducedSystem ERROR : constraint number",mypid_);
             printf(" cannot be found for row %d\n", rowIndex);
             for (j = 0; j < rowSize; j++) 
             {
                printf("ROW %4d COL = %d VAL = %e\n",rowIndex,colInd[j],colVal[j]);
                if (colVal[j] != 0.0 && colInd[j] >= StartRow 
                                     && colInd[j] <= (EndRow-nConstraints_)) 
                {
                   colIndex = colInd[j];
                   HYPRE_ParCSRMatrixGetRow(A_csr,colIndex,&rowSize2,&colInd2,
                                            &colVal2);
                   printf("      row %4d (%d) : \n",colIndex, rowSize2);
                   for (k = 0; k < rowSize2; k++) 
                      printf("    row %4d col = %d val = %e\n",colIndex,
                                            colInd2[k],colVal2[k]);
                   HYPRE_ParCSRMatrixRestoreRow(A_csr,colIndex,&rowSize2,
                                            &colInd2,&colVal2);
                }
             }
             printf("===================================================\n");
          }
          ncnt2++;
          MPI_Barrier(MPI_COMM_WORLD);
       }
       MPI_Finalize();
       exit(1);
    }
    delete [] constrListAux;

    //------------------------------------------------------------------
    // sort the local selected node list and its auxiliary list, then
    // form a global list of selected nodes on each processor
    // form the corresponding auxiliary list for later pruning
    //------------------------------------------------------------------

    dble_array = new double[nSelected];
    for ( i = 0; i < nSelected; i++ ) dble_array[i] = (double) i; 
    qsort1(selectedList, dble_array, 0, nSelected-1);
    for (i = 1; i < nSelected; i++) 
    {
       if ( selectedList[i] == selectedList[i-1] )
       {
          printf("%4d buildReducedSystem : repeated selected nodes %d \n", 
                 mypid_, selectedList[i]);
          exit(1);
       }
    }
    for (i = 0; i < nSelected; i++) selectedListAux[i] = (int) dble_array[i];
    delete [] dble_array;
    
    recvCntArray = new int[numProcs_];
    displArray   = new int[numProcs_];
    MPI_Allgather(&nSelected, 1, MPI_INT,recvCntArray, 1,MPI_INT, comm_);
    displArray[0] = 0;
    for ( i = 1; i < numProcs_; i++ ) 
       displArray[i] = displArray[i-1] + recvCntArray[i-1];
    for ( i = 0; i < nSelected; i++ ) 
       selectedListAux[i] += displArray[mypid_]; 
    MPI_Allgatherv(selectedList, nSelected, MPI_INT, globalSelectedList,
                   recvCntArray, displArray, MPI_INT, comm_);
    MPI_Allgatherv(selectedListAux, nSelected, MPI_INT, globalSelectedListAux,
                   recvCntArray, displArray, MPI_INT, comm_);
    for ( i = 0; i < nSelected; i++ ) 
       selectedListAux[i] -= displArray[mypid_]; 
    delete [] recvCntArray;
    delete [] displArray;

    if ( HYOutputLevel_ > 1 )
    {
       for ( i = 0; i < nSelected; i++ )
          printf("%4d buildReducedSystem : selectedList %d = %d(%d)\n",mypid_,
                 i,selectedList[i],selectedListAux[i]);
    }
 
    //******************************************************************
    // construct A21
    //------------------------------------------------------------------

    //------------------------------------------------------------------
    // calculate the dimension of A21
    //------------------------------------------------------------------

    A21NRows       = 2 * nConstraints_;
    A21NCols       = nRows - 2 * nConstraints_;
    A21GlobalNRows = 2 * globalNConstr;
    A21GlobalNCols = globalNRows - 2 * globalNConstr;
    A21StartRow    = 2 * ProcNConstr[mypid_];

    if ( HYOutputLevel_ > 0 )
    {
       printf("%4d buildReducedSystem : A21StartRow  = %d\n",mypid_,A21StartRow);
       printf("%4d buildReducedSystem : A21GlobalDim = %d %d\n", mypid_, 
                                        A21GlobalNRows, A21GlobalNCols);
       printf("%4d buildReducedSystem : A21LocalDim  = %d %d\n",mypid_,
                                        A21NRows, A21NCols);
    }

    //------------------------------------------------------------------
    // create a matrix context for A21
    //------------------------------------------------------------------

    ierr  = HYPRE_IJMatrixCreate(comm_,&A21,A21GlobalNRows,A21GlobalNCols);
    ierr += HYPRE_IJMatrixSetLocalStorageType(A21, HYPRE_PARCSR);
    ierr  = HYPRE_IJMatrixSetLocalSize(A21, A21NRows, A21NCols);
    assert(!ierr);

    //------------------------------------------------------------------
    // compute the number of nonzeros in the first nConstraint row of A21
    // (which consists of the rows in selectedList), the nnz will
    // be reduced by excluding the constraint and selected slave columns
    //------------------------------------------------------------------

    rowCount   = 0;
    maxRowSize = 0;
    newEndRow  = EndRow - nConstraints_;
    A21MatSize = new int[A21NRows];

    for ( i = 0; i < nSelected; i++ ) 
    {
       for ( j = 0; j < nSelected; j++ ) 
       {
          if ( selectedListAux[j] == i ) 
          {
             rowIndex = selectedList[j]; 
             break;
          }
       }
       HYPRE_ParCSRMatrixGetRow(A_csr,rowIndex,&rowSize,&colInd,&colVal);
       rowSize2 = 0;
       for (j = 0; j < rowSize; j++) 
       {
          colIndex = colInd[j];
	  searchIndex = hypre_BinarySearch(globalSelectedList,colIndex, 
                                           globalNSelected);
          if (searchIndex < 0 && 
              (colIndex <= newEndRow || colIndex > localEndRow_)) rowSize2++;
       }
       A21MatSize[rowCount] = rowSize2;
       maxRowSize = ( rowSize2 > maxRowSize ) ? rowSize2 : maxRowSize;
       HYPRE_ParCSRMatrixRestoreRow(A_csr,rowIndex,&rowSize,&colInd,&colVal);
       rowCount++;
    }

    //------------------------------------------------------------------
    // compute the number of nonzeros in the second nConstraint row of A21
    // (which consists of the rows in constraint equations), the nnz will
    // be reduced by excluding the selected slave columns only (since the
    // entries corresponding to the constraint columns are 0, and since
    // the selected matrix is a diagonal matrix, there is no need to 
    // search for slave equations in the off-processor list)
    //------------------------------------------------------------------

    rowCount = nSelected;
    for ( i = EndRow-nConstraints_+1; i <= EndRow; i++ ) 
    {
       HYPRE_ParCSRMatrixGetRow(A_csr,i,&rowSize,&colInd,&colVal);
       rowSize2 = 0;
       for (j = 0; j < rowSize; j++) 
       {
          if ( colVal[j] != 0.0 )
          {
             colIndex = colInd[j];
	     searchIndex = hypre_BinarySearch(selectedList,colIndex,nSelected); 
             if ( searchIndex < 0 ) rowSize2++;
          }
       }
       A21MatSize[rowCount] = rowSize2;
       maxRowSize = ( rowSize2 > maxRowSize ) ? rowSize2 : maxRowSize;
       HYPRE_ParCSRMatrixRestoreRow(A_csr,i,&rowSize,&colInd,&colVal);
       rowCount++;
    }

    //------------------------------------------------------------------
    // after fetching the row sizes, set up A21 with such sizes
    //------------------------------------------------------------------

    ierr  = HYPRE_IJMatrixSetRowSizes(A21, A21MatSize);
    ierr += HYPRE_IJMatrixInitialize(A21);
    assert(!ierr);
    delete [] A21MatSize;

    //------------------------------------------------------------------
    // next load the first nConstraint row to A21 extracted from A
    // (at the same time, the D block is saved for future use)
    //------------------------------------------------------------------

    rowCount  = A21StartRow;
    if ( nConstraints_ > 0 ) diagonal = new double[nConstraints_];
    else                    diagonal = NULL;
    newColInd = new int[maxRowSize+1];
    newColVal = new double[maxRowSize+1];

    diagCount = 0;
    for ( i = 0; i < nSelected; i++ )
    {
       for ( j = 0; j < nSelected; j++ ) 
       {
          if ( selectedListAux[j] == i ) 
          {
             rowIndex = selectedList[j]; 
             break;
          }
       }
       HYPRE_ParCSRMatrixGetRow(A_csr,rowIndex,&rowSize,&colInd,&colVal);
       newRowSize = 0;
       for (j = 0; j < rowSize; j++) 
       {
          if ( colVal[j] != 0.0 )
          {
             colIndex = colInd[j];
             if (colIndex <= newEndRow || colIndex > localEndRow_) 
             {
	        searchIndex = HYFEI_BinarySearch(globalSelectedList,colIndex, 
                                                 globalNSelected); 
                if ( searchIndex < 0 ) 
                {
                   searchIndex = - searchIndex - 1;
                   for ( procIndex = 0; procIndex < numProcs_; procIndex++ )
                      if ( ProcNRows[procIndex] > colIndex ) break;
                   procIndex--;
                   colIndex = colInd[j]-ProcNConstr[procIndex]-searchIndex;
                   newColInd[newRowSize]   = colIndex;
                   newColVal[newRowSize++] = colVal[j];
                   if ( colIndex < 0 || colIndex >= A21GlobalNCols )
                   {
                      printf("%4d buildReducedSystem WARNING : A21 ", mypid_);
                      printf("out of range %d - %d (%d)\n", rowCount, colIndex, 
                              A21GlobalNCols);
                   } 
                   if ( newRowSize > maxRowSize+1 ) 
                   {
                      printf("%4d buildReducedSystem : WARNING - ",mypid_);
                      printf("passing array boundary(1).\n");
                   }
                }
             }
             else if ( colIndex > newEndRow && colIndex <= EndRow ) 
             {
                if ( colVal[j] != 0.0 ) diagonal[diagCount++] = colVal[j];
                if ( abs(colVal[j]) < 1.0E-8 )
                {
                   printf("%4d buildReducedSystem WARNING : large entry ",mypid_);
                   printf("in invA22\n");
                }
             }
          } 
       }

       HYPRE_IJMatrixInsertRow(A21,newRowSize,rowCount,newColInd,newColVal);
       HYPRE_ParCSRMatrixRestoreRow(A_csr,rowIndex,&rowSize,&colInd,&colVal);
       if ( diagCount != (i+1) )
       {
          printf("%4d buildReducedSystem ERROR (3) : %d %d.\n", mypid_,
                  diagCount,i+1);
          exit(1);
       }
       rowCount++;
    }

    //------------------------------------------------------------------
    // send the diagonal to each processor that needs them
    //------------------------------------------------------------------

    recvCntArray = new int[numProcs_];
    displArray   = new int[numProcs_];
    MPI_Allgather(&diagCount, 1, MPI_INT, recvCntArray, 1, MPI_INT, comm_);
    displArray[0] = 0;
    for ( i = 1; i < numProcs_; i++ ) 
       displArray[i] = displArray[i-1] + recvCntArray[i-1];
    ncnt = displArray[numProcs_-1] + recvCntArray[numProcs_-1];
    if ( ncnt > 0 ) extDiagonal = new double[ncnt];
    else            extDiagonal = NULL;
    MPI_Allgatherv(diagonal, diagCount, MPI_DOUBLE, extDiagonal,
                   recvCntArray, displArray, MPI_DOUBLE, comm_);
    diagCount = ncnt;
    delete [] recvCntArray;
    delete [] displArray;
    if ( diagonal != NULL ) delete [] diagonal;

    //------------------------------------------------------------------
    // next load the second nConstraint rows to A21 extracted from A
    //------------------------------------------------------------------

    for ( i = EndRow-nConstraints_+1; i <= EndRow; i++ ) 
    {
       HYPRE_ParCSRMatrixGetRow(A_csr,i,&rowSize,&colInd,&colVal);
       newRowSize = 0;
       for (j = 0; j < rowSize; j++) 
       {
          colIndex    = colInd[j];
	  searchIndex = HYFEI_BinarySearch(globalSelectedList,colIndex,
                                           globalNSelected); 
          if ( searchIndex < 0 && colVal[j] != 0.0 ) 
          {
             searchIndex = - searchIndex - 1;
             for ( procIndex = 0; procIndex < numProcs_; procIndex++ )
                if ( ProcNRows[procIndex] > colIndex ) break;
             procIndex--;
             colIndex = colInd[j] - ProcNConstr[procIndex] - searchIndex;
             newColInd[newRowSize]   = colIndex;
             newColVal[newRowSize++] = colVal[j];
             if ( colIndex < 0 || colIndex >= A21GlobalNCols )
             {
                printf("%4d buildReducedSystem WARNING : A21(%d,%d) out of range\n",
                       mypid_, rowCount, colIndex, A21GlobalNCols);
             } 
             if ( newRowSize > maxRowSize+1 ) 
             {
                printf("%4d : buildReducedSystem WARNING : ",mypid_);
                printf("passing array boundary(2).\n");
             }
          } 
       }
       HYPRE_IJMatrixInsertRow(A21,newRowSize,rowCount,newColInd,newColVal);
       HYPRE_ParCSRMatrixRestoreRow(A_csr,i,&rowSize,&colInd,&colVal);
       rowCount++;
    }
    delete [] newColInd;
    delete [] newColVal;

    //------------------------------------------------------------------
    // finally assemble the matrix and sanitize
    //------------------------------------------------------------------

    HYPRE_IJMatrixAssemble(A21);
    A21_csr = (HYPRE_ParCSRMatrix) HYPRE_IJMatrixGetLocalStorage(A21);
    hypre_MatvecCommPkgCreate((hypre_ParCSRMatrix *) A21_csr);

    if ( HYOutputLevel_ > 3 )
    {
       ncnt = 0;
       MPI_Barrier(MPI_COMM_WORLD);
       while ( ncnt < numProcs_ ) 
       {
          if ( mypid_ == ncnt ) 
          {
             printf("====================================================\n");
             printf("%4d buildReducedSystem : matrix A21 assembled %d.\n",
                                        mypid_,A21StartRow);
             fflush(stdout);
             for ( i = A21StartRow; i < A21StartRow+2*nConstraints_; i++ ) 
             {
                HYPRE_ParCSRMatrixGetRow(A21_csr,i,&rowSize,&colInd,&colVal);
                printf("A21 ROW = %6d (%d)\n", i, rowSize);
                for ( j = 0; j < rowSize; j++ )
                   printf("   col = %6d, val = %e \n", colInd[j], colVal[j]);
                HYPRE_ParCSRMatrixRestoreRow(A21_csr,i,&rowSize,&colInd,&colVal);
             }
             printf("====================================================\n");
          }
          ncnt++;
          MPI_Barrier(MPI_COMM_WORLD);
       }
    }

    //******************************************************************
    // construct invA22
    //------------------------------------------------------------------

    //------------------------------------------------------------------
    // calculate the dimension of invA22
    //------------------------------------------------------------------

    invA22NRows       = A21NRows;
    invA22NCols       = invA22NRows;
    invA22GlobalNRows = A21GlobalNRows;
    invA22GlobalNCols = invA22GlobalNRows;
    if ( HYOutputLevel_ > 0 )
    {
       printf("%4d buildReducedSystem : A22GlobalDim = %d %d\n", mypid_, 
                        invA22GlobalNRows, invA22GlobalNCols);
       printf("%4d buildReducedSystem : A22LocalDim  = %d %d\n", mypid_, 
                        invA22NRows, invA22NCols);
    }

    //------------------------------------------------------------------
    // create a matrix context for A22
    //------------------------------------------------------------------

    ierr  = HYPRE_IJMatrixCreate(comm_,&invA22,invA22GlobalNRows,
                                 invA22GlobalNCols);
    ierr += HYPRE_IJMatrixSetLocalStorageType(invA22, HYPRE_PARCSR);
    ierr += HYPRE_IJMatrixSetLocalSize(invA22, invA22NRows, invA22NCols);
    assert(!ierr);

    //------------------------------------------------------------------
    // compute the no. of nonzeros in the first nConstraint row of invA22
    //------------------------------------------------------------------

    maxRowSize  = 0;
    invA22MatSize = new int[invA22NRows];
    for ( i = 0; i < nConstraints_; i++ ) invA22MatSize[i] = 1;

    //------------------------------------------------------------------
    // compute the number of nonzeros in the second nConstraints row of 
    // invA22 (consisting of [D and A22 block])
    //------------------------------------------------------------------

    for ( i = 0; i < nSelected; i++ ) 
    {
       for ( j = 0; j < nSelected; j++ ) 
       {
          if ( selectedListAux[j] == i ) 
          {
             rowIndex = selectedList[j]; 
             break;
          }
       }
       HYPRE_ParCSRMatrixGetRow(A_csr,rowIndex,&rowSize,&colInd,&colVal);
       rowSize2 = 1;
       for (j = 0; j < rowSize; j++) 
       {
          colIndex = colInd[j];
          if ( colVal[j] != 0.0 ) 
          {
             if ( colIndex >= StartRow && colIndex <= newEndRow ) 
             {
	        searchIndex = hypre_BinarySearch(selectedList, colIndex, 
                                                 nSelected); 
                if ( searchIndex >= 0 ) rowSize2++;
             } 
             else if ( colIndex < StartRow || colIndex > EndRow ) 
             {
	        searchIndex = hypre_BinarySearch(globalSelectedList,colIndex, 
                                                 globalNSelected); 
                if ( searchIndex >= 0 ) rowSize2++;
             }
          }
       }
       invA22MatSize[nConstraints_+i] = rowSize2;
       maxRowSize = ( rowSize2 > maxRowSize ) ? rowSize2 : maxRowSize;
       HYPRE_ParCSRMatrixRestoreRow(A_csr,rowIndex,&rowSize,&colInd,&colVal);
    }

    //------------------------------------------------------------------
    // after fetching the row sizes, set up invA22 with such sizes
    //------------------------------------------------------------------

    ierr  = HYPRE_IJMatrixSetRowSizes(invA22, invA22MatSize);
    ierr += HYPRE_IJMatrixInitialize(invA22);
    assert(!ierr);
    delete [] invA22MatSize;

    //------------------------------------------------------------------
    // next load the first nConstraints_ row to invA22 extracted from A
    // (that is, the D block)
    //------------------------------------------------------------------

    maxRowSize++;
    newColInd = new int[maxRowSize];
    newColVal = new double[maxRowSize];

    for ( i = 0; i < diagCount; i++ ) 
    {
       extDiagonal[i] = 1.0 / extDiagonal[i];
    }
    for ( i = 0; i < nConstraints_; i++ ) {
       newColInd[0] = A21StartRow + nConstraints_ + i; 
       rowIndex     = A21StartRow + i;
       if ( newColInd[0] < 0 || newColInd[0] >= invA22GlobalNCols )
       {
          printf("%4d buildReducedSystem WARNING : A22 out of range %d, %d (%d)\n", 
                 mypid_, rowIndex, newColInd[0], invA22GlobalNCols);
       } 
       newColVal[0] = extDiagonal[A21StartRow/2+i];
       ierr = HYPRE_IJMatrixInsertRow(invA22,1,rowIndex,newColInd,newColVal);
       assert(!ierr);
    }

    //------------------------------------------------------------------
    // next load the second nConstraints_ rows to A22 extracted from A
    //------------------------------------------------------------------

    for ( i = 0; i < nSelected; i++ ) 
    {
       for ( j = 0; j < nSelected; j++ ) 
       {
          if ( selectedListAux[j] == i ) 
          {
             rowIndex = selectedList[j]; 
             break;
          }
       }
       HYPRE_ParCSRMatrixGetRow(A_csr,rowIndex,&rowSize,&colInd,&colVal);
       newRowSize = 1;
       newColInd[0] = A21StartRow + i;
       newColVal[0] = extDiagonal[A21StartRow/2+i]; 
       for (j = 0; j < rowSize; j++) 
       {
          colIndex = colInd[j];
          if ( colVal[j] != 0.0 )
          {
	     searchIndex = hypre_BinarySearch(globalSelectedList,colIndex, 
                                              globalNSelected); 
             if ( searchIndex >= 0 ) 
             {
                searchIndex = globalSelectedListAux[searchIndex];
                for ( procIndex = 0; procIndex < numProcs_; procIndex++ )
                   if ( ProcNRows[procIndex] > colIndex ) break;
                if ( procIndex == numProcs_ )
                {
                   newColInd[newRowSize] = searchIndex + globalNConstr; 
                }
                else
                {
                   newColInd[newRowSize] = searchIndex + 
                                           ProcNConstr[procIndex]; 
                }
                if ( newColInd[newRowSize] < 0 || 
                     newColInd[newRowSize] >= invA22GlobalNCols )
                {
                   printf("%4d buildReducedSystem WARNING : A22 out of range",
                          mypid_);
                   printf(" %d - %d (%d)\n", rowCount, newColInd[newRowSize], 
                          invA22GlobalNCols);
                } 
                newColVal[newRowSize++] = - extDiagonal[A21StartRow/2+i] * 
                                        colVal[j] * extDiagonal[searchIndex];
                if ( newRowSize > maxRowSize )
                {
                   printf("%4d buildReducedSystem : WARNING - ",mypid_);
                   printf("passing array boundary(3).\n");
                }
      	     } 
	  } 
       }
       rowCount = A21StartRow + nConstraints_ + i;
       ierr = HYPRE_IJMatrixInsertRow(invA22, newRowSize, rowCount, 
                                      newColInd, newColVal);
       assert(!ierr);
       HYPRE_ParCSRMatrixRestoreRow(A_csr,rowIndex,&rowSize,&colInd,&colVal);
    }
    delete [] newColInd;
    delete [] newColVal;
    delete [] extDiagonal;

    //------------------------------------------------------------------
    // finally assemble the matrix and sanitize
    //------------------------------------------------------------------

    HYPRE_IJMatrixAssemble(invA22);
    invA22_csr = (HYPRE_ParCSRMatrix) HYPRE_IJMatrixGetLocalStorage(invA22);
    hypre_MatvecCommPkgCreate((hypre_ParCSRMatrix *) invA22_csr);

    if ( HYOutputLevel_ > 3 )
    {
       ncnt = 0;
       MPI_Barrier(MPI_COMM_WORLD);
       while ( ncnt < numProcs_ ) 
       {
          if ( mypid_ == ncnt ) 
          {
             printf("====================================================\n");
             printf("%4d buildReducedSystem : invA22 \n", mypid_);
             for ( i = A21StartRow; i < A21StartRow+2*nConstraints_; i++ ) 
             {
                HYPRE_ParCSRMatrixGetRow(invA22_csr,i,&rowSize,&colInd,&colVal);
                printf("invA22 ROW = %6d (%d)\n", i, rowSize);
                for ( j = 0; j < rowSize; j++ )
                   printf("   col = %6d, val = %e \n", colInd[j], colVal[j]);
                HYPRE_ParCSRMatrixRestoreRow(invA22_csr,i,&rowSize,&colInd,
                                             &colVal);
             }
             printf("====================================================\n");
          }
          MPI_Barrier(MPI_COMM_WORLD);
          ncnt++;
       }
    }

    //******************************************************************
    // perform the triple matrix product
    //------------------------------------------------------------------

    A21_csr = (HYPRE_ParCSRMatrix) HYPRE_IJMatrixGetLocalStorage(A21);
    invA22_csr = (HYPRE_ParCSRMatrix) HYPRE_IJMatrixGetLocalStorage(invA22);
    if ( HYOutputLevel_ > 0 )
    {
       printf("%4d buildReducedSystem : Triple matrix product starts\n",mypid_);
    }
    hypre_ParAMGBuildCoarseOperator( (hypre_ParCSRMatrix *) A21_csr,
                                     (hypre_ParCSRMatrix *) invA22_csr,
                                     (hypre_ParCSRMatrix *) A21_csr,
                                     (hypre_ParCSRMatrix **) &RAP_csr);
    if ( HYOutputLevel_ > 0 )
    {
       printf("%4d buildReducedSystem : Triple matrix product ends\n", mypid_);
    }

    if ( HYOutputLevel_ > 3 )
    {
       MPI_Barrier(MPI_COMM_WORLD);
       ncnt = 0;
       while ( ncnt < numProcs_ )
       {
          if ( mypid_ == ncnt )
          {
             for ( i = A21StartRow; i < A21StartRow+A21NCols; i++ ) {
                HYPRE_ParCSRMatrixGetRow(RAP_csr,i,&rowSize,&colInd, &colVal);
                printf("RAP ROW = %6d (%d)\n", i, rowSize);
                for ( j = 0; j < rowSize; j++ )
                   printf("   col = %6d, val = %e \n", colInd[j], colVal[j]);
                HYPRE_ParCSRMatrixRestoreRow(RAP_csr,i,&rowSize,&colInd,&colVal);
             }
          }
          MPI_Barrier(MPI_COMM_WORLD);
          ncnt++;
       }
    }

    //******************************************************************
    // finally formed the Schur complement reduced system by
    // extracting the A11 part of A and subtracting the RAP
    //------------------------------------------------------------------

    //------------------------------------------------------------------
    // first calculate the dimension of the reduced matrix
    //------------------------------------------------------------------

    newNRows       = nRows - 2 * nConstraints_;
    newGlobalNRows = globalNRows - 2 * globalNConstr;
    ierr  = HYPRE_IJMatrixCreate(comm_,&reducedA,
                                 newGlobalNRows,newGlobalNRows);
    ierr += HYPRE_IJMatrixSetLocalStorageType(reducedA, HYPRE_PARCSR);
    ierr += HYPRE_IJMatrixSetLocalSize(reducedA, newNRows, newNRows);
    assert(!ierr);

    //------------------------------------------------------------------
    // set up reducedA with proper sizes
    //------------------------------------------------------------------

    reducedAMatSize  = new int[newNRows];
    reducedAStartRow = ProcNRows[mypid_] - 2 * ProcNConstr[mypid_];
    rowCount = reducedAStartRow;
    rowIndex = 0;

    for ( i = StartRow; i <= newEndRow; i++ ) 
    {
       searchIndex = hypre_BinarySearch(selectedList, i, nSelected); 
       if ( searchIndex < 0 )  
       {
          HYPRE_ParCSRMatrixGetRow(A_csr,i,&rowSize,&colInd,&colVal);
          ierr = HYPRE_ParCSRMatrixGetRow(RAP_csr,rowCount,&rowSize2,
                                          &colInd2, &colVal2);
          assert( !ierr );
          newRowSize = rowSize + rowSize2;
          newColInd = new int[newRowSize];
          for (j = 0; j < rowSize; j++)  newColInd[j] = colInd[j]; 
          for (j = 0; j < rowSize2; j++) newColInd[rowSize+j] = colInd2[j];
          qsort0(newColInd, 0, newRowSize-1);
          ncnt = 0;
          for ( j = 0; j < newRowSize; j++ ) 
          {
             if ( newColInd[j] != newColInd[ncnt] ) 
             {
                ncnt++;
                newColInd[ncnt] = newColInd[j];
             }  
          }
          reducedAMatSize[rowIndex++] = ncnt;
         
          HYPRE_ParCSRMatrixRestoreRow(A_csr,i,&rowSize,&colInd,&colVal);
          ierr = HYPRE_ParCSRMatrixRestoreRow(RAP_csr,rowCount,&rowSize2,
                                              &colInd2,&colVal2);
          assert( !ierr );
          rowCount++;
       }
    }

    //------------------------------------------------------------------
    // create a matrix context for reducedA
    //------------------------------------------------------------------

    ierr  = HYPRE_IJMatrixSetRowSizes(reducedA, reducedAMatSize);
    ierr += HYPRE_IJMatrixInitialize(reducedA);
    assert(!ierr);
    delete [] reducedAMatSize;

    //------------------------------------------------------------------
    // load the reducedA matrix 
    //------------------------------------------------------------------

    rowCount = reducedAStartRow;
    for ( i = StartRow; i <= newEndRow; i++ ) 
    {
       searchIndex = hypre_BinarySearch(selectedList, i, nSelected); 
       if ( searchIndex < 0 )
       {
          HYPRE_ParCSRMatrixGetRow(A_csr, i, &rowSize, &colInd, &colVal);
          HYPRE_ParCSRMatrixGetRow(RAP_csr,rowCount,&rowSize2,&colInd2,
                                   &colVal2);
          newRowSize = rowSize + rowSize2;
          newColInd  = new int[newRowSize];
          newColVal  = new double[newRowSize];
          ncnt       = 0;
                  
          for ( j = 0; j < rowSize; j++ ) 
          {
             colIndex = colInd[j];
             for ( procIndex = 0; procIndex < numProcs_; procIndex++ )
                if ( ProcNRows[procIndex] > colIndex ) break;
             if ( procIndex == numProcs_ ) 
                ubound = globalNRows-(globalNConstr-ProcNConstr[numProcs_-1]);
             else
                ubound = ProcNRows[procIndex] - 
                         (ProcNConstr[procIndex]-ProcNConstr[procIndex-1]);
             procIndex--;
             if ( colIndex < ubound ) 
             {
                searchIndex = HYFEI_BinarySearch(globalSelectedList,colIndex, 
                                                 globalNSelected); 
                if ( searchIndex < 0 ) 
                {
                   searchIndex = - searchIndex - 1;
                   newColInd[ncnt] = colIndex - ProcNConstr[procIndex] - 
                                     searchIndex;
                   newColVal[ncnt++] = colVal[j]; 
                }
             }
          }
          for ( j = 0; j < rowSize2; j++ ) 
          {
             newColInd[ncnt+j] = colInd2[j]; 
             newColVal[ncnt+j] = - colVal2[j]; 
          }
          newRowSize = ncnt + rowSize2;
          qsort1(newColInd, newColVal, 0, newRowSize-1);
          ncnt = 0;
          for ( j = 0; j < newRowSize; j++ ) 
          {
             if ( j != ncnt && newColInd[j] == newColInd[ncnt] ) 
                newColVal[ncnt] += newColVal[j];
             else if ( newColInd[j] != newColInd[ncnt] ) 
             {
                ncnt++;
                newColVal[ncnt] = newColVal[j];
                newColInd[ncnt] = newColInd[j];
             }  
          } 
          newRowSize = ncnt + 1;
          // translate the newColInd
          ierr = HYPRE_IJMatrixInsertRow(reducedA, newRowSize, rowCount,
                                        newColInd, newColVal);
          assert(!ierr);
          HYPRE_ParCSRMatrixRestoreRow(A_csr,i,&rowSize,&colInd,&colVal);
          HYPRE_ParCSRMatrixRestoreRow(RAP_csr,rowCount,&rowSize2,&colInd2,
                                       &colVal2);
          rowCount++;
          delete [] newColInd;
          delete [] newColVal;
       }
    }

    //------------------------------------------------------------------
    // assemble the reduced matrix
    //------------------------------------------------------------------

    HYPRE_IJMatrixAssemble(reducedA);
    if ( HYOutputLevel_ > 0 )
    {
       printf("%4d buildReducedSystem : FINAL - reducedAStartRow = %d\n", 
                                       mypid_, reducedAStartRow);
    }

    reducedA_csr = (HYPRE_ParCSRMatrix) HYPRE_IJMatrixGetLocalStorage(reducedA);

    if ( HYOutputLevel_ > 3 )
    {
       MPI_Barrier(MPI_COMM_WORLD);
       ncnt = 0;
       while ( ncnt < numProcs_ )
       {
          if ( mypid_ == ncnt )
          {
             printf("====================================================\n");
             for (i=reducedAStartRow;i<reducedAStartRow+nRows-2*nConstraints_;i++)
             {
                printf("%d : reducedA ROW %d\n", mypid_, i);
                ierr = HYPRE_ParCSRMatrixGetRow(reducedA_csr,i,&rowSize,&colInd,
                                                &colVal);
                //qsort1(colInd, colVal, 0, rowSize-1);
                for ( j = 0; j < rowSize; j++ )
                   if ( colVal[j] != 0.0 )
                      printf("%4d %4d %20.13e\n", i+1, colInd[j]+1, colVal[j]);
                HYPRE_ParCSRMatrixRestoreRow(reducedA_csr,i,&rowSize,&colInd,
                                             &colVal);
             }
             printf("====================================================\n");
          }
          MPI_Barrier(MPI_COMM_WORLD);
          ncnt++;
       }
    }

    // *****************************************************************
    // form modified right hand side  (f1 = f1 - A12*invA22*f2)
    // *****************************************************************

    // *****************************************************************
    // form f2hat = invA22 * f2
    //------------------------------------------------------------------

    HYPRE_IJVectorCreate(comm_, &f2, A21GlobalNRows);
    HYPRE_IJVectorSetLocalStorageType(f2, HYPRE_PARCSR);
    if ( HYOutputLevel_ > 0 )
    {
       printf("%4d buildReducedSystem : A21 dims = %d %d %d\n", mypid_, 
               A21StartRow, A21NRows, A21GlobalNRows);
    }
    ierr =  HYPRE_IJVectorSetLocalPartitioning(f2,A21StartRow,
                                               A21StartRow+A21NRows);
    ierr += HYPRE_IJVectorAssemble(f2);
    ierr += HYPRE_IJVectorInitialize(f2);
    ierr += HYPRE_IJVectorZeroLocalComponents(f2);
    //assert(!ierr);

    HYPRE_IJVectorCreate(comm_, &f2hat, A21GlobalNRows);
    HYPRE_IJVectorSetLocalStorageType(f2hat, HYPRE_PARCSR);
    ierr =  HYPRE_IJVectorSetLocalPartitioning(f2hat,A21StartRow,
                                                  A21StartRow+A21NRows);
    ierr += HYPRE_IJVectorAssemble(f2hat);
    ierr += HYPRE_IJVectorInitialize(f2hat);
    ierr += HYPRE_IJVectorZeroLocalComponents(f2hat);
    //assert(!ierr);

    colInd = new int[nSelected*2];
    colVal = new double[nSelected*2];

    for ( i = 0; i < nSelected; i++ ) 
    {
       for ( j = 0; j < nSelected; j++ ) 
       {
          if ( selectedListAux[j] == i ) 
          {
             colInd[i] = selectedList[j];
             break;
          }
       }
       if ( colInd[i] < 0 )
       {
          printf("%4d buildReducedSystem ERROR : out of range %d\n", mypid_,
                  colInd[i]);
          exit(1);
       }
    }
    for ( i = 0; i < nSelected; i++ ) 
    {
       colInd[nSelected+i] = EndRow - nConstraints_ + i + 1;
    }
    HYPRE_IJVectorGetLocalComponents(HYb_, nSelected, colInd,NULL,colVal);
    for ( i = 0; i < nSelected*2; i++ ) colInd[i] = A21StartRow + i;
    ierr = HYPRE_IJVectorSetLocalComponents(f2, nSelected, colInd,
                                            NULL, colVal);
    assert( !ierr );
    f2_csr     = (HYPRE_ParVector) HYPRE_IJVectorGetLocalStorage(f2);
    f2hat_csr  = (HYPRE_ParVector) HYPRE_IJVectorGetLocalStorage(f2hat);
    HYPRE_ParCSRMatrixMatvec( 1.0, invA22_csr, f2_csr, 0.0, f2hat_csr );
    delete [] colVal;
    delete [] colInd;
    HYPRE_IJVectorDestroy(f2); 

    // *****************************************************************
    // set up A12 with proper sizes before forming f2til = A12 * f2hat
    //------------------------------------------------------------------

    //------------------------------------------------------------------
    // calculate the dimension of A12
    //------------------------------------------------------------------

    A12NRows       = A21NCols;
    A12NCols       = A21NRows;
    A12GlobalNRows = A21GlobalNCols;
    A12GlobalNCols = A21GlobalNRows;
    A12MatSize     = new int[A12NRows];
    A12StartRow    = ProcNRows[mypid_] - 2 * ProcNConstr[mypid_];
    if ( HYOutputLevel_ > 0 )
    {
       printf("%4d buildReducedSystem : A12GlobalDim = %d %d\n", mypid_, 
                        A12GlobalNRows, A12GlobalNCols);
       printf("%4d buildReducedSystem : A12LocalDim  = %d %d\n", mypid_, 
                        A12NRows, A12NCols);
    }

    //------------------------------------------------------------------
    // create a matrix context for A12
    //------------------------------------------------------------------

    ierr  = HYPRE_IJMatrixCreate(comm_,&A12,A12GlobalNRows,A12GlobalNCols);
    ierr += HYPRE_IJMatrixSetLocalStorageType(A12, HYPRE_PARCSR);
    ierr += HYPRE_IJMatrixSetLocalSize(A12, A12NRows, A12NCols);
    assert(!ierr);

    //------------------------------------------------------------------
    // compute the number of nonzeros in each row of A12
    // (which consists of the rows in selectedList and the constraints)
    //------------------------------------------------------------------

    rowCount = A12StartRow;
    rowIndex = 0;

    for ( i = StartRow; i <= newEndRow; i++ ) 
    {
       searchIndex = hypre_BinarySearch(selectedList, i, nSelected); 
       if ( searchIndex < 0 )  
       {
          HYPRE_ParCSRMatrixGetRow(A_csr,i,&rowSize,&colInd,&colVal);
          newRowSize = 0;
          for (j = 0; j < rowSize; j++)  
          {
             colIndex = colInd[j];
             for ( procIndex = 0; procIndex < numProcs_; procIndex++ )
                if ( ProcNRows[procIndex] > colIndex ) break;
             if ( procIndex == numProcs_ ) 
                ubound = globalNRows-(globalNConstr-ProcNConstr[numProcs_-1]);
             else
                ubound = ProcNRows[procIndex] - 
                         (ProcNConstr[procIndex]-ProcNConstr[procIndex-1]);
             procIndex--;
             if ( colIndex >= ubound ) newRowSize++; 
             else
             {
                if (hypre_BinarySearch(globalSelectedList,colIndex, 
                                                 globalNSelected) >= 0)
                   newRowSize++;
             }
          }
          A12MatSize[rowIndex++] = newRowSize;
          HYPRE_ParCSRMatrixRestoreRow(A_csr,i,&rowSize,&colInd,&colVal);
          rowCount++;
       }
    }
 
    //------------------------------------------------------------------
    // after fetching the row sizes, set up A12 with such sizes
    //------------------------------------------------------------------

    ierr  = HYPRE_IJMatrixSetRowSizes(A12, A12MatSize);
    ierr += HYPRE_IJMatrixInitialize(A12);
    assert(!ierr);
    delete [] A12MatSize;

    //------------------------------------------------------------------
    // load the A12 matrix 
    //------------------------------------------------------------------

    rowCount = A12StartRow;
    for ( i = StartRow; i <= newEndRow; i++ ) 
    {
       searchIndex = hypre_BinarySearch(selectedList, i, nSelected); 
       if ( searchIndex < 0 )
       {
          HYPRE_ParCSRMatrixGetRow(A_csr, i, &rowSize, &colInd, &colVal);
          newRowSize = 0;
          newColInd  = new int[rowSize];
          newColVal  = new double[rowSize];
          for (j = 0; j < rowSize; j++)  
          {
             colIndex = colInd[j];
             for ( procIndex = 0; procIndex < numProcs_; procIndex++ )
                if ( ProcNRows[procIndex] > colIndex ) break;
             if ( procIndex == numProcs_ ) 
                ubound = globalNRows-(globalNConstr-ProcNConstr[numProcs_-1]);
             else
                ubound = ProcNRows[procIndex] - 
                         (ProcNConstr[procIndex]-ProcNConstr[procIndex-1]);
             procIndex--;
             if ( colIndex >= ubound ) { 
                if ( procIndex != numProcs_ - 1 ) 
                {
                   newColInd[newRowSize] = colInd[j] - ubound + 
                                           ProcNConstr[procIndex] +
                                           ProcNConstr[procIndex+1];
                }
                else 
                {
                   newColInd[newRowSize] = colInd[j] - ubound + 
                                           ProcNConstr[procIndex] +
                                           globalNConstr;
                }
                if ( newColInd[newRowSize] < 0 || 
                     newColInd[newRowSize] >= A12GlobalNCols )
                {
                   printf("%4d buildReducedSystem WARNING : A12 col index out ",
                          mypid_);
                   printf("of range %d %d(%d)\n", mypid_, i, 
                           newColInd[newRowSize], A12GlobalNCols);
                }
                newColVal[newRowSize++] = colVal[j];
             } else
             {
                searchIndex = HYFEI_BinarySearch(globalSelectedList,colInd[j],
                                                 globalNSelected);
                if ( searchIndex >= 0) 
                {
                   searchIndex = globalSelectedListAux[searchIndex];
                   newColInd[newRowSize] = searchIndex + 
                                           ProcNConstr[procIndex]; 
                   if ( newColInd[newRowSize] < 0 || 
                        newColInd[newRowSize] >= A12GlobalNCols )
                   {
                      printf("%4d buildReducedSystem WARNING : A12 col index ",
                             mypid_);
                      printf("out of range %d %d(%d)\n", mypid_, i, 
                             newColInd[newRowSize], A12GlobalNCols);
                   }
                   newColVal[newRowSize++] = colVal[j];
                }
             }
          }
          ierr = HYPRE_IJMatrixInsertRow(A12, newRowSize, rowCount,
                                         newColInd, newColVal);
          assert(!ierr);
          HYPRE_ParCSRMatrixRestoreRow(A_csr,i,&rowSize,&colInd,&colVal);

          rowCount++;
          delete [] newColInd;
          delete [] newColVal;
       }
    }

    //------------------------------------------------------------------
    // assemble the A12 matrix 
    //------------------------------------------------------------------

    ierr = HYPRE_IJMatrixAssemble(A12);
    assert( !ierr );
    A12_csr = (HYPRE_ParCSRMatrix) HYPRE_IJMatrixGetLocalStorage(A12);

    if ( HYOutputLevel_ > 3 )
    {
       MPI_Barrier(MPI_COMM_WORLD);
       ncnt = 0;
       while ( ncnt < numProcs_ )
       {
          if ( mypid_ == ncnt )
          {
             printf("====================================================\n");
             for (i=A12StartRow;i<A12StartRow+A12NRows;i++)
             {
                printf("%d : A12 ROW %d\n", mypid_, i+1);
                HYPRE_ParCSRMatrixGetRow(A12_csr,i,&rowSize,&colInd,&colVal);
                //qsort1(colInd, colVal, 0, rowSize-1);
                for ( j = 0; j < rowSize; j++ )
                   if ( colVal[j] != 0.0 )
                      printf(" A12 %d %d %20.13e\n", i+1, colInd[j]+1, colVal[j]);
                HYPRE_ParCSRMatrixRestoreRow(A12_csr,i,&rowSize,&colInd,
                                             &colVal);
             }
             printf("====================================================\n");
          }
          MPI_Barrier(MPI_COMM_WORLD);
          ncnt++;
       }
    }

    //------------------------------------------------------------------
    // form reducedB_ = A12 * f2hat
    //------------------------------------------------------------------

    ierr  = HYPRE_IJVectorCreate(comm_, &reducedB_, newGlobalNRows);
    ierr += HYPRE_IJVectorSetLocalStorageType(reducedB_, HYPRE_PARCSR);
    ierr += HYPRE_IJVectorSetLocalPartitioning(reducedB_,reducedAStartRow,
                                               reducedAStartRow+newNRows);
    ierr += HYPRE_IJVectorAssemble(reducedB_);
    ierr += HYPRE_IJVectorInitialize(reducedB_);
    ierr += HYPRE_IJVectorZeroLocalComponents(reducedB_);
    assert( !ierr );

    reducedB_csr = (HYPRE_ParVector) HYPRE_IJVectorGetLocalStorage(reducedB_);
    HYPRE_ParCSRMatrixMatvec( -1.0, A12_csr, f2hat_csr, 0.0, reducedB_csr );
    HYPRE_IJMatrixDestroy(A12); 
    HYPRE_IJVectorDestroy(f2hat); 
    //for ( i = reducedAStartRow; i < reducedAStartRow+newNRows; i++ ) 
    //{
    //   HYPRE_IJVectorGetLocalComponents(reducedB_, 1, &i, NULL, &ddata);
    //   printf("A12 * invA22 * f2 %d = %e\n", i, ddata);
    //}

    //------------------------------------------------------------------
    // finally form reducedB = f1 - f2til
    //------------------------------------------------------------------

    rowCount = reducedAStartRow;
    for ( i = StartRow; i <= newEndRow; i++ ) 
    {
       if ( hypre_BinarySearch(selectedList, i, nSelected) < 0 ) 
       {
          HYPRE_IJVectorGetLocalComponents(HYb_, 1, &i, NULL, &ddata);
          HYPRE_IJVectorAddToLocalComponents(reducedB_,1,&rowCount,NULL,
                                             &ddata);
          HYPRE_IJVectorGetLocalComponents(reducedB_,1,&rowCount,NULL, 
                                           &searchValue);
          rowCount++;
       }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    //for ( i = reducedAStartRow; 
    //      i < reducedAStartRow+EndRow-StartRow+1-2*nConstraints_; i++ ) 
    //{
    //   HYPRE_IJVectorGetLocalComponents(reducedB_, 1, &i, NULL, &ddata);
    //   printf("RHS(2) %d = %e\n", i, ddata);
    //}

    //******************************************************************
    // set up the system with the new matrix
    //------------------------------------------------------------------

    reducedA_ = reducedA;
    ierr = HYPRE_IJVectorCreate(comm_, &reducedX_, newGlobalNRows);
    ierr = HYPRE_IJVectorSetLocalStorageType(reducedX_, HYPRE_PARCSR);
    ierr = HYPRE_IJVectorSetLocalPartitioning(reducedX_,reducedAStartRow,
                                              reducedAStartRow+newNRows);
    ierr = HYPRE_IJVectorAssemble(reducedX_);
    ierr = HYPRE_IJVectorInitialize(reducedX_);
    ierr = HYPRE_IJVectorZeroLocalComponents(reducedX_);
    assert(!ierr);

    ierr = HYPRE_IJVectorCreate(comm_, &reducedR_, newGlobalNRows);
    ierr = HYPRE_IJVectorSetLocalStorageType(reducedR_, HYPRE_PARCSR);
    ierr = HYPRE_IJVectorSetLocalPartitioning(reducedR_,reducedAStartRow,
                                              reducedAStartRow+newNRows);
    ierr = HYPRE_IJVectorAssemble(reducedR_);
    ierr = HYPRE_IJVectorInitialize(reducedR_);
    ierr = HYPRE_IJVectorZeroLocalComponents(reducedR_);
    assert(!ierr);

    currA_ = reducedA_;
    currB_ = reducedB_;
    currR_ = reducedR_;
    currX_ = reducedX_;

    //******************************************************************
    // save A21 and invA22 for solution recovery
    //------------------------------------------------------------------

    HYA21_    = A21; 
    HYinvA22_ = invA22; 
    systemReduced_ = 1;
    selectedList_ = selectedList;
    selectedListAux_ = selectedListAux;

    //------------------------------------------------------------------
    // final clean up
    //------------------------------------------------------------------

    delete [] globalSelectedList;
    delete [] globalSelectedListAux;
    delete [] ProcNRows;
    delete [] ProcNConstr;

    //if ( HYA_ != NULL ) {HYPRE_IJMatrixDestroy(HYA_); HYA_ = NULL;}
    if ( colIndices_ != NULL )
    {
       for ( i = 0; i < localEndRow_-localStartRow_+1; i++ )
          if ( colIndices_[i] != NULL ) delete [] colIndices_[i];
       delete [] colIndices_;
       colIndices_ = NULL;
    }
    if ( colValues_ != NULL )
    {
       for ( j = 0; j < localEndRow_-localStartRow_+1; j++ )
          if ( colValues_[j] != NULL ) delete [] colValues_[j];
       delete [] colValues_;
       colValues_ = NULL;
       if ( rowLengths_ != NULL ) 
       {
          delete [] rowLengths_;
          rowLengths_ = NULL;
       }
    }
}

//***************************************************************************
//***************************************************************************
//***************************************************************************
// reading a matrix from a file in ija format (first row : nrows, nnz)
// (read by a single processor)
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::HYFEI_Get_IJAMatrixFromFile(double **val, int **ia, 
             int **ja, int *N, double **rhs, char *matfile, char *rhsfile)
{
    int    i, j, Nrows, nnz, icount, rowindex, colindex, curr_row;
    int    k, m, *mat_ia, *mat_ja, ncnt, rnum;
    double dtemp, *mat_a, value, *rhs_local;
    FILE   *fp;

    //------------------------------------------------------------------
    // read matrix file 
    //------------------------------------------------------------------

    printf("Reading matrix file = %s \n", matfile );
    fp = fopen( matfile, "r" );
    if ( fp == NULL ) {
       printf("Error : file open error (filename=%s).\n", matfile);
       exit(1);
    }
    fscanf(fp, "%d %d", &Nrows, &nnz);
    if ( Nrows <= 0 || nnz <= 0 ) {
       printf("Error : nrows,nnz = %d %d\n", Nrows, nnz);
       exit(1);
    }
    mat_ia = new int[Nrows+1];
    mat_ja = new int[nnz];
    mat_a  = new double[nnz];
    mat_ia[0] = 0;

    curr_row = 0;
    icount   = 0;
    for ( i = 0; i < nnz; i++ ) {
       fscanf(fp, "%d %d %lg", &rowindex, &colindex, &value);
       rowindex--;
       colindex--;
       if ( rowindex != curr_row ) mat_ia[++curr_row] = icount;
       if ( rowindex < 0 || rowindex >= Nrows )
          printf("Error reading row %d (curr_row = %d)\n", rowindex, curr_row);
       if ( colindex < 0 || colindex >= Nrows )
          printf("Error reading col %d (rowindex = %d)\n", colindex, rowindex);
         //if ( value != 0.0 ) {
          mat_ja[icount] = colindex;
          mat_a[icount++]  = value;
         //}
    }
    fclose(fp);
    for ( i = curr_row+1; i <= Nrows; i++ ) mat_ia[i] = icount;
    (*val) = mat_a;
    (*ia)  = mat_ia;
    (*ja)  = mat_ja;
    (*N) = Nrows;
    printf("matrix has %6d rows and %7d nonzeros\n", Nrows, mat_ia[Nrows]);

    //------------------------------------------------------------------
    // read rhs file 
    //------------------------------------------------------------------

    printf("reading rhs file = %s \n", rhsfile );
    fp = fopen( rhsfile, "r" );
    if ( fp == NULL ) {
       printf("Error : file open error (filename=%s).\n", rhsfile);
       exit(1);
    }
    fscanf(fp, "%d", &ncnt);
    if ( ncnt <= 0 || ncnt != Nrows) {
       printf("Error : nrows = %d \n", ncnt);
       exit(1);
    }
    fflush(stdout);
    rhs_local = new double[Nrows];
    m = 0;
    for ( k = 0; k < ncnt; k++ ) {
       fscanf(fp, "%d %lg", &rnum, &dtemp);
       rhs_local[rnum-1] = dtemp; m++;
    }
    fflush(stdout);
    ncnt = m;
    fclose(fp);
    (*rhs) = rhs_local;
    printf("reading rhs done \n");
    for ( i = 0; i < Nrows; i++ ) {
       for ( j = mat_ia[i]; j < mat_ia[i+1]; j++ )
          mat_ja[j]++;
    }
    printf("returning from reading matrix\n");
}

//***************************************************************************
//***************************************************************************
//***************************************************************************
// The following is a test function for the above routines
//***************************************************************************
//***************************************************************************
//***************************************************************************

void fei_hypre_test(int argc, char *argv[])
{
    int    i, j, k, my_rank, num_procs, nrows, nnz, mybegin, myend, status;
    int    *ia, *ja, ncnt, index, chunksize, iterations, local_nrows;
    int    *rowLengths, **colIndices, blksize=1, *list, prec;
    double *val, *rhs, ddata, ddata_max;

    //------------------------------------------------------------------
    // initialize parallel platform
    //------------------------------------------------------------------

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    HYPRE_LinSysCore H(MPI_COMM_WORLD);

    //------------------------------------------------------------------
    // read the matrix and rhs and broadcast
    //------------------------------------------------------------------

    if ( my_rank == 0 ) {
       H.HYFEI_Get_IJAMatrixFromFile(&val, &ia, &ja, &nrows,
                                &rhs, "matrix.data", "rhs.data");
       nnz = ia[nrows];
       MPI_Bcast(&nrows, 1, MPI_INT, 0, MPI_COMM_WORLD);
       MPI_Bcast(&nnz,   1, MPI_INT, 0, MPI_COMM_WORLD);

       MPI_Bcast(ia,  nrows+1, MPI_INT,    0, MPI_COMM_WORLD);
       MPI_Bcast(ja,  nnz,     MPI_INT,    0, MPI_COMM_WORLD);
       MPI_Bcast(val, nnz,     MPI_DOUBLE, 0, MPI_COMM_WORLD);
       MPI_Bcast(rhs, nrows,   MPI_DOUBLE, 0, MPI_COMM_WORLD);

    } else {
       MPI_Bcast(&nrows, 1, MPI_INT, 0, MPI_COMM_WORLD);
       MPI_Bcast(&nnz,   1, MPI_INT, 0, MPI_COMM_WORLD);
       ia  = new int[nrows+1];
       ja  = new int[nnz];
       val = new double[nnz];
       rhs = new double[nrows];

       MPI_Bcast(ia,  nrows+1, MPI_INT,    0, MPI_COMM_WORLD);
       MPI_Bcast(ja,  nnz,     MPI_INT,    0, MPI_COMM_WORLD);
       MPI_Bcast(val, nnz,     MPI_DOUBLE, 0, MPI_COMM_WORLD);
       MPI_Bcast(rhs, nrows,   MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    chunksize = nrows / blksize;
    if ( chunksize * blksize != nrows )
    {
       printf("Cannot put into matrix blocks with block size 3\n");
       exit(1);
    }
    chunksize = chunksize / num_procs;
    mybegin = chunksize * my_rank * blksize;
    myend   = chunksize * (my_rank + 1) * blksize - 1;
    if ( my_rank == num_procs-1 ) myend = nrows - 1;
    printf("Processor %d : begin/end = %d %d\n", my_rank, mybegin, myend);
    fflush(stdout);

    //------------------------------------------------------------------
    // create matrix in the HYPRE context
    //------------------------------------------------------------------

    local_nrows = myend - mybegin + 1;
    H.createMatricesAndVectors(nrows, mybegin+1, local_nrows);

    rowLengths = new int[local_nrows];
    colIndices = new int*[local_nrows];
    for ( i = mybegin; i < myend+1; i++ ) 
    {
       ncnt = ia[i+1] - ia[i];
       rowLengths[i-mybegin] = ncnt;
       colIndices[i-mybegin] = new int[ncnt];
       k = 0;
       for (j = ia[i]; j < ia[i+1]; j++) colIndices[i-mybegin][k++] = ja[j];
    }

    H.allocateMatrix(colIndices, rowLengths);

    for ( i = mybegin; i < myend+1; i++ ) delete [] colIndices[i-mybegin];
    delete [] colIndices;
    delete [] rowLengths;

    //------------------------------------------------------------------
    // load the matrix 
    //------------------------------------------------------------------

    for ( i = mybegin; i <= myend; i++ ) {
       ncnt = ia[i+1] - ia[i];
       index = i + 1;
       H.sumIntoSystemMatrix(index, ncnt, &val[ia[i]], &ja[ia[i]]);
    }
    H.matrixLoadComplete();
    delete [] ia;
    delete [] ja;
    delete [] val;
    
    //------------------------------------------------------------------
    // load the right hand side 
    //------------------------------------------------------------------

    for ( i = mybegin; i <= myend; i++ ) 
    {
       index = i + 1;
       H.sumIntoRHSVector(1, &rhs[i], &index);
    }
    delete [] rhs;

    //------------------------------------------------------------------
    // set other parameters
    //------------------------------------------------------------------

    char *paramString = new char[100];

    strcpy(paramString, "solver gmres");
    H.parameters(1, &paramString);
    if ( my_rank == 0 )
    {
       printf("preconditioner (diagonal,parasails,boomeramg,ml) : ");
       scanf("%d", &prec);
    }
    MPI_Bcast(&prec,  1, MPI_INT, 0, MPI_COMM_WORLD);
    switch (prec)
    {
       case 0 : strcpy(paramString, "preconditioner diagonal");
                break;
       case 1 : strcpy(paramString, "preconditioner parasails");
                break;
       case 2 : strcpy(paramString, "preconditioner boomeramg");
                break;
       case 3 : strcpy(paramString, "preconditioner ml");
                break;
       default : strcpy(paramString, "preconditioner parasails");
                break;
    }

    H.parameters(1, &paramString);
    strcpy(paramString, "gmresDim 300");
    H.parameters(1, &paramString);
    strcpy(paramString, "maxIterations 300");
    H.parameters(1, &paramString);

    strcpy(paramString, "amgRelaxType hybrid");
    H.parameters(1, &paramString);
    strcpy(paramString, "amgRelaxWeight 0.5");
    H.parameters(1, &paramString);
    strcpy(paramString, "amgStrongThreshold 0.25");
    H.parameters(1, &paramString);
    strcpy(paramString, "amgNumSweeps 3");
    H.parameters(1, &paramString);

    strcpy(paramString, "mlNumPresweeps 2");
    H.parameters(1, &paramString);
    strcpy(paramString, "mlNumPostsweeps 2");
    H.parameters(1, &paramString);
    strcpy(paramString, "mlPresmootherType sgs");
    H.parameters(1, &paramString);
    strcpy(paramString, "mlPostsmootherType sgs");
    H.parameters(1, &paramString);
    strcpy(paramString, "mlRelaxWeight 0.5");
    H.parameters(1, &paramString);
    strcpy(paramString, "mlStrongThreshold 0.25");
    H.parameters(1, &paramString);

    strcpy(paramString, "parasailsNlevels 1");
    H.parameters(1, &paramString);

    //------------------------------------------------------------------
    // solve the system
    //------------------------------------------------------------------

    H.launchSolver(status, iterations);

    if ( status != 1 )
    {
       printf("%4d : HYPRE_LinSysCore : solve unsuccessful.\n", my_rank);
    } 
    else if ( my_rank == 0 )
    {
       printf("%4d : HYPRE_LinSysCore : solve successful.\n", my_rank);
       printf("                  iteration count = %4d\n", iterations);
    }

    if ( my_rank == 0 )
    {
       for ( i = H.localStartRow_-1; i < H.localEndRow_; i++ )
       {
          HYPRE_IJVectorGetLocalComponents(H.currX_,1,&i, NULL, &ddata);
          //H.getSolnEntry(i, ddata);
          printf("sol(%d): %e\n", i, ddata);
       }
    }

    //------------------------------------------------------------------
    // clean up 
    //------------------------------------------------------------------

    MPI_Finalize();
}

//***************************************************************************
//***************************************************************************
// This section investigates the use of domain decomposition preconditioner
// using AMG.  
//***************************************************************************
//***************************************************************************

HYPRE_IJMatrix localA;
HYPRE_IJVector localb;
HYPRE_IJVector localx;
int            myBegin, myEnd, myRank;

int HYPRE_SeqAMGSolve( HYPRE_Solver solver,
                       HYPRE_ParCSRMatrix A,
                       HYPRE_ParVector b,
                       HYPRE_ParVector x      )
{
   int                i, local_nrows, *indlist;
   hypre_ParCSRMatrix *A2 = (hypre_ParCSRMatrix *) A;
   hypre_ParVector    *b2 = (hypre_ParVector *) b;
   hypre_ParVector    *x2 = (hypre_ParVector *) x;
   hypre_Vector       *x_local = hypre_ParVectorLocalVector(x2);
   double             *x_data  = hypre_VectorData(x_local);
   hypre_Vector       *b_local = hypre_ParVectorLocalVector(b2);
   double             *b_data  = hypre_VectorData(b_local);
   HYPRE_ParCSRMatrix A_csr;
   HYPRE_ParVector    x_csr;
   HYPRE_ParVector    b_csr;

   local_nrows = myEnd - myBegin + 1;
   indlist = new int[local_nrows];
   for (i=0; i<local_nrows; i++) indlist[i] = i;
   HYPRE_IJVectorSetLocalComponents(localb,local_nrows,indlist, NULL, b_data);
   HYPRE_IJVectorSetLocalComponents(localx,local_nrows,indlist, NULL, x_data);
   A_csr  = (HYPRE_ParCSRMatrix) HYPRE_IJMatrixGetLocalStorage(localA);
   x_csr  = (HYPRE_ParVector)    HYPRE_IJVectorGetLocalStorage(localx);
   b_csr  = (HYPRE_ParVector)    HYPRE_IJVectorGetLocalStorage(localb);
   HYPRE_ParAMGSolve( solver, A_csr, b_csr, x_csr );
   hypre_ParVector    *u = (hypre_ParVector *) x_csr;
   hypre_Vector       *u_local = hypre_ParVectorLocalVector(u);
   double             *u_data  = hypre_VectorData(u_local);
   for (i=0; i<local_nrows; i++) x_data[i] = u_data[i];
   delete [] indlist;

   int    j, rowSize, *colInd;
   double *colVal;
   if ( myRank == 10 )
   {
      for ( i = 0; i < local_nrows; i++ )
      {
         HYPRE_ParCSRMatrixGetRow(A_csr,i,&rowSize,&colInd,&colVal);
         for ( j = 0; j < rowSize; j++ )
            printf("A(%d,%d) = %e;\n", i+1, colInd[j]+1, colVal[j]);
         HYPRE_ParCSRMatrixRestoreRow(A_csr,i,&rowSize,&colInd,&colVal);
      }
      for ( i = 0; i < local_nrows; i++ )
      {
         printf("x(%d) = %e;\n", i+1, u_data[i]);
      }
      for ( i = 0; i < local_nrows; i++ )
      {
         printf("b(%d) = %e;\n", i+1, b_data[i]);
      }
   }
   return 0;
}

//***************************************************************************
//***************************************************************************

void fei_hypre_dd(int argc, char *argv[])
{
    int                i, j, k, nrows, nnz;
    int                num_procs, status;
    int                *ia, *ja, ncnt, index, chunksize, iterations;
    int                local_nrows, eqnNum, *rowLengths, **colIndices;
    int                blksize=1, *list, *colInd, *newColInd;
    int                rowSize, newRowSize, maxRowSize;
    double             *val, *rhs, ddata, ddata_max, *colVal, *newColVal;
    MPI_Comm           newComm, dummyComm;

    HYPRE_Solver       SeqPrecon;
    HYPRE_Solver       PSolver;
    HYPRE_ParCSRMatrix A_csr;
    HYPRE_ParVector    x_csr;
    HYPRE_ParVector    b_csr;
    HYPRE_ParVector    r_csr;

    //******************************************************************
    // initialize parallel platform
    //------------------------------------------------------------------

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    HYPRE_LinSysCore H(MPI_COMM_WORLD);

    //******************************************************************
    // read the matrix and rhs and broadcast
    //------------------------------------------------------------------

    if ( myRank == 0 ) {
       H.HYFEI_Get_IJAMatrixFromFile(&val, &ia, &ja, &nrows,
                                &rhs, "matrix.data", "rhs.data");
       nnz = ia[nrows];
       MPI_Bcast(&nrows, 1, MPI_INT, 0, MPI_COMM_WORLD);
       MPI_Bcast(&nnz,   1, MPI_INT, 0, MPI_COMM_WORLD);

       MPI_Bcast(ia,  nrows+1, MPI_INT,    0, MPI_COMM_WORLD);
       MPI_Bcast(ja,  nnz,     MPI_INT,    0, MPI_COMM_WORLD);
       MPI_Bcast(val, nnz,     MPI_DOUBLE, 0, MPI_COMM_WORLD);
       MPI_Bcast(rhs, nrows,   MPI_DOUBLE, 0, MPI_COMM_WORLD);

    } else {
       MPI_Bcast(&nrows, 1, MPI_INT, 0, MPI_COMM_WORLD);
       MPI_Bcast(&nnz,   1, MPI_INT, 0, MPI_COMM_WORLD);
       ia  = new int[nrows+1];
       ja  = new int[nnz];
       val = new double[nnz];
       rhs = new double[nrows];

       MPI_Bcast(ia,  nrows+1, MPI_INT,    0, MPI_COMM_WORLD);
       MPI_Bcast(ja,  nnz,     MPI_INT,    0, MPI_COMM_WORLD);
       MPI_Bcast(val, nnz,     MPI_DOUBLE, 0, MPI_COMM_WORLD);
       MPI_Bcast(rhs, nrows,   MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    chunksize = nrows / blksize;
    if ( chunksize * blksize != nrows )
    {
       printf("Cannot put into matrix blocks with block size 3\n");
       exit(1);
    }
    chunksize = chunksize / num_procs;
    myBegin = chunksize * myRank * blksize;
    myEnd   = chunksize * (myRank + 1) * blksize - 1;
    if ( myRank == num_procs-1 ) myEnd = nrows - 1;
    printf("Processor %d : begin/end = %d %d\n", myRank, myBegin, myEnd);
    fflush(stdout);

    //******************************************************************
    // create and load the global matrix in the HYPRE context
    //------------------------------------------------------------------

    local_nrows = myEnd - myBegin + 1;
    H.createMatricesAndVectors(nrows, myBegin+1, local_nrows);

    rowLengths = new int[local_nrows];
    colIndices = new int*[local_nrows];
    for ( i = myBegin; i < myEnd+1; i++ ) 
    {
       ncnt = ia[i+1] - ia[i];
       rowLengths[i-myBegin] = ncnt;
       colIndices[i-myBegin] = new int[ncnt];
       k = 0;
       for (j = ia[i]; j < ia[i+1]; j++) colIndices[i-myBegin][k++] = ja[j];
    }

    H.allocateMatrix(colIndices, rowLengths);

    for ( i = myBegin; i < myEnd+1; i++ ) delete [] colIndices[i-myBegin];
    delete [] colIndices;
    delete [] rowLengths;

    for ( i = myBegin; i <= myEnd; i++ ) 
    {
       ncnt = ia[i+1] - ia[i];
       index = i + 1;
       H.sumIntoSystemMatrix(index, ncnt, &val[ia[i]], &ja[ia[i]]);
    }
    H.matrixLoadComplete();
    delete [] ia;
    delete [] ja;
    delete [] val;
    
    //******************************************************************
    // load the right hand side 
    //------------------------------------------------------------------

    for ( i = myBegin; i <= myEnd; i++ ) 
    {
       index = i + 1;
       H.sumIntoRHSVector(1, &rhs[i], &index);
    }
    delete [] rhs;

    //******************************************************************
    // create and load a local matrix 
    //------------------------------------------------------------------

    local_nrows = myEnd - myBegin + 1;
    for ( i = 0; i < num_procs; i++ )
    {
       if ( myRank == i )
          MPI_Comm_split(MPI_COMM_WORLD, i+1, 0, &newComm);
       else
          MPI_Comm_split(MPI_COMM_WORLD, MPI_UNDEFINED, 1, &dummyComm);
    }
    printf("%d : newComm = %d\n", H.mypid_, (int) newComm);
    MPI_Comm_rank(newComm, &i);
    MPI_Comm_size(newComm, &j);
    printf("%d : my new rank %d = %d(%d)\n", H.mypid_, (int) newComm, i,j);
    HYPRE_IJMatrixCreate(newComm,&localA,local_nrows,local_nrows);
    HYPRE_IJMatrixSetLocalStorageType(localA, HYPRE_PARCSR);
    HYPRE_IJMatrixSetLocalSize(localA, local_nrows, local_nrows);
    rowLengths = new int[local_nrows];
    A_csr  = (HYPRE_ParCSRMatrix) HYPRE_IJMatrixGetLocalStorage(H.HYA_);
    maxRowSize = 0;
    for ( i = myBegin; i <= myEnd; i++ )
    {
       rowLengths[i-myBegin] = 0;
       HYPRE_ParCSRMatrixGetRow(A_csr,i,&rowSize,&colInd,&colVal);
       for ( j = 0; j < rowSize; j++ )
          if ( colInd[j] >= myBegin && colInd[j] <= myEnd ) 
             rowLengths[i-myBegin]++;
       nnz += rowLengths[i-myBegin];
       maxRowSize = (rowLengths[i-myBegin] > maxRowSize) ? 
                            rowLengths[i-myBegin] : maxRowSize;
       HYPRE_ParCSRMatrixRestoreRow(A_csr,i,&rowSize,&colInd,&colVal);
    }
    HYPRE_IJMatrixSetRowSizes(localA, rowLengths);
    HYPRE_IJMatrixInitialize(localA);
    newColInd = new int[maxRowSize];
    newColVal = new double[maxRowSize];
    for ( i = 0; i < local_nrows; i++ )
    {
       eqnNum = myBegin + i;
       HYPRE_ParCSRMatrixGetRow(A_csr,eqnNum,&rowSize,&colInd,&colVal);
       nnz = 0;
       for ( j = 0; j < rowSize; j++ )
          if ( colInd[j] >= myBegin && colInd[j] <= myEnd ) 
          {
             newColInd[nnz] = colInd[j] - myBegin;
             newColVal[nnz++] = colVal[j];
          }
       HYPRE_ParCSRMatrixRestoreRow(A_csr,eqnNum,&rowSize,&colInd,&colVal);
       HYPRE_IJMatrixInsertRow(localA,nnz,i,newColInd,newColVal);
    }
    delete [] newColInd;
    delete [] newColVal;
    HYPRE_IJMatrixAssemble(localA);

    //******************************************************************
    // create and load local vectors 
    //------------------------------------------------------------------

    HYPRE_IJVectorCreate(newComm, &localx, local_nrows);
    HYPRE_IJVectorSetLocalStorageType(localx, HYPRE_PARCSR);
    HYPRE_IJVectorSetLocalPartitioning(localx, 0, local_nrows);
    HYPRE_IJVectorAssemble(localx);
    HYPRE_IJVectorInitialize(localx);
    HYPRE_IJVectorZeroLocalComponents(localx);
    HYPRE_IJVectorCreate(newComm, &localb, local_nrows);
    HYPRE_IJVectorSetLocalStorageType(localb, HYPRE_PARCSR);
    HYPRE_IJVectorSetLocalPartitioning(localb, 0, local_nrows);
    HYPRE_IJVectorAssemble(localb);
    HYPRE_IJVectorInitialize(localb);
    HYPRE_IJVectorZeroLocalComponents(localb);

    //******************************************************************
    // create a GMRES context
    //------------------------------------------------------------------

    printf("%d : creating GMRES \n", H.mypid_);
    HYPRE_ParCSRGMRESCreate(MPI_COMM_WORLD, &PSolver);
    HYPRE_ParCSRGMRESSetKDim(PSolver, 100);
    HYPRE_ParCSRGMRESSetMaxIter(PSolver, 1000);
    HYPRE_ParCSRGMRESSetTol(PSolver, 1.0E-8);
    A_csr  = (HYPRE_ParCSRMatrix) HYPRE_IJMatrixGetLocalStorage(H.HYA_);
    x_csr  = (HYPRE_ParVector)    HYPRE_IJVectorGetLocalStorage(H.HYx_);
    b_csr  = (HYPRE_ParVector)    HYPRE_IJVectorGetLocalStorage(H.HYb_);
    HYPRE_ParCSRGMRESSetup(PSolver, A_csr, b_csr, x_csr);

    //******************************************************************
    // create an AMG context
    //------------------------------------------------------------------

    HYPRE_ParAMGCreate(&SeqPrecon);
    HYPRE_ParAMGSetMaxIter(SeqPrecon, 1);
    HYPRE_ParAMGSetCycleType(SeqPrecon, 1);
    HYPRE_ParAMGSetMaxLevels(SeqPrecon, 25);
    HYPRE_ParAMGSetTol(SeqPrecon, 1.0E-16);
    HYPRE_ParAMGSetMeasureType(SeqPrecon, 0);
    HYPRE_ParCSRGMRESSetPrecond(PSolver, HYPRE_SeqAMGSolve, 
                                HYPRE_ParAMGSetup, SeqPrecon);
    A_csr  = (HYPRE_ParCSRMatrix) HYPRE_IJMatrixGetLocalStorage(localA);
    //HYPRE_ParAMGSetIOutDat(SeqPrecon, 2);
    //HYPRE_ParAMGSetDebugFlag(SeqPrecon, 1);
    HYPRE_ParAMGSetup( SeqPrecon, A_csr, NULL, NULL);
    MPI_Barrier(MPI_COMM_WORLD);

    //******************************************************************
    // solve 
    //------------------------------------------------------------------

    A_csr  = (HYPRE_ParCSRMatrix) HYPRE_IJMatrixGetLocalStorage(H.HYA_);
    x_csr  = (HYPRE_ParVector)    HYPRE_IJVectorGetLocalStorage(H.HYx_);
    b_csr  = (HYPRE_ParVector)    HYPRE_IJVectorGetLocalStorage(H.HYb_);
    HYPRE_ParCSRGMRESSolve(PSolver, A_csr, b_csr, x_csr);

    //------------------------------------------------------------------
    // clean up 
    //------------------------------------------------------------------

    MPI_Finalize();
}

