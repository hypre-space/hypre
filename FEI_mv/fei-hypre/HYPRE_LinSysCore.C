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
#ifndef NOFEI
#include "base/Data.h"
#include "base/basicTypes.h"
#include "base/LinearSystemCore.h"
#endif
#include "HYPRE.h"
#include "../../IJ_matrix_vector/HYPRE_IJ_mv.h"
#include "../../parcsr_matrix_vector/HYPRE_parcsr_mv.h"
#include "../../parcsr_linear_solvers/HYPRE_parcsr_ls.h"
#include "HYPRE_LinSysCore.h"

#define abs(x) (((x) > 0.0) ? x : -(x))

//---------------------------------------------------------------------------
// parcsr_matrix_vector.h is put here instead of in HYPRE_LinSysCore.h 
// because it gives warning when compiling cfei.cc
//---------------------------------------------------------------------------

#include "parcsr_matrix_vector/parcsr_matrix_vector.h"
#include "hypre_lsi_ddilut.h"

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

   int HYPRE_LSI_DDIlutSetOutputLevel(HYPRE_Solver, int);
   int hypre_BoomerAMGBuildCoarseOperator(hypre_ParCSRMatrix*,
                                       hypre_ParCSRMatrix*,
                                       hypre_ParCSRMatrix*,
                                       hypre_ParCSRMatrix**);
   void qsort0(int *, int, int);
   void qsort1(int *, double *, int, int);
   int  HYPRE_DummyFunction(HYPRE_Solver, HYPRE_ParCSRMatrix,
                            HYPRE_ParVector, HYPRE_ParVector) {return 0;}

   int   getMatrixCSR(HYPRE_IJMatrix,int nrows,int nnz,int*,int*,double*);
   int   HYPRE_LSI_Search(int*, int, int);
   void  HYPRE_LSI_Get_IJAMatrixFromFile(double **val, int **ia, int **ja, 
                  int *N, double **rhs, char *matfile, char *rhsfile);

#ifdef Y12M
   void y12maf_(int*,int*,double*,int*,int*,int*,int*,double*,
                int*,int*, double*,int*,double*,int*);
#endif
}

//***************************************************************************
// constructor
//---------------------------------------------------------------------------

HYPRE_LinSysCore::HYPRE_LinSysCore(MPI_Comm comm) : 
#ifdef FEI_V12
                  LinearSystemCore(comm),
#endif
                  comm_(comm),
                  HYA_(NULL),
                  HYA21_(NULL),
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
                  finalResNorm_(0.0),
                  rowLengths_(NULL),
                  colIndices_(NULL),
                  colValues_(NULL),
                  selectedList_(NULL),
                  selectedListAux_(NULL),
                  node2EqnMap(NULL),
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

    pilutRowSize_       = 0;    // how many nonzeros to keep in L and U
    pilutDropTol_       = 0.0;
    pilutMaxNnzPerRow_  = 0;    // register the max NNZ/per in matrix A

    ddilutFillin_       = 1.0;  // additional fillin other than A
    ddilutDropTol_      = 0.0;

    parasailsSym_       = 0;    // default is nonsymmetric
    parasailsThreshold_ = 0.1;
    parasailsNlevels_   = 1;
    parasailsFilter_    = 0.05;
    parasailsLoadbal_   = 0.0;
    parasailsReuse_     = 0;    // reuse pattern if nonzero

    superluOrdering_    = 0;    // natural ordering in SuperLU
    superluScale_[0]    = 'N';  // no scaling in SuperLUX
    gmresDim_           = 100;  // restart size in GMRES
    mlNumPreSweeps_     = 1;
    mlNumPostSweeps_    = 1;
    mlPresmootherType_  = 1;    // default Gauss-Seidel
    mlPostsmootherType_ = 1;    // default Gauss-Seidel
    mlRelaxWeight_      = 0.5;
    mlStrongThreshold_  = 0.08; // one suggested by Vanek/Brezina/Mandel

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
    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
    {
       printf("%4d : HYPRE_LinSysCore::entering destructor.\n",mypid_);
    }

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
    if (reducedA_ != NULL) {HYPRE_IJMatrixDestroy(reducedA_); reducedA_ = NULL;}
    if (reducedB_ != NULL) {HYPRE_IJVectorDestroy(reducedB_); reducedB_ = NULL;}
    if (reducedX_ != NULL) {HYPRE_IJVectorDestroy(reducedX_); reducedX_ = NULL;}
    if (reducedR_ != NULL) {HYPRE_IJVectorDestroy(reducedR_); reducedR_ = NULL;}
    if (HYA21_    != NULL) {HYPRE_IJMatrixDestroy(HYA21_);    HYA21_    = NULL;}
    if (HYinvA22_ != NULL) {HYPRE_IJMatrixDestroy(HYinvA22_); HYinvA22_ = NULL;}

    matrixVectorsCreated_ = 0;
    systemAssembled_ = 0;

    if ( node2EqnMap != NULL ) delete [] node2EqnMap;

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
          HYPRE_BoomerAMGDestroy( HYPrecon_ );

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
    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
    {
       printf("%4d : HYPRE_LinSysCore::leaving  destructor.\n",mypid_);
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
    int    i, nsweeps, rtype, olevel;
    double weight;
    char   param[256], param1[256], param2[80];

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
    {
       printf("%4d : HYPRE_LinSysCore::entering parameters function.\n",mypid_);
       if ( mypid_ == 0 )
       {
          printf("HYPRE_LinSysCore::parameters - numParams = %d\n", numParams);
          for ( int i = 0; i < numParams; i++ )
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
             printf("       HYPRE_LinSysCore::parameters outputLevel = %d\n",
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
          if (!strcmp(param2, "printMat")) HYOutputLevel_ |= HYFEI_PRINTMAT;
          if (!strcmp(param2, "printSol")) HYOutputLevel_ |= HYFEI_PRINTSOL;
          if (!strcmp(param2, "printReducedMat")) 
             HYOutputLevel_ |= HYFEI_PRINTREDMAT;
          if (!strcmp(param2, "ddilut")) HYOutputLevel_ |= HYFEI_DDILUT;
          if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
          {
             printf("       HYPRE_LinSysCore::parameters setDebug.\n");
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
             printf("       HYPRE_LinSysCore::parameters - schur reduction.\n");
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
             printf("       HYPRE_LinSysCore::parameters - slide reduction.\n");
          }
       }
       else if ( !strcmp(param1, "slideReduction2") )
       {
          slideReduction_ = 2;
          if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
          {
             printf("       HYPRE_LinSysCore::parameters - slide reduction.\n");
          }
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
             printf("       HYPRE_LinSysCore::parameters solver = %s\n",
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
             printf("       HYPRE_LinSysCore::parameters gmresDim = %d\n",
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
             printf("       HYPRE_LinSysCore::parameters gmresStopCrit = %s\n",
                    param2);
          }
       }

       //----------------------------------------------------------------
       // which preconditioner : diagonal, pilut, boomeramg, parasails
       //----------------------------------------------------------------

       else if ( !strcmp(param1, "preconditioner") )
       {
          sscanf(params[i],"%s %s", param, param2);
          if ( !strcmp(param2, "reuse" ) ) HYPreconReuse_ = 1;
          else
          {
             sscanf(params[i],"%s %s", param, HYPreconName_);
             selectPreconditioner(HYPreconName_);
          }
          if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
          {
             printf("       HYPRE_LinSysCore::parameters preconditioner = %s\n",
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
             printf("       HYPRE_LinSysCore::parameters maxIterations = %d\n",
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
             printf("       HYPRE_LinSysCore::parameters tolerance = %e\n",
                    tolerance_);
          }
       }

       //----------------------------------------------------------------
       // pilut preconditioner : max number of nonzeros to keep per row
       //----------------------------------------------------------------

       else if ( !strcmp(param1, "pilutRowSize") )
       {
          sscanf(params[i],"%s %d", param, &pilutRowSize_);
          if ( pilutRowSize_ < 1 ) pilutRowSize_ = 50;
          if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
          {
             printf("       HYPRE_LinSysCore::parameters pilutRowSize = %d\n",
                    pilutRowSize_);
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
             printf("       HYPRE_LinSysCore::parameters pilutDropTol = %e\n",
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
             printf("       HYPRE_LinSysCore::parameters ddilutFillin = %d\n",
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
             printf("       HYPRE_LinSysCore::parameters ddilutDropTol = %e\n",
                    ddilutDropTol_);
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
             printf("       HYPRE_LinSysCore::parameters superluOrdering = %s\n",
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
             printf("       HYPRE_LinSysCore::parameters superluScale = %s\n",
                    params);
          }
       }

       //----------------------------------------------------------------
       // amg preconditoner : coarsening type 
       //----------------------------------------------------------------

       else if ( !strcmp(param1, "amgCoarsenType") )
       {
          sscanf(params[i],"%s %s", param, param2);
          if      ( !strcmp(param2, "ruge" ) )    amgCoarsenType_ = 1;
          else if ( !strcmp(param2, "falgout" ) ) amgCoarsenType_ = 6;
          else if ( !strcmp(param2, "default" ) ) amgCoarsenType_ = 0;
          if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
          {
             printf("       HYPRE_LinSysCore::parameters amgCoarsenType = %s\n",
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
          for ( i = 0; i < 3; i++ ) amgNumSweeps_[i] = nsweeps;
          if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
          {
             printf("       HYPRE_LinSysCore::parameters amgNumSweeps = %d\n",
                    nsweeps);
          }
       }

       //---------------------------------------------------------------
       // amg preconditoner : which smoother to use
       //----------------------------------------------------------------

       else if ( !strcmp(param1, "amgRelaxType") )
       {
          sscanf(params[i],"%s %s", param, param2);
          if      ( !strcmp(param2, "jacobi" ) ) rtype = 2;
          else if ( !strcmp(param2, "gsSlow") )  rtype = 1;
          else if ( !strcmp(param2, "gsFast") )  rtype = 4;
          else if ( !strcmp(param2, "hybrid" ) ) rtype = 3;
          else if ( !strcmp(param2, "hybridsym" ) ) rtype = 5;
          else                                   rtype = 4;
          for ( i = 0; i < 3; i++ ) amgRelaxType_[i] = rtype;
          if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
          {
             printf("       HYPRE_LinSysCore::parameters amgRelaxType = %s\n",
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
          for ( i = 0; i < 25; i++ ) amgRelaxWeight_[i] = weight;
          if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
          {
             printf("       HYPRE_LinSysCore::parameters amgRelaxWeight = %e\n",
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
             printf("       HYPRE_LinSysCore::parameters amgStrongThreshold = %e\n",
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
             printf("       HYPRE_LinSysCore::parameters parasailsThreshold = %e\n",
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
             printf("       HYPRE_LinSysCore::parameters parasailsNlevels = %d\n",
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
             printf("       HYPRE_LinSysCore::parameters parasailsFilter = %e\n",
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
             printf("       HYPRE_LinSysCore::parameters parasailsLoadbal = %e\n",
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
             printf("       HYPRE_LinSysCore::parameters parasailsSym = %d\n",
                    parasailsSym_);
          }
       }
       else if ( !strcmp(param1, "parasailsUnSymmetric") )
       {
          parasailsSym_ = 0;
          if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
          {
             printf("       HYPRE_LinSysCore::parameters parasailsSym = %d\n",
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
             printf("       HYPRE_LinSysCore::parameters parasailsReuse = %d\n",
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
             printf("       HYPRE_LinSysCore::parameters mlNumPresweeps = %d\n",
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
             printf("       HYPRE_LinSysCore::parameters mlNumPostsweeps = %d\n",
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
             printf("       HYPRE_LinSysCore::parameters mlNumSweeps = %d\n",
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
          if      ( !strcmp(param2, "jacobi" ) ) rtype = 0;
          else if ( !strcmp(param2, "gs") )      rtype = 1;
          else if ( !strcmp(param2, "sgs") )     rtype = 2;
          else if ( !strcmp(param2, "vbjacobi")) rtype = 3;
          else if ( !strcmp(param2, "vbsgs") )   rtype = 4;
          else if ( !strcmp(param2, "vbsgsseq")) rtype = 5;
          else if ( !strcmp(param2, "ilut") )    rtype = 6;
          mlPresmootherType_  = rtype;
          if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
          {
             printf("       HYPRE_LinSysCore::parameters mlPresmootherType = %s\n",
                    param2);
          }
       }
       else if ( !strcmp(param1, "mlPostsmootherType") )
       {
          sscanf(params[i],"%s %s", param, param2);
          rtype = 1;
          if      ( !strcmp(param2, "jacobi" ) ) rtype = 0;
          else if ( !strcmp(param2, "gs") )      rtype = 1;
          else if ( !strcmp(param2, "sgs") )     rtype = 2;
          else if ( !strcmp(param2, "vbjacobi")) rtype = 3;
          else if ( !strcmp(param2, "vbsgs") )   rtype = 4;
          else if ( !strcmp(param2, "vbsgsseq")) rtype = 5;
          mlPostsmootherType_  = rtype;
          if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
          {
             printf("       HYPRE_LinSysCore::parameters mlPostsmootherType = %s\n",
                    param2);
          }
       }
       else if ( !strcmp(param1, "mlRelaxType") )
       {
          sscanf(params[i],"%s %s", param, param2);
          rtype = 1;
          if      ( !strcmp(param2, "jacobi" ) ) rtype = 0;
          else if ( !strcmp(param2, "gs") )      rtype = 1;
          else if ( !strcmp(param2, "sgs") )     rtype = 2;
          else if ( !strcmp(param2, "vbjacobi")) rtype = 3;
          else if ( !strcmp(param2, "vbsgs") )   rtype = 4;
          else if ( !strcmp(param2, "vbsgsseq")) rtype = 5;
          else if ( !strcmp(param2, "ilut") )    rtype = 6;
          mlPresmootherType_  = rtype;
          mlPostsmootherType_ = rtype;
          if ( rtype == 6 ) mlPostsmootherType_ = 1;
          if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
          {
             printf("       HYPRE_LinSysCore::parameters mlRelaxType = %s\n",
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
             printf("       HYPRE_LinSysCore::parameters mlRelaxWeight = %e\n",
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
             printf("       HYPRE_LinSysCore::parameters mlStrongThreshold = %e\n",
                    mlStrongThreshold_);
          }
       }

       //---------------------------------------------------------------
       // error 
       //---------------------------------------------------------------

       else
       {
          if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0 )
          {
             printf("HYPRE_LinSysCore::parameters WARNING : %s not recognized\n",
                    params[i]);
          }
       }
    }

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
    {
       printf("%4d : HYPRE_LinSysCore::leaving  parameters function.\n",mypid_);
    }

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

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
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

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
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

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
    {
       printf("%4d : HYPRE_LinSysCore::entering allocateMatrix.\n", mypid_);
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

    ierr = HYPRE_IJMatrixSetRowSizes(HYA_, rowLengths_);
    ierr = HYPRE_IJMatrixInitialize(HYA_);
    assert(!ierr);

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
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

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
    {
       printf("%4d : HYPRE_LinSysCore::entering resetMatrixAndVector.\n",mypid_);
    }

    if ( s != 0.0 && mypid_ == 0 )
    {
       printf("resetMatrixAndVector ERROR : cannot take nonzeros.\n");
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

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
    {
       printf("%4d : HYPRE_LinSysCore::leaving  resetMatrixAndVector.\n",
              mypid_);
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

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
    {
       printf("%4d : HYPRE_LinSysCore::entering sumIntoSystemMatrix.\n",mypid_);
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
          exit(1);
       }
       colValues_[localRow][index] += values[i];
    }

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
    {
       printf("%4d : HYPRE_LinSysCore::leaving  sumIntoSystemMatrix.\n",mypid_);
    }
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
       printf("%d : HYPRE_LinSysCore::entering sumIntoRHSVector.\n", mypid_);
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
       if ( indices[i] >= localStartRow_  && indices[i] <= localEndRow_ )
          local_ind[i] = indices[i] - 1;
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
       printf("%d : HYPRE_LinSysCore::leaving  sumIntoRHSVector.\n", mypid_);
    }
}

//***************************************************************************
// start assembling the matrix into its internal format
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::matrixLoadComplete()
{
    int i, j, numLocalEqns, leng, eqnNum, nnz;

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
    {
       printf("%4d : HYPRE_LinSysCore::entering matrixLoadComplete.\n",mypid_);
    }

    //-------------------------------------------------------------------
    // load the matrix stored locally to a HYPRE matrix
    //-------------------------------------------------------------------

    numLocalEqns = localEndRow_ - localStartRow_ + 1;
    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
    {
       printf("%4d : HYPRE_LinSysCore::matrixLoadComplete - NEqns = %d.\n",
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
       printf("%4d : HYPRE_LinSysCore::matrixLoadComplete - nnz = %d.\n",
               mypid_, nnz);
    }
    delete [] colValues_;
    colValues_ = NULL;

    HYPRE_IJMatrixAssemble(HYA_);
    systemAssembled_ = 1;
    currA_ = HYA_;
    currB_ = HYb_;
    currX_ = HYx_;
    currR_ = HYr_;

    if ( HYOutputLevel_ & HYFEI_PRINTMAT )
    {
       int    rowSize, *colInd, nnz, nrows;
       double *colVal, value;
       char   fname[40];
       FILE   *fp;
       HYPRE_ParCSRMatrix A_csr;

       A_csr  = (HYPRE_ParCSRMatrix) HYPRE_IJMatrixGetLocalStorage(currA_);
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
    }

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
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

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
    {
       printf("%4d : HYPRE_LinSysCore::entering enforceEssentialBC.\n",mypid_);
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
    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
    {
       printf("%4d : HYPRE_LinSysCore::leaving  enforceEssentialBC.\n",mypid_);
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
       printf("%4d : HYPRE_LinSysCore::entering enforceRemoteEssBC.\n",mypid_);
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

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
    {
       printf("%4d : HYPRE_LinSysCore::leaving  enforceRemoteEssBC.\n",mypid_);
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
       printf("%4d : HYPRE_LinSysCore::entering enforceOtherBC.\n",mypid_);
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

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
    {
       printf("%4d : HYPRE_LinSysCore::leaving  enforceOtherBC.\n",mypid_);
    }
}

//***************************************************************************
// put the pointer to the A matrix into the Data object
//---------------------------------------------------------------------------

#ifndef NOFEI
void HYPRE_LinSysCore::getMatrixPtr(Data& data) 
{
   (void) data;
   printf("HYPRE_LinSysCore::getmatrixPtr ERROR - not implemented yet.\n");
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
    printf("HYPRE_LinSysCore::copyInMatrix ERROR - not implemented yet.\n");
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
    printf("HYPRE_LinSysCore::copyOutMatrix ERROR - not implemented yet.\n");
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
    printf("HYPRE_LinSysCore::sumInMatrix ERROR - not implemented yet.\n");
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
       printf("%4d : HYPRE_LinSysCore::entering getRHSVectorPtr.\n",mypid_);
    }

    data.setTypeName("IJ_Vector");
    data.setDataPtr((void*) HYb_);

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
    {
       printf("%4d : HYPRE_LinSysCore::leaving  getRHSVectorPtr.\n",mypid_);
    }
}
#endif

//***************************************************************************

#ifndef NOFEI
void HYPRE_LinSysCore::copyInRHSVector(double scalar, const Data& data) 
{
    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
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

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
    {
       printf("%4d : HYPRE_LinSysCore::leaving  copyInRHSVector.\n",mypid_);
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

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
    {
       printf("%4d : HYPRE_LinSysCore::leaving  copyOutRHSVector.\n",mypid_);
    }
}
#endif 

//***************************************************************************

#ifndef NOFEI
void HYPRE_LinSysCore::sumInRHSVector(double scalar, const Data& data) 
{
    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
    {
       printf("%4d : HYPRE_LinSysCore::entering sumInRHSVector.\n",mypid_);
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
       printf("%4d : HYPRE_LinSysCore::leaving  sumInRHSVector.\n",mypid_);
    }
}
#endif 

//***************************************************************************

#ifndef NOFEI
void HYPRE_LinSysCore::destroyMatrixData(Data& data) 
{
    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
    {
       printf("%4d : HYPRE_LinSysCore::entering destroyMatrixData.\n",mypid_);
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
       printf("%4d : HYPRE_LinSysCore::leaving  destroyMatrixData.\n",mypid_);
    }
}
#endif 

//***************************************************************************

#ifndef NOFEI
void HYPRE_LinSysCore::destroyVectorData(Data& data) 
{
    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
    {
       printf("%4d : HYPRE_LinSysCore::entering destroyVectorData.\n",mypid_);
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
       printf("%4d : HYPRE_LinSysCore::leaving  destroyVectorData.\n",mypid_);
    }
}
#endif 

//***************************************************************************

void HYPRE_LinSysCore::setNumRHSVectors(int numRHSs, const int* rhsIDs) 
{
    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
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

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
    {
       printf("%4d : HYPRE_LinSysCore::leaving  setNumRHSVectors.\n",mypid_);
    }
}

//***************************************************************************

void HYPRE_LinSysCore::setRHSID(int rhsID) 
{
    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
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

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
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

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
    {
       printf("%4d : HYPRE_LinSysCore::entering putInitalGuess.\n",mypid_);
    }

    local_ind = new int[leng];
    for ( i = 0; i < leng; i++ ) // change to 0-based
    {
       if (eqnNumbers[i] >= localStartRow_ && eqnNumbers[i] <= localEndRow_)
          local_ind[i] = eqnNumbers[i] - 1;
       else
       {
          printf("%d : putInitialGuess ERROR - index %d out of range\n",
                       mypid_, eqnNumbers[i]);
          exit(1);
       }
    }

    ierr = HYPRE_IJVectorSetLocalComponents(HYx_,leng,local_ind,NULL,values);
    assert(!ierr);

    delete [] local_ind;

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
    {
       printf("%4d : HYPRE_LinSysCore::leaving  putInitalGuess.\n",mypid_);
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
       printf("%4d : HYPRE_LinSysCore::entering getSolution.\n",mypid_);
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
       printf("%4d : HYPRE_LinSysCore::leaving  getSolution.\n",mypid_);
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
       printf("%4d : HYPRE_LinSysCore::entering getSolnEntry.\n",mypid_);
    }

    equation = eqnNumber - 1; // construct 0-based index
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
       printf("%4d : HYPRE_LinSysCore::leaving  getSolnEntry.\n",mypid_);
    }
}

//***************************************************************************
// select which Krylov solver to use
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::selectSolver(char* name) 
{
    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
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
       if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
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
            break;

       case HYGMRES :
            HYPRE_ParCSRGMRESCreate(comm_, &HYSolver_);
            break;
    }

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
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

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
    {
       printf("%4d : HYPRE_LinSysCore::entering selectPreconditioner = %s.\n",
              mypid_, name);
    }
    HYPreconReuse_ = 0;

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

       else if ( HYPreconID_ == HYBOOMERAMG )
          HYPRE_LSI_DDIlutDestroy( HYPrecon_ );

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
    else if ( !strcmp(name, "ddilut") )
    {
       strcpy( HYPreconName_, name );
       HYPreconID_ = HYDDILUT;
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
          printf("                       set default to identity.\n");
       }
       strcpy( HYPreconName_, "identity" );
       HYPreconID_ = HYNONE;
#endif
    }
    else
    {
       if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
       {
          printf("selectPreconditioner error : invalid option.\n");
          printf("                     use default = identity.\n");
       }
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

#ifdef MLPACK
       case HYML :
            ierr = HYPRE_ParCSRMLCreate( comm_, &HYPrecon_ );
            break;
#endif
    }

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
    {
       printf("%4d : HYPRE_LinSysCore::leaving  selectPreconditioner.\n",mypid_);
    }
}

//***************************************************************************
// solve the linear system
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::formResidual(int* eqnNumbers, double* values, int leng)
{
    int                i, index, nrows, startRow, endRow;
    int                *int_array, *gint_array;
    HYPRE_ParCSRMatrix A_csr;
    HYPRE_ParVector    x_csr;
    HYPRE_ParVector    b_csr;
    HYPRE_ParVector    r_csr;

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
    {
       printf("%4d : HYPRE_LinSysCore::entering formResidual.\n", mypid_);
    }

    //*******************************************************************
    // error checking
    //-------------------------------------------------------------------

    if (slideReduction_  == 1) 
    {
       nrows = localEndRow_ - localStartRow_ + 1 - 2 * nConstraints_;
       int_array = new int[numProcs_];
       gint_array = new int[numProcs_];
       for ( i = 0; i < numProcs_; i++ ) int_array[i] = 0;
       int_array[mypid_] = 2 * nConstraints_;
       MPI_Allreduce(int_array,gint_array,numProcs_,MPI_INT,MPI_SUM,comm_);
       startRow = 0;
       for ( i = 0; i < mypid_; i++ ) startRow += gint_array[i];
       startRow = localStartRow_ - 1 - startRow;
       endRow   = startRow + nrows;
       delete [] int_array;
       delete [] gint_array;
    }
    else if (slideReduction_  == 2) 
    {
       nrows = localEndRow_ - localStartRow_ + 1 - nConstraints_;
       int_array = new int[numProcs_];
       gint_array = new int[numProcs_];
       for ( i = 0; i < numProcs_; i++ ) int_array[i] = 0;
       int_array[mypid_] = nConstraints_;
       MPI_Allreduce(int_array,gint_array,numProcs_,MPI_INT,MPI_SUM,comm_);
       startRow = 0;
       for ( i = 0; i < mypid_; i++ ) startRow += gint_array[i];
       startRow = localStartRow_ - 1 - startRow;
       endRow   = startRow + nrows;
       delete [] int_array;
       delete [] gint_array;
    }
    else if (schurReduction_ == 1) 
    {
       nrows = localEndRow_ - localStartRow_ + 1 - A21NRows_;
       int_array = new int[numProcs_];
       gint_array = new int[numProcs_];
       for ( i = 0; i < numProcs_; i++ ) int_array[i] = 0;
       int_array[mypid_] = nrows;
       MPI_Allreduce(int_array,gint_array,numProcs_,MPI_INT,MPI_SUM,comm_);
       startRow = 0;
       for ( i = 0; i < mypid_; i++ ) startRow += gint_array[i];
       endRow   = startRow + nrows;
       delete [] int_array;
       delete [] gint_array;
    }

    if (leng != nrows)
    {
       printf("%4d : HYPRE_LinSysCore::formResidual ERROR - leng != numLocalRows");
       printf("                        numLocalRows, leng = %d %d",nrows,leng);
       exit(1);
    }
    if ( ! systemAssembled_ )
    {
       printf("formResidual ERROR : system not yet assembled.\n");
       exit(1);
    }

    //*******************************************************************
    // fetch matrix and vector pointers
    //-------------------------------------------------------------------

    A_csr  = (HYPRE_ParCSRMatrix) HYPRE_IJMatrixGetLocalStorage(currA_);
    x_csr  = (HYPRE_ParVector)    HYPRE_IJVectorGetLocalStorage(currX_);
    b_csr  = (HYPRE_ParVector)    HYPRE_IJVectorGetLocalStorage(currB_);
    r_csr  = (HYPRE_ParVector)    HYPRE_IJVectorGetLocalStorage(currR_);

    //*******************************************************************
    // form residual vector
    //-------------------------------------------------------------------

    HYPRE_ParVectorCopy( b_csr, r_csr );
    HYPRE_ParCSRMatrixMatvec( -1.0, A_csr, x_csr, 1.0, r_csr );

    //*******************************************************************
    // fetch residual vector
    //-------------------------------------------------------------------

    for ( i = startRow; i < endRow; i++ )
    {
       index = i - startRow;
       HYPRE_IJVectorGetLocalComponents(currR_, 1, &i, NULL, &values[index]);
       eqnNumbers[index] = i + 1;
    }

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
    {
       printf("%4d : HYPRE_LinSysCore::leaving  formResidual.\n", mypid_);
    }
}

//***************************************************************************
// solve the linear system
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::launchSolver(int& solveStatus, int &iterations)
{
    int                i, j, num_iterations, status, *num_sweeps, *relax_type;
    int                ierr, localNRows, rowNum, index, x2NRows;
    int                startRow, *int_array, *gint_array, startRow2;
    int                rowSize, *colInd, nnz, nrows;
    double             rnorm, *relax_wt, ddata, *colVal, value;
    double             stime, etime, ptime, rtime1, rtime2;
    char               fname[40];
    FILE               *fp;
    HYPRE_ParCSRMatrix A_csr;
    HYPRE_ParVector    x_csr;
    HYPRE_ParVector    b_csr;
    HYPRE_ParVector    r_csr;

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
    {
       printf("%4d : HYPRE_LinSysCore::entering launchSolver.\n", mypid_);
    }

    //*******************************************************************
    // temporary kludge before FEI adds functions to address this
    //-------------------------------------------------------------------

    MPI_Barrier(MPI_COMM_WORLD);
    rtime1  = MPI_Wtime();

    if ( schurReduction_ == 1 )
    {
       buildSchurReducedSystem();
    }

    if ( schurReduction_ == 0 && slideReduction_ != 0 )
    {
       if ( constrList_ != NULL ) delete [] constrList_;
       constrList_ = NULL;
       if      ( slideReduction_ == 1 ) buildSlideReducedSystem();
       else if ( slideReduction_ == 2 ) buildSlideReducedSystem2();
    }
    MPI_Barrier(MPI_COMM_WORLD);
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
       MPI_Barrier(MPI_COMM_WORLD);
    }

    //*******************************************************************
    // choose PCG, GMRES or direct solver
    //-------------------------------------------------------------------

    MPI_Barrier(MPI_COMM_WORLD);
    status = 1;
    stime  = MPI_Wtime();

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
                  if ( pilutRowSize_ == 0 ) pilutRowSize_ = pilutMaxNnzPerRow_;
                  if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
                  {
                     printf("PILUT - row size = %d\n", pilutRowSize_);
                     printf("PILUT - drop tol = %e\n", pilutDropTol_);
                  }
                  HYPRE_ParCSRPilutSetFactorRowSize(HYPrecon_,pilutRowSize_);
                  HYPRE_ParCSRPilutSetDropTolerance(HYPrecon_,pilutDropTol_);
                  if ( HYPreconReuse_ == 1 )
                  {
                     HYPRE_ParCSRPCGSetPrecond(HYSolver_,
                                    HYPRE_ParCSRPilutSolve,
                                    HYPRE_DummyFunction,HYPrecon_);
                  }
                  else
                  {
                     HYPRE_ParCSRPCGSetPrecond(HYSolver_,
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
                     HYPRE_ParCSRPCGSetPrecond(HYSolver_,HYPRE_LSI_DDIlutSolve,
                                    HYPRE_DummyFunction, HYPrecon_);
                  }
                  else
                  {
                     HYPRE_ParCSRPCGSetPrecond(HYSolver_, HYPRE_LSI_DDIlutSolve,
                                    HYPRE_LSI_DDIlutSetup, HYPrecon_);
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
                     printf("AMG threshold    = %e\n", amgStrongThreshold_);
                     printf("AMG numsweeps    = %d\n", amgNumSweeps_[0]);
                     printf("AMG relax type   = %d\n", amgRelaxType_[0]);
                     printf("AMG relax weight = %e\n", amgRelaxWeight_[0]);
                  }
                  if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 && mypid_ == 0)
                  {
                     HYPRE_BoomerAMGSetIOutDat(HYPrecon_, 2);
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
          HYPRE_ParCSRPCGSetup(HYSolver_, A_csr, b_csr, x_csr);
          ptime  = MPI_Wtime();
          if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0 )
          {
             printf("***************************************************\n");
             HYPRE_ParCSRPCGSetLogging(HYSolver_, 1);
          }
          HYPRE_ParCSRPCGSolve(HYSolver_, A_csr, b_csr, x_csr);
          HYPRE_ParCSRPCGGetNumIterations(HYSolver_, &num_iterations);
          HYPRE_ParVectorCopy( b_csr, r_csr );
          HYPRE_ParCSRMatrixMatvec( -1.0, A_csr, x_csr, 1.0, r_csr );
          HYPRE_ParVectorInnerProd( r_csr, r_csr, &rnorm);
          rnorm = sqrt( rnorm );
          iterations = num_iterations;
          if ( num_iterations >= maxIterations_ ) status = 0;
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
                  if (pilutRowSize_ == 0) pilutRowSize_ = pilutMaxNnzPerRow_;
                  if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
                  {
                     printf("PILUT - row size = %d\n", pilutRowSize_);
                     printf("PILUT - drop tol = %e\n", pilutDropTol_);
                  }
                  HYPRE_ParCSRPilutSetFactorRowSize(HYPrecon_,pilutRowSize_);
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
                     printf("AMG threshold    = %e\n", amgStrongThreshold_);
                     printf("AMG numsweeps    = %d\n", amgNumSweeps_[0]);
                     printf("AMG relax type   = %d\n", amgRelaxType_[0]);
                     printf("AMG relax weight = %e\n", amgRelaxWeight_[0]);
                  }
                  if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 && mypid_ == 0)
                     HYPRE_BoomerAMGSetIOutDat(HYPrecon_, 2);

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
          if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0 )
          {
             printf("***************************************************\n");
             HYPRE_ParCSRGMRESSetLogging(HYSolver_, 1);
          }
          HYPRE_ParCSRGMRESSetup(HYSolver_, A_csr, b_csr, x_csr);
          ptime  = MPI_Wtime();
          HYPRE_ParCSRGMRESSolve(HYSolver_, A_csr, b_csr, x_csr);
          HYPRE_ParCSRGMRESGetNumIterations(HYSolver_, &num_iterations);
          HYPRE_ParVectorCopy( b_csr, r_csr );
          HYPRE_ParCSRMatrixMatvec( -1.0, A_csr, x_csr, 1.0, r_csr );
          HYPRE_ParVectorInnerProd( r_csr, r_csr, &rnorm);
          iterations = num_iterations;
          rnorm = sqrt( rnorm );
          if ( num_iterations >= maxIterations_ ) status = 0;
          break;

       //----------------------------------------------------------------
       // choose SuperLU (single processor) 
       //----------------------------------------------------------------

       case HYSUPERLU :

          if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0 )
             printf("%4d : launchSolver(SuperLU)\n",mypid_);
          solveUsingSuperLU(status);
          iterations = 1;
          //printf("SuperLU solver - return status = %d\n",status);
          break;

       //----------------------------------------------------------------
       // choose SuperLU (single processor) 
       //----------------------------------------------------------------

       case HYSUPERLUX :

          if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0 )
             printf("%4d : launchSolver(SuperLUX)\n",mypid_);
          solveUsingSuperLUX(status);
          iterations = 1;
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
          iterations = 1;
          //printf("Y12M solver - return status = %d\n",status);
          break;

#else
          printf("HYPRE_LinSysCore : Y12M not available. \n");
          exit(1);
          break; 
#endif
    }

    //*******************************************************************
    // register solver return information and print timing information
    //-------------------------------------------------------------------

    solveStatus = status;
    iterations = num_iterations;

    MPI_Barrier(MPI_COMM_WORLD);
    etime = MPI_Wtime();
    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0 )
    {
       printf("***************************************************\n");
       printf("*                Solver Statistics                *\n");
       printf("*-------------------------------------------------*\n");
       if ( schurReduction_ )
          printf("** HYPRE Schur reduction time      = %e\n",rtime2-rtime1);
       if ( slideReduction_ )
          printf("** HYPRE slide reduction time      = %e\n",rtime2-rtime1);
       printf("** HYPRE preconditioner setup time = %e\n", ptime - stime);
       printf("** HYPRE solution time             = %e\n", etime - ptime);
       printf("** HYPRE total time                = %e\n", etime - stime);
       printf("** HYPRE number of iterations      = %d\n", num_iterations);
       printf("** HYPRE final residual norm       = %e\n", rnorm);
       printf("***************************************************\n");
    }

    //*******************************************************************
    // recover solution for reduced system
    //-------------------------------------------------------------------

    if ( slideReduction_ == 1 )
    {
       buildSlideReducedSoln();
    }
    else if ( slideReduction_ == 2 )
    {
       buildSlideReducedSoln2();
    }
    else if ( schurReduction_ == 1 )
    {
       buildSchurReducedSoln();
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
       MPI_Barrier(MPI_COMM_WORLD);
       exit(0);
    }

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
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
       printf("solveUsingSuperLU ERROR - allocateMatrix not called.\n");
       status = -1;
       return;
    }
    if ( localStartRow_ != 1 )
    {
       printf("solveUsingSuperLU ERROR - row does not start at 1\n");
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

    nnz   = 0;
    for ( i = 0; i < nrows; i++ ) nnz += rowLengths_[i];

    new_ia = new int[nrows+1];
    new_ja = new int[nnz];
    new_a  = new double[nnz];
    A_csr  = (HYPRE_ParCSRMatrix) HYPRE_IJMatrixGetLocalStorage(currA_);

    nz_ptr = getMatrixCSR(currA_, nrows, nnz, new_ia, new_ja, new_a);

    nnz = nz_ptr;

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
          printf("HYPRE_LinSysCore::solveUsingSuperLU - FINAL NORM = %e.\n",rnorm);
    }

    //------------------------------------------------------------------
    // clean up 
    //------------------------------------------------------------------

    delete [] ind_array; 
    delete [] rhs; 
    delete [] perm_c; 
    delete [] perm_r; 
    free( new_ia ); 
    free( new_ja ); 
    free( new_a ); 
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
          printf("HYPRE_LinSysCore::solveUsingSuperLUX - FINAL NORM = %e.\n",rnorm);
    }

    //------------------------------------------------------------------
    // clean up 
    //------------------------------------------------------------------

    delete [] ind_array; 
    delete [] perm_c; 
    delete [] perm_r; 
    delete [] etree; 
    delete [] rhs; 
    free( new_ia );
    free( new_ja );
    free( new_a );
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
    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
    {
       printf("%4d : HYPRE_LinSysCore::loadConstraintNumbers - size = %d\n", 
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
       printf("%4d : HYPRE_LinSysCore::leaving  loadConstraintNumbers\n",
              mypid_);
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

void HYPRE_LinSysCore::getVersion(char **name)
{
    printf("HYPRE_LinsysCore : this function hasn't been implemented yet.\n");
    return;
}

//***************************************************************************
// create a node to equation map from the solution vector
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::createMapFromSoln()
{
    int    i, ierr, *equations, local_nrows;
    double *answers;

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
    {
       printf("%4d : HYPRE_LinSysCore::entering createMapSoln.\n",mypid_);
    }

    local_nrows = localEndRow_ - localStartRow_ + 1;
    equations   = new int[local_nrows];
    answers     = new double[local_nrows];
    node2EqnMap = new int[local_nrows];

    for (i = 0; i < local_nrows; i++) equations[i] = localStartRow_ + i - 1; 
    ierr = HYPRE_IJVectorGetLocalComponents(HYx_,local_nrows,equations,NULL,
                                            answers);
    assert(!ierr);
    delete [] equations;
    for (i = 0; i < local_nrows; i++) 
    {
       node2EqnMap[i] = (int) answers[i];
       if ( node2EqnMap[i] < localStartRow_-1 || node2EqnMap[i] > localEndRow_ )
          printf("%4d : createMapFromSoln WARNING : map index out of range\n",
                 mypid_);
    }

    if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
    {
       printf("%4d : HYPRE_LinSysCore::leaving  createMapFromSoln.\n",mypid_);
    }
}


