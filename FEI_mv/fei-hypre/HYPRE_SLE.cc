/*BHEADER**********************************************************************
 * (c) 1999   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

//---------------------------------------------------------------------------
// parcsr_matrix_vector.h is put here instead of in HYPRE_SLE.h because 
// it gives warning when compiling fei_proc.cc                  
//---------------------------------------------------------------------------

#include "HYPRE_SLE.h"
#include "parcsr_matrix_vector.h"

#ifdef SUPERLU
#include "dsp_defs.h"
#include "util.h"
#endif

//---------------------------------------------------------------------------
// These are external functions needed internally here
//---------------------------------------------------------------------------

extern "C" {
   int hypre_ParAMGBuildCoarseOperator( hypre_ParCSRMatrix *,
                                        hypre_ParCSRMatrix *,
                                        hypre_ParCSRMatrix *,
                                        hypre_ParCSRMatrix **);
   void qsort0(int *, int, int);
   void qsort1(int *, double *, int, int);
   int  hypre_BinarySearch(int *, int, int);
#ifdef Y12M
   void y12maf_(int*,int*,double*,int*,int*,int*,int*,double*,int*,int*,
                double*,int*,double*,int*);
#endif
}

//***************************************************************************
// constructor
//---------------------------------------------------------------------------

HYPRE_SLE::HYPRE_SLE(MPI_Comm PASSED_COMM_WORLD, int masterRank) : 
                                     BASE_SLE(PASSED_COMM_WORLD, masterRank)
{
    int i;

    //--------------------------------------------------
    // set communicator 
    //--------------------------------------------------

    comm = PASSED_COMM_WORLD;
    MPI_Comm_rank(comm, &my_pid);

#ifdef DEBUG
    printf("%4d : HYPRE_SLE::entering constructor.\n", my_pid);
#endif

    //--------------------------------------------------
    //default to one rhs vector (in parent class)
    //--------------------------------------------------

    numRHSs_ = 1;
    currentRHS_ = 0;

    //--------------------------------------------------
    // default method = pcg
    //--------------------------------------------------

    HYSolverName_ = new char[64];
    strcpy(HYSolverName_,"gmres");
    solverID_  = HYGMRES;
    pcg_solver = NULL;

    //--------------------------------------------------
    // default preconditioner = diagonal
    //--------------------------------------------------

    HYPrecondName_ = new char[64];
    strcpy(HYPrecondName_,"diagonal");
    preconID_    = HYDIAGONAL;
    pcg_precond  = NULL;

    //--------------------------------------------------
    // other parameters
    //--------------------------------------------------

    globalNumEqns_  = 0;      // total no. of equations
    StartRow_       = 0;      // local start row (1-based)
    EndRow_         = -1;     // local end row (1-based)

    HY_A            = NULL;   // A matrix
    HY_x            = NULL;   // solution vector
    HY_b            = NULL;   // right hand side vector
    HY_r            = NULL;   // residual vector

    colIndices      = NULL;   // store matrix info
    rowLengths      = NULL;   // store matrix info

    nConstr         = 0;      // information for slide surfaces
    nSlaves         = 0;
    slaveList       = NULL;

    max_iterations  = 1000;   // solver parameters and info
    final_res_norm  = 0.0;
    tolerance       = 1.0e-10;

    assemble_flag   = 0;      // store whether matrix has been assembled

    //--------------------------------------------------
    // parameters for controlling amg, pilut, and SuperLU
    //--------------------------------------------------

    amg_coarsen_type     = 0;      // default coarsening
    for ( i = 0; i < 3;  i++ ) amg_num_sweeps[i]    = 2;
    amg_num_sweeps[3] = 1;
    for ( i = 0; i < 3;  i++ ) amg_relax_type[i]    = 3;   // hybrid
    amg_relax_type[3] = 9;         // direct for the coarsest level
    for ( i = 0; i < 25; i++ ) amg_relax_weight[i]  = 0.0; // damping factor
    amg_strong_threshold = 0.25;
    pilut_row_size       = 0;      // how many nonzeros to keep in L and U
    pilut_drop_tol       = 0.0;
    pilut_max_nz_per_row = 0;
    parasails_nlevels    = 1;
    parasails_threshold  = 0.0;
    superlu_ordering     = 0;      // natural ordering in SuperLU
    superlu_scale[0]     = 'N';    // no scaling in SuperLUX
    krylov_dim           = 50;

#ifdef DEBUG
    printf("%4d : HYPRE_SLE::leaving constructor.\n", my_pid);
#endif
    return;
}

//***************************************************************************
// destruction
//---------------------------------------------------------------------------

HYPRE_SLE::~HYPRE_SLE()
{
    delete [] HYSolverName_;
    delete [] HYPrecondName_;
    deleteLinearAlgebraCore();

    int numRows = EndRow_ - StartRow_ + 1;
    if ( numRows > 0 && colIndices != NULL ) 
    {
       for ( int i = 0; i < numRows; i++ )
          if ( colIndices[i] != NULL ) delete [] colIndices;
       delete [] colIndices;
       if ( rowLengths != NULL ) delete [] rowLengths;
    }
    if ( slaveList != NULL ) delete [] slaveList;
    return;
}

//***************************************************************************
// this function takes parameters for setting internal things like solver
// and preconditioner choice, etc.
//---------------------------------------------------------------------------

void HYPRE_SLE::parameters(int numParams, char **paramStrings) 
{
#ifdef DEBUG
    printf("%4d : HYPRE_SLE::entering parameters function.\n",my_pid);
    if ( my_pid == 0 )
    {
       printf("HYPRE_SLE::parameters - numParams = %d\n", numParams);
       for ( int i = 0; i < numParams; i++ )
       {
          printf("           param %d = %s \n", i, paramStrings[i]);
       }
    }
#endif

    appendParamStrings(numParams, paramStrings);

    int    i, nsweeps, rtype;
    double weight;
    char   param[256], param2[80];

    //----------------------------------------------------------
    // which solver to pick : cg, gmres, superlu, superlux, y12m 
    //----------------------------------------------------------

    if ( getParam("solver",numParams,paramStrings,param) == 1)
       sscanf(param,"%s",HYSolverName_);

    //----------------------------------------------------------
    // for GMRES, the restart size
    //----------------------------------------------------------

    if ( getParam("gmres-dim",numParams,paramStrings,param) == 1)
    {
       sscanf(param,"%d", &krylov_dim);
       if ( krylov_dim < 1 ) krylov_dim = 50;
    }

    //----------------------------------------------------------
    // which preconditioner : diagonal, pilut, boomeramg, parasails
    //----------------------------------------------------------

    if ( getParam("preconditioner",numParams,paramStrings,param) == 1)
    {
       sscanf(param,"%s",HYPrecondName_);
       //selectPreconditioner(HYPrecondName_);
    }

    //----------------------------------------------------------
    // maximum number of iterations for pcg or gmres
    //----------------------------------------------------------

    if ( getParam("maxIterations",numParams,paramStrings,param) == 1)
       sscanf(param,"%d", &max_iterations);

    //----------------------------------------------------------
    // tolerance as termination criterion
    //----------------------------------------------------------

    if ( getParam("tolerance",numParams,paramStrings,param) == 1)
       sscanf(param,"%e", &tolerance);

    //----------------------------------------------------------
    // pilut preconditioner : max no. of nonzeros to keep per row
    //----------------------------------------------------------

    if ( getParam("pilut-row-size",numParams,paramStrings,param) == 1)
    {
       sscanf(param,"%d", &pilut_row_size);
       if ( pilut_row_size < 1 ) pilut_row_size = 50;
    }

    //----------------------------------------------------------
    // pilut preconditioner : threshold to drop small nonzeros
    //----------------------------------------------------------

    if ( getParam("pilut-drop-tol",numParams,paramStrings,param) == 1)
    {
       sscanf(param,"%e", &pilut_drop_tol);
       if ( pilut_drop_tol < 0.0 || pilut_drop_tol >= 1.0 ) 
       {
          pilut_drop_tol = 0.0;
          printf("HYPRE_SLE::parameters - invalid pilut drop tol => set to %e\n",
                                          pilut_drop_tol);
       }
    }

    //----------------------------------------------------------
    // superlu : ordering to use (natural, mmd)
    //----------------------------------------------------------

    if ( getParam("superlu-ordering",numParams,paramStrings,param) == 1)
    {
       sscanf(param,"%s", &param2);
       if      ( !strcmp(param2, "natural" ) ) superlu_ordering = 0;
       else if ( !strcmp(param2, "mmd") )      superlu_ordering = 2;
       else 
       {
          superlu_ordering = 0;
          printf("HYPRE_SLE::parameters - superlu ordering set to natural.\n");
       }
    }

    //----------------------------------------------------------
    // superlu : scaling none ('N') or both col/row ('B')
    //----------------------------------------------------------

    if ( getParam("superlu-scale",numParams,paramStrings,param) == 1)
    {
       sscanf(param,"%s", &param2);
       if      ( !strcmp(param2, "y" ) ) superlu_scale[0] = 'B';
       else                              superlu_scale[0] = 'N';
    }

    //----------------------------------------------------------
    // amg preconditoner : coarsen type (falgout, ruge, default)
    //----------------------------------------------------------

    if ( getParam("amg-coarsen-type",numParams,paramStrings,param) == 1)
    {
       sscanf(param,"%s", param2);
       if      ( !strcmp(param2, "falgout") ) amg_coarsen_type = 6;
       else if ( !strcmp(param2, "ruge")    ) amg_coarsen_type = 1;
       else                                   amg_coarsen_type = 0;
    }

    //----------------------------------------------------------
    // amg preconditoner : no of relaxation sweeps per level
    //----------------------------------------------------------

    if ( getParam("amg-num-sweeps",numParams,paramStrings,param) == 1)
    {
       sscanf(param,"%d", &nsweeps);
       if ( nsweeps < 1 ) nsweeps = 1; 
       for ( i = 0; i < 3; i++ ) amg_num_sweeps[i] = nsweeps;
    }

    //----------------------------------------------------------
    // amg preconditoner : which smoother to use
    //----------------------------------------------------------

    if ( getParam("amg-relax-type",numParams,paramStrings,param) == 1)
    {
       sscanf(param,"%s", param2);
       if      ( !strcmp(param2, "jacobi" ) ) rtype = 2;
       else if ( !strcmp(param2, "gs-slow") ) rtype = 1;
       else if ( !strcmp(param2, "gs-fast") ) rtype = 4;
       else if ( !strcmp(param2, "hybrid" ) ) rtype = 3;
       else if ( !strcmp(param2, "direct" ) ) rtype = 9;
       else 
       {
          rtype = 3;
          printf("HYPRE_SLE::parameters - invalid relax type => set to hybrid.\n");
       }
       for ( i = 0; i < 3; i++ ) amg_relax_type[i] = rtype;
    }

    //----------------------------------------------------------
    // amg preconditoner : damping factor for Jacobi smoother
    //----------------------------------------------------------

    if ( getParam("amg-relax-weight",numParams,paramStrings,param) == 1)
    {
       sscanf(param,"%e", &weight);
       if ( weight < 0.0 || weight > 1.0 ) 
       {
          weight = 0.5;
          printf("HYPRE_SLE::parameters - invalid relax weight => set to %e\n",
                                          amg_relax_weight);
       }
       for ( i = 0; i < 25; i++ ) amg_relax_weight[i] = weight;
    }

    //----------------------------------------------------------
    // amg preconditoner : threshold to determine strong coupling
    //----------------------------------------------------------

    if ( getParam("amg-strong-threshold",numParams,paramStrings,param) == 1)
    {
       sscanf(param,"%e", &amg_strong_threshold);
       if ( amg_strong_threshold < 0.0 || amg_strong_threshold > 1.0 ) 
       {
          amg_strong_threshold = 0.25;
          printf("HYPRE_SLE::parameters - invalid amg threshold => set to %e\n",
                                          amg_strong_threshold);
       }
    }

    //----------------------------------------------------------
    // parasails preconditoner : threshold ( >= 0.0 )
    //----------------------------------------------------------

    if ( getParam("parasails-threshold",numParams,paramStrings,param) == 1)
    {
       sscanf(param,"%e", &parasails_threshold);
       if ( parasails_threshold < 0.0 ) 
       {
          parasails_threshold = 0.0;
          printf("HYPRE_SLE::parameters - parasails threshold set to %e\n",
                                          parasails_threshold);
       }
    }

    //----------------------------------------------------------
    // parasails preconditoner : nlevels ( >= 1) 
    //----------------------------------------------------------

    if ( getParam("parasails-nlevels",numParams,paramStrings,param) == 1)
    {
       sscanf(param,"%d", &parasails_nlevels);
       if ( parasails_nlevels < 1 ) 
       {
          parasails_nlevels = 1;
          printf("HYPRE_SLE::parameters - parasails nlevels set to %d\n",
                                          parasails_nlevels);
       }
    }

    BASE_SLE::parameters(numParams, paramStrings); 

#ifdef DEBUG
    printf("%4d : HYPRE_SLE::leaving parameters function.\n", my_pid);
#endif

    return;
}

//***************************************************************************
// select which Krylov solver to use
//---------------------------------------------------------------------------

void HYPRE_SLE::selectSolver(char *name)
{

#ifdef DEBUG
    printf("%4d : HYPRE_SLE::entering selectSolver = %s.\n", my_pid, name);
#endif

    //--------------------------------------------------
    // if already been allocated, destroy it first
    //--------------------------------------------------

    if ( pcg_solver != NULL ) 
    {
       if ( solverID_ == HYPCG )   HYPRE_ParCSRPCGDestroy(pcg_solver);
       if ( solverID_ == HYGMRES ) HYPRE_ParCSRGMRESDestroy(pcg_solver);
    }

    //--------------------------------------------------
    // check for the validity of the solver name
    //--------------------------------------------------

    if ( !strcmp(name, "cg"  ) ) 
    {
       strcpy( HYSolverName_, name );
       solverID_ = HYPCG;
    } 
    else if ( !strcmp(name, "gmres") ) 
    {
       strcpy( HYSolverName_, name );
       solverID_ = HYGMRES;
    } 
    else if ( !strcmp(name, "superlu") ) 
    {
       strcpy( HYSolverName_, name );
       solverID_ = HYSUPERLU;
    } 
    else if ( !strcmp(name, "superlux") ) 
    {
       strcpy( HYSolverName_, name );
       solverID_ = HYSUPERLUX;
    } 
    else if ( !strcmp(name, "y12m") ) 
    {
       strcpy( HYSolverName_, name );
       solverID_ = HYY12M;
    } 
    else 
    {
       printf("HYPRE_SLE selectSolver : use default = gmres.\n");
       strcpy( HYSolverName_, "gmres" );
       solverID_ = HYGMRES;
    }
 
    //--------------------------------------------------
    // instantiate solver 
    //--------------------------------------------------

    switch ( solverID_ ) 
    {
       case HYPCG :
            HYPRE_ParCSRPCGCreate(comm, &pcg_solver);
            //HYPRE_ParCSRPCGSetTwoNorm(pcg_solver, 1);
            //HYPRE_ParCSRPCGSetRelChange(pcg_solver, 0);
            //HYPRE_ParCSRPCGSetLogging(pcg_solver, 1);
            break;

       case HYGMRES :
            HYPRE_ParCSRGMRESCreate(comm, &pcg_solver);
            //HYPRE_ParCSRGMRESSetLogging(pcg_solver, 1);
            break;
    }

#ifdef DEBUG
    printf("%4d : HYPRE_SLE::leaving selectSolver = %s.\n", my_pid, name);
#endif
    return;
}

//***************************************************************************
// select which preconditioner to use
//---------------------------------------------------------------------------

void HYPRE_SLE::selectPreconditioner(char *name)
{
    int ierr;

#ifdef DEBUG
    printf("%4d : HYPRE_SLE::entering selectPreconditioner = %s.\n", my_pid, name);
#endif

    //--------------------------------------------------
    // if already been allocated, destroy it first
    //--------------------------------------------------

    if ( pcg_precond != NULL ) 
    {
       if ( preconID_ == HYPILUT )
          HYPRE_ParCSRPilutDestroy( pcg_precond );

       else if ( preconID_ == HYPARASAILS )
          HYPRE_ParCSRParaSailsDestroy( pcg_precond );

       else if ( preconID_ == HYBOOMERAMG )
          HYPRE_ParAMGDestroy( pcg_precond );
    }

    //--------------------------------------------------
    // check for the validity of the preconditioner name
    //--------------------------------------------------

    if ( !strcmp(name, "diagonal"  ) ) 
    {
       strcpy( HYPrecondName_, name );
       preconID_ = HYDIAGONAL;
    } 
    else if ( !strcmp(name, "pilut") ) 
    {
       strcpy( HYPrecondName_, name );
       preconID_ = HYPILUT;
    } 
    else if ( !strcmp(name, "parasails") ) 
    {
       strcpy( HYPrecondName_, name );
       preconID_ = HYPARASAILS;
    } 
    else if ( !strcmp(name, "boomeramg") ) 
    {
       strcpy( HYPrecondName_, name );
       preconID_ = HYPARASAILS;
    } 
    else 
    {
       if ( solverID_ != HYSUPERLU )
       {
          printf("HYPRE_SLE selectPreconditioner error : invalid solver.\n");
          printf("                               use default = diagonal.\n");
       }
       strcpy( HYPrecondName_, "diagonal" );
       preconID_ = HYDIAGONAL;
    }
    if ( solverID_ != HYPCG && solverID_ != HYGMRES ) preconID_ = HYNONE;

    //--------------------------------------------------
    // instantiate preconditioner 
    //--------------------------------------------------

    switch ( preconID_ ) 
    {
       case HYDIAGONAL :
            pcg_precond = NULL;
            break;

       case HYPILUT :
            ierr = HYPRE_ParCSRPilutCreate( comm, &pcg_precond );
            assert( !ierr );
            HYPRE_ParCSRPilutSetMaxIter( pcg_precond, 10 );
            break;

       case HYPARASAILS :
            ierr = HYPRE_ParCSRParaSailsCreate( comm, &pcg_precond );
            assert( !ierr );
            break;

       case HYBOOMERAMG :
            HYPRE_ParAMGCreate(&pcg_precond);
            HYPRE_ParAMGSetMaxIter(pcg_precond, 1);
            HYPRE_ParAMGSetCycleType(pcg_precond, 1);
            HYPRE_ParAMGSetMaxLevels(pcg_precond, 25);
            HYPRE_ParAMGSetMeasureType(pcg_precond, 0);
            break;
    }
#ifdef DEBUG
    printf("%4d : HYPRE_SLE::leaving selectPreconditioner.\n", my_pid);
#endif
}

//***************************************************************************
// This function is called by the constructor, just initializes
// the pointers and other variables associated with the linear
// algebra core to NULL or appropriate initial values.
//---------------------------------------------------------------------------

void HYPRE_SLE::initLinearAlgebraCore()
{
    HY_A = (HYPRE_IJMatrix) NULL;
    HY_x = (HYPRE_IJVector) NULL;
    HY_b = (HYPRE_IJVector) NULL;
    HY_r = (HYPRE_IJVector) NULL;
    assemble_flag = 0;
}

//***************************************************************************
//This is a destructor-type function.
//This function deletes allocated memory associated with
//the linear algebra core objects/data structures.
//---------------------------------------------------------------------------

void HYPRE_SLE::deleteLinearAlgebraCore()
{
    if ( HY_A != NULL ) HYPRE_IJMatrixDestroy(HY_A);
    if ( HY_x != NULL ) HYPRE_IJVectorDestroy(HY_x);
    if ( HY_b != NULL ) HYPRE_IJVectorDestroy(HY_b);
    if ( HY_r != NULL ) HYPRE_IJVectorDestroy(HY_r);
    HY_A = NULL;
    HY_x = NULL;
    HY_b = NULL;
    HY_r = NULL;
    assemble_flag = 0;
}

//***************************************************************************
//This function is where we establish the structures/objects associated
//with the linear algebra library. i.e., do initial allocations, etc.
// Rows and columns are 1-based.
//---------------------------------------------------------------------------

void HYPRE_SLE::createLinearAlgebraCore(int globalNumEqns,
  int localStartRow, int localEndRow, int localStartCol, int localEndCol)
{
    int ierr;

#ifdef DEBUG
    printf("%4d : HYPRE_SLE::entering createLinearAlgebraCore.\n", my_pid);
    printf("%4d : HYPRE_SLE::startrow, endrow = %d %d\n",my_pid,
                                      localStartRow,localEndRow);
    printf("%4d : HYPRE_SLE::startcol, endcol = %d %d\n",my_pid,
                                      localStartCol,localEndCol);
#endif

    //--------------------------------------------------
    // error checking
    //--------------------------------------------------

    if ( localEndRow < localStartRow ) 
    {
       printf("HYPRE_SLE createLinearAlgebraCore : invalid row indices.\n");
       printf("          startrow, endrow        = %d %d\n",localStartRow,
                         localEndRow);
       exit(1);
    }
    if ( localEndCol < localStartCol ) 
    {
       printf("HYPRE_SLE createLinearAlgebraCore : invalid column indices.\n");
       printf("          startcol, endcol        = %d %d\n",localStartCol,
                         localEndCol);
       exit(1);
    }

    StartRow_      = localStartRow;
    EndRow_        = localEndRow;
    globalNumEqns_ = globalNumEqns;

    //--------------------------------------------------
    // instantiate the matrix
    //--------------------------------------------------

    ierr = HYPRE_IJMatrixCreate(comm, &HY_A, globalNumEqns, globalNumEqns);
    assert(!ierr);
    ierr = HYPRE_IJMatrixSetLocalStorageType(HY_A, HYPRE_PARCSR);
    assert(!ierr);
    ierr = HYPRE_IJMatrixSetLocalSize(HY_A, localEndRow-localStartRow+1, 
                                      localEndCol-localStartCol+1);
    assert(!ierr);

    //--------------------------------------------------
    // instantiate the right hand vector
    //--------------------------------------------------

    ierr = HYPRE_IJVectorCreate(comm, &HY_b, globalNumEqns);
    assert(!ierr);
    ierr = HYPRE_IJVectorSetLocalStorageType(HY_b, HYPRE_PARCSR);
    assert(!ierr);
    ierr = HYPRE_IJVectorSetLocalPartitioning(HY_b,localStartRow-1,localEndRow);
    assert(!ierr);
    ierr = HYPRE_IJVectorAssemble(HY_b);
    assert(!ierr);
    ierr = HYPRE_IJVectorInitialize(HY_b);
    assert(!ierr);
    ierr = HYPRE_IJVectorZeroLocalComponents(HY_b);
    assert(!ierr);

    //--------------------------------------------------
    // instantiate the solution vector
    //--------------------------------------------------

    ierr = HYPRE_IJVectorCreate(comm, &HY_x, globalNumEqns);
    assert(!ierr);
    ierr = HYPRE_IJVectorSetLocalStorageType(HY_x, HYPRE_PARCSR);
    assert(!ierr);
    ierr = HYPRE_IJVectorSetLocalPartitioning(HY_x,localStartRow-1,localEndRow);
    assert(!ierr);
    ierr = HYPRE_IJVectorAssemble(HY_x);
    assert(!ierr);
    ierr = HYPRE_IJVectorInitialize(HY_x);
    assert(!ierr);
    ierr = HYPRE_IJVectorZeroLocalComponents(HY_x);
    assert(!ierr);

    //--------------------------------------------------
    // instantiate the residual vector
    //--------------------------------------------------

    ierr = HYPRE_IJVectorCreate(comm, &HY_r, globalNumEqns);
    assert(!ierr);
    ierr = HYPRE_IJVectorSetLocalStorageType(HY_r, HYPRE_PARCSR);
    assert(!ierr);
    ierr = HYPRE_IJVectorSetLocalPartitioning(HY_r,localStartRow-1,localEndRow);
    assert(!ierr);
    ierr = HYPRE_IJVectorAssemble(HY_r);
    assert(!ierr);
    ierr = HYPRE_IJVectorInitialize(HY_r);
    assert(!ierr);
    ierr = HYPRE_IJVectorZeroLocalComponents(HY_r);
    assert(!ierr);

#ifdef DEBUG
    printf("%4d : HYPRE_SLE::leaving createLinearAlgebraCore.\n", my_pid);
#endif
}

//***************************************************************************
// Set the number of rows in the diagonal part and off diagonal part
// of the matrix, using the structure of the matrix, stored in rows.
// rows is an array that is 0-based.  localStartRow and localEndRow are 1-based.
//---------------------------------------------------------------------------

void HYPRE_SLE::matrixConfigure(IntArray* rows)
{

    int i, j, ierr, nsize, *indices, maxSize, minSize;

#ifdef DEBUG
    printf("%4d : HYPRE_SLE::entering matrixConfigure.\n", my_pid);
#endif

    //--------------------------------------------------
    // error checking
    //--------------------------------------------------

    if ( EndRow_ < StartRow_ ) {
       printf("HYPRE_SLE matrixConfigure : createLinearAlgebraCore should\n");
       printf("                            be called before this.\n");
       exit(1);
    }

    nsize      = EndRow_ - StartRow_ + 1;
    rowLengths = new int[nsize];
    colIndices = new int*[nsize];

    //--------------------------------------------------
    // store the column index information
    //--------------------------------------------------

    maxSize = 0;
    minSize = 1000000;
    for ( i = 0; i < nsize; i++ ) 
    {
       rowLengths[i] = rows[i].size();
       indices = &((rows[i])[0]);

       if ( rowLengths[i] > 0 ) colIndices[i] = new int[rowLengths[i]];
       else                     colIndices[i] = NULL;
       for ( j = 0; j < rowLengths[i]; j++ )
       {
          colIndices[i][j] = indices[j];
       }
       maxSize = ( rowLengths[i] > maxSize ) ? rowLengths[i] : maxSize;
       minSize = ( rowLengths[i] < minSize ) ? rowLengths[i] : minSize;
    }

#ifdef DEBUG
    printf("%4d : HYPRE_SLE matrixConfigure : max/min nnz/row = %d %d\n",
                                              my_pid, maxSize, minSize);
#endif
    MPI_Allreduce(&maxSize, &pilut_max_nz_per_row, 1, MPI_INT, MPI_MAX, comm);

    ierr = HYPRE_IJMatrixSetRowSizes(HY_A, rowLengths);
    assert(!ierr);

    ierr = HYPRE_IJMatrixInitialize(HY_A);
    assert(!ierr);

#ifdef DEBUG
    printf("%4d : HYPRE_SLE::leaving matrixConfigure.\n", my_pid);
#endif
}

//***************************************************************************
// This function is needed in order to construct a new problem with the 
// same sparsity pattern.
//---------------------------------------------------------------------------

void HYPRE_SLE::resetMatrixAndVector(double s)
{
    int  ierr, size;

#ifdef DEBUG
    printf("%d : HYPRE_SLE::entering resetMatrixAndVector.\n", my_pid);
#endif

    if ( s != 0.0 )
    {
       printf("%d : HYPRE_SLE::resetMatrixAndVector - cannot take nonzeros.\n",my_pid);
       exit(1);
    }

    HYPRE_IJVectorZeroLocalComponents(HY_b);
    assemble_flag = 0;

    //--------------------------------------------------
    // for now, since HYPRE does not yet support 
    // re-initializing the matrix, restart the whole thing
    //--------------------------------------------------
 
    if ( HY_A != NULL ) HYPRE_IJMatrixDestroy(HY_A);
    ierr = HYPRE_IJMatrixCreate(comm, &HY_A, globalNumEqns_, globalNumEqns_);
    assert(!ierr);
    ierr = HYPRE_IJMatrixSetLocalStorageType(HY_A, HYPRE_PARCSR);
    assert(!ierr);
    size = EndRow_ - StartRow_ + 1;
    ierr = HYPRE_IJMatrixSetLocalSize(HY_A, size, size);
    assert(!ierr);
    ierr = HYPRE_IJMatrixSetRowSizes(HY_A, rowLengths);
    assert(!ierr);
    ierr = HYPRE_IJMatrixInitialize(HY_A);
    assert(!ierr);

#ifdef DEBUG
    printf("%4d : HYPRE_SLE::leaving resetMatrixAndVector.\n", my_pid);
#endif
}

//***************************************************************************
// input is 1-based, but HYPRE vectors are 0-based
//---------------------------------------------------------------------------

void HYPRE_SLE::sumIntoRHSVector(int num, const int* indices, 
  const double* values)
{
    int    i, ierr, *local_ind;

#ifdef DEBUG
#ifdef DEBUG_LEVEL2
    printf("%d : HYPRE_SLE::entering sumIntoRHSVector.\n", my_pid);
    for ( i = 0; i < num; i++ )
    {
       printf("%d : HYPRE_SLE::sumIntoRHSVector - %d = %e.\n", 
                               my_pid, indices[i], values[i]);
    }
#endif
#endif

    local_ind = new int[num];
    for ( i = 0; i < num; i++ ) // change to 0-based
    {
       if ( indices[i] > 0 && indices[i] <= globalNumEqns_ )
          local_ind[i] = indices[i] - 1; 
       else
       {
          printf("%d : HYPRE_SLE::sumIntoRHSVector - index out of range = %d.\n", 
                                  my_pid, indices[i]);
          exit(1);
       }
    }

    ierr = HYPRE_IJVectorAddToLocalComponents(HY_b,num,local_ind,NULL,values);
    assert(!ierr);

    delete [] local_ind;

#ifdef DEBUG
#ifdef DEBUG_LEVEL2
    printf("%4d : HYPRE_SLE::leaving sumIntoRHSVector.\n", my_pid);
#endif
#endif
}

//***************************************************************************
// used for initializing the initial guess
//---------------------------------------------------------------------------

void HYPRE_SLE::putIntoSolnVector(int num, const int* indices,
  const double* values)
{
    int i, ierr, *local_ind;

    local_ind = new int[num];
    for ( i = 0; i < num; i++ ) // change to 0-based
    {
       if ( indices[i] > 0 && indices[i] <= globalNumEqns_ )
          local_ind[i] = indices[i] - 1; 
       else
       {
          printf("%d : HYPRE_SLE::putIntoSolnVector - index out of range = %d.\n", 
                                  my_pid, indices[i]);
          exit(1);
       }
    }

    ierr = HYPRE_IJVectorSetLocalComponents(HY_x,num,local_ind,NULL,values);
    assert(!ierr);

    delete [] local_ind;
}

//***************************************************************************
// used for getting the solution out of the solver, and into the application
//---------------------------------------------------------------------------

double HYPRE_SLE::accessSolnVector(int equation)
{
    double val;
    int eqnNumber, ierr;

    eqnNumber = equation - 1; // construct 0-based index
    if ( equation <= 0 || equation > globalNumEqns_ )
    {
       printf("%d : HYPRE_SLE::accessSolnVector - index out of range = %d.\n", 
                               my_pid, equation);
       exit(1);
    }

    ierr = HYPRE_IJVectorGetLocalComponents(HY_x, 1, &eqnNumber, NULL, &val);
    assert(!ierr);

    return val;
}

//***************************************************************************
// add nonzero entries into the matrix data structure
//---------------------------------------------------------------------------

void HYPRE_SLE::sumIntoSystemMatrix(int row, int numValues, 
                  const double* values, const int* scatterIndices)
{
    int i, ierr, *local_ind;

#ifdef DEBUG
#ifdef DEBUG_LEVEL2
    printf("%4d : HYPRE_SLE::entering sumIntoSystemMatrix.\n", my_pid);
    printf("%4d : HYPRE_SLE::row number = %d.\n", my_pid, row);
    for ( i = 0; i < numValues; i++ ) 
       printf("%4d : col = %d, data = %e\n", scatterIndices[i], values[i]);
#endif
#endif

    //--------------------------------------------------
    // error checking
    //--------------------------------------------------

    if ( assemble_flag == 1 )
    {
       printf("HYPRE_SLE::sumIntoSystemMatrix error : matrix assembled already.\n");
       exit(1);
    }
    if ( row < StartRow_ || row > EndRow_ ) 
    {
       printf("HYPRE_SLE::sumIntoSystemMatrix error : invalid row number.\n");
       exit(1);
    }
    if ( numValues > rowLengths[row-StartRow_] )
    {
       printf("HYPRE_SLE::sumIntoSystemMatrix error : row size too large.\n");
       exit(1);
    }

    //--------------------------------------------------
    // load the matrix
    //--------------------------------------------------

    local_ind = new int[numValues];

    for ( i = 0; i < numValues; i++ ) // change indices to 0-based
    {
       if ( scatterIndices[i] > 0 && scatterIndices[i] <= globalNumEqns_ )
          local_ind[i] = scatterIndices[i] - 1;
       else
       {
          printf("%d : HYPRE_SLE::sumIntoSystemMatrix - index out of range = %d.\n", 
                                  my_pid, scatterIndices[i]);
          exit(1);
       }
    }

    ierr = HYPRE_IJMatrixAddToRow(HY_A, numValues, row-1, local_ind, values);
    assert(!ierr);

    delete [] local_ind;

#ifdef DEBUG
#ifdef DEBUG_LEVEL2
    printf("%4d : HYPRE_SLE::leaving sumIntoSystemMatrix.\n", my_pid);
#endif
#endif
}

//***************************************************************************
//This function must enforce an essential boundary condition on
//the equations in 'globalEqn'. This means, that the following modification
//should be made to A and b, for each globalEqn:
//
//for(all local equations i){
//   if (i==globalEqn) b[i] = gamma/alpha;
//   else b[i] -= (gamma/alpha)*A[i,globalEqn];
//}
//all of row 'globalEqn' and column 'globalEqn' in A should be zeroed,
//except for 1.0 on the diagonal.
//---------------------------------------------------------------------------

void HYPRE_SLE::enforceEssentialBC(int* globalEqn, double* alpha,
                                  double* gamma, int leng) 
{
    int       i, j, localEqn, row_size, indices[100], *indices_temp, *tempEqn;
    double    values[100], *values_temp, *local_gamma;

#ifdef DEBUG
#ifdef DEBUG_LEVEL2
    printf("%4d : HYPRE_SLE::entering enforceEssentialBC.\n", my_pid);
#endif
#endif

    //--------------------------------------------------
    // error checking
    //--------------------------------------------------

    if ( assemble_flag == 1 )
    {
       printf("HYPRE_SLE::enforceEssentialBC error : matrix assembled already.\n");
       exit(1);
    }

    //--------------------------------------------------
    // modify gamma for rhs (change indices to 0-based)
    //--------------------------------------------------

    tempEqn = new int[leng];
    local_gamma = new double[leng];
    for ( i = 0; i < leng; i++ )
    {
       if ( globalEqn[i] > 0 && globalEqn[i] <= globalNumEqns_ )
          tempEqn[i] = globalEqn[i] - 1;
       else
       {
          printf("%d : HYPRE_SLE::enforceEssentialBC - index out of range = %d.\n", 
                                  my_pid, globalEqn[i]);
          exit(1);
       }
       local_gamma[i] = gamma[i] / alpha[i];
    }

    //--------------------------------------------------
    // To preserve symmetry, should zero out column globalEqn[i].
    // This requires getting row globalEqn[i], looping over
    // its columns, and making modifications in the cooresponding
    // rows. This requires use of the GetRow function. Also note
    // that some rows requiring modification will be on different
    // processors than globalEqn[i].
    //
    // For the moment, we only modify the row globalEqn[i].
    //--------------------------------------------------

    for ( i = 0; i < leng; i++ )
    {
       if (globalEqn[i] >= StartRow_ && globalEqn[i] <= EndRow_)
       {
          localEqn = globalEqn[i] - StartRow_;
          row_size = rowLengths[localEqn];

          // If row_size is larger than 100 (the allocated size of
          // values and indices), allocate temporary space.
          if (row_size > 100)
          {
             values_temp = new double[row_size];
             indices_temp = new int[row_size];
          }
          else
          {
             values_temp = values;
             indices_temp = indices;
          }

          // Set up identity row.
          for (j = 0; j < row_size; j++)
          {
             indices_temp[j] = colIndices[localEqn][j] - 1;

             if (indices_temp[j] == tempEqn[i])
             {
                values_temp[j]=1.0;
             }
             else
             {
                values_temp[j]=0.0;
             }
          }

          // Set row for boundary point to identity
          HYPRE_IJMatrixInsertRow( HY_A,  row_size, tempEqn[i], 
                                   indices_temp, values_temp);

          // Free temporary space
          if (row_size > 100)
          {
             delete []indices_temp;
             delete []values_temp;
          }

          // Set rhs for boundary point
          printf("Add to rhs %d = %e\n", tempEqn[i]+1, local_gamma[i]);
          HYPRE_IJVectorSetLocalComponents(HY_b, 1, &tempEqn[i],
                                           NULL, &local_gamma[i]);

       }
    }

    delete [] local_gamma;
    delete [] tempEqn;

#ifdef DEBUG
#ifdef DEBUG_LEVEL2
    printf("%4d : HYPRE_SLE::leaving enforceEssentialBC.\n", my_pid);
#endif
#endif
}

//***************************************************************************
//This function must enforce a natural or mixed boundary condition
//on equation 'globalEqn'. This means that the following modification should
//be made to A and b:
//
//A[globalEqn,globalEqn] += alpha/beta;
//b[globalEqn] += gamma/beta;
//
//Currently loops over boundary points and uses HYPRE_AddIJMatrixRow to
//modify the diagonal. Uses a single call to  HYPRE_AddToIJVectorLocalComponents
//to modify right hand side. 
//---------------------------------------------------------------------------

void HYPRE_SLE::enforceOtherBC(int* globalEqn, double* alpha, double* beta,
                              double* gamma, int leng)
{
    int    i, *tempEqn;
    double value, *local_gamma;

#ifdef DEBUG
#ifdef DEBUG_LEVEL2
    printf("%4d : HYPRE_SLE::entering enforceOtherBC.\n", my_pid);
#endif
#endif

    //--------------------------------------------------
    // error checking
    //--------------------------------------------------

    if ( assemble_flag == 1 )
    {
       printf("HYPRE_SLE::enforceOtherBC error : matrix assembled already.\n");
       exit(1);
    }

    //--------------------------------------------------
    // modify gamma for rhs (change indices to 0-based)
    //--------------------------------------------------

    tempEqn = new int[leng];
    local_gamma = new double[leng];

    for ( i = 0; i < leng; i++ )
    {
       if ( globalEqn[i] > 0 && globalEqn[i] <= globalNumEqns_ )
          tempEqn[i] = globalEqn[i] - 1;
       else
       {
          printf("%d : HYPRE_SLE::enforceOtherBC - index out of range = %d.\n", 
                                  my_pid, globalEqn[i]);
          exit(1);
       }
       local_gamma[i] = gamma[i] / beta[i];
    }

    //--------------------------------------------------
    // modify matrix and rhs
    //--------------------------------------------------

    for ( i = 0; i < leng; i++ )
    {
       if (globalEqn[i] >= StartRow_ && globalEqn[i] <= EndRow_)
       {
          value = alpha[i]/beta[i];
          HYPRE_IJMatrixAddToRow(HY_A, 1, tempEqn[i], &tempEqn[i], &value);
          HYPRE_IJVectorAddToLocalComponents(HY_b,1,&tempEqn[i],NULL,&local_gamma[i]);
       }
    }

    delete [] tempEqn;
    delete [] local_gamma;

#ifdef DEBUG
#ifdef DEBUG_LEVEL2
    printf("%4d : HYPRE_SLE::leaving enforceOtherBC.\n", my_pid);
#endif
#endif
}

//***************************************************************************
// start assembling the matrix into its internal format
//---------------------------------------------------------------------------

void HYPRE_SLE::matrixLoadComplete()
{
#ifdef DEBUG
    printf("%4d : HYPRE_SLE::entering matrixLoadComplete.\n", my_pid);
#endif

    HYPRE_IJMatrixAssemble(HY_A);
    assemble_flag = 1;

#if PRINTMAT
    HYPRE_ParCSRMatrix a = (HYPRE_ParCSRMatrix) 
        HYPRE_IJMatrixGetLocalStorage(HY_A);

    HYPRE_ParCSRMatrixPrint(a, "driver.out.a");
    exit(0);
#endif

#ifdef DEBUG
    printf("%4d : HYPRE_SLE::leaving matrixLoadComplete.\n", my_pid);
#endif
}

//***************************************************************************
// solve the linear system
//---------------------------------------------------------------------------

void HYPRE_SLE::launchSolver(int* solveStatus)
{
    int                i, num_iterations, status, *num_sweeps, *relax_type;
    double             rnorm, *relax_wt;
    HYPRE_ParCSRMatrix A_csr;
    HYPRE_ParVector    x_csr;
    HYPRE_ParVector    b_csr;
    HYPRE_ParVector    r_csr;

#ifdef DEBUG
    printf("%4d : HYPRE_SLE::entering launchSolver.\n", my_pid);
#endif

    //--------------------------------------------------
    // fetch matrix and vector pointers
    //--------------------------------------------------

    A_csr  = (HYPRE_ParCSRMatrix) HYPRE_IJMatrixGetLocalStorage(HY_A);
    x_csr  = (HYPRE_ParVector)    HYPRE_IJVectorGetLocalStorage(HY_x);
    b_csr  = (HYPRE_ParVector)    HYPRE_IJVectorGetLocalStorage(HY_b);
    r_csr  = (HYPRE_ParVector)    HYPRE_IJVectorGetLocalStorage(HY_r);

    //--------------------------------------------------
    // choose PCG or GMRES
    //--------------------------------------------------

    status = 1;

    switch ( solverID_ ) 
    {

       case HYPCG :

#ifdef DEBUG
            printf("%4d : HYPRE_SLE : lauchSolver(PCG) - matsize = %d\n",
                                                         my_pid, globalNumEqns_);
#endif
            switch ( preconID_ ) 
            {
               case HYDIAGONAL :
                    HYPRE_ParCSRPCGSetPrecond(pcg_solver,
                                   HYPRE_ParCSRDiagScale,
                                   HYPRE_ParCSRDiagScaleSetup,
                                   pcg_precond);
                    break;

               case HYPILUT :
                    if ( pilut_row_size == 0 )
                    {
                       pilut_row_size = (int) (1.2 * pilut_max_nz_per_row);
#ifdef DEBUG
                       printf("HYPRE_SLE:: PILUT - row size = %d\n",pilut_row_size);
                       printf("HYPRE_SLE:: PILUT - drop tol = %e\n",pilut_drop_tol);
#endif
                    }
                    HYPRE_ParCSRPilutSetFactorRowSize( pcg_precond, pilut_row_size );
                    HYPRE_ParCSRPilutSetDropTolerance(pcg_precond,pilut_drop_tol);
                    HYPRE_ParCSRPCGSetPrecond(pcg_solver,
                                   HYPRE_ParCSRPilutSolve,
                                   HYPRE_ParCSRPilutSetup,
                                   pcg_precond);
                    break;

               case HYPARASAILS :
                    HYPRE_ParCSRParaSailsSetParams(pcg_precond, parasails_threshold,
                                                   parasails_nlevels);
                    HYPRE_ParCSRPCGSetPrecond(pcg_solver,
                                   HYPRE_ParCSRParaSailsSolve,
                                   HYPRE_ParCSRParaSailsSetup,
                                   pcg_precond);
                    break;

               case HYBOOMERAMG :
                    HYPRE_ParAMGSetCoarsenType(pcg_precond, amg_coarsen_type);
                    HYPRE_ParAMGSetStrongThreshold(pcg_precond, amg_strong_threshold);
                    num_sweeps = hypre_CTAlloc(int,4);
                    for ( i = 0; i < 4; i++ ) num_sweeps[i] = amg_num_sweeps[i]; 
                    HYPRE_ParAMGSetNumGridSweeps(pcg_precond, num_sweeps);
                    relax_type = hypre_CTAlloc(int,4);
                    for ( i = 0; i < 4; i++ ) relax_type[i] = amg_relax_type[i]; 
                    HYPRE_ParAMGSetGridRelaxType(pcg_precond, relax_type);
                    relax_wt = hypre_CTAlloc(double,25);
                    for ( i = 0; i < 25; i++ ) relax_wt[i] = amg_relax_weight[i]; 
                    HYPRE_ParAMGSetRelaxWeight(pcg_precond, relax_wt);
#ifdef DEBUG
                    if ( my_pid == 0 )
                    {
                       printf("HYPRE_SLE::AMG coarsen type = %d\n",amg_coarsen_type);
                       printf("HYPRE_SLE::AMG threshold    = %e\n",amg_strong_threshold);
                       printf("HYPRE_SLE::AMG numsweeps    = %d\n",amg_num_sweeps[0]);
                       printf("HYPRE_SLE::AMG relax type   = %d\n",amg_relax_type[0]);
                       printf("HYPRE_SLE::AMG relax weight = %e\n",amg_relax_weight[0]);
                    }
#endif
                    HYPRE_ParCSRPCGSetPrecond(pcg_solver,
                                   HYPRE_ParAMGSolve,
                                   HYPRE_ParAMGSetup,
                                   pcg_precond);
                    break;
            }
            HYPRE_ParCSRPCGSetMaxIter(pcg_solver, max_iterations);
            HYPRE_ParCSRPCGSetTol(pcg_solver, tolerance);
            HYPRE_ParCSRPCGSetup(pcg_solver, A_csr, b_csr, x_csr);
            HYPRE_ParCSRPCGSolve(pcg_solver, A_csr, b_csr, x_csr);
            HYPRE_ParCSRPCGGetNumIterations(pcg_solver, &num_iterations);
            HYPRE_ParVectorCopy( b_csr, r_csr );
            HYPRE_ParCSRMatrixMatvec( -1.0, A_csr, x_csr, 1.0, r_csr );
            HYPRE_ParVectorInnerProd( r_csr, r_csr, &rnorm);
            rnorm = sqrt( rnorm );
            iterations_ = num_iterations;
            if ( my_pid == 0 )
            {
               printf("HYPRE_SLE :launchSolver(PCG) - NO. ITERATION =    %d.\n", 
                                                      num_iterations);
               printf("HYPRE_SLE::launchSolver(PCG) - FINAL NORM    =    %e.\n", rnorm);
            }
            if ( num_iterations >= max_iterations ) status = 0;
            break;
               
       case HYGMRES :

#ifdef DEBUG
            printf("%4d : HYPRE_SLE : lauchSolver(GMRES) - matsize = %d\n",
                                                         my_pid, globalNumEqns_);
#endif

            switch ( preconID_ ) 
            {
               case HYDIAGONAL :
                    HYPRE_ParCSRGMRESSetPrecond(pcg_solver,
                                   HYPRE_ParCSRDiagScale,
                                   HYPRE_ParCSRDiagScaleSetup,
                                   pcg_precond);
                    break;

               case HYPILUT :
                    if ( pilut_row_size == 0 )
                    {
                       pilut_row_size = (int) (1.2 * pilut_max_nz_per_row);
#ifdef DEBUG
                       printf("HYPRE_SLE:: PILUT - row size = %d\n",pilut_row_size);
                       printf("HYPRE_SLE:: PILUT - drop tol = %e\n",pilut_drop_tol);
#endif
                    }
                    HYPRE_ParCSRPilutSetFactorRowSize( pcg_precond, pilut_row_size );
                    HYPRE_ParCSRPilutSetDropTolerance(pcg_precond,pilut_drop_tol);
                    HYPRE_ParCSRGMRESSetPrecond(pcg_solver,
                                   HYPRE_ParCSRPilutSolve,
                                   HYPRE_ParCSRPilutSetup,
                                   pcg_precond);
                    break;

               case HYPARASAILS :
                    HYPRE_ParCSRParaSailsSetParams(pcg_precond, parasails_threshold,
                                                   parasails_nlevels);
                    HYPRE_ParCSRGMRESSetPrecond(pcg_solver,
                                   HYPRE_ParCSRParaSailsSolve,
                                   HYPRE_ParCSRParaSailsSetup,
                                   pcg_precond);
                    break;

               case HYBOOMERAMG :
                    HYPRE_ParAMGSetCoarsenType(pcg_precond, amg_coarsen_type);
                    HYPRE_ParAMGSetStrongThreshold(pcg_precond, amg_strong_threshold);
                    num_sweeps = hypre_CTAlloc(int,4);
                    for ( i = 0; i < 4; i++ ) num_sweeps[i] = amg_num_sweeps[i]; 
                    HYPRE_ParAMGSetNumGridSweeps(pcg_precond, num_sweeps);
                    relax_type = hypre_CTAlloc(int,4);
                    for ( i = 0; i < 4; i++ ) relax_type[i] = amg_relax_type[i]; 
                    HYPRE_ParAMGSetGridRelaxType(pcg_precond, relax_type);
                    relax_wt = hypre_CTAlloc(double,25);
                    for ( i = 0; i < 25; i++ ) relax_wt[i] = amg_relax_weight[i]; 
                    HYPRE_ParAMGSetRelaxWeight(pcg_precond, relax_wt);
#ifdef DEBUG
                    if ( my_pid == 0 )
                    {
                       printf("HYPRE_SLE::AMG coarsen type = %d\n",amg_coarsen_type);
                       printf("HYPRE_SLE::AMG threshold    = %e\n",amg_strong_threshold);
                       printf("HYPRE_SLE::AMG numsweeps    = %d\n",amg_num_sweeps[0]);
                       printf("HYPRE_SLE::AMG relax type   = %d\n",amg_relax_type[0]);
                       printf("HYPRE_SLE::AMG relax weight = %e\n",amg_relax_weight[0]);
                    }
#endif
                    HYPRE_ParCSRGMRESSetPrecond(pcg_solver,
                                   HYPRE_ParAMGSolve,
                                   HYPRE_ParAMGSetup,
                                   pcg_precond);
                    break;
            }
            HYPRE_ParCSRGMRESSetKDim(pcg_solver, krylov_dim);
            HYPRE_ParCSRGMRESSetMaxIter(pcg_solver, max_iterations);
            HYPRE_ParCSRGMRESSetTol(pcg_solver, tolerance);
            HYPRE_ParCSRGMRESSetup(pcg_solver, A_csr, b_csr, x_csr);
            HYPRE_ParCSRGMRESSolve(pcg_solver, A_csr, b_csr, x_csr);
            HYPRE_ParCSRGMRESGetNumIterations(pcg_solver, &num_iterations);
            HYPRE_ParVectorCopy( b_csr, r_csr );
            HYPRE_ParCSRMatrixMatvec( -1.0, A_csr, x_csr, 1.0, r_csr );
            HYPRE_ParVectorInnerProd( r_csr, r_csr, &rnorm);
            iterations_ = num_iterations;
            rnorm = sqrt( rnorm );
            if ( my_pid == 0 )
            {
               printf("HYPRE_SLE :launchSolver(GMRES) - NO. ITERATION =    %d.\n", 
                                                        num_iterations);
               printf("HYPRE_SLE::launchSolver(GMRES) - FINAL NORM    =    %e.\n", 
                                                        rnorm);
            }
            if ( num_iterations >= max_iterations ) status = 0;
            break;

       case HYSUPERLU :

#ifdef DEBUG
            printf("%4d : HYPRE_SLE : launchSolver(SuperLU) - matsize=%d\n",
                                      my_pid, globalNumEqns_);
#endif
            solveUsingSuperLU(status);
            iterations_ = 0;
            //printf("HYPRE_SLE : SuperLU solver - return status = %d\n",status);
            break;

       case HYSUPERLUX :

#ifdef DEBUG
            printf("%4d : HYPRE_SLE : launchSolver(SuperLUX) - matsize=%d\n",
                                      my_pid, globalNumEqns_);
#endif
            solveUsingSuperLUX(status);
            iterations_ = 0;
            //printf("HYPRE_SLE : SuperLUX solver - return status = %d\n",status);
            break;

       case HYY12M :

#ifdef DEBUG
            printf("%4d : HYPRE_SLE : launchSolver(Y12M) - matsize=%d\n",
                                         my_pid, globalNumEqns_);
#endif
            solveUsingY12M(status);
            iterations_ = 0;
            //printf("HYPRE_SLE : Y12M solver - return status = %d\n",status);
            break;
    }

    *solveStatus = status; 

#ifdef DEBUG
    printf("%4d : HYPRE_SLE::leaving launchSolver.\n", my_pid);
#endif
}

//***************************************************************************
// reading a matrix from a file in ija format (first row : nrows, nnz)
// (read by a single processor)
//---------------------------------------------------------------------------

void ML_Get_IJAMatrixFromFile(double **val, int **ia, int **ja, int *N, 
                              double **rhs, char *matfile, char *rhsfile)
{
   int    i, j, Nrows, nnz, icount, rowindex, colindex, curr_row;
   int    k, m, *mat_ia, *mat_ja, ncnt, rnum;
   double dtemp, *mat_a, value, *rhs_local;
   FILE   *fp;

   /* ========================================= */
   /* read matrix file                          */
   /* ========================================= */

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
   mat_ia = (int *)    malloc((Nrows+1) * sizeof(int));
   mat_ja = (int *)    malloc(nnz * sizeof(int));
   mat_a  = (double *) malloc(nnz * sizeof(double));
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
      /*if ( value != 0.0 ) {*/
         mat_ja[icount] = colindex;
         mat_a[icount++]  = value;
      /*}*/
   }
   fclose(fp);
   for ( i = curr_row+1; i <= Nrows; i++ ) mat_ia[i] = icount;
   (*val) = mat_a;
   (*ia)  = mat_ia;
   (*ja)  = mat_ja;
   (*N) = Nrows;
   printf("matrix has %6d rows and %7d nonzeros\n", Nrows, mat_ia[Nrows]);

   /* ========================================= */
   /* read rhs file                             */
   /* ========================================= */

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
   rhs_local = (double *) malloc(Nrows * sizeof(double));
   m = 0;
   for ( k = 0; k < ncnt; k++ ) {
      fscanf(fp, "%d %lg", &rnum, &dtemp);
      rhs_local[rnum-1] = dtemp; m++;
   }
   fflush(stdout);
   ncnt = m;
   fclose(fp);
   (*rhs) = rhs_local;

   for ( i = 0; i < Nrows; i++ ) {
      for ( j = mat_ia[i]; j < mat_ia[i+1]; j++ )
         mat_ja[j]++;
   }
}

//******************************************************************************
// Given the matrix (A) within the object, compute the reduced system and put
// it in place.  Additional information given are :
//
// Additional assumptions are :
//
//    - a given slave equation and the corresponding constraint equation
//      reside in the same processor
//    - the ordering of the slave equations is the same as the ordering of
//      the constraint equations
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

void HYPRE_SLE::buildReducedSystem()
{
    int    j, num_procs, nRows, globalNRows, colIndex;
    int    globalNConstr, globalNSelected, *globalSelectedList;
    int    nSelected, *tempList, i, reducedAStartRow;
    int    searchIndex, procIndex, A2StartRow;
    int    rowSize, *colInd, searchCount, A21NRows, A21GlobalNRows;
    int    A21NCols, A21GlobalNCols, rowCount, maxRowSize, newEndRow;
    int    *A21MatSize, rowIndex;
    int    sumCount, *newColInd, diagCount, newRowSize, ierr;
    int    invA22NRows, invA22GlobalNRows, invA22NCols, invA22GlobalNCols;
    int    *invA22MatSize, newNRows, newGlobalNRows;
    int    *colInd2, *selectedList, ncnt, ubound;
    int    rowSize2, *recvCntArray, *displArray, zeroDiagFlag, ncnt2;
    int    StartRow, EndRow, *reducedAMatSize;
    int    *ProcNRows, *ProcNConstr;

    double searchValue, *colVal, *colVal2, *newColVal, *diagonal;
    double *extDiagonal;

    HYPRE_IJMatrix     A21, invA22, reducedA;
    HYPRE_ParCSRMatrix A_csr, A21_csr, invA22_csr, RAP_csr, reducedA_csr;

    //******************************************************************
    // initial set up 
    //------------------------------------------------------------------

    //------------------------------------------------------------------
    // get machine information
    //------------------------------------------------------------------

    MPI_Comm_size(comm, &num_procs);

    //------------------------------------------------------------------
    // offset the row ranges to be 0 based (instead of 1-based)
    //------------------------------------------------------------------

    StartRow = StartRow_ - 1;
    EndRow   = EndRow_ - 1;
    printf("buildReducedSystem %d : StartRow,EndRow = %d %d\n",my_pid,StartRow,EndRow);

    //------------------------------------------------------------------
    // get information about nRows from all processors
    //------------------------------------------------------------------
 
    nRows       = EndRow_ - StartRow_ + 1;
    ProcNRows   = new int[num_procs];
    tempList    = new int[num_procs];
    for ( i = 0; i < num_procs; i++ ) tempList[i] = 0;
    tempList[my_pid] = nRows;
    MPI_Allreduce(tempList, ProcNRows, num_procs, MPI_INT, MPI_SUM, comm );
    delete [] tempList;
    printf("buildReducedSystem %d : localNRows = %d\n", my_pid, nRows);

    //------------------------------------------------------------------
    // compute the base NRows on each processor
    // (This is needed later on for column index conversion)
    //------------------------------------------------------------------

    globalNRows = 0;
    ncnt = 0;
    for ( i = 0; i < num_procs; i++ ) 
    {
       globalNRows   += ProcNRows[i];
       ncnt2          = ProcNRows[i];
       ProcNRows[i]   = ncnt;
       ncnt          += ncnt2;
    }

    //------------------------------------------------------------------
    // get the CSR matrix for A
    //------------------------------------------------------------------

    A_csr = (HYPRE_ParCSRMatrix) HYPRE_IJMatrixGetLocalStorage(HY_A);

    //******************************************************************
    // search the entire local matrix to find where the constraint
    // equations are
    //------------------------------------------------------------------
    
    if ( nConstr == 0 )
    {
       for ( i = StartRow; i <= EndRow; i++ ) 
       {
          ierr = HYPRE_ParCSRMatrixGetRow(A_csr,i,&rowSize,&colInd,&colVal);
          assert(!ierr);
          for (j = 0; j < rowSize; j++) 
             if ( colInd[j] == i && colVal[j] != 0.0 ) break;
          ierr = HYPRE_ParCSRMatrixRestoreRow(A_csr,i,&rowSize,&colInd,&colVal);
          assert(!ierr);
          if ( j == rowSize ) break;
       }
       nConstr = EndRow - i + 1;
    }
    printf("buildReducedSystem %d : no. constraint equations = %d\n",my_pid,nConstr);

    //******************************************************************
    // search the entire local matrix to find where the constraint
    // equations are
    //------------------------------------------------------------------
    
    globalNConstr = 0;
    tempList    = new int[num_procs];
    ProcNConstr = new int[num_procs];
    for ( i = 0; i < num_procs; i++ ) tempList[i] = 0;
    tempList[my_pid] = nConstr;
    MPI_Allreduce(tempList, ProcNConstr, num_procs, MPI_INT, MPI_SUM, comm );
    delete [] tempList;

    //------------------------------------------------------------------
    // compute the base NConstr on each processor
    // (This is needed later on for column index conversion)
    //------------------------------------------------------------------

    ncnt = 0;
    for ( i = 0; i < num_procs; i++ ) 
    {
       globalNConstr += ProcNConstr[i];
       ncnt2          = ProcNConstr[i];
       ProcNConstr[i] = ncnt;
       ncnt          += ncnt2;
    }
   
    //------------------------------------------------------------------
    // compute the starting row number for the reduced matrix in my
    // processor 
    //------------------------------------------------------------------

    A2StartRow = 2 * ProcNConstr[my_pid];
    printf("buildReducedSystem %d : A2StartRow = %d\n", my_pid, A2StartRow);

    //------------------------------------------------------------------
    // allocate array for storing indices of selected nodes
    //------------------------------------------------------------------

    globalNSelected = globalNConstr;
    if ( globalNSelected > 0 ) globalSelectedList = new int[globalNSelected];
    else                       globalSelectedList = NULL;
    nSelected = nConstr;
    if ( nConstr > 0 ) selectedList = new int[nConstr];
    else               selectedList = NULL;
   
    //------------------------------------------------------------------
    // compose candidate slave list (if not given by loadSlaveList func)
    //------------------------------------------------------------------

    if ( nConstr > 0 && nSlaves == 0 )
    {
       slaveList = new int[EndRow-nConstr-StartRow+1];
       for ( i = StartRow; i <= EndRow-nConstr; i++ ) 
       {
          ierr = HYPRE_ParCSRMatrixGetRow(A_csr,i,&rowSize,&colInd,&colVal);
          assert(!ierr);
          ncnt = 0;
          for (j = 0; j < rowSize; j++) 
          {
             colIndex = colInd[j];
             for (procIndex=0; procIndex < num_procs; procIndex++ )
                if ( colIndex < ProcNRows[procIndex] ) break;
             if ( procIndex == num_procs ) 
                ubound = globalNRows - (globalNConstr - ProcNConstr[procIndex-1]); 
             else                          
                ubound = ProcNRows[procIndex] - (ProcNConstr[procIndex] - 
                                                 ProcNConstr[procIndex-1]); 
             if ( colIndex >= ubound && colVal[j] != 0.0 ) ncnt++;
             if ( ncnt > 1 ) break;
          }
          ierr = HYPRE_ParCSRMatrixRestoreRow(A_csr,i,&rowSize,&colInd,&colVal);
          assert(!ierr);
          if ( j == rowSize && ncnt == 1 ) slaveList[nSlaves++] = i;
       }

       //******************************************************************
       // search the constraint equations for the selected nodes
       //------------------------------------------------------------------
    
       nSelected = 0;

       for ( i = EndRow-nConstr+1; i <= EndRow; i++ ) 
       {
          ierr = HYPRE_ParCSRMatrixGetRow(A_csr,i,&rowSize,&colInd,&colVal);
          assert(!ierr);
          searchIndex = -1;
          searchValue = -1.0E10;
          searchCount = 0;
          for (j = 0; j < rowSize; j++) 
          {
             if (colVal[j] != 0.0 && colInd[j] >= StartRow 
                                  && colInd[j] <= (EndRow-nConstr)) 
             {
	        if ( hypre_BinarySearch(slaveList, colInd[j], nSlaves) >= 0 ) 
                {
                    if ( colVal[j] > searchValue ) 
                    {
                       searchValue = colVal[j];
                       searchIndex = colInd[j];
                    }
                    searchCount++;
                    if ( searchCount >= 3 ) break;
                }
             }
          } 
          if ( searchIndex >= 0 )
          {
             selectedList[nSelected++] = searchIndex;
          } else {
             printf("%d : buildReducedSystem::ERROR (1).\n", my_pid);
             exit(1);
          }
          ierr = HYPRE_ParCSRMatrixRestoreRow(A_csr,i,&rowSize,&colInd,&colVal);
          assert(!ierr);
       }
    }
    else
    {
       nSelected = nSlaves;
       for ( i = 0; i < nSlaves; i++ ) selectedList[i] = slaveList[i];
    }   

    //******************************************************************
    // form a global list of selected nodes on each processor
    // form the corresponding auxiliary list for later pruning
    //------------------------------------------------------------------

    recvCntArray = new int[num_procs];
    displArray   = new int[num_procs];
    MPI_Allgather(&nSelected, 1, MPI_INT, recvCntArray, 1, MPI_INT, comm);
    displArray[0] = 0;
    for ( i = 1; i < num_procs; i++ ) 
       displArray[i] = displArray[i-1] + recvCntArray[i-1];
    MPI_Allgatherv(selectedList, nSelected, MPI_INT, globalSelectedList,
                   recvCntArray, displArray, MPI_INT, comm);
    delete [] recvCntArray;
    delete [] displArray;

    qsort0(globalSelectedList, 0, globalNSelected-1);
    qsort0(selectedList, 0, nSelected-1);

    printf("buildReducedSystem %d : nSelected = %d\n", my_pid, nSelected);
    for ( i = 0; i < nSelected; i++ )
       printf("buildReducedSystem %d : selectedList %d = %d\n",my_pid,i,selectedList[i]);
 
    //******************************************************************
    // construct A21
    //------------------------------------------------------------------

    //------------------------------------------------------------------
    // calculate the dimension of A21
    //------------------------------------------------------------------

    A21NRows       = 2 * nConstr;
    A21GlobalNRows = 2 * globalNConstr;
    A21NCols       = nRows - 2 * nConstr;
    A21GlobalNCols = globalNRows - 2 * globalNConstr;
    printf("A21 dimensions = %d %d\n", A21GlobalNRows, A21GlobalNCols);
    printf("buildReducedSystem %d : A21 local dim = %d %d\n",my_pid,A21NRows,A21NCols);

    //------------------------------------------------------------------
    // create a matrix context for A21
    //------------------------------------------------------------------

    ierr = HYPRE_IJMatrixCreate(comm,&A21,A21GlobalNRows,A21GlobalNCols);
    assert(!ierr);
    ierr = HYPRE_IJMatrixSetLocalStorageType(A21, HYPRE_PARCSR);
    assert(!ierr);
    ierr = HYPRE_IJMatrixSetLocalSize(A21, A21NRows, A21NCols);
    assert(!ierr);

    //------------------------------------------------------------------
    // compute the number of nonzeros in the first nConstr row of A21
    // (which consists of the rows in selectedList), the nnz will
    // be reduced by excluding the constraint and selected slave columns
    //------------------------------------------------------------------

    rowCount   = 0;
    maxRowSize = 0;
    newEndRow  = EndRow - nConstr;
    A21MatSize = new int[A21NRows];

    for ( i = 0; i < nSelected; i++ ) 
    {
       rowIndex = selectedList[i];
       HYPRE_ParCSRMatrixGetRow(A_csr,rowIndex,&rowSize,&colInd,&colVal);
       rowSize2 = 0;
       for (j = 0; j < rowSize; j++) 
       {
          colIndex = colInd[j];
	  searchIndex = hypre_BinarySearch(globalSelectedList, colIndex, 
                                           globalNSelected);
          if (searchIndex < 0 && (colIndex <= newEndRow || colIndex > EndRow_)) 
             rowSize2++;
       }
       A21MatSize[rowCount] = rowSize2;
       maxRowSize = ( rowSize2 > maxRowSize ) ? rowSize2 : maxRowSize;
       HYPRE_ParCSRMatrixRestoreRow(A_csr,rowIndex,&rowSize,&colInd,&colVal);
       rowCount++;
    }

    //------------------------------------------------------------------
    // compute the number of nonzeros in the second nConstr row of A21
    // (which consists of the rows in constraint equations), the nnz will
    // be reduced by excluding the selected slave columns only (since the
    // entries corresponding to the constraint columns are 0, and since
    // the selected matrix is a diagonal matrix, there is no need to 
    // search for slave equations in the off-processor list)
    //------------------------------------------------------------------

    rowCount = nSelected;
    for ( i = EndRow-nConstr+1; i <= EndRow; i++ ) 
    {
       ierr = HYPRE_ParCSRMatrixGetRow(A_csr,i,&rowSize,&colInd,&colVal);
       assert(!ierr);
       rowSize2 = 0;
       for (j = 0; j < rowSize; j++) 
       {
          if ( colVal[j] != 0.0 )
          {
             colIndex = colInd[j];
	     searchIndex = hypre_BinarySearch(selectedList, colIndex, nSelected); 
             if ( searchIndex < 0 ) rowSize2++;
          }
       }
       A21MatSize[rowCount] = rowSize2;
       maxRowSize = ( rowSize2 > maxRowSize ) ? rowSize2 : maxRowSize;
       ierr = HYPRE_ParCSRMatrixRestoreRow(A_csr,i,&rowSize,&colInd,&colVal);
       assert(!ierr);
       rowCount++;
    }

    //------------------------------------------------------------------
    // after fetching the row sizes, set up A21 with such sizes
    //------------------------------------------------------------------

    ierr = HYPRE_IJMatrixSetRowSizes(A21, A21MatSize);
    assert(!ierr);
    ierr = HYPRE_IJMatrixInitialize(A21);
    assert(!ierr);
    delete [] A21MatSize;

    //------------------------------------------------------------------
    // next load the first nConstr row to A21 extracted from A
    // (at the same time, the D block is saved for future use)
    //------------------------------------------------------------------

    rowCount  = A2StartRow;
    if ( nConstr > 0 ) diagonal = new double[nConstr];
    else               diagonal = NULL;
    newColInd = new int[maxRowSize+1];
    newColVal = new double[maxRowSize+1];

    diagCount = 0;
    for ( i = 0; i < nSelected; i++ )
    {
       rowIndex = selectedList[i];
       HYPRE_ParCSRMatrixGetRow(A_csr,rowIndex,&rowSize,&colInd,&colVal);
       newRowSize = 0;
       for (j = 0; j < rowSize; j++) 
       {
          if ( colVal[j] != 0.0 )
          {
             colIndex = colInd[j];
             if (colIndex <= newEndRow || colIndex > EndRow_) 
             {
	        searchIndex = hypre_BinarySearch(globalSelectedList, colIndex, 
                                                 globalNSelected); 
                if ( searchIndex < 0 ) 
                {
                   searchIndex = - searchIndex - 1;
                   for ( procIndex = 0; procIndex < num_procs; procIndex++ )
                      if ( ProcNRows[procIndex] > colIndex ) break;
                   procIndex--;
                   colIndex = colInd[j] - ProcNConstr[procIndex] - searchIndex;
                   newColInd[newRowSize]   = colIndex;
                   newColVal[newRowSize++] = colVal[j];
                   if ( newRowSize > maxRowSize+1 ) 
                      printf("#### : passing array boundary.\n");
                }
             }
             else if ( colIndex > newEndRow && colIndex <= EndRow ) 
             {
                if ( colVal[j] != 0.0 ) diagonal[diagCount++] = colVal[j];
             }
          } 
       }

       ierr = HYPRE_IJMatrixInsertRow(A21,newRowSize,rowCount,newColInd,newColVal);
       assert(!ierr);
       HYPRE_ParCSRMatrixRestoreRow(A_csr,rowIndex,&rowSize,&colInd,&colVal);
       if ( diagCount != (i+1) )
       {
          printf("buildReducedSystem:: ERROR (3) - %d %d.\n",diagCount,i+1);
          exit(1);
       }
       rowCount++;
    }

    //------------------------------------------------------------------
    // send the diagonal to each processor that needs them
    //------------------------------------------------------------------

    recvCntArray = new int[num_procs];
    displArray   = new int[num_procs];
    MPI_Allgather(&diagCount, 1, MPI_INT, recvCntArray, 1, MPI_INT, comm);
    displArray[0] = 0;
    for ( i = 1; i < num_procs; i++ ) 
       displArray[i] = displArray[i-1] + recvCntArray[i-1];
    ncnt = displArray[num_procs-1] + recvCntArray[num_procs-1];
    if ( ncnt > 0 ) extDiagonal = new double[ncnt];
    else            extDiagonal = NULL;
    MPI_Allgatherv(diagonal, diagCount, MPI_DOUBLE, extDiagonal,
                   recvCntArray, displArray, MPI_DOUBLE, comm);
    diagCount = ncnt;
    delete [] recvCntArray;
    delete [] displArray;
    if ( diagonal != NULL ) delete [] diagonal;

    //------------------------------------------------------------------
    // next load the second nConstr rows to A21 extracted from A
    //------------------------------------------------------------------

    for ( i = EndRow-nConstr+1; i <= EndRow; i++ ) 
    {
       HYPRE_ParCSRMatrixGetRow(A_csr,i,&rowSize,&colInd,&colVal);
       newRowSize = 0;
       for (j = 0; j < rowSize; j++) 
       {
          colIndex    = colInd[j];
	  searchIndex = hypre_BinarySearch(globalSelectedList,colIndex,
                                           globalNSelected); 
          if ( searchIndex < 0 && colVal[j] != 0.0 ) 
          {
             searchIndex = - searchIndex - 1;
             for ( procIndex = 0; procIndex < num_procs; procIndex++ )
                if ( ProcNRows[procIndex] > colIndex ) break;
             procIndex--;
             colIndex = colInd[j] - ProcNConstr[procIndex] - searchIndex;
             newColInd[newRowSize]   = colIndex;
             newColVal[newRowSize++] = colVal[j];
             if ( newRowSize > maxRowSize+1 ) 
                printf("#### : passing array boundary.\n");
          } 
       }
       ierr = HYPRE_IJMatrixInsertRow(A21,newRowSize,rowCount,newColInd,newColVal);
       assert(!ierr);
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

    ncnt = 100;
    while ( ncnt < num_procs ) {
       if ( my_pid == ncnt ) {
          printf("buildReducedSystem %d : matrix A21 assembled %d.\n",my_pid,A2StartRow);
          fflush(stdout);
          for ( i = A2StartRow; i < A2StartRow+2*nConstr; i++ ) {
             ierr = HYPRE_ParCSRMatrixGetRow(A21_csr,i,&rowSize,&colInd,&colVal);
             printf("ROW = %6d (%d)\n", i, rowSize);
             for ( j = 0; j < rowSize; j++ )
                printf("   col = %6d, val = %e \n", colInd[j], colVal[j]);
             HYPRE_ParCSRMatrixRestoreRow(A21_csr,i,&rowSize,&colInd,&colVal);
          }
       }
       MPI_Barrier(MPI_COMM_WORLD);
       ncnt++;
    }

    //******************************************************************
    // construct invA22
    //------------------------------------------------------------------

    //------------------------------------------------------------------
    // calculate the dimension of invA22
    //------------------------------------------------------------------

    invA22NRows       = A21NRows;
    invA22GlobalNRows = A21GlobalNRows;
    invA22NCols       = invA22NRows;
    invA22GlobalNCols = invA22GlobalNRows;
    printf("buildReducedSystem %d : A22 dimensions = %d %d\n", my_pid, 
                     invA22GlobalNRows, invA22GlobalNCols);
    printf("buildReducedSystem %d : A22 local dims = %d %d\n", my_pid, 
                     invA22NRows, invA22NCols);

    //------------------------------------------------------------------
    // create a matrix context for A22
    //------------------------------------------------------------------

    ierr = HYPRE_IJMatrixCreate(comm,&invA22,invA22GlobalNRows,invA22GlobalNCols);
    assert(!ierr);
    ierr = HYPRE_IJMatrixSetLocalStorageType(invA22, HYPRE_PARCSR);
    assert(!ierr);
    ierr = HYPRE_IJMatrixSetLocalSize(invA22, invA22NRows, invA22NCols);
    assert(!ierr);

    //------------------------------------------------------------------
    // compute the number of nonzeros in the first nConstr row of invA22
    //------------------------------------------------------------------

    maxRowSize  = 0;
    invA22MatSize = new int[invA22NRows];
    for ( i = 0; i < nConstr; i++ ) invA22MatSize[i] = 1;

    //------------------------------------------------------------------
    // compute the number of nonzeros in the second nConstr row of invA22
    // (consisting of [D and A22 block])
    //------------------------------------------------------------------

    for ( i = 0; i < nSelected; i++ ) 
    {
       rowIndex = selectedList[i];
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
	        searchIndex = hypre_BinarySearch(globalSelectedList, colIndex, 
                                                 globalNSelected); 
                if ( searchIndex >= 0 ) rowSize2++;
             }
          }
       }
       invA22MatSize[nConstr+i] = rowSize2;
       maxRowSize = ( rowSize2 > maxRowSize ) ? rowSize2 : maxRowSize;
       HYPRE_ParCSRMatrixRestoreRow(A_csr,rowIndex,&rowSize,&colInd,&colVal);
    }

    //------------------------------------------------------------------
    // after fetching the row sizes, set up invA22 with such sizes
    //------------------------------------------------------------------

    ierr = HYPRE_IJMatrixSetRowSizes(invA22, invA22MatSize);
    assert(!ierr);
    ierr = HYPRE_IJMatrixInitialize(invA22);
    assert(!ierr);
    delete [] invA22MatSize;

    //------------------------------------------------------------------
    // next load the first nConstr row to invA22 extracted from A
    // (that is, the D block)
    //------------------------------------------------------------------

    maxRowSize++;
    newColInd = new int[maxRowSize];
    newColVal = new double[maxRowSize];

    for ( i = 0; i < diagCount; i++ ) 
    {
       extDiagonal[i] = 1.0 / extDiagonal[i];
    }
    for ( i = 0; i < nConstr; i++ ) {
       newColInd[0] = A2StartRow + nConstr + i; 
       rowIndex     = A2StartRow + i;
       newColVal[0] = extDiagonal[A2StartRow/2+i];
       ierr = HYPRE_IJMatrixInsertRow(invA22,1,rowIndex,newColInd,newColVal);
       assert(!ierr);
    }

    //------------------------------------------------------------------
    // next load the second nConstr rows to A22 extracted from A
    //------------------------------------------------------------------

    for ( i = 0; i < nSelected; i++ ) 
    {
       rowIndex = selectedList[i];
       HYPRE_ParCSRMatrixGetRow(A_csr,rowIndex,&rowSize,&colInd,&colVal);
       newRowSize = 1;
       newColInd[0] = A2StartRow + i;
       newColVal[0] = extDiagonal[A2StartRow/2+i]; 
       for (j = 0; j < rowSize; j++) 
       {
          colIndex = colInd[j];

          if ( colVal[j] != 0.0 )
          {
	     searchIndex = hypre_BinarySearch(globalSelectedList, colIndex, 
                                              globalNSelected); 
             if ( searchIndex >= 0 ) 
             {
                for ( procIndex = 0; procIndex < num_procs; procIndex++ )
                   if ( ProcNRows[procIndex] > colIndex ) break;
                procIndex--;
                newColInd[newRowSize] = searchIndex + ProcNConstr[procIndex] + 
                                        nConstr;
                newColVal[newRowSize++] = - extDiagonal[A2StartRow/2+i] * colVal[j] *
                                            extDiagonal[searchIndex];
                if ( newRowSize > maxRowSize )
                    printf("##### : pass array boundary.\n");
      	     } 
	  } 
       }
       rowCount = A2StartRow + nConstr + i;
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

    ncnt = 100;
    while ( ncnt < num_procs ) {
       if ( my_pid == ncnt ) {
          for ( i = A2StartRow; i < A2StartRow+2*nConstr; i++ ) {
             HYPRE_ParCSRMatrixGetRow(invA22_csr,i,&rowSize,&colInd,&colVal);
             printf("ROW = %6d (%d)\n", i, rowSize);
             for ( j = 0; j < rowSize; j++ )
                printf("   col = %6d, val = %e \n", colInd[j], colVal[j]);
             HYPRE_ParCSRMatrixRestoreRow(invA22_csr,i,&rowSize,&colInd,&colVal);
          }
       }
       MPI_Barrier(MPI_COMM_WORLD);
       ncnt++;
    }

    //******************************************************************
    // perform the triple matrix product
    //------------------------------------------------------------------

    A21_csr = (HYPRE_ParCSRMatrix) HYPRE_IJMatrixGetLocalStorage(A21);
    invA22_csr = (HYPRE_ParCSRMatrix) HYPRE_IJMatrixGetLocalStorage(invA22);
    printf("buildReducedSystem %d : Triple matrix product starts\n", my_pid);
    hypre_ParAMGBuildCoarseOperator( (hypre_ParCSRMatrix *) A21_csr,
                                     (hypre_ParCSRMatrix *) invA22_csr,
                                     (hypre_ParCSRMatrix *) A21_csr,
                                     (hypre_ParCSRMatrix **) &RAP_csr);
    printf("buildReducedSystem %d : Triple matrix product ends\n", my_pid);
    ncnt = 100;
    while ( ncnt < num_procs )
    {
       if ( my_pid == ncnt )
       {
          for ( i = A2StartRow; i < A2StartRow+A21NCols; i++ ) {
             ierr = HYPRE_ParCSRMatrixGetRow(RAP_csr,i,&rowSize,&colInd, &colVal);
             assert(!ierr);
             printf("ROW = %6d (%d)\n", i, rowSize);
             for ( j = 0; j < rowSize; j++ )
                printf("   col = %6d, val = %e \n", colInd[j], colVal[j]);
             ierr = HYPRE_ParCSRMatrixRestoreRow(RAP_csr,i,&rowSize,&colInd,&colVal);
             assert(!ierr);
          }
       }
       MPI_Barrier(MPI_COMM_WORLD);
       ncnt++;
    }

    //******************************************************************
    // extract the A11 part of A and minus the RAP
    //------------------------------------------------------------------

    newNRows = nRows - 2 * nConstr;
    newGlobalNRows = globalNRows - 2 * globalNConstr;

    ierr = HYPRE_IJMatrixCreate(comm,&reducedA,newGlobalNRows,newGlobalNRows);
    assert(!ierr);
    ierr = HYPRE_IJMatrixSetLocalStorageType(reducedA, HYPRE_PARCSR);
    assert(!ierr);
    ierr = HYPRE_IJMatrixSetLocalSize(reducedA, newNRows, newNRows);
    assert(!ierr);

    //------------------------------------------------------------------
    // set up reducedA with proper sizes
    //------------------------------------------------------------------

    reducedAMatSize = new int[newNRows];
    reducedAStartRow = ProcNRows[my_pid] - 2 * ProcNConstr[my_pid];
    rowCount = reducedAStartRow;
    rowIndex = 0;

    for ( i = StartRow; i <= newEndRow; i++ ) 
    {
       searchIndex = hypre_BinarySearch(selectedList, i, nSelected); 
       if ( searchIndex < 0 )  
       {
          ierr = HYPRE_ParCSRMatrixGetRow(A_csr,i,&rowSize,&colInd,&colVal);
          assert( !ierr );
          ierr = HYPRE_ParCSRMatrixGetRow(RAP_csr,rowCount,&rowSize2,&colInd2,
                                          &colVal2);
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
         
          ierr = HYPRE_ParCSRMatrixRestoreRow(A_csr,i,&rowSize,&colInd,&colVal);
          assert( !ierr );
          ierr = HYPRE_ParCSRMatrixRestoreRow(RAP_csr,rowCount,&rowSize2,
                                              &colInd2,&colVal2);
          assert( !ierr );
          rowCount++;
       }
    }
    ierr = HYPRE_IJMatrixSetRowSizes(reducedA, reducedAMatSize);
    assert(!ierr);
    ierr = HYPRE_IJMatrixInitialize(reducedA);
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
          HYPRE_ParCSRMatrixGetRow(RAP_csr,rowCount,&rowSize2,&colInd2,&colVal2);
          newRowSize = rowSize + rowSize2;
          newColInd  = new int[newRowSize];
          newColVal  = new double[newRowSize];
          ncnt       = 0;
          for ( j = 0; j < rowSize; j++ ) 
          {
             colIndex = colInd[j];
             for ( procIndex = 0; procIndex < num_procs; procIndex++ )
                if ( ProcNRows[procIndex] > colIndex ) break;
             if ( procIndex == num_procs ) 
                ubound = globalNRows - ( globalNConstr - ProcNConstr[num_procs-1] );
             else
                ubound = ProcNRows[procIndex] - 
                         (ProcNConstr[procIndex]-ProcNConstr[procIndex-1]);
             procIndex--;
             if ( colIndex < ubound ) 
             {
                searchIndex = hypre_BinarySearch(globalSelectedList, colIndex, 
                                                 globalNSelected); 
                if ( searchIndex < 0 ) 
                {
                   searchIndex = - searchIndex - 1;
                   newColInd[ncnt] = colIndex - ProcNConstr[procIndex] - searchIndex;
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
          // translate the newColIndices
          ierr = HYPRE_IJMatrixInsertRow(reducedA, newRowSize, rowCount,
                                        newColInd, newColVal);
          assert(!ierr);
          HYPRE_ParCSRMatrixRestoreRow(A_csr,i,&rowSize,&colInd,&colVal);
          HYPRE_ParCSRMatrixRestoreRow(RAP_csr,rowCount,&rowSize2,&colInd2,&colVal2);
          rowCount++;
          delete [] newColInd;
          delete [] newColVal;
       }
    }
    HYPRE_IJMatrixAssemble(reducedA);
    reducedA_csr = (HYPRE_ParCSRMatrix) HYPRE_IJMatrixGetLocalStorage(reducedA);
    MPI_Barrier(MPI_COMM_WORLD);
    printf("buildReducedSystem FINAL : reducedAStartRow = %d\n", reducedAStartRow);
    ncnt = 100;
    while ( ncnt < num_procs )
    {
       if ( my_pid == ncnt )
       {
          for ( i = reducedAStartRow; i < reducedAStartRow+nRows-2*nConstr; i++ ) 
          {
             ierr = HYPRE_ParCSRMatrixGetRow(reducedA_csr,i,&rowSize,&colInd,
                                             &colVal);
             qsort1(colInd, colVal, 0, rowSize-1);
             printf("ROW = %6d (%d)\n", i+1, rowSize);
             for ( j = 0; j < rowSize; j++ )
                if ( colVal[j] != 0.0 )
                printf("   col = %6d, val = %e \n", colInd[j]+1, colVal[j]);
             HYPRE_ParCSRMatrixRestoreRow(reducedA_csr,i,&rowSize,&colInd,&colVal);
          }
       }
       MPI_Barrier(MPI_COMM_WORLD);
       ncnt++;
    }
    delete [] globalSelectedList;
    delete [] selectedList;
    delete [] ProcNRows;
    delete [] ProcNConstr;
}

//***************************************************************************
// this function solve the incoming linear system using SuperLU
//---------------------------------------------------------------------------

void HYPRE_SLE::solveUsingSuperLU(int& status)
{
    int                i, nnz, num_procs, startRow, endRow, nrows, ierr;
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

    MPI_Comm_size(comm, &num_procs);
    if ( num_procs > 1 )
    {
       printf("HYPRE_SLE::solveUsingSuperLU error - too many processors.\n");
       status = -1;
       return;
    }

    //------------------------------------------------------------------
    // need to construct a CSR matrix, and the column indices should
    // have been stored in colIndices and rowLengths
    //------------------------------------------------------------------
      
    if ( colIndices == NULL || rowLengths == NULL )
    {
       printf("HYPRE_SLE::solveUsingSuperLU error - matConfigure not called yet.\n");
       status = -1;
       return;
    }
    if ( StartRow_ != 1 )
    {
       printf("HYPRE_SLE::solveUsingSuperLU error - row does not start at 1.\n");
       status = -1;
       return;
    }
    nrows = EndRow_;
    nnz   = 0;
    for ( i = 0; i < nrows; i++ ) nnz += rowLengths[i];

    new_ia = new int[nrows+1];
    new_ja = new int[nnz];
    new_a  = new double[nnz];
    A_csr  = (HYPRE_ParCSRMatrix) HYPRE_IJMatrixGetLocalStorage(HY_A);

    nz_ptr = getMatrixCSR(nrows, nnz, new_ia, new_ja, new_a);

    nnz = nz_ptr;

    //------------------------------------------------------------------
    // set up SuperLU CSR matrix and the corresponding rhs
    //------------------------------------------------------------------

    dCreate_CompRow_Matrix(&A2,nrows,nrows,nnz,new_a,new_ja,new_ia,NR,_D,GE);
    ind_array = new int[nrows];
    for ( i = 0; i < nrows; i++ ) ind_array[i] = i;
    rhs = new double[nrows];
    ierr = HYPRE_IJVectorGetLocalComponents(HY_b, nrows, ind_array, NULL, rhs);
    assert(!ierr);
    dCreate_Dense_Matrix(&B, nrows, 1, rhs, nrows, DN, _D, GE);

    //------------------------------------------------------------------
    // set up the rest and solve (permc_spec=0 : natural ordering)
    //------------------------------------------------------------------
 
    perm_r = new int[nrows];
    perm_c = new int[nrows];
    permc_spec = superlu_ordering;
    get_perm_c(permc_spec, &A2, perm_c);
    panel_size = sp_ienv(1);

    dgssv(&A2, perm_c, perm_r, &L, &U, &B, &info);

    //------------------------------------------------------------------
    // postprocessing of the return status information
    //------------------------------------------------------------------

    if ( info == 0 ) {

        status = 1;
        Lstore = (SCformat *) L.Store;
        Ustore = (NRformat *) U.Store;
        //printf("No of nonzeros in factor L = %d\n", Lstore->nnz);
        //printf("No of nonzeros in factor U = %d\n", Ustore->nnz);
        printf("   SuperLU : NNZ in L+U = %d\n", Lstore->nnz+Ustore->nnz-nrows);

        //dQuerySpace(&L, &U, panel_size, &mem_usage);
        //printf("L\\U MB %.3f\ttotal MB needed %.3f\texpansions %d\n",
        //       mem_usage.for_lu/1e6, mem_usage.total_needed/1e6,
        //       mem_usage.expansions);

    } else {
        status = 0;
        printf("HYPRE_SLE::solveUsingSuperLU - dgssv error code = %d\n",info);
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
       ierr = HYPRE_IJVectorSetLocalComponents(HY_x,nrows,ind_array,NULL,soln);
       assert(!ierr);
       x_csr  = (HYPRE_ParVector) HYPRE_IJVectorGetLocalStorage(HY_x);
       b_csr  = (HYPRE_ParVector) HYPRE_IJVectorGetLocalStorage(HY_b);
       r_csr  = (HYPRE_ParVector) HYPRE_IJVectorGetLocalStorage(HY_r);
       ierr = HYPRE_ParVectorCopy( b_csr, r_csr );
       assert(!ierr);
       ierr = HYPRE_ParCSRMatrixMatvec( -1.0, A_csr, x_csr, 1.0, r_csr );
       assert(!ierr);
       ierr = HYPRE_ParVectorInnerProd( r_csr, r_csr, &rnorm);
       assert(!ierr);
       rnorm = sqrt( rnorm );
       printf("HYPRE_SLE::solveUsingSuperLU - FINAL NORM =               %e.\n", rnorm);
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
    printf("HYPRE_SLE::solveUsingSuperLU : not available.\n");
#endif
}

//***************************************************************************
// this function solve the incoming linear system using SuperLU
// using expert mode
//---------------------------------------------------------------------------

void HYPRE_SLE::solveUsingSuperLUX(int& status)
{
    int                i, k, nnz, num_procs, startRow, endRow, nrows, ierr;
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

    MPI_Comm_size(comm, &num_procs);
    if ( num_procs > 1 )
    {
       printf("HYPRE_SLE::solveUsingSuperLUX error - too many processors.\n");
       status = -1;
       return;
    }

    //------------------------------------------------------------------
    // need to construct a CSR matrix, and the column indices should
    // have been stored in colIndices and rowLengths
    //------------------------------------------------------------------
      
    if ( colIndices == NULL || rowLengths == NULL )
    {
       printf("HYPRE_SLE::solveUsingSuperLUX error - matConfigure not called yet.\n");
       status = -1;
       return;
    }
    if ( StartRow_ != 1 )
    {
       printf("HYPRE_SLE::solveUsingSuperLUX error - row does not start at 1.\n");
       status = -1;
       return;
    }
    nrows = EndRow_;
    colLengths = new int[nrows];
    for ( i = 0; i < nrows; i++ ) colLengths[i] = 0;
    
    maxRowSize = 0;
    A_csr  = (HYPRE_ParCSRMatrix) HYPRE_IJMatrixGetLocalStorage(HY_A);
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
    ierr = HYPRE_IJVectorGetLocalComponents(HY_b, nrows, ind_array, NULL, rhs);
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
    permc_spec = superlu_ordering;
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
        printf("   SuperLU : NNZ in L+U = %d\n", Lstore->nnz+Ustore->nnz-nrows);

        //dQuerySpace(&L, &U, panel_size, &mem_usage);
        //printf("L\\U MB %.3f\ttotal MB needed %.3f\texpansions %d\n",
        //       mem_usage.for_lu/1e6, mem_usage.total_needed/1e6,
        //       mem_usage.expansions);
    } else {
        printf("HYPRE_SLE::solveUsingSuperLUX - dgssvx error code = %d\n",info);
        status = 0;
    }

    //------------------------------------------------------------------
    // fetch the solution and find residual norm
    //------------------------------------------------------------------

    if ( status == 1 )
    {
       ierr = HYPRE_IJVectorSetLocalComponents(HY_x,nrows,ind_array,NULL,soln);
       assert(!ierr);
       x_csr  = (HYPRE_ParVector) HYPRE_IJVectorGetLocalStorage(HY_x);
       r_csr  = (HYPRE_ParVector) HYPRE_IJVectorGetLocalStorage(HY_r);
       b_csr    = (HYPRE_ParVector) HYPRE_IJVectorGetLocalStorage(HY_b);
       ierr = HYPRE_ParVectorCopy( b_csr, r_csr );
       assert(!ierr);
       ierr = HYPRE_ParCSRMatrixMatvec( -1.0, A_csr, x_csr, 1.0, r_csr );
       assert(!ierr);
       ierr = HYPRE_ParVectorInnerProd( r_csr, r_csr, &rnorm);
       assert(!ierr);
       rnorm = sqrt( rnorm );
       printf("HYPRE_SLE::solveUsingSuperLUX - FINAL NORM =             %e.\n", rnorm);
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
    printf("HYPRE_SLE::solveUsingSuperLUX : not available.\n");
#endif

}

//***************************************************************************
// this function solve the incoming linear system using Y12M
//---------------------------------------------------------------------------

void HYPRE_SLE::solveUsingY12M(int& status)
{
    int                i, k, nnz, num_procs, nrows, ierr;
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

    MPI_Comm_size(comm, &num_procs);
    if ( num_procs > 1 )
    {
       printf("HYPRE_SLE::solveUsingY12M error - too many processors.\n");
       status = 0;
       return;
    }

    //------------------------------------------------------------------
    // need to construct a CSR matrix, and the column indices should
    // have been stored in colIndices and rowLengths
    //------------------------------------------------------------------
      
    if ( colIndices == NULL || rowLengths == NULL )
    {
       printf("HYPRE_SLE::solveUsingY12M error - matConfigure not called yet.\n");
       status = -1;
       return;
    }
    if ( StartRow_ != 1 )
    {
       printf("HYPRE_SLE::solveUsingY12M error - row does not start at 1.\n");
       status = -1;
       return;
    }
    nrows = EndRow_;
    colLengths = new int[nrows];
    for ( i = 0; i < nrows; i++ ) colLengths[i] = 0;
    
    maxRowSize = 0;
    A_csr  = (HYPRE_ParCSRMatrix) HYPRE_IJMatrixGetLocalStorage(HY_A);
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
    ierr = HYPRE_IJVectorGetLocalComponents(HY_b, nrows, ind_array, NULL, rhs);
    assert(!ierr);

    //------------------------------------------------------------------
    // call Y12M to solve the linear system
    //------------------------------------------------------------------

    y12maf_(&nrows,&nnz,val,snr,&nn,rnr,&nn1,pivot,ha,&iha,aflag,iflag,rhs,&ifail);
    if ( ifail != 0 )
    {
       printf("HYPRE_SLE::solveUsingY12M warning - ifail = %d\n", ifail);
    }
 
    //------------------------------------------------------------------
    // postprocessing
    //------------------------------------------------------------------

    if ( ifail == 0 )
    {
       ierr = HYPRE_IJVectorSetLocalComponents(HY_x,nrows,ind_array,NULL,rhs);
       assert(!ierr);
       x_csr  = (HYPRE_ParVector) HYPRE_IJVectorGetLocalStorage(HY_x);
       r_csr  = (HYPRE_ParVector) HYPRE_IJVectorGetLocalStorage(HY_r);
       b_csr  = (HYPRE_ParVector) HYPRE_IJVectorGetLocalStorage(HY_b);
       ierr = HYPRE_ParVectorCopy( b_csr, r_csr );
       assert(!ierr);
       ierr = HYPRE_ParCSRMatrixMatvec( -1.0, A_csr, x_csr, 1.0, r_csr );
       assert(!ierr);
       ierr = HYPRE_ParVectorInnerProd( r_csr, r_csr, &rnorm);
       assert(!ierr);
       rnorm = sqrt( rnorm );
       printf("HYPRE_SLE::solveUsingY12M - final norm = %e.\n", rnorm);
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
    printf("HYPRE_SLE::solveUsingY12M - not available.\n");
#endif

}

//***************************************************************************
// this function extracts the matrix in a CSR format
//---------------------------------------------------------------------------

void HYPRE_SLE::loadSlaveList(int nslaves, int *slist)
{
#ifdef DEBUG
    if ( my_pid == 0 )
    {
       printf("HYPRE_SLE::loadSlaveList - number of slaves in P0 = %d\n", nslaves);
    }
#endif

    nSlaves = nslaves;
    nConstr = nslaves;
    if ( nslaves > 0 )
    {
       slaveList = new int[nslaves];
       for ( int i = 0; i < nslaves; i++ ) slaveList[i] = slist[i];
    }
}

//***************************************************************************
// this function extracts the matrix in a CSR format
//---------------------------------------------------------------------------

int HYPRE_SLE::getMatrixCSR(int nrows, int nnz, int *ia_ptr, int *ja_ptr,
                            double *a_ptr) 
{
    int                nz, i, j, ierr, rowSize, *colInd, nz_ptr, *colInd2;
    int                firstNnz;
    double             *colVal, *colVal2;
    HYPRE_ParCSRMatrix A_csr;

    nz        = 0;
    nz_ptr    = 0;
    ia_ptr[0] = nz_ptr;
    A_csr  = (HYPRE_ParCSRMatrix) HYPRE_IJMatrixGetLocalStorage(HY_A);
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
       if ( rowSize > rowLengths[i] )
          printf("HYPRE_SLE::getMatrixCSR warning at row %d - %d %d\n",
                                              i,rowSize,rowLengths[i]);
       qsort1(colInd2, colVal2, 0, rowSize-1);
       for ( j = 0; j < rowSize-1; j++ )
          if ( colInd2[j] == colInd2[j+1] )
             printf("HYPRE_SLE::getMatrixCSR - duplicate colind at row %d %d \n",i,colInd2[j]); 

       firstNnz = 0;
       for ( j = 0; j < rowSize; j++ )
       {
          if ( colVal2[j] != 0.0 )
          {
             if ( nz_ptr > 0 && firstNnz > 0 && colInd2[j] == ja_ptr[nz_ptr-1] ) 
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
                   printf("HYPRE_SLE::getMatrixCSR error (1) - %d %d.\n",i,nrows);
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
       printf("HYPRE_SLE::getMatrixCSR note : matrix sparsity has been\n");
       printf("           changed since matConfigure - %d > %d ?\n", nnz, nz_ptr);
       printf("           number of zeros            = %d \n", nz );
    }
    return nz_ptr;
}

//***************************************************************************
//***************************************************************************
//***************************************************************************
// The following are the test functions for the above routines
//***************************************************************************
//***************************************************************************
//***************************************************************************


//***************************************************************************
// This is a C++ test for the HYPRE FEI code in HYPRE_SLE.cc.
// It does not perform the test by calling the FEI interface functions
// which were implemented by Sandia. 
// The following is a friend function of HYPRE_SLE because it needs to call
// protected member functions of that class in order to do the test.
//
// This test program uses two processors, and sets up the following matrix.
// Rows 1-3 belong to proc 0, and rows 4-6 belong to proc 1.
// Five elements of the form [1 -1; -1 1] are summed.
// The right hand side elements are [1; -1].
//
//  1 -1  0  0  0  0      1
// -1  2 -1  0  0  0      0
//  0 -1  2 -1  0  0     -1
//  0  0 -1  2 -1  0      0
//  0  0  0 -1  2 -1      0
//  0  0  0  0 -1  1      0
//
//  We then enforce a essential BC in row 1 (alpha=2, beta=0, gamma=2)
//  and a mixed BC in row 6 (alpha=5, beta =5, gamma=10).
//  The resulting system is:
//
//  1  0  0  0  0  0      1
// -1  2 -1  0  0  0      0
//  0 -1  2 -1  0  0     -1
//  0  0 -1  2 -1  0      0
//  0  0  0 -1  2 -1      0
//  0  0  0  0 -1  2      2
//
//  The solution is [ 1.0, 0.5, 0.0, 0.5, 1.0, 1.5]
//  NOTE: for CG to perform converge on this "non-symetric" matrix, it
//  is necessary for the initial guess to satisfy the essential BC.
//---------------------------------------------------------------------------

void fei_hypre_test(int argc, char *argv[])
{
    int my_rank;
    int num_procs;
    int i;
    int status;

    IntArray *rows;
    rows = new IntArray[3];

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    assert(num_procs == 2);

    switch (my_rank)
    {
        case 0:
           rows[0].append(1);
           rows[0].append(2);
           rows[1].append(1);
           rows[1].append(2);
           rows[1].append(3);
           rows[2].append(2);
           rows[2].append(3);
           rows[2].append(4);
           break;
        case 1 :
           rows[0].append(3);
           rows[0].append(4);
           rows[0].append(5);
           rows[1].append(4);
           rows[1].append(5);
           rows[1].append(6);
           rows[2].append(5);
           rows[2].append(6);
           break;
    }

    const int ind1[] = {1, 2};
    const int ind2[] = {2, 3};
    const int ind3[] = {3, 4};
    const int ind4[] = {4, 5};
    const int ind5[] = {5, 6};

    const double val1[] = {1.0, -1.0};
    const double val2[] = {-1.0, 1.0};

    const int indg1[] = {1, 2, 3};
    const int indg2[] = {4, 5, 6};
    const double valg[] = {1.0, 0.2, 0.3};

    int dir_index[] = {1};
    double dir_alpha[] = {2.0};
    double dir_gamma[] = {2.0};

    int mix_index[] = {6};
    double mix_alpha[] = {5.0};
    double mix_beta[] = {5.0};
    double mix_gamma[] = {10.0};

    HYPRE_SLE H(MPI_COMM_WORLD, 0);

    if (my_rank == 0)
        H.createLinearAlgebraCore(6, 1, 3, 1, 3);
    else
        H.createLinearAlgebraCore(6, 4, 6, 4, 6);

    H.matrixConfigure(rows);

    switch (my_rank)
    {
	case 0:

            H.sumIntoSystemMatrix(1, 2, val1, ind1);
            H.sumIntoSystemMatrix(2, 2, val2, ind1);

            H.sumIntoSystemMatrix(2, 2, val1, ind2);
            H.sumIntoSystemMatrix(3, 2, val2, ind2);

            H.sumIntoSystemMatrix(3, 2, val1, ind3);

            H.sumIntoRHSVector(2, ind1, val1);   // rhs vector
            H.sumIntoRHSVector(2, ind2, val1);

            H.putIntoSolnVector(3, indg1, valg); // initial guess

	    break;

	case 1:

            H.sumIntoSystemMatrix(4, 2, val2, ind3);

            H.sumIntoSystemMatrix(4, 2, val1, ind4);
            H.sumIntoSystemMatrix(5, 2, val2, ind4);

            H.sumIntoSystemMatrix(5, 2, val1, ind5);
            H.sumIntoSystemMatrix(6, 2, val2, ind5);

            //H.sumIntoRHSVector(2, ind5, val1);   // rhs vector
            //H.putIntoSolnVector(3, indg2, valg); // initial guess//BUG

	    break;

        default:
	    assert(0);
    }

    H.enforceOtherBC(mix_index, mix_alpha, mix_beta, mix_gamma, 1);
    H.enforceEssentialBC(dir_index, dir_alpha, dir_gamma, 1);

    H.matrixLoadComplete();

    H.selectSolver("cg");
    H.selectPreconditioner("pilut");

    H.launchSolver(&status);
    assert(status == 1);

    // get the result
    for (i=1; i<=3; i++)
      if (my_rank == 0)
	printf("sol(%d): %f\n", i, H.accessSolnVector(i));
      else
	printf("sol(%d): %f\n", i+3, H.accessSolnVector(i+3));

    MPI_Finalize();

    // note implicit call to destructor at end of scope
}

//***************************************************************************
// This is a C++ test for the HYPRE FEI code in HYPRE_SLE.cc.
//---------------------------------------------------------------------------

void fei_hypre_test2(int argc, char *argv[])
{
    int    i, j, my_rank, num_procs, nrows, nnz, mybegin, myend, status;
    int    *ia, *ja, ncnt, index, chunksize, zeroDiagFlag;
    double *val, *rhs;

    //======================================================
    // initialize parallel platform
    //======================================================

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    HYPRE_SLE H(MPI_COMM_WORLD, 0);

    //======================================================
    // read the matrix and rhs and broadcast 
    //======================================================

    if ( my_rank == 0 ) {
       ML_Get_IJAMatrixFromFile(&val, &ia, &ja, &nrows, 
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
       ia  = (int    *) malloc( (nrows + 1) * sizeof( int ) );
       ja  = (int    *) malloc( nnz * sizeof( int ) );
       val = (double *) malloc( nnz * sizeof( double ) );
       rhs = (double *) malloc( nrows * sizeof( double ) );

       MPI_Bcast(ia,  nrows+1, MPI_INT,    0, MPI_COMM_WORLD);
       MPI_Bcast(ja,  nnz,     MPI_INT,    0, MPI_COMM_WORLD);
       MPI_Bcast(val, nnz,     MPI_DOUBLE, 0, MPI_COMM_WORLD);
       MPI_Bcast(rhs, nrows,   MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    chunksize = nrows / num_procs;
    mybegin = chunksize * my_rank;
    myend   = chunksize * (my_rank + 1) - 1;
    if ( my_rank == num_procs-1 ) myend = nrows - 1;
    printf("Processor %d : begin/end = %d %d\n", my_rank, mybegin, myend);
    fflush(stdout);

    //======================================================
    // create matrix and rhs in the hypre context 
    //======================================================

    H.createLinearAlgebraCore(nrows, mybegin+1, myend+1, mybegin+1, myend+1);

    IntArray  *rows;
    double    zero=0.0;

    rows = new IntArray[nrows]();
    for ( i = mybegin; i < myend+1; i++ ) {
       for ( j = ia[i]; j < ia[i+1]; j++ ) {
          rows[i-mybegin].append(ja[j]);
       }
    }

    H.matrixConfigure(rows);

    for ( i = mybegin; i <= myend; i++ ) { 
       ncnt = ia[i+1] - ia[i];
       index = i + 1;
       H.sumIntoSystemMatrix(index, ncnt, &val[ia[i]], &ja[ia[i]]);
    }

    H.matrixLoadComplete();

    for ( i = mybegin; i <= myend; i++ ) { 
       index = i + 1;
       H.sumIntoRHSVector(1, &index, &rhs[i]);
    }

    char *paramString = new char[100];
    strcpy(paramString, "pilut-row-size 50");
    H.parameters(1, &paramString);

    H.selectSolver("gmres");

    H.selectPreconditioner("boomeramg");

    H.launchSolver(&status);

    assert(status == 1);

    // get the result
    /*
    for (i=1; i<=10; i++)
       printf("sol(%d): %f\n", i, H.accessSolnVector(i));
    */

    MPI_Finalize();

    delete [] rows;
    free( ia );
    free( ja );
    free( val );
    free( rhs );

    // note implicit call to destructor at end of scope
}

