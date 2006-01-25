/*--------------------------------------------------------------------------
 * Test driver for unstructured matrix interface (IJ_matrix interface).
 * This is the version which uses the Babel interface.
 * Do `driver -help' for usage info.
 * This driver started from the driver for parcsr_linear_solvers, and it
 * works by first building a parcsr matrix as before and then "copying"
 * that matrix row-by-row into the IJMatrix interface. AJC 7/99.
 *--------------------------------------------------------------------------*/
/* As of October 2005, the solvers implemented are AMG, ParaSails, PCG, GMRES, diagonal
   scaling, and combinations thereof.  The babel (bHYPRE) interface is used exclusively.
*/
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>

#include "utilities.h"
#include "HYPRE.h"
#include "HYPRE_parcsr_mv.h"

#include "HYPRE_IJ_mv.h"
#include "HYPRE_parcsr_ls.h"
#include "krylov.h"

#include "bHYPRE.h"
#include "bHYPRE_IJParCSRMatrix_Impl.h"
#include "bHYPRE_IJParCSRVector_Impl.h"
#include "bHYPRE_ParCSRDiagScale_Impl.h"
#include "bHYPRE_Schwarz_Impl.h"
#include "bHYPRE_MPICommunicator_Impl.h"

const double dt_inf = 1.e40;
typedef struct
{
   /* Parameters which the user may set through the command line
      (some exceptions are  noted ) */
   int                 print_usage;
   int                 build_matrix_type;
   int                 build_matrix_arg_index;
   int                 build_rhs_type;
   int                 build_rhs_arg_index;
   int                 build_src_type;
   int                 build_src_arg_index;
   int                 build_funcs_type;
   int                 build_funcs_arg_index;
   int                 sparsity_known;
   int                 solver_id;
   int	               smooth_num_levels;
   int	               hpcg;
   int                 coarsen_type;
   int	               hybrid;
   int                 measure_type;
   int                 relax_default;
   int	               smooth_type;
   int                 max_levels;
   int                 debug_flag;
   int		       num_functions;
   int                 num_sweep;
   int                 smooth_num_sweep;
   double              dt;
   double              strong_threshold;
   double              trunc_factor;
   int                 cycle_type;
   int                 ioutdat;
   int                 poutdat;
   int                 k_dim; /* for GMRES */
   double              drop_tol;  /* for PILUT */
   int                 nonzeros_to_keep; /* for PILUT */
   double              schwarz_rlx_weight; /* for Schwarz and BoomerAMG */
   int                 variant; /* multiplicative; for Schwarz */
   int                 overlap; /* 1 layer overlap; for Schwarz */
   int                 domain_type; /* through agglomeration; for Schwarz */
   double              max_row_sum; /* for BoomerAMG */
   double              tol;
   double              pc_tol; /* for BoomerAMG, not yet user-settable */
   double              sai_threshold; /* for ParaSAILS */
   double              sai_filter; /* for ParaSAILS */
   /* Scalar command-line arguments provide some control over array values. */
   double             *relax_weight;  /* for BoomerAMG */
   int                *num_grid_sweeps;   /* for BoomerAMG */
   int                *grid_relax_type;   /* for BoomerAMG */
   int               **grid_relax_points; /* for BoomerAMG; not user-settable */
   double             *omega;  /* for BoomerAMG; not presently referenced or user-settable */
   int                 gsmg_samples;  /* for AMG-GSMG */
   int                 interp_type;   /* for AMG-GSMG */

} CommandLineParameters;

int BuildParFromFile (int argc , char *argv [], int arg_index , HYPRE_ParCSRMatrix *A_ptr );
int BuildParLaplacian (int argc , char *argv [], int arg_index , HYPRE_ParCSRMatrix *A_ptr );
int bBuildParLaplacian( int argc, char *argv[], int arg_index, bHYPRE_MPICommunicator bmpi_comm,
                        bHYPRE_IJParCSRMatrix  *bA_ptr );
int BuildParDifConv (int argc , char *argv [], int arg_index , HYPRE_ParCSRMatrix *A_ptr );
int BuildParFromOneFile (int argc , char *argv [], int arg_index , HYPRE_ParCSRMatrix *A_ptr );
int BuildFuncsFromFiles (int argc , char *argv [], int arg_index , bHYPRE_IJParCSRMatrix A , int **dof_func_ptr );
int BuildFuncsFromOneFile (int argc , char *argv [], int arg_index , bHYPRE_IJParCSRMatrix A , int **dof_func_ptr );
int BuildRhsParFromOneFile_ (int argc , char *argv [], int arg_index , int *partitioning , HYPRE_ParVector *b_ptr );
int BuildParLaplacian9pt (int argc , char *argv [], int arg_index , HYPRE_ParCSRMatrix *A_ptr );
int BuildParLaplacian27pt (int argc , char *argv [], int arg_index , HYPRE_ParCSRMatrix *A_ptr );
void ParseCommandLine_1( int argc, char *argv[], CommandLineParameters *clp );
void ParseCommandLine_2( int argc, char *argv[], CommandLineParameters *clp );
void BoomerAMG_DefaultParameters( CommandLineParameters *clp );
void PrintUsage( char *argv[] );
int IJMatrixVectorDebug(
   const bHYPRE_MPICommunicator bmpicomm, const int local_num_cols,
   const int first_local_col, const int last_local_col,
   const bHYPRE_IJParCSRMatrix  bHYPRE_parcsr_A,
   bHYPRE_IJParCSRVector  bHYPRE_b, bHYPRE_IJParCSRVector  bHYPRE_x );
int Demo_Matrix_AddToValues(
   bHYPRE_IJParCSRMatrix bHYPRE_parcsr_A, CommandLineParameters *clp,
   int first_local_row, int last_local_row );
void Print_BabelTimeCorrection( int myid, int argc, char *argv[],
                                CommandLineParameters *clp, MPI_Comm mpi_comm );
void BuildDefaultFuncs( CommandLineParameters *clp, int myid, int local_num_rows,
                        int first_local_row, int ** dof_func);
int Test_AMG( CommandLineParameters *clp, bHYPRE_IJParCSRMatrix bHYPRE_parcsr_A,
              bHYPRE_IJParCSRVector bHYPRE_b, bHYPRE_IJParCSRVector bHYPRE_x,
              int * dof_func,
              MPI_Comm mpi_comm, bHYPRE_MPICommunicator bmpicomm );
int PrecondAMG( CommandLineParameters *clp, int myid,
                bHYPRE_IJParCSRMatrix bHYPRE_parcsr_A,
                bHYPRE_Vector bHYPRE_Vector_b, bHYPRE_Vector bHYPRE_Vector_x,
                int * dof_func, bHYPRE_MPICommunicator bmpicomm,
                bHYPRE_Solver * bHYPRE_SolverPC );

int
main( int   argc,
      char *argv[] )
{
   CommandLineParameters * clp = hypre_CTAlloc( CommandLineParameters, 1 );

   int                 ierr = 0;
   int                 i,j; 
   int                 num_iterations; 
   /*double              norm;*/
   double tmp;
   double              final_res_norm;

   bHYPRE_MPICommunicator bmpicomm;
   bHYPRE_IJParCSRMatrix  bHYPRE_parcsr_A;
   bHYPRE_Operator        bHYPRE_op_A;
   bHYPRE_IJParCSRVector  bHYPRE_b;
   bHYPRE_IJParCSRVector  bHYPRE_x;
   bHYPRE_Vector          bHYPRE_Vector_x, bHYPRE_Vector_b;

   bHYPRE_BoomerAMG     bHYPRE_AMG = NULL;
   bHYPRE_PCG           bHYPRE_PCG;
   bHYPRE_HPCG          bHYPRE_HPCG;
   bHYPRE_GMRES         bHYPRE_GMRES;
   bHYPRE_HGMRES        bHYPRE_HGMRES;
   bHYPRE_BiCGSTAB      bHYPRE_BiCGSTAB;
   bHYPRE_CGNR          bHYPRE_CGNR;
   bHYPRE_ParCSRDiagScale  bHYPRE_ParCSRDiagScale;
   bHYPRE_ParaSails     bHYPRE_ParaSails;
   bHYPRE_Euclid        bHYPRE_Euclid;
   bHYPRE_Solver        bHYPRE_SolverPC;
   bHYPRE_Schwarz bHYPRE_Schwarz;

   int                 num_procs, myid;
   int                *dof_func;

   int		       time_index;
   MPI_Comm            mpi_comm = MPI_COMM_WORLD;
   int M, N;
   int first_local_row, last_local_row, local_num_rows;
   int first_local_col, last_local_col, local_num_cols;
   double *values;
   struct sidl_int__array* bHYPRE_grid_relax_points=NULL;

   int dimsl[2], dimsu[2];

   /*-----------------------------------------------------------
    * Initialize MPI
    *-----------------------------------------------------------*/

   /* Initialize MPI */
   MPI_Init(&argc, &argv);

   MPI_Comm_size( mpi_comm, &num_procs );
   MPI_Comm_rank( mpi_comm, &myid );
   bmpicomm = bHYPRE_MPICommunicator_CreateC( (void *)(&mpi_comm) );
/*
  hypre_InitMemoryDebug(myid);
*/
   /*-----------------------------------------------------------
    * Set defaults and read user-provided parameters
    *-----------------------------------------------------------*/
 
   ParseCommandLine_1( argc, argv, clp );

   /* defaults for BoomerAMG, uses results of ParseCommandLine_1,
      does nothing significant if BoomerAMG isn't needed;
      returns 4 arrays which we'll need to free up later... */
   BoomerAMG_DefaultParameters( clp );

   ParseCommandLine_2( argc, argv, clp );


   /*-----------------------------------------------------------
    * Print usage info, aka help
    *-----------------------------------------------------------*/
 
   if ( clp->print_usage )
   {
      if ( myid==0 )
         PrintUsage( argv );
      bHYPRE_MPICommunicator_deleteRef( bmpicomm );
      MPI_Finalize();
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/
 
   if (myid == 0)
   {
      printf("Running with these driver parameters:\n");
      printf("  solver ID    = %d\n\n", clp->solver_id);
   }

   /*-----------------------------------------------------------
    * Set up matrix
    *-----------------------------------------------------------*/

   if ( myid == 0 && clp->dt != dt_inf)
   {
      printf("  Backward Euler time step with dt = %e\n", clp->dt);
      printf("  Dirichlet 0 BCs are implicit in the spatial operator\n");
   }

   Print_BabelTimeCorrection( myid, argc, argv, clp, mpi_comm );

   time_index = hypre_InitializeTiming("Spatial operator");
   hypre_BeginTiming(time_index);

   if ( clp->build_matrix_type == -1 )

   {
      printf("build_matrix_type == -1 not currently implemented\n");
      return(-1);
   }
   else if ( clp->build_matrix_type == 0 )
   {
      printf("build_matrix_type == 0 not currently implemented\n");
      return(-1);
   }
   else if ( clp->build_matrix_type == 1 )
   {
      printf("build_matrix_type == 1 not currently implemented\n");
      return(-1);
   }
   else if ( clp->build_matrix_type == 2 )
   {
      bBuildParLaplacian(argc, argv, clp->build_matrix_arg_index, bmpicomm, &bHYPRE_parcsr_A);
   }
   else if ( clp->build_matrix_type == 3 )
   {
      printf("build_matrix_type == 3 not currently implemented\n");
      return(-1);
   }
   else if ( clp->build_matrix_type == 4 )
   {
      printf("build_matrix_type == 4 not currently implemented\n");
      return(-1);
   }
   else if ( clp->build_matrix_type == 5 )
   {
      printf("build_matrix_type == 5 not currently implemented\n");
      return(-1);
   }
   else
   {
      printf("You have asked for an unsupported problem with\n");
      printf("build_matrix_type = %d.\n", clp->build_matrix_type);
      return(-1);
   }

   ierr += bHYPRE_IJParCSRMatrix_GetLocalRange(
      bHYPRE_parcsr_A, &first_local_row, &last_local_row,
      &first_local_col, &last_local_col );
   local_num_rows = last_local_row - first_local_row + 1;
   local_num_cols = last_local_col - first_local_col +1;
   ierr += bHYPRE_IJParCSRMatrix_GetIntValue( bHYPRE_parcsr_A,
                                              "GlobalNumRows", &M );
   ierr += bHYPRE_IJParCSRMatrix_GetIntValue( bHYPRE_parcsr_A,
                                              "GlobalNumCols", &N );

   ierr += bHYPRE_IJParCSRMatrix_Assemble( bHYPRE_parcsr_A );


   hypre_EndTiming(time_index);
   hypre_PrintTiming("IJ Matrix Setup", MPI_COMM_WORLD);
   hypre_FinalizeTiming(time_index);
   hypre_ClearTiming();

   if (ierr)
   {
      printf("Error in driver building IJMatrix from parcsr matrix. \n");
      return(-1);
   }

   /* demonstration of use of AddToValues to change a matrix after it
      has been created and assembled... */
   ierr += Demo_Matrix_AddToValues( bHYPRE_parcsr_A, clp,
                                    first_local_row, last_local_row );

   /*-----------------------------------------------------------
    * Set up the RHS and initial guess
    *-----------------------------------------------------------*/

   time_index = hypre_InitializeTiming("RHS and Initial Guess");
   hypre_BeginTiming(time_index);

   if ( clp->build_rhs_type == 0 )
   {
      printf("build_rhs_type == 0 not currently implemented\n");
      return(-1);
   }
   else if ( clp->build_rhs_type == 1 )
   {
      printf("build_rhs_type == 1 not currently implemented\n");
      return(-1);
   }
   else if ( clp->build_rhs_type == 2 )
   {
      if (myid == 0)
      {
         printf("  RHS vector has unit components\n");
         printf("  Initial guess is 0\n");
      }

/* RHS */
      bHYPRE_b = bHYPRE_IJParCSRVector_Create( bmpicomm,
                                               first_local_row,
                                               last_local_row );

      ierr += bHYPRE_IJParCSRVector_Initialize( bHYPRE_b );


      values = hypre_CTAlloc(double, local_num_rows);
      for (i = 0; i < local_num_rows; i++)
         values[i] = 1.0;
      bHYPRE_IJParCSRVector_SetValues( bHYPRE_b, local_num_rows, NULL, values );
      hypre_TFree(values);

      ierr += bHYPRE_IJParCSRVector_Assemble( bHYPRE_b );


/* Initial guess */
      bHYPRE_x = bHYPRE_IJParCSRVector_Create( bmpicomm,
                                               first_local_row,
                                               last_local_row );

      ierr += bHYPRE_IJParCSRVector_Initialize( bHYPRE_x );

      values = hypre_CTAlloc(double, local_num_cols);
      for ( i=0; i<local_num_cols; ++i )
         values[i] = 0.;
      bHYPRE_IJParCSRVector_SetValues( bHYPRE_x, local_num_cols,
                                       NULL, values );
      hypre_TFree(values);

      ierr += bHYPRE_IJParCSRVector_Assemble( bHYPRE_x );

   }
   else if ( clp->build_rhs_type == 3 )
   {
      printf("build_rhs_type == 3 not currently implemented\n");
      return(-1);
   }
   else if ( clp->build_rhs_type == 4 )
   {
      printf("build_rhs_type == 4 not currently implemented\n");
      return(-1);
   }
   else if ( clp->build_rhs_type == 5 )
   {
      printf("build_rhs_type == 5 not currently implemented\n");
      return(-1);
   }

   hypre_EndTiming(time_index);
   hypre_PrintTiming("IJ Vector Setup", mpi_comm);
   hypre_FinalizeTiming(time_index);
   hypre_ClearTiming();
   


   /* dof_func for AMG ... */

   if (clp->num_functions > 1)
   {
      dof_func = NULL;
      if (clp->build_funcs_type == 1)
      {
	 BuildFuncsFromOneFile(argc, argv, clp->build_funcs_arg_index,
                               bHYPRE_parcsr_A, &dof_func);
      }
      else if (clp->build_funcs_type == 2)
      {
	 BuildFuncsFromFiles(argc, argv, clp->build_funcs_arg_index,
                             bHYPRE_parcsr_A, &dof_func);
      }
      else
      {
         BuildDefaultFuncs( clp, myid, local_num_rows, first_local_row,
                            &dof_func );
      }
   }
 
#ifdef HYPRE_IJMV_DEBUG
   ierr += IJMatrixVectorDebug(
      bmpicomm, local_num_cols, first_local_col, last_local_col,
      bHYPRE_parcsr_A, bHYPRE_b, bHYPRE_x );
#endif

   /*-----------------------------------------------------------
    * Solve the system using AMG
    *-----------------------------------------------------------*/

   if (clp->solver_id == 0)
   {
      ierr += Test_AMG( clp, bHYPRE_parcsr_A, bHYPRE_b, bHYPRE_x, dof_func,
                        mpi_comm, bmpicomm );
   }

   /*-----------------------------------------------------------
    * Solve the system using PCG 
    *-----------------------------------------------------------*/

   if (clp->solver_id == 1 || clp->solver_id == 2 || clp->solver_id == 8 || 
       clp->solver_id == 12 || clp->solver_id == 43)
      if ( clp->hpcg == 0 )
   {

      clp->ioutdat = 2;
      time_index = hypre_InitializeTiming("PCG Setup");
      hypre_BeginTiming(time_index);
 
      bHYPRE_op_A = bHYPRE_Operator__cast( bHYPRE_parcsr_A );
      bHYPRE_PCG = bHYPRE_PCG_Create( bmpicomm, bHYPRE_op_A );
      bHYPRE_Vector_b = bHYPRE_Vector__cast( bHYPRE_b );
      bHYPRE_Vector_x = bHYPRE_Vector__cast( bHYPRE_x );

      bHYPRE_PCG_SetIntParameter( bHYPRE_PCG, "MaxIterations", 500);
      bHYPRE_PCG_SetDoubleParameter( bHYPRE_PCG, "Tolerance", clp->tol);
      bHYPRE_PCG_SetIntParameter( bHYPRE_PCG, "TwoNorm", 1 );
      bHYPRE_PCG_SetIntParameter( bHYPRE_PCG, "RelChange", 0 );
      bHYPRE_PCG_SetIntParameter( bHYPRE_PCG, "PrintLevel", clp->ioutdat );

      if (clp->solver_id == 1)
      {
         /* use BoomerAMG as preconditioner */
         if (myid == 0) printf("Solver: AMG-PCG\n");
         ierr += PrecondAMG( clp, myid, bHYPRE_parcsr_A,
                             bHYPRE_Vector_b, bHYPRE_Vector_x, dof_func, bmpicomm,
                             &bHYPRE_SolverPC );
         ierr += bHYPRE_PCG_SetPreconditioner( bHYPRE_PCG, bHYPRE_SolverPC );
         ierr += bHYPRE_PCG_Setup( bHYPRE_PCG, bHYPRE_Vector_b, bHYPRE_Vector_x );

      }

      else if (clp->solver_id == 2)
      {
         /* use diagonal scaling as preconditioner */

         /* To call a bHYPRE solver:
            create, set comm, set operator, set other parameters,
            Setup (noop in this case), Apply */
         bHYPRE_ParCSRDiagScale = bHYPRE_ParCSRDiagScale_Create(
            bmpicomm, bHYPRE_parcsr_A );
         ierr += bHYPRE_ParCSRDiagScale_Setup( bHYPRE_ParCSRDiagScale,
                                               bHYPRE_Vector_b, bHYPRE_Vector_x );
         bHYPRE_SolverPC =
            bHYPRE_Solver__cast( bHYPRE_ParCSRDiagScale );
         ierr += bHYPRE_PCG_SetPreconditioner( bHYPRE_PCG, bHYPRE_SolverPC );
         ierr += bHYPRE_PCG_Setup( bHYPRE_PCG, bHYPRE_Vector_b, bHYPRE_Vector_x );

      }
      else if (clp->solver_id == 8)
      {
         /* use ParaSails preconditioner */
         if (myid == 0) printf("Solver: ParaSails-PCG\n");

         bHYPRE_ParaSails = bHYPRE_ParaSails_Create( bmpicomm, bHYPRE_parcsr_A );
         ierr += bHYPRE_ParaSails_SetDoubleParameter( bHYPRE_ParaSails, "Thresh",
                                                      clp->sai_threshold );
         ierr += bHYPRE_ParaSails_SetIntParameter( bHYPRE_ParaSails, "Nlevels",
                                                   clp->max_levels );
         ierr += bHYPRE_ParaSails_SetDoubleParameter( bHYPRE_ParaSails, "Filter",
                                                      clp->sai_filter );
         ierr += bHYPRE_ParaSails_SetIntParameter( bHYPRE_ParaSails, "Logging",
                                                   clp->ioutdat );
         hypre_assert( ierr==0 );
         bHYPRE_SolverPC = bHYPRE_Solver__cast( bHYPRE_ParaSails );
         ierr += bHYPRE_PCG_SetPreconditioner( bHYPRE_PCG, bHYPRE_SolverPC );
         ierr += bHYPRE_PCG_Setup( bHYPRE_PCG, bHYPRE_Vector_b, bHYPRE_Vector_x );

      }
      else if (clp->solver_id == 12)
      {
         /* use Schwarz preconditioner */
         if (myid == 0) printf("Solver: Schwarz-PCG\n");
         bHYPRE_Schwarz = bHYPRE_Schwarz_Create( bHYPRE_parcsr_A );
         ierr += bHYPRE_Schwarz_SetIntParameter(
            bHYPRE_Schwarz, "Variant", clp->variant );
         ierr += bHYPRE_Schwarz_SetIntParameter(
            bHYPRE_Schwarz, "Overlap", clp->overlap );
         ierr += bHYPRE_Schwarz_SetIntParameter(
            bHYPRE_Schwarz, "DomainType", clp->domain_type );
         ierr += bHYPRE_Schwarz_SetDoubleParameter(
            bHYPRE_Schwarz, "RelaxWeight", clp->schwarz_rlx_weight );
         hypre_assert( ierr==0 );
         bHYPRE_SolverPC = bHYPRE_Solver__cast( bHYPRE_Schwarz );
         ierr += bHYPRE_PCG_SetPreconditioner( bHYPRE_PCG, bHYPRE_SolverPC );
         ierr += bHYPRE_PCG_Setup( bHYPRE_PCG, bHYPRE_Vector_b, bHYPRE_Vector_x );
      }
      else if (clp->solver_id == 43)
      {
         /* use Euclid preconditioning */
         if (myid == 0) printf("Solver: Euclid-PCG\n");

         bHYPRE_Euclid = bHYPRE_Euclid_Create( bmpicomm, bHYPRE_parcsr_A );

         /* note: There are three three methods of setting run-time 
            parameters for Euclid: (see HYPRE_parcsr_ls.h); here
            we'll use what I think is simplest: let Euclid internally 
            parse the command line.
         */
         /*ierr += bHYPRE_Euclid_SetIntParameter( bHYPRE_Euclid, "-eu_stats", 1 );*/
         ierr += bHYPRE_Euclid_SetParameters( bHYPRE_Euclid, argc, argv );

         bHYPRE_SolverPC = bHYPRE_Solver__cast( bHYPRE_Euclid );
         ierr += bHYPRE_PCG_SetPreconditioner( bHYPRE_PCG, bHYPRE_SolverPC );
         ierr += bHYPRE_PCG_Setup( bHYPRE_PCG, bHYPRE_Vector_b, bHYPRE_Vector_x );
      }
 

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", mpi_comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
 
      time_index = hypre_InitializeTiming("PCG Solve");
      hypre_BeginTiming(time_index);

      ierr += bHYPRE_PCG_Apply( bHYPRE_PCG, bHYPRE_Vector_b, &bHYPRE_Vector_x );

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", mpi_comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      ierr += bHYPRE_PCG_GetIntValue( bHYPRE_PCG, "NumIterations",
                                      &num_iterations );
      ierr += bHYPRE_PCG_GetDoubleValue( bHYPRE_PCG, "Final Relative Residual Norm",
                                         &final_res_norm );

      bHYPRE_PCG_deleteRef( bHYPRE_PCG );
      if ( clp->solver_id == 1 )
      {
         bHYPRE_Solver_deleteRef( bHYPRE_SolverPC ); /* don't need if's if always do this */
      }
      else if ( clp->solver_id == 2 )
      {
         bHYPRE_ParCSRDiagScale_deleteRef( bHYPRE_ParCSRDiagScale );
      }
      else if (clp->solver_id == 8)
      {
         bHYPRE_ParaSails_deleteRef( bHYPRE_ParaSails );
      }
      else if (clp->solver_id == 12)
      {
         bHYPRE_Schwarz_deleteRef( bHYPRE_Schwarz );
      }
      else if (clp->solver_id == 43)
      {
         bHYPRE_Euclid_deleteRef( bHYPRE_Euclid );
      }

      if (myid == 0)
      {
         printf("\n");
         printf("Iterations = %d\n", num_iterations);
         printf("Final Relative Residual Norm = %e\n", final_res_norm);
         printf("\n");
      }
 
   }

   /*-----------------------------------------------------------
    * Solve the system using original PCG 
    *-----------------------------------------------------------*/

   if (clp->solver_id == 1 || clp->solver_id == 2 || clp->solver_id == 8 || 
       clp->solver_id == 12 || clp->solver_id == 43)
      if ( clp->hpcg != 0 )
   {

      time_index = hypre_InitializeTiming("HPCG Setup");
      hypre_BeginTiming(time_index);
 
      bHYPRE_HPCG = bHYPRE_HPCG_Create( bmpicomm );
      bHYPRE_Vector_b = bHYPRE_Vector__cast( bHYPRE_b );
      bHYPRE_Vector_x = bHYPRE_Vector__cast( bHYPRE_x );
      bHYPRE_Vector_Dot( bHYPRE_Vector_b, bHYPRE_Vector_b, &tmp );
      bHYPRE_Vector_Dot( bHYPRE_Vector_x, bHYPRE_Vector_x, &tmp );

      bHYPRE_op_A = bHYPRE_Operator__cast( bHYPRE_parcsr_A );
      bHYPRE_HPCG_SetOperator( bHYPRE_HPCG, bHYPRE_op_A );
      bHYPRE_HPCG_SetIntParameter( bHYPRE_HPCG, "MaxIterations", 500);
      bHYPRE_HPCG_SetDoubleParameter( bHYPRE_HPCG, "Tolerance", clp->tol);
      bHYPRE_HPCG_SetIntParameter( bHYPRE_HPCG, "TwoNorm", 1 );
      bHYPRE_HPCG_SetIntParameter( bHYPRE_HPCG, "RelChange", 0 );
      bHYPRE_HPCG_SetIntParameter( bHYPRE_HPCG, "PrintLevel", clp->ioutdat );

      if (clp->solver_id == 1)
      {
         /* use BoomerAMG as preconditioner */
	 clp->ioutdat = 1;
         if (myid == 0) printf("Solver: AMG-HPCG\n");
         bHYPRE_AMG = bHYPRE_BoomerAMG_Create( bmpicomm, bHYPRE_parcsr_A );
         bHYPRE_BoomerAMG_SetOperator( bHYPRE_AMG, bHYPRE_op_A );
         bHYPRE_BoomerAMG_SetDoubleParameter( bHYPRE_AMG, "Tolerance", clp->pc_tol);
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "CoarsenType",
                                        (clp->hybrid*clp->coarsen_type));
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "MeasureType",
                                           clp->measure_type);
         bHYPRE_BoomerAMG_SetDoubleParameter( bHYPRE_AMG, "StrongThreshold",
                                              clp->strong_threshold);
         bHYPRE_BoomerAMG_SetDoubleParameter( bHYPRE_AMG, "TruncFactor",
                                              clp->trunc_factor);
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "PrintLevel", clp->poutdat );
         bHYPRE_BoomerAMG_SetStringParameter( bHYPRE_AMG, "PrintFileName",
                                              "driver.out.log" );
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "MaxIter", 1 );
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "CycleType", clp->cycle_type );
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Cycle0NumSweeps",
                                           (clp->num_grid_sweeps)[0] );
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Cycle1NumSweeps",
                                           (clp->num_grid_sweeps)[1] );
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Cycle2NumSweeps",
                                           (clp->num_grid_sweeps)[2] );
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Cycle3NumSweeps",
                                           (clp->num_grid_sweeps)[3] );
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Cycle0RelaxType",
                                           (clp->grid_relax_type)[0] );
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Cycle1RelaxType",
                                           (clp->grid_relax_type)[1] );
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Cycle2RelaxType",
                                           (clp->grid_relax_type)[2] );
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Cycle3RelaxType",
                                           (clp->grid_relax_type)[3] );
         for ( i=0; i<clp->max_levels; ++i )
         {
            bHYPRE_BoomerAMG_SetLevelRelaxWt( bHYPRE_AMG, (clp->relax_weight)[i], i );
         }
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "SmoothType",
                                           clp->smooth_type );
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "SmoothNumSweeps",
                                           clp->smooth_num_sweep);

         dimsl[0] = 0;   dimsl[1] = 0;   dimsu[0] = 4;   dimsu[1] = 4;
         bHYPRE_grid_relax_points = sidl_int__array_createCol( 2, dimsl, dimsu );
         for ( i=0; i<4; ++i )
         {
            for ( j=0; j<(clp->num_grid_sweeps)[i]; ++j )
            {
               sidl_int__array_set2( bHYPRE_grid_relax_points, i, j,
                                     (clp->grid_relax_points)[i][j] );
            }
         }
         bHYPRE_BoomerAMG_SetIntArray2Parameter( bHYPRE_AMG, "GridRelaxPoints",
                                                 bHYPRE_grid_relax_points );
         sidl_int__array_deleteRef( bHYPRE_grid_relax_points );

         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "MaxLevels", clp->max_levels);
         bHYPRE_BoomerAMG_SetDoubleParameter( bHYPRE_AMG, "MaxRowSum",
                                              clp->max_row_sum);
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "NumFunctions",
                                           clp->num_functions);
         if (clp->num_functions > 1)
         {
            bHYPRE_BoomerAMG_SetIntArray1Parameter( bHYPRE_AMG, "DOFFunc",
                                                    dof_func, clp->num_functions );
         }
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Variant", clp->variant);
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Overlap", clp->overlap);
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "DomainType", clp->domain_type);
         bHYPRE_BoomerAMG_SetDoubleParameter( bHYPRE_AMG,
                                              "SchwarzRlxWeight",
                                              clp->schwarz_rlx_weight);

         bHYPRE_SolverPC = bHYPRE_Solver__cast( bHYPRE_AMG );
         ierr += bHYPRE_HPCG_SetPreconditioner( bHYPRE_HPCG, bHYPRE_SolverPC );
         ierr += bHYPRE_HPCG_Setup( bHYPRE_HPCG, bHYPRE_Vector_b, bHYPRE_Vector_x );

      }

      else if (clp->solver_id == 2)
      {
         /* use diagonal scaling as preconditioner */

         /* To call a bHYPRE solver:
            create, set comm, set operator, set other parameters,
            Setup (noop in this case), Apply */
         bHYPRE_ParCSRDiagScale = bHYPRE_ParCSRDiagScale_Create(
            bmpicomm, bHYPRE_parcsr_A );
         ierr += bHYPRE_ParCSRDiagScale_Setup( bHYPRE_ParCSRDiagScale,
                                               bHYPRE_Vector_b, bHYPRE_Vector_x );
         bHYPRE_SolverPC =
            bHYPRE_Solver__cast( bHYPRE_ParCSRDiagScale );
         ierr += bHYPRE_HPCG_SetPreconditioner( bHYPRE_HPCG, bHYPRE_SolverPC );
         ierr += bHYPRE_HPCG_Setup( bHYPRE_HPCG, bHYPRE_Vector_b, bHYPRE_Vector_x );

      }
      else if (clp->solver_id == 8)
      {
         /* use ParaSails preconditioner */
         if (myid == 0) printf("Solver: ParaSails-HPCG\n");

         bHYPRE_ParaSails = bHYPRE_ParaSails_Create( bmpicomm, bHYPRE_parcsr_A );
         ierr += bHYPRE_ParaSails_SetDoubleParameter( bHYPRE_ParaSails, "Thresh",
                                                      clp->sai_threshold );
         ierr += bHYPRE_ParaSails_SetIntParameter( bHYPRE_ParaSails, "Nlevels",
                                                   clp->max_levels );
         ierr += bHYPRE_ParaSails_SetDoubleParameter( bHYPRE_ParaSails, "Filter",
                                                      clp->sai_filter );
         ierr += bHYPRE_ParaSails_SetIntParameter( bHYPRE_ParaSails, "Logging",
                                                   clp->ioutdat );
         hypre_assert( ierr==0 );
         bHYPRE_SolverPC = bHYPRE_Solver__cast( bHYPRE_ParaSails );
         ierr += bHYPRE_HPCG_SetPreconditioner( bHYPRE_HPCG, bHYPRE_SolverPC );
         ierr += bHYPRE_HPCG_Setup( bHYPRE_HPCG, bHYPRE_Vector_b, bHYPRE_Vector_x );

      }
      else if (clp->solver_id == 12)
      {
#ifdef DO_THIS_LATER
         /* use Schwarz preconditioner */
         if (myid == 0) printf("Solver: Schwarz-HPCG\n");

	 HYPRE_SchwarzCreate(&pcg_precond);
	 HYPRE_SchwarzSetVariant(pcg_precond, clp->variant);
	 HYPRE_SchwarzSetOverlap(pcg_precond, clp->overlap);
	 HYPRE_SchwarzSetDomainType(pcg_precond, clp->domain_type);
         HYPRE_SchwarzSetRelaxWeight(pcg_precond, clp->schwarz_rlx_weight);

         HYPRE_HPCGSetPrecond(pcg_solver,
                             (HYPRE_PtrToSolverFcn) HYPRE_SchwarzSolve,
                             (HYPRE_PtrToSolverFcn) HYPRE_SchwarzSetup,
                             pcg_precond);
#endif  /*DO_THIS_LATER*/
      }
      else if (clp->solver_id == 43)
      {
#ifdef DO_THIS_LATER
         /* use Euclid preconditioning */
         if (myid == 0) printf("Solver: Euclid-HPCG\n");

         HYPRE_EuclidCreate(mpi_comm, &pcg_precond);

         /* note: There are three three methods of setting run-time 
            parameters for Euclid: (see HYPRE_parcsr_ls.h); here
            we'll use what I think is simplest: let Euclid internally 
            parse the command line.
         */   
         HYPRE_EuclidSetParams(pcg_precond, argc, argv);

         HYPRE_HPCGSetPrecond(pcg_solver,
                             (HYPRE_PtrToSolverFcn) HYPRE_EuclidSolve,
                             (HYPRE_PtrToSolverFcn) HYPRE_EuclidSetup,
                             pcg_precond);
#endif  /*DO_THIS_LATER*/
      }
 

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", mpi_comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
 
      time_index = hypre_InitializeTiming("HPCG Solve");
      hypre_BeginTiming(time_index);

      ierr += bHYPRE_HPCG_Apply( bHYPRE_HPCG, bHYPRE_Vector_b, &bHYPRE_Vector_x );

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", mpi_comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      ierr += bHYPRE_HPCG_GetIntValue( bHYPRE_HPCG, "NumIterations",
                                      &num_iterations );
      ierr += bHYPRE_HPCG_GetDoubleValue( bHYPRE_HPCG, "Final Relative Residual Norm",
                                         &final_res_norm );

      bHYPRE_HPCG_deleteRef( bHYPRE_HPCG );
      if ( clp->solver_id == 1 )
      {
         bHYPRE_BoomerAMG_deleteRef( bHYPRE_AMG );
      }
      else if ( clp->solver_id == 2 )
      {
         bHYPRE_ParCSRDiagScale_deleteRef( bHYPRE_ParCSRDiagScale );
      }
      else if (clp->solver_id == 8)
      {
         bHYPRE_ParaSails_deleteRef( bHYPRE_ParaSails );
      }
#ifdef DO_THIS_LATER
   else if (clp->solver_id == 12)
   {
   HYPRE_SchwarzDestroy(pcg_precond);
   }
   else if (clp->solver_id == 43)
   {
   / * HYPRE_EuclidPrintParams(pcg_precond); * /
   HYPRE_EuclidDestroy(pcg_precond);
   }
#endif  /*DO_THIS_LATER*/

      if (myid == 0)
      {
         printf("\n");
         printf("Iterations = %d\n", num_iterations);
         printf("Final Relative Residual Norm = %e\n", final_res_norm);
         printf("\n");
      }
 
   }

   /*-----------------------------------------------------------
    * Solve the system using GMRES, pure Babel-interface version
    *-----------------------------------------------------------*/

   if (clp->solver_id == 3 || clp->solver_id == 4 || clp->solver_id == 7 
       || clp->solver_id == 18 || clp->solver_id == 44)
      if ( clp->hpcg == 0 )
   {
      time_index = hypre_InitializeTiming("GMRES Setup");
      hypre_BeginTiming(time_index);

      bHYPRE_op_A = bHYPRE_Operator__cast( bHYPRE_parcsr_A );
      bHYPRE_GMRES = bHYPRE_GMRES_Create( bmpicomm, bHYPRE_op_A );
      bHYPRE_Vector_b = bHYPRE_Vector__cast( bHYPRE_b );
      bHYPRE_Vector_x = bHYPRE_Vector__cast( bHYPRE_x );

      ierr += bHYPRE_GMRES_SetIntParameter( bHYPRE_GMRES, "KDim", clp->k_dim );
      ierr += bHYPRE_GMRES_SetIntParameter( bHYPRE_GMRES, "MaxIter", 1000 );
      ierr += bHYPRE_GMRES_SetDoubleParameter( bHYPRE_GMRES, "Tol", clp->tol );
      ierr += bHYPRE_GMRES_SetIntParameter( bHYPRE_GMRES, "Logging", 1 );

      if (clp->solver_id == 3)
      {
         /* use BoomerAMG as preconditioner */
         if (myid == 0) printf("Solver: AMG-GMRES\n");
         ierr += PrecondAMG( clp, myid, bHYPRE_parcsr_A,
                             bHYPRE_Vector_b, bHYPRE_Vector_x, dof_func, bmpicomm,
                             &bHYPRE_SolverPC );
         ierr += bHYPRE_GMRES_SetPreconditioner( bHYPRE_GMRES, bHYPRE_SolverPC );
         ierr += bHYPRE_GMRES_Setup( bHYPRE_GMRES, bHYPRE_Vector_b, bHYPRE_Vector_x );
      }
      else if (clp->solver_id == 4)
      {
         /* use diagonal scaling as preconditioner */
         if (myid == 0) printf("Solver: DS-GMRES\n");

         bHYPRE_ParCSRDiagScale = bHYPRE_ParCSRDiagScale_Create(
            bmpicomm, bHYPRE_parcsr_A );
         ierr += bHYPRE_ParCSRDiagScale_Setup( bHYPRE_ParCSRDiagScale,
                                               bHYPRE_Vector_b, bHYPRE_Vector_x );
         bHYPRE_SolverPC =
            bHYPRE_Solver__cast( bHYPRE_ParCSRDiagScale );
         ierr += bHYPRE_GMRES_SetPreconditioner( bHYPRE_GMRES, bHYPRE_SolverPC );
         ierr += bHYPRE_GMRES_Setup( bHYPRE_GMRES, bHYPRE_Vector_b,
                                     bHYPRE_Vector_x );

      }
#ifdef DO_THIS_LATER
      else if (clp->solver_id == 7)
      {
         /* use PILUT as preconditioner */
         if (myid == 0) printf("Solver: PILUT-GMRES\n");

         ierr = HYPRE_ParCSRPilutCreate( mpi_comm, &pcg_precond ); 
         if (ierr) {
            printf("Error in ParPilutCreate\n");
         }

         HYPRE_GMRESSetPrecond(pcg_solver,
                               (HYPRE_PtrToSolverFcn) HYPRE_ParCSRPilutSolve,
                               (HYPRE_PtrToSolverFcn) HYPRE_ParCSRPilutSetup,
                               pcg_precond);

         if (clp->drop_tol >= 0 )
            HYPRE_ParCSRPilutSetDropTolerance( pcg_precond,
                                               clp->drop_tol );

         if (clp->nonzeros_to_keep >= 0 )
            HYPRE_ParCSRPilutSetFactorRowSize( pcg_precond,
                                               clp->nonzeros_to_keep );
      }
#endif  /*DO_THIS_LATER*/
      else if (clp->solver_id == 18)
      {
         /* use ParaSails preconditioner */
         if (myid == 0) printf("Solver: ParaSails-GMRES\n");

         bHYPRE_ParaSails = bHYPRE_ParaSails_Create( bmpicomm, bHYPRE_parcsr_A );
         ierr += bHYPRE_ParaSails_SetDoubleParameter( bHYPRE_ParaSails, "Thresh",
                                                      clp->sai_threshold );
         ierr += bHYPRE_ParaSails_SetIntParameter( bHYPRE_ParaSails, "Nlevels",
                                                   clp->max_levels );
         ierr += bHYPRE_ParaSails_SetDoubleParameter( bHYPRE_ParaSails, "Filter",
                                                      clp->sai_filter );
         ierr += bHYPRE_ParaSails_SetIntParameter( bHYPRE_ParaSails, "Logging",
                                                   clp->ioutdat );
         ierr += bHYPRE_ParaSails_SetIntParameter( bHYPRE_ParaSails, "Sym", 0 );
         hypre_assert( ierr==0 );
         bHYPRE_SolverPC = bHYPRE_Solver__cast( bHYPRE_ParaSails );
         ierr += bHYPRE_GMRES_SetPreconditioner( bHYPRE_GMRES, bHYPRE_SolverPC );
         ierr += bHYPRE_GMRES_Setup( bHYPRE_GMRES, bHYPRE_Vector_b,
                                     bHYPRE_Vector_x );

      }
#ifdef DO_THIS_LATER
      else if (clp->solver_id == 44)
      {
         /* use Euclid preconditioning */
         if (myid == 0) printf("Solver: Euclid-GMRES\n");

         HYPRE_EuclidCreate(mpi_comm, &pcg_precond);

         /* note: There are three three methods of setting run-time 
            parameters for Euclid: (see HYPRE_parcsr_ls.h); here
            we'll use what I think is simplest: let Euclid internally 
            parse the command line.
         */   
         HYPRE_EuclidSetParams(pcg_precond, argc, argv);

         HYPRE_GMRESSetPrecond (pcg_solver,
                                (HYPRE_PtrToSolverFcn) HYPRE_EuclidSolve,
                                (HYPRE_PtrToSolverFcn) HYPRE_EuclidSetup,
                                pcg_precond);
      }
#endif  /*DO_THIS_LATER*/
 
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", mpi_comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
   
      time_index = hypre_InitializeTiming("GMRES Solve");
      hypre_BeginTiming(time_index);

      ierr += bHYPRE_GMRES_Apply( bHYPRE_GMRES, bHYPRE_Vector_b, &bHYPRE_Vector_x );
 
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", mpi_comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      ierr += bHYPRE_GMRES_GetIntValue( bHYPRE_GMRES, "NumIterations",
                                        &num_iterations );
      ierr += bHYPRE_GMRES_GetDoubleValue( bHYPRE_GMRES, "Final Relative Residual Norm",
                                           &final_res_norm );
 
      bHYPRE_GMRES_deleteRef( bHYPRE_GMRES );
 
      if (clp->solver_id == 3)
      {
         bHYPRE_Solver_deleteRef( bHYPRE_SolverPC ); /* don't need if's if always do this */
      }
      else if ( clp->solver_id == 4 )
      {
         bHYPRE_ParCSRDiagScale_deleteRef( bHYPRE_ParCSRDiagScale );
      }
#ifdef DO_THIS_LATER
      if (clp->solver_id == 7)
      {
         HYPRE_ParCSRPilutDestroy(pcg_precond);
      }
#endif  /*DO_THIS_LATER*/
      else if (clp->solver_id == 18)
      {
	 bHYPRE_ParaSails_deleteRef ( bHYPRE_ParaSails );
      }
#ifdef DO_THIS_LATER
      else if (clp->solver_id == 44)
      {
         /* HYPRE_EuclidPrintParams(pcg_precond); */
         HYPRE_EuclidDestroy(pcg_precond);
      }
#endif  /*DO_THIS_LATER*/

      if (myid == 0)
      {
         printf("\n");
         printf("GMRES Iterations = %d\n", num_iterations);
         printf("Final GMRES Relative Residual Norm = %e\n", final_res_norm);
         printf("\n");
      }
   }

   /*-----------------------------------------------------------
    * Solve the system using GMRES, Babel interface working through the HYPRE interface
    *-----------------------------------------------------------*/

   if (clp->solver_id == 3 || clp->solver_id == 4 || clp->solver_id == 7 
       || clp->solver_id == 18 || clp->solver_id == 44)
      if ( clp->hpcg != 0 )
   {
      time_index = hypre_InitializeTiming("GMRES Setup");
      hypre_BeginTiming(time_index);

      bHYPRE_HGMRES = bHYPRE_HGMRES_Create( bmpicomm );
      bHYPRE_Vector_b = bHYPRE_Vector__cast( bHYPRE_b );
      bHYPRE_Vector_x = bHYPRE_Vector__cast( bHYPRE_x );
      bHYPRE_op_A = bHYPRE_Operator__cast( bHYPRE_parcsr_A );
      bHYPRE_HGMRES_SetOperator( bHYPRE_HGMRES, bHYPRE_op_A );

      ierr += bHYPRE_HGMRES_SetIntParameter( bHYPRE_HGMRES, "KDim", clp->k_dim );
      ierr += bHYPRE_HGMRES_SetIntParameter( bHYPRE_HGMRES, "MaxIter", 1000 );
      ierr += bHYPRE_HGMRES_SetDoubleParameter( bHYPRE_HGMRES, "Tol", clp->tol );
      ierr += bHYPRE_HGMRES_SetIntParameter( bHYPRE_HGMRES, "Logging", 1 );

      if (clp->solver_id == 3)
      {
         /* use BoomerAMG as preconditioner */
         if (myid == 0) printf("Solver: AMG-GMRES\n");

         bHYPRE_AMG = bHYPRE_BoomerAMG_Create( bmpicomm, bHYPRE_parcsr_A );
         bHYPRE_BoomerAMG_SetDoubleParameter( bHYPRE_AMG, "Tolerance", clp->pc_tol);
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "CoarsenType",
                                        (clp->hybrid*clp->coarsen_type));
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "MeasureType",
                                           clp->measure_type);
         bHYPRE_BoomerAMG_SetDoubleParameter( bHYPRE_AMG, "StrongThreshold",
                                              clp->strong_threshold);
         bHYPRE_BoomerAMG_SetDoubleParameter( bHYPRE_AMG, "TruncFactor",
                                              clp->trunc_factor);
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "PrintLevel", clp->poutdat );
         bHYPRE_BoomerAMG_SetStringParameter( bHYPRE_AMG, "PrintFileName",
                                              "driver.out.log" );
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "MaxIter", 1 );
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "CycleType", clp->cycle_type );
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Cycle0NumSweeps",
                                           (clp->num_grid_sweeps)[0] );
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Cycle1NumSweeps",
                                           (clp->num_grid_sweeps)[1] );
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Cycle2NumSweeps",
                                           (clp->num_grid_sweeps)[2] );
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Cycle3NumSweeps",
                                           (clp->num_grid_sweeps)[3] );
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Cycle0RelaxType",
                                           (clp->grid_relax_type)[0] );
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Cycle1RelaxType",
                                           (clp->grid_relax_type)[1] );
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Cycle2RelaxType",
                                           (clp->grid_relax_type)[2] );
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Cycle3RelaxType",
                                           (clp->grid_relax_type)[3] );
         for ( i=0; i<clp->max_levels; ++i )
         {
            bHYPRE_BoomerAMG_SetLevelRelaxWt( bHYPRE_AMG, (clp->relax_weight)[i], i );
         }
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "SmoothType",
                                           clp->smooth_type );
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "SmoothNumSweeps",
                                           clp->smooth_num_sweep);

         dimsl[0] = 0;   dimsl[1] = 0;   dimsu[0] = 4;   dimsu[1] = 4;
         bHYPRE_grid_relax_points = sidl_int__array_createCol( 2, dimsl, dimsu );
         for ( i=0; i<4; ++i )
         {
            for ( j=0; j<(clp->num_grid_sweeps)[i]; ++j )
            {
               sidl_int__array_set2( bHYPRE_grid_relax_points, i, j,
                                     (clp->grid_relax_points)[i][j] );
            }
         }
         bHYPRE_BoomerAMG_SetIntArray2Parameter( bHYPRE_AMG, "GridRelaxPoints",
                                                 bHYPRE_grid_relax_points );
         sidl_int__array_deleteRef( bHYPRE_grid_relax_points );

         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "MaxLevels", clp->max_levels);
         bHYPRE_BoomerAMG_SetDoubleParameter( bHYPRE_AMG, "MaxRowSum",
                                              clp->max_row_sum);
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "NumFunctions",
                                           clp->num_functions);
         if (clp->num_functions > 1)
         {
            bHYPRE_BoomerAMG_SetIntArray1Parameter( bHYPRE_AMG, "DOFFunc",
                                                    dof_func, clp->num_functions );
         }
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Variant", clp->variant);
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Overlap", clp->overlap);
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "DomainType", clp->domain_type);

         bHYPRE_SolverPC = bHYPRE_Solver__cast( bHYPRE_AMG );
         ierr += bHYPRE_HGMRES_SetPreconditioner( bHYPRE_HGMRES, bHYPRE_SolverPC );
         ierr += bHYPRE_HGMRES_Setup( bHYPRE_HGMRES, bHYPRE_Vector_b,
                                     bHYPRE_Vector_x );
      }
      else if (clp->solver_id == 4)
      {
         /* use diagonal scaling as preconditioner */
         if (myid == 0) printf("Solver: DS-GMRES\n");

         bHYPRE_ParCSRDiagScale = bHYPRE_ParCSRDiagScale_Create(
            bmpicomm, bHYPRE_parcsr_A );
         ierr += bHYPRE_ParCSRDiagScale_Setup( bHYPRE_ParCSRDiagScale,
                                               bHYPRE_Vector_b, bHYPRE_Vector_x );
         bHYPRE_SolverPC =
            bHYPRE_Solver__cast( bHYPRE_ParCSRDiagScale );
         ierr += bHYPRE_HGMRES_SetPreconditioner( bHYPRE_HGMRES, bHYPRE_SolverPC );
         ierr += bHYPRE_HGMRES_Setup( bHYPRE_HGMRES, bHYPRE_Vector_b,
                                     bHYPRE_Vector_x );

      }
#ifdef DO_THIS_LATER
      else if (clp->solver_id == 7)
      {
         /* use PILUT as preconditioner */
         if (myid == 0) printf("Solver: PILUT-GMRES\n");

         ierr = HYPRE_ParCSRPilutCreate( mpi_comm, &pcg_precond ); 
         if (ierr) {
            printf("Error in ParPilutCreate\n");
         }

         HYPRE_GMRESSetPrecond(pcg_solver,
                               (HYPRE_PtrToSolverFcn) HYPRE_ParCSRPilutSolve,
                               (HYPRE_PtrToSolverFcn) HYPRE_ParCSRPilutSetup,
                               pcg_precond);

         if (clp->drop_tol >= 0 )
            HYPRE_ParCSRPilutSetDropTolerance( pcg_precond,
                                               clp->drop_tol );

         if (clp->nonzeros_to_keep >= 0 )
            HYPRE_ParCSRPilutSetFactorRowSize( pcg_precond,
                                               clp->nonzeros_to_keep );
      }
#endif  /*DO_THIS_LATER*/
      else if (clp->solver_id == 18)
      {
         /* use ParaSails preconditioner */
         if (myid == 0) printf("Solver: ParaSails-GMRES\n");

         bHYPRE_ParaSails = bHYPRE_ParaSails_Create( bmpicomm, bHYPRE_parcsr_A );
         ierr += bHYPRE_ParaSails_SetDoubleParameter( bHYPRE_ParaSails, "Thresh",
                                                      clp->sai_threshold );
         ierr += bHYPRE_ParaSails_SetIntParameter( bHYPRE_ParaSails, "Nlevels",
                                                   clp->max_levels );
         ierr += bHYPRE_ParaSails_SetDoubleParameter( bHYPRE_ParaSails, "Filter",
                                                      clp->sai_filter );
         ierr += bHYPRE_ParaSails_SetIntParameter( bHYPRE_ParaSails, "Logging",
                                                   clp->ioutdat );
         ierr += bHYPRE_ParaSails_SetIntParameter( bHYPRE_ParaSails, "Sym", 0 );
         hypre_assert( ierr==0 );
         bHYPRE_SolverPC = bHYPRE_Solver__cast( bHYPRE_ParaSails );
         ierr += bHYPRE_HGMRES_SetPreconditioner( bHYPRE_HGMRES, bHYPRE_SolverPC );
         ierr += bHYPRE_HGMRES_Setup( bHYPRE_HGMRES, bHYPRE_Vector_b,
                                     bHYPRE_Vector_x );

      }
#ifdef DO_THIS_LATER
      else if (clp->solver_id == 44)
      {
         /* use Euclid preconditioning */
         if (myid == 0) printf("Solver: Euclid-GMRES\n");

         HYPRE_EuclidCreate(mpi_comm, &pcg_precond);

         /* note: There are three three methods of setting run-time 
            parameters for Euclid: (see HYPRE_parcsr_ls.h); here
            we'll use what I think is simplest: let Euclid internally 
            parse the command line.
         */   
         HYPRE_EuclidSetParams(pcg_precond, argc, argv);

         HYPRE_GMRESSetPrecond (pcg_solver,
                                (HYPRE_PtrToSolverFcn) HYPRE_EuclidSolve,
                                (HYPRE_PtrToSolverFcn) HYPRE_EuclidSetup,
                                pcg_precond);
      }
#endif  /*DO_THIS_LATER*/
 
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", mpi_comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
   
      time_index = hypre_InitializeTiming("GMRES Solve");
      hypre_BeginTiming(time_index);

      ierr += bHYPRE_HGMRES_Apply( bHYPRE_HGMRES, bHYPRE_Vector_b, &bHYPRE_Vector_x );
 
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", mpi_comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      ierr += bHYPRE_HGMRES_GetIntValue( bHYPRE_HGMRES, "NumIterations",
                                        &num_iterations );
      ierr += bHYPRE_HGMRES_GetDoubleValue( bHYPRE_HGMRES, "Final Relative Residual Norm",
                                           &final_res_norm );
 
      bHYPRE_HGMRES_deleteRef( bHYPRE_HGMRES );
 
      if (clp->solver_id == 3)
      {
         bHYPRE_BoomerAMG_deleteRef( bHYPRE_AMG );
      }
      else if ( clp->solver_id == 2 )
      {
         bHYPRE_ParCSRDiagScale_deleteRef( bHYPRE_ParCSRDiagScale );
      }
#ifdef DO_THIS_LATER
      if (clp->solver_id == 7)
      {
         HYPRE_ParCSRPilutDestroy(pcg_precond);
      }
#endif  /*DO_THIS_LATER*/
      else if (clp->solver_id == 18)
      {
	 bHYPRE_ParaSails_deleteRef ( bHYPRE_ParaSails );
      }
#ifdef DO_THIS_LATER
      else if (clp->solver_id == 44)
      {
         /* HYPRE_EuclidPrintParams(pcg_precond); */
         HYPRE_EuclidDestroy(pcg_precond);
      }
#endif  /*DO_THIS_LATER*/

      if (myid == 0)
      {
         printf("\n");
         printf("GMRES Iterations = %d\n", num_iterations);
         printf("Final GMRES Relative Residual Norm = %e\n", final_res_norm);
         printf("\n");
      }
   }

   /*-----------------------------------------------------------
    * Solve the system using BiCGSTAB 
    *-----------------------------------------------------------*/

   if (clp->solver_id == 9 || clp->solver_id == 10 || clp->solver_id == 11 || clp->solver_id == 45)
   {
      time_index = hypre_InitializeTiming("BiCGSTAB Setup");
      hypre_BeginTiming(time_index);
 
      bHYPRE_op_A = bHYPRE_Operator__cast( bHYPRE_parcsr_A );
      bHYPRE_BiCGSTAB = bHYPRE_BiCGSTAB_Create( bmpicomm, bHYPRE_op_A );
      bHYPRE_Vector_b = bHYPRE_Vector__cast( bHYPRE_b );
      bHYPRE_Vector_x = bHYPRE_Vector__cast( bHYPRE_x );

      bHYPRE_BiCGSTAB_SetIntParameter( bHYPRE_BiCGSTAB, "MaxIterations", 500 );
      bHYPRE_BiCGSTAB_SetDoubleParameter( bHYPRE_BiCGSTAB, "Tolerance", clp->tol );
      bHYPRE_BiCGSTAB_SetIntParameter( bHYPRE_BiCGSTAB, "Logging", 1 );
      bHYPRE_BiCGSTAB_SetIntParameter( bHYPRE_BiCGSTAB, "PrintLevel", clp->ioutdat );
 
      if (clp->solver_id == 9)
      {
         /* use BoomerAMG as preconditioner */
         if (myid == 0) printf("Solver: AMG-BiCGSTAB\n");
         ierr += PrecondAMG( clp, myid, bHYPRE_parcsr_A,
                             bHYPRE_Vector_b, bHYPRE_Vector_x, dof_func, bmpicomm,
                             &bHYPRE_SolverPC );
         ierr += bHYPRE_BiCGSTAB_SetPreconditioner(
            bHYPRE_BiCGSTAB, bHYPRE_SolverPC );
         ierr += bHYPRE_BiCGSTAB_Setup(
            bHYPRE_BiCGSTAB, bHYPRE_Vector_b, bHYPRE_Vector_x );
      }
      else if (clp->solver_id == 10)
      {
         /* use diagonal scaling as preconditioner */
         if (myid == 0) printf("Solver: DS-BiCGSTAB\n");

         bHYPRE_ParCSRDiagScale = bHYPRE_ParCSRDiagScale_Create(
            bmpicomm, bHYPRE_parcsr_A );
         ierr += bHYPRE_ParCSRDiagScale_Setup( bHYPRE_ParCSRDiagScale,
                                               bHYPRE_Vector_b, bHYPRE_Vector_x );
         bHYPRE_SolverPC =
            bHYPRE_Solver__cast( bHYPRE_ParCSRDiagScale );
         ierr += bHYPRE_BiCGSTAB_SetPreconditioner(
            bHYPRE_BiCGSTAB, bHYPRE_SolverPC );
         ierr += bHYPRE_BiCGSTAB_Setup(
            bHYPRE_BiCGSTAB, bHYPRE_Vector_b, bHYPRE_Vector_x );

      }
      else if (clp->solver_id == 11)
      {
         hypre_assert( "solver 11 not implemented"==0 );
#ifdef DO_THIS_LATER
         /* use PILUT as preconditioner */
         if (myid == 0) printf("Solver: PILUT-BiCGSTAB\n");

         ierr = HYPRE_ParCSRPilutCreate( mpi_comm, &pcg_precond ); 
         if (ierr) {
            printf("Error in ParPilutCreate\n");
         }

         HYPRE_BiCGSTABSetPrecond(pcg_solver,
                                  (HYPRE_PtrToSolverFcn) HYPRE_ParCSRPilutSolve,
                                  (HYPRE_PtrToSolverFcn) HYPRE_ParCSRPilutSetup,
                                  pcg_precond);

         if (clp->drop_tol >= 0 )
            HYPRE_ParCSRPilutSetDropTolerance( pcg_precond,
                                               clp->drop_tol );

         if (clp->nonzeros_to_keep >= 0 )
            HYPRE_ParCSRPilutSetFactorRowSize( pcg_precond,
                                               clp->nonzeros_to_keep );
#endif  /*DO_THIS_LATER*/
      }
      else if (clp->solver_id == 45)
      {
         hypre_assert( "solver 45 not implemented"==0 );
#ifdef DO_THIS_LATER
         /* use Euclid preconditioning */
         if (myid == 0) printf("Solver: Euclid-BICGSTAB\n");

         HYPRE_EuclidCreate(mpi_comm, &pcg_precond);

         /* note: There are three three methods of setting run-time 
            parameters for Euclid: (see HYPRE_parcsr_ls.h); here
            we'll use what I think is simplest: let Euclid internally 
            parse the command line.
         */   
         HYPRE_EuclidSetParams(pcg_precond, argc, argv);

         HYPRE_BiCGSTABSetPrecond(pcg_solver,
                                  (HYPRE_PtrToSolverFcn) HYPRE_EuclidSolve,
                                  (HYPRE_PtrToSolverFcn) HYPRE_EuclidSetup,
                                  pcg_precond);
#endif  /*DO_THIS_LATER*/
      }
 
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", mpi_comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
   
      time_index = hypre_InitializeTiming("BiCGSTAB Solve");
      hypre_BeginTiming(time_index);
 
      ierr += bHYPRE_BiCGSTAB_Apply(
         bHYPRE_BiCGSTAB, bHYPRE_Vector_b, &bHYPRE_Vector_x );
 
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", mpi_comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
 
      ierr += bHYPRE_BiCGSTAB_GetIntValue( bHYPRE_BiCGSTAB, "NumIterations",
                                           &num_iterations );
      ierr += bHYPRE_BiCGSTAB_GetDoubleValue(
         bHYPRE_BiCGSTAB, "Final Relative Residual Norm", &final_res_norm );

      bHYPRE_BiCGSTAB_deleteRef( bHYPRE_BiCGSTAB );
 
      if (clp->solver_id == 9)
      {
         bHYPRE_Solver_deleteRef( bHYPRE_SolverPC ); /* don't need if's if always do this */
      }
      else if (clp->solver_id == 10)
      {
         bHYPRE_ParCSRDiagScale_deleteRef( bHYPRE_ParCSRDiagScale );
      }
      else if (clp->solver_id == 11)
      {
#ifdef DO_THIS_LATER
         HYPRE_ParCSRPilutDestroy(pcg_precond);
#endif  /*DO_THIS_LATER*/
      }
      else if (clp->solver_id == 45)
      {
#ifdef DO_THIS_LATER
         /* HYPRE_EuclidPrintParams(pcg_precond); */
         HYPRE_EuclidDestroy(pcg_precond);
#endif  /*DO_THIS_LATER*/
      }

      if (myid == 0)
      {
         printf("\n");
         printf("BiCGSTAB Iterations = %d\n", num_iterations);
         printf("Final BiCGSTAB Relative Residual Norm = %e\n", final_res_norm);
         printf("\n");
      }
   }
   /*-----------------------------------------------------------
    * Solve the system using CGNR 
    *-----------------------------------------------------------*/

   if (clp->solver_id == 5 || clp->solver_id == 6)
   {
      time_index = hypre_InitializeTiming("CGNR Setup");
      hypre_BeginTiming(time_index);

      bHYPRE_op_A = bHYPRE_Operator__cast( bHYPRE_parcsr_A );
      bHYPRE_CGNR = bHYPRE_CGNR_Create( bmpicomm, bHYPRE_op_A );
      bHYPRE_Vector_b = bHYPRE_Vector__cast( bHYPRE_b );
      bHYPRE_Vector_x = bHYPRE_Vector__cast( bHYPRE_x );

      bHYPRE_CGNR_SetIntParameter( bHYPRE_CGNR, "MaxIterations", 1000 );
      bHYPRE_CGNR_SetDoubleParameter( bHYPRE_CGNR, "Tolerance", clp->tol );
      bHYPRE_CGNR_SetLogging( bHYPRE_CGNR, 2 );
      bHYPRE_CGNR_SetIntParameter( bHYPRE_CGNR, "PrintLevel", clp->ioutdat );
 
      if (clp->solver_id == 5)
      {
         /* use BoomerAMG as preconditioner */
         if (myid == 0) printf("Solver: AMG-CGNR\n");

         ierr += PrecondAMG( clp, myid, bHYPRE_parcsr_A,
                             bHYPRE_Vector_b, bHYPRE_Vector_x, dof_func, bmpicomm,
                             &bHYPRE_SolverPC );
         ierr += bHYPRE_CGNR_SetPreconditioner( bHYPRE_CGNR, bHYPRE_SolverPC );
         ierr += bHYPRE_CGNR_Setup( bHYPRE_CGNR, bHYPRE_Vector_b, bHYPRE_Vector_x );

      }
      else if (clp->solver_id == 6)
      {
         /* use diagonal scaling as preconditioner */
         if (myid == 0) printf("Solver: DS-CGNR\n");
         bHYPRE_ParCSRDiagScale = bHYPRE_ParCSRDiagScale_Create( bmpicomm, bHYPRE_parcsr_A );
         ierr += bHYPRE_ParCSRDiagScale_Setup( bHYPRE_ParCSRDiagScale,
                                               bHYPRE_Vector_b, bHYPRE_Vector_x );
         bHYPRE_SolverPC =
            bHYPRE_Solver__cast( bHYPRE_ParCSRDiagScale );
         ierr += bHYPRE_CGNR_SetPreconditioner( bHYPRE_CGNR, bHYPRE_SolverPC );
         ierr += bHYPRE_CGNR_Setup( bHYPRE_CGNR, bHYPRE_Vector_b, bHYPRE_Vector_x );

      }
 
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", mpi_comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
   
      time_index = hypre_InitializeTiming("CGNR Solve");
      hypre_BeginTiming(time_index);
 
      ierr += bHYPRE_CGNR_Apply( bHYPRE_CGNR, bHYPRE_Vector_b, &bHYPRE_Vector_x );
 
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", mpi_comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
 
      ierr += bHYPRE_CGNR_GetIntValue( bHYPRE_CGNR, "NumIterations",
                                       &num_iterations );
      ierr += bHYPRE_CGNR_GetDoubleValue( bHYPRE_CGNR, "Final Relative Residual Norm",
                                          &final_res_norm );

      bHYPRE_CGNR_deleteRef( bHYPRE_CGNR );
 
      if (clp->solver_id == 5)
      {
         bHYPRE_Solver_deleteRef( bHYPRE_SolverPC ); /* don't need if's if always do this */
      }
      else if ( clp->solver_id == 6 )
      {
         bHYPRE_ParCSRDiagScale_deleteRef( bHYPRE_ParCSRDiagScale );
      }
      if (myid == 0)
      {
         printf("\n");
         printf("Iterations = %d\n", num_iterations);
         printf("Final Relative Residual Norm = %e\n", final_res_norm);
         printf("\n");
      }
   }

   /*-----------------------------------------------------------
    * Print the solution and other info
    *-----------------------------------------------------------*/

   bHYPRE_IJParCSRVector_Print( bHYPRE_b, "driver.out.b");
   bHYPRE_IJParCSRVector_Print( bHYPRE_x, "driver.out.x");

   /*-----------------------------------------------------------
    * Finalize things
    *-----------------------------------------------------------*/

   /* Programming note: some or all of these sidl array objects, e.g.
    * bHYPRE_num_grid_sweeps, contain data which have been incorporated
    * into bHYPRE_AMG (the sidl objects themselves were just temporary
    * carriers for the data).  The Babel deleteRef doesn't seem to be able
    * to handle doing it twice, so some are commented-out.
    */

   bHYPRE_IJParCSRMatrix_deleteRef( bHYPRE_parcsr_A );
   bHYPRE_IJParCSRVector_deleteRef( bHYPRE_b );
   bHYPRE_IJParCSRVector_deleteRef( bHYPRE_x );

   /* These can be (and do get) freed by HYPRE programs, but not always.
      All are obsolete, better to not pass them in. */
   if ( (clp->num_grid_sweeps) )
      hypre_TFree((clp->num_grid_sweeps));
   if ( (clp->relax_weight) )
      hypre_TFree((clp->relax_weight));
   if ( (clp->grid_relax_points) ) {
      for ( i=0; i<4; ++i )
      {
         if ( (clp->grid_relax_points)[i] )
         {
            hypre_TFree( (clp->grid_relax_points)[i] );
         }
      }
      hypre_TFree( (clp->grid_relax_points) );
   }
   if ( (clp->grid_relax_type) )
      hypre_TFree( (clp->grid_relax_type) );

   bHYPRE_MPICommunicator_deleteRef( bmpicomm );
   MPI_Finalize();

   return (0);
}

/*----------------------------------------------------------------------
 * Build matrix from file. Expects three files on each processor.
 * filename.D.n contains the diagonal part, filename.O.n contains
 * the offdiagonal part and filename.INFO.n contains global row
 * and column numbers, number of columns of offdiagonal matrix
 * and the mapping of offdiagonal column numbers to global column numbers.
 * Parameters given in command line.
 *----------------------------------------------------------------------*/

int
BuildParFromFile( int                  argc,
                  char                *argv[],
                  int                  arg_index,
                  HYPRE_ParCSRMatrix  *A_ptr     )
{
   char               *filename;

   HYPRE_ParCSRMatrix A;

   int                 myid;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   MPI_Comm_rank(MPI_COMM_WORLD, &myid );

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/

   if (arg_index < argc)
   {
      filename = argv[arg_index];
   }
   else
   {
      printf("Error: No filename specified \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/
 
   if (myid == 0)
   {
      printf("  FromFile: %s\n", filename);
   }

   /*-----------------------------------------------------------
    * Generate the matrix 
    *-----------------------------------------------------------*/
 
   HYPRE_ParCSRMatrixRead(MPI_COMM_WORLD, filename,&A);

   *A_ptr = A;

   return (0);
}

/*----------------------------------------------------------------------
 * Build standard 7-point laplacian in 3D with grid and anisotropy.
 * Parameters given in command line.
 *----------------------------------------------------------------------*/

int
bBuildParLaplacian( int                  argc,
                    char                *argv[],
                    int                  arg_index,
                    bHYPRE_MPICommunicator bmpi_comm,
                    bHYPRE_IJParCSRMatrix  *bA_ptr     )
{
   int                 nx, ny, nz;
   int                 P, Q, R;
   double              cx, cy, cz;

   bHYPRE_IJParCSRMatrix  bA;

   int                 num_procs, myid;
   int                 p, q, r;
   double             *values;
   int                 nvalues = 4;
   MPI_Comm mpi_comm = bHYPRE_MPICommunicator__get_data(bmpi_comm)->mpi_comm;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   MPI_Comm_size(mpi_comm, &num_procs );
   MPI_Comm_rank(mpi_comm, &myid );

   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/
 
   nx = 10;
   ny = 10;
   nz = 10;

   P  = 1;
   Q  = num_procs;
   R  = 1;

   cx = 1.;
   cy = 1.;
   cz = 1.;

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/
   arg_index = 0;
   while (arg_index < argc)
   {
      if ( strcmp(argv[arg_index], "-n") == 0 )
      {
         arg_index++;
         nx = atoi(argv[arg_index++]);
         ny = atoi(argv[arg_index++]);
         nz = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-P") == 0 )
      {
         arg_index++;
         P  = atoi(argv[arg_index++]);
         Q  = atoi(argv[arg_index++]);
         R  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-c") == 0 )
      {
         arg_index++;
         cx = atof(argv[arg_index++]);
         cy = atof(argv[arg_index++]);
         cz = atof(argv[arg_index++]);
      }
      else
      {
         arg_index++;
      }
   }

   /*-----------------------------------------------------------
    * Check a few things
    *-----------------------------------------------------------*/

   if ((P*Q*R) != num_procs)
   {
      printf("Error: Invalid number of processors or processor topology \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/
 
   if (myid == 0)
   {
      printf("  Laplacian:\n");
      printf("    (nx, ny, nz) = (%d, %d, %d)\n", nx, ny, nz);
      printf("    (Px, Py, Pz) = (%d, %d, %d)\n", P,  Q,  R);
      printf("    (cx, cy, cz) = (%f, %f, %f)\n\n", cx, cy, cz);
   }

   /*-----------------------------------------------------------
    * Set up the grid structure
    *-----------------------------------------------------------*/

   /* compute p,q,r from P,Q,R and myid */
   p = myid % P;
   q = (( myid - p)/P) % Q;
   r = ( myid - p - P*q)/( P*Q );

   /*-----------------------------------------------------------
    * Generate the matrix 
    *-----------------------------------------------------------*/
 
   values = hypre_CTAlloc(double, 4);

   values[1] = -cx;
   values[2] = -cy;
   values[3] = -cz;

   values[0] = 0.;
   if (nx > 1)
   {
      values[0] += 2.0*cx;
   }
   if (ny > 1)
   {
      values[0] += 2.0*cy;
   }
   if (nz > 1)
   {
      values[0] += 2.0*cz;
   }

   bA = bHYPRE_IJParCSRMatrix_GenerateLaplacian(
      bmpi_comm, nx, ny, nz, P, Q, R, p, q, r,
      values, nvalues, 7 );

   hypre_TFree(values);

   *bA_ptr = bA;

   return (0);
}

/* non-Babel version used only for timings... */
int
BuildParLaplacian( int                  argc,
                   char                *argv[],
                   int                  arg_index,
                   HYPRE_ParCSRMatrix  *A_ptr     )
{
   int                 nx, ny, nz;
   int                 P, Q, R;
   double              cx, cy, cz;

   HYPRE_ParCSRMatrix  A;

   int                 num_procs, myid;
   int                 p, q, r;
   double             *values;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   MPI_Comm_size(MPI_COMM_WORLD, &num_procs );
   MPI_Comm_rank(MPI_COMM_WORLD, &myid );

   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/
 
   nx = 10;
   ny = 10;
   nz = 10;

   P  = 1;
   Q  = num_procs;
   R  = 1;

   cx = 1.;
   cy = 1.;
   cz = 1.;

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/
   arg_index = 0;
   while (arg_index < argc)
   {
      if ( strcmp(argv[arg_index], "-n") == 0 )
      {
         arg_index++;
         nx = atoi(argv[arg_index++]);
         ny = atoi(argv[arg_index++]);
         nz = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-P") == 0 )
      {
         arg_index++;
         P  = atoi(argv[arg_index++]);
         Q  = atoi(argv[arg_index++]);
         R  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-c") == 0 )
      {
         arg_index++;
         cx = atof(argv[arg_index++]);
         cy = atof(argv[arg_index++]);
         cz = atof(argv[arg_index++]);
      }
      else
      {
         arg_index++;
      }
   }

   /*-----------------------------------------------------------
    * Check a few things
    *-----------------------------------------------------------*/

   if ((P*Q*R) != num_procs)
   {
      printf("Error: Invalid number of processors or processor topology \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/
 
   if (myid == 0)
   {
      printf("  Laplacian:\n");
      printf("    (nx, ny, nz) = (%d, %d, %d)\n", nx, ny, nz);
      printf("    (Px, Py, Pz) = (%d, %d, %d)\n", P,  Q,  R);
      printf("    (cx, cy, cz) = (%f, %f, %f)\n\n", cx, cy, cz);
   }

   /*-----------------------------------------------------------
    * Set up the grid structure
    *-----------------------------------------------------------*/

   /* compute p,q,r from P,Q,R and myid */
   p = myid % P;
   q = (( myid - p)/P) % Q;
   r = ( myid - p - P*q)/( P*Q );

   /*-----------------------------------------------------------
    * Generate the matrix 
    *-----------------------------------------------------------*/
 
   values = hypre_CTAlloc(double, 4);

   values[1] = -cx;
   values[2] = -cy;
   values[3] = -cz;

   values[0] = 0.;
   if (nx > 1)
   {
      values[0] += 2.0*cx;
   }
   if (ny > 1)
   {
      values[0] += 2.0*cy;
   }
   if (nz > 1)
   {
      values[0] += 2.0*cz;
   }

   A = (HYPRE_ParCSRMatrix) GenerateLaplacian(MPI_COMM_WORLD, 
		nx, ny, nz, P, Q, R, p, q, r, values);

   hypre_TFree(values);

   *A_ptr = A;

   return (0);
}

/*----------------------------------------------------------------------
 * Build standard 7-point convection-diffusion operator 
 * Parameters given in command line.
 * Operator:
 *
 *  -cx Dxx - cy Dyy - cz Dzz + ax Dx + ay Dy + az Dz = f
 *
 *----------------------------------------------------------------------*/

int
BuildParDifConv( int                  argc,
                 char                *argv[],
                 int                  arg_index,
                 HYPRE_ParCSRMatrix  *A_ptr     )
{
   int                 nx, ny, nz;
   int                 P, Q, R;
   double              cx, cy, cz;
   double              ax, ay, az;
   double              hinx,hiny,hinz;

   HYPRE_ParCSRMatrix  A;

   int                 num_procs, myid;
   int                 p, q, r;
   double             *values;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   MPI_Comm_size(MPI_COMM_WORLD, &num_procs );
   MPI_Comm_rank(MPI_COMM_WORLD, &myid );

   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/
 
   nx = 10;
   ny = 10;
   nz = 10;

   hinx = 1./(nx+1);
   hiny = 1./(ny+1);
   hinz = 1./(nz+1);

   P  = 1;
   Q  = num_procs;
   R  = 1;

   cx = 1.;
   cy = 1.;
   cz = 1.;

   ax = 1.;
   ay = 1.;
   az = 1.;

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/
   arg_index = 0;
   while (arg_index < argc)
   {
      if ( strcmp(argv[arg_index], "-n") == 0 )
      {
         arg_index++;
         nx = atoi(argv[arg_index++]);
         ny = atoi(argv[arg_index++]);
         nz = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-P") == 0 )
      {
         arg_index++;
         P  = atoi(argv[arg_index++]);
         Q  = atoi(argv[arg_index++]);
         R  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-c") == 0 )
      {
         arg_index++;
         cx = atof(argv[arg_index++]);
         cy = atof(argv[arg_index++]);
         cz = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-a") == 0 )
      {
         arg_index++;
         ax = atof(argv[arg_index++]);
         ay = atof(argv[arg_index++]);
         az = atof(argv[arg_index++]);
      }
      else
      {
         arg_index++;
      }
   }

   /*-----------------------------------------------------------
    * Check a few things
    *-----------------------------------------------------------*/

   if ((P*Q*R) != num_procs)
   {
      printf("Error: Invalid number of processors or processor topology \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/
 
   if (myid == 0)
   {
      printf("  Convection-Diffusion: \n");
      printf("    -cx Dxx - cy Dyy - cz Dzz + ax Dx + ay Dy + az Dz = f\n");  
      printf("    (nx, ny, nz) = (%d, %d, %d)\n", nx, ny, nz);
      printf("    (Px, Py, Pz) = (%d, %d, %d)\n", P,  Q,  R);
      printf("    (cx, cy, cz) = (%f, %f, %f)\n", cx, cy, cz);
      printf("    (ax, ay, az) = (%f, %f, %f)\n\n", ax, ay, az);
   }

   /*-----------------------------------------------------------
    * Set up the grid structure
    *-----------------------------------------------------------*/

   /* compute p,q,r from P,Q,R and myid */
   p = myid % P;
   q = (( myid - p)/P) % Q;
   r = ( myid - p - P*q)/( P*Q );

   /*-----------------------------------------------------------
    * Generate the matrix 
    *-----------------------------------------------------------*/
 
   values = hypre_CTAlloc(double, 7);

   values[1] = -cx/(hinx*hinx);
   values[2] = -cy/(hiny*hiny);
   values[3] = -cz/(hinz*hinz);
   values[4] = -cx/(hinx*hinx) + ax/hinx;
   values[5] = -cy/(hiny*hiny) + ay/hiny;
   values[6] = -cz/(hinz*hinz) + az/hinz;

   values[0] = 0.;
   if (nx > 1)
   {
      values[0] += 2.0*cx/(hinx*hinx) - 1.*ax/hinx;
   }
   if (ny > 1)
   {
      values[0] += 2.0*cy/(hiny*hiny) - 1.*ay/hiny;
   }
   if (nz > 1)
   {
      values[0] += 2.0*cz/(hinz*hinz) - 1.*az/hinz;
   }

   A = (HYPRE_ParCSRMatrix) GenerateDifConv(MPI_COMM_WORLD,
                                            nx, ny, nz, P, Q, R, p, q, r, values);

   hypre_TFree(values);

   *A_ptr = A;

   return (0);
}

/*----------------------------------------------------------------------
 * Build matrix from one file on Proc. 0. Expects matrix to be in
 * CSR format. Distributes matrix across processors giving each about
 * the same number of rows.
 * Parameters given in command line.
 *----------------------------------------------------------------------*/

int
BuildParFromOneFile( int                  argc,
                     char                *argv[],
                     int                  arg_index,
                     HYPRE_ParCSRMatrix  *A_ptr     )
{
   char               *filename;

   HYPRE_ParCSRMatrix  A;
   HYPRE_CSRMatrix  A_CSR = NULL;

   int                 myid;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   MPI_Comm_rank(MPI_COMM_WORLD, &myid );

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/

   if (arg_index < argc)
   {
      filename = argv[arg_index];
   }
   else
   {
      printf("Error: No filename specified \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/
 
   if (myid == 0)
   {
      printf("  FromFile: %s\n", filename);

      /*-----------------------------------------------------------
       * Generate the matrix 
       *-----------------------------------------------------------*/
 
      A_CSR = HYPRE_CSRMatrixRead(filename);
   }
   HYPRE_CSRMatrixToParCSRMatrix(MPI_COMM_WORLD, A_CSR, NULL, NULL, &A);

   *A_ptr = A;

   if (myid == 0) HYPRE_CSRMatrixDestroy(A_CSR);

   return (0);
}

/*----------------------------------------------------------------------
 * Build Function array from files on different processors
 *----------------------------------------------------------------------*/

int
BuildFuncsFromFiles(    int                  argc,
                        char                *argv[],
                        int                  arg_index,
                        bHYPRE_IJParCSRMatrix   parcsr_A,
                        int                **dof_func_ptr     )
{
/*----------------------------------------------------------------------
 * Build Function array from files on different processors
 *----------------------------------------------------------------------*/

   printf (" Feature is not implemented yet!\n");	
   return(0);

}


int
BuildFuncsFromOneFile(  int                  argc,
                        char                *argv[],
                        int                  arg_index,
                        bHYPRE_IJParCSRMatrix   bHYPRE_parcsr_A,
                        int                **dof_func_ptr     )
{
   char           *filename;

   int             myid, num_procs;
   int            *partitioning;
   int            *dof_func;
   int            *dof_func_local;
   int             i, j;
   int             local_size, global_size;
   MPI_Request	  *requests;
   MPI_Status	  *status, status0;
   MPI_Comm	   comm;

   HYPRE_ParCSRMatrix parcsr_A;
   struct bHYPRE_IJParCSRMatrix__data * temp_data;
   void               *object;

   /*-----------------------------------------------------------
    * extract HYPRE_ParCSRMatrix from bHYPRE_IJParCSRMatrix
    *-----------------------------------------------------------*/
      temp_data = bHYPRE_IJParCSRMatrix__get_data( bHYPRE_parcsr_A );
      HYPRE_IJMatrixGetObject( temp_data->ij_A, &object);
      parcsr_A = (HYPRE_ParCSRMatrix) object;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   comm = MPI_COMM_WORLD;
   MPI_Comm_rank(MPI_COMM_WORLD, &myid );
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs );

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/

   if (arg_index < argc)
   {
      filename = argv[arg_index];
   }
   else
   {
      printf("Error: No filename specified \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/
 
   if (myid == 0)
   {
      FILE *fp;
      printf("  Funcs FromFile: %s\n", filename);

      /*-----------------------------------------------------------
       * read in the data
       *-----------------------------------------------------------*/
      fp = fopen(filename, "r");

      fscanf(fp, "%d", &global_size);
      dof_func = hypre_CTAlloc(int, global_size);

      for (j = 0; j < global_size; j++)
      {
         fscanf(fp, "%d", &dof_func[j]);
      }

      fclose(fp);
 
   }
   HYPRE_ParCSRMatrixGetRowPartitioning(parcsr_A, &partitioning);
   local_size = partitioning[myid+1]-partitioning[myid];
   dof_func_local = hypre_CTAlloc(int,local_size);

   if (myid == 0)
   {
      requests = hypre_CTAlloc(MPI_Request,num_procs-1);
      status = hypre_CTAlloc(MPI_Status,num_procs-1);
      j = 0;
      for (i=1; i < num_procs; i++)
      {
         MPI_Isend(&dof_func[partitioning[i]],
                   partitioning[i+1]-partitioning[i],
                   MPI_INT, i, 0, comm, &requests[j++]);
      }
      for (i=0; i < local_size; i++)
      {
         dof_func_local[i] = dof_func[i];
      }
      MPI_Waitall(num_procs-1,requests, status);
      hypre_TFree(requests);
      hypre_TFree(status);
   }
   else
   {
      MPI_Recv(dof_func_local,local_size,MPI_INT,0,0,comm,&status0);
   }

   *dof_func_ptr = dof_func_local;

   if (myid == 0) hypre_TFree(dof_func);

   return (0);
}

/*----------------------------------------------------------------------
 * Build Rhs from one file on Proc. 0. Distributes vector across processors 
 * giving each about using the distribution of the matrix A.
 *----------------------------------------------------------------------*/

int
BuildRhsParFromOneFile_( int                  argc,
                         char                *argv[],
                         int                  arg_index,
                         int                 *partitioning,
                         HYPRE_ParVector     *b_ptr     )
{
   char           *filename;

   HYPRE_ParVector b;
   HYPRE_Vector    b_CSR;

   int             myid;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   MPI_Comm_rank(MPI_COMM_WORLD, &myid );

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/

   if (arg_index < argc)
   {
      filename = argv[arg_index];
   }
   else
   {
      printf("Error: No filename specified \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/
 
   if (myid == 0)
   {
      printf("  Rhs FromFile: %s\n", filename);

      /*-----------------------------------------------------------
       * Generate the matrix 
       *-----------------------------------------------------------*/
 
      b_CSR = HYPRE_VectorRead(filename);
   }
   HYPRE_VectorToParVector(MPI_COMM_WORLD, b_CSR, partitioning,&b); 

   *b_ptr = b;

   HYPRE_VectorDestroy(b_CSR);

   return (0);
}

/*----------------------------------------------------------------------
 * Build standard 9-point laplacian in 2D with grid and anisotropy.
 * Parameters given in command line.
 *----------------------------------------------------------------------*/

int
BuildParLaplacian9pt( int                  argc,
                      char                *argv[],
                      int                  arg_index,
                      HYPRE_ParCSRMatrix  *A_ptr     )
{
   int                 nx, ny;
   int                 P, Q;

   HYPRE_ParCSRMatrix  A;

   int                 num_procs, myid;
   int                 p, q;
   double             *values;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   MPI_Comm_size(MPI_COMM_WORLD, &num_procs );
   MPI_Comm_rank(MPI_COMM_WORLD, &myid );

   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/
 
   nx = 10;
   ny = 10;

   P  = 1;
   Q  = num_procs;

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/
   arg_index = 0;
   while (arg_index < argc)
   {
      if ( strcmp(argv[arg_index], "-n") == 0 )
      {
         arg_index++;
         nx = atoi(argv[arg_index++]);
         ny = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-P") == 0 )
      {
         arg_index++;
         P  = atoi(argv[arg_index++]);
         Q  = atoi(argv[arg_index++]);
      }
      else
      {
         arg_index++;
      }
   }

   /*-----------------------------------------------------------
    * Check a few things
    *-----------------------------------------------------------*/

   if ((P*Q) != num_procs)
   {
      printf("Error: Invalid number of processors or processor topology \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/
 
   if (myid == 0)
   {
      printf("  Laplacian 9pt:\n");
      printf("    (nx, ny) = (%d, %d)\n", nx, ny);
      printf("    (Px, Py) = (%d, %d)\n\n", P,  Q);
   }

   /*-----------------------------------------------------------
    * Set up the grid structure
    *-----------------------------------------------------------*/

   /* compute p,q from P,Q and myid */
   p = myid % P;
   q = ( myid - p)/P;

   /*-----------------------------------------------------------
    * Generate the matrix 
    *-----------------------------------------------------------*/
 
   values = hypre_CTAlloc(double, 2);

   values[1] = -1.;

   values[0] = 0.;
   if (nx > 1)
   {
      values[0] += 2.0;
   }
   if (ny > 1)
   {
      values[0] += 2.0;
   }
   if (nx > 1 && ny > 1)
   {
      values[0] += 4.0;
   }

   A = (HYPRE_ParCSRMatrix) GenerateLaplacian9pt(MPI_COMM_WORLD,
                                                 nx, ny, P, Q, p, q, values);

   hypre_TFree(values);

   *A_ptr = A;

   return (0);
}
/*----------------------------------------------------------------------
 * Build 27-point laplacian in 3D,
 * Parameters given in command line.
 *----------------------------------------------------------------------*/

int
BuildParLaplacian27pt( int                  argc,
                       char                *argv[],
                       int                  arg_index,
                       HYPRE_ParCSRMatrix  *A_ptr     )
{
   int                 nx, ny, nz;
   int                 P, Q, R;

   HYPRE_ParCSRMatrix  A;

   int                 num_procs, myid;
   int                 p, q, r;
   double             *values;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   MPI_Comm_size(MPI_COMM_WORLD, &num_procs );
   MPI_Comm_rank(MPI_COMM_WORLD, &myid );

   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/
 
   nx = 10;
   ny = 10;
   nz = 10;

   P  = 1;
   Q  = num_procs;
   R  = 1;

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/
   arg_index = 0;
   while (arg_index < argc)
   {
      if ( strcmp(argv[arg_index], "-n") == 0 )
      {
         arg_index++;
         nx = atoi(argv[arg_index++]);
         ny = atoi(argv[arg_index++]);
         nz = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-P") == 0 )
      {
         arg_index++;
         P  = atoi(argv[arg_index++]);
         Q  = atoi(argv[arg_index++]);
         R  = atoi(argv[arg_index++]);
      }
      else
      {
         arg_index++;
      }
   }

   /*-----------------------------------------------------------
    * Check a few things
    *-----------------------------------------------------------*/

   if ((P*Q*R) != num_procs)
   {
      printf("Error: Invalid number of processors or processor topology \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/
 
   if (myid == 0)
   {
      printf("  Laplacian_27pt:\n");
      printf("    (nx, ny, nz) = (%d, %d, %d)\n", nx, ny, nz);
      printf("    (Px, Py, Pz) = (%d, %d, %d)\n\n", P,  Q,  R);
   }

   /*-----------------------------------------------------------
    * Set up the grid structure
    *-----------------------------------------------------------*/

   /* compute p,q,r from P,Q,R and myid */
   p = myid % P;
   q = (( myid - p)/P) % Q;
   r = ( myid - p - P*q)/( P*Q );

   /*-----------------------------------------------------------
    * Generate the matrix 
    *-----------------------------------------------------------*/
 
   values = hypre_CTAlloc(double, 2);

   values[0] = 26.0;
   if (nx == 1 || ny == 1 || nz == 1)
      values[0] = 8.0;
   if (nx*ny == 1 || nx*nz == 1 || ny*nz == 1)
      values[0] = 2.0;
   values[1] = -1.;

   A = (HYPRE_ParCSRMatrix) GenerateLaplacian27pt(MPI_COMM_WORLD,
                                                  nx, ny, nz, P, Q, R, p, q, r, values);

   hypre_TFree(values);

   *A_ptr = A;

   return (0);
}

void ParseCommandLine_1( int argc, char *argv[], CommandLineParameters *clp )
{
   /*-----------------------------------------------------------
    * Parse much of the command line, and some defaults
    *-----------------------------------------------------------*/
   int arg_index;
 
   clp->dt = dt_inf;
   clp->build_matrix_type = 2;
   clp->build_matrix_arg_index = argc;
   clp->build_rhs_type = 2;
   clp->build_rhs_arg_index = argc;
   clp->build_src_type = -1;
   clp->build_src_arg_index = argc;
   clp->build_funcs_type = 0;
   clp->build_funcs_arg_index = argc;
   clp->debug_flag = 0;
   clp->print_usage = 0;
   clp->sparsity_known = 0;
   clp->solver_id = 0;
   clp->smooth_num_levels = 0;
   clp->hpcg = 0;
   clp->coarsen_type = 6;
   clp->hybrid = 1;
   clp->measure_type = 0;
   clp->relax_default = 3;
   clp->smooth_type = 6;
   clp->max_levels = 25;
   clp->num_functions = 1;
   clp->num_sweep = 1;
   clp->smooth_num_sweep = 1;

   arg_index = 1;

   while ( (arg_index < argc) && (!(clp->print_usage)) )
   {
      if ( strcmp(argv[arg_index], "-fromijfile") == 0 )
      {
         arg_index++;
         clp->build_matrix_type      = -1;
         clp->build_matrix_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-fromparcsrfile") == 0 )
      {
         arg_index++;
         clp->build_matrix_type      = 0;
         clp->build_matrix_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-fromonecsrfile") == 0 )
      {
         arg_index++;
         clp->build_matrix_type      = 1;
         clp->build_matrix_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-laplacian") == 0 )
      {
         arg_index++;
         clp->build_matrix_type      = 2;
         clp->build_matrix_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-9pt") == 0 )
      {
         arg_index++;
         clp->build_matrix_type      = 3;
         clp->build_matrix_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-27pt") == 0 )
      {
         arg_index++;
         clp->build_matrix_type      = 4;
         clp->build_matrix_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-difconv") == 0 )
      {
         arg_index++;
         clp->build_matrix_type      = 5;
         clp->build_matrix_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-funcsfromonefile") == 0 )
      {
         arg_index++;
         clp->build_funcs_type      = 1;
         clp->build_funcs_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-funcsfromfile") == 0 )
      {
         arg_index++;
         clp->build_funcs_type      = 2;
         clp->build_funcs_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-exact_size") == 0 )
      {
         arg_index++;
         clp->sparsity_known = 1;
      }
      else if ( strcmp(argv[arg_index], "-storage_low") == 0 )
      {
         arg_index++;
         clp->sparsity_known = 2;
      }
      else if ( strcmp(argv[arg_index], "-concrete_parcsr") == 0 )
      {
         arg_index++;
         clp->build_matrix_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-solver") == 0 )
      {
         arg_index++;
         clp->solver_id = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-hpcg") == 0 )
      {
         arg_index++;
         clp->hpcg = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-rhsfromfile") == 0 )
      {
         arg_index++;
         clp->build_rhs_type      = 0;
         clp->build_rhs_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-rhsfromonefile") == 0 )
      {
         arg_index++;
         clp->build_rhs_type      = 1;
         clp->build_rhs_arg_index = arg_index;
      }      
      else if ( strcmp(argv[arg_index], "-rhsisone") == 0 )
      {
         arg_index++;
         clp->build_rhs_type      = 2;
         clp->build_rhs_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-rhsrand") == 0 )
      {
         arg_index++;
         clp->build_rhs_type      = 3;
         clp->build_rhs_arg_index = arg_index;
      }    
      else if ( strcmp(argv[arg_index], "-xisone") == 0 )
      {
         arg_index++;
         clp->build_rhs_type      = 4;
         clp->build_rhs_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-rhszero") == 0 )
      {
         arg_index++;
         clp->build_rhs_type      = 5;
         clp->build_rhs_arg_index = arg_index;
      }    
      else if ( strcmp(argv[arg_index], "-srcfromfile") == 0 )
      {
         arg_index++;
         clp->build_src_type      = 0;
         clp->build_rhs_type      = -1;
         clp->build_src_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-srcfromonefile") == 0 )
      {
         arg_index++;
         clp->build_src_type      = 1;
         clp->build_rhs_type      = -1;
         clp->build_src_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-srcisone") == 0 )
      {
         arg_index++;
         clp->build_src_type      = 2;
         clp->build_rhs_type      = -1;
         clp->build_src_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-srcrand") == 0 )
      {
         arg_index++;
         clp->build_src_type      = 3;
         clp->build_rhs_type      = -1;
         clp->build_src_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-srczero") == 0 )
      {
         arg_index++;
         clp->build_src_type      = 4;
         clp->build_rhs_type      = -1;
         clp->build_src_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-cljp") == 0 )
      {
         arg_index++;
         clp->coarsen_type      = 0;
      }    
      else if ( strcmp(argv[arg_index], "-ruge") == 0 )
      {
         arg_index++;
         clp->coarsen_type      = 1;
      }    
      else if ( strcmp(argv[arg_index], "-ruge2b") == 0 )
      {
         arg_index++;
         clp->coarsen_type      = 2;
      }    
      else if ( strcmp(argv[arg_index], "-ruge3") == 0 )
      {
         arg_index++;
         clp->coarsen_type      = 3;
      }    
      else if ( strcmp(argv[arg_index], "-ruge3c") == 0 )
      {
         arg_index++;
         clp->coarsen_type      = 4;
      }    
      else if ( strcmp(argv[arg_index], "-rugerlx") == 0 )
      {
         arg_index++;
         clp->coarsen_type      = 5;
      }    
      else if ( strcmp(argv[arg_index], "-falgout") == 0 )
      {
         arg_index++;
         clp->coarsen_type      = 6;
      }    
      else if ( strcmp(argv[arg_index], "-nohybrid") == 0 )
      {
         arg_index++;
         clp->hybrid      = -1;
      }    
      else if ( strcmp(argv[arg_index], "-gm") == 0 )
      {
         arg_index++;
         clp->measure_type      = 1;
      }    
      else if ( strcmp(argv[arg_index], "-rlx") == 0 )
      {
         arg_index++;
         clp->relax_default = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-smtype") == 0 )
      {
         arg_index++;
         clp->smooth_type = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-smlv") == 0 )
      {
         arg_index++;
         clp->smooth_num_levels = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-mxl") == 0 )
      {
         arg_index++;
         clp->max_levels = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-dbg") == 0 )
      {
         arg_index++;
         clp->debug_flag = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-nf") == 0 )
      {
         arg_index++;
         clp->num_functions = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-ns") == 0 )
      {
         arg_index++;
         clp->num_sweep = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-sns") == 0 )
      {
         arg_index++;
         clp->smooth_num_sweep = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-dt") == 0 )
      {
         arg_index++;
         clp->dt = atof(argv[arg_index++]);
         clp->build_rhs_type = -1;
         if ( clp->build_src_type == -1 ) clp->build_src_type = 2;
      }
      else if ( strcmp(argv[arg_index], "-help") == 0 )
      {
         clp->print_usage = 1;
      }
      else
      {
         arg_index++;
      }
   }

   /* max_levels for ParaSails - overrides user input if any */
   if (clp->solver_id == 8 || clp->solver_id == 18)
   {
      clp->max_levels = 1;
   }

}

void ParseCommandLine_2( int argc, char *argv[], CommandLineParameters *clp )
{
   /*-----------------------------------------------------------
    * Parse more of the command line, and some defaults.
    * This depends on clp parameters set by ParseCommandLine_1, for example
    * solver_type and max_levels.
    *-----------------------------------------------------------*/

   int                 arg_index, i;

   clp->gsmg_samples = 5;
   clp->interp_type  = 200;
   clp->variant = 0;  /* multiplicative */
   clp->overlap = 1;  /* 1 layer overlap */
   clp->domain_type = 2; /* through agglomeration */
   clp->schwarz_rlx_weight = 1.;
   clp->ioutdat = 3;
   clp->poutdat = 1;
   clp->k_dim = 5;
   clp->drop_tol = -1;
   clp->nonzeros_to_keep = -1;
   clp->max_row_sum = 1.;
   clp->tol = 1.e-8;
   clp->pc_tol = 0.;
   clp->sai_threshold = 0.1;
   clp->sai_filter = 0.1;

   arg_index = 0;
   while (arg_index < argc)
   {
      if ( strcmp(argv[arg_index], "-k") == 0 )
      {
         arg_index++;
         clp->k_dim = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-w") == 0 )
      {
         arg_index++;
         if (clp->solver_id == 0 || clp->solver_id == 1 || clp->solver_id == 3 
             || clp->solver_id == 5 )
         {
            (clp->relax_weight)[0] = atof(argv[arg_index++]);
            for (i=1; i < clp->max_levels; i++)
            {
               (clp->relax_weight)[i] = (clp->relax_weight)[0];
            }
         }
      }
      else if ( strcmp(argv[arg_index], "-sw") == 0 )
      {
         arg_index++;
         clp->schwarz_rlx_weight = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-th") == 0 )
      {
         arg_index++;
         clp->strong_threshold  = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-tol") == 0 )
      {
         arg_index++;
         clp->tol  = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-mxrs") == 0 )
      {
         arg_index++;
         clp->max_row_sum  = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-sai_th") == 0 )
      {
         arg_index++;
         clp->sai_threshold  = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-sai_filt") == 0 )
      {
         arg_index++;
         clp->sai_filter  = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-drop_tol") == 0 )
      {
         arg_index++;
         clp->drop_tol  = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-nonzeros_to_keep") == 0 )
      {
         arg_index++;
         clp->nonzeros_to_keep  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-tr") == 0 )
      {
         arg_index++;
         clp->trunc_factor  = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-iout") == 0 )
      {
         arg_index++;
         clp->ioutdat  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-pout") == 0 )
      {
         arg_index++;
         clp->poutdat  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-var") == 0 )
      {
         arg_index++;
         clp->variant  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-ov") == 0 )
      {
         arg_index++;
         clp->overlap  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-dom") == 0 )
      {
         arg_index++;
         clp->domain_type  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-mu") == 0 )
      {
         arg_index++;
         clp->cycle_type  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-numsamp") == 0 )
      {
         arg_index++;
         clp->gsmg_samples  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-interptype") == 0 )
      {
         arg_index++;
         clp->interp_type  = atoi(argv[arg_index++]);
      }
      else
      {
         arg_index++;
      }
   }
}

void BoomerAMG_DefaultParameters( CommandLineParameters *clp )
{
   int i;

   clp->relax_weight = NULL;
   clp->num_grid_sweeps = NULL;  
   clp->grid_relax_type = NULL;   
   clp->grid_relax_points = NULL;

   if (clp->solver_id == 0 || clp->solver_id == 1 || clp->solver_id == 3
       || clp->solver_id == 5
       || clp->solver_id == 9 || clp->solver_id == 13 || clp->solver_id == 14
       || clp->solver_id == 15 || clp->solver_id == 20)
   {

      clp->strong_threshold = 0.25;
      clp->trunc_factor = 0.;
      clp->cycle_type = 1;

      clp->num_grid_sweeps   = hypre_CTAlloc(int,4);
      clp->grid_relax_type   = hypre_CTAlloc(int,4);
      clp->grid_relax_points = hypre_CTAlloc(int *,4);
      clp->relax_weight      = hypre_CTAlloc(double, clp->max_levels);
      clp->omega      = hypre_CTAlloc(double, clp->max_levels);

      for (i=0; i < clp->max_levels; i++)
      {
         (clp->relax_weight)[i] = 1.;
         (clp->omega)[i] = 1.;
      }

      /* for CGNR preconditioned with Boomeramg, only relaxation scheme 0 is
         implemented, i.e. Jacobi relaxation */
      if (clp->solver_id == 5) 
      {
         /* fine grid */
         clp->relax_default = 7;
         (clp->grid_relax_type)[0] = clp->relax_default; 
         (clp->num_grid_sweeps)[0] = clp->num_sweep;
         (clp->grid_relax_points)[0] = hypre_CTAlloc(int, clp->num_sweep); 
         for (i=0; i<clp->num_sweep; i++)
         {
            (clp->grid_relax_points)[0][i] = 0;
         };
         /* down cycle */
         (clp->grid_relax_type)[1] = clp->relax_default; 
         (clp->num_grid_sweeps)[1] = clp->num_sweep;
         (clp->grid_relax_points)[1] = hypre_CTAlloc(int, clp->num_sweep); 
         for (i=0; i<clp->num_sweep; i++)
         {
            (clp->grid_relax_points)[1][i] = 0;
         } 
         /* up cycle */
         (clp->grid_relax_type)[2] = clp->relax_default; 
         (clp->num_grid_sweeps)[2] = clp->num_sweep;
         (clp->grid_relax_points)[2] = hypre_CTAlloc(int, clp->num_sweep); 
         for (i=0; i<clp->num_sweep; i++)
         {
            (clp->grid_relax_points)[2][i] = 0;
         } 
      }
      else if (clp->coarsen_type == 5)
      {
         /* fine grid */
         (clp->num_grid_sweeps)[0] = 3;
         (clp->grid_relax_type)[0] = clp->relax_default; 
         (clp->grid_relax_points)[0] = hypre_CTAlloc(int, 3); 
         (clp->grid_relax_points)[0][0] = -2;
         (clp->grid_relax_points)[0][1] = -1;
         (clp->grid_relax_points)[0][2] = 1;
   
         /* down cycle */
         (clp->num_grid_sweeps)[1] = 4;
         (clp->grid_relax_type)[1] = clp->relax_default; 
         (clp->grid_relax_points)[1] = hypre_CTAlloc(int, 4); 
         (clp->grid_relax_points)[1][0] = -1;
         (clp->grid_relax_points)[1][1] = 1;
         (clp->grid_relax_points)[1][2] = -2;
         (clp->grid_relax_points)[1][3] = -2;
   
         /* up cycle */
         (clp->num_grid_sweeps)[2] = 4;
         (clp->grid_relax_type)[2] = clp->relax_default; 
         (clp->grid_relax_points)[2] = hypre_CTAlloc(int, 4); 
         (clp->grid_relax_points)[2][0] = -2;
         (clp->grid_relax_points)[2][1] = -2;
         (clp->grid_relax_points)[2][2] = 1;
         (clp->grid_relax_points)[2][3] = -1;
      }
      else
      {   
         /* fine grid */
         (clp->num_grid_sweeps)[0] = 2*clp->num_sweep;
         (clp->grid_relax_type)[0] = clp->relax_default; 
         (clp->grid_relax_points)[0] = hypre_CTAlloc(int, 2*clp->num_sweep); 
         for (i=0; i<2*clp->num_sweep; i+=2)
         {
            (clp->grid_relax_points)[0][i] = 1;
            (clp->grid_relax_points)[0][i+1] = -1;
         }

         /* down cycle */
         (clp->num_grid_sweeps)[1] = 2*clp->num_sweep;
         (clp->grid_relax_type)[1] = clp->relax_default; 
         (clp->grid_relax_points)[1] = hypre_CTAlloc(int, 2*clp->num_sweep); 
         for (i=0; i<2*clp->num_sweep; i+=2)
         {
            (clp->grid_relax_points)[1][i] = 1;
            (clp->grid_relax_points)[1][i+1] = -1;
         }

         /* up cycle */
         (clp->num_grid_sweeps)[2] = 2*clp->num_sweep;
         (clp->grid_relax_type)[2] = clp->relax_default; 
         (clp->grid_relax_points)[2] = hypre_CTAlloc(int, 2*clp->num_sweep); 
         for (i=0; i<2*clp->num_sweep; i+=2)
         {
            (clp->grid_relax_points)[2][i] = -1;
            (clp->grid_relax_points)[2][i+1] = 1;
         }
      }

      /* coarsest grid */
      (clp->num_grid_sweeps)[3] = 1;
      (clp->grid_relax_type)[3] = 9;
      (clp->grid_relax_points)[3] = hypre_CTAlloc(int, 1);
      (clp->grid_relax_points)[3][0] = 0;
   }
}

void PrintUsage( char *argv[] )
{
   printf("\n");
   printf("Usage: %s [<options>]\n", argv[0]);
   printf("\n");
   printf("  -fromijfile <filename>     : ");
   printf("matrix read in IJ format from distributed files\n");
   printf("  -fromparcsrfile <filename> : ");
   printf("matrix read in ParCSR format from distributed files\n");
   printf("  -fromonecsrfile <filename> : ");
   printf("matrix read in CSR format from a file on one processor\n");
   printf("\n");
   printf("  -laplacian [<options>] : build 5pt 2D laplacian problem (default) \n");
   printf(" only the default is supported at present\n" );
   printf("  -9pt [<opts>]          : build 9pt 2D laplacian problem\n");
   printf("  -27pt [<opts>]         : build 27pt 3D laplacian problem\n");
   printf("  -difconv [<opts>]      : build convection-diffusion problem\n");
   printf("    -n <nx> <ny> <nz>    : total problem size \n");
   printf("    -P <Px> <Py> <Pz>    : processor topology\n");
   printf("    -c <cx> <cy> <cz>    : diffusion coefficients\n");
   printf("    -a <ax> <ay> <az>    : convection coefficients\n");
   printf("\n");
   printf("  -exact_size            : inserts immediately into ParCSR structure\n");
   printf("  -storage_low           : allocates not enough storage for aux struct\n");
   printf("  -concrete_parcsr       : use parcsr matrix type as concrete type\n");
   printf("\n");
   printf("  -rhsfromfile           : rhs read in IJ form from distributed files\n");
   printf("  -rhsfromonefile        : rhs read from a file one one processor\n");
   printf("  -rhsrand               : rhs is random vector\n");
   printf("  -rhsisone              : rhs is vector with unit components (default)\n");
   printf(" only the default is supported at present\n" );
   printf("  -xisone                : solution of all ones\n");
   printf("  -rhszero               : rhs is zero vector\n");
   printf("\n");
   printf(" the backward Euler and src options are not supported yet\n");
#ifdef DO_THIS_LATER
   printf("  -dt <val>              : specify finite backward Euler time step\n");
   printf("                         :    -rhsfromfile, -rhsfromonefile, -rhsrand,\n");
   printf("                         :    -rhsrand, or -xisone will be ignored\n");
   printf("  -srcfromfile           : backward Euler source read in IJ form from distributed files\n");
   printf("  -srcfromonefile        : ");
   printf("backward Euler source read from a file on one processor\n");
   printf("  -srcrand               : ");
   printf("backward Euler source is random vector with components in range 0 - 1\n");
   printf("  -srcisone              : ");
   printf("backward Euler source is vector with unit components (default)\n");
   printf("  -srczero               : ");
   printf("backward Euler source is zero-vector\n");
   printf("The backward Euler source options have not been implemented.\n");
   printf("\n");
#endif /* DO_THIS_LATER */
   printf("  -solver <ID>           : solver ID\n");
   printf("        0=AMG                1=AMG-PCG        \n");
   printf("        2=DS-PCG             3=AMG-GMRES      \n");
   printf("        4=DS-GMRES           5=AMG-CGNR       \n");     
   printf("        6=DS-CGNR            7*=PILUT-GMRES    \n");     
   printf("        8=ParaSails-PCG      9=AMG-BiCGSTAB   \n");
   printf("       10=DS-BiCGSTAB       11*=PILUT-BiCGSTAB \n");
   printf("       12=Schwarz-PCG      18=ParaSails-GMRES\n");     
   printf("        43=Euclid-PCG       44*=Euclid-GMRES   \n");
   printf("       45*=Euclid-BICGSTAB\n");
   printf("Solvers marked with '*' have not yet been implemented.\n");
   printf("   -hpcg 1               : for HYPRE-interface version of PCG or GMRES solver\n");
   printf("\n");
   printf("   -cljp                 : CLJP coarsening \n");
   printf("   -ruge                 : Ruge coarsening (local)\n");
   printf("   -ruge3                : third pass on boundary\n");
   printf("   -ruge3c               : third pass on boundary, keep c-points\n");
   printf("   -ruge2b               : 2nd pass is global\n");
   printf("   -rugerlx              : relaxes special points\n");
   printf("   -falgout              : local ruge followed by LJP\n");
   printf("   -nohybrid             : no switch in coarsening\n");
   printf("   -gm                   : use global measures\n");
   printf("\n");
   printf("  -rlx  <val>            : relaxation type\n");
   printf("       0=Weighted Jacobi  \n");
   printf("       1=Gauss-Seidel (very slow!)  \n");
   printf("       3=Hybrid Jacobi/Gauss-Seidel  \n");
   printf("  -ns <val>              : Use <val> sweeps on each level\n");
   printf("                           (default C/F down, F/C up, F/C fine\n");
   printf("\n"); 
   printf("  -mu   <val>            : set AMG cycles (1=V, 2=W, etc.)\n"); 
   printf("  -th   <val>            : set AMG threshold Theta = val \n");
   printf("  -tr   <val>            : set AMG interpolation truncation factor = val \n");
   printf("  -mxrs <val>            : set AMG maximum row sum threshold for dependency weakening \n");
   printf("  -nf <val>              : set number of functions for systems AMG\n");
     
   printf("  -w   <val>             : set Jacobi relax weight = val\n");
   printf("  -k   <val>             : dimension Krylov space for GMRES\n");
   printf("  -mxl  <val>            : maximum number of levels (AMG, ParaSAILS)\n");
   printf("  -tol  <val>            : set solver convergence tolerance = val\n");
   printf("\n");
   printf("  -sai_th   <val>        : set ParaSAILS threshold = val \n");
   printf("  -sai_filt <val>        : set ParaSAILS filter = val \n");
   printf("\n");  
   printf("  -drop_tol  <val>       : set threshold for dropping in PILUT\n");
   printf("  -nonzeros_to_keep <val>: number of nonzeros in each row to keep\n");
   printf("\n");  
   printf("  -iout <val>            : set output flag\n");
   printf("       0=no output    1=matrix stats\n"); 
   printf("       2=cycle stats  3=matrix & cycle stats\n"); 
   printf("\n");  
   printf("  -dbg <val>             : set debug flag\n");
   printf("       0=no debugging\n       1=internal timing\n       2=interpolation truncation\n       3=more detailed timing in coarsening routine\n");

}

int IJMatrixVectorDebug(
   const bHYPRE_MPICommunicator bmpicomm, const int local_num_cols,
   const int first_local_col, const int last_local_col,
   const bHYPRE_IJParCSRMatrix  bHYPRE_parcsr_A,
   bHYPRE_IJParCSRVector bHYPRE_b, bHYPRE_IJParCSRVector bHYPRE_x )
{

   /*-----------------------------------------------------------
    * Matrix-Vector and Vector Operation Debugging code begun by adapting
    * from Rob Falgout's sstruct tests
    *-----------------------------------------------------------*/

   bHYPRE_IJParCSRVector  bHYPRE_y;
   bHYPRE_IJParCSRVector  bHYPRE_y2;
   bHYPRE_Vector  y;
   bHYPRE_Vector bHYPRE_Vector_x;
   int *indices;
   double *values;
   double tmp;
   int ierr = 0;
   int i;

   /*  Apply, y=A*b: result is 1's on the interior of the grid */
   bHYPRE_y = bHYPRE_IJParCSRVector_Create( bmpicomm,
                                            first_local_col,
                                            last_local_col );
   ierr += bHYPRE_IJParCSRVector_Initialize( bHYPRE_y );
   y = bHYPRE_Vector__cast( bHYPRE_y );

   bHYPRE_IJParCSRMatrix_Apply( bHYPRE_parcsr_A,
                                bHYPRE_Vector__cast( bHYPRE_b ),
                                &y );

   bHYPRE_IJParCSRMatrix_Print( bHYPRE_parcsr_A, "test.A" );
   bHYPRE_IJParCSRVector_Print( bHYPRE_y, "test.apply" );
   bHYPRE_Vector_deleteRef( y );

   /* SetValues, x=1; result is all 1's */
   indices = hypre_CTAlloc(int, local_num_cols);
   values = hypre_CTAlloc(double, local_num_cols);
   for ( i=0; i<local_num_cols; ++i )
   {
      indices[i] = i+first_local_col;
      values[i] = 1.0;
   }
   bHYPRE_IJParCSRVector_SetValues( bHYPRE_x, local_num_cols, indices, values );
   hypre_TFree(indices);
   hypre_TFree(values);
   bHYPRE_IJParCSRVector_Print( bHYPRE_x, "test.setvalues" );

   /* Copy, b=x; result is all 1's */
   bHYPRE_Vector_x = bHYPRE_Vector__cast( bHYPRE_x );
   bHYPRE_IJParCSRVector_Copy( bHYPRE_b, bHYPRE_Vector_x );
   bHYPRE_IJParCSRVector_Print( bHYPRE_b, "test.copy" );

   /* Clone y=b and copy data; result is all 1's */
   bHYPRE_IJParCSRVector_Clone( bHYPRE_b, &y );
   bHYPRE_y = bHYPRE_IJParCSRVector__cast( y );
   bHYPRE_IJParCSRVector_Copy( bHYPRE_y, bHYPRE_Vector__cast( bHYPRE_b ) );
   bHYPRE_IJParCSRVector_Print( bHYPRE_y, "test.clone" );
   bHYPRE_Vector_deleteRef( y );

   /* Read y2=y; result is all 1's */
   bHYPRE_y2 = bHYPRE_IJParCSRVector_Create( bmpicomm,
                                             first_local_col,
                                             last_local_col );
   ierr += bHYPRE_IJParCSRVector_Initialize( bHYPRE_y2 );
   bHYPRE_IJParCSRVector_Read( bHYPRE_y2, "test.clone", bmpicomm );
   bHYPRE_IJParCSRVector_Print( bHYPRE_y2, "test.read" );

   bHYPRE_IJParCSRVector_deleteRef( bHYPRE_y2 );

   /* Scale, x=2*x; result is all 2's */
   bHYPRE_IJParCSRVector_Scale( bHYPRE_x, 2.0 );
   bHYPRE_IJParCSRVector_Print( bHYPRE_x, "test.scale" );

   /* Dot, tmp = b.x; at this point all b[i]==1, all x[i]==2 */
   bHYPRE_IJParCSRVector_Dot( bHYPRE_b, bHYPRE_Vector_x, &tmp );
   hypre_assert( tmp==2*N );

   /* Axpy, b=b-0.5*x; result is all 0's */
   bHYPRE_IJParCSRVector_Axpy( bHYPRE_b, -0.5, bHYPRE_Vector_x );
   bHYPRE_IJParCSRVector_Print( bHYPRE_b, "test.axpy" );

   /* tested by other parts of this driver program: ParCSRVector_GetObject */

   /* Clear and AddToValues, b=1, which restores its initial value of 1 */
   indices = hypre_CTAlloc(int, local_num_cols);
   values = hypre_CTAlloc(double, local_num_cols);
   for ( i=0; i<local_num_cols; ++i )
   {
      indices[i] = i+first_local_col;
      values[i] = 1.0;
   }
   bHYPRE_IJParCSRVector_Clear( bHYPRE_b );
   bHYPRE_IJParCSRVector_AddToValues
      ( bHYPRE_b, local_num_cols, indices, values );
   hypre_TFree(indices);
   hypre_TFree(values);
   bHYPRE_IJParCSRVector_Print( bHYPRE_b, "test.addtovalues" );

   /* Clear,x=0, which restores its initial value of 0 */
   bHYPRE_IJParCSRVector_Clear( bHYPRE_x );
   bHYPRE_IJParCSRVector_Print( bHYPRE_x, "test.clear" );

   return ierr;
}

int Demo_Matrix_AddToValues(
   bHYPRE_IJParCSRMatrix bHYPRE_parcsr_A, CommandLineParameters *clp,
   int first_local_row, int last_local_row )
{
   /* This is to emphasize that one can IJMatrixAddToValues after an
      IJMatrixRead or an IJMatrixAssemble.  After an IJMatrixRead,
      assembly is unnecessary if the sparsity pattern of the matrix is
      not changed somehow.  If one has not used IJMatrixRead, one has
      the opportunity to IJMatrixAddTo before a IJMatrixAssemble. */

   int * ncols    = hypre_CTAlloc(int, last_local_row - first_local_row + 1);
   int * rows     = hypre_CTAlloc(int, last_local_row - first_local_row + 1);
   int * col_inds = hypre_CTAlloc(int, last_local_row - first_local_row + 1);
   double * values   = hypre_CTAlloc(double, last_local_row - first_local_row + 1);
   double val;
   int i, j;
   int ierr = 0;
   int local_num_rows = last_local_row - first_local_row + 1;

   if (clp->dt < dt_inf)
      val = 1./clp->dt;
   else 
      val = 0.;  /* Use zero to avoid unintentional loss of significance */

   for (i = first_local_row; i <= last_local_row; i++)
   {
      j = i - first_local_row;
      rows[j] = i;
      ncols[j] = 1;
      col_inds[j] = i;
      values[j] = val;
   }
      
   ierr += bHYPRE_IJParCSRMatrix_AddToValues
      ( bHYPRE_parcsr_A, local_num_rows, ncols, rows, col_inds, values, local_num_rows );

   hypre_TFree(values);
   hypre_TFree(col_inds);
   hypre_TFree(rows);
   hypre_TFree(ncols);


   /* If sparsity pattern is not changed since last IJMatrixAssemble call,
      this should be a no-op */

   ierr += bHYPRE_IJParCSRMatrix_Assemble( bHYPRE_parcsr_A );

   return ierr;
}

void Print_BabelTimeCorrection( int myid, int argc, char *argv[],
                                CommandLineParameters *clp, MPI_Comm mpi_comm
   )
{
   /*
   bBuildParLaplacian and the function it calls, bHYPRE_IJMatrix_GenerateLaplacian,
   include part of what is (and should be) called "matrix setup" in ij.c,
   but they also include matrix values computation which is not considered part
   of "setup".  So here we do the values computation alone, just so we'll know the non-setup
   part of the setup timing computation done below.  Hypre timing functions
   don't have a subtraction feature, so you the "user" will have to do it yourself.
   */
   HYPRE_ParCSRMatrix    parcsr_A;/* only for timing computation */
   int time_index;

   time_index = hypre_InitializeTiming("LaplacianComputation");
   hypre_BeginTiming(time_index);
   if ( clp->build_matrix_type == 2 )
   {
      BuildParLaplacian(argc, argv, clp->build_matrix_arg_index, &parcsr_A);
      HYPRE_ParCSRMatrixDestroy( parcsr_A );
   }
   else
   {
      if ( myid==0 ) printf("timing only correct for build_matrix_type==2\n");
   }
   hypre_EndTiming(time_index);
   hypre_PrintTiming( "Laplacian Computation, deduct from Matrix Setup", mpi_comm );
   hypre_FinalizeTiming(time_index);
   hypre_ClearTiming();
}

void BuildDefaultFuncs( CommandLineParameters *clp, int myid,
                        int local_num_rows, int first_local_row, int **dof_func )
{
   int local_num_vars, j, k;
   int                 indx, rest, tms;

   local_num_vars = local_num_rows;
   *dof_func = hypre_CTAlloc(int,local_num_vars);
   if (myid == 0)
      printf (" Number of unknown functions = %d \n", clp->num_functions);
   rest = first_local_row-((first_local_row/(clp->num_functions))*(clp->num_functions));
   indx = (clp->num_functions)-rest;
   if (rest == 0) indx = 0;
   k = (clp->num_functions) - 1;
   for (j = indx-1; j > -1; j--)
   {
      (*dof_func)[j] = k--;
   }
   tms = local_num_vars/(clp->num_functions);
   if (tms*(clp->num_functions)+indx > local_num_vars) tms--;
   for (j=0; j < tms; j++)
   {
      for (k=0; k < clp->num_functions; k++)
      {
         (*dof_func)[indx++] = k;
      }
   }
   k = 0;
   while (indx < local_num_vars)
      (*dof_func)[indx++] = k++;
}


int Test_AMG( CommandLineParameters *clp, bHYPRE_IJParCSRMatrix bHYPRE_parcsr_A,
              bHYPRE_IJParCSRVector bHYPRE_b, bHYPRE_IJParCSRVector bHYPRE_x,
              int * dof_func,
              MPI_Comm mpi_comm, bHYPRE_MPICommunicator bmpicomm )
{
   int ierr = 0;
   int                 log_level, i, j, myid;
   int		       time_index;
   int dimsl[2], dimsu[2];
   struct sidl_int__array* bHYPRE_grid_relax_points=NULL;
   bHYPRE_Vector          bHYPRE_Vector_x, bHYPRE_Vector_b;
   bHYPRE_BoomerAMG        bHYPRE_AMG;

   MPI_Comm_rank( mpi_comm, &myid );
   if (myid == 0) printf("Solver:  AMG\n");
   time_index = hypre_InitializeTiming("BoomerAMG Setup");
   hypre_BeginTiming(time_index);


   /* To call a bHYPRE solver:
      create, set comm, set operator, set other parameters,
      Setup (noop in this case), Apply */
   bHYPRE_AMG = bHYPRE_BoomerAMG_Create( bmpicomm, bHYPRE_parcsr_A );
   bHYPRE_Vector_b = bHYPRE_Vector__cast( bHYPRE_b );
   bHYPRE_Vector_x = bHYPRE_Vector__cast( bHYPRE_x );

   bHYPRE_BoomerAMG_SetDoubleParameter( bHYPRE_AMG, "Tolerance", clp->tol);
   bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "PrintLevel", clp->ioutdat ); 

   bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "CoarsenType",
                                     (clp->hybrid*clp->coarsen_type));
   bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "MeasureType",
                                     clp->measure_type);
   bHYPRE_BoomerAMG_SetDoubleParameter( bHYPRE_AMG, "StrongThreshold",
                                        clp->strong_threshold);
   bHYPRE_BoomerAMG_SetDoubleParameter( bHYPRE_AMG, "TruncFactor",
                                        clp->trunc_factor);
   bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "CycleType", clp->cycle_type);
   bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Cycle0NumSweeps",
                                     (clp->num_grid_sweeps)[0] );
   bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Cycle1NumSweeps",
                                     (clp->num_grid_sweeps)[1] );
   bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Cycle2NumSweeps",
                                     (clp->num_grid_sweeps)[2] );
   bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Cycle3NumSweeps",
                                     (clp->num_grid_sweeps)[3] );
   bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Cycle0RelaxType",
                                     (clp->grid_relax_type)[0] );
   bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Cycle1RelaxType",
                                     (clp->grid_relax_type)[1] );
   bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Cycle2RelaxType",
                                     (clp->grid_relax_type)[2] );
   bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Cycle3RelaxType",
                                     (clp->grid_relax_type)[3] );
   for ( i=0; i<clp->max_levels; ++i )
   {
      bHYPRE_BoomerAMG_SetLevelRelaxWt( bHYPRE_AMG, (clp->relax_weight)[i], i );
   }
   bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "SmoothType",
                                     clp->smooth_type );
   bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "SmoothNumSweeps",
                                     clp->smooth_num_sweep);
   dimsl[0] = 0;   dimsl[1] = 0;   dimsu[0] = 4;   dimsu[1] = 4;
   bHYPRE_grid_relax_points = sidl_int__array_createCol( 2, dimsl, dimsu );
   for ( i=0; i<4; ++i )
   {
      for ( j=0; j<(clp->num_grid_sweeps)[i]; ++j )
      {
         sidl_int__array_set2( bHYPRE_grid_relax_points, i, j,
                               (clp->grid_relax_points)[i][j] );
      }
   }
   bHYPRE_BoomerAMG_SetIntArray2Parameter( bHYPRE_AMG, "GridRelaxPoints",
                                           bHYPRE_grid_relax_points );
   sidl_int__array_deleteRef( bHYPRE_grid_relax_points );


   bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "MaxLevels", clp->max_levels);
   bHYPRE_BoomerAMG_SetDoubleParameter( bHYPRE_AMG, "MaxRowSum",
                                        clp->max_row_sum);
   bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "DebugFlag", clp->debug_flag);
   bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Variant", clp->variant);
   bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Overlap", clp->overlap);
   bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "DomainType", clp->domain_type);
   bHYPRE_BoomerAMG_SetDoubleParameter( bHYPRE_AMG,
                                        "SchwarzRlxWeight",
                                        clp->schwarz_rlx_weight);
   bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "NumFunctions",
                                     clp->num_functions);
   if (clp->num_functions > 1)
   {
      bHYPRE_BoomerAMG_SetIntArray1Parameter( bHYPRE_AMG, "DOFFunc",
                                              dof_func, clp->num_functions );
   }
   log_level = 3;
   bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Logging", log_level );

   ierr += bHYPRE_BoomerAMG_Setup( bHYPRE_AMG, bHYPRE_Vector_b,
                                   bHYPRE_Vector_x );
   hypre_EndTiming(time_index);
   hypre_PrintTiming("Setup phase times", mpi_comm);
   hypre_FinalizeTiming(time_index);
   hypre_ClearTiming();
 
   time_index = hypre_InitializeTiming("BoomerAMG Solve");
   hypre_BeginTiming(time_index);

   ierr += bHYPRE_BoomerAMG_Apply( bHYPRE_AMG, bHYPRE_Vector_b, &bHYPRE_Vector_x );

   hypre_EndTiming(time_index);
   hypre_PrintTiming("Solve phase times", mpi_comm);
   hypre_FinalizeTiming(time_index);
   hypre_ClearTiming();

   bHYPRE_BoomerAMG_deleteRef( bHYPRE_AMG );

   return ierr;
}

int PrecondAMG( CommandLineParameters *clp, int myid,
                bHYPRE_IJParCSRMatrix bHYPRE_parcsr_A,
                bHYPRE_Vector bHYPRE_Vector_b, bHYPRE_Vector bHYPRE_Vector_x,
                int * dof_func, bHYPRE_MPICommunicator bmpicomm,
                bHYPRE_Solver * bHYPRE_SolverPC )
{
   int ierr = 0;
   int dimsl[2], dimsu[2];
   int i, j;
   struct sidl_int__array* bHYPRE_grid_relax_points=NULL;
   bHYPRE_BoomerAMG bHYPRE_AMG;

   bHYPRE_AMG = bHYPRE_BoomerAMG_Create( bmpicomm, bHYPRE_parcsr_A );
   bHYPRE_BoomerAMG_SetDoubleParameter( bHYPRE_AMG, "Tolerance", clp->pc_tol);
   bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "InterpType",
                                     (clp->hybrid*clp->interp_type));
   bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "NumSamples",
                                     (clp->hybrid*clp->gsmg_samples));
   bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "CoarsenType",
                                     (clp->hybrid*clp->coarsen_type));
   bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "MeasureType",
                                     clp->measure_type);
   bHYPRE_BoomerAMG_SetDoubleParameter( bHYPRE_AMG, "StrongThreshold",
                                        clp->strong_threshold);
   bHYPRE_BoomerAMG_SetDoubleParameter( bHYPRE_AMG, "TruncFactor",
                                        clp->trunc_factor);
   bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "PrintLevel", clp->poutdat );
   bHYPRE_BoomerAMG_SetStringParameter( bHYPRE_AMG, "PrintFileName",
                                        "driver.out.log" );
   bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "MaxIter", 1 );
   bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "CycleType", clp->cycle_type );
   bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Cycle0NumSweeps",
                                     (clp->num_grid_sweeps)[0] );
   bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Cycle1NumSweeps",
                                     (clp->num_grid_sweeps)[1] );
   bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Cycle2NumSweeps",
                                     (clp->num_grid_sweeps)[2] );
   bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Cycle3NumSweeps",
                                     (clp->num_grid_sweeps)[3] );
   bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Cycle0RelaxType",
                                     (clp->grid_relax_type)[0] );
   bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Cycle1RelaxType",
                                     (clp->grid_relax_type)[1] );
   bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Cycle2RelaxType",
                                     (clp->grid_relax_type)[2] );
   bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Cycle3RelaxType",
                                     (clp->grid_relax_type)[3] );
   for ( i=0; i<clp->max_levels; ++i )
   {
      bHYPRE_BoomerAMG_SetLevelRelaxWt( bHYPRE_AMG, (clp->relax_weight)[i], i );
   }
   bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "SmoothType",
                                     clp->smooth_type );
   bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "SmoothNumSweeps",
                                     clp->smooth_num_sweep);

   dimsl[0] = 0;   dimsl[1] = 0;   dimsu[0] = 4;   dimsu[1] = 4;
   bHYPRE_grid_relax_points = sidl_int__array_createCol( 2, dimsl, dimsu );
   for ( i=0; i<4; ++i )
   {
      for ( j=0; j<(clp->num_grid_sweeps)[i]; ++j )
      {
         sidl_int__array_set2( bHYPRE_grid_relax_points, i, j,
                               (clp->grid_relax_points)[i][j] );
      }
   }
   bHYPRE_BoomerAMG_SetIntArray2Parameter( bHYPRE_AMG, "GridRelaxPoints",
                                           bHYPRE_grid_relax_points );
   sidl_int__array_deleteRef( bHYPRE_grid_relax_points );

   bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "MaxLevels", clp->max_levels);
   bHYPRE_BoomerAMG_SetDoubleParameter( bHYPRE_AMG, "MaxRowSum",
                                        clp->max_row_sum);
   bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "NumFunctions",
                                     clp->num_functions);
   if (clp->num_functions > 1)
   {
      bHYPRE_BoomerAMG_SetIntArray1Parameter( bHYPRE_AMG, "DOFFunc",
                                              dof_func, clp->num_functions );
   }
   bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Variant", clp->variant);
   bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Overlap", clp->overlap);
   bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "DomainType", clp->domain_type);
   bHYPRE_BoomerAMG_SetDoubleParameter( bHYPRE_AMG,
                                        "SchwarzRlxWeight",
                                        clp->schwarz_rlx_weight);
   *bHYPRE_SolverPC = bHYPRE_Solver__cast( bHYPRE_AMG );
   return ierr;
}
