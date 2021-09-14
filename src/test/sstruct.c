/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "_hypre_utilities.h"
#include "_hypre_parcsr_mv.h" // Delete this


#include "HYPRE_sstruct_ls.h"
#include "HYPRE_struct_ls.h"
#include "HYPRE_krylov.h"
#include "_hypre_sstruct_mv.h"
#include "sstruct_helpers.h"

/* begin lobpcg */

#include <time.h>

#include "HYPRE_lobpcg.h"

#define NO_SOLVER -9198

/* end lobpcg */

#define DEBUG 0
#define DEBUG_SSGRAPH 0

char infile_default[50] = "sstruct.in.default";

/*--------------------------------------------------------------------------
 * Print usage info
 *--------------------------------------------------------------------------*/

HYPRE_Int
PrintUsage( char *progname,
            HYPRE_Int   myid )
{
   if ( myid == 0 )
   {
      hypre_printf("\n");
      hypre_printf("Usage: %s [-in <filename>] [<options>]\n", progname);
      hypre_printf("       %s -help | -version | -vernum \n", progname);
      hypre_printf("\n");
      hypre_printf("  -in <filename> : input file (default is `%s')\n",
                   infile_default);
      hypre_printf("\n");
      hypre_printf("  -pt <pt1> <pt2> ... : set part(s) for subsequent options\n");
      hypre_printf("  -pooldist <p>       : pool distribution to use\n");
      hypre_printf("  -r <rx> <ry> <rz>   : refine part(s)\n");
      hypre_printf("  -P <Px> <Py> <Pz>   : refine and distribute part(s)\n");
      hypre_printf("  -b <bx> <by> <bz>   : refine and block part(s)\n");
      hypre_printf("  -solver <ID>        : solver ID (default = 39)\n");
      hypre_printf("                        -3 - ParCSR  Matvec\n");
      hypre_printf("                        -2 - Struct  Matvec\n");
      hypre_printf("                        -1 - SStruct Matvec\n");
      hypre_printf("                         0 - SMG split solver\n");
      hypre_printf("                         1 - PFMG split solver\n");
      hypre_printf("                         3 - SysPFMG\n");
      hypre_printf("                         4 - SSAMG\n");
      hypre_printf("                         5 - BoomerAMG\n");
      hypre_printf("                         8 - 1-step Jacobi split solver\n");
      hypre_printf("                        10 - PCG with SMG split precond\n");
      hypre_printf("                        11 - PCG with PFMG split precond\n");
      hypre_printf("                        13 - PCG with SysPFMG precond\n");
      hypre_printf("                        14 - PCG with SSAMG precond\n");
      hypre_printf("                        18 - PCG with diagonal scaling\n");
      hypre_printf("                        19 - PCG\n");
      hypre_printf("                        20 - PCG with BoomerAMG precond\n");
      hypre_printf("                        21 - PCG with EUCLID precond\n");
      hypre_printf("                        22 - PCG with ParaSails precond\n");
      hypre_printf("                        28 - PCG with diagonal scaling\n");
      hypre_printf("                        30 - GMRES with SMG split precond\n");
      hypre_printf("                        31 - GMRES with PFMG split precond\n");
      hypre_printf("                        33 - GMRES with SysPFMG precond\n");
      hypre_printf("                        34 - GMRES with SSAMG precond\n");
      hypre_printf("                        38 - GMRES with diagonal scaling\n");
      hypre_printf("                        39 - GMRES\n");
      hypre_printf("                        40 - GMRES with BoomerAMG precond\n");
      hypre_printf("                        41 - GMRES with EUCLID precond\n");
      hypre_printf("                        42 - GMRES with ParaSails precond\n");
      hypre_printf("                        50 - BiCGSTAB with SMG split precond\n");
      hypre_printf("                        51 - BiCGSTAB with PFMG split precond\n");
      hypre_printf("                        53 - BiCGSTAB with SysPFMG precond\n");
      hypre_printf("                        54 - BiCGSTAB with SSAMG precond\n");
      hypre_printf("                        58 - BiCGSTAB with diagonal scaling\n");
      hypre_printf("                        59 - BiCGSTAB\n");
      hypre_printf("                        60 - BiCGSTAB with BoomerAMG precond\n");
      hypre_printf("                        61 - BiCGSTAB with EUCLID precond\n");
      hypre_printf("                        62 - BiCGSTAB with ParaSails precond\n");
      hypre_printf("                        70 - Flexible GMRES with SMG split precond\n");
      hypre_printf("                        71 - Flexible GMRES with PFMG split precond\n");
      hypre_printf("                        73 - Flexible GMRES with SysPFMG precond\n");
      hypre_printf("                        74 - Flexible GMRES with SSAMG precond\n");
      hypre_printf("                        78 - Flexible GMRES with diagonal scaling\n");
      hypre_printf("                        80 - Flexible GMRES with BoomerAMG precond\n");
      hypre_printf("                        90 - LGMRES with BoomerAMG precond\n");
      hypre_printf("                        120- ParCSRHybrid with DSCG/BoomerAMG precond\n");
      hypre_printf("                        150- AMS solver\n");
      hypre_printf("                        200- Struct SMG\n");
      hypre_printf("                        201- Struct PFMG\n");
      hypre_printf("                        203- Struct PFMG constant coefficients\n");
      hypre_printf("                        204- Struct PFMG constant coefficients variable diagonal\n");
      hypre_printf("                        205- Struct Cyclic Reduction\n");
      hypre_printf("                        208- Struct Jacobi\n");
      hypre_printf("                        210- Struct CG with SMG precond\n");
      hypre_printf("                        211- Struct CG with PFMG precond\n");
      hypre_printf("                        217- Struct CG with 2-step Jacobi\n");
      hypre_printf("                        218- Struct CG with diagonal scaling\n");
      hypre_printf("                        219- Struct CG\n");
      hypre_printf("                        220- Struct Hybrid with SMG precond\n");
      hypre_printf("                        221- Struct Hybrid with PFMG precond\n");
      hypre_printf("                        230- Struct GMRES with SMG precond\n");
      hypre_printf("                        231- Struct GMRES with PFMG precond\n");
      hypre_printf("                        237- Struct GMRES with 2-step Jacobi\n");
      hypre_printf("                        238- Struct GMRES with diagonal scaling\n");
      hypre_printf("                        239- Struct GMRES\n");
      hypre_printf("                        240- Struct BiCGSTAB with SMG precond\n");
      hypre_printf("                        241- Struct BiCGSTAB with PFMG precond\n");
      hypre_printf("                        247- Struct BiCGSTAB with 2-step Jacobi\n");
      hypre_printf("                        248- Struct BiCGSTAB with diagonal scaling\n");
      hypre_printf("                        249- Struct BiCGSTAB\n");
      hypre_printf("  -reps              : number of times to repeat\n");
      hypre_printf("  -sym               : check symmetry of matrix A\n");
      hypre_printf("  -Aones             : compute A times vector of ones\n");
      hypre_printf("  -print             : print out the system\n");
      hypre_printf("  -rhsfromcosine     : solution is cosine function (default)\n");
      hypre_printf("  -rhszero           : rhs vector has zero components\n");
      hypre_printf("  -rhsone            : rhs vector has unit components\n");
      hypre_printf("  -x0zero            : initial solution (x0) has zero components \n");
      hypre_printf("  -x0one             : initial solution (x0) has unit components \n");
      hypre_printf("  -x0rand            : initial solution (x0) has random components \n");
      hypre_printf("  -xone              : solution (x) is vector with unit components\n");
      hypre_printf("  -tol <val>         : convergence tolerance (def 1e-9)\n");
      hypre_printf("  -solver_type <ID>  : Solver type for Hybrid\n");
      hypre_printf("                        1 - PCG (default)\n");
      hypre_printf("                        2 - GMRES\n");
      hypre_printf("                        3 - BiCGSTAB (only ParCSRHybrid)\n");
      hypre_printf("  -recompute <bool>  : Recompute residual in PCG?\n");
      hypre_printf("  -final_res <bool>  : Compute final residual (def 0) \n");
      hypre_printf("  -itr <val>         : maximum number of iterations (def 100);\n");
      hypre_printf("  -k <val>           : dimension Krylov space for GMRES (def 10);\n");
      hypre_printf("  -aug <val>         : number of augmentation vectors LGMRES (def 2);\n");
      hypre_printf("  -rel_change        : conv based on relative change of x (def 0);\n");
      hypre_printf("  -kprint <val>      : print level for krylov solvers  (def 2);\n");
      hypre_printf("  -plevel <val>      : print level for prec/solvers (def 1);\n");
      hypre_printf("  -pfreq <val>       : print frequency for prec/solvers (def 1);\n");
      hypre_printf("  -lvl <val>         : maximum number of levels (default 100);\n");
      hypre_printf("  -v <n_pre>         : # of pre-relaxation sweeps (def 1)\n");
      hypre_printf("     <n_post>        : # of pos-relaxation sweeps (def 1)\n");
      hypre_printf("     <n_coarse>      : # of coarse grid solver sweeps (def 1)\n");
      hypre_printf("  -max_coarse <val>  : maximum coarse size (def 1) \n");
      hypre_printf("  -csolver <ID>      : SSAMG - Coarse solver type\n");
      hypre_printf("                        0 - Weighted Jacobi (default)\n");
      hypre_printf("                        1 - BoomerAMG\n");
      hypre_printf("  -skip <s>          : PFMG, SysPFMG and SSAMG- skip relaxation (0 or 1)\n");
      hypre_printf("  -rap <r>           : coarse grid operator type\n");
      hypre_printf("                        0 - Galerkin (default)\n");
      hypre_printf("                        1 - non-Galerkin ParFlow operators\n");
      hypre_printf("                        2 - Galerkin, general operators\n");
      hypre_printf("  -relax <r>         : (S)Struct - relaxation type\n");
      hypre_printf("                        0 - Jacobi\n");
      hypre_printf("                        1 - Weighted Jacobi (default)\n");
      hypre_printf("                        2 - R/B Gauss-Seidel\n");
      hypre_printf("                        3 - R/B Gauss-Seidel (nonsymmetric)\n");
      hypre_printf("\n");
      hypre_printf("                       ParCSR - relaxation type\n");
      hypre_printf("                        0 - Weighted Jacobi\n");
      hypre_printf("                        1 - Gauss-Seidel (very slow!)\n");
      hypre_printf("                        3 - Hybrid Gauss-Seidel\n");
      hypre_printf("                        4 - Hybrid backward Gauss-Seidel\n");
      hypre_printf("                        6 - Hybrid symmetric Gauss-Seidel\n");
      hypre_printf("                        8 - symmetric L1-Gauss-Seidel\n");
      hypre_printf("                       13 - forward L1-Gauss-Seidel\n");
      hypre_printf("                       14 - backward L1-Gauss-Seidel\n");
      hypre_printf("                       15 - CG\n");
      hypre_printf("                       16 - Chebyshev\n");
      hypre_printf("                       17 - FCF-Jacobi\n");
      hypre_printf("                       18 - L1-Jacobi (may be used with -CF)\n");
      hypre_printf("                        9 - Gauss elimination (coarsest grid only) \n");
      hypre_printf("                       99 - Gauss elim. with pivoting (coarsest grid)\n");
      hypre_printf("  -rlx_coarse  <val> : ParCSR - set relaxation type for coarsest grid\n");
      hypre_printf("  -rlx_down    <val> : ParCSR - set relaxation type for down cycle\n");
      hypre_printf("  -rlx_up      <val> : ParCSR - set relaxation type for up cycle\n");
      hypre_printf("  -agg_nl <val>      : ParCSR - set number of agg. coarsening levels (0)\n");
      hypre_printf("  -w <jacobi_weight> : jacobi weight\n");
      hypre_printf("  -solver_type <ID>  : Struct- solver type for Hybrid\n");
      hypre_printf("                        1 - PCG (default)\n");
      hypre_printf("                        2 - GMRES\n");
      hypre_printf("  -cf <cf>           : Struct- convergence factor for Hybrid\n");
      hypre_printf("  -crtdim <tdim>     : Struct- cyclic reduction tdim\n");
      hypre_printf("  -cri <ix> <iy> <iz>: Struct- cyclic reduction base_index\n");
      hypre_printf("  -crs <sx> <sy> <sz>: Struct- cyclic reduction base_stride\n");
      hypre_printf("  -old_default       : sets old BoomerAMG defaults, possibly better for 2D problems\n");
      hypre_printf("  -vis               : save the solution for GLVis visualization");
      hypre_printf("  -seed <val>        : use <val> as the seed for the pseudo-random number generator\n");
      hypre_printf("                       (default seed is based on the time of the run)\n");


      /* begin lobpcg */

      hypre_printf("\nLOBPCG options:\n");
      hypre_printf("\n");
      hypre_printf("  -lobpcg            : run LOBPCG instead of PCG\n");
      hypre_printf("\n");
      hypre_printf("  -solver none       : no HYPRE preconditioner is used\n");
      hypre_printf("\n");
      hypre_printf("  -itr <val>         : maximal number of LOBPCG iterations (default 100);\n");
      hypre_printf("\n");
      hypre_printf("  -vrand <val>       : compute <val> eigenpairs using random initial vectors (default 1)\n");
      hypre_printf("\n");
      hypre_printf("  -seed <val>        : use <val> as the seed for the pseudo-random number generator\n");
      hypre_printf("                       (default seed is based on the time of the run)\n");
      hypre_printf("\n");
      hypre_printf("  -orthchk           : check eigenvectors for orthonormality\n");
      hypre_printf("\n");
      hypre_printf("  -verb <val>        : verbosity level\n");
      hypre_printf("  -verb 0            : no print\n");
      hypre_printf("  -verb 1            : print initial eigenvalues and residuals, iteration number, number of\n");
      hypre_printf("                       non-convergent eigenpairs and final eigenvalues and residuals (default)\n");
      hypre_printf("  -verb 2            : print eigenvalues and residuals on each iteration\n");
      hypre_printf("\n");
      hypre_printf("  -pcgitr <val>      : maximum number of inner PCG iterations for preconditioning (default 1);\n");
      hypre_printf("                       if <val> = 0 then the preconditioner is applied directly\n");
      hypre_printf("\n");
      hypre_printf("  -pcgtol <val>      : residual tolerance for inner iterations (default 0.01)\n");
      hypre_printf("\n");
      hypre_printf("  -vout <val>        : file output level\n");
      hypre_printf("  -vout 0            : no files created (default)\n");
      hypre_printf("  -vout 1            : write eigenvalues to values.txt and residuals to residuals.txt\n");
      hypre_printf("  -vout 2            : in addition to the above, write the eigenvalues history (the matrix whose\n");
      hypre_printf("                       i-th column contains eigenvalues at (i+1)-th iteration) to val_hist.txt and\n");
      hypre_printf("                       residuals history to res_hist.txt\n");
      hypre_printf("\nNOTE: in this test driver LOBPCG only works with solvers 10, 11, 13, and 18\n");
      hypre_printf("\ndefault solver is 10\n");

      /* end lobpcg */

      hypre_printf("\n");
   }

   return 0;
}

/*--------------------------------------------------------------------------
 * Test driver for semi-structured matrix interface
 *--------------------------------------------------------------------------*/

hypre_int
main( hypre_int argc,
      char *argv[] )
{
   MPI_Comm              comm = hypre_MPI_COMM_WORLD;

   char                 *infile;
   ProblemData           global_data;
   ProblemData           data;
   ProblemPartData       pdata;
   HYPRE_Int             nparts;
   HYPRE_Int             pooldist;
   HYPRE_Int            *parts;
   Index                *refine;
   Index                *distribute;
   Index                *block;
   HYPRE_Int             solver_id, object_type;
   HYPRE_Int             print_system;
   HYPRE_Int             check_symmetry;
   HYPRE_Int             check_Aones;
   HYPRE_Int             sol_type;
   HYPRE_Int             sol0_type;
   HYPRE_Real            rhs_value;
   HYPRE_Real            scale;

   HYPRE_SStructGrid     grid, G_grid;
   HYPRE_SStructStencil *stencils, *G_stencils;
   HYPRE_SStructGraph    graph, G_graph;
   HYPRE_SStructMatrix   A, G;
   HYPRE_SStructVector   b;
   HYPRE_SStructVector   r;
   HYPRE_SStructVector   x;
   HYPRE_SStructSolver   solver;
   HYPRE_SStructSolver   precond;

   HYPRE_IJMatrix        ij_A;
   HYPRE_ParCSRMatrix    par_A;
   HYPRE_ParVector       par_b;
   HYPRE_ParVector       par_x;
   HYPRE_Solver          par_solver;
   HYPRE_Solver          par_precond;

   HYPRE_StructMatrix    sA;
   HYPRE_StructVector    sb;
   HYPRE_StructVector    sx;
   HYPRE_StructSolver    struct_solver;
   HYPRE_StructSolver    struct_precond;

   Index                 ilower, iupper;
   HYPRE_Real           *values;

   HYPRE_Int             num_iterations;
   HYPRE_Real            final_res_norm;
   HYPRE_Real            real_res_norm;
   HYPRE_Real            rhs_norm;
   HYPRE_Real            x0_norm;

   HYPRE_Int             num_procs, myid;
   HYPRE_Int             time_index;

   /* parameters for multigrid */
   HYPRE_Real            jacobi_weight;
   HYPRE_Real            strong_threshold;
   HYPRE_Int             P_max_elmts;
   HYPRE_Int             coarsen_type;
   HYPRE_Int             usr_jacobi_weight;
   HYPRE_Int             rap;
   HYPRE_Int             max_levels;
   HYPRE_Int             n_pre, n_post, n_coarse;
   HYPRE_Int             relax[4];
   HYPRE_Int             relax_is_set;
   HYPRE_Int             max_coarse_size;
   HYPRE_Int             csolver_type;
   HYPRE_Int             skip;
   HYPRE_Int             agg_num_levels;

   /* parameters for Solvers */
   HYPRE_Int             rel_change;
   HYPRE_Int             solver_type;
   HYPRE_Int             recompute_res;
   HYPRE_Int             final_res;
   HYPRE_Int             max_iterations;
   HYPRE_Int             krylov_print_level;
   HYPRE_Int             print_level;
   HYPRE_Int             print_freq;
   HYPRE_Real            tol;

   /* parameters for GMRES */
   HYPRE_Int	         k_dim;

   /* parameters for LGMRES */
   HYPRE_Int	         aug_dim;

   /* Misc */
   HYPRE_Int             vis;
   HYPRE_Int             seed;
   HYPRE_Int             reps;

   HYPRE_Real            cf_tol;

   HYPRE_Int             cycred_tdim;
   Index                 cycred_index, cycred_stride;

   HYPRE_Int             arg_index, part, var, box, s, entry, i, j, k;
   HYPRE_Int             gradient_matrix;

   /* begin lobpcg */

   HYPRE_SStructSolver      lobpcg_solver;

   HYPRE_Int                lobpcgFlag = 0;
   HYPRE_Int                lobpcgSeed = 0;
   HYPRE_Int                blockSize = 1;
   HYPRE_Int                verbosity = 1;
   HYPRE_Int                iterations;
   HYPRE_Int                maxIterations = 100;
   HYPRE_Int                checkOrtho = 0;
   HYPRE_Int                printLevel = 0;
   HYPRE_Int                pcgIterations = 0;
   HYPRE_Int                pcgMode = 0;
   HYPRE_Int                old_default = 0;
   HYPRE_Real               pcgTol = 1e-2;
   HYPRE_Real               nonOrthF;
   HYPRE_Real              *eigenvalues = NULL;
   HYPRE_Real              *residuals;

   utilities_FortranMatrix *residualNorms;
   utilities_FortranMatrix *residualNormsHistory;
   utilities_FortranMatrix *eigenvaluesHistory;
   utilities_FortranMatrix *printBuffer;
   utilities_FortranMatrix *gramXX;
   utilities_FortranMatrix *identity;
   mv_InterfaceInterpreter *interpreter;
   mv_MultiVectorPtr        eigenvectors = NULL;
   mv_MultiVectorPtr        constrains = NULL;
   HYPRE_MatvecFunctions    matvec_fn;

   FILE                    *filePtr;

   /* end lobpcg */

#if defined(HYPRE_USING_GPU)
   HYPRE_Int spgemm_use_cusparse = 0;
#endif
   HYPRE_ExecutionPolicy default_exec_policy = HYPRE_EXEC_DEVICE;
   HYPRE_MemoryLocation memory_location = HYPRE_MEMORY_DEVICE;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   /* Initialize MPI */
   hypre_MPI_Init(&argc, &argv);

   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs);
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid);

   /*-----------------------------------------------------------------
    * GPU Device binding
    * Must be done before HYPRE_Init() and should not be changed after
    *-----------------------------------------------------------------*/
   hypre_bind_device(myid, num_procs, hypre_MPI_COMM_WORLD);

   /*-----------------------------------------------------------
    * Initialize : must be the first HYPRE function to call
    *-----------------------------------------------------------*/
   HYPRE_Init();

   /*-----------------------------------------------------------
    * Read input file
    *-----------------------------------------------------------*/

   arg_index = 1;

   /* parse command line for input file name */
   infile = infile_default;
   if (argc > 1)
   {
      if ( strcmp(argv[arg_index], "-in") == 0 )
      {
         arg_index++;
         infile = argv[arg_index++];
      }
      else if ( strcmp(argv[arg_index], "-help") == 0 )
      {
         PrintUsage(argv[0], myid);
         exit(1);
      }
      else if ( strcmp(argv[arg_index], "-version") == 0 )
      {
         char *version_string;
         HYPRE_Version(&version_string);
         hypre_printf("%s\n", version_string);
         hypre_TFree(version_string, HYPRE_MEMORY_HOST);
         exit(1);
      }
      else if ( strcmp(argv[arg_index], "-vernum") == 0 )
      {
         HYPRE_Int major, minor, patch, single;
         HYPRE_VersionNumber(&major, &minor, &patch, &single);
         hypre_printf("HYPRE Version %d.%d.%d\n", major, minor, patch);
         hypre_printf("HYPRE Single = %d\n", single);
         exit(1);
      }
   }

   /*-----------------------------------------------------------
    * Read data from input file
    *-----------------------------------------------------------*/

   ReadData(infile, &global_data);

   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/

   reps  = 10;
   skip  = 0;
   rap   = 0;
   relax_is_set = 0;
   relax[0] = 1;
   relax[1] = -1; /* Relax up */
   relax[2] = -1; /* Relax down */
   relax[3] = -1; /* Relax coarse */
   usr_jacobi_weight= 0;
   solver_type = 1;
   recompute_res = 0;   /* What should be the default here? */
   cf_tol = 0.90;
   num_iterations = -1;
   max_iterations = 100;
   max_levels = 25;
   max_coarse_size = -1; /* depends on object_type */
   csolver_type = 0;
   tol = 1.0e-6;
   rel_change = 0;
   k_dim = 5;
   aug_dim = 2;
   print_level = 1;
   print_freq = 1;
   krylov_print_level = 1;
   final_res = 0;
   final_res_norm = 0.0;

   nparts = global_data.nparts;
   pooldist = 0;

   parts      = hypre_TAlloc(HYPRE_Int,  nparts, HYPRE_MEMORY_HOST);
   refine     = hypre_TAlloc(Index,  nparts, HYPRE_MEMORY_HOST);
   distribute = hypre_TAlloc(Index,  nparts, HYPRE_MEMORY_HOST);
   block      = hypre_TAlloc(Index,  nparts, HYPRE_MEMORY_HOST);
   for (part = 0; part < nparts; part++)
   {
      parts[part] = part;
      for (j = 0; j < 3; j++)
      {
         refine[part][j]     = 1;
         distribute[part][j] = 1;
         block[part][j]      = 1;
      }
   }
   cycred_tdim = 0;
   for (i = 0; i < 3; i++)
   {
      cycred_index[i]  = 0;
      cycred_stride[i] = 1;
   }

   solver_id = 39;
   print_system = 0;
   check_symmetry = 0;
   check_Aones = 0;
   rhs_value = 1.0;
   sol_type = 0;
   sol0_type = 0;
   skip = 0;
   n_pre  = 1;
   n_post = 1;
   n_coarse = 1;
   strong_threshold = 0.25;
   P_max_elmts = 4;
   agg_num_levels = 0;
   coarsen_type = 10;
   vis = 0;
   seed = 1;

   old_default = 0;

   if (global_data.rhs_true || global_data.fem_rhs_true)
   {
      sol_type = -1;
   }

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/

   while (arg_index < argc)
   {
      if ( strcmp(argv[arg_index], "-pt") == 0 )
      {
         arg_index++;
         nparts = 0;
         while ( strncmp(argv[arg_index], "-", 1) != 0 )
         {
            parts[nparts++] = atoi(argv[arg_index++]);
         }
      }
      else if ( strcmp(argv[arg_index], "-pooldist") == 0 )
      {
         arg_index++;
         pooldist = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-r") == 0 )
      {
         arg_index++;
         for (i = 0; i < nparts; i++)
         {
            part = parts[i];
            k = arg_index;
            for (j = 0; j < 3; j++)
            {
               refine[part][j] = atoi(argv[k++]);
            }
         }
         arg_index += 3;
      }
      else if ( strcmp(argv[arg_index], "-P") == 0 )
      {
         arg_index++;
         for (i = 0; i < nparts; i++)
         {
            part = parts[i];
            k = arg_index;
            for (j = 0; j < 3; j++)
            {
               distribute[part][j] = atoi(argv[k++]);
            }
         }
         arg_index += 3;
      }
      else if ( strcmp(argv[arg_index], "-b") == 0 )
      {
         arg_index++;
         for (i = 0; i < nparts; i++)
         {
            part = parts[i];
            k = arg_index;
            for (j = 0; j < 3; j++)
            {
               block[part][j] = atoi(argv[k++]);
            }
         }
         arg_index += 3;
      }
      else if ( strcmp(argv[arg_index], "-solver") == 0 )
      {
         arg_index++;

	 /* begin lobpcg */
	 if ( strcmp(argv[arg_index], "none") == 0 ) {
            solver_id = NO_SOLVER;
            arg_index++;
	 }
	 else /* end lobpcg */
            solver_id = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-print") == 0 )
      {
         arg_index++;
         print_system = 1;
      }
      else if (strcmp(argv[arg_index], "-repeats") == 0 )
      {
         arg_index++;
         reps = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-sym") == 0 )
      {
         arg_index++;
         check_symmetry = 1;
      }
      else if ( strcmp(argv[arg_index], "-Aones") == 0 )
      {
         arg_index++;
         check_Aones = 1;
      }
      else if ( strcmp(argv[arg_index], "-vis") == 0 )
      {
         arg_index++;
         vis = 1;
      }
      else if ( strcmp(argv[arg_index], "-seed") == 0 )
      {
         arg_index++;
         seed = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-rhszero") == 0 )
      {
         arg_index++;
         rhs_value = 0.0;
         sol_type = -1;
      }
      else if ( strcmp(argv[arg_index], "-rhsone") == 0 )
      {
         arg_index++;
         rhs_value = 1.0;
         sol_type = -1;
      }
      else if ( strcmp(argv[arg_index], "-xfromcosine") == 0 )
      {
         arg_index++;
         sol_type = 0;
      }
      else if ( strcmp(argv[arg_index], "-xone") == 0 )
      {
         arg_index++;
         sol_type = 1;
      }
      else if ( strcmp(argv[arg_index], "-x0zero") == 0 )
      {
         arg_index++;
         sol0_type = 0;
      }
      else if ( strcmp(argv[arg_index], "-x0one") == 0 )
      {
         arg_index++;
         sol0_type = 1;
      }
      else if ( strcmp(argv[arg_index], "-x0rand") == 0 )
      {
         arg_index++;
         sol0_type = 2;
      }
      else if ( strcmp(argv[arg_index], "-tol") == 0 )
      {
         arg_index++;
         tol = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-itr") == 0 )
      {
         arg_index++;
         max_iterations = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-k") == 0 )
      {
         arg_index++;
         k_dim = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-aug") == 0 )
      {
         arg_index++;
         aug_dim = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-kprint") == 0 )
      {
         arg_index++;
         krylov_print_level = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-plevel") == 0 )
      {
         arg_index++;
         print_level = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-pfreq") == 0 )
      {
         arg_index++;
         print_freq = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-lvl") == 0 )
      {
         arg_index++;
         max_levels = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-v") == 0 )
      {
         arg_index++;
         n_pre = atoi(argv[arg_index++]);
         n_post = atoi(argv[arg_index++]);
         n_coarse = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-max_coarse") == 0 )
      {
         arg_index++;
         max_coarse_size = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-csolver") == 0 )
      {
         arg_index++;
         csolver_type = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-skip") == 0 )
      {
         arg_index++;
         skip = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-rap") == 0 )
      {
         arg_index++;
         rap = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-relax") == 0 )
      {
         arg_index++;
         relax[0] = atoi(argv[arg_index++]);
         relax_is_set = 1;
      }
      else if ( strcmp(argv[arg_index], "-relax_up") == 0 )
      {
         arg_index++;
         relax[1] = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-relax_down") == 0 )
      {
         arg_index++;
         relax[2] = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-relax_coarse") == 0 )
      {
         arg_index++;
         relax[3] = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-w") == 0 )
      {
         arg_index++;
         jacobi_weight= atof(argv[arg_index++]);
         usr_jacobi_weight= 1; /* flag user weight */
      }
      else if ( strcmp(argv[arg_index], "-agg_nl") == 0 )
      {
         arg_index++;
         agg_num_levels  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-th") == 0 )
      {
         arg_index++;
         strong_threshold  = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-Pmx") == 0 )
      {
         arg_index++;
         P_max_elmts  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-coarsen") == 0 )
      {
         arg_index++;
         coarsen_type = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-solver_type") == 0 )
      {
         arg_index++;
         solver_type = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-recompute") == 0 )
      {
         arg_index++;
         recompute_res = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-final_res") == 0 )
      {
         arg_index++;
         final_res = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-cf") == 0 )
      {
         arg_index++;
         cf_tol = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-crtdim") == 0 )
      {
         arg_index++;
         cycred_tdim = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-cri") == 0 )
      {
         arg_index++;
         for (i = 0; i < 3; i++)
         {
            cycred_index[i] = atoi(argv[arg_index++]);
         }
      }
      else if ( strcmp(argv[arg_index], "-crs") == 0 )
      {
         arg_index++;
         for (i = 0; i < 3; i++)
         {
            cycred_stride[i] = atoi(argv[arg_index++]);
         }
      }
      else if ( strcmp(argv[arg_index], "-old_default") == 0 )
      {
         /* uses old BoomerAMG defaults */
         arg_index++;
         old_default = 1;
      }
      /* begin lobpcg */
      else if ( strcmp(argv[arg_index], "-lobpcg") == 0 )
      {
         /* use lobpcg */
         arg_index++;
         lobpcgFlag = 1;
      }
      else if ( strcmp(argv[arg_index], "-orthchk") == 0 )
      {
         /* lobpcg: check orthonormality */
         arg_index++;
         checkOrtho = 1;
      }
      else if ( strcmp(argv[arg_index], "-verb") == 0 )
      {
         /* lobpcg: verbosity level */
         arg_index++;
         verbosity = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-vrand") == 0 )
      {                         /* lobpcg: block size */
         arg_index++;
         blockSize = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-seed") == 0 )
      {
         /* lobpcg: seed for srand */
         arg_index++;
         lobpcgSeed = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-itr") == 0 )
      {
         /* lobpcg: max # of iterations */
         arg_index++;
         maxIterations = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-pcgitr") == 0 )
      {
         /* lobpcg: max inner pcg iterations */
         arg_index++;
         pcgIterations = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-pcgtol") == 0 )
      {
         /* lobpcg: inner pcg iterations tolerance */
         arg_index++;
         pcgTol = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-pcgmode") == 0 )
      {
         /* lobpcg: initial guess for inner pcg */
         arg_index++;      /* 0: zero, otherwise rhs */
         pcgMode = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-vout") == 0 )
      {
         /* lobpcg: print level */
         arg_index++;
         old_default = 1;
      }
      else
      {
         arg_index++;
         /*break;*/
      }
   }

   /* default memory location */
   HYPRE_SetMemoryLocation(memory_location);

   /* default execution policy */
   HYPRE_SetExecutionPolicy(default_exec_policy);

   HYPRE_SetStructExecutionPolicy(HYPRE_EXEC_DEVICE);

#if defined(HYPRE_USING_GPU)
   HYPRE_SetSpGemmUseCusparse(spgemm_use_cusparse);
#endif

   if ( solver_id == 39 && lobpcgFlag )
   {
      solver_id = 10;
   }

   /* end lobpcg */

   /*-----------------------------------------------------------
    * Print driver parameters TODO
    *-----------------------------------------------------------*/
   if (myid == 0)
   {
   }

   /*-----------------------------------------------------------
    * Distribute data
    *-----------------------------------------------------------*/

   DistributeData(global_data, pooldist, refine, distribute, block,
                  num_procs, myid, &data);

   /*-----------------------------------------------------------
    * Check a few things
    *-----------------------------------------------------------*/
   if (solver_id >= 200)
   {
      pdata = data.pdata[0];
      if (nparts > 1)
      {
         if (!myid)
         {
            hypre_printf("Warning: Invalid number of parts for Struct Solver. Part 0 taken. \n");
         }
      }

      if (pdata.nvars > 1)
      {
         if (!myid)
         {
            hypre_printf("Error: Invalid number of nvars for Struct Solver \n");
         }
         exit(1);
      }
   }

   /*-----------------------------------------------------------
    * Synchronize so that timings make sense
    *-----------------------------------------------------------*/

   hypre_MPI_Barrier(comm);

   /*-----------------------------------------------------------
    * Determine object type
    *-----------------------------------------------------------*/

   object_type = HYPRE_SSTRUCT;

   /* determine if we build a gradient matrix */
   gradient_matrix = 0;
   if (solver_id == 150)
   {
      gradient_matrix = 1;
      /* for now, change solver 150 to solver 28 */
      solver_id = 28;
   }

   if ( ((solver_id >= 20) && (solver_id < 30)) ||
        ((solver_id >= 40) && (solver_id < 50)) ||
        ((solver_id >= 60) && (solver_id < 70)) ||
        ((solver_id >= 80) && (solver_id < 90)) ||
        ((solver_id >= 90) && (solver_id < 100)) ||
        (solver_id == 120) || (solver_id == 5) ||
        (solver_id == -3))
   {
      object_type = HYPRE_PARCSR;
   }

   else if (solver_id >= 200 || solver_id == -2)
   {
      object_type = HYPRE_STRUCT;
   }

   if (myid == 0)
   {
      switch (object_type)
      {
         case HYPRE_STRUCT:
            hypre_printf("Setting object type to Struct\n");
            break;

         case HYPRE_SSTRUCT:
            hypre_printf("Setting object type to SStruct\n");
            break;

         case HYPRE_PARCSR:
            hypre_printf("Setting object type to ParCSR\n");
            break;
      }
   }

   /* Change default input parameters according to the object type */
   if (object_type == HYPRE_PARCSR)
   {
      if (!relax_is_set)
      {
         relax[0] = -1;
      }

      if (max_coarse_size == -1)
      {
         max_coarse_size = 9;
      }
   }
   else if (object_type == HYPRE_STRUCT)
   {
      if (max_coarse_size == -1)
      {
         max_coarse_size = 0;
      }
   }
   else if (object_type == HYPRE_SSTRUCT)
   {
      if (max_coarse_size == -1)
      {
         max_coarse_size = 0;
      }
   }

   /*-----------------------------------------------------------
    * Set up the linear system
    *-----------------------------------------------------------*/

   time_index = hypre_InitializeTiming("SStruct Interface");
   hypre_BeginTiming(time_index);

   BuildGrid(comm, data, &grid);
   BuildStencils(data, grid, &stencils);
   BuildGraph(comm, data, grid, object_type, stencils, &graph);
   BuildMatrix(comm, data, grid, stencils, graph, &A);
   BuildVector(comm, data, grid, object_type, rhs_value, &b);

   /*-----------------------------------------------------------
    * Create solution vector
    *-----------------------------------------------------------*/

   HYPRE_SStructVectorCreate(comm, grid, &x);
   HYPRE_SStructVectorSetObjectType(x, object_type);
   HYPRE_SStructVectorInitialize(x);

   switch (sol_type)
   {
      case 0:
         /*-----------------------------------------------------------
          * Set RHS such that the solution vector is given by
          *
          *  u(part,var,i,j,k) = (part+1)*(var+1)*cosine[(i+j+k)/10]
          *-----------------------------------------------------------*/
         values = hypre_TAlloc(HYPRE_Real, hypre_max(data.max_boxsize, data.fem_nsparse),
                               HYPRE_MEMORY_HOST);

         for (part = 0; part < data.nparts; part++)
         {
            pdata = data.pdata[part];
            for (var = 0; var < pdata.nvars; var++)
            {
               scale = (part + 1.0)*(var + 1.0);
               for (box = 0; box < pdata.nboxes; box++)
               {
               /* GetVariableBox(pdata.ilowers[box], pdata.iuppers[box],
                                 pdata.vartypes[var], ilower, iupper); */
                  GetVariableBox(pdata.ilowers[box], pdata.iuppers[box],
                                 var, ilower, iupper);
                  SetCosineVector(scale, ilower, iupper, values);
                  HYPRE_SStructVectorSetBoxValues(x, part, ilower, iupper,
                                                  var, values);
               }
            }
         }

         hypre_TFree(values, HYPRE_MEMORY_HOST);
         break;

      case 1:
         HYPRE_SStructVectorSetConstantValues(x, 1.0);
         break;
   }

   HYPRE_SStructVectorAssemble(x);

//   /*-----------------------------------------------------------
//    * RDF: Temporary test in case SStructMatrixMatvec still has a bug...
//    *
//    * Get the objects out
//    * NOTE: This should go after the cosine part, but for the bug
//    *-----------------------------------------------------------*/
//
//   if (object_type == HYPRE_PARCSR)
//   {
//      HYPRE_SStructMatrixGetObject(A, (void **) &par_A);
//      HYPRE_SStructVectorGetObject(b, (void **) &par_b);
//      HYPRE_SStructVectorGetObject(x, (void **) &par_x);
//   }
//   else if (object_type == HYPRE_STRUCT)
//   {
//      HYPRE_SStructMatrixGetObject(A, (void **) &sA);
//      HYPRE_SStructVectorGetObject(b, (void **) &sb);
//      HYPRE_SStructVectorGetObject(x, (void **) &sx);
//   }
//
//   if (sol_type == 0 || sol_type == 1)
//   {
//      /* This if/else is due to a bug in SStructMatvec */
//      if (object_type == HYPRE_SSTRUCT)
//      {
//         /* Apply A to x to yield righthand side */
//         hypre_SStructMatvec(1.0, A, x, 0.0, b);
//      }
//      else if (object_type == HYPRE_PARCSR)
//      {
//         /* Apply A to x to yield righthand side */
//         HYPRE_ParCSRMatrixMatvec(1.0, par_A, par_x, 0.0, par_b );
//      }
//      else if (object_type == HYPRE_STRUCT)
//      {
//         /* Apply A to x to yield righthand side */
//         hypre_StructMatvec(1.0, sA, sx, 0.0, sb);
//      }
//   }

   if (sol_type == 0 || sol_type == 1)
   {
      HYPRE_SStructMatrixMatvec(1.0, A, x, 0.0, b);
   }

   HYPRE_SStructVectorDestroy(x);

   /*-----------------------------------------------------------
    * Set initial solution
    *-----------------------------------------------------------*/

   HYPRE_SStructVectorCreate(comm, grid, &x);
   HYPRE_SStructVectorSetObjectType(x, object_type);
   HYPRE_SStructVectorInitialize(x);
   switch (sol0_type)
   {
      case 0:
         HYPRE_SStructVectorSetConstantValues(x, 0.0);
         break;

      case 1:
         HYPRE_SStructVectorSetConstantValues(x, 1.0);
         break;

      case 2:
         HYPRE_SStructVectorSetRandomValues(x, seed);
         break;
   }
   HYPRE_SStructVectorAssemble(x);

   /* Compute norms */
   HYPRE_SStructInnerProd(b, b, &rhs_norm);
   HYPRE_SStructInnerProd(x, x, &x0_norm);
   rhs_norm = sqrt(rhs_norm);
   x0_norm = sqrt(x0_norm);

   /*-----------------------------------------------------------
    * Build residual vector
    *-----------------------------------------------------------*/
   HYPRE_SStructVectorCreate(comm, grid, &r);
   HYPRE_SStructVectorSetObjectType(r, object_type);
   HYPRE_SStructVectorInitialize(r);
   HYPRE_SStructVectorAssemble(r);
   if (print_system)
   {
      HYPRE_SStructVectorCopy(b, r);
      HYPRE_SStructMatrixMatvec(-1.0, A, x, 1.0, r);
      HYPRE_SStructVectorPrint("sstruct.out.r0", r, 0);
   }

   hypre_EndTiming(time_index);
   hypre_PrintTiming("SStruct Interface", comm);
   hypre_FinalizeTiming(time_index);
   hypre_ClearTiming();

   /*-----------------------------------------------------------
    * Get the objects out
    * NOTE: This should go after the cosine part, but for the bug
    *-----------------------------------------------------------*/

   if (object_type == HYPRE_PARCSR)
   {
      HYPRE_SStructMatrixGetObject(A, (void **) &par_A);
      HYPRE_SStructVectorGetObject(b, (void **) &par_b);
      HYPRE_SStructVectorGetObject(x, (void **) &par_x);
   }
   else if (object_type == HYPRE_STRUCT)
   {
      HYPRE_SStructMatrixGetObject(A, (void **) &sA);
      HYPRE_SStructVectorGetObject(b, (void **) &sb);
      HYPRE_SStructVectorGetObject(x, (void **) &sx);
   }

   /*-----------------------------------------------------------
    * Set up a gradient matrix G
    *-----------------------------------------------------------*/

   if (gradient_matrix)
   {
      HYPRE_SStructVariable vartypes[1] = {HYPRE_SSTRUCT_VARIABLE_NODE};
      HYPRE_Int offsets[3][2][3] = { {{0,0,0}, {-1,0,0}},
                                     {{0,0,0}, {0,-1,0}},
                                     {{0,0,0}, {0,0,-1}} };
      HYPRE_Real stencil_values[2] = {1.0, -1.0};

      /* Set up the domain grid */

      HYPRE_SStructGridCreate(comm, data.ndim, data.nparts, &G_grid);
      for (part = 0; part < data.nparts; part++)
      {
         pdata = data.pdata[part];
         for (box = 0; box < pdata.nboxes; box++)
         {
            HYPRE_SStructGridSetExtents(G_grid, part,
                                        pdata.ilowers[box], pdata.iuppers[box]);
         }
         HYPRE_SStructGridSetVariables(G_grid, part, 1, vartypes);
         for (box = 0; box < pdata.glue_nboxes; box++)
         {
            if (pdata.glue_shared[box])
            {
               HYPRE_SStructGridSetSharedPart(G_grid, part,
                                              pdata.glue_ilowers[box],
                                              pdata.glue_iuppers[box],
                                              pdata.glue_offsets[box],
                                              pdata.glue_nbor_parts[box],
                                              pdata.glue_nbor_ilowers[box],
                                              pdata.glue_nbor_iuppers[box],
                                              pdata.glue_nbor_offsets[box],
                                              pdata.glue_index_maps[box],
                                              pdata.glue_index_dirs[box]);
            }
            else
            {
               HYPRE_SStructGridSetNeighborPart(G_grid, part,
                                                pdata.glue_ilowers[box],
                                                pdata.glue_iuppers[box],
                                                pdata.glue_nbor_parts[box],
                                                pdata.glue_nbor_ilowers[box],
                                                pdata.glue_nbor_iuppers[box],
                                                pdata.glue_index_maps[box],
                                                pdata.glue_index_dirs[box]);
            }
         }
      }
      HYPRE_SStructGridAssemble(G_grid);

      /* Set up the gradient stencils */

      G_stencils = hypre_CTAlloc(HYPRE_SStructStencil, data.ndim, HYPRE_MEMORY_HOST);
      for (s = 0; s < data.ndim; s++)
      {
         HYPRE_SStructStencilCreate(data.ndim, 2, &G_stencils[s]);
         for (entry = 0; entry < 2; entry++)
         {
            HYPRE_SStructStencilSetEntry(
               G_stencils[s], entry, offsets[s][entry], 0);
         }
      }

      /* Set up the gradient graph */

      HYPRE_SStructGraphCreate(comm, grid, &G_graph);
      HYPRE_SStructGraphSetDomainGrid(G_graph, G_grid);
      HYPRE_SStructGraphSetObjectType(G_graph, HYPRE_PARCSR);
      for (part = 0; part < data.nparts; part++)
      {
         pdata = data.pdata[part];
         for (var = 0; var < data.ndim; var++)
         {
            HYPRE_SStructGraphSetStencil(G_graph, part, var, G_stencils[var]);
         }
      }
      HYPRE_SStructGraphAssemble(G_graph);

      /* Set up the matrix */

      HYPRE_SStructMatrixCreate(comm, G_graph, &G);
      HYPRE_SStructMatrixSetObjectType(G, HYPRE_PARCSR);
      HYPRE_SStructMatrixInitialize(G);
      for (part = 0; part < data.nparts; part++)
      {
         pdata = data.pdata[part];
         for (var = 0; var < data.ndim; var++)
         {
            for (i = 0; i < 2; i++)
            {
               for (j = 0; j < pdata.max_boxsize; j++)
               {
                  values[j] = stencil_values[i];
               }
               for (box = 0; box < pdata.nboxes; box++)
               {
                  GetVariableBox(pdata.ilowers[box], pdata.iuppers[box],
                                 pdata.vartypes[var], ilower, iupper);
                  HYPRE_SStructMatrixSetBoxValues(G, part, ilower, iupper,
                                                  var, 1, &i, values);
               }
            }
         }
      }

      HYPRE_SStructMatrixAssemble(G);
   }

   /*-----------------------------------------------------------
    * Convert SStructMatrix to IJMatrix
    *-----------------------------------------------------------*/

   if (print_system || check_symmetry)
   {
      HYPRE_SStructMatrixToIJMatrix(A, 0, &ij_A);
   }

   /*-----------------------------------------------------------
    * Print out the system and initial guess
    *-----------------------------------------------------------*/

   if (print_system)
   {
      HYPRE_SStructVectorGather(b);
      HYPRE_SStructVectorGather(x);
      HYPRE_SStructMatrixPrint("sstruct.out.A",  A, 0);
      HYPRE_SStructVectorPrint("sstruct.out.b",  b, 0);
      HYPRE_SStructVectorPrint("sstruct.out.x0", x, 0);

      if (gradient_matrix)
      {
         HYPRE_SStructMatrixPrint("sstruct.out.G",  G, 0);
      }

      if (object_type != HYPRE_PARCSR)
      {
         HYPRE_IJMatrixPrint(ij_A, "IJ.out.A");
      }
   }

   if (check_symmetry)
   {
      HYPRE_IJMatrix  ij_AT, ij_B;
      HYPRE_Real      B_norm;

      /* Compute Frobenius norm of (A - A^T) */
      HYPRE_IJMatrixTranspose(ij_A, &ij_AT);
      HYPRE_IJMatrixAdd(1.0, ij_A, -1.0, ij_AT, &ij_B);
      HYPRE_IJMatrixNorm(ij_B, &B_norm);
      HYPRE_IJMatrixPrint(ij_B, "IJ.out.B");
      if (!myid)
      {
         hypre_printf("Frobenius norm (A - A^T) = %20.15e\n\n", B_norm);
      }

      /* Free memory */
      HYPRE_IJMatrixDestroy(ij_AT);
      HYPRE_IJMatrixDestroy(ij_B);
   }

   if (check_Aones)
   {
      HYPRE_SStructVector ones;
      HYPRE_SStructVector Aones;

      HYPRE_SStructVectorCreate(comm, grid, &ones);
      HYPRE_SStructVectorInitialize(ones);
      HYPRE_SStructVectorSetConstantValues(ones, 1.0);
      HYPRE_SStructVectorAssemble(ones);

      HYPRE_SStructVectorCreate(comm, grid, &Aones);
      HYPRE_SStructVectorInitialize(Aones);
      HYPRE_SStructVectorAssemble(Aones);

      HYPRE_SStructMatrixMatvec(1.0, A, ones, 0.0, Aones);
      HYPRE_SStructVectorPrint("sstruct.out.Aones", Aones, 0);

      HYPRE_SStructVectorDestroy(ones);
      HYPRE_SStructVectorDestroy(Aones);
   }

   /*-----------------------------------------------------------
    * Debugging code
    *-----------------------------------------------------------*/

#if DEBUG
   {
      FILE *file;
      char  filename[255];

      /* result is 1's on the interior of the grid */
      hypre_SStructMatvec(1.0, A, b, 0.0, x);
      HYPRE_SStructVectorPrint("sstruct.out.matvec", x, 0);

      /* result is all 1's */
      hypre_SStructCopy(b, x);
      HYPRE_SStructVectorPrint("sstruct.out.copy", x, 0);

      /* result is all 2's */
      hypre_SStructScale(2.0, x);
      HYPRE_SStructVectorPrint("sstruct.out.scale", x, 0);

      /* result is all 0's */
      hypre_SStructAxpy(-2.0, b, x);
      HYPRE_SStructVectorPrint("sstruct.out.axpy", x, 0);

      /* result is 1's with 0's on some boundaries */
      hypre_SStructCopy(b, x);
      hypre_sprintf(filename, "sstruct.out.gatherpre.%05d", myid);
      file = fopen(filename, "w");
      for (part = 0; part < data.nparts; part++)
      {
         pdata = data.pdata[part];
         for (var = 0; var < pdata.nvars; var++)
         {
            for (box = 0; box < pdata.nboxes; box++)
            {
               GetVariableBox(pdata.ilowers[box], pdata.iuppers[box],
                              pdata.vartypes[var], ilower, iupper);
               HYPRE_SStructVectorGetBoxValues(x, part, ilower, iupper,
                                               var, values);
               hypre_fprintf(file, "\nPart %d, var %d, box %d:\n", part, var, box);
               for (i = 0; i < pdata.boxsizes[box]; i++)
               {
                  hypre_fprintf(file, "%e\n", values[i]);
               }
            }
         }
      }
      fclose(file);

      /* result is all 1's */
      HYPRE_SStructVectorGather(x);
      hypre_sprintf(filename, "sstruct.out.gatherpost.%05d", myid);
      file = fopen(filename, "w");
      for (part = 0; part < data.nparts; part++)
      {
         pdata = data.pdata[part];
         for (var = 0; var < pdata.nvars; var++)
         {
            for (box = 0; box < pdata.nboxes; box++)
            {
               GetVariableBox(pdata.ilowers[box], pdata.iuppers[box],
                              pdata.vartypes[var], ilower, iupper);
               HYPRE_SStructVectorGetBoxValues(x, part, ilower, iupper,
                                               var, values);
               hypre_fprintf(file, "\nPart %d, var %d, box %d:\n", part, var, box);
               for (i = 0; i < pdata.boxsizes[box]; i++)
               {
                  hypre_fprintf(file, "%e\n", values[i]);
               }
            }
         }
      }

      /* re-initializes x to 0 */
      hypre_SStructAxpy(-1.0, b, x);
   }
#endif

   hypre_TFree(values, HYPRE_MEMORY_HOST);

   /*-----------------------------------------------------------
    * Solve the system using SysPFMG, SSAMG, Split or BoomerAMG
    *-----------------------------------------------------------*/

   if (solver_id == 3)
   {
      time_index = hypre_InitializeTiming("SysPFMG Setup");
      hypre_BeginTiming(time_index);

      HYPRE_SStructSysPFMGCreate(comm, &solver);
      HYPRE_SStructSysPFMGSetMaxIter(solver, max_iterations);
      HYPRE_SStructSysPFMGSetTol(solver, tol);
      HYPRE_SStructSysPFMGSetRelChange(solver, rel_change);
      /* weighted Jacobi = 1; red-black GS = 2 */
      HYPRE_SStructSysPFMGSetRelaxType(solver, relax[0]);
      if (usr_jacobi_weight)
      {
         HYPRE_SStructSysPFMGSetJacobiWeight(solver, jacobi_weight);
      }
      HYPRE_SStructSysPFMGSetNumPreRelax(solver, n_pre);
      HYPRE_SStructSysPFMGSetNumPostRelax(solver, n_post);
      HYPRE_SStructSysPFMGSetSkipRelax(solver, skip);
      /*HYPRE_StructPFMGSetDxyz(solver, dxyz);*/
      HYPRE_SStructSysPFMGSetPrintLevel(solver, print_level);
      HYPRE_SStructSysPFMGSetLogging(solver, 1);
      HYPRE_SStructSysPFMGSetup(solver, A, b, x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      time_index = hypre_InitializeTiming("SysPFMG Solve");
      hypre_BeginTiming(time_index);

      HYPRE_SStructSysPFMGSolve(solver, A, b, x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      HYPRE_SStructSysPFMGGetNumIterations(solver, &num_iterations);
      HYPRE_SStructSysPFMGGetFinalRelativeResidualNorm(solver, &final_res_norm);

      HYPRE_SStructSysPFMGDestroy(solver);
   }

   else if (solver_id == 4)
   {
      time_index = hypre_InitializeTiming("SSAMG Setup");
      hypre_BeginTiming(time_index);

      HYPRE_SStructSSAMGCreate(comm, &solver);
      HYPRE_SStructSSAMGSetMaxIter(solver, max_iterations);
      HYPRE_SStructSSAMGSetMaxLevels(solver, max_levels);
      HYPRE_SStructSSAMGSetTol(solver, tol);
      HYPRE_SStructSSAMGSetRelChange(solver, rel_change);
      HYPRE_SStructSSAMGSetSkipRelax(solver, skip);
      /* weighted Jacobi = 1; red-black GS = 2 */
      HYPRE_SStructSSAMGSetRelaxType(solver, relax[0]);
      if (usr_jacobi_weight)
      {
         HYPRE_SStructSSAMGSetRelaxWeight(solver, jacobi_weight);
      }
      HYPRE_SStructSSAMGSetNumPreRelax(solver, n_pre);
      HYPRE_SStructSSAMGSetNumPostRelax(solver, n_post);
      HYPRE_SStructSSAMGSetNumCoarseRelax(solver, n_coarse);
      HYPRE_SStructSSAMGSetMaxCoarseSize(solver, max_coarse_size);
      HYPRE_SStructSSAMGSetCoarseSolverType(solver, csolver_type);
      HYPRE_SStructSSAMGSetNonGalerkinRAP(solver, rap);
      HYPRE_SStructSSAMGSetPrintLevel(solver, print_level);
      HYPRE_SStructSSAMGSetPrintFreq(solver, print_freq);
      HYPRE_SStructSSAMGSetLogging(solver, 1);
      HYPRE_SStructSSAMGSetup(solver, A, b, x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      time_index = hypre_InitializeTiming("SSAMG Solve");
      hypre_BeginTiming(time_index);

      HYPRE_SStructSSAMGSolve(solver, A, b, x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      HYPRE_SStructSSAMGGetNumIterations(solver, &num_iterations);
      HYPRE_SStructSSAMGGetFinalRelativeResidualNorm(solver, &final_res_norm);

      HYPRE_SStructSSAMGDestroy(solver);
   }

   else if (solver_id == 5)
   {
      time_index = hypre_InitializeTiming("BoomerAMG Setup");
      hypre_BeginTiming(time_index);

      HYPRE_BoomerAMGCreate(&par_solver);
      HYPRE_BoomerAMGSetStrongThreshold(par_solver, strong_threshold);
      HYPRE_BoomerAMGSetPMaxElmts(par_solver, P_max_elmts);
      HYPRE_BoomerAMGSetCoarsenType(par_solver, coarsen_type);
      HYPRE_BoomerAMGSetMaxIter(par_solver, max_iterations);
      HYPRE_BoomerAMGSetMaxLevels(par_solver, max_levels);
      HYPRE_BoomerAMGSetMaxCoarseSize(par_solver, max_coarse_size);
      HYPRE_BoomerAMGSetTol(par_solver, tol);
      HYPRE_BoomerAMGSetPrintLevel(par_solver, print_level);
      HYPRE_BoomerAMGSetLogging(par_solver, 1);
      HYPRE_BoomerAMGSetCycleNumSweeps(par_solver, n_pre, 1);
      HYPRE_BoomerAMGSetCycleNumSweeps(par_solver, n_post, 2);
      HYPRE_BoomerAMGSetCycleNumSweeps(par_solver, n_pre, 3);
      if (usr_jacobi_weight)
      {
         HYPRE_BoomerAMGSetRelaxWt(par_solver, jacobi_weight);
      }
      if (relax[0] > -1)
      {
         HYPRE_BoomerAMGSetRelaxType(par_solver, relax[0]);
      }
      for (i = 1; i < 4; i++)
      {
         if (relax[i] > -1)
         {
            HYPRE_BoomerAMGSetCycleRelaxType(par_solver, relax[i], i);
         }
      }
      HYPRE_BoomerAMGSetAggNumLevels(par_solver, agg_num_levels);
      if (old_default)
      {
         HYPRE_BoomerAMGSetOldDefault(par_solver);
      }
      HYPRE_BoomerAMGSetup(par_solver, par_A, par_b, par_x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      time_index = hypre_InitializeTiming("BoomerAMG Solve");
      hypre_BeginTiming(time_index);

      HYPRE_BoomerAMGSolve(par_solver, par_A, par_b, par_x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      HYPRE_BoomerAMGGetNumIterations(par_solver, &num_iterations);
      HYPRE_BoomerAMGGetFinalRelativeResidualNorm(par_solver, &final_res_norm);
      HYPRE_BoomerAMGDestroy(par_solver);
   }

   else if ((solver_id >= 0) && (solver_id < 10) && (solver_id != 3))
   {
      time_index = hypre_InitializeTiming("Split Setup");
      hypre_BeginTiming(time_index);

      HYPRE_SStructSplitCreate(comm, &solver);
      HYPRE_SStructSplitSetPrintLevel(solver, print_level);
      HYPRE_SStructSplitSetLogging(solver, 1);
      HYPRE_SStructSplitSetMaxIter(solver, max_iterations);
      HYPRE_SStructSplitSetTol(solver, tol);
      if (solver_id == 0)
      {
         HYPRE_SStructSplitSetStructSolver(solver, HYPRE_SMG);
      }
      else if (solver_id == 1)
      {
         HYPRE_SStructSplitSetStructSolver(solver, HYPRE_PFMG);
      }
      else if (solver_id == 8)
      {
         HYPRE_SStructSplitSetStructSolver(solver, HYPRE_Jacobi);
      }
      HYPRE_SStructSplitSetup(solver, A, b, x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      time_index = hypre_InitializeTiming("Split Solve");
      hypre_BeginTiming(time_index);

      HYPRE_SStructSplitSolve(solver, A, b, x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      HYPRE_SStructSplitGetNumIterations(solver, &num_iterations);
      HYPRE_SStructSplitGetFinalRelativeResidualNorm(solver, &final_res_norm);

      HYPRE_SStructSplitDestroy(solver);
   }

   /*-----------------------------------------------------------
    * Solve the system using PCG
    *-----------------------------------------------------------*/

   else if (!lobpcgFlag && (solver_id >= 10) && (solver_id < 20))
   {
      time_index = hypre_InitializeTiming("PCG Setup");
      hypre_BeginTiming(time_index);

      HYPRE_SStructPCGCreate(comm, &solver);
      HYPRE_PCGSetMaxIter( (HYPRE_Solver) solver, max_iterations );
      HYPRE_PCGSetTol( (HYPRE_Solver) solver, tol );
      HYPRE_PCGSetTwoNorm( (HYPRE_Solver) solver, 1 );
      HYPRE_PCGSetRelChange( (HYPRE_Solver) solver, rel_change );
      HYPRE_PCGSetPrintLevel( (HYPRE_Solver) solver, krylov_print_level );
      HYPRE_PCGSetRecomputeResidual( (HYPRE_Solver) solver, recompute_res);

      if ((solver_id == 10) || (solver_id == 11))
      {
         /* use Split solver as preconditioner */
         HYPRE_SStructSplitCreate(comm, &precond);
         HYPRE_SStructSplitSetMaxIter(precond, 1);
         HYPRE_SStructSplitSetPrintLevel(precond, print_level);
         HYPRE_SStructSplitSetLogging(precond, 0);
         HYPRE_SStructSplitSetTol(precond, 0.0);
         HYPRE_SStructSplitSetZeroGuess(precond);
         if (solver_id == 10)
         {
            HYPRE_SStructSplitSetStructSolver(precond, HYPRE_SMG);
         }
         else if (solver_id == 11)
         {
            HYPRE_SStructSplitSetStructSolver(precond, HYPRE_PFMG);
         }
         HYPRE_PCGSetPrecond( (HYPRE_Solver) solver,
                              (HYPRE_PtrToSolverFcn) HYPRE_SStructSplitSolve,
                              (HYPRE_PtrToSolverFcn) HYPRE_SStructSplitSetup,
                              (HYPRE_Solver) precond);
      }
      else if (solver_id == 13)
      {
         /* use SysPFMG solver as preconditioner */
         HYPRE_SStructSysPFMGCreate(comm, &precond);
         HYPRE_SStructSysPFMGSetMaxIter(precond, 1);
         HYPRE_SStructSysPFMGSetTol(precond, 0.0);
         HYPRE_SStructSysPFMGSetZeroGuess(precond);
         /* weighted Jacobi = 1; red-black GS = 2 */
         HYPRE_SStructSysPFMGSetRelaxType(precond, relax[0]);
         if (usr_jacobi_weight)
         {
            HYPRE_SStructSysPFMGSetJacobiWeight(precond, jacobi_weight);
         }
         HYPRE_SStructSysPFMGSetNumPreRelax(precond, n_pre);
         HYPRE_SStructSysPFMGSetNumPostRelax(precond, n_post);
         HYPRE_SStructSysPFMGSetSkipRelax(precond, skip);
         HYPRE_SStructSysPFMGSetPrintLevel(precond, print_level);
         HYPRE_SStructSysPFMGSetLogging(precond, 0);
         /*HYPRE_SStructSysPFMGSetDxyz(precond, dxyz);*/
         HYPRE_PCGSetPrecond( (HYPRE_Solver) solver,
                              (HYPRE_PtrToSolverFcn) HYPRE_SStructSysPFMGSolve,
                              (HYPRE_PtrToSolverFcn) HYPRE_SStructSysPFMGSetup,
                              (HYPRE_Solver) precond);

      }
      else if (solver_id == 14)
      {
         /* use SSAMG solver as preconditioner */
         HYPRE_SStructSSAMGCreate(comm, &precond);
         HYPRE_SStructSSAMGSetMaxIter(precond, 1);
         HYPRE_SStructSSAMGSetMaxLevels(precond, max_levels);
         HYPRE_SStructSSAMGSetTol(precond, 0.0);
         HYPRE_SStructSSAMGSetZeroGuess(precond);
         HYPRE_SStructSSAMGSetSkipRelax(precond, skip);
         HYPRE_SStructSSAMGSetRelaxType(precond, relax[0]);
         if (usr_jacobi_weight)
         {
            HYPRE_SStructSSAMGSetRelaxWeight(precond, jacobi_weight);
         }
         HYPRE_SStructSSAMGSetNumPreRelax(precond, n_pre);
         HYPRE_SStructSSAMGSetNumPostRelax(precond, n_post);
         HYPRE_SStructSSAMGSetNumCoarseRelax(precond, n_coarse);
         HYPRE_SStructSSAMGSetMaxCoarseSize(precond, max_coarse_size);
         HYPRE_SStructSSAMGSetCoarseSolverType(precond, csolver_type);
         HYPRE_SStructSSAMGSetNonGalerkinRAP(precond, rap);
         HYPRE_SStructSSAMGSetPrintLevel(precond, print_level);
         HYPRE_SStructSSAMGSetLogging(precond, 0);

         HYPRE_PCGSetPrecond( (HYPRE_Solver) solver,
                              (HYPRE_PtrToSolverFcn) HYPRE_SStructSSAMGSolve,
                              (HYPRE_PtrToSolverFcn) HYPRE_SStructSSAMGSetup,
                              (HYPRE_Solver) precond);
      }
      else if (solver_id == 18)
      {
         /* use diagonal scaling as preconditioner */
         precond = NULL;
         HYPRE_PCGSetPrecond( (HYPRE_Solver) solver,
                              (HYPRE_PtrToSolverFcn) HYPRE_SStructDiagScale,
                              (HYPRE_PtrToSolverFcn) HYPRE_SStructDiagScaleSetup,
                              (HYPRE_Solver) precond);
      }

      HYPRE_PCGSetup( (HYPRE_Solver) solver, (HYPRE_Matrix) A,
                      (HYPRE_Vector) b, (HYPRE_Vector) x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      time_index = hypre_InitializeTiming("PCG Solve");
      hypre_BeginTiming(time_index);

      HYPRE_PCGSolve( (HYPRE_Solver) solver, (HYPRE_Matrix) A,
                      (HYPRE_Vector) b, (HYPRE_Vector) x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      HYPRE_PCGGetNumIterations( (HYPRE_Solver) solver, &num_iterations );
      HYPRE_PCGGetFinalRelativeResidualNorm( (HYPRE_Solver) solver, &final_res_norm );
      HYPRE_SStructPCGDestroy(solver);

      if ((solver_id == 10) || (solver_id == 11))
      {
         HYPRE_SStructSplitDestroy(precond);
      }
      else if (solver_id == 13)
      {
         HYPRE_SStructSysPFMGDestroy(precond);
      }
      else if (solver_id == 14)
      {
         HYPRE_SStructSSAMGDestroy(precond);
      }
   }

   /* begin lobpcg */

   /*-----------------------------------------------------------
    * Solve the eigenvalue problem using LOBPCG
    *-----------------------------------------------------------*/

   if ( lobpcgFlag && ( solver_id < 10 || solver_id >= 20 ) && verbosity )
      hypre_printf("\nLOBPCG works with solvers 10, 11, 13 and 18 only\n");

   if ( lobpcgFlag && (solver_id >= 10) && (solver_id < 20) ) {

      interpreter = hypre_CTAlloc(mv_InterfaceInterpreter, 1, HYPRE_MEMORY_HOST);

      HYPRE_SStructSetupInterpreter( interpreter );
      HYPRE_SStructSetupMatvec(&matvec_fn);

      if (myid != 0)
      {
         verbosity = 0;
      }

      if (pcgIterations > 0)
      {
         time_index = hypre_InitializeTiming("PCG Setup");
         hypre_BeginTiming(time_index);

         HYPRE_SStructPCGCreate(comm, &solver);
         HYPRE_PCGSetMaxIter( (HYPRE_Solver) solver, pcgIterations );
         HYPRE_PCGSetTol( (HYPRE_Solver) solver, pcgTol );
         HYPRE_PCGSetTwoNorm( (HYPRE_Solver) solver, 1 );
         HYPRE_PCGSetRelChange( (HYPRE_Solver) solver, 0 );
         HYPRE_PCGSetPrintLevel( (HYPRE_Solver) solver, 0 );

         if ((solver_id == 10) || (solver_id == 11))
         {
            /* use Split solver as preconditioner */
            HYPRE_SStructSplitCreate(comm, &precond);
            HYPRE_SStructSplitSetMaxIter(precond, 1);
            HYPRE_SStructSplitSetTol(precond, 0.0);
            HYPRE_SStructSplitSetZeroGuess(precond);
            HYPRE_SStructSplitSetPrintLevel(precond, print_level);
            HYPRE_SStructSplitSetLogging(precond, 0);
            if (solver_id == 10)
            {
               HYPRE_SStructSplitSetStructSolver(precond, HYPRE_SMG);
            }
            else if (solver_id == 11)
            {
               HYPRE_SStructSplitSetStructSolver(precond, HYPRE_PFMG);
            }
            HYPRE_PCGSetPrecond( (HYPRE_Solver) solver,
                                 (HYPRE_PtrToSolverFcn) HYPRE_SStructSplitSolve,
                                 (HYPRE_PtrToSolverFcn) HYPRE_SStructSplitSetup,
                                 (HYPRE_Solver) precond);
         }

         else if (solver_id == 13)
         {
            /* use SysPFMG solver as preconditioner */
            HYPRE_SStructSysPFMGCreate(comm, &precond);
            HYPRE_SStructSysPFMGSetMaxIter(precond, 1);
            HYPRE_SStructSysPFMGSetTol(precond, 0.0);
            HYPRE_SStructSysPFMGSetZeroGuess(precond);
            /* weighted Jacobi = 1; red-black GS = 2 */
            HYPRE_SStructSysPFMGSetRelaxType(precond, relax[0]);
            HYPRE_SStructSysPFMGSetNumPreRelax(precond, n_pre);
            HYPRE_SStructSysPFMGSetNumPostRelax(precond, n_post);
            HYPRE_SStructSysPFMGSetSkipRelax(precond, skip);
            /*HYPRE_StructPFMGSetDxyz(precond, dxyz);*/
            HYPRE_SStructSysPFMGSetPrintLevel(precond, print_level);
            HYPRE_SStructSysPFMGSetLogging(precond, 0);
            HYPRE_PCGSetPrecond( (HYPRE_Solver) solver,
                                 (HYPRE_PtrToSolverFcn) HYPRE_SStructSysPFMGSolve,
                                 (HYPRE_PtrToSolverFcn) HYPRE_SStructSysPFMGSetup,
                                 (HYPRE_Solver) precond);

         }
         else if (solver_id == 18)
         {
            /* use diagonal scaling as preconditioner */
            precond = NULL;
            HYPRE_PCGSetPrecond( (HYPRE_Solver) solver,
                                 (HYPRE_PtrToSolverFcn) HYPRE_SStructDiagScale,
                                 (HYPRE_PtrToSolverFcn) HYPRE_SStructDiagScaleSetup,
                                 (HYPRE_Solver) precond);
         }
         else if (solver_id != NO_SOLVER )
         {
            if (verbosity)
            {
               hypre_printf("Solver ID not recognized - running inner PCG iterations without preconditioner\n\n");
            }
         }

         hypre_EndTiming(time_index);
         hypre_PrintTiming("Setup phase times", comm);
         hypre_FinalizeTiming(time_index);
         hypre_ClearTiming();

         HYPRE_LOBPCGCreate(interpreter, &matvec_fn, (HYPRE_Solver*)&lobpcg_solver);
         HYPRE_LOBPCGSetMaxIter((HYPRE_Solver)lobpcg_solver, maxIterations);
         HYPRE_LOBPCGSetPrecondUsageMode((HYPRE_Solver)lobpcg_solver, pcgMode);
         HYPRE_LOBPCGSetTol((HYPRE_Solver)lobpcg_solver, tol);
         HYPRE_LOBPCGSetPrintLevel((HYPRE_Solver)lobpcg_solver, verbosity);

         HYPRE_LOBPCGSetPrecond((HYPRE_Solver)lobpcg_solver,
                                (HYPRE_PtrToSolverFcn) HYPRE_PCGSolve,
                                (HYPRE_PtrToSolverFcn) HYPRE_PCGSetup,
                                (HYPRE_Solver)solver);

         HYPRE_LOBPCGSetup((HYPRE_Solver)lobpcg_solver, (HYPRE_Matrix)A,
                           (HYPRE_Vector)b, (HYPRE_Vector)x);

         eigenvectors = mv_MultiVectorCreateFromSampleVector( interpreter,
                                                              blockSize,
                                                              x );
         eigenvalues = hypre_CTAlloc(HYPRE_Real,  blockSize, HYPRE_MEMORY_HOST);

         if (lobpcgSeed)
         {
            mv_MultiVectorSetRandom(eigenvectors, lobpcgSeed);
         }
         else
         {
            mv_MultiVectorSetRandom(eigenvectors, (HYPRE_Int)time(0));
         }

         time_index = hypre_InitializeTiming("PCG Solve");
         hypre_BeginTiming(time_index);

         HYPRE_LOBPCGSolve((HYPRE_Solver)lobpcg_solver, constrains,
                           eigenvectors, eigenvalues );

         hypre_EndTiming(time_index);
         hypre_PrintTiming("Solve phase times", comm);
         hypre_FinalizeTiming(time_index);
         hypre_ClearTiming();

         if (checkOrtho)
         {
            gramXX = utilities_FortranMatrixCreate();
            identity = utilities_FortranMatrixCreate();

            utilities_FortranMatrixAllocateData( blockSize, blockSize, gramXX );
            utilities_FortranMatrixAllocateData( blockSize, blockSize, identity );

            lobpcg_MultiVectorByMultiVector( eigenvectors, eigenvectors, gramXX );
            utilities_FortranMatrixSetToIdentity( identity );
            utilities_FortranMatrixAdd( -1, identity, gramXX, gramXX );
            nonOrthF = utilities_FortranMatrixFNorm( gramXX );
            if ( myid == 0 )
               hypre_printf("Non-orthonormality of eigenvectors: %12.5e\n", nonOrthF);

            utilities_FortranMatrixDestroy( gramXX );
            utilities_FortranMatrixDestroy( identity );
         }

         if (printLevel)
         {
            if (myid == 0)
            {
               if ((filePtr = fopen("values.txt", "w")))
               {
                  hypre_fprintf(filePtr, "%d\n", blockSize);
                  for (i = 0; i < blockSize; i++)
                  {
                     hypre_fprintf(filePtr, "%22.14e\n", eigenvalues[i]);
                  }
                  fclose(filePtr);
               }

               if ((filePtr = fopen("residuals.txt", "w")))
               {
                  residualNorms = HYPRE_LOBPCGResidualNorms( (HYPRE_Solver)lobpcg_solver );
                  residuals = utilities_FortranMatrixValues( residualNorms );
                  hypre_fprintf(filePtr, "%d\n", blockSize);
                  for (i = 0; i < blockSize; i++)
                  {
                     hypre_fprintf(filePtr, "%22.14e\n", residuals[i]);
                  }
                  fclose(filePtr);
               }

               if (printLevel > 1)
               {
                  printBuffer = utilities_FortranMatrixCreate();

                  iterations = HYPRE_LOBPCGIterations( (HYPRE_Solver)lobpcg_solver );

                  eigenvaluesHistory = HYPRE_LOBPCGEigenvaluesHistory( (HYPRE_Solver)lobpcg_solver );
                  utilities_FortranMatrixSelectBlock( eigenvaluesHistory,
                                                      1, blockSize, 1, iterations + 1,
                                                      printBuffer );
                  utilities_FortranMatrixPrint(printBuffer, "val_hist.txt");

                  residualNormsHistory = HYPRE_LOBPCGResidualNormsHistory( (HYPRE_Solver)lobpcg_solver );
                  utilities_FortranMatrixSelectBlock(residualNormsHistory,
                                                     1, blockSize, 1, iterations + 1,
                                                     printBuffer );
                  utilities_FortranMatrixPrint(printBuffer, "res_hist.txt");

                  utilities_FortranMatrixDestroy(printBuffer);
               }
            }
         }

         HYPRE_SStructPCGDestroy(solver);

         if ((solver_id == 10) || (solver_id == 11))
         {
            HYPRE_SStructSplitDestroy(precond);
         }
         else if (solver_id == 13)
         {
            HYPRE_SStructSysPFMGDestroy(precond);
         }

         HYPRE_LOBPCGDestroy((HYPRE_Solver)lobpcg_solver);
         mv_MultiVectorDestroy( eigenvectors );
         hypre_TFree(eigenvalues, HYPRE_MEMORY_HOST);
      }
      else
      {
         time_index = hypre_InitializeTiming("LOBPCG Setup");
         hypre_BeginTiming(time_index);

         HYPRE_LOBPCGCreate(interpreter, &matvec_fn, (HYPRE_Solver*)&solver);
         HYPRE_LOBPCGSetMaxIter( (HYPRE_Solver) solver, maxIterations );
         HYPRE_LOBPCGSetTol( (HYPRE_Solver) solver, tol );
         HYPRE_LOBPCGSetPrintLevel( (HYPRE_Solver) solver, verbosity );

         if ((solver_id == 10) || (solver_id == 11))
         {
            /* use Split solver as preconditioner */
            HYPRE_SStructSplitCreate(comm, &precond);
            HYPRE_SStructSplitSetMaxIter(precond, 1);
            HYPRE_SStructSplitSetTol(precond, 0.0);
            HYPRE_SStructSplitSetZeroGuess(precond);
            HYPRE_SStructSplitSetPrintLevel(precond, print_level);
            HYPRE_SStructSplitSetLogging(precond, 0);
            if (solver_id == 10)
            {
               HYPRE_SStructSplitSetStructSolver(precond, HYPRE_SMG);
            }
            else if (solver_id == 11)
            {
               HYPRE_SStructSplitSetStructSolver(precond, HYPRE_PFMG);
            }
            HYPRE_LOBPCGSetPrecond( (HYPRE_Solver) solver,
                                    (HYPRE_PtrToSolverFcn) HYPRE_SStructSplitSolve,
                                    (HYPRE_PtrToSolverFcn) HYPRE_SStructSplitSetup,
                                    (HYPRE_Solver) precond);
         }

         else if (solver_id == 13)
         {
            /* use SysPFMG solver as preconditioner */
            HYPRE_SStructSysPFMGCreate(comm, &precond);
            HYPRE_SStructSysPFMGSetMaxIter(precond, 1);
            HYPRE_SStructSysPFMGSetTol(precond, 0.0);
            HYPRE_SStructSysPFMGSetZeroGuess(precond);
            /* weighted Jacobi = 1; red-black GS = 2 */
            HYPRE_SStructSysPFMGSetRelaxType(precond, relax[0]);
            HYPRE_SStructSysPFMGSetNumPreRelax(precond, n_pre);
            HYPRE_SStructSysPFMGSetNumPostRelax(precond, n_post);
            HYPRE_SStructSysPFMGSetSkipRelax(precond, skip);
            /*HYPRE_StructPFMGSetDxyz(precond, dxyz);*/
            HYPRE_SStructSysPFMGSetPrintLevel(precond, print_level);
            HYPRE_SStructSysPFMGSetLogging(precond, 0);
            HYPRE_LOBPCGSetPrecond( (HYPRE_Solver) solver,
                                    (HYPRE_PtrToSolverFcn) HYPRE_SStructSysPFMGSolve,
                                    (HYPRE_PtrToSolverFcn) HYPRE_SStructSysPFMGSetup,
                                    (HYPRE_Solver) precond);

         }
         else if (solver_id == 18)
         {
            /* use diagonal scaling as preconditioner */
            precond = NULL;
            HYPRE_LOBPCGSetPrecond( (HYPRE_Solver) solver,
                                    (HYPRE_PtrToSolverFcn) HYPRE_SStructDiagScale,
                                    (HYPRE_PtrToSolverFcn) HYPRE_SStructDiagScaleSetup,
                                    (HYPRE_Solver) precond);
         }
         else if (solver_id != NO_SOLVER )
         {
            if ( verbosity )
               hypre_printf("Solver ID not recognized - running LOBPCG without preconditioner\n\n");
         }

         HYPRE_LOBPCGSetup( (HYPRE_Solver) solver, (HYPRE_Matrix) A,
                            (HYPRE_Vector) b, (HYPRE_Vector) x);

         hypre_EndTiming(time_index);
         hypre_PrintTiming("Setup phase times", comm);
         hypre_FinalizeTiming(time_index);
         hypre_ClearTiming();

         eigenvectors = mv_MultiVectorCreateFromSampleVector( interpreter,
                                                              blockSize,
                                                              x );
         eigenvalues = hypre_CTAlloc(HYPRE_Real,  blockSize, HYPRE_MEMORY_HOST);

         if (lobpcgSeed)
         {
            mv_MultiVectorSetRandom(eigenvectors, lobpcgSeed);
         }
         else
         {
            mv_MultiVectorSetRandom(eigenvectors, (HYPRE_Int)time(0));
         }

         time_index = hypre_InitializeTiming("LOBPCG Solve");
         hypre_BeginTiming(time_index);

         HYPRE_LOBPCGSolve( (HYPRE_Solver) solver, constrains,
                            eigenvectors, eigenvalues );

         hypre_EndTiming(time_index);
         hypre_PrintTiming("Solve phase times", comm);
         hypre_FinalizeTiming(time_index);
         hypre_ClearTiming();

         if (checkOrtho)
         {
            gramXX = utilities_FortranMatrixCreate();
            identity = utilities_FortranMatrixCreate();

            utilities_FortranMatrixAllocateData(blockSize, blockSize, gramXX);
            utilities_FortranMatrixAllocateData(blockSize, blockSize, identity);

            lobpcg_MultiVectorByMultiVector(eigenvectors, eigenvectors, gramXX);
            utilities_FortranMatrixSetToIdentity(identity);
            utilities_FortranMatrixAdd(-1, identity, gramXX, gramXX);
            nonOrthF = utilities_FortranMatrixFNorm(gramXX);
            if (myid == 0)
            {
               hypre_printf("Non-orthonormality of eigenvectors: %12.5e\n", nonOrthF);
            }

            utilities_FortranMatrixDestroy( gramXX );
            utilities_FortranMatrixDestroy( identity );
         }

         if (printLevel)
         {
            if (myid == 0)
            {
               if ((filePtr = fopen("values.txt", "w")))
               {
                  hypre_fprintf(filePtr, "%d\n", blockSize);
                  for (i = 0; i < blockSize; i++)
                  {
                     hypre_fprintf(filePtr, "%22.14e\n", eigenvalues[i]);
                  }
                  fclose(filePtr);
               }

               if ((filePtr = fopen("residuals.txt", "w")))
               {
                  residualNorms = HYPRE_LOBPCGResidualNorms( (HYPRE_Solver)solver );
                  residuals = utilities_FortranMatrixValues( residualNorms );
                  hypre_fprintf(filePtr, "%d\n", blockSize);
                  for (i = 0; i < blockSize; i++)
                  {
                     hypre_fprintf(filePtr, "%22.14e\n", residuals[i]);
                  }
                  fclose(filePtr);
               }

               if (printLevel > 1)
               {

                  printBuffer = utilities_FortranMatrixCreate();

                  iterations = HYPRE_LOBPCGIterations( (HYPRE_Solver)solver );

                  eigenvaluesHistory = HYPRE_LOBPCGEigenvaluesHistory( (HYPRE_Solver)solver );
                  utilities_FortranMatrixSelectBlock( eigenvaluesHistory,
                                                      1, blockSize, 1, iterations + 1,
                                                      printBuffer );
                  utilities_FortranMatrixPrint(printBuffer, "val_hist.txt");

                  residualNormsHistory = HYPRE_LOBPCGResidualNormsHistory( (HYPRE_Solver)solver );
                  utilities_FortranMatrixSelectBlock(residualNormsHistory,
                                                     1, blockSize, 1, iterations + 1,
                                                     printBuffer);
                  utilities_FortranMatrixPrint(printBuffer, "res_hist.txt");

                  utilities_FortranMatrixDestroy(printBuffer);
               }
            }
         }

         HYPRE_LOBPCGDestroy((HYPRE_Solver)solver);

         if ((solver_id == 10) || (solver_id == 11))
         {
            HYPRE_SStructSplitDestroy(precond);
         }
         else if (solver_id == 13)
         {
            HYPRE_SStructSysPFMGDestroy(precond);
         }

         mv_MultiVectorDestroy( eigenvectors );
         hypre_TFree(eigenvalues, HYPRE_MEMORY_HOST);
      }

      hypre_TFree( interpreter , HYPRE_MEMORY_HOST);
   }
   /* end lobpcg */

   /*-----------------------------------------------------------
    * Solve the system using ParCSR version of PCG
    *-----------------------------------------------------------*/

   else if ((solver_id >= 20) && (solver_id < 30))
   {
      time_index = hypre_InitializeTiming("PCG Setup");
      hypre_BeginTiming(time_index);

      HYPRE_ParCSRPCGCreate(comm, &par_solver);
      HYPRE_PCGSetMaxIter( par_solver, max_iterations );
      HYPRE_PCGSetTol( par_solver, tol );
      HYPRE_PCGSetTwoNorm( par_solver, 1 );
      HYPRE_PCGSetRelChange( par_solver, rel_change );
      HYPRE_PCGSetPrintLevel( par_solver, krylov_print_level );
      HYPRE_PCGSetRecomputeResidual( (HYPRE_Solver) par_solver, recompute_res);

      if (solver_id == 20)
      {
         /* use BoomerAMG as preconditioner */
         HYPRE_BoomerAMGCreate(&par_precond);
         HYPRE_BoomerAMGSetStrongThreshold(par_precond, strong_threshold);
         HYPRE_BoomerAMGSetPMaxElmts(par_precond, P_max_elmts);
         HYPRE_BoomerAMGSetCoarsenType(par_precond, coarsen_type);
         HYPRE_BoomerAMGSetMaxIter(par_precond, 1);
         HYPRE_BoomerAMGSetMaxLevels(par_precond, max_levels);
         HYPRE_BoomerAMGSetMaxCoarseSize(par_precond, max_coarse_size);
         HYPRE_BoomerAMGSetTol(par_precond, 0.0);
         HYPRE_BoomerAMGSetPrintLevel(par_precond, print_level);
         HYPRE_BoomerAMGSetLogging(par_precond, 0);
         HYPRE_BoomerAMGSetPrintFileName(par_precond, "sstruct.out.log");
         HYPRE_BoomerAMGSetCycleNumSweeps(par_precond, n_pre, 1);
         HYPRE_BoomerAMGSetCycleNumSweeps(par_precond, n_post, 2);
         HYPRE_BoomerAMGSetCycleNumSweeps(par_precond, n_coarse, 3);
         if (usr_jacobi_weight)
         {
            HYPRE_BoomerAMGSetRelaxWt(par_precond, jacobi_weight);
         }
         if (relax[0] > -1)
         {
            HYPRE_BoomerAMGSetRelaxType(par_precond, relax[0]);
         }
         for (i = 1; i < 4; i++)
         {
            if (relax[i] > -1)
            {
               HYPRE_BoomerAMGSetCycleRelaxType(par_precond, relax[i], i);
            }
         }
         HYPRE_BoomerAMGSetAggNumLevels(par_precond, agg_num_levels);
         if (old_default)
         {
            HYPRE_BoomerAMGSetOldDefault(par_precond);
         }

         HYPRE_PCGSetPrecond( par_solver,
                              (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve,
                              (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup,
                              par_precond );
      }
      else if (solver_id == 21)
      {
         /* use Euclid as preconditioner */
         HYPRE_EuclidCreate(comm, &par_precond);
         HYPRE_EuclidSetParams(par_precond, argc, argv);
         HYPRE_PCGSetPrecond(par_solver,
                             (HYPRE_PtrToSolverFcn) HYPRE_EuclidSolve,
                             (HYPRE_PtrToSolverFcn) HYPRE_EuclidSetup,
                             par_precond);
      }
      else if (solver_id == 22)
      {
         /* use ParaSails as preconditioner */
         HYPRE_ParCSRParaSailsCreate(comm, &par_precond );
	 HYPRE_ParCSRParaSailsSetParams(par_precond, 0.1, 1);
         HYPRE_PCGSetPrecond( par_solver,
                              (HYPRE_PtrToSolverFcn) HYPRE_ParCSRParaSailsSolve,
                              (HYPRE_PtrToSolverFcn) HYPRE_ParCSRParaSailsSetup,
                              par_precond );
      }

      else if (solver_id == 28)
      {
         /* use diagonal scaling as preconditioner */
         par_precond = NULL;
         HYPRE_PCGSetPrecond(  par_solver,
                               (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScale,
                               (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScaleSetup,
                               par_precond );
      }

      HYPRE_PCGSetup( par_solver, (HYPRE_Matrix) par_A,
                      (HYPRE_Vector) par_b, (HYPRE_Vector) par_x );

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      time_index = hypre_InitializeTiming("PCG Solve");
      hypre_BeginTiming(time_index);

      HYPRE_PCGSolve( par_solver, (HYPRE_Matrix) par_A,
                      (HYPRE_Vector) par_b, (HYPRE_Vector) par_x );

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      HYPRE_PCGGetNumIterations( par_solver, &num_iterations );
      HYPRE_PCGGetFinalRelativeResidualNorm( par_solver, &final_res_norm );
      HYPRE_ParCSRPCGDestroy(par_solver);

      if (solver_id == 20)
      {
         HYPRE_BoomerAMGDestroy(par_precond);
      }
      else if (solver_id == 21)
      {
         HYPRE_EuclidDestroy(par_precond);
      }
      else if (solver_id == 22)
      {
         HYPRE_ParCSRParaSailsDestroy(par_precond);
      }
   }

   /*-----------------------------------------------------------
    * Solve the system using GMRES
    *-----------------------------------------------------------*/

   else if ((solver_id >= 30) && (solver_id < 40))
   {
      time_index = hypre_InitializeTiming("GMRES Setup");
      hypre_BeginTiming(time_index);

      HYPRE_SStructGMRESCreate(comm, &solver);
      HYPRE_GMRESSetKDim( (HYPRE_Solver) solver, k_dim );
      HYPRE_GMRESSetMaxIter( (HYPRE_Solver) solver, max_iterations );
      HYPRE_GMRESSetTol( (HYPRE_Solver) solver, tol );
      HYPRE_GMRESSetPrintLevel( (HYPRE_Solver) solver, krylov_print_level );
      HYPRE_GMRESSetLogging( (HYPRE_Solver) solver, 1 );

      if ((solver_id == 30) || (solver_id == 31))
      {
         /* use Split solver as preconditioner */
         HYPRE_SStructSplitCreate(comm, &precond);
         HYPRE_SStructSplitSetMaxIter(precond, 1);
         HYPRE_SStructSplitSetTol(precond, 0.0);
         HYPRE_SStructSplitSetZeroGuess(precond);
         HYPRE_SStructSplitSetPrintLevel(precond, print_level);
         HYPRE_SStructSplitSetLogging(precond, 0);
         if (solver_id == 30)
         {
            HYPRE_SStructSplitSetStructSolver(precond, HYPRE_SMG);
         }
         else if (solver_id == 31)
         {
            HYPRE_SStructSplitSetStructSolver(precond, HYPRE_PFMG);
         }
         HYPRE_GMRESSetPrecond( (HYPRE_Solver) solver,
                                (HYPRE_PtrToSolverFcn) HYPRE_SStructSplitSolve,
                                (HYPRE_PtrToSolverFcn) HYPRE_SStructSplitSetup,
                                (HYPRE_Solver) precond );
      }
      else if (solver_id == 33)
      {
         /* use SysPFMG solver as preconditioner */
         HYPRE_SStructSysPFMGCreate(comm, &precond);
         HYPRE_SStructSysPFMGSetMaxIter(precond, 1);
         HYPRE_SStructSysPFMGSetTol(precond, 0.0);
         HYPRE_SStructSysPFMGSetZeroGuess(precond);
         /* weighted Jacobi = 1; red-black GS = 2 */
         HYPRE_SStructSysPFMGSetRelaxType(precond, relax[0]);
         HYPRE_SStructSysPFMGSetNumPreRelax(precond, n_pre);
         HYPRE_SStructSysPFMGSetNumPostRelax(precond, n_post);
         HYPRE_SStructSysPFMGSetSkipRelax(precond, skip);
         HYPRE_SStructSysPFMGSetPrintLevel(precond, print_level);
         HYPRE_SStructSysPFMGSetLogging(precond, 0);
         /*HYPRE_SStructSysPFMGSetDxyz(precond, dxyz);*/
         HYPRE_GMRESSetPrecond( (HYPRE_Solver) solver,
                                (HYPRE_PtrToSolverFcn) HYPRE_SStructSysPFMGSolve,
                                (HYPRE_PtrToSolverFcn) HYPRE_SStructSysPFMGSetup,
                                (HYPRE_Solver) precond);
      }
      else if (solver_id == 34)
      {
         /* use SSAMG solver as preconditioner */
         HYPRE_SStructSSAMGCreate(comm, &precond);
         HYPRE_SStructSSAMGSetMaxIter(precond, 1);
         HYPRE_SStructSSAMGSetMaxLevels(precond, max_levels);
         HYPRE_SStructSSAMGSetTol(precond, 0.0);
         HYPRE_SStructSSAMGSetZeroGuess(precond);
         HYPRE_SStructSSAMGSetSkipRelax(precond, skip);
         HYPRE_SStructSSAMGSetRelaxType(precond, relax[0]);
         HYPRE_SStructSSAMGSetNonGalerkinRAP(precond, rap);
         if (usr_jacobi_weight)
         {
            HYPRE_SStructSSAMGSetRelaxWeight(precond, jacobi_weight);
         }
         HYPRE_SStructSSAMGSetNumPreRelax(precond, n_pre);
         HYPRE_SStructSSAMGSetNumPostRelax(precond, n_post);
         HYPRE_SStructSSAMGSetNumCoarseRelax(precond, n_coarse);
         HYPRE_SStructSSAMGSetMaxCoarseSize(precond, max_coarse_size);
         HYPRE_SStructSSAMGSetCoarseSolverType(precond, csolver_type);
         HYPRE_SStructSSAMGSetPrintLevel(precond, print_level);
         HYPRE_SStructSSAMGSetLogging(precond, 0);

         HYPRE_GMRESSetPrecond( (HYPRE_Solver) solver,
                                (HYPRE_PtrToSolverFcn) HYPRE_SStructSSAMGSolve,
                                (HYPRE_PtrToSolverFcn) HYPRE_SStructSSAMGSetup,
                                (HYPRE_Solver) precond);
      }
      else if (solver_id == 38)
      {
         /* use diagonal scaling as preconditioner */
         precond = NULL;
         HYPRE_GMRESSetPrecond( (HYPRE_Solver) solver,
                                (HYPRE_PtrToSolverFcn) HYPRE_SStructDiagScale,
                                (HYPRE_PtrToSolverFcn) HYPRE_SStructDiagScaleSetup,
                                (HYPRE_Solver) precond );
      }

      HYPRE_GMRESSetup( (HYPRE_Solver) solver, (HYPRE_Matrix) A,
                        (HYPRE_Vector) b, (HYPRE_Vector) x );

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      time_index = hypre_InitializeTiming("GMRES Solve");
      hypre_BeginTiming(time_index);

      HYPRE_GMRESSolve( (HYPRE_Solver) solver, (HYPRE_Matrix) A,
                        (HYPRE_Vector) b, (HYPRE_Vector) x );

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      HYPRE_GMRESGetNumIterations( (HYPRE_Solver) solver, &num_iterations );
      HYPRE_GMRESGetFinalRelativeResidualNorm( (HYPRE_Solver) solver, &final_res_norm );
      HYPRE_SStructGMRESDestroy(solver);

      if ((solver_id == 30) || (solver_id == 31))
      {
         HYPRE_SStructSplitDestroy(precond);
      }
      else if (solver_id == 33)
      {
         HYPRE_SStructSysPFMGDestroy(precond);
      }
      else if (solver_id == 34)
      {
         HYPRE_SStructSSAMGDestroy(precond);
      }
   }

   /*-----------------------------------------------------------
    * Solve the system using ParCSR version of GMRES
    *-----------------------------------------------------------*/

   else if ((solver_id >= 40) && (solver_id < 50))
   {
      time_index = hypre_InitializeTiming("GMRES Setup");
      hypre_BeginTiming(time_index);

      HYPRE_ParCSRGMRESCreate(comm, &par_solver);
      HYPRE_GMRESSetKDim(par_solver, k_dim);
      HYPRE_GMRESSetMaxIter(par_solver, max_iterations);
      HYPRE_GMRESSetTol(par_solver, tol);
      HYPRE_GMRESSetPrintLevel(par_solver, krylov_print_level);
      HYPRE_GMRESSetLogging(par_solver, 1);

      if (solver_id == 40)
      {
         /* use BoomerAMG as preconditioner */
         HYPRE_BoomerAMGCreate(&par_precond);
         HYPRE_BoomerAMGSetStrongThreshold(par_precond, strong_threshold);
         HYPRE_BoomerAMGSetPMaxElmts(par_precond, P_max_elmts);
         HYPRE_BoomerAMGSetCoarsenType(par_precond, coarsen_type);
         HYPRE_BoomerAMGSetMaxIter(par_precond, 1);
         HYPRE_BoomerAMGSetMaxLevels(par_precond, max_levels);
         HYPRE_BoomerAMGSetMaxCoarseSize(par_precond, max_coarse_size);
         HYPRE_BoomerAMGSetTol(par_precond, 0.0);
         HYPRE_BoomerAMGSetLogging(par_precond, 0);
         HYPRE_BoomerAMGSetPrintLevel(par_precond, print_level);
         HYPRE_BoomerAMGSetPrintFileName(par_precond, "sstruct.out.log");
         HYPRE_BoomerAMGSetCycleNumSweeps(par_precond, n_pre, 1);
         HYPRE_BoomerAMGSetCycleNumSweeps(par_precond, n_post, 2);
         HYPRE_BoomerAMGSetCycleNumSweeps(par_precond, n_pre, 3);
         if (usr_jacobi_weight)
         {
            HYPRE_BoomerAMGSetRelaxWt(par_precond, jacobi_weight);
         }
         if (relax[0] > -1)
         {
            HYPRE_BoomerAMGSetRelaxType(par_precond, relax[0]);
         }
         for (i = 1; i < 4; i++)
         {
            if (relax[i] > -1)
            {
               HYPRE_BoomerAMGSetCycleRelaxType(par_precond, relax[i], i);
            }
         }
         HYPRE_BoomerAMGSetAggNumLevels(par_precond, agg_num_levels);
         if (old_default)
         {
            HYPRE_BoomerAMGSetOldDefault(par_precond);
         }
         HYPRE_GMRESSetPrecond( par_solver,
                                (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve,
                                (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup,
                                par_precond);
      }
      else if (solver_id == 41)
      {
         /* use Euclid as preconditioner */
         HYPRE_EuclidCreate(comm, &par_precond);
         HYPRE_EuclidSetParams(par_precond, argc, argv);
         HYPRE_GMRESSetPrecond(par_solver,
                               (HYPRE_PtrToSolverFcn) HYPRE_EuclidSolve,
                               (HYPRE_PtrToSolverFcn) HYPRE_EuclidSetup,
                               par_precond);
      }
      else if (solver_id == 42)
      {
         /* use ParaSails as preconditioner */
         HYPRE_ParCSRParaSailsCreate(comm, &par_precond );
	 HYPRE_ParCSRParaSailsSetParams(par_precond, 0.1, 1);
	 HYPRE_ParCSRParaSailsSetSym(par_precond, 0);
         HYPRE_GMRESSetPrecond( par_solver,
                                (HYPRE_PtrToSolverFcn) HYPRE_ParCSRParaSailsSolve,
                                (HYPRE_PtrToSolverFcn) HYPRE_ParCSRParaSailsSetup,
                                par_precond);
      }

      HYPRE_GMRESSetup( par_solver, (HYPRE_Matrix) par_A,
                        (HYPRE_Vector) par_b, (HYPRE_Vector) par_x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      time_index = hypre_InitializeTiming("GMRES Solve");
      hypre_BeginTiming(time_index);

      HYPRE_GMRESSolve( par_solver, (HYPRE_Matrix) par_A,
                        (HYPRE_Vector) par_b, (HYPRE_Vector) par_x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      HYPRE_GMRESGetNumIterations( par_solver, &num_iterations);
      HYPRE_GMRESGetFinalRelativeResidualNorm( par_solver, &final_res_norm);
      HYPRE_ParCSRGMRESDestroy(par_solver);

      if (solver_id == 40)
      {
         HYPRE_BoomerAMGDestroy(par_precond);
      }
      else if (solver_id == 41)
      {
         HYPRE_EuclidDestroy(par_precond);
      }
      else if (solver_id == 42)
      {
         HYPRE_ParCSRParaSailsDestroy(par_precond);
      }
   }

   /*-----------------------------------------------------------
    * Solve the system using BiCGSTAB
    *-----------------------------------------------------------*/

   else if ((solver_id >= 50) && (solver_id < 60))
   {
      time_index = hypre_InitializeTiming("BiCGSTAB Setup");
      hypre_BeginTiming(time_index);

      HYPRE_SStructBiCGSTABCreate(comm, &solver);
      HYPRE_BiCGSTABSetMaxIter( (HYPRE_Solver) solver, max_iterations );
      HYPRE_BiCGSTABSetTol( (HYPRE_Solver) solver, tol );
      HYPRE_BiCGSTABSetPrintLevel( (HYPRE_Solver) solver, krylov_print_level );
      HYPRE_BiCGSTABSetLogging( (HYPRE_Solver) solver, 1 );

      if ((solver_id == 50) || (solver_id == 51))
      {
         /* use Split solver as preconditioner */
         HYPRE_SStructSplitCreate(comm, &precond);
         HYPRE_SStructSplitSetMaxIter(precond, 1);
         HYPRE_SStructSplitSetTol(precond, 0.0);
         HYPRE_SStructSplitSetZeroGuess(precond);
         HYPRE_SStructSplitSetPrintLevel(precond, print_level);
         HYPRE_SStructSplitSetLogging(precond, 0);
         if (solver_id == 50)
         {
            HYPRE_SStructSplitSetStructSolver(precond, HYPRE_SMG);
         }
         else if (solver_id == 51)
         {
            HYPRE_SStructSplitSetStructSolver(precond, HYPRE_PFMG);
         }
         HYPRE_BiCGSTABSetPrecond( (HYPRE_Solver) solver,
                                   (HYPRE_PtrToSolverFcn) HYPRE_SStructSplitSolve,
                                   (HYPRE_PtrToSolverFcn) HYPRE_SStructSplitSetup,
                                   (HYPRE_Solver) precond );
      }
      else if (solver_id == 53)
      {
         /* use SysPFMG solver as preconditioner */
         HYPRE_SStructSysPFMGCreate(comm, &precond);
         HYPRE_SStructSysPFMGSetMaxIter(precond, 1);
         HYPRE_SStructSysPFMGSetTol(precond, 0.0);
         HYPRE_SStructSysPFMGSetZeroGuess(precond);
         /* weighted Jacobi = 1; red-black GS = 2 */
         HYPRE_SStructSysPFMGSetRelaxType(precond, relax[0]);
         HYPRE_SStructSysPFMGSetNumPreRelax(precond, n_pre);
         HYPRE_SStructSysPFMGSetNumPostRelax(precond, n_post);
         HYPRE_SStructSysPFMGSetSkipRelax(precond, skip);
         /*HYPRE_SStructSysPFMGSetDxyz(precond, dxyz);*/
         HYPRE_SStructSysPFMGSetPrintLevel(precond, print_level);
         HYPRE_SStructSysPFMGSetLogging(precond, 0);

         HYPRE_BiCGSTABSetPrecond( (HYPRE_Solver) solver,
                                   (HYPRE_PtrToSolverFcn) HYPRE_SStructSysPFMGSolve,
                                   (HYPRE_PtrToSolverFcn) HYPRE_SStructSysPFMGSetup,
                                   (HYPRE_Solver) precond);
      }
      else if (solver_id == 54)
      {
         /* use SSAMG solver as preconditioner */
         HYPRE_SStructSSAMGCreate(comm, &precond);
         HYPRE_SStructSSAMGSetMaxIter(precond, 1);
         HYPRE_SStructSSAMGSetMaxLevels(precond, max_levels);
         HYPRE_SStructSSAMGSetTol(precond, 0.0);
         HYPRE_SStructSSAMGSetZeroGuess(precond);
         HYPRE_SStructSSAMGSetSkipRelax(precond, skip);
         HYPRE_SStructSSAMGSetRelaxType(precond, relax[0]);
         HYPRE_SStructSSAMGSetNonGalerkinRAP(precond, rap);
         if (usr_jacobi_weight)
         {
            HYPRE_SStructSSAMGSetRelaxWeight(precond, jacobi_weight);
         }
         HYPRE_SStructSSAMGSetNumPreRelax(precond, n_pre);
         HYPRE_SStructSSAMGSetNumPostRelax(precond, n_post);
         HYPRE_SStructSSAMGSetNumCoarseRelax(precond, n_coarse);
         HYPRE_SStructSSAMGSetMaxCoarseSize(precond, max_coarse_size);
         HYPRE_SStructSSAMGSetCoarseSolverType(precond, csolver_type);
         HYPRE_SStructSSAMGSetPrintLevel(precond, print_level);
         HYPRE_SStructSSAMGSetLogging(precond, 0);

         HYPRE_BiCGSTABSetPrecond( (HYPRE_Solver) solver,
                                   (HYPRE_PtrToSolverFcn) HYPRE_SStructSSAMGSolve,
                                   (HYPRE_PtrToSolverFcn) HYPRE_SStructSSAMGSetup,
                                   (HYPRE_Solver) precond);
      }
      else if (solver_id == 58)
      {
         /* use diagonal scaling as preconditioner */
         precond = NULL;
         HYPRE_BiCGSTABSetPrecond( (HYPRE_Solver) solver,
                                   (HYPRE_PtrToSolverFcn) HYPRE_SStructDiagScale,
                                   (HYPRE_PtrToSolverFcn) HYPRE_SStructDiagScaleSetup,
                                   (HYPRE_Solver) precond );
      }

      HYPRE_BiCGSTABSetup( (HYPRE_Solver) solver, (HYPRE_Matrix) A,
                           (HYPRE_Vector) b, (HYPRE_Vector) x );

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      time_index = hypre_InitializeTiming("BiCGSTAB Solve");
      hypre_BeginTiming(time_index);

      HYPRE_BiCGSTABSolve( (HYPRE_Solver) solver, (HYPRE_Matrix) A,
                           (HYPRE_Vector) b, (HYPRE_Vector) x );

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      HYPRE_BiCGSTABGetNumIterations( (HYPRE_Solver) solver, &num_iterations );
      HYPRE_BiCGSTABGetFinalRelativeResidualNorm( (HYPRE_Solver) solver, &final_res_norm );
      HYPRE_SStructBiCGSTABDestroy(solver);

      if ((solver_id == 50) || (solver_id == 51))
      {
         HYPRE_SStructSplitDestroy(precond);
      }
      else if (solver_id == 53)
      {
         HYPRE_SStructSysPFMGDestroy(precond);
      }
      else if (solver_id == 54)
      {
         HYPRE_SStructSSAMGDestroy(precond);
      }
   }

   /*-----------------------------------------------------------
    * Solve the system using ParCSR version of BiCGSTAB
    *-----------------------------------------------------------*/

   else if ((solver_id >= 60) && (solver_id < 70))
   {
      time_index = hypre_InitializeTiming("BiCGSTAB Setup");
      hypre_BeginTiming(time_index);

      HYPRE_ParCSRBiCGSTABCreate(comm, &par_solver);
      HYPRE_BiCGSTABSetMaxIter(par_solver, max_iterations);
      HYPRE_BiCGSTABSetTol(par_solver, tol);
      HYPRE_BiCGSTABSetPrintLevel(par_solver, krylov_print_level);
      HYPRE_BiCGSTABSetLogging(par_solver, 1);

      if (solver_id == 60)
      {
         /* use BoomerAMG as preconditioner */
         HYPRE_BoomerAMGCreate(&par_precond);
         HYPRE_BoomerAMGSetStrongThreshold(par_precond, strong_threshold);
         HYPRE_BoomerAMGSetPMaxElmts(par_precond, P_max_elmts);
         HYPRE_BoomerAMGSetCoarsenType(par_precond, coarsen_type);
         HYPRE_BoomerAMGSetMaxIter(par_precond, 1);
         HYPRE_BoomerAMGSetMaxLevels(par_precond, max_levels);
         HYPRE_BoomerAMGSetMaxCoarseSize(par_precond, max_coarse_size);
         HYPRE_BoomerAMGSetTol(par_precond, 0.0);
         HYPRE_BoomerAMGSetLogging(par_precond, 0);
         HYPRE_BoomerAMGSetPrintLevel(par_precond, print_level);
         HYPRE_BoomerAMGSetPrintFileName(par_precond, "sstruct.out.log");
         HYPRE_BoomerAMGSetCycleNumSweeps(par_precond, n_pre, 1);
         HYPRE_BoomerAMGSetCycleNumSweeps(par_precond, n_post, 2);
         HYPRE_BoomerAMGSetCycleNumSweeps(par_precond, n_pre, 3);
         if (usr_jacobi_weight)
         {
            HYPRE_BoomerAMGSetRelaxWt(par_precond, jacobi_weight);
         }
         if (relax[0] > -1)
         {
            HYPRE_BoomerAMGSetRelaxType(par_precond, relax[0]);
         }
         for (i = 1; i < 4; i++)
         {
            if (relax[i] > -1)
            {
               HYPRE_BoomerAMGSetCycleRelaxType(par_precond, relax[i], i);
            }
         }
         HYPRE_BoomerAMGSetAggNumLevels(par_precond, agg_num_levels);
         if (old_default)
         {
            HYPRE_BoomerAMGSetOldDefault(par_precond);
         }
         HYPRE_BiCGSTABSetPrecond( par_solver,
                                   (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve,
                                   (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup,
                                   par_precond);
      }
      else if (solver_id == 61)
      {
         /* use Euclid as preconditioner */
         HYPRE_EuclidCreate(comm, &par_precond);
         HYPRE_EuclidSetParams(par_precond, argc, argv);
         HYPRE_BiCGSTABSetPrecond(par_solver,
                                  (HYPRE_PtrToSolverFcn) HYPRE_EuclidSolve,
                                  (HYPRE_PtrToSolverFcn) HYPRE_EuclidSetup,
                                  par_precond);
      }
      else if (solver_id == 62)
      {
         /* use ParaSails as preconditioner */
         HYPRE_ParCSRParaSailsCreate(comm, &par_precond );
	 HYPRE_ParCSRParaSailsSetParams(par_precond, 0.1, 1);
	 HYPRE_ParCSRParaSailsSetSym(par_precond, 0);
         HYPRE_BiCGSTABSetPrecond( par_solver,
                                   (HYPRE_PtrToSolverFcn) HYPRE_ParCSRParaSailsSolve,
                                   (HYPRE_PtrToSolverFcn) HYPRE_ParCSRParaSailsSetup,
                                   par_precond);
      }

      HYPRE_BiCGSTABSetup( par_solver, (HYPRE_Matrix) par_A,
                           (HYPRE_Vector) par_b, (HYPRE_Vector) par_x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      time_index = hypre_InitializeTiming("BiCGSTAB Solve");
      hypre_BeginTiming(time_index);

      HYPRE_BiCGSTABSolve( par_solver, (HYPRE_Matrix) par_A,
                           (HYPRE_Vector) par_b, (HYPRE_Vector) par_x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      HYPRE_BiCGSTABGetNumIterations( par_solver, &num_iterations);
      HYPRE_BiCGSTABGetFinalRelativeResidualNorm( par_solver, &final_res_norm);
      HYPRE_ParCSRBiCGSTABDestroy(par_solver);

      if (solver_id == 60)
      {
         HYPRE_BoomerAMGDestroy(par_precond);
      }
      else if (solver_id == 61)
      {
         HYPRE_EuclidDestroy(par_precond);
      }
      else if (solver_id == 62)
      {
         HYPRE_ParCSRParaSailsDestroy(par_precond);
      }
   }

   /*-----------------------------------------------------------
    * Solve the system using Flexible GMRES
    *-----------------------------------------------------------*/

   else if ((solver_id >= 70) && (solver_id < 80))
   {
      time_index = hypre_InitializeTiming("FlexGMRES Setup");
      hypre_BeginTiming(time_index);

      HYPRE_SStructFlexGMRESCreate(comm, &solver);
      HYPRE_FlexGMRESSetKDim( (HYPRE_Solver) solver, k_dim );
      HYPRE_FlexGMRESSetMaxIter( (HYPRE_Solver) solver, max_iterations );
      HYPRE_FlexGMRESSetTol( (HYPRE_Solver) solver, tol );
      HYPRE_FlexGMRESSetPrintLevel( (HYPRE_Solver) solver, krylov_print_level );
      HYPRE_FlexGMRESSetLogging( (HYPRE_Solver) solver, 1 );

      if ((solver_id == 70) || (solver_id == 71))
      {
         /* use Split solver as preconditioner */
         HYPRE_SStructSplitCreate(comm, &precond);
         HYPRE_SStructSplitSetMaxIter(precond, 1);
         HYPRE_SStructSplitSetTol(precond, 0.0);
         HYPRE_SStructSplitSetZeroGuess(precond);
         HYPRE_SStructSplitSetPrintLevel(precond, print_level);
         HYPRE_SStructSplitSetLogging(precond, 0);
         if (solver_id == 70)
         {
            HYPRE_SStructSplitSetStructSolver(precond, HYPRE_SMG);
         }
         else if (solver_id == 71)
         {
            HYPRE_SStructSplitSetStructSolver(precond, HYPRE_PFMG);
         }
         HYPRE_FlexGMRESSetPrecond( (HYPRE_Solver) solver,
                                    (HYPRE_PtrToSolverFcn) HYPRE_SStructSplitSolve,
                                    (HYPRE_PtrToSolverFcn) HYPRE_SStructSplitSetup,
                                    (HYPRE_Solver) precond );
      }
      else if (solver_id == 73)
      {
         /* use SysPFMG solver as preconditioner */
         HYPRE_SStructSysPFMGCreate(comm, &precond);
         HYPRE_SStructSysPFMGSetMaxIter(precond, 1);
         HYPRE_SStructSysPFMGSetTol(precond, 0.0);
         HYPRE_SStructSysPFMGSetZeroGuess(precond);
         /* weighted Jacobi = 1; red-black GS = 2 */
         HYPRE_SStructSysPFMGSetRelaxType(precond, relax[0]);
         HYPRE_SStructSysPFMGSetNumPreRelax(precond, n_pre);
         HYPRE_SStructSysPFMGSetNumPostRelax(precond, n_post);
         HYPRE_SStructSysPFMGSetSkipRelax(precond, skip);
         /*HYPRE_SStructSysPFMGSetDxyz(precond, dxyz);*/
         HYPRE_SStructSysPFMGSetPrintLevel(precond, print_level);
         HYPRE_SStructSysPFMGSetLogging(precond, 0);

         HYPRE_FlexGMRESSetPrecond( (HYPRE_Solver) solver,
                                    (HYPRE_PtrToSolverFcn) HYPRE_SStructSysPFMGSolve,
                                    (HYPRE_PtrToSolverFcn) HYPRE_SStructSysPFMGSetup,
                                    (HYPRE_Solver) precond);
      }
      else if (solver_id == 74)
      {
         /* use SSAMG solver as preconditioner */
         HYPRE_SStructSSAMGCreate(comm, &precond);
         HYPRE_SStructSSAMGSetMaxIter(precond, 1);
         HYPRE_SStructSSAMGSetMaxLevels(precond, max_levels);
         HYPRE_SStructSSAMGSetTol(precond, 0.0);
         HYPRE_SStructSSAMGSetZeroGuess(precond);
         HYPRE_SStructSSAMGSetSkipRelax(precond, skip);
         HYPRE_SStructSSAMGSetRelaxType(precond, relax[0]);
         HYPRE_SStructSSAMGSetNonGalerkinRAP(precond, rap);
         if (usr_jacobi_weight)
         {
            HYPRE_SStructSSAMGSetRelaxWeight(precond, jacobi_weight);
         }
         HYPRE_SStructSSAMGSetNumPreRelax(precond, n_pre);
         HYPRE_SStructSSAMGSetNumPostRelax(precond, n_post);
         HYPRE_SStructSSAMGSetNumCoarseRelax(precond, n_coarse);
         HYPRE_SStructSSAMGSetMaxCoarseSize(precond, max_coarse_size);
         HYPRE_SStructSSAMGSetCoarseSolverType(precond, csolver_type);
         HYPRE_SStructSSAMGSetPrintLevel(precond, print_level);
         HYPRE_SStructSSAMGSetLogging(precond, 0);

         HYPRE_FlexGMRESSetPrecond( (HYPRE_Solver) solver,
                                    (HYPRE_PtrToSolverFcn) HYPRE_SStructSSAMGSolve,
                                    (HYPRE_PtrToSolverFcn) HYPRE_SStructSSAMGSetup,
                                    (HYPRE_Solver) precond);
      }
      else if (solver_id == 78)
      {
         /* use diagonal scaling as preconditioner */
         precond = NULL;
         HYPRE_FlexGMRESSetPrecond( (HYPRE_Solver) solver,
                                    (HYPRE_PtrToSolverFcn) HYPRE_SStructDiagScale,
                                    (HYPRE_PtrToSolverFcn) HYPRE_SStructDiagScaleSetup,
                                    (HYPRE_Solver) precond );
      }

      HYPRE_FlexGMRESSetup( (HYPRE_Solver) solver, (HYPRE_Matrix) A,
                            (HYPRE_Vector) b, (HYPRE_Vector) x );

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      time_index = hypre_InitializeTiming("FlexGMRES Solve");
      hypre_BeginTiming(time_index);

      HYPRE_FlexGMRESSolve( (HYPRE_Solver) solver, (HYPRE_Matrix) A,
                            (HYPRE_Vector) b, (HYPRE_Vector) x );

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      HYPRE_FlexGMRESGetNumIterations( (HYPRE_Solver) solver, &num_iterations );
      HYPRE_FlexGMRESGetFinalRelativeResidualNorm( (HYPRE_Solver) solver, &final_res_norm );
      HYPRE_SStructFlexGMRESDestroy(solver);

      if ((solver_id == 70) || (solver_id == 71))
      {
         HYPRE_SStructSplitDestroy(precond);
      }
      else if (solver_id == 73)
      {
         HYPRE_SStructSysPFMGDestroy(precond);
      }
      else if (solver_id == 74)
      {
         HYPRE_SStructSSAMGDestroy(precond);
      }
   }

   /*-----------------------------------------------------------
    * Solve the system using ParCSR version of Flexible GMRES
    *-----------------------------------------------------------*/

   else if ((solver_id >= 80) && (solver_id < 90))
   {
      time_index = hypre_InitializeTiming("FlexGMRES Setup");
      hypre_BeginTiming(time_index);

      HYPRE_ParCSRFlexGMRESCreate(comm, &par_solver);
      HYPRE_FlexGMRESSetKDim(par_solver, k_dim);
      HYPRE_FlexGMRESSetMaxIter(par_solver, max_iterations);
      HYPRE_FlexGMRESSetTol(par_solver, tol);
      HYPRE_FlexGMRESSetPrintLevel(par_solver, krylov_print_level);
      HYPRE_FlexGMRESSetLogging(par_solver, 1);

      if (solver_id == 80)
      {
         /* use BoomerAMG as preconditioner */
         HYPRE_BoomerAMGCreate(&par_precond);
         HYPRE_BoomerAMGSetStrongThreshold(par_precond, strong_threshold);
         HYPRE_BoomerAMGSetPMaxElmts(par_precond, P_max_elmts);
         HYPRE_BoomerAMGSetCoarsenType(par_precond, coarsen_type);
         HYPRE_BoomerAMGSetMaxIter(par_precond, 1);
         HYPRE_BoomerAMGSetMaxLevels(par_precond, max_levels);
         HYPRE_BoomerAMGSetMaxCoarseSize(par_precond, max_coarse_size);
         HYPRE_BoomerAMGSetTol(par_precond, 0.0);
         HYPRE_BoomerAMGSetLogging(par_precond, 0);
         HYPRE_BoomerAMGSetPrintLevel(par_precond, print_level);
         HYPRE_BoomerAMGSetPrintFileName(par_precond, "sstruct.out.log");
         HYPRE_BoomerAMGSetCycleNumSweeps(par_precond, n_pre, 1);
         HYPRE_BoomerAMGSetCycleNumSweeps(par_precond, n_post, 2);
         HYPRE_BoomerAMGSetCycleNumSweeps(par_precond, n_pre, 3);
         if (usr_jacobi_weight)
         {
            HYPRE_BoomerAMGSetRelaxWt(par_precond, jacobi_weight);
         }
         if (relax[0] > -1)
         {
            HYPRE_BoomerAMGSetRelaxType(par_precond, relax[0]);
         }
         for (i = 1; i < 4; i++)
         {
            if (relax[i] > -1)
            {
               HYPRE_BoomerAMGSetCycleRelaxType(par_precond, relax[i], i);
            }
         }
         HYPRE_BoomerAMGSetAggNumLevels(par_precond, agg_num_levels);
         if (old_default)
         {
            HYPRE_BoomerAMGSetOldDefault(par_precond);
         }
         HYPRE_FlexGMRESSetPrecond( par_solver,
                                    (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve,
                                    (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup,
                                    par_precond);
      }

      HYPRE_FlexGMRESSetup( par_solver, (HYPRE_Matrix) par_A,
                            (HYPRE_Vector) par_b, (HYPRE_Vector) par_x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      time_index = hypre_InitializeTiming("FlexGMRES Solve");
      hypre_BeginTiming(time_index);

      HYPRE_FlexGMRESSolve( par_solver, (HYPRE_Matrix) par_A,
                            (HYPRE_Vector) par_b, (HYPRE_Vector) par_x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      HYPRE_FlexGMRESGetNumIterations( par_solver, &num_iterations);
      HYPRE_FlexGMRESGetFinalRelativeResidualNorm( par_solver, &final_res_norm);
      HYPRE_ParCSRFlexGMRESDestroy(par_solver);

      if (solver_id == 80)
      {
         HYPRE_BoomerAMGDestroy(par_precond);
      }
   }

   /*-----------------------------------------------------------
    * Solve the system using ParCSR version of LGMRES
    *-----------------------------------------------------------*/

   else if ((solver_id >= 90) && (solver_id < 100))
   {
      time_index = hypre_InitializeTiming("LGMRES Setup");
      hypre_BeginTiming(time_index);

      HYPRE_ParCSRLGMRESCreate(comm, &par_solver);
      HYPRE_LGMRESSetKDim(par_solver, k_dim);
      HYPRE_LGMRESSetAugDim(par_solver, aug_dim);
      HYPRE_LGMRESSetMaxIter(par_solver, max_iterations);
      HYPRE_LGMRESSetTol(par_solver, tol);
      HYPRE_LGMRESSetPrintLevel(par_solver, krylov_print_level);
      HYPRE_LGMRESSetLogging(par_solver, 1);

      if (solver_id == 90)
      {
         /* use BoomerAMG as preconditioner */
         HYPRE_BoomerAMGCreate(&par_precond);
         HYPRE_BoomerAMGSetStrongThreshold(par_precond, strong_threshold);
         HYPRE_BoomerAMGSetPMaxElmts(par_precond, P_max_elmts);
         HYPRE_BoomerAMGSetCoarsenType(par_precond, coarsen_type);
         HYPRE_BoomerAMGSetMaxIter(par_precond, 1);
         HYPRE_BoomerAMGSetMaxLevels(par_precond, max_levels);
         HYPRE_BoomerAMGSetMaxCoarseSize(par_precond, max_coarse_size);
         HYPRE_BoomerAMGSetTol(par_precond, 0.0);
         HYPRE_BoomerAMGSetLogging(par_precond, 0);
         HYPRE_BoomerAMGSetPrintLevel(par_precond, print_level);
         HYPRE_BoomerAMGSetPrintFileName(par_precond, "sstruct.out.log");
         HYPRE_BoomerAMGSetCycleNumSweeps(par_precond, n_pre, 1);
         HYPRE_BoomerAMGSetCycleNumSweeps(par_precond, n_post, 2);
         HYPRE_BoomerAMGSetCycleNumSweeps(par_precond, n_pre, 3);
         if (usr_jacobi_weight)
         {
            HYPRE_BoomerAMGSetRelaxWt(par_precond, jacobi_weight);
         }
         if (relax[0] > -1)
         {
            HYPRE_BoomerAMGSetRelaxType(par_precond, relax[0]);
         }
         for (i = 1; i < 4; i++)
         {
            if (relax[i] > -1)
            {
               HYPRE_BoomerAMGSetCycleRelaxType(par_precond, relax[i], i);
            }
         }
         HYPRE_BoomerAMGSetAggNumLevels(par_precond, agg_num_levels);
         if (old_default)
         {
            HYPRE_BoomerAMGSetOldDefault(par_precond);
         }
         HYPRE_LGMRESSetPrecond( par_solver,
                                 (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve,
                                 (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup,
                                 par_precond);
      }

      HYPRE_LGMRESSetup( par_solver, (HYPRE_Matrix) par_A,
                         (HYPRE_Vector) par_b, (HYPRE_Vector) par_x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      time_index = hypre_InitializeTiming("LGMRES Solve");
      hypre_BeginTiming(time_index);

      HYPRE_LGMRESSolve( par_solver, (HYPRE_Matrix) par_A,
                         (HYPRE_Vector) par_b, (HYPRE_Vector) par_x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      HYPRE_LGMRESGetNumIterations( par_solver, &num_iterations);
      HYPRE_LGMRESGetFinalRelativeResidualNorm( par_solver, &final_res_norm);
      HYPRE_ParCSRLGMRESDestroy(par_solver);

      if (solver_id == 90)
      {
         HYPRE_BoomerAMGDestroy(par_precond);
      }
   }

   /*-----------------------------------------------------------
    * Solve the system using ParCSR hybrid DSCG/BoomerAMG
    *-----------------------------------------------------------*/

   else if (solver_id == 120)
   {
      time_index = hypre_InitializeTiming("Hybrid Setup");
      hypre_BeginTiming(time_index);

      HYPRE_ParCSRHybridCreate(&par_solver);
      HYPRE_ParCSRHybridSetTol(par_solver, tol);
      HYPRE_ParCSRHybridSetTwoNorm(par_solver, 1);
      HYPRE_ParCSRHybridSetRelChange(par_solver, rel_change);
      HYPRE_ParCSRHybridSetPrintLevel(par_solver, print_level);
      HYPRE_ParCSRHybridSetLogging(par_solver, 1);
      HYPRE_ParCSRHybridSetSolverType(par_solver, solver_type);
      HYPRE_ParCSRHybridSetRecomputeResidual(par_solver, recompute_res);

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP)
      /*
      HYPRE_ParCSRHybridSetPMaxElmts(par_solver, 8);
      HYPRE_ParCSRHybridSetRelaxType(par_solver, 18);
      HYPRE_ParCSRHybridSetCycleRelaxType(par_solver, 9, 3);
      HYPRE_ParCSRHybridSetCoarsenType(par_solver, 8);
      HYPRE_ParCSRHybridSetInterpType(par_solver, 3);
      HYPRE_ParCSRHybridSetMaxCoarseSize(par_solver, 20);
      */
#endif

#if SECOND_TIME
      hypre_ParVector *par_x2 =
         hypre_ParVectorCreate(hypre_ParVectorComm(par_x), hypre_ParVectorGlobalSize(par_x),
                               hypre_ParVectorPartitioning(par_x));
      hypre_ParVectorInitialize(par_x2);
      hypre_ParVectorCopy(par_x, par_x2);

      HYPRE_ParCSRHybridSetup(par_solver,par_A,par_b,par_x);
      HYPRE_ParCSRHybridSolve(par_solver,par_A,par_b,par_x);

      hypre_ParVectorCopy(par_x2, par_x);
#endif

#if defined(HYPRE_USING_NVTX)
      hypre_GpuProfilingPushRange("HybridSolve");
#endif
      //cudaProfilerStart();

      HYPRE_ParCSRHybridSetup(par_solver,par_A,par_b,par_x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      time_index = hypre_InitializeTiming("Hybrid Solve");
      hypre_BeginTiming(time_index);

      HYPRE_ParCSRHybridSolve(par_solver,par_A,par_b,par_x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      HYPRE_ParCSRHybridGetNumIterations(par_solver, &num_iterations);
      HYPRE_ParCSRHybridGetFinalRelativeResidualNorm(par_solver, &final_res_norm);

      HYPRE_ParCSRHybridDestroy(par_solver);

#if defined(HYPRE_USING_NVTX)
      hypre_GpuProfilingPopRange();
#endif
      //cudaProfilerStop();

#if SECOND_TIME
      hypre_ParVectorDestroy(par_x2);
#endif
   }

   /*-----------------------------------------------------------
    * Solve the system using Struct solvers
    *-----------------------------------------------------------*/

   else if (solver_id == 200)
   {
      time_index = hypre_InitializeTiming("SMG Setup");
      hypre_BeginTiming(time_index);

      HYPRE_StructSMGCreate(comm, &struct_solver);
      HYPRE_StructSMGSetMemoryUse(struct_solver, 0);
      HYPRE_StructSMGSetMaxIter(struct_solver, max_iterations);
      HYPRE_StructSMGSetTol(struct_solver, tol);
      HYPRE_StructSMGSetRelChange(struct_solver, rel_change);
      HYPRE_StructSMGSetNumPreRelax(struct_solver, n_pre);
      HYPRE_StructSMGSetNumPostRelax(struct_solver, n_post);
      HYPRE_StructSMGSetPrintLevel(struct_solver, print_level);
      HYPRE_StructSMGSetLogging(struct_solver, 1);
      HYPRE_StructSMGSetup(struct_solver, sA, sb, sx);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      time_index = hypre_InitializeTiming("SMG Solve");
      hypre_BeginTiming(time_index);

      HYPRE_StructSMGSolve(struct_solver, sA, sb, sx);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      HYPRE_StructSMGGetNumIterations(struct_solver, &num_iterations);
      HYPRE_StructSMGGetFinalRelativeResidualNorm(struct_solver, &final_res_norm);
      HYPRE_StructSMGDestroy(struct_solver);
   }

   else if ( solver_id == 201 || solver_id == 203 || solver_id == 204 )
   {
      time_index = hypre_InitializeTiming("PFMG Setup");
      hypre_BeginTiming(time_index);

      HYPRE_StructPFMGCreate(comm, &struct_solver);
      HYPRE_StructPFMGSetMaxLevels(struct_solver, max_levels);
      HYPRE_StructPFMGSetMaxIter(struct_solver, max_iterations);
      HYPRE_StructPFMGSetTol(struct_solver, tol);
      HYPRE_StructPFMGSetRelChange(struct_solver, rel_change);
      HYPRE_StructPFMGSetRAPType(struct_solver, rap);
      HYPRE_StructPFMGSetRelaxType(struct_solver, relax[0]);
      if (usr_jacobi_weight)
      {
         HYPRE_StructPFMGSetJacobiWeight(struct_solver, jacobi_weight);
      }
      HYPRE_StructPFMGSetNumPreRelax(struct_solver, n_pre);
      HYPRE_StructPFMGSetNumPostRelax(struct_solver, n_post);
      HYPRE_StructPFMGSetSkipRelax(struct_solver, skip);
      /*HYPRE_StructPFMGSetDxyz(struct_solver, dxyz);*/
      HYPRE_StructPFMGSetPrintLevel(struct_solver, print_level);
      HYPRE_StructPFMGSetLogging(struct_solver, 1);
      HYPRE_StructPFMGSetup(struct_solver, sA, sb, sx);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      time_index = hypre_InitializeTiming("PFMG Solve");
      hypre_BeginTiming(time_index);

      HYPRE_StructPFMGSolve(struct_solver, sA, sb, sx);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      HYPRE_StructPFMGGetNumIterations(struct_solver, &num_iterations);
      HYPRE_StructPFMGGetFinalRelativeResidualNorm(struct_solver, &final_res_norm);
      HYPRE_StructPFMGDestroy(struct_solver);
   }

   /*-----------------------------------------------------------
    * Solve the system using Cyclic Reduction
    *-----------------------------------------------------------*/

   else if ( solver_id == 205 )
   {
      HYPRE_StructVector  sr;

      time_index = hypre_InitializeTiming("CycRed Setup");
      hypre_BeginTiming(time_index);

      HYPRE_StructCycRedCreate(comm, &struct_solver);
      HYPRE_StructCycRedSetTDim(struct_solver, cycred_tdim);
      HYPRE_StructCycRedSetBase(struct_solver, data.ndim,
                                cycred_index, cycred_stride);
      HYPRE_StructCycRedSetup(struct_solver, sA, sb, sx);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      time_index = hypre_InitializeTiming("CycRed Solve");
      hypre_BeginTiming(time_index);

      HYPRE_StructCycRedSolve(struct_solver, sA, sb, sx);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      num_iterations = 1;
      HYPRE_StructVectorCreate(comm,
                               hypre_StructVectorGrid(sb), &sr);
      HYPRE_StructVectorInitialize(sr);
      HYPRE_StructVectorAssemble(sr);
      HYPRE_StructVectorCopy(sb, sr);
      hypre_StructMatvec(-1.0, sA, sx, 1.0, sr);
      /* Using an inner product instead of a norm to help with testing */
      final_res_norm = hypre_StructInnerProd(sr, sr);
      if (final_res_norm < 1.0e-20)
      {
         final_res_norm = 0.0;
      }
      HYPRE_StructVectorDestroy(sr);

      HYPRE_StructCycRedDestroy(struct_solver);
   }

   /*-----------------------------------------------------------
    * Solve the system using Jacobi
    *-----------------------------------------------------------*/

   else if ( solver_id == 208 )
   {
      time_index = hypre_InitializeTiming("Jacobi Setup");
      hypre_BeginTiming(time_index);

      HYPRE_StructJacobiCreate(comm, &struct_solver);
      HYPRE_StructJacobiSetMaxIter(struct_solver, max_iterations);
      HYPRE_StructJacobiSetTol(struct_solver, tol);
      HYPRE_StructJacobiSetup(struct_solver, sA, sb, sx);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      time_index = hypre_InitializeTiming("Jacobi Solve");
      hypre_BeginTiming(time_index);

      HYPRE_StructJacobiSolve(struct_solver, sA, sb, sx);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      HYPRE_StructJacobiGetNumIterations(struct_solver, &num_iterations);
      HYPRE_StructJacobiGetFinalRelativeResidualNorm(struct_solver,
                                                     &final_res_norm);
      HYPRE_StructJacobiDestroy(struct_solver);
   }

   /*-----------------------------------------------------------
    * Solve the system using CG
    *-----------------------------------------------------------*/

   else if ((solver_id > 209) && (solver_id < 220))
   {
      time_index = hypre_InitializeTiming("PCG Setup");
      hypre_BeginTiming(time_index);

      HYPRE_StructPCGCreate(comm, &struct_solver);
      HYPRE_PCGSetMaxIter( (HYPRE_Solver)struct_solver, max_iterations );
      HYPRE_PCGSetTol( (HYPRE_Solver)struct_solver, tol );
      HYPRE_PCGSetTwoNorm( (HYPRE_Solver)struct_solver, 1 );
      HYPRE_PCGSetRelChange( (HYPRE_Solver)struct_solver, rel_change );
      HYPRE_PCGSetPrintLevel( (HYPRE_Solver)struct_solver, krylov_print_level );
      HYPRE_PCGSetRecomputeResidual( (HYPRE_Solver)struct_solver, recompute_res);

      if (solver_id == 210)
      {
         /* use symmetric SMG as preconditioner */
         HYPRE_StructSMGCreate(comm, &struct_precond);
         HYPRE_StructSMGSetMemoryUse(struct_precond, 0);
         HYPRE_StructSMGSetMaxIter(struct_precond, 1);
         HYPRE_StructSMGSetTol(struct_precond, 0.0);
         HYPRE_StructSMGSetZeroGuess(struct_precond);
         HYPRE_StructSMGSetNumPreRelax(struct_precond, n_pre);
         HYPRE_StructSMGSetNumPostRelax(struct_precond, n_post);
         HYPRE_StructSMGSetPrintLevel(struct_precond, print_level);
         HYPRE_StructSMGSetLogging(struct_precond, 0);
         HYPRE_PCGSetPrecond( (HYPRE_Solver) struct_solver,
                              (HYPRE_PtrToSolverFcn) HYPRE_StructSMGSolve,
                              (HYPRE_PtrToSolverFcn) HYPRE_StructSMGSetup,
                              (HYPRE_Solver) struct_precond);
      }

      else if (solver_id == 211)
      {
         /* use symmetric PFMG as preconditioner */
         HYPRE_StructPFMGCreate(comm, &struct_precond);
         HYPRE_StructPFMGSetMaxLevels(struct_precond, max_levels);
         HYPRE_StructPFMGSetMaxIter(struct_precond, 1);
         HYPRE_StructPFMGSetTol(struct_precond, 0.0);
         HYPRE_StructPFMGSetZeroGuess(struct_precond);
         HYPRE_StructPFMGSetRAPType(struct_precond, rap);
         HYPRE_StructPFMGSetRelaxType(struct_precond, relax[0]);
         if (usr_jacobi_weight)
         {
            HYPRE_StructPFMGSetJacobiWeight(struct_precond, jacobi_weight);
         }
         HYPRE_StructPFMGSetNumPreRelax(struct_precond, n_pre);
         HYPRE_StructPFMGSetNumPostRelax(struct_precond, n_post);
         HYPRE_StructPFMGSetSkipRelax(struct_precond, skip);
         /*HYPRE_StructPFMGSetDxyz(struct_precond, dxyz);*/
         HYPRE_StructPFMGSetPrintLevel(struct_precond, print_level);
         HYPRE_StructPFMGSetLogging(struct_precond, 0);
         HYPRE_PCGSetPrecond( (HYPRE_Solver) struct_solver,
                              (HYPRE_PtrToSolverFcn) HYPRE_StructPFMGSolve,
                              (HYPRE_PtrToSolverFcn) HYPRE_StructPFMGSetup,
                              (HYPRE_Solver) struct_precond);
      }

      else if (solver_id == 217)
      {
         /* use two-step Jacobi as preconditioner */
         HYPRE_StructJacobiCreate(comm, &struct_precond);
         HYPRE_StructJacobiSetMaxIter(struct_precond, 2);
         HYPRE_StructJacobiSetTol(struct_precond, 0.0);
         HYPRE_StructJacobiSetZeroGuess(struct_precond);
         HYPRE_PCGSetPrecond( (HYPRE_Solver) struct_solver,
                              (HYPRE_PtrToSolverFcn) HYPRE_StructJacobiSolve,
                              (HYPRE_PtrToSolverFcn) HYPRE_StructJacobiSetup,
                              (HYPRE_Solver) struct_precond);
      }

      else if (solver_id == 218)
      {
         /* use diagonal scaling as preconditioner */
         struct_precond = NULL;
         HYPRE_PCGSetPrecond( (HYPRE_Solver) struct_solver,
                              (HYPRE_PtrToSolverFcn) HYPRE_StructDiagScale,
                              (HYPRE_PtrToSolverFcn) HYPRE_StructDiagScaleSetup,
                              (HYPRE_Solver) struct_precond);
      }

      HYPRE_PCGSetup
         ( (HYPRE_Solver)struct_solver, (HYPRE_Matrix)sA, (HYPRE_Vector)sb,
           (HYPRE_Vector)sx );

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      time_index = hypre_InitializeTiming("PCG Solve");
      hypre_BeginTiming(time_index);

      HYPRE_PCGSolve
         ( (HYPRE_Solver) struct_solver, (HYPRE_Matrix)sA, (HYPRE_Vector)sb,
           (HYPRE_Vector)sx);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      HYPRE_PCGGetNumIterations( (HYPRE_Solver)struct_solver, &num_iterations );
      HYPRE_PCGGetFinalRelativeResidualNorm( (HYPRE_Solver)struct_solver, &final_res_norm );
      HYPRE_StructPCGDestroy(struct_solver);

      if (solver_id == 210)
      {
         HYPRE_StructSMGDestroy(struct_precond);
      }
      else if (solver_id == 211)
      {
         HYPRE_StructPFMGDestroy(struct_precond);
      }
      else if (solver_id == 217)
      {
         HYPRE_StructJacobiDestroy(struct_precond);
      }
   }

   /*-----------------------------------------------------------
    * Solve the system using Hybrid
    *-----------------------------------------------------------*/

   else if ((solver_id > 219) && (solver_id < 230))
   {
      time_index = hypre_InitializeTiming("Hybrid Setup");
      hypre_BeginTiming(time_index);

      HYPRE_StructHybridCreate(comm, &struct_solver);
      HYPRE_StructHybridSetDSCGMaxIter(struct_solver, max_iterations);
      HYPRE_StructHybridSetPCGMaxIter(struct_solver, max_iterations);
      HYPRE_StructHybridSetTol(struct_solver, tol);
      /*HYPRE_StructHybridSetPCGAbsoluteTolFactor(struct_solver, 1.0e-200);*/
      HYPRE_StructHybridSetConvergenceTol(struct_solver, cf_tol);
      HYPRE_StructHybridSetTwoNorm(struct_solver, 1);
      HYPRE_StructHybridSetRelChange(struct_solver, rel_change);
      if (solver_type == 2) /* for use with GMRES */
      {
         HYPRE_StructHybridSetStopCrit(struct_solver, 0);
         HYPRE_StructHybridSetKDim(struct_solver, 10);
      }
      HYPRE_StructHybridSetPrintLevel(struct_solver, krylov_print_level);
      HYPRE_StructHybridSetLogging(struct_solver, 1);
      HYPRE_StructHybridSetSolverType(struct_solver, solver_type);
      HYPRE_StructHybridSetRecomputeResidual(struct_solver, recompute_res);

      if (solver_id == 220)
      {
         /* use symmetric SMG as preconditioner */
         HYPRE_StructSMGCreate(comm, &struct_precond);
         HYPRE_StructSMGSetMemoryUse(struct_precond, 0);
         HYPRE_StructSMGSetMaxIter(struct_precond, 1);
         HYPRE_StructSMGSetTol(struct_precond, 0.0);
         HYPRE_StructSMGSetZeroGuess(struct_precond);
         HYPRE_StructSMGSetNumPreRelax(struct_precond, n_pre);
         HYPRE_StructSMGSetNumPostRelax(struct_precond, n_post);
         HYPRE_StructSMGSetPrintLevel(struct_precond, print_level);
         HYPRE_StructSMGSetLogging(struct_precond, 0);
         HYPRE_StructHybridSetPrecond(struct_solver,
                                      HYPRE_StructSMGSolve,
                                      HYPRE_StructSMGSetup,
                                      struct_precond);
      }

      else if (solver_id == 221)
      {
         /* use symmetric PFMG as preconditioner */
         HYPRE_StructPFMGCreate(comm, &struct_precond);
         HYPRE_StructPFMGSetMaxLevels(struct_precond, max_levels);
         HYPRE_StructPFMGSetMaxIter(struct_precond, 1);
         HYPRE_StructPFMGSetTol(struct_precond, 0.0);
         HYPRE_StructPFMGSetZeroGuess(struct_precond);
         HYPRE_StructPFMGSetRAPType(struct_precond, rap);
         HYPRE_StructPFMGSetRelaxType(struct_precond, relax[0]);
         if (usr_jacobi_weight)
         {
            HYPRE_StructPFMGSetJacobiWeight(struct_precond, jacobi_weight);
         }
         HYPRE_StructPFMGSetNumPreRelax(struct_precond, n_pre);
         HYPRE_StructPFMGSetNumPostRelax(struct_precond, n_post);
         HYPRE_StructPFMGSetSkipRelax(struct_precond, skip);
         /*HYPRE_StructPFMGSetDxyz(struct_precond, dxyz);*/
         HYPRE_StructPFMGSetPrintLevel(struct_precond, print_level);
         HYPRE_StructPFMGSetLogging(struct_precond, 0);
         HYPRE_StructHybridSetPrecond(struct_solver,
                                      HYPRE_StructPFMGSolve,
                                      HYPRE_StructPFMGSetup,
                                      struct_precond);
      }

      HYPRE_StructHybridSetup(struct_solver, sA, sb, sx);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      time_index = hypre_InitializeTiming("Hybrid Solve");
      hypre_BeginTiming(time_index);

      HYPRE_StructHybridSolve(struct_solver, sA, sb, sx);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      HYPRE_StructHybridGetNumIterations(struct_solver, &num_iterations);
      HYPRE_StructHybridGetFinalRelativeResidualNorm(struct_solver, &final_res_norm);
      HYPRE_StructHybridDestroy(struct_solver);

      if (solver_id == 220)
      {
         HYPRE_StructSMGDestroy(struct_precond);
      }
      else if (solver_id == 221)
      {
         HYPRE_StructPFMGDestroy(struct_precond);
      }
   }

   /*-----------------------------------------------------------
    * Solve the system using GMRES
    *-----------------------------------------------------------*/

   else if ((solver_id > 229) && (solver_id < 240))
   {
      time_index = hypre_InitializeTiming("GMRES Setup");
      hypre_BeginTiming(time_index);

      HYPRE_StructGMRESCreate(comm, &struct_solver);
      HYPRE_GMRESSetKDim( (HYPRE_Solver) struct_solver, k_dim );
      HYPRE_GMRESSetMaxIter( (HYPRE_Solver) struct_solver, max_iterations );
      HYPRE_GMRESSetTol( (HYPRE_Solver) struct_solver, tol );
      HYPRE_GMRESSetRelChange( (HYPRE_Solver) struct_solver, rel_change );
      HYPRE_GMRESSetPrintLevel( (HYPRE_Solver) struct_solver, krylov_print_level );
      HYPRE_GMRESSetLogging( (HYPRE_Solver) struct_solver, 1 );

      if (solver_id == 230)
      {
         /* use symmetric SMG as preconditioner */
         HYPRE_StructSMGCreate(comm, &struct_precond);
         HYPRE_StructSMGSetMemoryUse(struct_precond, 0);
         HYPRE_StructSMGSetMaxIter(struct_precond, 1);
         HYPRE_StructSMGSetTol(struct_precond, 0.0);
         HYPRE_StructSMGSetZeroGuess(struct_precond);
         HYPRE_StructSMGSetNumPreRelax(struct_precond, n_pre);
         HYPRE_StructSMGSetNumPostRelax(struct_precond, n_post);
         HYPRE_StructSMGSetPrintLevel(struct_precond, print_level);
         HYPRE_StructSMGSetLogging(struct_precond, 0);
         HYPRE_GMRESSetPrecond( (HYPRE_Solver)struct_solver,
                                (HYPRE_PtrToSolverFcn) HYPRE_StructSMGSolve,
                                (HYPRE_PtrToSolverFcn) HYPRE_StructSMGSetup,
                                (HYPRE_Solver)struct_precond);
      }

      else if (solver_id == 231)
      {
         /* use symmetric PFMG as preconditioner */
         HYPRE_StructPFMGCreate(comm, &struct_precond);
         HYPRE_StructPFMGSetMaxLevels(struct_precond, max_levels);
         HYPRE_StructPFMGSetMaxIter(struct_precond, 1);
         HYPRE_StructPFMGSetTol(struct_precond, 0.0);
         HYPRE_StructPFMGSetZeroGuess(struct_precond);
         HYPRE_StructPFMGSetRAPType(struct_precond, rap);
         HYPRE_StructPFMGSetRelaxType(struct_precond, relax[0]);
         if (usr_jacobi_weight)
         {
            HYPRE_StructPFMGSetJacobiWeight(struct_precond, jacobi_weight);
         }
         HYPRE_StructPFMGSetNumPreRelax(struct_precond, n_pre);
         HYPRE_StructPFMGSetNumPostRelax(struct_precond, n_post);
         HYPRE_StructPFMGSetSkipRelax(struct_precond, skip);
         /*HYPRE_StructPFMGSetDxyz(struct_precond, dxyz);*/
         HYPRE_StructPFMGSetPrintLevel(struct_precond, print_level);
         HYPRE_StructPFMGSetLogging(struct_precond, 0);
         HYPRE_GMRESSetPrecond( (HYPRE_Solver)struct_solver,
                                (HYPRE_PtrToSolverFcn) HYPRE_StructPFMGSolve,
                                (HYPRE_PtrToSolverFcn) HYPRE_StructPFMGSetup,
                                (HYPRE_Solver)struct_precond);
      }
      else if (solver_id == 237)
      {
         /* use two-step Jacobi as preconditioner */
         HYPRE_StructJacobiCreate(comm, &struct_precond);
         HYPRE_StructJacobiSetMaxIter(struct_precond, 2);
         HYPRE_StructJacobiSetTol(struct_precond, 0.0);
         HYPRE_StructJacobiSetZeroGuess(struct_precond);
         HYPRE_GMRESSetPrecond( (HYPRE_Solver)struct_solver,
                                (HYPRE_PtrToSolverFcn) HYPRE_StructJacobiSolve,
                                (HYPRE_PtrToSolverFcn) HYPRE_StructJacobiSetup,
                                (HYPRE_Solver)struct_precond);
      }

      else if (solver_id == 238)
      {
         /* use diagonal scaling as preconditioner */
         struct_precond = NULL;
         HYPRE_GMRESSetPrecond( (HYPRE_Solver)struct_solver,
                                (HYPRE_PtrToSolverFcn) HYPRE_StructDiagScale,
                                (HYPRE_PtrToSolverFcn) HYPRE_StructDiagScaleSetup,
                                (HYPRE_Solver)struct_precond);
      }

      HYPRE_GMRESSetup
         ( (HYPRE_Solver)struct_solver, (HYPRE_Matrix)sA, (HYPRE_Vector)sb,
           (HYPRE_Vector)sx );

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      time_index = hypre_InitializeTiming("GMRES Solve");
      hypre_BeginTiming(time_index);

      HYPRE_GMRESSolve
         ( (HYPRE_Solver)struct_solver, (HYPRE_Matrix)sA, (HYPRE_Vector)sb,
           (HYPRE_Vector)sx);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      HYPRE_GMRESGetNumIterations( (HYPRE_Solver)struct_solver, &num_iterations);
      HYPRE_GMRESGetFinalRelativeResidualNorm( (HYPRE_Solver)struct_solver, &final_res_norm);
      HYPRE_StructGMRESDestroy(struct_solver);

      if (solver_id == 230)
      {
         HYPRE_StructSMGDestroy(struct_precond);
      }
      else if (solver_id == 231)
      {
         HYPRE_StructPFMGDestroy(struct_precond);
      }
      else if (solver_id == 237)
      {
         HYPRE_StructJacobiDestroy(struct_precond);
      }
   }

   /*-----------------------------------------------------------
    * Solve the system using BiCGTAB
    *-----------------------------------------------------------*/

   else if ((solver_id > 239) && (solver_id < 250))
   {
      time_index = hypre_InitializeTiming("BiCGSTAB Setup");
      hypre_BeginTiming(time_index);

      HYPRE_StructBiCGSTABCreate(comm, &struct_solver);
      HYPRE_BiCGSTABSetMaxIter( (HYPRE_Solver)struct_solver, max_iterations );
      HYPRE_BiCGSTABSetTol( (HYPRE_Solver)struct_solver, tol );
      HYPRE_BiCGSTABSetPrintLevel( (HYPRE_Solver)struct_solver, krylov_print_level );
      HYPRE_BiCGSTABSetLogging( (HYPRE_Solver)struct_solver, 1 );

      if (solver_id == 240)
      {
         /* use symmetric SMG as preconditioner */
         HYPRE_StructSMGCreate(comm, &struct_precond);
         HYPRE_StructSMGSetMemoryUse(struct_precond, 0);
         HYPRE_StructSMGSetMaxIter(struct_precond, 1);
         HYPRE_StructSMGSetTol(struct_precond, 0.0);
         HYPRE_StructSMGSetZeroGuess(struct_precond);
         HYPRE_StructSMGSetNumPreRelax(struct_precond, n_pre);
         HYPRE_StructSMGSetNumPostRelax(struct_precond, n_post);
         HYPRE_StructSMGSetPrintLevel(struct_precond, print_level);
         HYPRE_StructSMGSetLogging(struct_precond, 0);
         HYPRE_BiCGSTABSetPrecond( (HYPRE_Solver)struct_solver,
                                   (HYPRE_PtrToSolverFcn) HYPRE_StructSMGSolve,
                                   (HYPRE_PtrToSolverFcn) HYPRE_StructSMGSetup,
                                   (HYPRE_Solver)struct_precond);
      }

      else if (solver_id == 241)
      {
         /* use symmetric PFMG as preconditioner */
         HYPRE_StructPFMGCreate(comm, &struct_precond);
         HYPRE_StructPFMGSetMaxLevels(struct_precond, max_levels);
         HYPRE_StructPFMGSetMaxIter(struct_precond, 1);
         HYPRE_StructPFMGSetTol(struct_precond, 0.0);
         HYPRE_StructPFMGSetZeroGuess(struct_precond);
         HYPRE_StructPFMGSetRAPType(struct_precond, rap);
         HYPRE_StructPFMGSetRelaxType(struct_precond, relax[0]);
         if (usr_jacobi_weight)
         {
            HYPRE_StructPFMGSetJacobiWeight(struct_precond, jacobi_weight);
         }
         HYPRE_StructPFMGSetNumPreRelax(struct_precond, n_pre);
         HYPRE_StructPFMGSetNumPostRelax(struct_precond, n_post);
         HYPRE_StructPFMGSetSkipRelax(struct_precond, skip);
         /*HYPRE_StructPFMGSetDxyz(struct_precond, dxyz);*/
         HYPRE_StructPFMGSetPrintLevel(struct_precond, print_level);
         HYPRE_StructPFMGSetLogging(struct_precond, 0);
         HYPRE_BiCGSTABSetPrecond( (HYPRE_Solver)struct_solver,
                                   (HYPRE_PtrToSolverFcn) HYPRE_StructPFMGSolve,
                                   (HYPRE_PtrToSolverFcn) HYPRE_StructPFMGSetup,
                                   (HYPRE_Solver)struct_precond);
      }

      else if (solver_id == 247)
      {
         /* use two-step Jacobi as preconditioner */
         HYPRE_StructJacobiCreate(comm, &struct_precond);
         HYPRE_StructJacobiSetMaxIter(struct_precond, 2);
         HYPRE_StructJacobiSetTol(struct_precond, 0.0);
         HYPRE_StructJacobiSetZeroGuess(struct_precond);
         HYPRE_BiCGSTABSetPrecond( (HYPRE_Solver)struct_solver,
                                   (HYPRE_PtrToSolverFcn) HYPRE_StructJacobiSolve,
                                   (HYPRE_PtrToSolverFcn) HYPRE_StructJacobiSetup,
                                   (HYPRE_Solver)struct_precond);
      }

      else if (solver_id == 248)
      {
         /* use diagonal scaling as preconditioner */
         struct_precond = NULL;
         HYPRE_BiCGSTABSetPrecond( (HYPRE_Solver)struct_solver,
                                   (HYPRE_PtrToSolverFcn) HYPRE_StructDiagScale,
                                   (HYPRE_PtrToSolverFcn) HYPRE_StructDiagScaleSetup,
                                   (HYPRE_Solver)struct_precond);
      }

      HYPRE_BiCGSTABSetup
         ( (HYPRE_Solver)struct_solver, (HYPRE_Matrix)sA, (HYPRE_Vector)sb,
           (HYPRE_Vector)sx );

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      time_index = hypre_InitializeTiming("BiCGSTAB Solve");
      hypre_BeginTiming(time_index);

      HYPRE_BiCGSTABSolve
         ( (HYPRE_Solver)struct_solver, (HYPRE_Matrix)sA, (HYPRE_Vector)sb,
           (HYPRE_Vector)sx);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      HYPRE_BiCGSTABGetNumIterations( (HYPRE_Solver)struct_solver, &num_iterations);
      HYPRE_BiCGSTABGetFinalRelativeResidualNorm( (HYPRE_Solver)struct_solver, &final_res_norm);
      HYPRE_StructBiCGSTABDestroy(struct_solver);

      if (solver_id == 240)
      {
         HYPRE_StructSMGDestroy(struct_precond);
      }
      else if (solver_id == 241)
      {
         HYPRE_StructPFMGDestroy(struct_precond);
      }
      else if (solver_id == 247)
      {
         HYPRE_StructJacobiDestroy(struct_precond);
      }
   }

   /* begin lobpcg */

   /*-----------------------------------------------------------
    * Solve the eigenvalue problem using LOBPCG
    *-----------------------------------------------------------*/

   else if (lobpcgFlag)
   {
      if (solver_id < 10 || solver_id >= 20)
      {
         if (verbosity)
         {
            hypre_printf("\nLOBPCG works with solvers 10, 11, 13, 14 and 18 only\n");
         }
      }
      else
      {
         interpreter = hypre_CTAlloc(mv_InterfaceInterpreter,1, HYPRE_MEMORY_HOST);
         HYPRE_SStructSetupInterpreter(interpreter);
         HYPRE_SStructSetupMatvec(&matvec_fn);

         eigenvectors = mv_MultiVectorCreateFromSampleVector(interpreter,
                                                             blockSize,
                                                             x);
         eigenvalues = hypre_CTAlloc(HYPRE_Real, blockSize, HYPRE_MEMORY_HOST);
         if (seed)
         {
            mv_MultiVectorSetRandom(eigenvectors, seed);
         }
         else
         {
            mv_MultiVectorSetRandom(eigenvectors, (HYPRE_Int) time(0));
         }

         if (myid != 0)
         {
            verbosity = 0;
         }

         if (pcgIterations > 0)
         {
            time_index = hypre_InitializeTiming("PCG Setup");
            hypre_BeginTiming(time_index);

            HYPRE_SStructPCGCreate(comm, &solver);
            HYPRE_PCGSetMaxIter((HYPRE_Solver) solver, pcgIterations);
            HYPRE_PCGSetTol((HYPRE_Solver) solver, pcgTol);
            HYPRE_PCGSetTwoNorm((HYPRE_Solver) solver, 1);
            HYPRE_PCGSetRelChange((HYPRE_Solver) solver, rel_change);
            HYPRE_PCGSetPrintLevel((HYPRE_Solver) solver, krylov_print_level);
            HYPRE_PCGSetLogging((HYPRE_Solver) solver, 1);

            if ((solver_id == 10) || (solver_id == 11))
            {
               /* use Split solver as preconditioner */
               HYPRE_SStructSplitCreate(comm, &precond);
               HYPRE_SStructSplitSetMaxIter(precond, 1);
               HYPRE_SStructSplitSetTol(precond, 0.0);
               HYPRE_SStructSplitSetZeroGuess(precond);
               HYPRE_SStructSplitSetPrintLevel(precond, print_level);
               HYPRE_SStructSplitSetLogging(precond, 0);
               if (solver_id == 10)
               {
                  HYPRE_SStructSplitSetStructSolver(precond, HYPRE_SMG);
               }
               else if (solver_id == 11)
               {
                  HYPRE_SStructSplitSetStructSolver(precond, HYPRE_PFMG);
               }
               HYPRE_PCGSetPrecond( (HYPRE_Solver) solver,
                                    (HYPRE_PtrToSolverFcn) HYPRE_SStructSplitSolve,
                                    (HYPRE_PtrToSolverFcn) HYPRE_SStructSplitSetup,
                                    (HYPRE_Solver) precond);
            }
            else if (solver_id == 13)
            {
               /* use SysPFMG solver as preconditioner */
               HYPRE_SStructSysPFMGCreate(comm, &precond);
               HYPRE_SStructSysPFMGSetMaxIter(precond, 1);
               HYPRE_SStructSysPFMGSetTol(precond, 0.0);
               HYPRE_SStructSysPFMGSetZeroGuess(precond);
               /* weighted Jacobi = 1; red-black GS = 2 */
               HYPRE_SStructSysPFMGSetRelaxType(precond, relax[0]);
               HYPRE_SStructSysPFMGSetNumPreRelax(precond, n_pre);
               HYPRE_SStructSysPFMGSetNumPostRelax(precond, n_post);
               HYPRE_SStructSysPFMGSetSkipRelax(precond, skip);
               /*HYPRE_SStructSysPFMGSetDxyz(precond, dxyz);*/
               HYPRE_SStructSysPFMGSetPrintLevel(precond, print_level);
               HYPRE_SStructSysPFMGSetLogging(precond, 0);

               HYPRE_PCGSetPrecond( (HYPRE_Solver) solver,
                                    (HYPRE_PtrToSolverFcn) HYPRE_SStructSysPFMGSolve,
                                    (HYPRE_PtrToSolverFcn) HYPRE_SStructSysPFMGSetup,
                                    (HYPRE_Solver) precond);
            }
            else if (solver_id == 14)
            {
               /* use SSAMG solver as preconditioner */
               HYPRE_SStructSSAMGCreate(comm, &precond);
               HYPRE_SStructSSAMGSetMaxIter(precond, 1);
               HYPRE_SStructSSAMGSetMaxLevels(precond, max_levels);
               HYPRE_SStructSSAMGSetTol(precond, 0.0);
               HYPRE_SStructSSAMGSetZeroGuess(precond);
               HYPRE_SStructSSAMGSetSkipRelax(precond, skip);
               HYPRE_SStructSSAMGSetRelaxType(precond, relax[0]);
               HYPRE_SStructSSAMGSetNonGalerkinRAP(precond, rap);
               if (usr_jacobi_weight)
               {
                  HYPRE_SStructSSAMGSetRelaxWeight(precond, jacobi_weight);
               }
               HYPRE_SStructSSAMGSetNumPreRelax(precond, n_pre);
               HYPRE_SStructSSAMGSetNumPostRelax(precond, n_post);
               HYPRE_SStructSSAMGSetNumCoarseRelax(precond, n_coarse);
               HYPRE_SStructSSAMGSetMaxCoarseSize(precond, max_coarse_size);
               HYPRE_SStructSSAMGSetCoarseSolverType(precond, csolver_type);
               HYPRE_SStructSSAMGSetPrintLevel(precond, print_level);
               HYPRE_SStructSSAMGSetLogging(precond, 0);

               HYPRE_PCGSetPrecond( (HYPRE_Solver) solver,
                                    (HYPRE_PtrToSolverFcn) HYPRE_SStructSSAMGSolve,
                                    (HYPRE_PtrToSolverFcn) HYPRE_SStructSSAMGSetup,
                                    (HYPRE_Solver) precond);
            }
            else if (solver_id == 18)
            {
               /* use diagonal scaling as preconditioner */
               precond = NULL;
               HYPRE_PCGSetPrecond( (HYPRE_Solver) solver,
                                    (HYPRE_PtrToSolverFcn) HYPRE_SStructDiagScale,
                                    (HYPRE_PtrToSolverFcn) HYPRE_SStructDiagScaleSetup,
                                    (HYPRE_Solver) precond);
            }
            else if (solver_id != NO_SOLVER )
            {
               if (verbosity)
               {
                  hypre_printf("Solver ID not recognized. ");
                  hypre_printf("Running inner PCG iterations without preconditioner\n\n");
	       }
            }

            hypre_EndTiming(time_index);
            hypre_PrintTiming("Setup phase times", comm);
            hypre_FinalizeTiming(time_index);
            hypre_ClearTiming();

            HYPRE_LOBPCGCreate(interpreter, &matvec_fn, (HYPRE_Solver*)&lobpcg_solver);
            HYPRE_LOBPCGSetMaxIter((HYPRE_Solver)lobpcg_solver, max_iterations);
            HYPRE_LOBPCGSetPrecondUsageMode((HYPRE_Solver)lobpcg_solver, pcgMode);
            HYPRE_LOBPCGSetTol((HYPRE_Solver)lobpcg_solver, tol);
            HYPRE_LOBPCGSetPrintLevel((HYPRE_Solver)lobpcg_solver, verbosity);

            HYPRE_LOBPCGSetPrecond((HYPRE_Solver)lobpcg_solver,
                                   (HYPRE_PtrToSolverFcn) HYPRE_PCGSolve,
                                   (HYPRE_PtrToSolverFcn) HYPRE_PCGSetup,
                                   (HYPRE_Solver)solver);
            HYPRE_LOBPCGSetup((HYPRE_Solver)lobpcg_solver, (HYPRE_Matrix)A,
                              (HYPRE_Vector)b, (HYPRE_Vector)x);

            time_index = hypre_InitializeTiming("PCG Solve");
            hypre_BeginTiming(time_index);

            HYPRE_LOBPCGSolve((HYPRE_Solver)lobpcg_solver, constrains,
                              eigenvectors, eigenvalues );

            hypre_EndTiming(time_index);
            hypre_PrintTiming("Solve phase times", comm);
            hypre_FinalizeTiming(time_index);
            hypre_ClearTiming();

            HYPRE_SStructPCGDestroy(solver);

            if ((solver_id == 10) || (solver_id == 11))
	    {
               HYPRE_SStructSplitDestroy(precond);
            }
            else if (solver_id == 13)
            {
               HYPRE_SStructSysPFMGDestroy(precond);
            }
            else if (solver_id == 14)
            {
               HYPRE_SStructSSAMGDestroy(precond);
            }

            HYPRE_LOBPCGDestroy((HYPRE_Solver)lobpcg_solver);
         }
         else
         {
            time_index = hypre_InitializeTiming("LOBPCG Setup");
            hypre_BeginTiming(time_index);

            HYPRE_LOBPCGCreate(interpreter, &matvec_fn, (HYPRE_Solver*)&solver);
            HYPRE_LOBPCGSetMaxIter( (HYPRE_Solver) solver, max_iterations );
            HYPRE_LOBPCGSetTol( (HYPRE_Solver) solver, tol );
            HYPRE_LOBPCGSetPrintLevel( (HYPRE_Solver) solver, verbosity );

            if ((solver_id == 10) || (solver_id == 11))
            {
               /* use Split solver as preconditioner */
               HYPRE_SStructSplitCreate(comm, &precond);
               HYPRE_SStructSplitSetMaxIter(precond, 1);
               HYPRE_SStructSplitSetTol(precond, 0.0);
               HYPRE_SStructSplitSetZeroGuess(precond);
               HYPRE_SStructSplitSetPrintLevel(precond, print_level);
               HYPRE_SStructSplitSetLogging(precond, 0);
               if (solver_id == 10)
               {
                  HYPRE_SStructSplitSetStructSolver(precond, HYPRE_SMG);
               }
               else if (solver_id == 11)
               {
	          HYPRE_SStructSplitSetStructSolver(precond, HYPRE_PFMG);
               }
               HYPRE_LOBPCGSetPrecond((HYPRE_Solver) solver,
                                      (HYPRE_PtrToSolverFcn) HYPRE_SStructSplitSolve,
                                      (HYPRE_PtrToSolverFcn) HYPRE_SStructSplitSetup,
                                      (HYPRE_Solver) precond);
            }
            else if (solver_id == 13)
            {
               /* use SysPFMG solver as preconditioner */
               HYPRE_SStructSysPFMGCreate(comm, &precond);
               HYPRE_SStructSysPFMGSetMaxIter(precond, 1);
               HYPRE_SStructSysPFMGSetTol(precond, 0.0);
               HYPRE_SStructSysPFMGSetZeroGuess(precond);
               /* weighted Jacobi = 1; red-black GS = 2 */
               HYPRE_SStructSysPFMGSetRelaxType(precond, 1);
               HYPRE_SStructSysPFMGSetNumPreRelax(precond, n_pre);
               HYPRE_SStructSysPFMGSetNumPostRelax(precond, n_post);
               HYPRE_SStructSysPFMGSetSkipRelax(precond, skip);
               /*HYPRE_SStructSysPFMGSetDxyz(precond, dxyz);*/
               HYPRE_SStructSysPFMGSetPrintLevel(precond, print_level);
               HYPRE_SStructSysPFMGSetLogging(precond, 0);

               HYPRE_LOBPCGSetPrecond( (HYPRE_Solver) solver,
                                       (HYPRE_PtrToSolverFcn) HYPRE_SStructSysPFMGSolve,
                                       (HYPRE_PtrToSolverFcn) HYPRE_SStructSysPFMGSetup,
                                       (HYPRE_Solver) precond);
            }
            else if (solver_id == 14)
            {
               /* use SSAMG solver as preconditioner */
               HYPRE_SStructSSAMGCreate(comm, &precond);
               HYPRE_SStructSSAMGSetMaxIter(precond, 1);
               HYPRE_SStructSSAMGSetMaxLevels(precond, max_levels);
               HYPRE_SStructSSAMGSetTol(precond, 0.0);
               HYPRE_SStructSSAMGSetZeroGuess(precond);
               HYPRE_SStructSSAMGSetSkipRelax(precond, skip);
               HYPRE_SStructSSAMGSetRelaxType(precond, relax[0]);
               HYPRE_SStructSSAMGSetNonGalerkinRAP(precond, rap);
               if (usr_jacobi_weight)
               {
                  HYPRE_SStructSSAMGSetRelaxWeight(precond, jacobi_weight);
               }
               HYPRE_SStructSSAMGSetNumPreRelax(precond, n_pre);
               HYPRE_SStructSSAMGSetNumPostRelax(precond, n_post);
               HYPRE_SStructSSAMGSetNumCoarseRelax(precond, n_coarse);
               HYPRE_SStructSSAMGSetMaxCoarseSize(precond, max_coarse_size);
               HYPRE_SStructSSAMGSetCoarseSolverType(precond, csolver_type);
               HYPRE_SStructSSAMGSetPrintLevel(precond, print_level);
               HYPRE_SStructSSAMGSetLogging(precond, 0);

               HYPRE_LOBPCGSetPrecond((HYPRE_Solver) solver,
                                      (HYPRE_PtrToSolverFcn) HYPRE_SStructSSAMGSolve,
                                      (HYPRE_PtrToSolverFcn) HYPRE_SStructSSAMGSetup,
                                      (HYPRE_Solver) precond);
            }
            else if (solver_id == 18)
            {
               /* use diagonal scaling as preconditioner */
               precond = NULL;
               HYPRE_LOBPCGSetPrecond((HYPRE_Solver) solver,
                                      (HYPRE_PtrToSolverFcn) HYPRE_SStructDiagScale,
                                      (HYPRE_PtrToSolverFcn) HYPRE_SStructDiagScaleSetup,
                                      (HYPRE_Solver) precond);
            }
            else if (solver_id != NO_SOLVER )
            {
               if (verbosity)
               {
                  hypre_printf("Solver ID not recognized. ");
                  hypre_printf("Running inner PCG iterations without preconditioner\n\n");
               }
	    }

            HYPRE_LOBPCGSetup((HYPRE_Solver) solver, (HYPRE_Matrix) A,
                              (HYPRE_Vector) b, (HYPRE_Vector) x);

            hypre_EndTiming(time_index);
            hypre_PrintTiming("Setup phase times", comm);
            hypre_FinalizeTiming(time_index);
            hypre_ClearTiming();

            time_index = hypre_InitializeTiming("LOBPCG Solve");
            hypre_BeginTiming(time_index);

            HYPRE_LOBPCGSolve((HYPRE_Solver) solver, constrains,
                              eigenvectors, eigenvalues );

            hypre_EndTiming(time_index);
            hypre_PrintTiming("Solve phase times", comm);
            hypre_FinalizeTiming(time_index);
            hypre_ClearTiming();

            HYPRE_LOBPCGDestroy((HYPRE_Solver)solver);

            if ((solver_id == 10) || (solver_id == 11))
            {
               HYPRE_SStructSplitDestroy(precond);
            }
            else if (solver_id == 13)
            {
               HYPRE_SStructSysPFMGDestroy(precond);
            }
            else if (solver_id == 14)
            {
                HYPRE_SStructSSAMGDestroy(precond);
            }
         }

         if (checkOrtho)
         {
            gramXX = utilities_FortranMatrixCreate();
            identity = utilities_FortranMatrixCreate();

            utilities_FortranMatrixAllocateData(blockSize, blockSize, gramXX);
            utilities_FortranMatrixAllocateData(blockSize, blockSize, identity);

            lobpcg_MultiVectorByMultiVector(eigenvectors, eigenvectors, gramXX);
            utilities_FortranMatrixSetToIdentity(identity);
            utilities_FortranMatrixAdd(-1, identity, gramXX, gramXX);
            nonOrthF = utilities_FortranMatrixFNorm(gramXX);
            if (myid == 0)
            {
               hypre_printf("Non-orthonormality of eigenvectors: %12.5e\n", nonOrthF);
            }

            utilities_FortranMatrixDestroy(gramXX);
            utilities_FortranMatrixDestroy(identity);
         }

         if (print_level)
         {
            if (myid == 0)
            {
               if ((filePtr = fopen("values.txt", "w")))
               {
                  hypre_fprintf(filePtr, "%d\n", blockSize);
                  for (i = 0; i < blockSize; i++)
                  {
                     hypre_fprintf(filePtr, "%22.14e\n", eigenvalues[i]);
                  }
                  fclose(filePtr);
               }
            }

            if ((filePtr = fopen("residuals.txt", "w")))
            {
               residualNorms = HYPRE_LOBPCGResidualNorms((HYPRE_Solver)lobpcg_solver);
               residuals = utilities_FortranMatrixValues(residualNorms);
               hypre_fprintf(filePtr, "%d\n", blockSize);
               for (i = 0; i < blockSize; i++)
               {
                  hypre_fprintf(filePtr, "%22.14e\n", residuals[i]);
               }
               fclose(filePtr);
            }

            if (print_level > 1)
            {
               printBuffer = utilities_FortranMatrixCreate();
               iterations = HYPRE_LOBPCGIterations((HYPRE_Solver)lobpcg_solver);
               eigenvaluesHistory = HYPRE_LOBPCGEigenvaluesHistory((HYPRE_Solver)lobpcg_solver);
               utilities_FortranMatrixSelectBlock(eigenvaluesHistory,
                                                  1, blockSize, 1, iterations + 1,
                                                  printBuffer);
               utilities_FortranMatrixPrint(printBuffer, "val_hist.txt");

               residualNormsHistory = HYPRE_LOBPCGResidualNormsHistory((HYPRE_Solver)lobpcg_solver);
               utilities_FortranMatrixSelectBlock(residualNormsHistory,
                                                  1, blockSize, 1, iterations + 1,
                                                  printBuffer);
               utilities_FortranMatrixPrint(printBuffer, "res_hist.txt");
               utilities_FortranMatrixDestroy(printBuffer);
            }
         }

         mv_MultiVectorDestroy(eigenvectors);
         hypre_TFree(eigenvalues, HYPRE_MEMORY_HOST);
         hypre_TFree(interpreter, HYPRE_MEMORY_HOST);
      }
   }

   /* end lobpcg */

   /*-----------------------------------------------------------
    * Test matvec
    *-----------------------------------------------------------*/

   else if (solver_id < 0)
   {
      void  *matvec_data;

      hypre_SStructMatvecCreate(&matvec_data);
      hypre_SStructMatvecSetup(matvec_data, A, x);

      time_index = hypre_InitializeTiming("Matvec");
      hypre_BeginTiming(time_index);

      for (i = 0; i < reps; i++)
      {
         hypre_SStructMatvecCompute(matvec_data, -1.0, A, x, 1.0, b, r);
      }

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Total Matvec time", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      hypre_SStructMatvecDestroy(matvec_data);
   }

   /*-----------------------------------------------------------
    * Gather the solution vector
    *-----------------------------------------------------------*/

   HYPRE_SStructVectorGather(x);

   /*-----------------------------------------------------------
    * Compute real residual
    *-----------------------------------------------------------*/

   if (final_res || print_system)
   {
      HYPRE_SStructVectorCopy(b, r);
      HYPRE_SStructMatrixMatvec(-1.0, A, x, 1.0, r);
      HYPRE_SStructInnerProd(r, r, &real_res_norm);
      real_res_norm = sqrt(real_res_norm);
      if (rhs_norm > 0)
      {
         real_res_norm = real_res_norm/rhs_norm;
      }
   }

   /*-----------------------------------------------------------
    * Print the solution and other info
    *-----------------------------------------------------------*/

   if (print_system)
   {
      HYPRE_SStructVectorPrint("sstruct.out.x", x, 0);
      HYPRE_SStructVectorPrint("sstruct.out.r", r, 0);
#if 0
      FILE *file;
      char  filename[255];

      /* print out with shared data replicated */
      values = hypre_TAlloc(HYPRE_Real, data.max_boxsize, HYPRE_MEMORY_HOST);
      for (part = 0; part < data.nparts; part++)
      {
         pdata = data.pdata[part];
         for (var = 0; var < pdata.nvars; var++)
         {
            hypre_sprintf(filename, "sstruct.out.xx.%02d.%02d.%05d", part, var, myid);
            if ((file = fopen(filename, "w")) == NULL)
            {
               hypre_printf("Error: can't open output file %s\n", filename);
               exit(1);
            }
            for (box = 0; box < pdata.nboxes; box++)
            {
               GetVariableBox(pdata.ilowers[box], pdata.iuppers[box],
                              pdata.vartypes[var], ilower, iupper);
               HYPRE_SStructVectorGetBoxValues(x, part, ilower, iupper,
                                               var, values);
               hypre_fprintf(file, "\nBox %d:\n\n", box);
               size = 1;
               for (j = 0; j < data.ndim; j++)
               {
                  size*= (iupper[j] - ilower[j] + 1);
               }
               for (j = 0; j < size; j++)
               {
                  hypre_fprintf(file, "%.14e\n", values[j]);
               }
            }
            fflush(file);
            fclose(file);
         }
      }
      hypre_TFree(values, HYPRE_MEMORY_HOST);
#endif
   }

   if (myid == 0 /* begin lobpcg */ && !lobpcgFlag /* end lobpcg */)
   {
      hypre_printf("\n");
      hypre_printf("Iterations = %d\n", num_iterations);
      hypre_printf("Final Relative Residual Norm = %e\n", final_res_norm);
      if (final_res)
      {
         hypre_printf("Real Relative Residual Norm  = %e\n", real_res_norm);
      }
      hypre_printf("\n");
   }

   if (vis)
   {
      HYPRE_SStructGridPrintGLVis(grid, "sstruct.msh", NULL, NULL);
      HYPRE_SStructVectorPrintGLVis(b,  "sstruct.rhs");
      HYPRE_SStructVectorPrintGLVis(x,  "sstruct.sol");
   }

   /*-----------------------------------------------------------
    * Verify GetBoxValues()
    *-----------------------------------------------------------*/

#if 0
   {
      HYPRE_SStructVector   xnew;
      HYPRE_ParVector       par_xnew;
      HYPRE_StructVector    sxnew;
      HYPRE_Real            rnorm, bnorm;

      HYPRE_SStructVectorCreate(comm, grid, &xnew);
      HYPRE_SStructVectorSetObjectType(xnew, object_type);
      HYPRE_SStructVectorInitialize(xnew);

      /* get/set replicated shared data */
      values = hypre_TAlloc(HYPRE_Real,  data.max_boxsize, HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
      for (part = 0; part < data.nparts; part++)
      {
         pdata = data.pdata[part];
         for (var = 0; var < pdata.nvars; var++)
         {
            for (box = 0; box < pdata.nboxes; box++)
            {
               GetVariableBox(pdata.ilowers[box], pdata.iuppers[box],
                              pdata.vartypes[var], ilower, iupper);
               HYPRE_SStructVectorGetBoxValues(x, part, ilower, iupper,
                                               var, values);
               HYPRE_SStructVectorSetBoxValues(xnew, part, ilower, iupper,
                                               var, values);
            }
         }
      }
      hypre_TFree(values, HYPRE_MEMORY_HOST);

      HYPRE_SStructVectorAssemble(xnew);

      /* Compute residual norm - this if/else is due to a bug in SStructMatvec */
      if (object_type == HYPRE_SSTRUCT)
      {
         HYPRE_SStructInnerProd(b, b, &bnorm);
         hypre_SStructMatvec(-1.0, A, xnew, 1.0, b);
         HYPRE_SStructInnerProd(b, b, &rnorm);
      }
      else if (object_type == HYPRE_PARCSR)
      {
         bnorm = hypre_ParVectorInnerProd(par_b, par_b);
         HYPRE_SStructVectorGetObject(xnew, (void **) &par_xnew);
         HYPRE_ParCSRMatrixMatvec(-1.0, par_A, par_xnew, 1.0, par_b );
         rnorm = hypre_ParVectorInnerProd(par_b, par_b);
      }
      else if (object_type == HYPRE_STRUCT)
      {
         bnorm = hypre_StructInnerProd(sb, sb);
         HYPRE_SStructVectorGetObject(xnew, (void **) &sxnew);
         hypre_StructMatvec(-1.0, sA, sxnew, 1.0, sb);
         rnorm = hypre_StructInnerProd(sb, sb);
      }
      bnorm = sqrt(bnorm);
      rnorm = sqrt(rnorm);

      if (myid == 0)
      {
         hypre_printf("\n");
         hypre_printf("solver relnorm = %16.14e\n", final_res_norm);
         hypre_printf("check  relnorm = %16.14e, bnorm = %16.14e, rnorm = %16.14e\n",
                      (rnorm/bnorm), bnorm, rnorm);
         hypre_printf("\n");
      }

      HYPRE_SStructVectorDestroy(xnew);
   }
#endif

   /*-----------------------------------------------------------
    * Finalize things
    *-----------------------------------------------------------*/

   HYPRE_SStructGridDestroy(grid);
   for (s = 0; s < data.nstencils; s++)
   {
      HYPRE_SStructStencilDestroy(stencils[s]);
   }
   hypre_TFree(stencils, HYPRE_MEMORY_HOST);
   HYPRE_SStructGraphDestroy(graph);
   HYPRE_SStructMatrixDestroy(A);
   HYPRE_SStructVectorDestroy(b);
   HYPRE_SStructVectorDestroy(x);
   HYPRE_SStructVectorDestroy(r);
   if (gradient_matrix)
   {
      for (s = 0; s < data.ndim; s++)
      {
         HYPRE_SStructStencilDestroy(G_stencils[s]);
      }
      hypre_TFree(G_stencils, HYPRE_MEMORY_HOST);
      HYPRE_SStructGraphDestroy(G_graph);
      HYPRE_SStructGridDestroy(G_grid);
      HYPRE_SStructMatrixDestroy(G);
   }
   if ((print_system || check_symmetry) && (object_type == HYPRE_SSTRUCT))
   {
      HYPRE_IJMatrixDestroy(ij_A);
   }

   DestroyData(data);

   hypre_TFree(parts, HYPRE_MEMORY_HOST);
   hypre_TFree(refine, HYPRE_MEMORY_HOST);
   hypre_TFree(distribute, HYPRE_MEMORY_HOST);
   hypre_TFree(block, HYPRE_MEMORY_HOST);

   /* Finalize Hypre */
   HYPRE_Finalize();

   /* Finalize MPI */
   hypre_MPI_Finalize();

   return (0);
}
