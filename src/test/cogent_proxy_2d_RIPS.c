/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/*
   Example cogent_proxy_2d

   Verion 6 (2/6/2022)

   Compile with:   make cogent_proxy_2d

   To see options: cogent_proxy_2d -help

   Description:    This code solves a system corresponding to a discretization
                   of the system of equations

                   (M + N_1)u - N_2 v = 1
                   Mu         +     v = 0

                   on the unit square, where the matrices M, N_1 and N_2 are obtained from
                   the a second-order, centered-difference discretization of the operators

                   M(w)   = -a*w_xx
                   N_1(w) = -b*w_yy
                   N_2(w) = -c*w_yy

                   where a, b and c are positive constants.  The corresponding boundary
                   conditions are zero Dirichlet in x and periodic in y.

                   The domain is split into an Nx x 1 processor grid.
                   Each processor's piece of the grid has nx x ny cells
                   We use cell-centered variables, and, therefore, the
                   nodes are not shared. Note that we have two variables, u and
                   v, and need only one part to describe the domain.

*/

#include <math.h>
#include "_hypre_utilities.h"
#include "HYPRE_sstruct_ls.h"
#include "HYPRE_krylov.h"
#include "_hypre_parcsr_mv.h"
#include "_hypre_parcsr_ls.h"



int main (int argc, char *argv[])
{
   int i, j;

   int myid, num_procs;

   int nx, ny, Nx, pi, pj;
   HYPRE_Real hx, hy, hx2, hy2;
   int ilower[2], iupper[2];

   int solver_id;

   int object_type = HYPRE_PARCSR;

   HYPRE_SStructGrid     grid;
   HYPRE_SStructStencil  stencil_v;
   HYPRE_SStructStencil  stencil_u;
   HYPRE_SStructGraph    graph;
   HYPRE_SStructMatrix   A;
   HYPRE_SStructVector   b;
   HYPRE_SStructVector   x;

   HYPRE_Solver          par_solver;

   /* solver params */
   int max_iter = 10;
   double tol = 1.e-6;
   int print_level = 3;
   int print_cgrid = 0;

   /* ILU params */
   int ILU_type = 11;
   int ILU_fill_level = 0;
   double ILU_drop_threshold = -1.;
   double ILU_B_drop_threshold = -1.;
   double ILU_EF_drop_threshold = -1.;
   double ILU_Schur_drop_threshold = -1.;
   int ILU_max_nnz_per_row = 1000;
   int ILU_max_schur_iter = 5;

   /* MGR params */
   HYPRE_Int mgr_bsize = 2;
   HYPRE_Int mgr_nlevels = 1;
   HYPRE_Int mgr_non_c_to_f = 1;
   HYPRE_Int mgr_frelax_method = 1;
   HYPRE_Int *mgr_num_cindexes = NULL;
   HYPRE_Int **mgr_cindexes = NULL;
   HYPRE_Int mgr_relax_type = 0;
   HYPRE_Int mgr_num_relax_sweeps = 1;
   HYPRE_Int mgr_interp_type = 3;
   HYPRE_Int mgr_num_interp_sweeps = 1;
   HYPRE_Int mgr_gsmooth_type = 0;
   HYPRE_Int mgr_num_gsmooth_sweeps = 0;
   HYPRE_Int mgr_restrict_type = 0;
   HYPRE_Int mgr_num_restrict_sweeps = 0;
   HYPRE_Int mgr_cpoint = 0;
   HYPRE_Real mgr_csolve_threshold = 0.25;
   HYPRE_Real mgr_csolve_max_iter = 20;
   HYPRE_Int mgr_use_non_galerkin = 0;
   HYPRE_Int mgr_pmax = 0;

   HYPRE_Int strength_method = -1;

   HYPRE_Int semicoarsening_dir = 0;
   HYPRE_Int semicoarsening_offset = 0;

   /* Initialize MPI */
   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

   /* Set defaults */
   Nx = 1;
   nx = 8;
   ny = 8;
   solver_id = 0;
   HYPRE_Real a_fac = 1.e-5;
   HYPRE_Real b_fac = 2.e2;
   HYPRE_Real c_fac = 0.352;
   //   double R0 = 1.6;
   //   double axisym_fac = sqrt(2.*3.14159*R0);
   double axisym_fac = 1.;

   /* Parse command line */
   {
      int arg_index = 0;
      int print_usage = 0;

      while (arg_index < argc)
      {
         if ( strcmp(argv[arg_index], "-Nx") == 0 )
         {
            arg_index++;
            Nx = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-nx") == 0 )
         {
            arg_index++;
            nx = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-ny") == 0 )
         {
            arg_index++;
            ny = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-a") == 0 )
         {
            arg_index++;
            a_fac = atof(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-b") == 0 )
         {
            arg_index++;
            b_fac = atof(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-c") == 0 )
         {
            arg_index++;
            c_fac = atof(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-solver") == 0 )
         {
            arg_index++;
            solver_id = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-max_iter") == 0 )
         {
            arg_index++;
            max_iter = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-tol") == 0 )
         {
            arg_index++;
            tol = atof(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-print_level") == 0 )
         {
            arg_index++;
            print_level = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-print_cgrid") == 0 )
         {
            arg_index++;
            print_cgrid = atoi(argv[arg_index++]);
         }
         /* ILU options */
         else if ( strcmp(argv[arg_index], "-ilu_type") == 0 )
         {
            arg_index++;
            ILU_type = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-ilu_fill_level") == 0 )
         {
            arg_index++;
            ILU_fill_level = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-ilu_drop") == 0 )
         {
            arg_index++;
            ILU_drop_threshold = atof(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-ilu_B_drop") == 0 )
         {
            arg_index++;
            ILU_B_drop_threshold = atof(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-ilu_EF_drop") == 0 )
         {
            arg_index++;
            ILU_EF_drop_threshold = atof(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-ilu_schur_drop") == 0 )
         {
            arg_index++;
            ILU_Schur_drop_threshold = atof(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-ilu_max_nnz") == 0 )
         {
            arg_index++;
            ILU_max_nnz_per_row = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-ilu_max_schur_iter") == 0 )
         {
            arg_index++;
            ILU_max_schur_iter = atoi(argv[arg_index++]);
         }
         /* MGR options */
         else if ( strcmp(argv[arg_index], "-mgr_bsize") == 0 )
         {
            /* mgr block size */
            arg_index++;
            mgr_bsize = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-mgr_nlevels") == 0 )
         {
            /* mgr number of coarsening levels */
            arg_index++;
            mgr_nlevels = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-mgr_cpoint") == 0 )
         {
            /* coarse point index in block system */
            arg_index++;
            mgr_cpoint = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-mgr_non_c_to_f") == 0 )
         {
            /* mgr intermediate coarse grid strategy */
            arg_index++;
            mgr_non_c_to_f = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-mgr_frelax_method") == 0 )
         {
            /* mgr F-relaxation strategy: single/ multi level */
            arg_index++;
            mgr_frelax_method = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-mgr_relax_type") == 0 )
         {
            /* relax type for "single level" F-relaxation */
            arg_index++;
            mgr_relax_type = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-mgr_relax_sweeps") == 0 )
         {
            /* number of relaxation sweeps */
            arg_index++;
            mgr_num_relax_sweeps = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-mgr_interp_type") == 0 )
         {
            /* interpolation type */
            arg_index++;
            mgr_interp_type = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-mgr_interp_sweeps") == 0 )
         {
            /* number of interpolation sweeps*/
            arg_index++;
            mgr_num_interp_sweeps = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-mgr_gsmooth_type") == 0 )
         {
            /* global smoother type */
            arg_index++;
            mgr_gsmooth_type = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-mgr_gsmooth_sweeps") == 0 )
         {
            /* number of global smooth sweeps*/
            arg_index++;
            mgr_num_gsmooth_sweeps = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-mgr_restrict_type") == 0 )
         {
            /* restriction type */
            arg_index++;
            mgr_restrict_type = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-mgr_restrict_sweeps") == 0 )
         {
            /* number of restriction sweeps*/
            arg_index++;
            mgr_num_restrict_sweeps = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-mgr_csolve_threshold") == 0 )
         {
            /* number of restriction sweeps*/
            arg_index++;
            mgr_csolve_threshold = atof(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-mgr_csolve_max_iter") == 0 )
         {
            /* number of restriction sweeps*/
            arg_index++;
            mgr_csolve_max_iter = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-mgr_use_non_galerkin") == 0 )
         {
            /* use non Galerkin coarse grid */
            arg_index++;
            mgr_use_non_galerkin = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-mgr_pmax") == 0 )
         {
            /* Pmax elements for non Galerkin case */
            arg_index++;
            mgr_pmax = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-strength_method") == 0 )
         {
            /* Strength of connection method */
            arg_index++;
            strength_method = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-sdir") == 0 )
         {
            /* Semicoarsening direction */
            arg_index++;
            semicoarsening_dir = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-soffset") == 0 )
         {
            /* Semicoarsening offset */
            arg_index++;
            semicoarsening_offset = atoi(argv[arg_index++]);
         }
         /* end mgr options */
         else if ( strcmp(argv[arg_index], "-help") == 0 )
         {
            print_usage = 1;
            break;
         }
         else
         {
            arg_index++;
         }
      }

      if ((print_usage) && (myid == 0))
      {
         printf("\n");
         printf("Usage: %s [<options>]\n", argv[0]);
         printf("\n");
         printf("  -Nx <int>             : number of processors in x (default: 1)\n");
         printf("  -nx <int>             : grid size per processor in x (default: 8)\n");
         printf("  -ny <int>             : grid size per processor in y (default: 8)\n");
         printf("  -a <double>           : a factor (default: 1.e-5)\n");
         printf("  -b <double>           : b factor (default: 2.e2)\n");
         printf("  -solver <int>         : solver ID\n");
         printf("  -max_iter <int>       : max solver iterations (default 20) \n");
         printf("  -tol <double>         : solver convergence tolerance (default 1.e-7) \n");
         printf("  -print_level <int>    : print level (default 3) \n");
         printf("                          0  - hypre-ILU (default)\n");
         printf("                          1  - AMG \n");
         printf("                          2  - MGR \n");
         printf("  -ilu_type <int>        : ILU solver type (default 11) \n");
         printf("  -ilu_fill_level <int>  : ILU(k) fill level k (default 0) \n");
         printf("  -ilu_drop <double>     : ILU drop tolerance (default 1.e-2) \n");
         printf("  -ilu_B_drop <double>      : ILU B block drop tolerance (default 1.e-2) \n");
         printf("  -ilu_EF_drop <double>     : ILU E and F block drop tolerance (default 1.e-2) \n");
         printf("  -ilu_schur_drop <double>  : ILU Schur drop tolerance (default 1.e-2) \n");
         printf("  -ilu_max_nnz <int>        : ILU max number of nonzeros per row (default 1000) \n");
         printf("  -ilu_max_schur_iter <int> : ILU max Schur iteratations (default 5) \n");
         /* MGR options */
         hypre_printf("  -mgr_bsize   <val>               : set block size = val\n");
         hypre_printf("  -mgr_nlevels   <val>             : set number of coarsening levels = val\n");
         hypre_printf("                                     to be kept till the coarsest grid = val\n");
         hypre_printf("  -mgr_non_c_to_f   <val>          : set strategy for intermediate coarse grid \n");
         hypre_printf("  -mgr_non_c_to_f   0              : Allow some non Cpoints to be labeled \n");
         hypre_printf("                                     Cpoints on intermediate grid \n");
         hypre_printf("  -mgr_non_c_to_f   1              : set non Cpoints strictly to Fpoints \n");
         hypre_printf("  -mgr_frelax_method   <val>       : set F-relaxation strategy \n");
         hypre_printf("  -mgr_frelax_method   0           : Use 'single-level smoother' strategy \n");
         hypre_printf("                                     for F-relaxation \n");
         hypre_printf("  -mgr_frelax_method   1           : Use a 'multi-level smoother' strategy \n");
         hypre_printf("                                     for F-relaxation \n");
         hypre_printf("  -mgr_csolve_threshold <val>      : AMG coarse solve strong threshold = val\n");
         hypre_printf("  -mgr_csolve_max_iter <val>       : AMG coarse solve max iterations = val\n");
         hypre_printf("  -strength_method <val>       : method for creating AMG strength matrix= val\n");
         /* end MGR options */
         printf("\n");
      }

      if (print_usage)
      {
         MPI_Finalize();
         return (0);
      }
   }

   /* Set up the Nx x 1 processor grid.  pi, pj indicate
      the position of the executing process in the processor grid.
      The local problem size nx x ny. */

   pj = 0;
   pi = myid;

   /* Figure out the extents of each processor's piece of the grid. */
   ilower[0] = pi * nx;
   ilower[1] = pj * ny;

   iupper[0] = ilower[0] + nx - 1;
   iupper[1] = ilower[1] + ny - 1;

   /* Get the mesh size */

   hx  = 0.3 * axisym_fac / (Nx * nx);
   hy  = 3.77 * axisym_fac / ny;
   hx2 = hx * hx;
   hy2 = hy * hy;

   /* 1. Set up a grid - we have one part and two variables */
   {
      int nparts = 1;
      int part = 0;
      int ndim = 2;
      int periodic[2];

      /* Create an empty 2D grid object */
      HYPRE_SStructGridCreate(MPI_COMM_WORLD, ndim, nparts, &grid);

      /* Add a new box to the grid */
      HYPRE_SStructGridSetExtents(grid, part, ilower, iupper);

      /* Set the variable type and number of variables on each part.*/
      {
         int i;
         int nvars = 2;
         HYPRE_SStructVariable vartypes[2] = {HYPRE_SSTRUCT_VARIABLE_CELL,
                                              HYPRE_SSTRUCT_VARIABLE_CELL
                                             };

         for (i = 0; i < nparts; i++)
         {
            HYPRE_SStructGridSetVariables(grid, i, nvars, vartypes);
         }
      }

      /* Make the second direction periodic */
      periodic[0] = 0;
      periodic[1] = ny;
      HYPRE_SStructGridSetPeriodic(grid, part, periodic);

      /* This is a collective call finalizing the grid assembly.
         The grid is now ``ready to be used'' */
      HYPRE_SStructGridAssemble(grid);
   }

   /* 2. Define the discretization stencils */
   {
      int entry;
      int stencil_size;
      int var;
      int ndim = 2;

      int M_offsets[5][2] = {{0, -1}, {-1, 0}, {0, 0}, {1, 0}, {0, 1}};
      int N_offsets[3][2] = {{0, -1}, {0, 0}, {0, 1}};
      int I_offsets[1][2] = {{0, 0}};

      /* Stencil object for variable u (labeled as variable 0) */
      {
         int stencil_size_uu = 5;
         int stencil_size_uv = 3;
         stencil_size = stencil_size_uu + stencil_size_uv;

         HYPRE_SStructStencilCreate(ndim, stencil_size, &stencil_u);

         /* The first stencil_size_uu entries are for the u-u connections */
         var = 0; /* connect to variable 0 */
         for (entry = 0; entry < stencil_size_uu; entry++)
         {
            HYPRE_SStructStencilSetEntry(stencil_u, entry, M_offsets[entry], var);
         }

         /* The last stencil_size_uv entries are for the u-v connections */
         var = 1;  /* connect to variable 1 */
         for (entry = stencil_size_uu, j = 0; entry < stencil_size; entry++, j++)
         {
            HYPRE_SStructStencilSetEntry(stencil_u, entry, N_offsets[j], var);
         }
      }

      /* Stencil object for variable v  (variable 1) */
      {
         int stencil_size_vu = 5;
         int stencil_size_vv = 1;
         stencil_size = stencil_size_vu + stencil_size_vv;

         HYPRE_SStructStencilCreate(ndim, stencil_size, &stencil_v);

         /* These are the v-u connections */
         var = 0; /* Connect to variable 0 */
         for (entry = 0; entry < stencil_size_vu; entry++)
         {
            HYPRE_SStructStencilSetEntry(stencil_v, entry, M_offsets[entry], var);
         }

         /* These are the v-v connections */
         var = 1; /* Connect to variable 1 */
         for (entry = stencil_size_vu, j = 0; entry < stencil_size; entry++, j++)
         {
            HYPRE_SStructStencilSetEntry(stencil_v, entry, I_offsets[j], var);
         }
      }
   }

   /* 3. Set up the Graph  - this determines the non-zero structure
      of the matrix and allows non-stencil relationships between the parts. */
   {
      int var;
      int part = 0;

      /* Create the graph object */
      HYPRE_SStructGraphCreate(MPI_COMM_WORLD, grid, &graph);
      HYPRE_SStructGraphSetObjectType(graph, object_type);

      /* Assign the u-stencil we created to variable u (variable 0) */
      var = 0;
      HYPRE_SStructGraphSetStencil(graph, part, var, stencil_u);

      /* Assign the v-stencil we created to variable v (variable 1) */
      var = 1;
      HYPRE_SStructGraphSetStencil(graph, part, var, stencil_v);

      /* Assemble the graph */
      HYPRE_SStructGraphAssemble(graph);
   }

   /* 4. Set up the SStruct Matrix */
   {
      int nentries;
      int nvalues;
      int var;
      int part = 0;
      int num_local_cells = nx * ny;
      double M_stencil[5];
      double N1_stencil[3];
      double N2_stencil[3];

      M_stencil[0]  = 0.;
      M_stencil[1]  = -a_fac * ( 1. / hx2  );
      M_stencil[2]  = -a_fac * (-2. / hx2  );
      M_stencil[3]  = -a_fac * ( 1. / hx2  );
      M_stencil[4]  = 0.;

      N1_stencil[0] =   -b_fac / hy2;
      N1_stencil[1] = 2.*b_fac / hy2;
      N1_stencil[2] =   -b_fac / hy2;

      N2_stencil[0] =   -c_fac / hy2;
      N2_stencil[1] = 2.*c_fac / hy2;
      N2_stencil[2] =   -c_fac / hy2;

      /* Create an empty matrix object */
      HYPRE_SStructMatrixCreate(MPI_COMM_WORLD, graph, &A);

      /* Set the object type (by default HYPRE_SSTRUCT). This determines the
         data structure used to store the matrix.  If you want to use
         unstructured solvers, e.g. BoomerAMG, the object type should be
         HYPRE_PARCSR. If the problem is purely structured (with one part), you
         may want to use HYPRE_STRUCT to access the structured solvers.  */
      HYPRE_SStructMatrixSetObjectType(A, object_type);

      /* Indicate that the matrix coefficients are ready to be set */
      HYPRE_SStructMatrixInitialize(A);

      /* Each processor must set the stencil values for their boxes on each part.
         In this example, we only set stencil entries and therefore use
         HYPRE_SStructMatrixSetBoxValues.  If we need to set non-stencil entries,
         we have to use HYPRE_SStructMatrixSetValues. */

      /* First set the u-stencil entries.  Note that
         HYPRE_SStructMatrixSetBoxValues can only set values corresponding
         to stencil entries for the same variable. Therefore, we must set the
         entries for each variable within a stencil with separate function calls.
         For example, below the u-u connections and u-v connections are handled
         in separate calls.  */
      {
         int     i, j;
         HYPRE_Real *u_values;
         int     u_u_indices[5] = {0, 1, 2, 3, 4};

         var = 0; /* Set values for the u connections */

         /*  First the u-u connections */
         nentries = 5;
         nvalues = nentries * num_local_cells;
         u_values = (HYPRE_Real*) calloc(nvalues, sizeof(HYPRE_Real));

         for (i = 0; i < nvalues; i += nentries)
         {
            for (j = 0; j < nentries; ++j)
            {
               u_values[i + j] = 0.; //M_stencil[j];
            }

            u_values[i] += 0.;//N1_stencil[0];
            u_values[i + 2] += 0.; //N1_stencil[1];
            u_values[i + 4] += 0.; //N1_stencil[2];
         }

         HYPRE_SStructMatrixAddToBoxValues(A, part, ilower, iupper,
                                           var, nentries,
                                           u_u_indices, u_values);
         free(u_values);

         /* Next the u-v connections */
         int u_v_indices[3] = {5, 6, 7};
         nentries = 3;
         nvalues = nentries * num_local_cells;
         u_values = (HYPRE_Real*) calloc(nvalues, sizeof(HYPRE_Real));

         for (i = 0; i < nvalues; i += nentries)
         {
            for (j = 0; j < nentries; ++j)
            {
               u_values[i + j] = -N2_stencil[j];
            }
         }

         HYPRE_SStructMatrixAddToBoxValues(A, part, ilower, iupper,
                                           var, nentries,
                                           u_v_indices, u_values);
         free(u_values);
      }
      /*  Now set the v-stencil entries */
      {
         int     i, j;
         HYPRE_Real *v_values;
         int  v_u_indices[5] = {0, 1, 2, 3, 4};

         var = 1; /* the v connections */

         /* the v-u connections */
         nentries = 5;
         nvalues = nentries * num_local_cells;
         v_values = (HYPRE_Real*) calloc(nvalues, sizeof(HYPRE_Real));

         for (i = 0; i < nvalues; i += nentries)
         {
            for (j = 0; j < nentries; ++j)
            {
               v_values[i + j] = M_stencil[j];
            }
         }

         HYPRE_SStructMatrixAddToBoxValues(A, part, ilower, iupper,
                                           var, nentries,
                                           v_u_indices, v_values);

         free(v_values);

         /* the v-v connections */
         int v_v_indices[1] = {5};
         nentries = 1;
         nvalues = nentries * num_local_cells;
         v_values = (HYPRE_Real*) calloc(nvalues, sizeof(HYPRE_Real));

         for (i = 0; i < nvalues; i += nentries)
         {
            v_values[i] = 1.;
         }

         HYPRE_SStructMatrixAddToBoxValues(A, part, ilower, iupper,
                                           var, nentries,
                                           v_v_indices, v_values);

         free(v_values);
      }
   }

   /* 5. Incorporate the zero Dirichlet boundary condition in the first coordinate:
         at those boundaries, set the stencil entries that reach across it to zero.*/
   {
      int bc_ilower[2];
      int bc_iupper[2];
      int nentries = 1;
      int var;
      int stencil_indices[1];

      int part = 0;

      /* Recall: pi, pj describe position in the processor grid */
      if (pi == 0)
      {
         int nvalues  = nentries * ny; /*  number of stencil entries times the length
                                         of one side of my grid box */

         HYPRE_Real* values = (HYPRE_Real*) calloc(nvalues, sizeof(HYPRE_Real));
         for (j = 0; j < nvalues; j++)
         {
            values[j] = 0.0;
         }

         /* Bottom plane of grid points */
         bc_ilower[0] = pi * nx;
         bc_ilower[1] = pj * ny;

         bc_iupper[0] = bc_ilower[0];
         bc_iupper[1] = bc_ilower[1] + ny - 1;

         stencil_indices[0] = 1;

         var = 0;
         HYPRE_SStructMatrixSetBoxValues(A, part, bc_ilower, bc_iupper,
                                         var, nentries,
                                         stencil_indices, values);

         var = 1;
         HYPRE_SStructMatrixSetBoxValues(A, part, bc_ilower, bc_iupper,
                                         var, nentries,
                                         stencil_indices, values);
         free(values);
      }

      if (pi == Nx - 1)
      {
         int nvalues  = nentries * ny; /*  number of stencil entries times the length
                                         of one side of my grid box */

         HYPRE_Real* values = (HYPRE_Real*) calloc(nvalues, sizeof(HYPRE_Real));
         for (j = 0; j < nvalues; j++)
         {
            values[j] = 0.0;
         }

         /* upper plane of grid points */
         bc_ilower[0] = pi * nx + nx - 1;
         bc_ilower[1] = pj * ny;

         bc_iupper[0] = bc_ilower[0];
         bc_iupper[1] = bc_ilower[1] + ny - 1;

         stencil_indices[0] = 3;

         var = 0;
         HYPRE_SStructMatrixSetBoxValues(A, part, bc_ilower, bc_iupper,
                                         var, nentries,
                                         stencil_indices, values);

         var = 1;
         HYPRE_SStructMatrixSetBoxValues(A, part, bc_ilower, bc_iupper,
                                         var, nentries,
                                         stencil_indices, values);
         free(values);
      }

      /* Done with the boundary modifications, since the other direction
         is periodic
      */

   }

   /* This is a collective call finalizing the matrix assembly.
      The matrix is now ``ready to be used'' */
   HYPRE_SStructMatrixAssemble(A);


   HYPRE_SStructMatrixPrint("A", A, 0);

   /* 5. Set up SStruct Vectors for b and x */
   {
      int    nvalues = nx * ny;
      HYPRE_Real *values;
      int part = 0;
      int var;

      values = (HYPRE_Real*) calloc(nvalues, sizeof(HYPRE_Real));

      /* Create an empty vector object */
      HYPRE_SStructVectorCreate(MPI_COMM_WORLD, grid, &b);
      HYPRE_SStructVectorCreate(MPI_COMM_WORLD, grid, &x);

      /* Set the object type for the vectors
         to be the same as was already set for the matrix */
      HYPRE_SStructVectorSetObjectType(b, object_type);
      HYPRE_SStructVectorSetObjectType(x, object_type);

      /* Indicate that the vector coefficients are ready to be set */
      HYPRE_SStructVectorInitialize(b);
      HYPRE_SStructVectorInitialize(x);

      /* Set the values for b */
      /* project out kernel for a consistent rhs (for 4th-order problem) */
      /* This is hardcoded to use semicoarsening in y */
      HYPRE_Real gamma = (double)1. / ny;
      HYPRE_Real *pvalues = (HYPRE_Real*) calloc(nvalues, sizeof(HYPRE_Real));
      for (i = 0; i < nvalues; i ++)
      {
         values[i] = 1.0;   //(double)rand() / (double)RAND_MAX; //1.0;
      }
      memcpy(pvalues, values, nvalues * sizeof(HYPRE_Real));
      for (HYPRE_Int j = 0; j < ny; j++)
      {
         for (i = 0; i < nx; i++)
         {
            HYPRE_Int idx = i + j * nx;
            for (HYPRE_Int k = 0; k < ny; k++)
            {
               HYPRE_Int pos = idx + k * semicoarsening_offset;
               HYPRE_Real val = pos >= nvalues ? pvalues[pos % nvalues] : pvalues[pos];
               values[idx] -= gamma * val;
            }
         }
      }
      free(pvalues);
      for (i = 0; i < nvalues; i++)
      {
         printf("values[%d] = %f, gamma = %f \n", i, values[i], gamma);
      }

      var = 0;
      HYPRE_SStructVectorSetBoxValues(b, part, ilower, iupper, var, values);

      for (i = 0; i < nvalues; i ++)
      {
         values[i] = 0.0;
      }
      var = 1;
      HYPRE_SStructVectorSetBoxValues(b, part, ilower, iupper, var, values);

      /* Set the values for the initial guess */
      //      for (i = 0; i < nvalues; i ++)
      //         values[i] = (double)rand() / (double)RAND_MAX;
      var = 0;
      HYPRE_SStructVectorSetBoxValues(x, part, ilower, iupper, var, values);

      var = 1;
      HYPRE_SStructVectorSetBoxValues(x, part, ilower, iupper, var, values);

      free(values);

      /* This is a collective call finalizing the vector assembly.
         The vector is now ``ready to be used'' */
      HYPRE_SStructVectorAssemble(b);
      HYPRE_SStructVectorAssemble(x);
   }

   /* 6. Set up and use a solver
      (Solver options can be found in the Reference Manual.) */
   {
      HYPRE_Real final_res_norm;
      int its;

      /* Get the object for the matrix and vectors. */
      HYPRE_ParCSRMatrix    par_A;
      HYPRE_ParVector       par_b;
      HYPRE_ParVector       par_x;
      HYPRE_SStructMatrixGetObject(A, (void **) &par_A);
      HYPRE_SStructVectorGetObject(b, (void **) &par_b);
      HYPRE_SStructVectorGetObject(x, (void **) &par_x);

      if (solver_id == 0) /* ILU */
      {
         HYPRE_ILUCreate(&par_solver);

         /*
         ILU_type options:
         0: Block-Jacobi with ILU(k)
         1: Block-Jacobi with ILUT
         10: GMRES with ILU(k)
         11: GMRES with ILUT
         20: Newton Schulz Hotelling with ILU(k)
         21: Newton Schulz Hotelling with ILUT
         30: Restricted Additive Schwarz with ILU(k)
         31: Restricted Additive Schwarz with ILUT
         40: DDPQ-GMRES with ILU(k)
         41: DDPQ-GMRES with ILUT
         50: GMRES with RAP-ILU(0) using MILU(0) for P
         */

         HYPRE_ILUSetType(par_solver, ILU_type);

         if (ILU_fill_level >= 0)
         {
            HYPRE_ILUSetLevelOfFill(par_solver, ILU_fill_level);
         }

         if (ILU_drop_threshold >= 0.)
         {
            HYPRE_ILUSetDropThreshold(par_solver, ILU_drop_threshold);
         }

         if (ILU_B_drop_threshold >= 0. &&
             ILU_EF_drop_threshold >= 0. &&
             ILU_Schur_drop_threshold >= 0. )
         {
            double ILU_drop_threshold_array[3] = {ILU_B_drop_threshold,
                                                  ILU_EF_drop_threshold,
                                                  ILU_Schur_drop_threshold
                                                 };
            HYPRE_ILUSetDropThresholdArray(par_solver, ILU_drop_threshold_array);
         }

         HYPRE_ILUSetMaxNnzPerRow(par_solver, ILU_max_nnz_per_row);
         HYPRE_ILUSetMaxIter(par_solver, max_iter);
         HYPRE_ILUSetSchurMaxIter(par_solver, ILU_max_schur_iter);
         HYPRE_ILUSetTol(par_solver, tol);
         HYPRE_ILUSetPrintLevel(par_solver, print_level);

         /* do the setup */
         HYPRE_ILUSetup(par_solver, par_A, par_b, par_x);

         /* do the solve */
         HYPRE_ILUSolve(par_solver, par_A, par_b, par_x);

         /* get some info */
         HYPRE_ILUGetNumIterations(par_solver, &its);
         HYPRE_ILUGetFinalRelativeResidualNorm(par_solver, &final_res_norm);

         /* clean up */
         HYPRE_ILUDestroy(par_solver);
      }
      else if (solver_id == 1) /* AMG */
      {
         HYPRE_BoomerAMGCreate(&par_solver);
         HYPRE_BoomerAMGSetCoarsenType(par_solver, 6);
         HYPRE_BoomerAMGSetOldDefault(par_solver);
         HYPRE_BoomerAMGSetStrongThreshold(par_solver, 0.25);
         HYPRE_BoomerAMGSetTol(par_solver, tol);
         HYPRE_BoomerAMGSetPrintLevel(par_solver, 3);
         HYPRE_BoomerAMGSetPrintFileName(par_solver, "polcor.out.log");
         HYPRE_BoomerAMGSetMaxIter(par_solver, max_iter);
         //        HYPRE_BoomerAMGSetUseAuxStrengthMatrix(par_solver, 1);
         /* do the setup */
         HYPRE_BoomerAMGSetup(par_solver, par_A, par_b, par_x);

         /* do the solve */
         HYPRE_BoomerAMGSolve(par_solver, par_A, par_b, par_x);

         /* get some info */
         HYPRE_BoomerAMGGetNumIterations(par_solver, &its);
         HYPRE_BoomerAMGGetFinalRelativeResidualNorm(par_solver,
                                                     &final_res_norm);
         /* clean up */
         HYPRE_BoomerAMGDestroy(par_solver);
      }
      else if (solver_id == 2) /* MGR */
      {
         HYPRE_MGRCreate(&par_solver);

         mgr_num_cindexes = hypre_CTAlloc(HYPRE_Int,  mgr_nlevels, HYPRE_MEMORY_HOST);
         for (i = 0; i < mgr_nlevels; i++)
         {
            /* assume 1 coarse index per level */
            mgr_num_cindexes[i] = 1;
         }
         mgr_cindexes = hypre_CTAlloc(HYPRE_Int*,  mgr_nlevels, HYPRE_MEMORY_HOST);
         for (i = 0; i < mgr_nlevels; i++)
         {
            mgr_cindexes[i] = hypre_CTAlloc(HYPRE_Int,  mgr_num_cindexes[i], HYPRE_MEMORY_HOST);
         }
         for (i = 0; i < mgr_nlevels; i++)
         {
            /* assume coarse point is at index 0 */
            mgr_cindexes[i][0] = mgr_cpoint;
         }

         /* set MGR data by block */
#if 0
         {
            HYPRE_BigInt rowstart = hypre_ParCSRMatrixFirstRowIndex(par_A);
            HYPRE_BigInt rowend = hypre_ParCSRMatrixLastRowIndex(par_A);
            HYPRE_Int fsize = (rowend - rowstart + 1) / 2 ;
            HYPRE_BigInt next_block = rowstart + fsize;
            //        hypre_printf("%d: row_start = %d, next_block = %d, diff = %d \n",myid, rowstart, next_block,(rowend - rowstart + 1) );

            HYPRE_BigInt idx_array[2] = {rowstart, next_block};
            HYPRE_MGRSetCpointsByContiguousBlock( par_solver, mgr_bsize, mgr_nlevels, idx_array,
                                                  mgr_num_cindexes, mgr_cindexes);
         }
#else
         {
            int var, n;
            HYPRE_Int* CF_index = (HYPRE_Int*) calloc(nx * ny * 2, sizeof(HYPRE_Int));
            HYPRE_Int* ptr = CF_index;
            for (var = 0; var < 2; ++var)
            {
               for (n = 0; n < nx * ny; ++n) { *ptr++ = var; }
            }
            HYPRE_MGRSetCpointsByPointMarkerArray(par_solver, mgr_bsize, mgr_nlevels, mgr_num_cindexes,
                                                  mgr_cindexes, CF_index);
         }
#endif

         //         HYPRE_MGRSetBlockJacobiBlockSize(par_solver, 3);

         /* set intermediate coarse grid strategy */
         HYPRE_MGRSetNonCpointsToFpoints(par_solver, mgr_non_c_to_f);
         /* set F relaxation strategy */
         HYPRE_MGRSetFRelaxMethod(par_solver, mgr_frelax_method);
         /* set relax type for single level F-relaxation and post-relaxation */
         HYPRE_MGRSetRelaxType(par_solver, mgr_relax_type);
         HYPRE_MGRSetNumRelaxSweeps(par_solver, mgr_num_relax_sweeps);
         /* set interpolation type */
         HYPRE_MGRSetRestrictType(par_solver, mgr_restrict_type);
         HYPRE_MGRSetNumRestrictSweeps(par_solver, mgr_num_restrict_sweeps);
         HYPRE_MGRSetInterpType(par_solver, mgr_interp_type);
         HYPRE_MGRSetNumInterpSweeps(par_solver, mgr_num_interp_sweeps);
         /* set print level */
         HYPRE_MGRSetPrintLevel(par_solver, print_level);
         /* set max iterations */
         HYPRE_MGRSetMaxIter(par_solver, max_iter);
         HYPRE_MGRSetTol(par_solver, tol);

         HYPRE_MGRSetGlobalSmoothType(par_solver, mgr_gsmooth_type);
         HYPRE_MGRSetMaxGlobalSmoothIters( par_solver, mgr_num_gsmooth_sweeps );

         HYPRE_MGRSetCoarseGridMethod(par_solver, &mgr_use_non_galerkin);
         HYPRE_MGRSetPMaxElmts(par_solver, mgr_pmax);  // no truncation
         //HYPRE_Int num_functions = 1;
         //HYPRE_MGRSetLevelFRelaxNumFunctions(par_solver, &num_functions);

         /* Create the coarse grid solver */

         HYPRE_Solver coarse_solver;
         HYPRE_BoomerAMGCreate(&coarse_solver);
         HYPRE_BoomerAMGSetCGCIts(coarse_solver, 1);
         HYPRE_BoomerAMGSetInterpType(coarse_solver, 0);
         HYPRE_BoomerAMGSetPostInterpType(coarse_solver, 0);
         HYPRE_BoomerAMGSetCoarsenType(coarse_solver, 6);
         HYPRE_BoomerAMGSetPMaxElmts(coarse_solver, 0);  // no truncation
         HYPRE_BoomerAMGSetCycleType(coarse_solver, 1);
         HYPRE_BoomerAMGSetFCycle(coarse_solver, 0);
         HYPRE_BoomerAMGSetNumSweeps(coarse_solver, 1);
         HYPRE_BoomerAMGSetRelaxType(coarse_solver, 3);
         HYPRE_BoomerAMGSetRelaxOrder(coarse_solver, 1);
         HYPRE_BoomerAMGSetMaxLevels(coarse_solver, 25);
         HYPRE_BoomerAMGSetStrongThreshold(coarse_solver, mgr_csolve_threshold);
         HYPRE_BoomerAMGSetMaxIter(coarse_solver, mgr_csolve_max_iter);
         HYPRE_BoomerAMGSetTol(coarse_solver, 1.e-12 );
         HYPRE_BoomerAMGSetPrintLevel(coarse_solver, 3);
         //        HYPRE_BoomerAMGSetUseAuxStrengthMatrix(coarse_solver, strength_method);
         //        HYPRE_BoomerAMGSetMinCoarseSize(coarse_solver, 100);
         //        HYPRE_BoomerAMGSetMaxCoarseSize(coarse_solver, 500);
         /* set the MGR coarse solver. Comment out to use default Coarse Grid solver in MGR */
         HYPRE_MGRSetCoarseSolver( par_solver, HYPRE_BoomerAMGSolve, HYPRE_BoomerAMGSetup, coarse_solver);

         hypre_MGRSetFrelaxPrintLevel(par_solver, 3);

         /* setup MGR solver */
         HYPRE_MGRSetup(par_solver, par_A, par_b, par_x);

         if (print_cgrid)
         {
            HYPRE_Int* cgrid = (HYPRE_Int*) calloc(nx * ny, sizeof(HYPRE_Int));
            HYPRE_BoomerAMGGetGridHierarchy (coarse_solver, cgrid);

            FILE* fd = fopen ("cgrid", "w");

            fprintf(fd, "%d %d\n", nx, ny);
            for (i = 0; i < nx * ny; ++i)
            {
               fprintf(fd, "%d\n", cgrid[i]);
            }

            fclose(fd);
            free(cgrid);
         }
         /* MGR solve */
         HYPRE_MGRSolve(par_solver, par_A, par_b, par_x);

         HYPRE_MGRGetNumIterations(par_solver, &its);
         HYPRE_MGRGetFinalRelativeResidualNorm(par_solver, &final_res_norm);

         /* free memory */
         if (mgr_num_cindexes)
         {
            hypre_TFree(mgr_num_cindexes, HYPRE_MEMORY_HOST);
         }
         mgr_num_cindexes = NULL;

         if (mgr_cindexes)
         {
            for ( i = 0; i < mgr_nlevels; i++)
            {
               if (mgr_cindexes[i])
               {
                  hypre_TFree(mgr_cindexes[i], HYPRE_MEMORY_HOST);
               }
            }
            hypre_TFree(mgr_cindexes, HYPRE_MEMORY_HOST);
            mgr_cindexes = NULL;
         }

         HYPRE_BoomerAMGDestroy(coarse_solver);
         HYPRE_MGRDestroy(par_solver);
      }
      else if (solver_id == 3) /* semicoarsening MGR (sMGR) */
      {

         HYPRE_MGRCreate(&par_solver);

         mgr_num_cindexes = hypre_CTAlloc(HYPRE_Int,  mgr_nlevels, HYPRE_MEMORY_HOST);
         for (i = 0; i < mgr_nlevels; i++)
         {
            /* assume 1 coarse index per level */
            mgr_num_cindexes[i] = 1;
         }
         mgr_cindexes = hypre_CTAlloc(HYPRE_Int*,  mgr_nlevels, HYPRE_MEMORY_HOST);
         for (i = 0; i < mgr_nlevels; i++)
         {
            mgr_cindexes[i] = hypre_CTAlloc(HYPRE_Int,  mgr_num_cindexes[i], HYPRE_MEMORY_HOST);
         }
         for (i = 0; i < mgr_nlevels; i++)
         {
            /* assume coarse point is at index 0 */
            mgr_cindexes[i][0] = mgr_cpoint;
         }

         /* set MGR data by marker array */
         int var, n;
         HYPRE_Int* CF_index = (HYPRE_Int*) calloc(nx * ny * 2, sizeof(HYPRE_Int));
         HYPRE_Int* ptr = CF_index;
         for (var = 0; var < 2; ++var)
         {
            for (n = 0; n < nx * ny; ++n) { *ptr++ = var; }
         }
         HYPRE_MGRSetCpointsByPointMarkerArray(par_solver, mgr_bsize, mgr_nlevels, mgr_num_cindexes,
                                               mgr_cindexes, CF_index);

         /* set intermediate coarse grid strategy */
         HYPRE_MGRSetNonCpointsToFpoints(par_solver, mgr_non_c_to_f);
         /* set F relaxation strategy */
         HYPRE_MGRSetFRelaxMethod(par_solver, 0);
         /* set relax type for single level F-relaxation and post-relaxation */
         HYPRE_MGRSetRelaxType(par_solver, 0);
         HYPRE_MGRSetNumRelaxSweeps(par_solver, 1);
         /* set interpolation type */
         HYPRE_MGRSetRestrictType(par_solver, 0);
         HYPRE_MGRSetInterpType(par_solver, 4);
         HYPRE_MGRSetNumInterpSweeps(par_solver, mgr_num_interp_sweeps);
         /* set print level */
         HYPRE_MGRSetPrintLevel(par_solver, print_level);
         /* set max iterations */
         HYPRE_MGRSetMaxIter(par_solver, 1);
         HYPRE_MGRSetTol(par_solver, 0.);

         hypre_MGRPrintCoarseSystem(par_solver, 1);


         /* Create the inner MGR coarse grid solver */
         HYPRE_Solver mgr_coarse_solver;
         HYPRE_MGRCreate(&mgr_coarse_solver);
         HYPRE_MGRSetPrintLevel(mgr_coarse_solver, print_level);
         hypre_MGRSetFrelaxPrintLevel(mgr_coarse_solver, 3);
#if 1
         /* set coarsening data for inner mgr solver */
         {
            HYPRE_Int smgr_nblocks = semicoarsening_dir == 0 ? nx : ny;
            HYPRE_Int smgr_nlevels = ceil(log2(smgr_nblocks));
            printf("smgr_nblocks = %d, smgr_nlevels = %d \n", smgr_nblocks, smgr_nlevels);

            /* Assume C-labels come first. So C F C F C F C ... */
            HYPRE_Int *lvl_num_cpts = (int*)calloc(smgr_nlevels, sizeof(int));
            HYPRE_Int **lvl_cpts = (int**)malloc(smgr_nlevels * sizeof(int*));
            HYPRE_Int num_c_pts = smgr_nblocks;
            HYPRE_Int skip = 1;
            for (i = 0; i < smgr_nlevels; i++)
            {
               num_c_pts = (num_c_pts + 1) / 2; // ensure we have ceiling of num_c_pts / 2
               lvl_cpts[i] = (int*)calloc(num_c_pts, sizeof(int));
               lvl_num_cpts[i] = num_c_pts;
               printf("lvl_num_cpts[%d] = %d \n", i, num_c_pts);
               skip *= 2;
               int jc = 0;
               int counter = 0;
               /* populate coarse point labels for each level */
               while (counter < smgr_nblocks)
               {
                  lvl_cpts[i][jc] = counter;
                  counter += skip;
                  jc++;
               }
            }
            /* now generate marker labels for domain points */
            HYPRE_Int* smgr_CF_index = (HYPRE_Int*) calloc(nx * ny, sizeof(HYPRE_Int));
            ptr = smgr_CF_index;
            HYPRE_Int ctr = 0;
            for (j = 0; j < ny; j++)
            {
               for (i = 0; i < nx; i++)
               {
                  *ptr++ = semicoarsening_dir == 0 ? i : j;
                  //   printf("smgr_CF_index[%d] = %d \n", i+j*nx, smgr_CF_index[ctr++]);
               }
            }

            /*
            for(i=0; i<smgr_nlevels; i++)
            {
               printf("level_cpts[%d]: ", i);
               for(j=0; j<lvl_num_cpts[i]; j++)
               {
                  printf(" %d ", lvl_cpts[i][j]);
               }
               printf("\n");
            }
            */

            /* set inner MGR options */
            HYPRE_MGRSetCpointsByPointMarkerArray(mgr_coarse_solver, smgr_nblocks, smgr_nlevels, lvl_num_cpts,
                                                  lvl_cpts, smgr_CF_index);

            /* set intermediate coarse grid strategy */
            HYPRE_MGRSetNonCpointsToFpoints(mgr_coarse_solver, mgr_non_c_to_f);
            /* set F relaxation strategy */
            HYPRE_MGRSetFRelaxMethod(mgr_coarse_solver, 0);
            /* set relax type for single level F-relaxation and post-relaxation */
            HYPRE_MGRSetRelaxType(mgr_coarse_solver, mgr_relax_type);
            HYPRE_MGRSetNumRelaxSweeps(mgr_coarse_solver, mgr_num_relax_sweeps);
            /* set restriction type */
            HYPRE_MGRSetRestrictType(mgr_coarse_solver, mgr_restrict_type);
            /* set interpolation type */
            HYPRE_Int *mgr_level_interp_type = (int*)calloc(smgr_nlevels, sizeof(int));
            for (i = 0; i < smgr_nlevels; i++)
            {
               mgr_level_interp_type[i] = 8;
            }
            HYPRE_MGRSetLevelInterpType(mgr_coarse_solver, mgr_level_interp_type);
            //HYPRE_MGRSetInterpType(mgr_coarse_solver, mgr_interp_type);

            HYPRE_MGRSetTwoPointInterpOffsets(mgr_coarse_solver, (-semicoarsening_offset),
                                              semicoarsening_offset);

            /* set print level */
            HYPRE_MGRSetPrintLevel(mgr_coarse_solver, print_level);
            /* set max iterations */
            HYPRE_MGRSetMaxIter(mgr_coarse_solver, max_iter);
            HYPRE_MGRSetTol(mgr_coarse_solver, tol);

            hypre_ParCSRMatrixPrintIJ(par_A, 1, 1, "IJ.out.A");
            hypre_ParVectorPrintIJ(par_b, 1, "IJ.rhs");
            hypre_ParVectorPrintIJ(par_x, 1, "IJ.sol");
            hypre_MGRPrintCoarseSystem(mgr_coarse_solver, 1);

            /* Free memory */
            if (lvl_num_cpts) { free(lvl_num_cpts); }
            if (lvl_cpts) { free(lvl_cpts); }
            if (mgr_level_interp_type) { free(mgr_level_interp_type); }
         }
#endif
         /* Create the coarse grid solver (for inner MGR solver)*/
         HYPRE_Solver coarse_solver;
         HYPRE_BoomerAMGCreate(&coarse_solver);
         HYPRE_BoomerAMGSetCGCIts(coarse_solver, 1);
         HYPRE_BoomerAMGSetInterpType(coarse_solver, 0);
         HYPRE_BoomerAMGSetPostInterpType(coarse_solver, 0);
         HYPRE_BoomerAMGSetCoarsenType(coarse_solver, 6);
         HYPRE_BoomerAMGSetPMaxElmts(coarse_solver, 0);  // no truncation
         HYPRE_BoomerAMGSetCycleType(coarse_solver, 1);
         HYPRE_BoomerAMGSetFCycle(coarse_solver, 0);
         HYPRE_BoomerAMGSetNumSweeps(coarse_solver, 1);
         HYPRE_BoomerAMGSetRelaxType(coarse_solver, 3);
         HYPRE_BoomerAMGSetRelaxOrder(coarse_solver, 1);
         HYPRE_BoomerAMGSetMaxLevels(coarse_solver, 25);
         HYPRE_BoomerAMGSetStrongThreshold(coarse_solver, mgr_csolve_threshold);
         HYPRE_BoomerAMGSetMaxIter(coarse_solver, mgr_csolve_max_iter);
         HYPRE_BoomerAMGSetTol(coarse_solver, 0.);
         HYPRE_BoomerAMGSetPrintLevel(coarse_solver, 3);
         //        HYPRE_BoomerAMGSetUseAuxStrengthMatrix(coarse_solver, strength_method);
         //        HYPRE_BoomerAMGSetMinCoarseSize(coarse_solver, 100);
         //        HYPRE_BoomerAMGSetMaxCoarseSize(coarse_solver, 500);
         /* set the inner MGR coarse solver. Comment out to use default Coarse Grid solver in MGR */
         HYPRE_MGRSetCoarseSolver( mgr_coarse_solver, HYPRE_BoomerAMGSolve, HYPRE_BoomerAMGSetup,
                                   coarse_solver);

         /* set the outer MGR coarse solver. Comment out to use default Coarse Grid solver in MGR */
         HYPRE_MGRSetCoarseSolver( par_solver, HYPRE_MGRSolve, HYPRE_MGRSetup, mgr_coarse_solver);
         /* setup MGR solver */
         HYPRE_MGRSetup(par_solver, par_A, par_b, par_x);

         if (print_cgrid)
         {
            HYPRE_Int* cgrid = (HYPRE_Int*) calloc(nx * ny, sizeof(HYPRE_Int));
            HYPRE_MGRGetGridHierarchy (mgr_coarse_solver, cgrid);

            FILE* fd = fopen ("mgrcgrid", "w");

            fprintf(fd, "%d %d\n", nx, ny);
            for (i = 0; i < nx * ny; ++i)
            {
               fprintf(fd, "%d\n", cgrid[i]);
            }

            fclose(fd);
            free(cgrid);
         }
         /* MGR solve */
         HYPRE_MGRSolve(par_solver, par_A, par_b, par_x);

         HYPRE_MGRGetNumIterations(par_solver, &its);
         HYPRE_MGRGetFinalRelativeResidualNorm(par_solver, &final_res_norm);

         /* free memory */
         if (mgr_num_cindexes)
         {
            hypre_TFree(mgr_num_cindexes, HYPRE_MEMORY_HOST);
         }
         mgr_num_cindexes = NULL;

         if (mgr_cindexes)
         {
            for ( i = 0; i < mgr_nlevels; i++)
            {
               if (mgr_cindexes[i])
               {
                  hypre_TFree(mgr_cindexes[i], HYPRE_MEMORY_HOST);
               }
            }
            hypre_TFree(mgr_cindexes, HYPRE_MEMORY_HOST);
            mgr_cindexes = NULL;
         }

         HYPRE_BoomerAMGDestroy(coarse_solver);
         HYPRE_MGRDestroy(mgr_coarse_solver);
         HYPRE_MGRDestroy(par_solver);
      }
      else
      {
         if (myid == 0) { printf("\n ERROR: Invalid solver id specified.\n"); }
      }

      if (myid == 0)
      {
         printf("\n");
         printf("Iterations = %d\n", its);
         printf("Final Relative Residual Norm = %g\n", (double)final_res_norm);
         printf("\n");
      }
   }

#if 0
   /* Write out the solution */
   if ( num_procs == 1 )
   {
      int i, j, m;
      double* x_data = (HYPRE_Real*) calloc(nx * ny, sizeof(HYPRE_Real));

      HYPRE_SStructVectorGather(x);

      HYPRE_SStructVectorGetBoxValues(x, 0, ilower, iupper, 0, x_data);

      FILE* fd = fopen ("y", "w");

      m = 0;
      for (j = 0; j < ny; ++j)
      {
         for (i = 0; i < nx; ++i)
         {
            fprintf(fd, "%20.12e ", x_data[m++]);
         }
         fprintf(fd, "\n");
      }

      fclose(fd);

      free(x_data);
   }
#endif

   /* Free memory */
   HYPRE_SStructVectorDestroy(b);
   HYPRE_SStructVectorDestroy(x);
   HYPRE_SStructMatrixDestroy(A);
   HYPRE_SStructGraphDestroy(graph);
   HYPRE_SStructStencilDestroy(stencil_v);
   HYPRE_SStructStencilDestroy(stencil_u);
   HYPRE_SStructGridDestroy(grid);

   /* Finalize MPI */
   MPI_Finalize();

   return (0);
}
