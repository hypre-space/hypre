/*--------------------------------------------------------------------------
 * Test driver for the StructIJ version of the structured grid interface.  
 * Here, the structgrid interface is implemented on top of the IJ interface,
 * which in turn constructs a parcsr matrix under the hood.
 *--------------------------------------------------------------------------*/
 
/*------------------------------------------------------------------------
 * The problem is the standard 7-point laplacian in 3D with grid and 
 * anisotropy determined as command line arguments, as in the
 * struct_linear_solvers test driver.  1D and 2D problems are possible
 * too, but these have not been tested.  Do `driver -help' for usage info.
 *------------------------------------------------------------------------*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "utilities.h"
#include "HYPRE_structIJ_mv.h"
#include "HYPRE_parcsr_ls.h"

int
main( int   argc,
      char *argv[] )
{
   int                  arg_index;
   int                  print_usage;
   int                  nx, ny, nz;
   int                  P, Q, R;
   int                  bx, by, bz;
   double               cx, cy, cz;
   int                  solver_id;
   int                  ierr;

   int                  p, q, r;
   int                  dim;
   int                  nblocks, volume;

   int                  istart[3];
   int                **iupper;
   int                **ilower;

   int                **offsets;

   HYPRE_StructGrid     grid;
   HYPRE_StructStencil  stencil;

   int                 *stencil_indices;
   double              *values;

   int                  i, s;
   int                  ix, iy, iz, ib;

   int                  ioutdat;
   int                  debug_flag;
   int                  max_levels = 25;
   int                  num_iterations; 
   double               final_res_norm;

   HYPRE_StructIJMatrix A_structij;
   HYPRE_StructIJVector b_structij;
   HYPRE_StructIJVector x_structij;
   HYPRE_ParCSRMatrix   A_parcsr;
   HYPRE_ParVector      b_parcsr;
   HYPRE_ParVector      x_parcsr;

   HYPRE_Solver         amg_solver;
   HYPRE_Solver         pcg_solver;
   HYPRE_Solver         pcg_precond;

   int                  num_procs, myid;
   int                 *partitioning;
   int                 *part_b;
   int                 *part_x;
   int                 *row_sizes;
   int                 *diag_sizes;
   int                 *offdiag_sizes;

   int		        time_index;

   /* parameters for BoomerAMG */
   double   strong_threshold;
   double   trunc_factor;
   int      cycle_type;
   int      coarsen_type = 0;
   int      hybrid = 1;
   int      measure_type = 0;
   int     *num_grid_sweeps;  
   int     *grid_relax_type;   
   int    **grid_relax_points;
   int      relax_default;
   double  *relax_weight; 
   double   tol = 1.0e-6;

   /* parameters for PILUT */
   double   drop_tol = -1;
   int      nonzeros_to_keep = -1;

   /* parameters for GMRES */
   int	    k_dim;

   MPI_Init(&argc, &argv);

   MPI_Comm_size(MPI_COMM_WORLD, &num_procs );
   MPI_Comm_rank(MPI_COMM_WORLD, &myid );

   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/

   dim = 3;

   nx = 10;
   ny = 10;
   nz = 10;

   P  = num_procs;
   Q  = 1;
   R  = 1;

   bx = 1;
   by = 1;
   bz = 1;

   cx = 1.0;
   cy = 1.0;
   cz = 1.0;

   solver_id = 0;

   istart[0] = -17;
   istart[1] = 0;
   istart[2] = 32;
 
   relax_default = 3;
   debug_flag = 0;

   ioutdat = 3;

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/
 
   print_usage = 0;
   arg_index = 1;

   while ( (arg_index < argc) && (!print_usage) )
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
      else if ( strcmp(argv[arg_index], "-b") == 0 )
      {
         arg_index++;
         bx = atoi(argv[arg_index++]);
         by = atoi(argv[arg_index++]);
         bz = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-c") == 0 )
      {
         arg_index++;
         cx = atof(argv[arg_index++]);
         cy = atof(argv[arg_index++]);
         cz = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-d") == 0 )
      {
         arg_index++;
         dim = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-solver") == 0 )
      {
         arg_index++;
         solver_id = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-ruge") == 0 )
      {
         arg_index++;
         coarsen_type      = 1;
      }    
      else if ( strcmp(argv[arg_index], "-ruge2b") == 0 )
      {
         arg_index++;
         coarsen_type      = 2;
      }    
      else if ( strcmp(argv[arg_index], "-ruge3") == 0 )
      {
         arg_index++;
         coarsen_type      = 3;
      }    
      else if ( strcmp(argv[arg_index], "-ruge3c") == 0 )
      {
         arg_index++;
         coarsen_type      = 4;
      }    
      else if ( strcmp(argv[arg_index], "-rugerlx") == 0 )
      {
         arg_index++;
         coarsen_type      = 5;
      }    
      else if ( strcmp(argv[arg_index], "-falgout") == 0 )
      {
         arg_index++;
         coarsen_type      = 6;
      }    
      else if ( strcmp(argv[arg_index], "-nohybrid") == 0 )
      {
         arg_index++;
         hybrid      = -1;
      }    
      else if ( strcmp(argv[arg_index], "-gm") == 0 )
      {
         arg_index++;
         measure_type      = 1;
      }    
      else if ( strcmp(argv[arg_index], "-rlx") == 0 )
      {
         arg_index++;
         relax_default = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-dbg") == 0 )
      {
         arg_index++;
         debug_flag = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-help") == 0 )
      {
         print_usage = 1;
      }
      else
      {
         arg_index++;
      }
   }

   /* for CGNR preconditioned with Boomeramg, only relaxation scheme 2 is
      implemented, i.e. Jacobi relaxation with Matvec */
   if (solver_id == 5) relax_default = 2;

   /* defaults for BoomerAMG */
   if (solver_id == 0 || solver_id == 1 || solver_id == 3 || solver_id == 5)
   {
   strong_threshold = 0.25;
   trunc_factor = 0.0;
   cycle_type = 1;

   num_grid_sweeps   = hypre_CTAlloc(int,4);
   grid_relax_type   = hypre_CTAlloc(int,4);
   grid_relax_points = hypre_CTAlloc(int *,4);
   relax_weight      = hypre_CTAlloc(double, max_levels);

   for (i=0; i < max_levels; i++)
	relax_weight[i] = 0.0;

   if (coarsen_type == 5)
   {
      /* fine grid */
      num_grid_sweeps[0] = 3;
      grid_relax_type[0] = relax_default; 
      grid_relax_points[0] = hypre_CTAlloc(int, 4); 
      grid_relax_points[0][0] = -2;
      grid_relax_points[0][1] = -1;
      grid_relax_points[0][2] = 1;
   
      /* down cycle */
      num_grid_sweeps[1] = 4;
      grid_relax_type[1] = relax_default; 
      grid_relax_points[1] = hypre_CTAlloc(int, 4); 
      grid_relax_points[1][0] = -1;
      grid_relax_points[1][1] = 1;
      grid_relax_points[1][2] = -2;
      grid_relax_points[1][3] = -2;
   
      /* up cycle */
      num_grid_sweeps[2] = 4;
      grid_relax_type[2] = relax_default; 
      grid_relax_points[2] = hypre_CTAlloc(int, 4); 
      grid_relax_points[2][0] = -2;
      grid_relax_points[2][1] = -2;
      grid_relax_points[2][2] = 1;
      grid_relax_points[2][3] = -1;
   }
   else
   {   
      /* fine grid */
      num_grid_sweeps[0] = 2;
      grid_relax_type[0] = relax_default; 
      grid_relax_points[0] = hypre_CTAlloc(int, 2); 
      grid_relax_points[0][0] = 1;
      grid_relax_points[0][1] = -1;
  
      /* down cycle */
      num_grid_sweeps[1] = 2;
      grid_relax_type[1] = relax_default; 
      grid_relax_points[1] = hypre_CTAlloc(int, 2); 
      grid_relax_points[1][0] = 1;
      grid_relax_points[1][1] = -1;
   
      /* up cycle */
      num_grid_sweeps[2] = 2;
      grid_relax_type[2] = relax_default; 
      grid_relax_points[2] = hypre_CTAlloc(int, 2); 
      grid_relax_points[2][0] = -1;
      grid_relax_points[2][1] = 1;
   }
   /* coarsest grid */
   num_grid_sweeps[3] = 1;
   grid_relax_type[3] = 9;
   grid_relax_points[3] = hypre_CTAlloc(int, 1);
   grid_relax_points[3][0] = 0;
   }
   /* defaults for GMRES */

   k_dim = 5;

   arg_index = 0;
   while (arg_index < argc)
   {
      if ( strcmp(argv[arg_index], "-k") == 0 )
      {
         arg_index++;
         k_dim = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-w") == 0 )
      {
         arg_index++;
        if (solver_id == 0 || solver_id == 1 || solver_id == 3 
		|| solver_id == 5 )
        {
         relax_weight[0] = atof(argv[arg_index++]);
         for (i=1; i < max_levels; i++)
	   relax_weight[i] = relax_weight[0];
        }
      }
      else if ( strcmp(argv[arg_index], "-th") == 0 )
      {
         arg_index++;
         strong_threshold  = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-tol") == 0 )
      {
         arg_index++;
         tol  = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-drop_tol") == 0 )
      {
         arg_index++;
         drop_tol  = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-nonzeros_to_keep") == 0 )
      {
         arg_index++;
         nonzeros_to_keep  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-tr") == 0 )
      {
         arg_index++;
         trunc_factor  = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-iout") == 0 )
      {
         arg_index++;
         ioutdat  = atoi(argv[arg_index++]);
      }
      else
      {
         arg_index++;
      }
   }

   /*-----------------------------------------------------------
    * Print usage info
    *-----------------------------------------------------------*/
 
   if ( (print_usage) && (myid == 0) )
   {
      printf("\n");
      printf("Usage: %s [<options>]\n", argv[0]);
      printf("\n");
      printf("  -n <nx> <ny> <nz>    : problem size per block\n");
      printf("  -b <bx> <by> <bz>    : blocking per processor\n");
      printf("  -P <Px> <Py> <Pz>    : processor topology\n");
      printf("  -c <cx> <cy> <cz>    : diffusion coefficients\n");
      printf("\n");
      printf("  -rhsrand             : rhs is random vector\n");
      printf("  -xisone              : solution of all ones\n");
      printf("\n");
      printf("  -solver <ID>         : solver ID\n");
      printf("       0=AMG         1=AMG-PCG       \n");
      printf("       2=DS-PCG      3=AMG-GMRES     \n");
      printf("       4=DS-GMRES    5=AMG-CGNR      \n");     
      printf("       6=DS-CGNR     7=PILUT-GMRES   \n");     
      printf("       8=ParaSails-PCG \n");     
      printf("\n");
      printf("  -ruge                : Ruge coarsening (local)\n");
      printf("  -ruge3               : third pass on boundary\n");
      printf("  -ruge3c              : third pass on boundary, keep c-points\n");
      printf("  -ruge2b              : 2nd pass is global\n");
      printf("  -rugerlx             : relaxes special points\n");
      printf("  -falgout             : local ruge followed by LJP\n");
      printf("  -nohybrid            : no switch in coarsening\n");
      printf("  -gm                  : use global measures\n");
      printf("\n");
      printf("  -rlx <val>           : relaxation type\n");
      printf("       0=Weighted Jacobi  \n");
      printf("       1=Gauss-Seidel (very slow!)  \n");
      printf("       3=Hybrid Jacobi/Gauss-Seidel  \n");
      printf("\n");  
      printf("  -th <val>            : set AMG threshold Theta = val \n");
      printf("  -tr <val>            : set AMG interpolation truncation factor = val \n");
      printf("  -tol <val>           : set AMG convergence tolerance to val\n");
      printf("  -w  <val>            : set Jacobi relax weight = val\n");
      printf("  -k  <val>            : dimension Krylov space for GMRES\n");
      printf("\n");  
      printf("  -drop_tol  <val>     : set threshold for dropping in PILUT\n");
      printf("  -nonzeros_to_keep <val>: number of nonzeros in each row to keep\n");
      printf("\n");  
      printf("  -iout <val>          : set output flag\n");
      printf("       0=no output    1=matrix stats\n"); 
      printf("       2=cycle stats  3=matrix & cycle stats\n"); 
      printf("\n");  
      printf("  -dbg <val>           : set debug flag\n");
      printf("       0=no debugging\n       1=internal timing\n       2=interpolation truncation\n       3=more detailed timing in coarsening routine\n");
      exit(1);
   }

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
      printf("Running with these driver parameters:\n");
      printf("  (nx, ny, nz)    = (%d, %d, %d)\n", nx, ny, nz);
      printf("  (Px, Py, Pz)    = (%d, %d, %d)\n", P,  Q,  R);
      printf("  (bx, by, bz)    = (%d, %d, %d)\n", bx, by, bz);
      printf("  (cx, cy, cz)    = (%f, %f, %f)\n", cx, cy, cz);
      printf("  dim             = %d\n", dim);
      printf("  solver ID       = %d\n", solver_id);
   }

   /*-----------------------------------------------------------
    * Set up the grid structure
    *-----------------------------------------------------------*/

   switch (dim)
   {
      case 1:
         volume  = nx;
         nblocks = bx;
         stencil_indices = hypre_CTAlloc(int, 2);
         offsets = hypre_CTAlloc(int*, 2);
         offsets[0] = hypre_CTAlloc(int, 1);
         offsets[0][0] = -1; 
         offsets[1] = hypre_CTAlloc(int, 1);
         offsets[1][0] = 0; 
         /* compute p from P and myid */
         p = myid % P;
         break;
      case 2:
         volume  = nx*ny;
         nblocks = bx*by;
         stencil_indices = hypre_CTAlloc(int, 3);
         offsets = hypre_CTAlloc(int*, 3);
         offsets[0] = hypre_CTAlloc(int, 2);
         offsets[0][0] = -1; 
         offsets[0][1] = 0; 
         offsets[1] = hypre_CTAlloc(int, 2);
         offsets[1][0] = 0; 
         offsets[1][1] = -1; 
         offsets[2] = hypre_CTAlloc(int, 2);
         offsets[2][0] = 0; 
         offsets[2][1] = 0; 
         /* compute p,q from P,Q and myid */
         p = myid % P;
         q = (( myid - p)/P) % Q;
         break;
      case 3:
         volume  = nx*ny*nz;
         nblocks = bx*by*bz;
         stencil_indices = hypre_CTAlloc(int, 7);
         offsets = hypre_CTAlloc(int*, 7);
         offsets[0] = hypre_CTAlloc(int, 3);
         offsets[0][0] = -1; 
         offsets[0][1] = 0; 
         offsets[0][2] = 0; 
         offsets[1] = hypre_CTAlloc(int, 3);
         offsets[1][0] = 0; 
         offsets[1][1] = -1; 
         offsets[1][2] = 0; 
         offsets[2] = hypre_CTAlloc(int, 3);
         offsets[2][0] = 0; 
         offsets[2][1] = 0; 
         offsets[2][2] = -1; 
         offsets[3] = hypre_CTAlloc(int, 3);
         offsets[3][0] = 0; 
         offsets[3][1] = 0; 
         offsets[3][2] = 0; 
         offsets[4] = hypre_CTAlloc(int, 3);
         offsets[4][0] = 1; 
         offsets[4][1] = 0; 
         offsets[4][2] = 0; 
         offsets[5] = hypre_CTAlloc(int, 3);
         offsets[5][0] = 0; 
         offsets[5][1] = 1; 
         offsets[5][2] = 0; 
         offsets[6] = hypre_CTAlloc(int, 3);
         offsets[6][0] = 0; 
         offsets[6][1] = 0; 
         offsets[6][2] = 1; 
         /* compute p,q,r from P,Q,R and myid */
         p = myid % P;
         q = (( myid - p)/P) % Q;
         r = ( myid - p - P*q)/( P*Q );
         break;
   }

   ilower = hypre_CTAlloc(int*, nblocks);
   iupper = hypre_CTAlloc(int*, nblocks);
   for (i = 0; i < nblocks; i++)
   {
      ilower[i] = hypre_CTAlloc(int, dim);
      iupper[i] = hypre_CTAlloc(int, dim);
   }

   /* compute ilower and iupper from (p,q,r), (bx,by,bz), and (nx,ny,nz) */
   ib = 0;
   switch (dim)
   {
      case 1:
         for (ix = 0; ix < bx; ix++)
         {
            ilower[ib][0] = istart[0]+ nx*(bx*p+ix);
            iupper[ib][0] = istart[0]+ nx*(bx*p+ix+1) - 1;
            ib++;
         }
         break;
      case 2:
         for (iy = 0; iy < by; iy++)
            for (ix = 0; ix < bx; ix++)
            {
               ilower[ib][0] = istart[0]+ nx*(bx*p+ix);
               iupper[ib][0] = istart[0]+ nx*(bx*p+ix+1) - 1;
               ilower[ib][1] = istart[1]+ ny*(by*q+iy);
               iupper[ib][1] = istart[1]+ ny*(by*q+iy+1) - 1;
               ib++;
            }
         break;
      case 3:
         for (iz = 0; iz < bz; iz++)
            for (iy = 0; iy < by; iy++)
               for (ix = 0; ix < bx; ix++)
               {
                  ilower[ib][0] = istart[0]+ nx*(bx*p+ix);
                  iupper[ib][0] = istart[0]+ nx*(bx*p+ix+1) - 1;
                  ilower[ib][1] = istart[1]+ ny*(by*q+iy);
                  iupper[ib][1] = istart[1]+ ny*(by*q+iy+1) - 1;
                  ilower[ib][2] = istart[2]+ nz*(bz*r+iz);
                  iupper[ib][2] = istart[2]+ nz*(bz*r+iz+1) - 1;
                  ib++;
               }
         break;
   } 

   HYPRE_StructGridCreate(MPI_COMM_WORLD, dim, &grid);
   for (ib = 0; ib < nblocks; ib++)
   {
      HYPRE_StructGridSetExtents(grid, ilower[ib], iupper[ib]);
   }
   HYPRE_StructGridAssemble(grid);

   /*-----------------------------------------------------------
    * Set up the stencil structure
    *-----------------------------------------------------------*/
 
   HYPRE_StructStencilCreate(dim, 7, &stencil);
   for (s = 0; s < 7; s++)
   {
      HYPRE_StructStencilSetElement(stencil, s, offsets[s]);
   }

   /*-----------------------------------------------------------
    * Set up the matrix structure
    *-----------------------------------------------------------*/
 
   ierr  = HYPRE_StructIJMatrixCreate(MPI_COMM_WORLD, grid, stencil, &A_structij);
   ierr += HYPRE_StructIJMatrixSetSymmetric(A_structij, 1);  
   ierr += HYPRE_StructIJMatrixInitialize(A_structij);

   /*-----------------------------------------------------------
    * Fill in the matrix elements
    *-----------------------------------------------------------*/

   values = hypre_CTAlloc(double, 7*volume);

   /* Set the coefficients for the grid */
   for (i = 0; i < 7*volume; i += 7)
   {
      for (s = 0; s < 7; s++)
      {
         stencil_indices[s] = s;
         switch (dim)
         {
            case 1:
               values[i  ] = -cx;
               values[i+1] = 2.0*(cx);
               break;
            case 2:
               values[i  ] = -cx;
               values[i+1] = -cy;
               values[i+2] = 2.0*(cx+cy);
               break;
            case 3:
               values[i  ] = -cx;
               values[i+1] = -cy;
               values[i+2] = -cz;
               values[i+3] = 2.0*(cx+cy+cz);
               values[i+4] = -cx;
               values[i+5] = -cy;
               values[i+6] = -cz;
               break;
         }
      }
   }

   for (ib = 0; ib < nblocks; ib++)
   {
      ierr += HYPRE_StructIJMatrixSetBoxValues (A_structij, ilower[ib], iupper[ib], 
                                                7, stencil_indices, values);   
   }

   ierr += HYPRE_StructIJMatrixAssemble(A_structij);

   if (ierr)
   {
       printf("Error in driver building IJMatrix.\n");
       return(-1);
   }

   /*-----------------------------------------------------------
    * Get the underlying matrix
    *-----------------------------------------------------------*/

   A_parcsr = (HYPRE_ParCSRMatrix) HYPRE_StructIJMatrixGetLocalStorage( A_structij );

   /*-----------------------------------------------------------
    * Need partitioning information for setting up vectors
    *-----------------------------------------------------------*/

   HYPRE_ParCSRMatrixGetRowPartitioning(A_parcsr, &partitioning); 

   part_b = hypre_CTAlloc(int, num_procs+1);
   part_x = hypre_CTAlloc(int, num_procs+1);
   for (i=0; i < num_procs+1; i++)
   {
      part_b[i] = partitioning[i];
      part_x[i] = partitioning[i];
   }

   /*-----------------------------------------------------------
    * Set up the RHS b = (1, 1, ..., 1)^T and initial guess x = 0
    *-----------------------------------------------------------*/

   hypre_TFree(values);
   values = hypre_CTAlloc(double, volume);

   ierr  = HYPRE_StructIJVectorCreate(MPI_COMM_WORLD, grid, stencil, &b_structij);
   ierr += HYPRE_StructIJVectorInitialize(b_structij);  
/*
   SetPartitioning comes after Initialize because Initialize sets 
   LocalStorageType, and that must be done before SetPartitioning.
*/
   ierr += HYPRE_StructIJVectorSetPartitioning(b_structij, (const int *) part_b);  
   for (i = 0; i < volume; i++)
   {
      values[i] = 1.0;
   }
   for (ib = 0; ib < nblocks; ib++)
   {
      ierr += HYPRE_StructIJVectorSetBoxValues(b_structij, ilower[ib], iupper[ib], 
                                               values); 
   }
   ierr += HYPRE_StructIJVectorAssemble(b_structij);

   b_parcsr = (HYPRE_ParVector) HYPRE_StructIJVectorGetLocalStorage( b_structij );

   ierr  = HYPRE_StructIJVectorCreate(MPI_COMM_WORLD, grid, stencil, &x_structij);
   ierr += HYPRE_StructIJVectorInitialize(x_structij);
   ierr += HYPRE_StructIJVectorSetPartitioning(x_structij, (const int *) part_x);  
   for (i = 0; i < volume; i++)
   {
      values[i] = 0.0;
   }
   for (ib = 0; ib < nblocks; ib++)
   {
      ierr += HYPRE_StructIJVectorSetBoxValues(x_structij, ilower[ib], iupper[ib], 
                                               values);
   }
   ierr += HYPRE_StructIJVectorAssemble(x_structij);

   x_parcsr = (HYPRE_ParVector) HYPRE_StructIJVectorGetLocalStorage( x_structij );
 
   hypre_TFree(values);

   /*-----------------------------------------------------------
    * Solve the system using AMG
    *-----------------------------------------------------------*/

   if (solver_id == 0)
   {
      if (myid == 0) printf("Solver:  AMG\n");
      time_index = hypre_InitializeTiming("BoomerAMG Setup");
      hypre_BeginTiming(time_index);

      HYPRE_ParAMGCreate(&amg_solver); 
      HYPRE_ParAMGSetCoarsenType(amg_solver, (hybrid*coarsen_type));
      HYPRE_ParAMGSetMeasureType(amg_solver, measure_type);
      HYPRE_ParAMGSetTol(amg_solver, tol);
      HYPRE_ParAMGSetStrongThreshold(amg_solver, strong_threshold);
      HYPRE_ParAMGSetTruncFactor(amg_solver, trunc_factor);
/* note: log is written to standard output, not to file */
      HYPRE_ParAMGSetLogging(amg_solver, ioutdat, "driver.out.log"); 
      HYPRE_ParAMGSetCycleType(amg_solver, cycle_type);
      HYPRE_ParAMGSetNumGridSweeps(amg_solver, num_grid_sweeps);
      HYPRE_ParAMGSetGridRelaxType(amg_solver, grid_relax_type);
      HYPRE_ParAMGSetRelaxWeight(amg_solver, relax_weight);
      HYPRE_ParAMGSetGridRelaxPoints(amg_solver, grid_relax_points);
      HYPRE_ParAMGSetMaxLevels(amg_solver, max_levels);
      HYPRE_ParAMGSetDebugFlag(amg_solver, debug_flag);

      HYPRE_ParAMGSetup(amg_solver, A_parcsr, b_parcsr, x_parcsr);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
 
      time_index = hypre_InitializeTiming("BoomerAMG Solve");
      hypre_BeginTiming(time_index);

      HYPRE_ParAMGSolve(amg_solver, A_parcsr, b_parcsr, x_parcsr);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      HYPRE_ParAMGDestroy(amg_solver);
   }

   /*-----------------------------------------------------------
    * Solve the system using PCG 
    *-----------------------------------------------------------*/

   if (solver_id == 1 || solver_id == 2 || solver_id == 8)
   {
      time_index = hypre_InitializeTiming("PCG Setup");
      hypre_BeginTiming(time_index);
 
      HYPRE_ParCSRPCGCreate(MPI_COMM_WORLD, &pcg_solver);
      HYPRE_ParCSRPCGSetMaxIter(pcg_solver, 500);
      HYPRE_ParCSRPCGSetTol(pcg_solver, tol);
      HYPRE_ParCSRPCGSetTwoNorm(pcg_solver, 1);
      HYPRE_ParCSRPCGSetRelChange(pcg_solver, 0);
      HYPRE_ParCSRPCGSetLogging(pcg_solver, 1);
 
      if (solver_id == 1)
      {
         /* use BoomerAMG as preconditioner */
         if (myid == 0) printf("Solver: AMG-PCG\n");
         HYPRE_ParAMGCreate(&pcg_precond); 
         HYPRE_ParAMGSetCoarsenType(pcg_precond, (hybrid*coarsen_type));
         HYPRE_ParAMGSetMeasureType(pcg_precond, measure_type);
         HYPRE_ParAMGSetStrongThreshold(pcg_precond, strong_threshold);
         HYPRE_ParAMGSetLogging(pcg_precond, ioutdat, "driver.out.log");
         HYPRE_ParAMGSetMaxIter(pcg_precond, 1);
         HYPRE_ParAMGSetCycleType(pcg_precond, cycle_type);
         HYPRE_ParAMGSetNumGridSweeps(pcg_precond, num_grid_sweeps);
         HYPRE_ParAMGSetGridRelaxType(pcg_precond, grid_relax_type);
         HYPRE_ParAMGSetRelaxWeight(pcg_precond, relax_weight);
         HYPRE_ParAMGSetGridRelaxPoints(pcg_precond, grid_relax_points);
         HYPRE_ParAMGSetMaxLevels(pcg_precond, max_levels);
         HYPRE_ParCSRPCGSetPrecond(pcg_solver,
                                   HYPRE_ParAMGSolve,
                                   HYPRE_ParAMGSetup,
                                   pcg_precond);
      }
      else if (solver_id == 2)
      {
         
         /* use diagonal scaling as preconditioner */
         if (myid == 0) printf("Solver: DS-PCG\n");
         pcg_precond = NULL;

         HYPRE_ParCSRPCGSetPrecond(pcg_solver,
                                   HYPRE_ParCSRDiagScale,
                                   HYPRE_ParCSRDiagScaleSetup,
                                   pcg_precond);
      }
      else if (solver_id == 8)
      {
         /* use ParaSails preconditioner */
         if (myid == 0) printf("Solver: ParaSails-PCG\n");

	 HYPRE_ParCSRParaSailsCreate(MPI_COMM_WORLD, &pcg_precond);
	 HYPRE_ParCSRParaSailsSetParams(pcg_precond, 0.1, 1);

         HYPRE_ParCSRPCGSetPrecond(pcg_solver,
                                   HYPRE_ParCSRParaSailsSolve,
                                   HYPRE_ParCSRParaSailsSetup,
                                   pcg_precond);
      }
 
      HYPRE_ParCSRPCGSetup(pcg_solver, A_parcsr, b_parcsr, x_parcsr);
 
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
   
      time_index = hypre_InitializeTiming("PCG Solve");
      hypre_BeginTiming(time_index);
 
      HYPRE_ParCSRPCGSolve(pcg_solver, A_parcsr, b_parcsr, x_parcsr);
 
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
 
      HYPRE_ParCSRPCGGetNumIterations(pcg_solver, &num_iterations);
      HYPRE_ParCSRPCGGetFinalRelativeResidualNorm(pcg_solver, &final_res_norm);
      HYPRE_ParCSRPCGDestroy(pcg_solver);
 
      if (solver_id == 1)
      {
         HYPRE_ParAMGDestroy(pcg_precond);
      }
      else if (solver_id == 8)
      {
	 HYPRE_ParCSRParaSailsDestroy(pcg_precond);
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
    * Solve the system using GMRES 
    *-----------------------------------------------------------*/

   if (solver_id == 3 || solver_id == 4 || solver_id == 7)
   {
      time_index = hypre_InitializeTiming("GMRES Setup");
      hypre_BeginTiming(time_index);
 
      HYPRE_ParCSRGMRESCreate(MPI_COMM_WORLD, &pcg_solver);
      HYPRE_ParCSRGMRESSetKDim(pcg_solver, k_dim);
      HYPRE_ParCSRGMRESSetMaxIter(pcg_solver, 100);
      HYPRE_ParCSRGMRESSetTol(pcg_solver, tol);
      HYPRE_ParCSRGMRESSetLogging(pcg_solver, 1);
 
      if (solver_id == 3)
      {
         /* use BoomerAMG as preconditioner */
         if (myid == 0) printf("Solver: AMG-GMRES\n");

         HYPRE_ParAMGCreate(&pcg_precond); 
         HYPRE_ParAMGSetCoarsenType(pcg_precond, (hybrid*coarsen_type));
         HYPRE_ParAMGSetMeasureType(pcg_precond, measure_type);
         HYPRE_ParAMGSetStrongThreshold(pcg_precond, strong_threshold);
         HYPRE_ParAMGSetLogging(pcg_precond, ioutdat, "driver.out.log");
         HYPRE_ParAMGSetMaxIter(pcg_precond, 1);
         HYPRE_ParAMGSetCycleType(pcg_precond, cycle_type);
         HYPRE_ParAMGSetNumGridSweeps(pcg_precond, num_grid_sweeps);
         HYPRE_ParAMGSetGridRelaxType(pcg_precond, grid_relax_type);
         HYPRE_ParAMGSetRelaxWeight(pcg_precond, relax_weight);
         HYPRE_ParAMGSetGridRelaxPoints(pcg_precond, grid_relax_points);
         HYPRE_ParAMGSetMaxLevels(pcg_precond, max_levels);
         HYPRE_ParCSRGMRESSetPrecond(pcg_solver,
                                   HYPRE_ParAMGSolve,
                                   HYPRE_ParAMGSetup,
                                   pcg_precond);
      }
      else if (solver_id == 4)
      {
         /* use diagonal scaling as preconditioner */
         if (myid == 0) printf("Solver: DS-GMRES\n");
         pcg_precond = NULL;

         HYPRE_ParCSRGMRESSetPrecond(pcg_solver,
                                     HYPRE_ParCSRDiagScale,
                                     HYPRE_ParCSRDiagScaleSetup,
                                     pcg_precond);
      }
      else if (solver_id == 7)
      {
         /* use PILUT as preconditioner */
         if (myid == 0) printf("Solver: Pilut-GMRES\n");

         ierr = HYPRE_ParCSRPilutCreate( MPI_COMM_WORLD, &pcg_precond ); 
         if (ierr) {
	   printf("Error in ParPilutCreate\n");
         }

         HYPRE_ParCSRGMRESSetPrecond(pcg_solver,
                                   HYPRE_ParCSRPilutSolve,
                                   HYPRE_ParCSRPilutSetup,
                                   pcg_precond);

         if (drop_tol >= 0 )
            HYPRE_ParCSRPilutSetDropTolerance( pcg_precond,
               drop_tol );

         if (nonzeros_to_keep >= 0 )
            HYPRE_ParCSRPilutSetFactorRowSize( pcg_precond,
               nonzeros_to_keep );
      }
 
      HYPRE_ParCSRGMRESSetup(pcg_solver, A_parcsr, b_parcsr, x_parcsr);
 
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
   
      time_index = hypre_InitializeTiming("GMRES Solve");
      hypre_BeginTiming(time_index);
 
      HYPRE_ParCSRGMRESSolve(pcg_solver, A_parcsr, b_parcsr, x_parcsr);
 
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
 
      HYPRE_ParCSRGMRESGetNumIterations(pcg_solver, &num_iterations);
      HYPRE_ParCSRGMRESGetFinalRelativeResidualNorm(pcg_solver,&final_res_norm);
      HYPRE_ParCSRGMRESDestroy(pcg_solver);
 
      if (solver_id == 3)
      {
         HYPRE_ParAMGDestroy(pcg_precond);
      }

      if (solver_id == 7)
      {
         HYPRE_ParCSRPilutDestroy(pcg_precond);
      }

      if (myid == 0)
      {
         printf("\n");
         printf("GMRES Iterations = %d\n", num_iterations);
         printf("Final GMRES Relative Residual Norm = %e\n", final_res_norm);
         printf("\n");
      }
   }
   /*-----------------------------------------------------------
    * Solve the system using CGNR 
    *-----------------------------------------------------------*/

   if (solver_id == 5 || solver_id == 6)
   {
      time_index = hypre_InitializeTiming("CGNR Setup");
      hypre_BeginTiming(time_index);
 
      HYPRE_ParCSRCGNRCreate(MPI_COMM_WORLD, &pcg_solver);
      HYPRE_ParCSRCGNRSetMaxIter(pcg_solver, 1000);
      HYPRE_ParCSRCGNRSetTol(pcg_solver, tol);
      HYPRE_ParCSRCGNRSetLogging(pcg_solver, 1);
 
      if (solver_id == 5)
      {
         /* use BoomerAMG as preconditioner */
         if (myid == 0) printf("Solver: AMG-CGNR\n");

         HYPRE_ParAMGCreate(&pcg_precond); 
         HYPRE_ParAMGSetCoarsenType(pcg_precond, (hybrid*coarsen_type));
         HYPRE_ParAMGSetMeasureType(pcg_precond, measure_type);
         HYPRE_ParAMGSetStrongThreshold(pcg_precond, strong_threshold);
         HYPRE_ParAMGSetLogging(pcg_precond, ioutdat, "driver.out.log");
         HYPRE_ParAMGSetMaxIter(pcg_precond, 1);
         HYPRE_ParAMGSetCycleType(pcg_precond, cycle_type);
         HYPRE_ParAMGSetNumGridSweeps(pcg_precond, num_grid_sweeps);
         HYPRE_ParAMGSetGridRelaxType(pcg_precond, grid_relax_type);
         HYPRE_ParAMGSetRelaxWeight(pcg_precond, relax_weight);
         HYPRE_ParAMGSetGridRelaxPoints(pcg_precond, grid_relax_points);
         HYPRE_ParAMGSetMaxLevels(pcg_precond, max_levels);
         HYPRE_ParCSRCGNRSetPrecond(pcg_solver,
                                   HYPRE_ParAMGSolve,
                                   HYPRE_ParAMGSolveT,
                                   HYPRE_ParAMGSetup,
                                   pcg_precond);
      }
      else if (solver_id == 6)
      {
         /* use diagonal scaling as preconditioner */
         if (myid == 0) printf("Solver: DS-CGNR\n");
         pcg_precond = NULL;

         HYPRE_ParCSRCGNRSetPrecond(pcg_solver,
                                   HYPRE_ParCSRDiagScale,
                                   HYPRE_ParCSRDiagScale,
                                   HYPRE_ParCSRDiagScaleSetup,
                                   pcg_precond);
      }
 
      HYPRE_ParCSRCGNRSetup(pcg_solver, A_parcsr, b_parcsr, x_parcsr);
 
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
   
      time_index = hypre_InitializeTiming("CGNR Solve");
      hypre_BeginTiming(time_index);
 
      HYPRE_ParCSRCGNRSolve(pcg_solver, A_parcsr, b_parcsr, x_parcsr);
 
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
 
      HYPRE_ParCSRCGNRGetNumIterations(pcg_solver, &num_iterations);
      HYPRE_ParCSRCGNRGetFinalRelativeResidualNorm(pcg_solver,&final_res_norm);
      HYPRE_ParCSRCGNRDestroy(pcg_solver);
 
      if (solver_id == 5)
      {
         HYPRE_ParAMGDestroy(pcg_precond);
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

#if 0
   HYPRE_PrintCSRVector(x_parcsr, "driver.out.x");
#endif

   /*-----------------------------------------------------------
    * Clean up
    *-----------------------------------------------------------*/

   HYPRE_StructGridDestroy(grid);
   HYPRE_StructStencilDestroy(stencil);
   HYPRE_StructIJMatrixDestroy(A_structij);
   HYPRE_StructIJVectorDestroy(b_structij);
   HYPRE_StructIJVectorDestroy(x_structij);
   HYPRE_ParCSRMatrixDestroy(A_parcsr);
   HYPRE_ParVectorDestroy(b_parcsr);
   HYPRE_ParVectorDestroy(x_parcsr);

   for (i = 0; i < nblocks; i++)
   {
      hypre_TFree(iupper[i]);
      hypre_TFree(ilower[i]);
   }
   hypre_TFree(ilower);
   hypre_TFree(iupper);
   hypre_TFree(stencil_indices);

   for ( i = 0; i < 7; i++)
      hypre_TFree(offsets[i]);
   hypre_TFree(offsets);

   MPI_Finalize();

   return (0);
}
