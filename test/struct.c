#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "utilities.h"
#include "HYPRE_struct_ls.h"
#include "krylov.h"

#define HYPRE_MFLOPS 0
#if HYPRE_MFLOPS
#include "struct_mv.h"
#endif

#include "struct_mv.h"

#ifdef HYPRE_DEBUG
#include <cegdb.h>
#endif

int  SetStencilBndry(HYPRE_StructMatrix A,HYPRE_StructGrid gridmatrix,int* period);

int  AddValuesMatrix(HYPRE_StructMatrix A,HYPRE_StructGrid gridmatrix,
                                     double            cx,
				     double            cy,
				     double            cz) ;

int AddValuesVector( hypre_StructGrid  *gridvector,
                     hypre_StructVector *zvector,
                     int                *period, 
                     double             value  )  ;

/*--------------------------------------------------------------------------
 * Test driver for structured matrix interface (structured storage)
 *--------------------------------------------------------------------------*/
 
/*----------------------------------------------------------------------
 * Standard 7-point laplacian in 3D with grid and anisotropy determined
 * as command line arguments.  Do `driver -help' for usage info.
 *----------------------------------------------------------------------*/

int
main( int   argc,
      char *argv[] )
{
   int                 arg_index;
   int                 print_usage;
   int                 nx, ny, nz;
   int                 P, Q, R;
   int                 bx, by, bz;
   int                 px, py, pz;
   double              cx, cy, cz;
   int                 solver_id;
   int                 solver_type;

   /*double              dxyz[3];*/

   int                 A_num_ghost[6] = {0, 0, 0, 0, 0, 0};
   int                 v_num_ghost[3] = {0,0,0};
                     
   HYPRE_StructMatrix  A;
   HYPRE_StructVector  b;
   HYPRE_StructVector  x;

   HYPRE_StructSolver  solver;
   HYPRE_StructSolver  precond;
   int                 num_iterations;
   int                 time_index;
   double              final_res_norm;
   double              cf_tol;

   int                 num_procs, myid;

   int                 p, q, r;
   int                 dim;
   int                 n_pre, n_post;
   int                 nblocks ;
   int                 skip;
   int                 jump;

   int               **iupper;
   int               **ilower;

   int                 istart[3];
   int                 periodic[3];
   int               **offsets;

   HYPRE_StructGrid    grid;
   HYPRE_StructGrid    readgrid;
   HYPRE_StructStencil stencil;

   int                 i, s;
   int                 ix, iy, iz, ib;

   /* GEC0203   */
   int             read_fromfile_param;
   int             read_fromfile_index;
   int             read_rhsfromfile_param;
   int             read_rhsfromfile_index;
   int             read_x0fromfile_param;
   int             read_x0fromfile_index;
   int             periodx0[3] = {0,0,0};
   int             *readperiodic;
   int             sum;
   /*       */

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

#ifdef HYPRE_USE_PTHREADS
   HYPRE_InitPthreads(4);
#endif  

 
   /* Initialize MPI */
   MPI_Init(&argc, &argv);

   MPI_Comm_size(MPI_COMM_WORLD, &num_procs );
   MPI_Comm_rank(MPI_COMM_WORLD, &myid );


#ifdef HYPRE_DEBUG
   cegdb(&argc, &argv, myid);
#endif

   hypre_InitMemoryDebug(myid);

   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/
 
   dim = 3;

   skip = 0;
   jump = 0;

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

   n_pre  = 1;
   n_post = 1;

   solver_id = 0;
   solver_type = 1;

   istart[0] = -3;
   istart[1] = -3;
   istart[2] = -3;

   px = 0;
   py = 0;
   pz = 0;

   cf_tol = 0.90;


   /* setting defaults for the reading parameters    */
   read_fromfile_param = 0;
   read_fromfile_index = argc;
   read_rhsfromfile_param = 0;
   read_rhsfromfile_index = argc;
   read_x0fromfile_param = 0;
   read_x0fromfile_index = argc;
   sum = 0;

   /* ghosts for the building of matrix: default  */
   for (i = 0; i < dim; i++)
   {
      A_num_ghost[2*i] = 1;
      A_num_ghost[2*i + 1] = 1;
   }

   /*       */

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/
 
   print_usage = 0;
   arg_index = 1;
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
      else if ( strcmp(argv[arg_index], "-b") == 0 )
      {
         arg_index++;
         bx = atoi(argv[arg_index++]);
         by = atoi(argv[arg_index++]);
         bz = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-p") == 0 )
      {
         arg_index++;
         px = atoi(argv[arg_index++]);
         py = atoi(argv[arg_index++]);
         pz = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-c") == 0 )
      {
         arg_index++;
         cx = atof(argv[arg_index++]);
         cy = atof(argv[arg_index++]);
         cz = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-v") == 0 )
      {
         arg_index++;
         n_pre = atoi(argv[arg_index++]);
         n_post = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-d") == 0 )
      {
         arg_index++;
         dim = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-skip") == 0 )
      {
         arg_index++;
         skip = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-jump") == 0 )
      {
         arg_index++;
         jump = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-solver") == 0 )
      {
         arg_index++;
         solver_id = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-solver_type") == 0 )
      {
         arg_index++;
         solver_type = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-cf") == 0 )
      {
         arg_index++;
         cf_tol = atof(argv[arg_index++]);
      }
            /* GEC0203 parsing the arguments to read from file the linear system */
      else if ( strcmp(argv[arg_index], "-fromfile") == 0 )
      {
         arg_index++;
         read_fromfile_param = 1;
         read_fromfile_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-rhsfromfile") == 0 )
      {
         arg_index++;
         read_rhsfromfile_param = 1;
         read_rhsfromfile_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-x0fromfile") == 0 )
      {
         arg_index++;
         read_x0fromfile_param = 1;
         read_x0fromfile_index = arg_index;
      }
      /* GEC0203    */
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
   
   sum = read_x0fromfile_param + read_rhsfromfile_param +read_fromfile_param; 

   /*-----------------------------------------------------------
    * Print usage info
    *-----------------------------------------------------------*/
 
   if ( (print_usage) && (myid == 0) )
   {
      printf("\n");
      printf("Usage: %s [<options>]\n", argv[0]);
      printf("\n");
      printf("  -n <nx> <ny> <nz>    : problem size per block\n");
      printf("  -P <Px> <Py> <Pz>    : processor topology\n");
      printf("  -b <bx> <by> <bz>    : blocking per processor\n");
      printf("  -p <px> <py> <pz>    : periodicity in each dimension\n");
      printf("  -c <cx> <cy> <cz>    : diffusion coefficients\n");
      printf("  -v <n_pre> <n_post>  : number of pre and post relaxations\n");
      printf("  -d <dim>             : problem dimension (2 or 3)\n");
      printf("  -skip <s>            : skip some relaxation in PFMG (0 or 1)\n");
      printf("  -jump <num>          : num levels to jump in SparseMSG\n");
      printf("  -solver <ID>         : solver ID (default = 0)\n");
      /*   */
      printf("  -fromfile <name>     : prefix name for matrixfiles\n");
      printf("  -rhsfromfile <name>  : prefix name for rhsfiles\n");
      printf("  -x0fromfile <name>   : prefix name for firstguessfiles\n");
      /*   */
      printf("                         0  - SMG\n");
      printf("                         1  - PFMG\n");
      printf("                         2  - SparseMSG\n");
      printf("                         10 - CG with SMG precond\n");
      printf("                         11 - CG with PFMG precond\n");
      printf("                         12 - CG with SparseMSG precond\n");
      printf("                         17 - CG with 2-step Jacobi\n");
      printf("                         18 - CG with diagonal scaling\n");
      printf("                         19 - CG\n");
      printf("                         20 - Hybrid with SMG precond\n");
      printf("                         21 - Hybrid with PFMG precond\n");
      printf("                         22 - Hybrid with SparseMSG precond\n");
      printf("                         30 - GMRES with SMG precond\n");
      printf("                         31 - GMRES with PFMG precond\n");
      printf("                         32 - GMRES with SparseMSG precond\n");
      printf("                         37 - GMRES with 2-step Jacobi\n");
      printf("                         38 - GMRES with diagonal scaling\n");
      printf("                         39 - GMRES\n");
      printf("  -solver_type <ID>    : solver type for Hybrid(default = PCG)\n");
      printf("                         2 - GMRES\n");
      printf("  -cf <cf>             : convergence factor for Hybrid\n");
      printf("\n");
   }

   if ( print_usage )
   {
      exit(1);
   }

   /*-----------------------------------------------------------
    * Check a few things
    *-----------------------------------------------------------*/

   if ((P*Q*R) != num_procs)
   {
      printf("Error: Invalid number of processors or processor topology \n");
      exit(1);
   }

   if ((px+py+pz) != 0 && solver_id != 0 )
   {
      printf("\n*** Warning: Periodic implemented only for solver 0 ***\n\n");
      /* exit(1); */
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/
 
   if (myid == 0 && sum == 0)
   {
      printf("Running with these driver parameters:\n");
      printf("  (nx, ny, nz)    = (%d, %d, %d)\n", nx, ny, nz);
      printf("  (Px, Py, Pz)    = (%d, %d, %d)\n", P,  Q,  R);
      printf("  (bx, by, bz)    = (%d, %d, %d)\n", bx, by, bz);
      printf("  (px, py, pz)    = (%d, %d, %d)\n", px, py, pz);
      printf("  (cx, cy, cz)    = (%f, %f, %f)\n", cx, cy, cz);
      printf("  (n_pre, n_post) = (%d, %d)\n", n_pre, n_post);
      printf("  dim             = %d\n", dim);
      printf("  skip            = %d\n", skip);
      printf("  jump            = %d\n", jump);
      printf("  solver ID       = %d\n", solver_id);
   }

   if (myid == 0 && sum > 0)
   {
      printf("Running with these driver parameters:\n");
      printf("  (cx, cy, cz)    = (%f, %f, %f)\n", cx, cy, cz);
      printf("  (n_pre, n_post) = (%d, %d)\n", n_pre, n_post);
      printf("  dim             = %d\n", dim);
      printf("  skip            = %d\n", skip);
      printf("  jump            = %d\n", jump);
      printf("  solver ID       = %d\n", solver_id);
      printf("  the grid is read from  file \n");
	     
   }
  

   /*-----------------------------------------------------------
    * Synchronize so that timings make sense
    *-----------------------------------------------------------*/

   MPI_Barrier(MPI_COMM_WORLD);

   time_index = hypre_InitializeTiming("Struct Interface");
   hypre_BeginTiming(time_index);


   /*-----------------------------------------------------------
    * Set up the stencil structure (7 points) when matrix is NOT read from file
    * Set up also the grid structure used when NO files are read
    *-----------------------------------------------------------*/

   switch (dim)
   {
         case 1:
             nblocks = bx;
             offsets = hypre_CTAlloc(int*, 2);
             offsets[0] = hypre_CTAlloc(int, 1);
             offsets[0][0] = -1; 
             offsets[1] = hypre_CTAlloc(int, 1);
             offsets[1][0] = 0; 
         /* compute p from P and myid */
             p = myid % P;
             break;
         case 2:
             nblocks = bx*by;
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
             nblocks = bx*by*bz;
             offsets = hypre_CTAlloc(int*, 4);
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
         /* compute p,q,r from P,Q,R and myid */
             p = myid % P;
             q = (( myid - p)/P) % Q;
             r = ( myid - p - P*q)/( P*Q );
             break;
   }

   /*-----------------------------------------------------------
    * Set up the stencil structure needed for matrix creation
    * which is always the case for read_fromfile_param == 0
    *-----------------------------------------------------------*/
 
   HYPRE_StructStencilCreate(dim, dim + 1, &stencil);
   for (s = 0; s < dim + 1; s++)
   {
      HYPRE_StructStencilSetElement(stencil, s, offsets[s]);
   }

   /*-----------------------------------------------------------
    * Set up periodic
    *-----------------------------------------------------------*/

   periodic[0] = px;
   periodic[1] = py;
   periodic[2] = pz;

   /*-----------------------------------------------------------
    * Set up dxyz for PFMG solver
    *-----------------------------------------------------------*/

#if 0
   dxyz[0] = 1.0e+123;
   dxyz[1] = 1.0e+123;
   dxyz[2] = 1.0e+123;
   if (cx > 0)
   {
      dxyz[0] = sqrt(1.0 / cx);
   }
   if (cy > 0)
   {
      dxyz[1] = sqrt(1.0 / cy);
   }
   if (cz > 0)
   {
      dxyz[2] = sqrt(1.0 / cz);
   }
#endif

   /* We do the extreme cases first 
    * reading everything from files => sum = 3
    * building things from scratch (grid,stencils,extents) sum = 0
    *                                                            */

   if ( (read_fromfile_param ==1) &&
        (read_x0fromfile_param ==1) &&
        (read_rhsfromfile_param ==1) 
       )
   {
   printf("\nreading all the linear system from files: matrix, rhs and x0\n");
     /* ghost selection for reading the matrix and vectors */
      for (i = 0; i < dim; i++)
      {
          A_num_ghost[2*i] = 1;
          A_num_ghost[2*i + 1] = 1;
          v_num_ghost[2*i] = 1;
          v_num_ghost[2*i + 1] = 1;
          
      }

      A = (HYPRE_StructMatrix) hypre_StructMatrixRead(MPI_COMM_WORLD,
                                     argv[read_fromfile_index],A_num_ghost);

      b = (HYPRE_StructVector) hypre_StructVectorRead(MPI_COMM_WORLD,
                                     argv[read_rhsfromfile_index],v_num_ghost);

      x = (HYPRE_StructVector) hypre_StructVectorRead(MPI_COMM_WORLD,
                                     argv[read_x0fromfile_index],v_num_ghost);

   }

  /* beginning of sum == 0  */
   if (sum == 0 )    /* no read from any file..*/
   {

   /*-----------------------------------------------------------
    * prepare space for the extents
    *-----------------------------------------------------------*/

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
      HYPRE_StructGridSetPeriodic(grid, periodic);
      HYPRE_StructGridAssemble(grid);

   /*-----------------------------------------------------------
    * Set up the matrix structure
    *-----------------------------------------------------------*/

      for (i = 0; i < dim; i++)
      {
         A_num_ghost[2*i] = 1;
         A_num_ghost[2*i + 1] = 1;
      }

      HYPRE_StructMatrixCreate(MPI_COMM_WORLD, grid, stencil, &A);
      HYPRE_StructMatrixSetSymmetric(A, 1);
      HYPRE_StructMatrixSetNumGhost(A, A_num_ghost);
      HYPRE_StructMatrixInitialize(A);
   /*-----------------------------------------------------------
    * Fill in the matrix elements
    *-----------------------------------------------------------*/
   
      AddValuesMatrix(A,grid,cx,cy,cz);

   /* Zero out stencils reaching to real boundary */

      SetStencilBndry(A,grid,periodic); 
      HYPRE_StructMatrixAssemble(A);

#if 0
   HYPRE_StructMatrixPrint("drive.out.A", A, 0);
#endif

   /*-----------------------------------------------------------
    * Set up the linear system
    *-----------------------------------------------------------*/

      HYPRE_StructVectorCreate(MPI_COMM_WORLD, grid, &b);
      HYPRE_StructVectorInitialize(b);

   /*-----------------------------------------------------------
    * For periodic b.c. in all directions, need rhs to satisfy 
    * compatibility condition. Achieved by setting a source and
    *  sink of equal strength.  All other problems have rhs = 1.
    *-----------------------------------------------------------*/

      AddValuesVector(grid,b,periodic,1.0);
      HYPRE_StructVectorAssemble(b);

#if 0
   HYPRE_StructVectorPrint("drive.out.b", b, 0);
#endif
   
      HYPRE_StructVectorCreate(MPI_COMM_WORLD, grid, &x);
      HYPRE_StructVectorInitialize(x);
    
  
      AddValuesVector(grid,x,periodx0,0.0);
      HYPRE_StructVectorAssemble(x);

#if 0
   HYPRE_StructVectorPrint("drive.out.x0", x, 0);
#endif

   /* finishing the setup of linear system here extreme case
    * end of if sum == 0 
    */

      HYPRE_StructGridDestroy(grid);

      for (i = 0; i < nblocks; i++)
      {
         hypre_TFree(iupper[i]);
         hypre_TFree(ilower[i]);
      }
      hypre_TFree(ilower);
      hypre_TFree(iupper);
   }

   if ( (sum > 0 ) && (sum < 3))   /* the grid will be read from file.  */
   {

      if (read_fromfile_param == 0) /* the grid will come from rhs or from x0 */
      {

         if ((read_rhsfromfile_param > 0) && (read_x0fromfile_param == 0))
         {                     

   /* read right hand side, extract grid, construct matrix, construct x0 */

           printf("\ninitial rhs from file prefix :%s\n",argv[read_rhsfromfile_index]);

            /* ghost selection for vector  */
                for (i = 0; i < dim; i++)
                {
                  v_num_ghost[2*i] = 1;
                  v_num_ghost[2*i + 1] = 1;
                }
  
                  b = (HYPRE_StructVector) hypre_StructVectorRead(MPI_COMM_WORLD,
                                     argv[read_rhsfromfile_index],v_num_ghost);
    
    
                readgrid = hypre_StructVectorGrid(b) ;
                readperiodic = hypre_StructGridPeriodic(readgrid);  

                HYPRE_StructVectorCreate(MPI_COMM_WORLD, readgrid, &x);
                HYPRE_StructVectorInitialize(x);
  
                AddValuesVector(readgrid,x,periodx0,0.0);
                HYPRE_StructVectorAssemble(x);
 
                HYPRE_StructMatrixCreate(MPI_COMM_WORLD, readgrid, stencil, &A);
                HYPRE_StructMatrixSetSymmetric(A, 1);
                HYPRE_StructMatrixSetNumGhost(A, A_num_ghost);
                HYPRE_StructMatrixInitialize(A);
   /*-----------------------------------------------------------
    * Fill in the matrix elements
    *-----------------------------------------------------------*/
   
                AddValuesMatrix(A,readgrid,cx,cy,cz);

   /* Zero out stencils reaching to real boundary */

                SetStencilBndry(A,readgrid,readperiodic); 
                HYPRE_StructMatrixAssemble(A);

#if 0
   HYPRE_StructVectorPrint("drive.readout.b", b, 0);
   HYPRE_StructVectorPrint("drive.readout.x0", x, 0);
   HYPRE_StructMatrixPrint("drive.readout.A", A, 0);
#endif

		   /* done with one case rhs=1 x0 = 0  */
         }   

   /*  case when rhs=0 and read x0=1  */

         if ((read_rhsfromfile_param == 0) && (read_x0fromfile_param > 0))
         {                     

   /* read right hand side, extract grid, construct matrix, construct x0 */

              printf("\ninitial x0 from file prefix :%s\n",argv[read_x0fromfile_index]);

                /* ghost selection for vector  */
                for (i = 0; i < dim; i++)
                {
                  v_num_ghost[2*i] = 1;
                  v_num_ghost[2*i + 1] = 1;
                }
  
                x = (HYPRE_StructVector) hypre_StructVectorRead(MPI_COMM_WORLD,
					       argv[read_x0fromfile_index],v_num_ghost);

		 readgrid = hypre_StructVectorGrid(x) ;
		 readperiodic = hypre_StructGridPeriodic(readgrid);  

                HYPRE_StructVectorCreate(MPI_COMM_WORLD, readgrid, &b);
                HYPRE_StructVectorInitialize(b);
                AddValuesVector(readgrid,b,readperiodic,1.0);

                HYPRE_StructVectorAssemble(b);

                HYPRE_StructMatrixCreate(MPI_COMM_WORLD, readgrid, stencil, &A);
                HYPRE_StructMatrixSetSymmetric(A, 1);
                HYPRE_StructMatrixSetNumGhost(A, A_num_ghost);
                HYPRE_StructMatrixInitialize(A);
   /*-----------------------------------------------------------
    * Fill in the matrix elements
    *-----------------------------------------------------------*/
   
                AddValuesMatrix(A,readgrid,cx,cy,cz);

   /* Zero out stencils reaching to real boundary */

                SetStencilBndry(A,readgrid,readperiodic); 
                HYPRE_StructMatrixAssemble(A);
#if 0
   HYPRE_StructVectorPrint("drive.readout.b", b, 0);
   HYPRE_StructVectorPrint("drive.readout.x0", x, 0);
   HYPRE_StructMatrixPrint("drive.readout.A", A, 0);
#endif

		   /* done with one case rhs=0 x0 = 1  */
	 }



   /* the other case when read rhs > 0 and read x0 > 0  */
         if ((read_rhsfromfile_param > 0) && (read_x0fromfile_param > 0))
         {                    

   /* read right hand side, extract grid, construct matrix, construct x0 */

                printf("\ninitial rhs  from file prefix :%s\n",argv[read_rhsfromfile_index]);
                printf("\ninitial x0  from file prefix :%s\n",argv[read_x0fromfile_index]);

                /* ghost selection for vector  */
                for (i = 0; i < dim; i++)
                {
                v_num_ghost[2*i] = 1;
                v_num_ghost[2*i + 1] = 1;
                }
  
                b = (HYPRE_StructVector) hypre_StructVectorRead(MPI_COMM_WORLD,
                                     argv[read_rhsfromfile_index],v_num_ghost);

                x = (HYPRE_StructVector) hypre_StructVectorRead(MPI_COMM_WORLD,
					       argv[read_x0fromfile_index],v_num_ghost);

	        readgrid= hypre_StructVectorGrid(b) ;
		readperiodic = hypre_StructGridPeriodic(readgrid); 

                HYPRE_StructMatrixCreate(MPI_COMM_WORLD, readgrid, stencil, &A);
                HYPRE_StructMatrixSetSymmetric(A, 1);
                HYPRE_StructMatrixSetNumGhost(A, A_num_ghost);
                HYPRE_StructMatrixInitialize(A);
   /*-----------------------------------------------------------
    * Fill in the matrix elements
    *-----------------------------------------------------------*/
   
                AddValuesMatrix(A,readgrid,cx,cy,cz);

   /* Zero out stencils reaching to real boundary */

                SetStencilBndry(A,readgrid,readperiodic); 
                HYPRE_StructMatrixAssemble(A);
#if 0
   HYPRE_StructVectorPrint("drive.readout.x0", x, 0);
   HYPRE_StructVectorPrint("drive.readout.b", b, 0);  
   HYPRE_StructMatrixPrint("drive.readout.A", A, 0);
#endif

	 }	   /* done with one case rhs=1 x0 = 1  */
         
      }  /* done with the case where you no read matrix  */
                
   

      if (read_fromfile_param == 1)  /* still sum > 0  */
      {   
         printf("\nreading matrix from file:%s\n",argv[read_fromfile_index]);
        /* ghost selection for reading the matrix  */
         for (i = 0; i < dim; i++)
         {
            A_num_ghost[2*i] = 1;
            A_num_ghost[2*i + 1] = 1;
         }

         A = (HYPRE_StructMatrix) hypre_StructMatrixRead(MPI_COMM_WORLD,
                                     argv[read_fromfile_index],A_num_ghost);

	 readgrid = hypre_StructMatrixGrid(A);
	 readperiodic  =  hypre_StructGridPeriodic(readgrid);  

         if ((read_rhsfromfile_param > 0) && (read_x0fromfile_param == 0))
         {                
               /* read right hand side ,construct x0 */

                printf("\ninitial rhs from file prefix :%s\n",argv[read_rhsfromfile_index]);

                /* ghost selection for vector  */
                for (i = 0; i < dim; i++)
                {
                v_num_ghost[2*i] = 1;
                v_num_ghost[2*i + 1] = 1;
                }
  
                b = (HYPRE_StructVector) hypre_StructVectorRead(MPI_COMM_WORLD,
				           argv[read_rhsfromfile_index],v_num_ghost);

                HYPRE_StructVectorCreate(MPI_COMM_WORLD, readgrid,&x);
                HYPRE_StructVectorInitialize(x);
                AddValuesVector(readgrid,x,periodx0,0.0);
                HYPRE_StructVectorAssemble(x);


#if 0
    HYPRE_StructVectorPrint("drive.readout.x0", x, 0);
    HYPRE_StructVectorPrint("drive.readout.b", b, 0);
    HYPRE_StructMatrixPrint("drive.readout.A", A, 0);
#endif
	 }

         if ((read_rhsfromfile_param == 0) && (read_x0fromfile_param > 0))
         {                   

   /* read x0, construct rhs*/

                printf("\ninitial x0 from file prefix :%s\n",argv[read_x0fromfile_index]);

                /* ghost selection for vector  */
                for (i = 0; i < dim; i++)
                {
                v_num_ghost[2*i] = 1;
                v_num_ghost[2*i + 1] = 1;
                }
  
                  x = (HYPRE_StructVector) hypre_StructVectorRead(MPI_COMM_WORLD,
					       argv[read_x0fromfile_index],v_num_ghost);

                HYPRE_StructVectorCreate(MPI_COMM_WORLD, readgrid, &b);
                HYPRE_StructVectorInitialize(b);
                AddValuesVector(readgrid,b,readperiodic,1.0);
                HYPRE_StructVectorAssemble(b);
#if 0
    HYPRE_StructVectorPrint("drive.readout.x0", x, 0);
    HYPRE_StructVectorPrint("drive.readout.b", b, 0);
    HYPRE_StructMatrixPrint("drive.readout.A", A, 0);
#endif          
	 }

         if ((read_rhsfromfile_param == 0) && (read_x0fromfile_param == 0))
         {                    
               /* construct x0 , construct b*/

                HYPRE_StructVectorCreate(MPI_COMM_WORLD, readgrid, &b);
                HYPRE_StructVectorInitialize(b);
                AddValuesVector(readgrid,b,readperiodic,1.0);
                HYPRE_StructVectorAssemble(b);


                HYPRE_StructVectorCreate(MPI_COMM_WORLD, readgrid, &x);
                HYPRE_StructVectorInitialize(x);
                AddValuesVector(readgrid,x,periodx0,0.0);
                HYPRE_StructVectorAssemble(x); 

#if 0
  HYPRE_StructVectorPrint("drive.readout.x0", x, 0);
  HYPRE_StructVectorPrint("drive.readout.b", b, 0);
  HYPRE_StructMatrixPrint("drive.readout.A", A, 0);
#endif
		  
	 }   
      }    /* finish the read of matrix  */
   
   }        /* finish the sum > 0 case   */
 

		  /* linear system complete  */

   hypre_EndTiming(time_index);
   hypre_PrintTiming("Struct Interface", MPI_COMM_WORLD);
   hypre_FinalizeTiming(time_index);
   hypre_ClearTiming();

   /*-----------------------------------------------------------
    * Solve the system using SMG
    *-----------------------------------------------------------*/

#if !HYPRE_MFLOPS

   if (solver_id == 0)
   {
      time_index = hypre_InitializeTiming("SMG Setup");
      hypre_BeginTiming(time_index);

      HYPRE_StructSMGCreate(MPI_COMM_WORLD, &solver);
      HYPRE_StructSMGSetMemoryUse(solver, 0);
      HYPRE_StructSMGSetMaxIter(solver, 50);
      HYPRE_StructSMGSetTol(solver, 1.0e-06);
      HYPRE_StructSMGSetRelChange(solver, 0);
      HYPRE_StructSMGSetNumPreRelax(solver, n_pre);
      HYPRE_StructSMGSetNumPostRelax(solver, n_post);
      HYPRE_StructSMGSetLogging(solver, 1);
      HYPRE_StructSMGSetup(solver, A, b, x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      time_index = hypre_InitializeTiming("SMG Solve");
      hypre_BeginTiming(time_index);

      HYPRE_StructSMGSolve(solver, A, b, x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
   
      HYPRE_StructSMGGetNumIterations(solver, &num_iterations);
      HYPRE_StructSMGGetFinalRelativeResidualNorm(solver, &final_res_norm);
      HYPRE_StructSMGDestroy(solver);
   }

   /*-----------------------------------------------------------
    * Solve the system using PFMG
    *-----------------------------------------------------------*/

   else if (solver_id == 1)
   {
      time_index = hypre_InitializeTiming("PFMG Setup");
      hypre_BeginTiming(time_index);

      HYPRE_StructPFMGCreate(MPI_COMM_WORLD, &solver);
      HYPRE_StructPFMGSetMaxIter(solver, 50);
      HYPRE_StructPFMGSetTol(solver, 1.0e-06);
      HYPRE_StructPFMGSetRelChange(solver, 0);
      /* weighted Jacobi = 1; red-black GS = 2 */
      HYPRE_StructPFMGSetRelaxType(solver, 1);
      HYPRE_StructPFMGSetNumPreRelax(solver, n_pre);
      HYPRE_StructPFMGSetNumPostRelax(solver, n_post);
      HYPRE_StructPFMGSetSkipRelax(solver, skip);
      /*HYPRE_StructPFMGSetDxyz(solver, dxyz);*/
      HYPRE_StructPFMGSetLogging(solver, 1);
      HYPRE_StructPFMGSetup(solver, A, b, x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      time_index = hypre_InitializeTiming("PFMG Solve");
      hypre_BeginTiming(time_index);

      HYPRE_StructPFMGSolve(solver, A, b, x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
   
      HYPRE_StructPFMGGetNumIterations(solver, &num_iterations);
      HYPRE_StructPFMGGetFinalRelativeResidualNorm(solver, &final_res_norm);
      HYPRE_StructPFMGDestroy(solver);
   }

   /*-----------------------------------------------------------
    * Solve the system using SparseMSG
    *-----------------------------------------------------------*/

   else if (solver_id == 2)
   {
      time_index = hypre_InitializeTiming("SparseMSG Setup");
      hypre_BeginTiming(time_index);

      HYPRE_StructSparseMSGCreate(MPI_COMM_WORLD, &solver);
      HYPRE_StructSparseMSGSetMaxIter(solver, 50);
      HYPRE_StructSparseMSGSetJump(solver, jump);
      HYPRE_StructSparseMSGSetTol(solver, 1.0e-06);
      HYPRE_StructSparseMSGSetRelChange(solver, 0);
      /* weighted Jacobi = 1; red-black GS = 2 */
      HYPRE_StructSparseMSGSetRelaxType(solver, 1);
      HYPRE_StructSparseMSGSetNumPreRelax(solver, n_pre);
      HYPRE_StructSparseMSGSetNumPostRelax(solver, n_post);
      HYPRE_StructSparseMSGSetLogging(solver, 1);
      HYPRE_StructSparseMSGSetup(solver, A, b, x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      time_index = hypre_InitializeTiming("SparseMSG Solve");
      hypre_BeginTiming(time_index);

      HYPRE_StructSparseMSGSolve(solver, A, b, x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
   
      HYPRE_StructSparseMSGGetNumIterations(solver, &num_iterations);
      HYPRE_StructSparseMSGGetFinalRelativeResidualNorm(solver,
                                                        &final_res_norm);
      HYPRE_StructSparseMSGDestroy(solver);
   }

   /*-----------------------------------------------------------
    * Solve the system using CG
    *-----------------------------------------------------------*/

   if ((solver_id > 9) && (solver_id < 20))
   {
      time_index = hypre_InitializeTiming("PCG Setup");
      hypre_BeginTiming(time_index);

      HYPRE_StructPCGCreate(MPI_COMM_WORLD, &solver);
      HYPRE_PCGSetMaxIter( (HYPRE_Solver)solver, 50 );
      HYPRE_PCGSetTol( (HYPRE_Solver)solver, 1.0e-06 );
      HYPRE_PCGSetTwoNorm( (HYPRE_Solver)solver, 1 );
      HYPRE_PCGSetRelChange( (HYPRE_Solver)solver, 0 );
      HYPRE_PCGSetPrintLevel( (HYPRE_Solver)solver, 1 );

      if (solver_id == 10)
      {
         /* use symmetric SMG as preconditioner */
         HYPRE_StructSMGCreate(MPI_COMM_WORLD, &precond);
         HYPRE_StructSMGSetMemoryUse(precond, 0);
         HYPRE_StructSMGSetMaxIter(precond, 1);
         HYPRE_StructSMGSetTol(precond, 0.0);
         HYPRE_StructSMGSetZeroGuess(precond);
         HYPRE_StructSMGSetNumPreRelax(precond, n_pre);
         HYPRE_StructSMGSetNumPostRelax(precond, n_post);
         HYPRE_StructSMGSetLogging(precond, 0);
         HYPRE_PCGSetPrecond( (HYPRE_Solver) solver,
                              (HYPRE_PtrToSolverFcn) HYPRE_StructSMGSolve,
                              (HYPRE_PtrToSolverFcn) HYPRE_StructSMGSetup,
                              (HYPRE_Solver) precond);
      }

      else if (solver_id == 11)
      {
         /* use symmetric PFMG as preconditioner */
         HYPRE_StructPFMGCreate(MPI_COMM_WORLD, &precond);
         HYPRE_StructPFMGSetMaxIter(precond, 1);
         HYPRE_StructPFMGSetTol(precond, 0.0);
         HYPRE_StructPFMGSetZeroGuess(precond);
         /* weighted Jacobi = 1; red-black GS = 2 */
         HYPRE_StructPFMGSetRelaxType(precond, 1);
         HYPRE_StructPFMGSetNumPreRelax(precond, n_pre);
         HYPRE_StructPFMGSetNumPostRelax(precond, n_post);
         HYPRE_StructPFMGSetSkipRelax(precond, skip);
         /*HYPRE_StructPFMGSetDxyz(precond, dxyz);*/
         HYPRE_StructPFMGSetLogging(precond, 0);
         HYPRE_PCGSetPrecond( (HYPRE_Solver) solver,
                              (HYPRE_PtrToSolverFcn) HYPRE_StructPFMGSolve,
                              (HYPRE_PtrToSolverFcn) HYPRE_StructPFMGSetup,
                              (HYPRE_Solver) precond);
      }

      else if (solver_id == 12)
      {
         /* use symmetric SparseMSG as preconditioner */
         HYPRE_StructSparseMSGCreate(MPI_COMM_WORLD, &precond);
         HYPRE_StructSparseMSGSetMaxIter(precond, 1);
         HYPRE_StructSparseMSGSetJump(precond, jump);
         HYPRE_StructSparseMSGSetTol(precond, 0.0);
         HYPRE_StructSparseMSGSetZeroGuess(precond);
         /* weighted Jacobi = 1; red-black GS = 2 */
         HYPRE_StructSparseMSGSetRelaxType(precond, 1);
         HYPRE_StructSparseMSGSetNumPreRelax(precond, n_pre);
         HYPRE_StructSparseMSGSetNumPostRelax(precond, n_post);
         HYPRE_StructSparseMSGSetLogging(precond, 0);
         HYPRE_PCGSetPrecond( (HYPRE_Solver) solver,
                              (HYPRE_PtrToSolverFcn) HYPRE_StructSparseMSGSolve,
                              (HYPRE_PtrToSolverFcn) HYPRE_StructSparseMSGSetup,
                              (HYPRE_Solver) precond);
      }

      else if (solver_id == 17)
      {
         /* use two-step Jacobi as preconditioner */
         HYPRE_StructJacobiCreate(MPI_COMM_WORLD, &precond);
         HYPRE_StructJacobiSetMaxIter(precond, 2);
         HYPRE_StructJacobiSetTol(precond, 0.0);
         HYPRE_StructJacobiSetZeroGuess(precond);
         HYPRE_PCGSetPrecond( (HYPRE_Solver) solver,
                               (HYPRE_PtrToSolverFcn) HYPRE_StructJacobiSolve,
                               (HYPRE_PtrToSolverFcn) HYPRE_StructJacobiSetup,
                               (HYPRE_Solver) precond);
      }

      else if (solver_id == 18)
      {
         /* use diagonal scaling as preconditioner */
#ifdef HYPRE_USE_PTHREADS
         for (i = 0; i < hypre_NumThreads; i++)
         {
            precond[i] = NULL;
         }
#else
         precond = NULL;
#endif
         HYPRE_PCGSetPrecond( (HYPRE_Solver) solver,
                              (HYPRE_PtrToSolverFcn) HYPRE_StructDiagScale,
                              (HYPRE_PtrToSolverFcn) HYPRE_StructDiagScaleSetup,
                              (HYPRE_Solver) precond);
      }

      HYPRE_PCGSetup
         ( (HYPRE_Solver)solver, (HYPRE_Matrix)A, (HYPRE_Vector)b, (HYPRE_Vector)x );

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
   
      time_index = hypre_InitializeTiming("PCG Solve");
      hypre_BeginTiming(time_index);

      HYPRE_PCGSolve
         ( (HYPRE_Solver) solver, (HYPRE_Matrix)A, (HYPRE_Vector)b, (HYPRE_Vector)x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      HYPRE_PCGGetNumIterations( (HYPRE_Solver)solver, &num_iterations );
      HYPRE_PCGGetFinalRelativeResidualNorm( (HYPRE_Solver)solver, &final_res_norm );
      HYPRE_StructPCGDestroy(solver);

      if (solver_id == 10)
      {
         HYPRE_StructSMGDestroy(precond);
      }
      else if (solver_id == 11)
      {
         HYPRE_StructPFMGDestroy(precond);
      }
      else if (solver_id == 12)
      {
         HYPRE_StructSparseMSGDestroy(precond);
      }
      else if (solver_id == 17)
      {
         HYPRE_StructJacobiDestroy(precond);
      }
   }

   /*-----------------------------------------------------------
    * Solve the system using Hybrid
    *-----------------------------------------------------------*/

   if ((solver_id > 19) && (solver_id < 30))
   {
      time_index = hypre_InitializeTiming("Hybrid Setup");
      hypre_BeginTiming(time_index);

      HYPRE_StructHybridCreate(MPI_COMM_WORLD, &solver);
      HYPRE_StructHybridSetDSCGMaxIter(solver, 100);
      HYPRE_StructHybridSetPCGMaxIter(solver, 50);
      HYPRE_StructHybridSetTol(solver, 1.0e-06);
      HYPRE_StructHybridSetConvergenceTol(solver, cf_tol);
      HYPRE_StructHybridSetTwoNorm(solver, 1);
      HYPRE_StructHybridSetRelChange(solver, 0);
      if (solver_type == 2) /* for use with GMRES */
      {
         HYPRE_StructHybridSetStopCrit(solver, 0);
         HYPRE_StructHybridSetKDim(solver, 10);
      }
      HYPRE_StructHybridSetLogging(solver, 1);
      HYPRE_StructHybridSetSolverType(solver, solver_type);

      if (solver_id == 20)
      {
         /* use symmetric SMG as preconditioner */
         HYPRE_StructSMGCreate(MPI_COMM_WORLD, &precond);
         HYPRE_StructSMGSetMemoryUse(precond, 0);
         HYPRE_StructSMGSetMaxIter(precond, 1);
         HYPRE_StructSMGSetTol(precond, 0.0);
         HYPRE_StructSMGSetZeroGuess(precond);
         HYPRE_StructSMGSetNumPreRelax(precond, n_pre);
         HYPRE_StructSMGSetNumPostRelax(precond, n_post);
         HYPRE_StructSMGSetLogging(precond, 0);
         HYPRE_StructHybridSetPrecond(solver,
                                      HYPRE_StructSMGSolve,
                                      HYPRE_StructSMGSetup,
                                      precond);
      }

      else if (solver_id == 21)
      {
         /* use symmetric PFMG as preconditioner */
         HYPRE_StructPFMGCreate(MPI_COMM_WORLD, &precond);
         HYPRE_StructPFMGSetMaxIter(precond, 1);
         HYPRE_StructPFMGSetTol(precond, 0.0);
         HYPRE_StructPFMGSetZeroGuess(precond);
         /* weighted Jacobi = 1; red-black GS = 2 */
         HYPRE_StructPFMGSetRelaxType(precond, 1);
         HYPRE_StructPFMGSetNumPreRelax(precond, n_pre);
         HYPRE_StructPFMGSetNumPostRelax(precond, n_post);
         HYPRE_StructPFMGSetSkipRelax(precond, skip);
         /*HYPRE_StructPFMGSetDxyz(precond, dxyz);*/
         HYPRE_StructPFMGSetLogging(precond, 0);
         HYPRE_StructHybridSetPrecond(solver,
                                      HYPRE_StructPFMGSolve,
                                      HYPRE_StructPFMGSetup,
                                      precond);
      }

      else if (solver_id == 22)
      {
         /* use symmetric SparseMSG as preconditioner */
         HYPRE_StructSparseMSGCreate(MPI_COMM_WORLD, &precond);
         HYPRE_StructSparseMSGSetJump(precond, jump);
         HYPRE_StructSparseMSGSetMaxIter(precond, 1);
         HYPRE_StructSparseMSGSetTol(precond, 0.0);
         HYPRE_StructSparseMSGSetZeroGuess(precond);
         /* weighted Jacobi = 1; red-black GS = 2 */
         HYPRE_StructSparseMSGSetRelaxType(precond, 1);
         HYPRE_StructSparseMSGSetNumPreRelax(precond, n_pre);
         HYPRE_StructSparseMSGSetNumPostRelax(precond, n_post);
         HYPRE_StructSparseMSGSetLogging(precond, 0);
         HYPRE_StructHybridSetPrecond(solver,
                                      HYPRE_StructSparseMSGSolve,
                                      HYPRE_StructSparseMSGSetup,
                                      precond);
      }

      HYPRE_StructHybridSetup(solver, A, b, x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
   
      time_index = hypre_InitializeTiming("Hybrid Solve");
      hypre_BeginTiming(time_index);

      HYPRE_StructHybridSolve(solver, A, b, x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      HYPRE_StructHybridGetNumIterations(solver, &num_iterations);
      HYPRE_StructHybridGetFinalRelativeResidualNorm(solver, &final_res_norm);
      HYPRE_StructHybridDestroy(solver);

      if (solver_id == 20)
      {
         HYPRE_StructSMGDestroy(precond);
      }
      else if (solver_id == 21)
      {
         HYPRE_StructPFMGDestroy(precond);
      }
      else if (solver_id == 22)
      {
         HYPRE_StructSparseMSGDestroy(precond);
      }
   }

   /*-----------------------------------------------------------
    * Solve the system using GMRES
    *-----------------------------------------------------------*/

   if ((solver_id > 29) && (solver_id < 40))
   {
      time_index = hypre_InitializeTiming("GMRES Setup");
      hypre_BeginTiming(time_index);

      HYPRE_StructGMRESCreate(MPI_COMM_WORLD, &solver);
      HYPRE_GMRESSetMaxIter( (HYPRE_Solver)solver, 50 );
      HYPRE_GMRESSetTol( (HYPRE_Solver)solver, 1.0e-06 );
      HYPRE_GMRESSetRelChange( (HYPRE_Solver)solver, 0 );
      HYPRE_GMRESSetLogging( (HYPRE_Solver)solver, 1 );

      if (solver_id == 30)
      {
         /* use symmetric SMG as preconditioner */
         HYPRE_StructSMGCreate(MPI_COMM_WORLD, &precond);
         HYPRE_StructSMGSetMemoryUse(precond, 0);
         HYPRE_StructSMGSetMaxIter(precond, 1);
         HYPRE_StructSMGSetTol(precond, 0.0);
         HYPRE_StructSMGSetZeroGuess(precond);
         HYPRE_StructSMGSetNumPreRelax(precond, n_pre);
         HYPRE_StructSMGSetNumPostRelax(precond, n_post);
         HYPRE_StructSMGSetLogging(precond, 0);
         HYPRE_GMRESSetPrecond( (HYPRE_Solver)solver,
                                (HYPRE_PtrToSolverFcn) HYPRE_StructSMGSolve,
                                (HYPRE_PtrToSolverFcn) HYPRE_StructSMGSetup,
                                (HYPRE_Solver)precond);
      }

      else if (solver_id == 31)
      {
         /* use symmetric PFMG as preconditioner */
         HYPRE_StructPFMGCreate(MPI_COMM_WORLD, &precond);
         HYPRE_StructPFMGSetMaxIter(precond, 1);
         HYPRE_StructPFMGSetTol(precond, 0.0);
         HYPRE_StructPFMGSetZeroGuess(precond);
         /* weighted Jacobi = 1; red-black GS = 2 */
         HYPRE_StructPFMGSetRelaxType(precond, 1);
         HYPRE_StructPFMGSetNumPreRelax(precond, n_pre);
         HYPRE_StructPFMGSetNumPostRelax(precond, n_post);
         HYPRE_StructPFMGSetSkipRelax(precond, skip);
         /*HYPRE_StructPFMGSetDxyz(precond, dxyz);*/
         HYPRE_StructPFMGSetLogging(precond, 0);
         HYPRE_GMRESSetPrecond( (HYPRE_Solver)solver,
                                (HYPRE_PtrToSolverFcn) HYPRE_StructPFMGSolve,
                                (HYPRE_PtrToSolverFcn) HYPRE_StructPFMGSetup,
                                (HYPRE_Solver)precond);
      }

      else if (solver_id == 32)
      {
         /* use symmetric SparseMSG as preconditioner */
         HYPRE_StructSparseMSGCreate(MPI_COMM_WORLD, &precond);
         HYPRE_StructSparseMSGSetMaxIter(precond, 1);
         HYPRE_StructSparseMSGSetJump(precond, jump);
         HYPRE_StructSparseMSGSetTol(precond, 0.0);
         HYPRE_StructSparseMSGSetZeroGuess(precond);
         /* weighted Jacobi = 1; red-black GS = 2 */
         HYPRE_StructSparseMSGSetRelaxType(precond, 1);
         HYPRE_StructSparseMSGSetNumPreRelax(precond, n_pre);
         HYPRE_StructSparseMSGSetNumPostRelax(precond, n_post);
         HYPRE_StructSparseMSGSetLogging(precond, 0);
         HYPRE_GMRESSetPrecond( (HYPRE_Solver)solver,
                                (HYPRE_PtrToSolverFcn) HYPRE_StructSparseMSGSolve,
                                (HYPRE_PtrToSolverFcn) HYPRE_StructSparseMSGSetup,
                                (HYPRE_Solver)precond);
      }

      else if (solver_id == 37)
      {
         /* use two-step Jacobi as preconditioner */
         HYPRE_StructJacobiCreate(MPI_COMM_WORLD, &precond);
         HYPRE_StructJacobiSetMaxIter(precond, 2);
         HYPRE_StructJacobiSetTol(precond, 0.0);
         HYPRE_StructJacobiSetZeroGuess(precond);
         HYPRE_GMRESSetPrecond( (HYPRE_Solver)solver,
                                (HYPRE_PtrToSolverFcn) HYPRE_StructJacobiSolve,
                                (HYPRE_PtrToSolverFcn) HYPRE_StructJacobiSetup,
                                (HYPRE_Solver)precond);
      }

      else if (solver_id == 38)
      {
         /* use diagonal scaling as preconditioner */
#ifdef HYPRE_USE_PTHREADS
         for (i = 0; i < hypre_NumThreads; i++)
         {
            precond[i] = NULL;
         }
#else
         precond = NULL;
#endif
         HYPRE_GMRESSetPrecond( (HYPRE_Solver)solver,
                                (HYPRE_PtrToSolverFcn) HYPRE_StructDiagScale,
                                (HYPRE_PtrToSolverFcn) HYPRE_StructDiagScaleSetup,
                                (HYPRE_Solver)precond);
      }

      HYPRE_GMRESSetup
         ( (HYPRE_Solver)solver, (HYPRE_Matrix)A, (HYPRE_Vector)b, (HYPRE_Vector)x );

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
   
      time_index = hypre_InitializeTiming("GMRES Solve");
      hypre_BeginTiming(time_index);

      HYPRE_GMRESSolve
         ( (HYPRE_Solver)solver, (HYPRE_Matrix)A, (HYPRE_Vector)b, (HYPRE_Vector)x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      HYPRE_GMRESGetNumIterations( (HYPRE_Solver)solver, &num_iterations);
      HYPRE_GMRESGetFinalRelativeResidualNorm( (HYPRE_Solver)solver, &final_res_norm);
      HYPRE_StructGMRESDestroy(solver);

      if (solver_id == 30)
      {
         HYPRE_StructSMGDestroy(precond);
      }
      else if (solver_id == 31)
      {
         HYPRE_StructPFMGDestroy(precond);
      }
      else if (solver_id == 32)
      {
         HYPRE_StructSparseMSGDestroy(precond);
      }
      else if (solver_id == 37)
      {
         HYPRE_StructJacobiDestroy(precond);
      }
   }

   /*-----------------------------------------------------------
    * Print the solution and other info
    *-----------------------------------------------------------*/

#if 0
   HYPRE_StructVectorPrint("drive.out.x", x, 0);
#endif

   if (myid == 0)
   {
      printf("\n");
      printf("Iterations = %d\n", num_iterations);
      printf("Final Relative Residual Norm = %e\n", final_res_norm);
      printf("\n");
   }

#endif

   /*-----------------------------------------------------------
    * Compute MFLOPs for Matvec
    *-----------------------------------------------------------*/

#if HYPRE_MFLOPS
   {
      void *matvec_data;
      int   i, imax, N;

      /* compute imax */
      N = (P*nx)*(Q*ny)*(R*nz);
      imax = (5*1000000) / N;

      matvec_data = hypre_StructMatvecCreate();
      hypre_StructMatvecSetup(matvec_data, A, x);

      time_index = hypre_InitializeTiming("Matvec");
      hypre_BeginTiming(time_index);

      for (i = 0; i < imax; i++)
      {
         hypre_StructMatvecCompute(matvec_data, 1.0, A, x, 1.0, b);
      }
      /* this counts mult-adds */
      hypre_IncFLOPCount(7*N*imax);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Matvec time", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      hypre_StructMatvecDestroy(matvec_data);
   }
#endif

   /*-----------------------------------------------------------
    * Finalize things
    *-----------------------------------------------------------*/

   HYPRE_StructStencilDestroy(stencil);
   HYPRE_StructMatrixDestroy(A);
   HYPRE_StructVectorDestroy(b);
   HYPRE_StructVectorDestroy(x);

   for ( i = 0; i < (dim + 1); i++)
      hypre_TFree(offsets[i]);
   hypre_TFree(offsets);

   hypre_FinalizeMemoryDebug();

   /* Finalize MPI */
   MPI_Finalize();

#ifdef HYPRE_USE_PTHREADS
   HYPRE_DestroyPthreads();
#endif  

   return (0);
}
/*-------------------------------------------------------------------------
 * i can actually put any grid here and it will do it   
 *-------------------------------------------------------------------------*/

int
AddValuesVector( hypre_StructGrid  *gridvector,
                 hypre_StructVector *zvector,
                 int                *period, 
                 double             value  )
{
#include  "struct_mv.h"
 int ierr = 0;
 hypre_BoxArray     *gridboxes;
 int                i,ib;
 hypre_IndexRef     ilower;
 hypre_IndexRef     iupper;
 hypre_Box          *box;
 double             *values;
 int                volume,dim;

 gridboxes =  hypre_StructGridBoxes(gridvector);
 dim       =  hypre_StructGridDim(gridvector);

  ib=0;
  hypre_ForBoxI(ib, gridboxes)
       {
            box      = hypre_BoxArrayBox(gridboxes, ib);
            volume   =  hypre_BoxVolume(box);
	    values   = hypre_CTAlloc(double, volume);

	    if ((dim == 2 && period[0] != 0 && period[1] != 0) ||
               (dim == 3 && period[0] != 0 && period[1] != 0 && period[2] != 0))
            {
                for (i = 0; i < volume; i++)
                {
                   values[i] = 0.0;
                }
                   values[0]         =  value;
                  values[volume - 1] = -value;
            }
            else
            {
               for (i = 0; i < volume; i++)
               {
                  values[i] = value;
               }
            }

            ilower = hypre_BoxIMin(box);
	    iupper = hypre_BoxIMax(box);
            HYPRE_StructVectorSetBoxValues(zvector, ilower, iupper, values);
	    hypre_TFree(values);

       }

 return ierr;
}
/********************************/
/* now the addvalues to matrix  MATRIXMATRIX */
int
AddValuesMatrix(HYPRE_StructMatrix A,HYPRE_StructGrid gridmatrix,
                                     double            cx,
				     double            cy,
				     double            cz)
{

#include  "struct_mv.h"
  int ierr=0;
  hypre_BoxArray     *gridboxes;
 int                i,s,bi;
 hypre_IndexRef     ilower;
 hypre_IndexRef     iupper;
 hypre_Box          *box;
 double             *values;
 int                volume,dim;
  int               *stencil_indices;

 gridboxes =  hypre_StructGridBoxes(gridmatrix);
 dim       =  hypre_StructGridDim(gridmatrix);

  bi=0;
  hypre_ForBoxI(bi, gridboxes)
       {
            box      = hypre_BoxArrayBox(gridboxes, bi);
            volume   =  hypre_BoxVolume(box);
            values   = hypre_CTAlloc(double, (dim +1)*volume);
            stencil_indices = hypre_CTAlloc(int, (dim +1));

	    for (i = 0; i < (dim + 1)*volume; i += (dim + 1))
            {
               for (s = 0; s < (dim + 1); s++)
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
                        break;
                  }
               }
               
           }

           ilower = hypre_BoxIMin(box);
	   iupper = hypre_BoxIMax(box);
           HYPRE_StructMatrixSetBoxValues(A, ilower, iupper, (dim+1),
                                     stencil_indices, values);

	    hypre_TFree(values);
            hypre_TFree(stencil_indices);
       }

     return ierr;
}



/***************************ZEROING THE EXTREMES OF THE GRID *****************/

int
SetStencilBndry(HYPRE_StructMatrix A,HYPRE_StructGrid gridmatrix,int* period)
{
#include  "struct_mv.h"
  int ierr=0;
  hypre_BoxArray     *gridboxes;
  int                size,i,j,d,ib;
  int                **ilower;
  int                **iupper;
  int                 *vol;
  int                *istart;
  hypre_Box          *box;
  hypre_Box          *dummybox;
  hypre_Box          *boundingbox;
  double             *values;
  int                volume,dim;
  int               *stencil_indices;

  gridboxes =  hypre_StructGridBoxes(gridmatrix);
  boundingbox = hypre_StructGridBoundingBox(gridmatrix);
  istart      = hypre_BoxIMin(boundingbox);
  size      =  hypre_StructGridNumBoxes(gridmatrix);
  dim       =  hypre_StructGridDim(gridmatrix);
  stencil_indices = hypre_CTAlloc(int, (dim +1));

  vol    = hypre_CTAlloc(int, size);
  ilower = hypre_CTAlloc(int*, size);
  iupper = hypre_CTAlloc(int*, size);
   for (i = 0; i < size; i++)
   {
      ilower[i] = hypre_CTAlloc(int, dim);
      iupper[i] = hypre_CTAlloc(int, dim);
   }

  i = 0;
  ib = 0;
  hypre_ForBoxI(i, gridboxes)
       {
	    dummybox = hypre_BoxCreate( );
            box      = hypre_BoxArrayBox(gridboxes, i);
            volume   =  hypre_BoxVolume(box);
            vol[i]   = volume;
            hypre_CopyBox(box,dummybox);
            for (d = 0; d < dim; d++)
	    {
	      ilower[ib][d] = hypre_BoxIMinD(dummybox,d);
	      iupper[ib][d] = hypre_BoxIMaxD(dummybox,d);
            }
	    ib++ ;
            hypre_BoxDestroy(dummybox);
      	}

  for (d = 0; d < dim; d++)
  {
      for (ib = 0; ib < size; ib++)
      {
         values = hypre_CTAlloc(double, vol[ib]);
        
         for (i = 0; i < vol[ib]; i++)
	 {
	      values[i] = 0.0;
	 }

         if( ilower[ib][d] == istart[d] && period[d] == 0 )
         {
            j = iupper[ib][d];
            iupper[ib][d] = istart[d];
            stencil_indices[0] = d;
            HYPRE_StructMatrixSetBoxValues(A, ilower[ib], iupper[ib],
                                           1, stencil_indices, values);
            iupper[ib][d] = j;
         }
         hypre_TFree(values);
      }
  }
  
   hypre_TFree(vol);
   hypre_TFree(stencil_indices);
   for (ib =0 ; ib < size ; ib++)
   {
     hypre_TFree(ilower[ib]);
     hypre_TFree(iupper[ib]);
   }
   hypre_TFree(ilower);
   hypre_TFree(iupper);

  

   return ierr;
}
