/*--------------------------------------------------------------------------
 * Test driver for unstructured matrix-vector interface.
 * Do `driver -help' for usage info.
 * This driver started from the driver for parcsr_linear_solvers, and it
 * works by first building a parcsr matrix as before and then "copying"
 * that matrix row-by-row into the IJMatrix interface.
 *--------------------------------------------------------------------------*/
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "utilities.h"
#include "HYPRE.h"
#include "HYPRE_parcsr_mv.h"

#include "HYPRE_IJ_mv.h"
#include "HYPRE_parcsr_ls.h"
#include "krylov.h"

int BuildParFromFile (int argc , char *argv [], int arg_index , HYPRE_ParCSRMatrix *A_ptr );
int BuildParLaplacian (int argc , char *argv [], int arg_index , HYPRE_ParCSRMatrix *A_ptr );
int BuildParDifConv (int argc , char *argv [], int arg_index , HYPRE_ParCSRMatrix *A_ptr );
int BuildParFromOneFile (int argc , char *argv [], int arg_index , HYPRE_ParCSRMatrix *A_ptr );
int BuildRhsParFromOneFile (int argc , char *argv [], int arg_index , int *partitioning , HYPRE_ParVector *b_ptr );
int BuildParLaplacian9pt (int argc , char *argv [], int arg_index , HYPRE_ParCSRMatrix *A_ptr );
int BuildParLaplacian27pt (int argc , char *argv [], int arg_index , HYPRE_ParCSRMatrix *A_ptr );

#define SECOND_TIME 0
 
int
main( int   argc,
      char *argv[] )
{
   int                 arg_index;
   int                 print_usage;
   int                 generate_matrix = 0;
   int                 build_matrix_type;
   int                 build_matrix_arg_index;
   int                 build_rhs_type;
   int                 build_rhs_arg_index;
   int                 build_funcs_type;
   int                 build_funcs_arg_index;
   int                 ioutdat;
   int                 debug_flag;
   int                 ierr,i,j,k; 
   int                 indx, rest, tms;
   double              max_row_sum = 1.0;
   double              norm;
   void		      *object;

   HYPRE_IJMatrix      ij_matrix; 
   HYPRE_IJVector      ij_b;
   HYPRE_IJVector      ij_x;
   HYPRE_IJVector      ij_v;

   /* concrete underlying type for ij_matrix defaults to parcsr. AJC. */
   /* int                 ij_matrix_object_type=HYPRE_PARCSR; */
   int                 ij_vector_object_type=HYPRE_PARCSR;

   HYPRE_ParCSRMatrix  parcsr_A;
   HYPRE_ParCSRMatrix  A;
   HYPRE_ParVector     b;
   HYPRE_ParVector     x;

   int                 num_procs, myid;
   int                 global_n;
   int                 local_row;
   int                *partitioning;
   int                *part_b;
   int                *part_x;
   int                *indices;
   int                *row;
   int                *row_sizes;
   int                *diag_sizes;
   int                *offdiag_sizes;

   int		       time_index;
   MPI_Comm comm;
   int M, N;
   int first_local_row, last_local_row;
   int first_local_col, last_local_col;
   int size, *col_ind;
   int local_num_vars;
   double *values;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   /* Initialize MPI */
   MPI_Init(&argc, &argv);

   MPI_Comm_size(MPI_COMM_WORLD, &num_procs );
   MPI_Comm_rank(MPI_COMM_WORLD, &myid );
/*
   hypre_InitMemoryDebug(myid);
*/
   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/
 
   build_matrix_type      = 1;
   build_matrix_arg_index = argc;
   build_rhs_type = 0;
   build_rhs_arg_index = argc;
   build_funcs_type = 0;
   build_funcs_arg_index = argc;
   debug_flag = 0;

   ioutdat = 1;

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/
 
   print_usage = 0;
   arg_index = 1;

   while ( (arg_index < argc) && (!print_usage) )
   {
      if ( strcmp(argv[arg_index], "-fromfile") == 0 )
      {
         arg_index++;
         build_matrix_type      = 0;
         build_matrix_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-fromonefile") == 0 )
      {
         arg_index++;
         build_matrix_type      = 2;
         build_matrix_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-funcsfromonefile") == 0 )
      {
         arg_index++;
         build_funcs_type      = 1;
         build_funcs_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-funcsfromfile") == 0 )
      {
         arg_index++;
         build_funcs_type      = 2;
         build_funcs_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-laplacian") == 0 )
      {
         arg_index++;
         build_matrix_type      = 1;
         build_matrix_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-9pt") == 0 )
      {
         arg_index++;
         build_matrix_type      = 3;
         build_matrix_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-27pt") == 0 )
      {
         arg_index++;
         build_matrix_type      = 4;
         build_matrix_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-difconv") == 0 )
      {
         arg_index++;
         build_matrix_type      = 5;
         build_matrix_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-exact_size") == 0 )
      {
         arg_index++;
         generate_matrix = 1;
      }
      else if ( strcmp(argv[arg_index], "-storage_low") == 0 )
      {
         arg_index++;
         generate_matrix = 2;
      }
      else if ( strcmp(argv[arg_index], "-concrete_parcsr") == 0 )
      {
         arg_index++;
         /* ij_matrix_object_type      = HYPRE_PARCSR; */
         build_matrix_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-rhsfromfile") == 0 )
      {
         arg_index++;
         build_rhs_type      = 1;
         build_rhs_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-rhsfromonefile") == 0 )
      {
         arg_index++;
         build_rhs_type      = 2;
         build_rhs_arg_index = arg_index;
      }      
      else if ( strcmp(argv[arg_index], "-rhsrand") == 0 )
      {
         arg_index++;
         build_rhs_type      = 3;
         build_rhs_arg_index = arg_index;
      }    
      else if ( strcmp(argv[arg_index], "-rhszero") == 0 )
      {
         arg_index++;
         build_rhs_type      = 5;
         build_rhs_arg_index = arg_index;
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

   arg_index = 0;
   while (arg_index < argc)
   {
      if ( strcmp(argv[arg_index], "-mxrs") == 0 )
      {
         arg_index++;
         max_row_sum  = atof(argv[arg_index++]);
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
      printf("  -fromfile <filename>   : problem defining matrix from distributed file\n");
      printf("  -fromonefile <filename>: problem defining matrix from standard CSR file\n");
      printf("\n");
      printf("  -laplacian [<options>] : build laplacian problem\n");
      printf("  -9pt [<opts>] : build 9pt 2D laplacian problem\n");
      printf("  -27pt [<opts>] : build 27pt 3D laplacian problem\n");
      printf("  -difconv [<opts>]      : build convection-diffusion problem\n");
      printf("    -n <nx> <ny> <nz>    : total problem size \n");
      printf("    -P <Px> <Py> <Pz>    : processor topology\n");
      printf("    -c <cx> <cy> <cz>    : diffusion coefficients\n");
      printf("    -a <ax> <ay> <az>    : convection coefficients\n");
      printf("\n");
      printf("   -exact_size           : inserts immediately into ParCSR structure\n");
      printf("   -storage_low          : allocates not enough storage for aux struct\n");
      printf("   -concrete_parcsr      : use parcsr matrix type as concrete type\n");
      printf("\n");
      printf("   -rhsfromfile          : from distributed file (NOT YET)\n");
      printf("   -rhsfromonefile       : from vector file \n");
      printf("   -rhsrand              : rhs is random vector\n");
      printf("\n");
      printf("  -iout <val>            : set output flag\n");
      printf("       0=no output    1=matrix stats\n"); 
      printf("\n");  
      printf("  -dbg <val>             : set debug flag\n");
      printf("       0=no debugging\n       1=internal timing\n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/
 
   if (myid == 0)
   {
      printf("Running with these driver parameters:\n");
   }

   /*-----------------------------------------------------------
    * Set up matrix
    *-----------------------------------------------------------*/

   time_index = hypre_InitializeTiming("IJ Interface");
   hypre_BeginTiming(time_index);

   if ( build_matrix_type == 0 )
   {
      BuildParFromFile(argc, argv, build_matrix_arg_index, &parcsr_A);
   }
   else if ( build_matrix_type == 1 )
   {
      BuildParLaplacian(argc, argv, build_matrix_arg_index, &parcsr_A);
   }
   else if ( build_matrix_type == 2 )
   {
      BuildParFromOneFile(argc, argv, build_matrix_arg_index, &parcsr_A);
   }
   else if ( build_matrix_type == 3 )
   {
      BuildParLaplacian9pt(argc, argv, build_matrix_arg_index, &parcsr_A);
   }
   else if ( build_matrix_type == 4 )
   {
      BuildParLaplacian27pt(argc, argv, build_matrix_arg_index, &parcsr_A);
   }
   else if ( build_matrix_type == 5 )
   {
      BuildParDifConv(argc, argv, build_matrix_arg_index, &parcsr_A);
   }
   else
   {
      printf("You have asked for an unsupported problem, problem = %d.\n", 
		build_matrix_type);
      return(-1);
   }

    
   /*-----------------------------------------------------------
    * Copy the parcsr matrix into the IJMatrix through interface calls
    *-----------------------------------------------------------*/

   ierr = HYPRE_ParCSRMatrixGetComm( parcsr_A, &comm );
   ierr = HYPRE_ParCSRMatrixGetLocalRange( parcsr_A,
             &first_local_row, &last_local_row ,
             &first_local_col, &last_local_col );

   ierr += HYPRE_ParCSRMatrixGetDims( parcsr_A, &M, &N );

   ierr += HYPRE_IJMatrixCreate( comm, first_local_row, last_local_row,
                                       first_local_col, last_local_col,
                                       &ij_matrix );

   ierr += HYPRE_IJMatrixSetObjectType( ij_matrix, HYPRE_PARCSR );
/* the following shows how to build an ij_matrix if one has only an
   estimate for the row sizes */
   if (generate_matrix == 1)
   {   
/*  build ij_matrix using exact row_sizes for diag and offdiag */

      diag_sizes = hypre_CTAlloc(int, last_local_row - first_local_row + 1);
      offdiag_sizes = hypre_CTAlloc(int, last_local_row - first_local_row + 1);
      local_row = 0;
      for (i=first_local_row; i<= last_local_row; i++)
      {
         ierr += HYPRE_ParCSRMatrixGetRow( parcsr_A, i, &size, 
		&col_ind, &values );

         for (j=0; j < size; j++)
         {
	 if (col_ind[j] < first_local_row || col_ind[j] > last_local_row)
	       offdiag_sizes[local_row]++;
	 else
	       diag_sizes[local_row]++;
         }
         local_row++;
         ierr += HYPRE_ParCSRMatrixRestoreRow( parcsr_A, i, &size, 
		&col_ind, &values );
      }
      ierr += HYPRE_IJMatrixSetDiagOffdSizes ( ij_matrix, 
					(const int *) diag_sizes,
					(const int *) offdiag_sizes );
      hypre_TFree(diag_sizes);
      hypre_TFree(offdiag_sizes);
      
      ierr = HYPRE_IJMatrixInitialize( ij_matrix );
      row = hypre_CTAlloc(int,1);
      
      for (i=first_local_row; i<= last_local_row; i++)
      {
         ierr += HYPRE_ParCSRMatrixGetRow( parcsr_A, i, &size, 
		&col_ind, &values );

	 row[0] = i;
         ierr += HYPRE_IJMatrixSetValues(ij_matrix, 1, &size,
                                (const int *) row,
                                (const int *) col_ind,
                                (const double *) values );

         ierr += HYPRE_ParCSRMatrixRestoreRow( parcsr_A, i, &size, 
		&col_ind, &values );

      }
      hypre_TFree(row);
      ierr += HYPRE_IJMatrixAssemble( ij_matrix );
   }
   else
   {
      row_sizes = hypre_CTAlloc(int, last_local_row - first_local_row + 1);

      size = 5; /* this is in general too low, and supposed to test
		   the capability of the reallocation of the interface */ 
      if (generate_matrix == 0) /* tries a more accurate estimate of the
				   storage */
      {
	 if (build_matrix_type == 1) size = 7;
	 if (build_matrix_type == 3) size = 9;
	 if (build_matrix_type == 4) size = 27;
      }

      for (i=0; i < last_local_row - first_local_row + 1; i++)
         row_sizes[i] = size;

      ierr = HYPRE_IJMatrixSetRowSizes ( ij_matrix, (const int *) row_sizes );

      hypre_TFree(row_sizes);

      ierr = HYPRE_IJMatrixInitialize( ij_matrix );
      row = hypre_CTAlloc(int,1);

      /* Loop through all locally stored rows and insert them into ij_matrix */
      for (i=first_local_row; i<= last_local_row; i++)
      {
         ierr += HYPRE_ParCSRMatrixGetRow( parcsr_A, i, &size, 
		&col_ind, &values );

	 row[0] = i;
         ierr += HYPRE_IJMatrixSetValues(ij_matrix, 1, &size,
                                (const int *) row,
                                (const int *) col_ind,
                                (const double *) values );

         ierr += HYPRE_ParCSRMatrixRestoreRow( parcsr_A, i, &size, 
		&col_ind, &values );
      }
      hypre_TFree(row);
      ierr += HYPRE_IJMatrixAssemble( ij_matrix );
   }
   if (ierr)
   {
       printf("Error in driver building IJMatrix from parcsr matrix. \n");
       return(-1);
   }

   /*-----------------------------------------------------------
    * Fetch the resulting underlying matrix out
    *-----------------------------------------------------------*/

    ierr = HYPRE_IJMatrixGetObject( ij_matrix, &object);
    A = (HYPRE_ParCSRMatrix) object;

#if 0
    /* compare the two matrices that should be the same */
    HYPRE_ParCSRMatrixPrint(parcsr_A, "driver.out.parcsr_A");
    HYPRE_ParCSRMatrixPrint(A, "driver.out.A");
#endif

    HYPRE_ParCSRMatrixDestroy(parcsr_A);
   /*-----------------------------------------------------------
    * Set up the RHS and initial guess
    *-----------------------------------------------------------*/

   HYPRE_ParCSRMatrixGetRowPartitioning(A, &partitioning); 
   HYPRE_ParCSRMatrixGetDims(A, &global_n, &global_n);

   part_b = hypre_CTAlloc(int, num_procs+1);
   part_x = hypre_CTAlloc(int, num_procs+1);
   for (i=0; i < num_procs+1; i++)
   {
      part_b[i] = partitioning[i];
      part_x[i] = partitioning[i];
   }
   hypre_EndTiming(time_index);


   HYPRE_IJVectorCreate(MPI_COMM_WORLD, part_x[myid],
                        part_x[myid+1]-1, &ij_v);
   HYPRE_IJVectorSetObjectType(ij_v, ij_vector_object_type );
   HYPRE_IJVectorInitialize(ij_v);
   /* hypre_IJVectorZeroValues(ij_v);  */

   values  = hypre_CTAlloc(double, part_x[myid+1] - part_x[myid]);

  /*-------------------------------------------------------------------
   * Check HYPRE_IJVectorSet(Get)Values calls
   *
   * All local components changed -- NULL indices
   *-------------------------------------------------------------------*/

   for (i = 0; i < part_x[myid+1] - part_x[myid]; i++)
     values[i] = 1.;

   HYPRE_IJVectorSetValues(ij_v, part_x[myid+1] - part_x[myid],
                           NULL, values);

   for (i = 0; i < part_x[myid+1] - part_x[myid]; i++)
     values[i] = (double)i;

   HYPRE_IJVectorAddToValues(ij_v, (part_x[myid+1] - part_x[myid])/2,
                             NULL, values);

   HYPRE_IJVectorGetValues(ij_v, part_x[myid+1] - part_x[myid],
                           NULL, values);

   ierr = 0;
   for (i = 0; i < (part_x[myid+1]-part_x[myid])/2; i++)
     if (values[i] != (double)i + 1.) ++ierr;
   for (i = (part_x[myid+1]-part_x[myid])/2; i < part_x[myid+1]-part_x[myid]; i++)
     if (values[i] != 1.) ++ierr;
   if (ierr)
   {
     printf("One of HYPRE_IJVectorSet(AddTo,Get)Values\n");
     printf("calls with NULL indices bad\n");
     printf("IJVector Error 1 with ierr = %d\n", ierr);
     exit(1);
   }

  /*-------------------------------------------------------------------
   * All local components changed, assigned reverse-ordered values
   *   as specified by indices
   *-------------------------------------------------------------------*/

   /* hypre_IJVectorZeroValues(ij_v);  */

   indices = hypre_CTAlloc(int, part_x[myid+1] - part_x[myid]);

   for (i = 0; i < part_x[myid+1] - part_x[myid]; i++)
   {
     values[i] = (double)i;
     indices[i] = part_x[myid+1] - 1 - i;
   }

   HYPRE_IJVectorSetValues(ij_v, part_x[myid+1] - part_x[myid],
                           indices, values);

   for (i = 0; i < part_x[myid+1] - part_x[myid]; i++)
     values[i] = (double)i*i;

   HYPRE_IJVectorAddToValues(ij_v, part_x[myid+1] - part_x[myid],
                             indices, values);

   HYPRE_IJVectorGetValues(ij_v, part_x[myid+1] - part_x[myid],
                           indices, values);

   hypre_TFree(indices);

   ierr = 0;
   for (i = 0; i < part_x[myid+1] - part_x[myid]; i++)
     if (values[i] != (double)(i*i + i)) ++ierr;
   if (ierr)
   {
     printf("One of HYPRE_IJVectorSet(Get)Values\n");
     printf("calls bad\n");
     printf("IJVector Error 2 with ierr = %d\n", ierr);
     exit(1);
   }

   hypre_BeginTiming(time_index);

   if ( build_rhs_type == 0 )
   {
      HYPRE_ParVectorCreate(MPI_COMM_WORLD, global_n, part_b, &b);
      HYPRE_ParVectorInitialize(b);
      HYPRE_ParVectorSetConstantValues(b, 1.0);

      HYPRE_IJVectorCreate(MPI_COMM_WORLD, part_x[myid], part_x[myid+1]-1, &ij_x);
      HYPRE_IJVectorSetObjectType(ij_x,ij_vector_object_type );
      HYPRE_IJVectorInitialize(ij_x);
      hypre_IJVectorZeroValues(ij_x);
      ierr = HYPRE_IJVectorGetObject( ij_x, &object );
      x = (HYPRE_ParVector) object;
   }
   else if (build_rhs_type == 1)
   {
      /* BuildRHSParFromFile(argc, argv, build_rhs_arg_index, &b); */
      printf("Rhs from file not yet implemented.  Defaults to b=0\n");
      HYPRE_IJVectorCreate(MPI_COMM_WORLD, part_x[myid],
                           part_x[myid+1]-1, &ij_b);
      HYPRE_IJVectorSetObjectType(ij_b,ij_vector_object_type );
      HYPRE_IJVectorInitialize(ij_b);
      /* hypre_IJVectorZeroValues(ij_b);  */

      HYPRE_IJVectorCreate(MPI_COMM_WORLD, part_x[myid],
                           part_x[myid+1]-1, &ij_x);
      HYPRE_IJVectorSetObjectType(ij_x,ij_vector_object_type );
      HYPRE_IJVectorInitialize(ij_x);
      /* hypre_IJVectorZeroValues(ij_x);  */

      values = hypre_CTAlloc(double, part_x[myid+1] - part_x[myid]);

      for (i = 0; i < part_x[myid+1] - part_x[myid]; i++)
         values[i] = 1.0;

      HYPRE_IJVectorSetValues(ij_x, part_x[myid+1]-part_x[myid], 
                              NULL, values);
      hypre_TFree(values);

   /*-----------------------------------------------------------
    * Fetch the resulting underlying vectors out
    *-----------------------------------------------------------*/

      ierr = HYPRE_IJVectorGetObject( ij_b, &object );
      b = (HYPRE_ParVector) object;
      ierr = HYPRE_IJVectorGetObject( ij_x, &object );
      x = (HYPRE_ParVector) object;

   }
   else if ( build_rhs_type == 2 )
   {
      BuildRhsParFromOneFile(argc, argv, build_rhs_arg_index, part_b, &b);

      HYPRE_IJVectorCreate(MPI_COMM_WORLD, part_x[myid],
                           part_x[myid+1]-1, &ij_x);
      HYPRE_IJVectorSetObjectType(ij_x,ij_vector_object_type );
      HYPRE_IJVectorInitialize(ij_x);
      /* hypre_IJVectorZeroValues(ij_x);  */
      ierr = HYPRE_IJVectorGetObject( ij_x, &object );
      x = (HYPRE_ParVector) object;

   }
   else if ( build_rhs_type == 3 )
   {

      HYPRE_ParVectorCreate(MPI_COMM_WORLD, global_n, part_b,&b);
      HYPRE_ParVectorInitialize(b);
      HYPRE_ParVectorSetRandomValues(b, 22775);
      HYPRE_ParVectorInnerProd(b,b,&norm);
      norm = 1.0/sqrt(norm);
      ierr = HYPRE_ParVectorScale(norm, b);      

      HYPRE_IJVectorCreate(MPI_COMM_WORLD, part_x[myid],
                           part_x[myid+1]-1, &ij_x);
      HYPRE_IJVectorSetObjectType(ij_x,ij_vector_object_type );
      HYPRE_IJVectorInitialize(ij_x);
      ierr = HYPRE_IJVectorGetObject( ij_x, &object );
      x = (HYPRE_ParVector) object;
   }
   else if ( build_rhs_type == 5 )
   {

      HYPRE_ParVectorCreate(MPI_COMM_WORLD, global_n, part_x, &x);
      HYPRE_ParVectorInitialize(x);
      HYPRE_ParVectorSetConstantValues(x, 1.0);

      HYPRE_ParVectorCreate(MPI_COMM_WORLD, global_n, part_b, &b);
      HYPRE_ParVectorInitialize(b);
      HYPRE_ParVectorSetConstantValues(b, 0.0);
   }
   else /* if ( build_rhs_type == 0 ) */
   {
      HYPRE_ParVectorCreate(MPI_COMM_WORLD, global_n, part_b, &b);
      HYPRE_ParVectorInitialize(b);
      HYPRE_ParVectorSetConstantValues(b, 1.0);

      HYPRE_IJVectorCreate(MPI_COMM_WORLD, part_x[myid],
                           part_x[myid+1]-1, &ij_x);
      HYPRE_IJVectorSetObjectType(ij_x,ij_vector_object_type );
      HYPRE_IJVectorInitialize(ij_x);
      ierr = HYPRE_IJVectorGetObject( ij_x, &object );
      x = (HYPRE_ParVector) object;
   }

   hypre_EndTiming(time_index);
   hypre_PrintTiming("IJ Interface", MPI_COMM_WORLD);
   hypre_FinalizeTiming(time_index);
   hypre_ClearTiming();
   
   /*-----------------------------------------------------------
    * Print the solution and other info
    *-----------------------------------------------------------*/

#if 0
   HYPRE_PrintCSRVector(x, "driver.out.x");
#endif

   /*-----------------------------------------------------------
    * Finalize things
    *-----------------------------------------------------------*/

   HYPRE_IJVectorDestroy(ij_v);

/*--------------------------------------------------------------
 * Partitionings are destroyed by previous Destroy calls (not good?)
 * hypre_TFree(part_x);
 * hypre_TFree(part_b);
 *--------------------------------------------------------------*/

   HYPRE_IJMatrixDestroy(ij_matrix);
   if (build_rhs_type == 1)
      HYPRE_IJVectorDestroy(ij_b);
   else
      HYPRE_ParVectorDestroy(b);
   if (build_rhs_type > -1 && build_rhs_type < 4)
      HYPRE_IJVectorDestroy(ij_x);
   else
      HYPRE_ParVectorDestroy(x);
/*
   hypre_FinalizeMemoryDebug();
*/
   /* Finalize MPI */
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

   cx = 1.0;
   cy = 1.0;
   cz = 1.0;

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
      printf("    (cx, cy, cz) = (%f, %f, %f)\n", cx, cy, cz);
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

   values[0] = 0.0;
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

   hinx = 1.0/(nx+1);
   hiny = 1.0/(ny+1);
   hinz = 1.0/(nz+1);

   P  = 1;
   Q  = num_procs;
   R  = 1;

   cx = 1.0;
   cy = 1.0;
   cz = 1.0;

   ax = 1.0;
   ay = 1.0;
   az = 1.0;

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
      printf("    (ax, ay, az) = (%f, %f, %f)\n", ax, ay, az);
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

   values[0] = 0.0;
   if (nx > 1)
   {
      values[0] += 2.0*cx/(hinx*hinx) - 1.0*ax/hinx;
   }
   if (ny > 1)
   {
      values[0] += 2.0*cy/(hiny*hiny) - 1.0*ay/hiny;
   }
   if (nz > 1)
   {
      values[0] += 2.0*cz/(hinz*hinz) - 1.0*az/hinz;
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
 * Build Rhs from one file on Proc. 0. Distributes vector across processors 
 * giving each about using the distribution of the matrix A.
 *----------------------------------------------------------------------*/

int
BuildRhsParFromOneFile( int                  argc,
                        char                *argv[],
                        int                  arg_index,
                        int		    *partitioning,
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
      printf("    (Px, Py) = (%d, %d)\n", P,  Q);
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

   values[1] = -1.0;

   values[0] = 0.0;
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
      printf("    (Px, Py, Pz) = (%d, %d, %d)\n", P,  Q,  R);
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
   values[1] = -1.0;

   A = (HYPRE_ParCSRMatrix) GenerateLaplacian27pt(MPI_COMM_WORLD,
                               nx, ny, nz, P, Q, R, p, q, r, values);

   hypre_TFree(values);

   *A_ptr = A;

   return (0);
}
