 
#include "headers.h"
 
/* debugging header */
#include <cegdb.h>

/* malloc debug stuff */
char malloc_logpath_memory[256];
 
/*--------------------------------------------------------------------------
 * Driver to convert an input matrix file that matlab can read into 
 * several input matrix files for the parallel smg code to use.
 *
 * Original Version:  12-16-97
 * Author: pnb
 *
 * Assume only one box in the input file.
 *--------------------------------------------------------------------------*/
 
/*----------------------------------------------------------------------
 *    Read matrix and vector files from disk.
 *----------------------------------------------------------------------*/
int
main( int   argc,
      char *argv[] )
{
   int                 matrix_num_ghost[6] = { 0, 0, 0, 0, 0, 0};
   int                 vector_num_ghost[6] = { 0, 0, 0, 0, 0, 0};

   zzz_StructMatrix   *matrix_root, **sub_matrices;

   zzz_StructGrid     **sub_grids, *grid_root;
   zzz_BoxArray       *boxes, *boxes_2;
   zzz_Box            *box;
   int                 dim;

   zzz_StructStencil  *stencil;
   zzz_Index         **stencil_shape;
   zzz_Index           *ilower, *iupper;
   int                 stencil_size;

   zzz_BoxArray       *data_space, *data_space_2;

   FILE               *file, *file_root;
   int                 sub_i, sub_j, sub_k; 
                            /* Number of subdivisions to use for i,j,k */
   int                 num_files; /* Total number of files to write out */
   int                 i, j, k, i_file, idummy;
   int                 num_values, num_values_2;
   int                 imin, jmin, kmin, imax, jmax, kmax;
   int                 del_i, del_j, del_k;

   int                 myid, num_procs;
   int                 symmetric;

   char  filename_root[255]; /* Filename root */
   char  filename[256]    ;
                     
   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   /* Initialize MPI */
   MPI_Init(&argc, &argv);

   MPI_Comm_size(MPI_COMM_WORLD, &num_procs );
   MPI_Comm_rank(MPI_COMM_WORLD, &myid );

   cegdb(&argc, &argv, myid);

   /* malloc debug stuff */
   malloc_logpath = malloc_logpath_memory;
   sprintf(malloc_logpath, "malloc.log.%04d", myid);

   /* set number of subdivisions to use in decomposing the matrix into
      several different files */
   if (argc > 4)
     {
       sprintf(filename_root, argv[1]);
       sub_i = atoi(argv[2]);
       sub_j = atoi(argv[3]);
       sub_k = atoi(argv[4]);
     }
   else
     {
       printf("Illegal input.  Usage:\n\n");
       printf("     mpirun -np 1 one_to_many filename sub_i sub_j sub_k\n\n");
       printf("     where filename = file containing matrix to subdivide,\n");
       printf("           sub_i = number of subdivisions in i,\n");
       printf("           sub_j = number of subdivisions in j, and\n");
       printf("           sub_k = number of subdivisions in i.\n");
       printf("The number of files written is sub_i*sub_j*sub_k.\n");
       exit(1);
     }
   /* sub_i = 2; sub_j = 2; sub_k = 1; */
   /* set filename_root to zin_matrix for the moment */
   /* sprintf(filename_root, "zin_matrix_test"); */

   /* set number of files to write output to*/
   num_files = sub_i*sub_j*sub_k;
 
   /* open root file */
   if ((file_root = fopen(filename_root, "r")) == NULL)
     {
       printf("Error: can't open input file %s\n", filename_root);
       exit(1);
     }
   /*-----------------------------------------------------------
    * Read in the matrix Symmetric, Grid and Stencil information
    * for the root file.
    *-----------------------------------------------------------*/

   fscanf(file_root, "StructMatrix\n");

   fscanf(file_root, "\nSymmetric: %d\n", &symmetric);

   /* read grid info */
   fscanf(file_root, "\nGrid:\n");
   grid_root = zzz_ReadStructGrid(file_root);

   /* read stencil info */
   fscanf(file_root, "\nStencil:\n");
   dim = zzz_StructGridDim(grid_root);
   fscanf(file_root, "%d\n", &stencil_size);
   stencil_shape = zzz_CTAlloc(zzz_Index *, stencil_size);
   for (i = 0; i < stencil_size; i++)
     {
       stencil_shape[i] = zzz_NewIndex();
       fscanf(file_root, "%d: %d %d %d\n", &idummy,
	      &zzz_IndexX(stencil_shape[i]),
	      &zzz_IndexY(stencil_shape[i]),
	      &zzz_IndexZ(stencil_shape[i]));
     }
   stencil = zzz_NewStructStencil(dim, stencil_size, stencil_shape);

   /*----------------------------------------
    * Initialize the matrix
    *----------------------------------------*/

   matrix_root = zzz_NewStructMatrix(&MPI_COMM_WORLD, grid_root, stencil);
   zzz_StructMatrixSymmetric(matrix_root) = symmetric;
   zzz_SetStructMatrixNumGhost(matrix_root, matrix_num_ghost);
   zzz_InitializeStructMatrix(matrix_root);

   /*----------------------------------------
    * Read data
    *----------------------------------------*/

   boxes      = zzz_StructGridBoxes(grid_root);
   data_space = zzz_StructMatrixDataSpace(matrix_root);
   num_values = zzz_StructMatrixNumValues(matrix_root);
 
   fscanf(file_root, "\nData:\n");
   zzz_ReadBoxArrayData(file_root, boxes, data_space, num_values,
                        zzz_StructMatrixData(matrix_root));

   /*----------------------------------------
    * Close input file
    *----------------------------------------*/

   fclose(file_root);

   /* Write data read in to output file for testing */
   /* zzz_PrintStructMatrix("zout_matrix_test", matrix_root, 0); */

   /* Get min and max values for box in this file */
   /* Assume only one box in the input file for now */
   box = zzz_BoxArrayBox(boxes, 0);
   imin = zzz_BoxIMinX(box);
   jmin = zzz_BoxIMinY(box);
   kmin = zzz_BoxIMinZ(box);
 
   imax = zzz_BoxIMaxX(box);
   jmax = zzz_BoxIMaxY(box);
   kmax = zzz_BoxIMaxZ(box);

   sub_grids = zzz_CTAlloc(zzz_StructGrid *, num_files);
   sub_matrices = zzz_CTAlloc(zzz_StructMatrix *, num_files);
   ilower = zzz_NewIndex();
   iupper = zzz_NewIndex();

   del_i = (imax - imin + 1)/sub_i;
   del_j = (jmax - jmin + 1)/sub_j;
   del_k = (kmax - kmin + 1)/sub_k;
   if (del_i < 1)
     {
       printf("Error: too many subdivisions in i, sub_i = %d\n", sub_i);
       exit(1);
     }
   if (del_j < 1)
     {
       printf("Error: too many subdivisions in j, sub_j = %d\n", sub_j);
       exit(1);
     }
   if (del_k < 1)
     {
       printf("Error: too many subdivisions in k, sub_k = %d\n", sub_k);
       exit(1);
     }
   i_file = -1;
   for (k = 0; k < sub_k; k++)
     {
       for (j = 0; j < sub_j; j++)
	 {
	   for (i = 0; i < sub_i; i++)
	     {
	       i_file += 1;
	       sub_grids[i_file] = zzz_NewStructGrid(&MPI_COMM_WORLD, dim);

	       zzz_IndexX(ilower) = imin + i*del_i;
	       zzz_IndexY(ilower) = jmin + j*del_j;
	       zzz_IndexZ(ilower) = kmin + k*del_k;
	       /* Coding to handle an odd number of points in i,j,k */
	       if (i+1 == sub_i)
		 {
		   zzz_IndexX(iupper) = imax;
		 }
	       else
		 {
		   zzz_IndexX(iupper) = imin + (i+1)*del_i - 1;
		 }
	       if (j+1 == sub_j)
		 {
		   zzz_IndexY(iupper) = jmax;
		 }
	       else
		 {
		   zzz_IndexY(iupper) = jmin + (j+1)*del_j - 1;
		 }
	       if (k+1 == sub_k)
		 {
		   zzz_IndexZ(iupper) = kmax;
		 }
	       else
		 {
		   zzz_IndexZ(iupper) = kmin + (k+1)*del_k - 1;
		 }

	       zzz_SetStructGridExtents(sub_grids[i_file], ilower, iupper);
	       zzz_AssembleStructGrid(sub_grids[i_file]);

	       sub_matrices[i_file] = 
		 zzz_NewStructMatrix(&MPI_COMM_WORLD, 
				     sub_grids[i_file], stencil); 
	       zzz_StructMatrixSymmetric(sub_matrices[i_file]) = symmetric;
	       zzz_SetStructMatrixNumGhost(sub_matrices[i_file], 
					   matrix_num_ghost); 
	       zzz_InitializeStructMatrix(sub_matrices[i_file]);

	       /*----------------------------------------------
		* Load data from root matrix into sub_matrices
		*----------------------------------------------*/

	       boxes_2      = zzz_StructGridBoxes(sub_grids[i_file]);
	       data_space_2 = zzz_StructMatrixDataSpace(sub_matrices[i_file]);
	       num_values_2 = zzz_StructMatrixNumValues(sub_matrices[i_file]);
 
	       zzz_CopyBoxArrayData(boxes, data_space, num_values,
				    zzz_StructMatrixData(matrix_root),
				    boxes_2, data_space_2, num_values_2,
				    zzz_StructMatrixData(sub_matrices[i_file]));

	     }
	 }
     }
   zzz_FreeIndex(ilower);
   zzz_FreeIndex(iupper);
       

   /* Write the Symmtric, Grid and Stencil information to the output files */
   for (i_file = 0; i_file < num_files; i_file++)
     {
       sprintf(filename, "%s.%05d", filename_root, i_file);
       if ((file = fopen(filename, "w")) == NULL)
	 {
	   printf("Error: can't open output file %s\n", filename);
	   exit(1);
	 }
       fprintf(file, "StructMatrix\n");

       fprintf(file, "\nSymmetric: %d\n", symmetric);

       /* read grid info */
       fprintf(file, "\nGrid:\n");
       zzz_PrintStructGrid(file, sub_grids[i_file]);

       /* read stencil info */
       fprintf(file, "\nStencil:\n");
       fprintf(file, "%d\n", stencil_size);
       for (i = 0; i < stencil_size; i++)
	 {
	   fprintf(file, "%d: %d %d %d\n", i,
		  zzz_IndexX(stencil_shape[i]),
		  zzz_IndexY(stencil_shape[i]),
		  zzz_IndexZ(stencil_shape[i]));
	 }

       fprintf(file, "\nData:\n");
       boxes_2      = zzz_StructGridBoxes(sub_grids[i_file]);
       data_space_2 = zzz_StructMatrixDataSpace(sub_matrices[i_file]);
       num_values_2 = zzz_StructMatrixNumValues(sub_matrices[i_file]);
       
       zzz_PrintBoxArrayData(file, boxes_2, data_space_2, num_values_2,
			     zzz_StructMatrixData(sub_matrices[i_file]));

       fclose(file);
     }


   /*-----------------------------------------------------------
    * Finalize things
    *-----------------------------------------------------------*/

   zzz_FreeStructGrid(zzz_StructMatrixGrid(matrix_root));
   zzz_FreeStructStencil(zzz_StructMatrixUserStencil(matrix_root));
   zzz_FreeStructMatrix(matrix_root);
   /* zzz_FreeStructGrid(zzz_StructVectorGrid(vector)); */
   /* zzz_FreeStructVector(vector); */
   /* zzz_FreeStructVector(tmp_vector); */

   /* malloc debug stuff */
   malloc_verify(0);
   malloc_shutdown();

   /* Finalize MPI */
   MPI_Finalize();

   return 0;
}

