 
#include "headers.h"
 
#ifdef HYPRE_DEBUG
#include <cegdb.h>
#endif

/*--------------------------------------------------------------------------
 * Driver to convert an input matrix file for the parallel smg code into
 * several input matrix files for the parallel smg code to use.
 *
 * Original Version:  12-16-97
 * Author: pnb
 * 12-19-97 pnb   cleaned up comments
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

   hypre_StructMatrix   *matrix_root, **sub_matrices;

   hypre_StructGrid     **sub_grids, *grid_root;
   hypre_BoxArray       *boxes, *boxes_2;
   hypre_Box            *box;
   int                 dim;

   hypre_StructStencil  *stencil, **stencils;
   hypre_Index          *stencil_shape, **stencil_shapes;
   hypre_Index           ilower, iupper;
   int                 stencil_size;

   hypre_BoxArray       *data_space, *data_space_2;

   FILE               *file, *file_root;
   int                 sub_i, sub_j, sub_k; 
                            /* Number of subdivisions to use for i,j,k */
   int                 num_files; /* Total number of files to write out */
   int                 i, j, k, ii, i_file, idummy;
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

#ifdef HYPRE_DEBUG
   cegdb(&argc, &argv, myid);
#endif

   hypre_InitMemoryDebug(myid);

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
   grid_root = hypre_ReadStructGrid(MPI_COMM_WORLD, file_root);

   /* read stencil info */
   fscanf(file_root, "\nStencil:\n");
   dim = hypre_StructGridDim(grid_root);
   fscanf(file_root, "%d\n", &stencil_size);
   stencil_shape = hypre_CTAlloc(hypre_Index, stencil_size);
   for (i = 0; i < stencil_size; i++)
     {
       fscanf(file_root, "%d: %d %d %d\n", &idummy,
	      &hypre_IndexX(stencil_shape[i]),
	      &hypre_IndexY(stencil_shape[i]),
	      &hypre_IndexZ(stencil_shape[i]));
     }
   stencil = hypre_NewStructStencil(dim, stencil_size, stencil_shape);

   /*----------------------------------------
    * Initialize the matrix
    *----------------------------------------*/

   matrix_root = hypre_NewStructMatrix(MPI_COMM_WORLD, grid_root, stencil);
   hypre_StructMatrixSymmetric(matrix_root) = symmetric;
   hypre_SetStructMatrixNumGhost(matrix_root, matrix_num_ghost);
   hypre_InitializeStructMatrix(matrix_root);

   /*----------------------------------------
    * Read data
    *----------------------------------------*/

   boxes      = hypre_StructGridBoxes(grid_root);
   data_space = hypre_StructMatrixDataSpace(matrix_root);
   num_values = hypre_StructMatrixNumValues(matrix_root);
 
   fscanf(file_root, "\nData:\n");
   hypre_ReadBoxArrayData(file_root, boxes, data_space, num_values,
                        hypre_StructMatrixData(matrix_root));
   hypre_AssembleStructMatrix(matrix_root, NULL, NULL, NULL);

   /*----------------------------------------
    * Close input file
    *----------------------------------------*/

   fclose(file_root);

   /* Write data read in to output file for testing */
   /* hypre_PrintStructMatrix("zout_matrix_test", matrix_root, 0); */

   /* Get min and max values for box in this file */
   /* Assume only one box in the input file for now */
   box = hypre_BoxArrayBox(boxes, 0);
   imin = hypre_BoxIMinX(box);
   jmin = hypre_BoxIMinY(box);
   kmin = hypre_BoxIMinZ(box);
 
   imax = hypre_BoxIMaxX(box);
   jmax = hypre_BoxIMaxY(box);
   kmax = hypre_BoxIMaxZ(box);

   sub_grids = hypre_CTAlloc(hypre_StructGrid *, num_files);
   sub_matrices = hypre_CTAlloc(hypre_StructMatrix *, num_files);
   stencils = hypre_CTAlloc(hypre_StructStencil *, num_files);
   stencil_shapes = hypre_CTAlloc(hypre_Index *, num_files);

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
	       sub_grids[i_file] = hypre_NewStructGrid(MPI_COMM_WORLD, dim);
	       stencil_shapes[i_file] = hypre_CTAlloc(hypre_Index, stencil_size);
	       for (ii = 0; ii < stencil_size; ii++)
		 {
		   hypre_IndexX(stencil_shapes[i_file][ii]) = 
		     hypre_IndexX(stencil_shape[ii]);
		   hypre_IndexY(stencil_shapes[i_file][ii]) = 
		     hypre_IndexY(stencil_shape[ii]);
		   hypre_IndexZ(stencil_shapes[i_file][ii]) = 
		     hypre_IndexZ(stencil_shape[ii]);
		 }
	       stencils[i_file] = hypre_NewStructStencil(dim, stencil_size,
						       stencil_shapes[i_file]);

	       hypre_IndexX(ilower) = imin + i*del_i;
	       hypre_IndexY(ilower) = jmin + j*del_j;
	       hypre_IndexZ(ilower) = kmin + k*del_k;
	       /* Coding to handle an odd number of points in i,j,k */
	       if (i+1 == sub_i)
		 {
		   hypre_IndexX(iupper) = imax;
		 }
	       else
		 {
		   hypre_IndexX(iupper) = imin + (i+1)*del_i - 1;
		 }
	       if (j+1 == sub_j)
		 {
		   hypre_IndexY(iupper) = jmax;
		 }
	       else
		 {
		   hypre_IndexY(iupper) = jmin + (j+1)*del_j - 1;
		 }
	       if (k+1 == sub_k)
		 {
		   hypre_IndexZ(iupper) = kmax;
		 }
	       else
		 {
		   hypre_IndexZ(iupper) = kmin + (k+1)*del_k - 1;
		 }

	       hypre_SetStructGridExtents(sub_grids[i_file], ilower, iupper);
	       hypre_AssembleStructGrid(sub_grids[i_file]);

	       sub_matrices[i_file] = 
		 hypre_NewStructMatrix(MPI_COMM_WORLD, sub_grids[i_file],
				     stencils[i_file]); 
	       hypre_StructMatrixSymmetric(sub_matrices[i_file]) = symmetric;
	       hypre_SetStructMatrixNumGhost(sub_matrices[i_file], 
					   matrix_num_ghost); 
	       hypre_InitializeStructMatrix(sub_matrices[i_file]);

	       /*----------------------------------------------
		* Load data from root matrix into sub_matrices
		*----------------------------------------------*/

	       boxes_2      = hypre_StructGridBoxes(sub_grids[i_file]);
	       data_space_2 = hypre_StructMatrixDataSpace(sub_matrices[i_file]);
	       num_values_2 = hypre_StructMatrixNumValues(sub_matrices[i_file]);
 
	       hypre_CopyBoxArrayData(boxes, data_space, num_values,
				    hypre_StructMatrixData(matrix_root),
				    boxes_2, data_space_2, num_values_2,
				    hypre_StructMatrixData(sub_matrices[i_file]));

	       hypre_AssembleStructMatrix(sub_matrices[i_file]);
	     }
	 }
     }
       

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

       /* write grid info */
       fprintf(file, "\nGrid:\n");
       hypre_PrintStructGrid(file, sub_grids[i_file]);

       /* write stencil info */
       fprintf(file, "\nStencil:\n");
       fprintf(file, "%d\n", stencil_size);
       for (i = 0; i < stencil_size; i++)
	 {
	   fprintf(file, "%d: %d %d %d\n", i,
		  hypre_IndexX(stencil_shape[i]),
		  hypre_IndexY(stencil_shape[i]),
		  hypre_IndexZ(stencil_shape[i]));
	 }

       /* write data info */
       fprintf(file, "\nData:\n");
       boxes_2      = hypre_StructGridBoxes(sub_grids[i_file]);
       data_space_2 = hypre_StructMatrixDataSpace(sub_matrices[i_file]);
       num_values_2 = hypre_StructMatrixNumValues(sub_matrices[i_file]);
       
       hypre_PrintBoxArrayData(file, boxes_2, data_space_2, num_values_2,
			     hypre_StructMatrixData(sub_matrices[i_file]));

       fclose(file);
     }


   /*-----------------------------------------------------------
    * Finalize things
    *-----------------------------------------------------------*/

   hypre_FreeStructGrid(hypre_StructMatrixGrid(matrix_root));
   hypre_FreeStructMatrix(matrix_root);
   for (i_file = 0; i_file < num_files; i_file++)
     {
       hypre_FreeStructGrid(hypre_StructMatrixGrid(sub_matrices[i_file]));
       hypre_FreeStructMatrix(sub_matrices[i_file]);
     }

   hypre_TFree(sub_grids);
   hypre_TFree(sub_matrices);
   hypre_TFree(stencils);
   hypre_TFree(stencil_shapes);

   hypre_FinalizeMemoryDebug();

   /* Finalize MPI */
   MPI_Finalize();

   return 0;
}

