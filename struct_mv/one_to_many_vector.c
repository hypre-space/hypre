 
#include "headers.h"
 
#ifdef HYPRE_DEBUG
#include <cegdb.h>
#endif

/*--------------------------------------------------------------------------
 * Driver to convert an input vector file for the parallel smg code into 
 * several input vector files for the parallel smg code to use.
 *
 * Original Version:  12-16-97
 * Author: pnb
 * 12-19-97 pnb  commented out the temporary write of the input vector file.
 *
 * Assume only one box in the input file.
 *--------------------------------------------------------------------------*/
 
/*----------------------------------------------------------------------
 *    Read vector and vector files from disk.
 *----------------------------------------------------------------------*/
int
main( int   argc,
      char *argv[] )
{
   int                 vector_num_ghost[6] = { 0, 0, 0, 0, 0, 0};

   hypre_StructVector   *vector_root, **sub_vectors;

   hypre_StructGrid     **sub_grids, *grid_root;
   hypre_BoxArray       *boxes, *boxes_2;
   hypre_Box            *box;
   int                 dim;

   hypre_Index           ilower, iupper;

   hypre_BoxArray       *data_space, *data_space_2;

   FILE               *file, *file_root;
   int                 sub_i, sub_j, sub_k; 
                            /* Number of subdivisions to use for i,j,k */
   int                 num_files; /* Total number of files to write out */
   int                 i, j, k, i_file;
   int                 num_values, num_values_2;
   int                 imin, jmin, kmin, imax, jmax, kmax;
   int                 del_i, del_j, del_k;

   int                 myid, num_procs;

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

   /* set number of subdivisions to use in decomposing the vector into
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
       printf("     where filename = file containing vector to subdivide,\n");
       printf("           sub_i = number of subdivisions in i,\n");
       printf("           sub_j = number of subdivisions in j, and\n");
       printf("           sub_k = number of subdivisions in i.\n");
       printf("The number of files written is sub_i*sub_j*sub_k.\n");
       exit(1);
     }
   /* sub_i = 2; sub_j = 2; sub_k = 1; */
   /* set filename_root to zin_vector for the moment */
   /* sprintf(filename_root, "zin_vector_test"); */

   /* set number of files to write output to*/
   num_files = sub_i*sub_j*sub_k;
 
   /* open root file */
   if ((file_root = fopen(filename_root, "r")) == NULL)
     {
       printf("Error: can't open input file %s\n", filename_root);
       exit(1);
     }
   /*-----------------------------------------------------------
    * Read in the vector Grid information for the root file.
    *-----------------------------------------------------------*/

   fscanf(file_root, "StructVector\n");

   /* read grid info */
   fscanf(file_root, "\nGrid:\n");
   grid_root = hypre_ReadStructGrid(MPI_COMM_WORLD, file_root);
   dim = hypre_StructGridDim(grid_root);

   /*----------------------------------------
    * Initialize the vector
    *----------------------------------------*/

   vector_root = hypre_NewStructVector(MPI_COMM_WORLD, grid_root);
   hypre_SetStructVectorNumGhost(vector_root, vector_num_ghost);
   hypre_InitializeStructVector(vector_root);

   /*----------------------------------------
    * Read data
    *----------------------------------------*/

   boxes      = hypre_StructGridBoxes(grid_root);
   data_space = hypre_StructVectorDataSpace(vector_root);
   num_values = 1;
 
   fscanf(file_root, "\nData:\n");
   hypre_ReadBoxArrayData(file_root, boxes, data_space, num_values,
                        hypre_StructVectorData(vector_root));

   /*----------------------------------------
    * Close input file
    *----------------------------------------*/

   fclose(file_root);

   /* Write data read in to output file for testing */
   /* hypre_PrintStructVector("zout_vector_test", vector_root, 0); */

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
   sub_vectors = hypre_CTAlloc(hypre_StructVector *, num_files);

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

	       sub_vectors[i_file] = 
		 hypre_NewStructVector(MPI_COMM_WORLD, sub_grids[i_file]); 
	       hypre_SetStructVectorNumGhost(sub_vectors[i_file], 
					   vector_num_ghost); 
	       hypre_InitializeStructVector(sub_vectors[i_file]);

	       /*----------------------------------------------
		* Load data from root vector into sub_vectors
		*----------------------------------------------*/

	       boxes_2      = hypre_StructGridBoxes(sub_grids[i_file]);
	       data_space_2 = hypre_StructVectorDataSpace(sub_vectors[i_file]);
	       num_values_2 = 1;
 
	       hypre_CopyBoxArrayData(boxes, data_space, num_values,
				    hypre_StructVectorData(vector_root),
				    boxes_2, data_space_2, num_values_2,
				    hypre_StructVectorData(sub_vectors[i_file]));

	     }
	 }
     }
       

   /* Write the Grid information to the output files */
   for (i_file = 0; i_file < num_files; i_file++)
     {
       sprintf(filename, "%s.%05d", filename_root, i_file);
       if ((file = fopen(filename, "w")) == NULL)
	 {
	   printf("Error: can't open output file %s\n", filename);
	   exit(1);
	 }
       fprintf(file, "StructVector\n");

       /* write grid info */
       fprintf(file, "\nGrid:\n");
       hypre_PrintStructGrid(file, sub_grids[i_file]);

       /* write data info */
       fprintf(file, "\nData:\n");
       boxes_2      = hypre_StructGridBoxes(sub_grids[i_file]);
       data_space_2 = hypre_StructVectorDataSpace(sub_vectors[i_file]);
       num_values_2 = 1;
       
       hypre_PrintBoxArrayData(file, boxes_2, data_space_2, num_values_2,
			     hypre_StructVectorData(sub_vectors[i_file]));

       fclose(file);
     }


   /*-----------------------------------------------------------
    * Finalize things
    *-----------------------------------------------------------*/

   hypre_FreeStructGrid(hypre_StructVectorGrid(vector_root));
   hypre_FreeStructVector(vector_root);
   for (i_file = 0; i_file < num_files; i_file++)
     {
       hypre_FreeStructGrid(hypre_StructVectorGrid(sub_vectors[i_file]));
       hypre_FreeStructVector(sub_vectors[i_file]);
     }
   hypre_TFree(sub_grids);
   hypre_TFree(sub_vectors);

   hypre_FinalizeMemoryDebug();

   /* Finalize MPI */
   MPI_Finalize();

   return 0;
}

