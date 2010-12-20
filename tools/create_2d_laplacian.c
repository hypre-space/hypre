#include <stdio.h> 
 
/* debugging header */
#include <cegdb.h>

/* malloc debug stuff */
char malloc_logpath_memory[256];
 
/*--------------------------------------------------------------------------
 * Driver to create an input matrix file that one_to_many can then read to
 * create several input matrix files for the parallel smg code to use.
 *
 * This file generates a 2d Laplacian matrix file of arbitrary order.
 *
 * Original Version:  12-16-97
 * Author: pnb
 *
 *--------------------------------------------------------------------------*/
 
HYPRE_Int
main( HYPRE_Int   argc,
      char *argv[] )
{
   HYPRE_Int                 dim;          /* dimension of the problem */
   HYPRE_Int                 symmetric;    /* =1 for symmetric matrix storage */
   HYPRE_Int                 nx, ny;       /* Number of points in x and y */
   HYPRE_Int                 stencil_size; /* size of the stencil */
   double              del_x, del_y; /* Delta x and Delta y spacings */
   double              xlength, ylength; /* Lengths of x and y grids */
   double              data[3]; /* array to hold coefficients */
   HYPRE_Int                 stencil_shape[3][3];
   FILE               *file;
   HYPRE_Int                 ix, jy, i, j;

   char  filename[256]; /* Filename */
                     
   if (argc > 5)
     {
       hypre_sprintf(filename, argv[1]);
       nx = atoi(argv[2]);
       ny = atoi(argv[3]);
       xlength = atoi(argv[4]);
       ylength = atoi(argv[5]);
     }
   else
     {
       hypre_printf("Illegal input.\nUsage:\n\n");
       hypre_printf("     create_2d_laplacian filename nx ny xlength ylength\n\n");
       hypre_printf("     where filename = output file containing matrix,\n");
       hypre_printf("           nx = number of pts in x direction,\n");
       hypre_printf("           ny = number of pts in y direction,\n");
       hypre_printf("           xlength = total length in x, and\n");
       hypre_printf("           ylength = total length in y.\n\n");
       hypre_printf("The matrix generated is for the 2-d Laplacian on a\n");
       hypre_printf("(0,xlength) x (0,ylength) rectangle using a 5-pt stencil\n");
       hypre_printf("with delta_x = xlength/nx and delta_y = ylength/ny.\n");
       exit(1);
     }
   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   /* set filename_root to zin_matrix for the moment */
   /* hypre_sprintf(filename, "two_d_laplacian_input"); */

   /* open file */
   if ((file = fopen(filename, "w")) == NULL)
     {
       hypre_printf("Error: can't open input file %s\n", filename);
       exit(1);
     }
   /*--------------------------------------------------------------
    * Load the matrix data.
    *--------------------------------------------------------------*/

   symmetric = 1;
   dim = 2;
   stencil_size = 3;
   stencil_shape[0][0] = 0; stencil_shape[0][1] = 0; stencil_shape[0][2] = 0;
   stencil_shape[1][0] =-1; stencil_shape[1][1] = 0; stencil_shape[1][2] = 0;
   stencil_shape[2][0] = 0; stencil_shape[2][1] = 1; stencil_shape[2][2] = 0;

   del_x = xlength/nx;
   del_y = ylength/ny;

   data[0] = 2.0*(1.0/(del_x*del_x) + 1.0/(del_y*del_y));
   data[1] = -1.0/(del_x*del_x);
   data[2] = -1.0/(del_y*del_y);

   /*--------------------------------------------------------------
    * Write out the matrix Symmetric, Grid and Stencil information.
    *--------------------------------------------------------------*/

   hypre_fprintf(file, "StructMatrix\n");

   hypre_fprintf(file, "\nSymmetric: %d\n", symmetric);

   /* write grid info */
   hypre_fprintf(file, "\nGrid:\n");
   hypre_fprintf(file, "%d\n", dim);
   hypre_fprintf(file, "%d\n", 1);
   hypre_fprintf(file, "0:  (%d, %d, %d)  x  (%d, %d, %d)\n",0,0,0,
	   nx-1,ny-1,0);

   /* write stencil info */
   hypre_fprintf(file, "\nStencil:\n");
   hypre_fprintf(file, "%d\n", stencil_size);
   for (i = 0;  i < stencil_size; i++)
   {
      hypre_fprintf(file, "%d: %d %d %d\n", i,
             stencil_shape[i][0],stencil_shape[i][1],stencil_shape[i][2]);
   }

   /* write matrix data values */
   hypre_fprintf(file, "\nData:\n");
   for (jy = 0; jy < ny; jy++)
     {
       for (ix = 0; ix < nx; ix++)
	 {
	   for (j = 0; j < stencil_size; j++)
	     {
	       hypre_fprintf(file, "0: (%d, %d, 0; %d) %e\n",ix,jy,j,data[j]);
	     }
	 }
     }

   /*----------------------------------------
    * Close input file
    *----------------------------------------*/

   fclose(file);

   /*-----------------------------------------------------------
    * Finalize things
    *-----------------------------------------------------------*/

   /* hypre_FreeStructGrid(hypre_StructVectorGrid(vector)); */
   /* hypre_FreeStructVector(vector); */
   /* hypre_FreeStructVector(tmp_vector); */

   /* malloc debug stuff */
   malloc_verify(0);
   malloc_shutdown();

   return 0;
}

