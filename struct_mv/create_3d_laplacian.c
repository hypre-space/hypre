#include <stdio.h> 
 
/* debugging header */
#include <cegdb.h>

/* malloc debug stuff */
char malloc_logpath_memory[256];
 
/*--------------------------------------------------------------------------
 * Driver to create an input matrix file that one_to_many can then read to
 * create several input matrix files for the parallel smg code to use.
 *
 * This file generates a 3d Laplacian matrix file of arbitrary order.
 *
 * Original Version:  12-16-97
 * Author: pnb
 *
 *--------------------------------------------------------------------------*/
 
int
main( int   argc,
      char *argv[] )
{
   int                 dim;          /* dimension of the problem */
   int                 symmetric;    /* =1 for symmetric matrix storage */
   int                 nx, ny, nz;   /* Number of points in x and y */
   int                 stencil_size; /* size of the stencil */
   double              del_x, del_y, del_z; /* Delta x,y,z spacings */
   double              xlength, ylength, zlength; /* Lengths of x,y,z grids */
   double              data[7]; /* array to hold coefficients */
   int                 stencil_shape[4][3];
   FILE               *file;
   int                 ix, jy, kz, i, j;

   char  filename[256]; /* Filename */
                     
   if (argc > 7)
     {
       sprintf(filename, argv[1]);
       nx = atoi(argv[2]);
       ny = atoi(argv[3]);
       nz = atoi(argv[4]);
       xlength = atoi(argv[5]);
       ylength = atoi(argv[6]);
       zlength = atoi(argv[7]);
     }
   else
     {
       printf("Illegal input.\nUsage:\n\n");
       printf("  create_3d_laplacian filename nx ny nz xlength ylength zlength\n\n");
       printf("  where filename = output file containing matrix,\n");
       printf("        nx = number of pts in x direction,\n");
       printf("        ny = number of pts in y direction,\n");
       printf("        nz = number of pts in z direction,\n");
       printf("        xlength = total length in x, and\n");
       printf("        ylength = total length in y.\n");
       printf("        zlength = total length in z.\n\n");
       printf("The matrix generated is for the 3-d Laplacian on a\n");
       printf("(0,xlength) x (0,ylength) x (0,zlength) box using a 7-pt stencil\n");
       printf("with delta_x = xlength/nx, delta_y = ylength/ny, \n");
       printf("and delta_z = zlength/nz.\n");
       exit(1);
     }

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   /* set filename_root to zin_matrix for the moment */
   /* sprintf(filename, "three_d_laplacian_input"); */

   /* open file */
   if ((file = fopen(filename, "w")) == NULL)
     {
       printf("Error: can't open input file %s\n", filename);
       exit(1);
     }
   /*--------------------------------------------------------------
    * Load the matrix data.
    *--------------------------------------------------------------*/

   symmetric = 1;
   dim = 3;
   stencil_size = 4;

   stencil_shape[0][0]= 0;stencil_shape[0][1]= 0;stencil_shape[0][2]= 0;
   stencil_shape[1][0]=-1;stencil_shape[1][1]= 0;stencil_shape[1][2]= 0;
   stencil_shape[2][0]= 0;stencil_shape[2][1]= 1;stencil_shape[2][2]= 0;
   stencil_shape[3][0]= 0;stencil_shape[3][1]= 0;stencil_shape[3][2]= 1;

   del_x = xlength/nx;
   del_y = ylength/ny;
   del_z = zlength/nz;

   data[0] = 2.0*(1.0/(del_x*del_x) + 1.0/(del_y*del_y) + 1.0/(del_z*del_z));
   data[1] = -1.0/(del_x*del_x);
   data[2] = -1.0/(del_y*del_y);
   data[3] = -1.0/(del_z*del_z);

   /*--------------------------------------------------------------
    * Write out the matrix Symmetric, Grid and Stencil information.
    *--------------------------------------------------------------*/

   fprintf(file, "StructMatrix\n");

   fprintf(file, "\nSymmetric: %d\n", symmetric);

   /* write grid info */
   fprintf(file, "\nGrid:\n");
   fprintf(file, "%d\n", dim);
   fprintf(file, "%d\n", 1);
   fprintf(file, "0:  (%d, %d, %d)  x  (%d, %d, %d)\n",0,0,0,
	   nx-1,ny-1,nz-1);

   /* write stencil info */
   fprintf(file, "\nStencil:\n");
   fprintf(file, "%d\n", stencil_size);
   for (i = 0;  i < stencil_size; i++)
   {
      fprintf(file, "%d: %d %d %d\n", i,
             stencil_shape[i][0],stencil_shape[i][1],stencil_shape[i][2]);
   }

   /* write matrix data values */
   fprintf(file, "\nData:\n");
   for (kz = 0; kz < nz; kz++)
     {
       for (jy = 0; jy < ny; jy++)
	 {
	   for (ix = 0; ix < nx; ix++)
	     {
	       for (j = 0; j < stencil_size; j++)
		 {
		   fprintf(file, "0: (%d, %d, %d; %d) %e\n",
			   ix,jy,kz,j,data[j]);
		 }
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

   /* zzz_FreeStructGrid(zzz_StructVectorGrid(vector)); */
   /* zzz_FreeStructVector(vector); */
   /* zzz_FreeStructVector(tmp_vector); */

   /* malloc debug stuff */
   malloc_verify(0);
   malloc_shutdown();

   return 0;
}

