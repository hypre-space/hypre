 
#include "headers.h"
 
/* debugging header */
#include <cegdb.h>

/* malloc debug stuff */
char malloc_logpath_memory[256];
 
/*--------------------------------------------------------------------------
 * Test driver for structured matrix interface (structured storage)
 *--------------------------------------------------------------------------*/
 
/*----------------------------------------------------------------------
 * Example 1: (nonsymmetric storage)
 *
 *    Standard 5-point laplacian in 2D on a 10 x 7 grid,
 *    ignoring boundary conditions for simplicity.
 *----------------------------------------------------------------------*/

int   main(argc, argv)
int   argc;
char *argv[];
{
   double              zero=0.0, one=1.0;
   int                 Nx = 8;
   int                 Ny = 6;
   int                 nx, ny;
   zzz_Index          *ilower;
   zzz_Index          *iupper;
   zzz_Index          *ilower2;
   zzz_Index          *iupper2;
   zzz_Box            *box;
   zzz_Box            *box2;
                     
   int                 dim = 2;
                     
   int                 offsets[5][2] = {{ 0,  0},
                                        {-1,  0},
                                        { 1,  0},
                                        { 0, -1},
                                        { 0,  1}};
                     
   double              coeffs[5] = { 4, -1, -1, -1, -1};
                     
   zzz_StructGrid     *grid;
   zzz_StructStencil  *stencil;
   zzz_Index         **stencil_shape;
   zzz_StructMatrix   *matrix;
   zzz_StructVector   *vector;
   zzz_StructVector   *tmp_vec0;
   zzz_StructVector   *tmp_vec1;

   double             *values;
   int                *stencil_indices;

   zzz_Index          *index;
   zzz_Index          *stride;
   int                 i, j, d;

   int                 num_procs, myid;
   int                 P, Q, p, q;

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

   /*-----------------------------------------------------------
    * Determine grid distribution
    *-----------------------------------------------------------*/

   ilower = zzz_NewIndex();
   iupper = zzz_NewIndex();

   P = Q = (int) sqrt(num_procs);
   p = myid % P;
   q = (myid - p) / P;
   nx = Nx / P;
   ny = Ny / Q;
   zzz_IndexD(ilower, 0) = nx*p;
   zzz_IndexD(ilower, 1) = ny*q;
   zzz_IndexD(iupper, 0) = zzz_IndexD(ilower, 0) + nx - 1;
   zzz_IndexD(iupper, 1) = zzz_IndexD(ilower, 1) + ny - 1;

   /*-----------------------------------------------------------
    * Set up the grid structure
    *-----------------------------------------------------------*/
 
   grid = zzz_NewStructGrid(MPI_COMM_WORLD, dim);
   zzz_SetStructGridExtents(grid, ilower, iupper);
   if ((p == 1) && (q == 0))
   {
      zzz_IndexD(ilower2, 0) = zzz_IndexD(ilower, 0) + nx;
      zzz_IndexD(ilower2, 1) = zzz_IndexD(ilower, 1);
      zzz_IndexD(iupper2, 0) = zzz_IndexD(iupper, 0) + nx;
      zzz_IndexD(iupper2, 1) = zzz_IndexD(iupper, 1);
      zzz_SetStructGridExtents(grid, ilower2, iupper2);
   }
   zzz_AssembleStructGrid(grid);

   /*-----------------------------------------------------------
    * Set up the stencil structure
    *-----------------------------------------------------------*/
 
   stencil_shape = ctalloc(zzz_Index *, 5);
   for (i = 0; i < 5; i++)
   {
      stencil_shape[i] = zzz_NewIndex();
      for (d = 0; d < dim; d++)
         zzz_IndexD(stencil_shape[i], d) = offsets[i][d];
   }
   stencil = zzz_NewStructStencil(dim, 5, stencil_shape);

   /*-----------------------------------------------------------
    * Set up the matrix structure
    *-----------------------------------------------------------*/
 
   matrix = zzz_NewStructMatrix(grid, stencil);
   zzz_InitializeStructMatrix(matrix);

   /*-----------------------------------------------------------
    * Fill in the matrix elements
    *-----------------------------------------------------------*/

   index = zzz_NewIndex();
   stride = zzz_NewIndex();
   for (d = 0; d < 3; d++)
      zzz_IndexD(stride, d) = 5;
   stencil_indices = ctalloc(int, 5);
   for (i = 0; i < 5; i++)
      stencil_indices[i] = i;

   box = zzz_NewBox(ilower, iupper);
   values = ctalloc(double, 5*zzz_BoxVolume(box));
   zzz_BoxLoop1(box, index,
                box, zzz_BoxIMin(box), stride, i,
                {
                   for (j = 0; j < 5; j++)
                      values[i+j] = coeffs[j];
                });
   zzz_SetStructMatrixBoxValues(matrix, box, 5, stencil_indices, values);
   tfree(values);

   if ((p == 1) && (q == 0))
   {
      box2 = zzz_NewBox(ilower2, iupper2);
      values = ctalloc(double, 5*zzz_BoxVolume(box2));
      zzz_BoxLoop1(box2, index,
                   box2, zzz_BoxIMin(box2), stride, i,
                   {
                      for (j = 0; j < 5; j++)
                         values[i+j] = coeffs[j];
                   });
      zzz_SetStructMatrixBoxValues(matrix, box2, 5, stencil_indices, values);
      tfree(values);
   }

   zzz_AssembleStructMatrix(matrix);

   /*-----------------------------------------------------------
    * Output information about the matrix
    *-----------------------------------------------------------*/

   zzz_PrintStructMatrix("zout_matrix", matrix, 0);

   /*-----------------------------------------------------------
    * Set up the vector structure
    *-----------------------------------------------------------*/
 
   vector = zzz_NewStructVector(grid);
   zzz_InitializeStructVector(vector);

   /*-----------------------------------------------------------
    * Fill in the vector elements
    *-----------------------------------------------------------*/
 
   for (d = 0; d < 3; d++)
      zzz_IndexD(stride, d) = 1;

   values = ctalloc(double, zzz_BoxVolume(box));
   zzz_BoxLoop1(box, index,
                box, zzz_BoxIMin(box), stride, i,
                {
                   values[i] = 1;
                });
   zzz_SetStructVectorBoxValues(vector, box, values);
   tfree(values);

   if ((p == 1) && (q == 0))
   {
      values = ctalloc(double, zzz_BoxVolume(box2));
      zzz_BoxLoop1(box2, index,
                   box2, zzz_BoxIMin(box2), stride, i,
                   {
                      values[i] = 1;
                   });
      zzz_SetStructVectorBoxValues(vector, box2, values);
      tfree(values);
   }

   zzz_AssembleStructVector(vector);

   /*-----------------------------------------------------------
    * Output information about the vector
    *-----------------------------------------------------------*/
 
   zzz_PrintStructVector("zout_vector", vector, 0);

   /*-----------------------------------------------------------
    * Finalize things
    *-----------------------------------------------------------*/

   zzz_FreeIndex(ilower);
   zzz_FreeIndex(iupper);
   zzz_FreeIndex(index);
   zzz_FreeIndex(stride);

   zzz_FreeStructStencil(stencil);

   tfree(stencil_indices);

   zzz_FreeStructMatrix(matrix);
   zzz_FreeStructVector(vector);
   zzz_FreeStructGrid(grid);

   /* malloc debug stuff */
   malloc_verify(0);
   malloc_shutdown();

   /* Finalize MPI */
   MPI_Finalize();
}

