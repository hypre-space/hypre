#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "utilities.h"
#include "HYPRE_ls.h"
#include "Hypre_Box.h"
#include "Hypre_StructStencil.h"
#include "Hypre_StructuredGrid.h"
#include "Hypre_StructMatrix.h"
#include "Hypre_StructVector.h"
#include "Hypre_MPI_Com.h"
#include "Hypre_StructJacobi.h"

Hypre_StructMatrix Hypre_StructMatrix_new();
/* ... without this, compiler thinks Hypre_StructMatrix_new returns
 an int.  If you #include "Hypre_StructMatrix_Skel.h" (which declares
 Hypre_StructMatrix_new), you get a bunch of errors about stencils.
 */

#ifdef HYPRE_DEBUG
#include <cegdb.h>
#endif

/*--------------------------------------------------------------------------
 * Test driver for structured matrix interface (structured storage)
 * based on slsbabel.c of Brent S. & ?
 *--------------------------------------------------------------------------*/

int
main( int argc, char *argv[] )
{
   Hypre_Box box;
   Hypre_StructStencil stencil;
   Hypre_MPI_Com comm;
   Hypre_StructuredGrid grid;
   Hypre_StructMatrix mat;
   Hypre_StructVector vecb, vecx;
   Hypre_StructJacobi solver;
   
   int resultCode, size, i, d, s, symmetric;
   int dim = 3;
   array1int lower;
   array1int upper;
   int ilower[3], iupper[3];
   int offsets[12];
   array1int arroffsets;
   array1int intvals;
   array1double doubvals;

   double cx = 1.0; /* diffusion coefficients */
   double cy = 1.0;
   double cz = 1.0;
   int stencil_indices[4];
#define MAT_SIZE 256000
   double matrix_values[MAT_SIZE];
#define VEC_SIZE 64000
   double vector_values[VEC_SIZE];
   int nx = 10;
   int ny = 10;
   int nz = 10;
   int volume = nx*ny*nz;
   int istart[3];  /* for grid boundaries */

   if ( volume*(dim+1)>=MAT_SIZE ) printf( "matrix dimensioned to small!\n" );
   if ( volume>=VEC_SIZE ) printf( "vector dimensioned to small!\n" );

   istart[0] = -17;
   istart[1] = 0;
   istart[2] = 32;

   /* Make and destroy a sample box ... */
   ilower[0] = 0;
   ilower[1] = 0;
   ilower[2] = 0;
   iupper[0] = 1;
   iupper[1] = 2;
   iupper[2] = 3;
   lower.data = ilower;
   upper.data = iupper;
   
   box = Hypre_Box_new();
   resultCode = Hypre_Box_NewBox( box, lower, upper, dim );
   printf( "NewBox result = %i\n", resultCode );
   Hypre_Box_print( box );

   Hypre_Box_destructor( box );

   /* Make a "real" box, for our grid ... */
   ilower[0] = istart[0];
   iupper[0] = istart[0]+ nx - 1;
   ilower[1] = istart[1];
   iupper[1] = istart[1]+ ny - 1;
   ilower[2] = istart[2];
   iupper[2] = istart[2]+ nz - 1;
   lower.data = ilower;
   upper.data = iupper;
   box = Hypre_Box_new();
   Hypre_Box_NewBox( box, lower, upper, dim );
   Hypre_Box_print( box );


   /* Make a stencil ... */
   /* First define the shape of the standard 7-point stencil in 3D:
      points are (0,0,0), (+-dx,0,0),(0,+-dy,0),(0,0,+-dz).
      But since we will be using a symmetric matrix, we only need to
      define the offsets for four of the stencil's points:
      (-dx,0,0),(0,-dy,0),(0,0,-dz),(0,0,0):
      */
   size = dim+1;
   offsets[0*3 +0] = -1; 
   offsets[0*3 +1] = 0; 
   offsets[0*3 +2] = 0; 
   offsets[1*3 +0] = 0; 
   offsets[1*3 +1] = -1; 
   offsets[1*3 +2] = 0; 
   offsets[2*3 +0] = 0; 
   offsets[2*3 +1] = 0; 
   offsets[2*3 +2] = -1; 
   offsets[3*3 +0] = 0; 
   offsets[3*3 +1] = 0; 
   offsets[3*3 +2] = 0; 
   
   /* Now set up the stencil object and intialize it */
   stencil = Hypre_StructStencil_new();
   Hypre_StructStencil_NewStencil( stencil, dim, size );
   for (i = 0; i<size; ++i)
   {
      arroffsets.data = &offsets[3*i];
      resultCode = Hypre_StructStencil_SetElement(stencil, i, &arroffsets);
      if ( resultCode!=0 ) printf( "StructStencil bad code=%i\n", resultCode );
   };
   Hypre_StructStencil_print( stencil );

   /* Make a MPI_Com object. */
   comm = Hypre_MPI_Com_new();
   /* With -DHYPRE_SEQUENTIAL, this function is typedefed to
      hypre_MPI_Init which exists nowhere:
      MPI_Init( &argc, &argv ); */

   /* Make a grid ... */

   grid = Hypre_StructuredGrid_new();
   Hypre_StructuredGrid_NewGrid( grid, comm, dim );
   Hypre_StructuredGrid_SetGridExtents( grid, box );
   /* probably makes sense only with MPI ... */
   Hypre_StructuredGrid_Assemble( grid );

   Hypre_StructuredGrid_print( grid );

   /* Make a matrix ... */

   mat = Hypre_StructMatrix_new();
   symmetric = 1;
   Hypre_StructMatrix_NewMatrix( mat, grid, stencil, symmetric );
   Hypre_StructMatrix_print( mat );

   /* set the matrix elements */

   /* Set the coefficients for the grid */
   for ( i=0; i<(dim+1)*volume; i+=(dim+1) )
   {
      for ( s=0; s<(dim+1); ++s )
      {
         stencil_indices[s] = s;
         matrix_values[i  ] = -cx;
         matrix_values[i+1] = -cy;
         matrix_values[i+2] = -cz;
         matrix_values[i+3] = 2.0*(cx+cy+cz);
      }
   }

   intvals.lower[0] = 0;
   intvals.upper[0] = dim+1;
   doubvals.lower[0] = 0;
   doubvals.upper[0] = (dim+1)*volume;
   intvals.data = stencil_indices;
   doubvals.data = matrix_values;
   Hypre_StructMatrix_SetValues( mat, box, intvals, doubvals );


   /* Zero out stencils reaching to real boundary; based on slsbabel.c */

   for ( i=0; i<volume; ++i )
   {
      matrix_values[i] = 0.0;
   };

   for ( d=0; d<dim; ++d )
   {
      if( ilower[d] == istart[d] )
      {
         i = iupper[d];
/*** UGGH. This changes box, a very very bad coding style, fix it soon ... ***/
         iupper[d] = istart[d];
         stencil_indices[0] = d;
         intvals.data = stencil_indices;
         doubvals.data = matrix_values;

         Hypre_StructMatrix_SetValues( mat, box, intvals, doubvals );

         iupper[d] = i;
      }
   }

   Hypre_StructMatrix_print( mat );

   /* Make two vectors */

   vecb = Hypre_StructVector_new();
   Hypre_StructVector_NewVector( vecb, grid );

   for ( i=0; i<volume; ++i )
   {
      vector_values[i] = 1.0;
   }

   doubvals.data = vector_values;
   Hypre_StructVector_SetValues( vecb, box, intvals, doubvals );

   Hypre_StructVector_print( vecb );

   vecx = Hypre_StructVector_new();
   Hypre_StructVector_NewVector( vecx, grid );

   for ( i=0; i<volume; ++i )
   {
      vector_values[i] = 0.0;
   }

   doubvals.data = vector_values;
   Hypre_StructVector_SetValues( vecx, box, intvals, doubvals );

   Hypre_StructVector_print( vecx );
   
   /* Make a linear solver */

   solver = Hypre_StructJacobi_new();
/* First call of Setup is to call HYPRE_StructJacobiCreate (args are
   solver and comm), which is needed before SetParameter.
   Second call of Setup is to call HYPRE_StructJacobiSetup, which apparantly
   should be called once parameters are set.
   This needs to be reorganized! */
   Hypre_StructJacobi_Setup( solver, mat, vecb, vecx, comm );
   Hypre_StructJacobi_SetParameter( solver, "tol", 1.0e-4 );
   Hypre_StructJacobi_SetParameter( solver, "max_iter", 40 );
   Hypre_StructJacobi_Setup( solver, mat, vecb, vecx, comm );
   Hypre_StructJacobi_Apply( solver, vecb, &vecx );

   Hypre_StructVector_print( vecx );

   return( 0 );
}
