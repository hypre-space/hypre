/*BHEADER**********************************************************************
 * (c) 1997   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * ZZZ_StructMatrix interface
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * ZZZ_NewStructMatrix
 *--------------------------------------------------------------------------*/

ZZZ_StructMatrix 
ZZZ_NewStructMatrix( MPI_Comm          *comm,
                     ZZZ_StructGrid     grid,
                     ZZZ_StructStencil  stencil )
{
   return ( (ZZZ_StructMatrix)
            zzz_NewStructMatrix( comm,
                                 (zzz_StructGrid *) grid,
                                 (zzz_StructStencil *) stencil ) );
}

/*--------------------------------------------------------------------------
 * ZZZ_FreeStructMatrix
 *--------------------------------------------------------------------------*/

int 
ZZZ_FreeStructMatrix( ZZZ_StructMatrix matrix )
{
   return( zzz_FreeStructMatrix( (zzz_StructMatrix *) matrix ) );
}

/*--------------------------------------------------------------------------
 * ZZZ_InitializeStructMatrix
 *--------------------------------------------------------------------------*/

int
ZZZ_InitializeStructMatrix( ZZZ_StructMatrix matrix )
{
   return ( zzz_InitializeStructMatrix( (zzz_StructMatrix *) matrix ) );
}

/*--------------------------------------------------------------------------
 * ZZZ_SetStructMatrixValues
 *--------------------------------------------------------------------------*/

int 
ZZZ_SetStructMatrixValues( ZZZ_StructMatrix  matrix,
                           int              *grid_index,
                           int               num_stencil_indices,
                           int              *stencil_indices,
                           double           *values              )
{
   zzz_StructMatrix *new_matrix = (zzz_StructMatrix *) matrix;
   zzz_Index         new_grid_index;

   int               d;
   int               ierr;

   for (d = 0; d < zzz_StructGridDim(zzz_StructMatrixGrid(new_matrix)); d++)
   {
      zzz_IndexD(new_grid_index, d) = grid_index[d];
   }

   ierr = zzz_SetStructMatrixValues( new_matrix,
                                     new_grid_index,
                                     num_stencil_indices, stencil_indices,
                                     values );

   return (ierr);
}

/*--------------------------------------------------------------------------
 * ZZZ_SetStructMatrixCoeffs
 *--------------------------------------------------------------------------*/

int 
ZZZ_SetStructMatrixCoeffs( ZZZ_StructMatrix  matrix,
                           int              *grid_index,
                           double           *values              )
{
   zzz_StructMatrix *new_matrix = (zzz_StructMatrix *) matrix;
   zzz_Index         new_grid_index;

   int                 d;
   int                 s;
   int                 ierr;
   int                 stencil_size;
   zzz_StructStencil  *stencil;
   int                *stencil_indicies;

   stencil = zzz_StructMatrixStencil(new_matrix);
   stencil_size = zzz_StructStencilSize(stencil);
   stencil_indicies = zzz_CTAlloc(int, stencil_size);
   for (s = 0; s < stencil_size; s++)
     {
       stencil_indicies[s] = s;
     }

   for (d = 0; d < zzz_StructGridDim(zzz_StructMatrixGrid(new_matrix)); d++)
   {
      zzz_IndexD(new_grid_index, d) = grid_index[d];
   }

   ierr = zzz_SetStructMatrixValues( new_matrix,
                                     new_grid_index,
                                     stencil_size, stencil_indicies,
                                     values );

   zzz_TFree(stencil_indicies);

   return (ierr);
}

/*--------------------------------------------------------------------------
 * ZZZ_SetStructMatrixBoxValues
 *--------------------------------------------------------------------------*/

int 
ZZZ_SetStructMatrixBoxValues( ZZZ_StructMatrix  matrix,
                              int              *ilower,
                              int              *iupper,
                              int               num_stencil_indices,
                              int              *stencil_indices,
                              double           *values              )
{
   zzz_StructMatrix *new_matrix = (zzz_StructMatrix *) matrix;
   zzz_Index         new_ilower;
   zzz_Index         new_iupper;
   zzz_Box          *new_value_box;
                    
   int               d;
   int               ierr;

   for (d = 0; d < zzz_StructGridDim(zzz_StructMatrixGrid(new_matrix)); d++)
   {
      zzz_IndexD(new_ilower, d) = ilower[d];
      zzz_IndexD(new_iupper, d) = iupper[d];
   }
   new_value_box = zzz_NewBox(new_ilower, new_iupper);

   ierr = zzz_SetStructMatrixBoxValues( new_matrix,
                                        new_value_box,
                                        num_stencil_indices, stencil_indices,
                                        values );

   zzz_FreeBox(new_value_box);

   return (ierr);
}

/*--------------------------------------------------------------------------
 * ZZZ_AssembleStructMatrix
 *--------------------------------------------------------------------------*/

int 
ZZZ_AssembleStructMatrix( ZZZ_StructMatrix matrix )
{
   return( zzz_AssembleStructMatrix( (zzz_StructMatrix *) matrix ) );
}

/*--------------------------------------------------------------------------
 * ZZZ_SetStructMatrixNumGhost
 *--------------------------------------------------------------------------*/
 
void
ZZZ_SetStructMatrixNumGhost( ZZZ_StructMatrix  matrix,
                             int              *num_ghost )
{
   zzz_SetStructMatrixNumGhost( (zzz_StructMatrix *) matrix, num_ghost);
}

/*--------------------------------------------------------------------------
 * ZZZ_StructMatrixGrid
 *--------------------------------------------------------------------------*/

ZZZ_StructGrid
ZZZ_StructMatrixGrid( ZZZ_StructMatrix matrix )
{
   return ( (ZZZ_StructGrid) (zzz_StructMatrixGrid( (zzz_StructMatrix *) matrix) ) );
}

/*--------------------------------------------------------------------------
 * ZZZ_SetStructMatrixSymmetric
 *--------------------------------------------------------------------------*/
 
void
ZZZ_SetStructMatrixSymmetric( ZZZ_StructMatrix  matrix,
                              int               symmetric )
{
   zzz_StructMatrixSymmetric( (zzz_StructMatrix *) matrix ) = symmetric;
}

/*--------------------------------------------------------------------------
 * ZZZ_PrintStructMatrix
 *--------------------------------------------------------------------------*/

void 
ZZZ_PrintStructMatrix( char            *filename,
                       ZZZ_StructMatrix matrix,
                       int              all )
{
   zzz_PrintStructMatrix( filename,
                          (zzz_StructMatrix *) matrix,
                          all );
}
