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
 * Header file for ZZZ_mv library
 *
 *****************************************************************************/

#ifndef ZZZ_MV_HEADER
#define ZZZ_MV_HEADER

/*--------------------------------------------------------------------------
 * Structures
 *--------------------------------------------------------------------------*/

typedef void *ZZZ_StructStencil;
typedef void *ZZZ_StructGrid;
typedef void *ZZZ_StructMatrix;
typedef void *ZZZ_StructVector;

/*--------------------------------------------------------------------------
 * Prototypes
 *--------------------------------------------------------------------------*/

#ifdef __STDC__
# define        P(s) s
#else
# define P(s) ()
#endif
 
 
/* ZZZ_struct_grid.c */
ZZZ_StructGrid ZZZ_NewStructGrid P((MPI_Comm *comm , int dim ));
void ZZZ_FreeStructGrid P((ZZZ_StructGrid grid ));
void ZZZ_SetStructGridExtents P((ZZZ_StructGrid grid , int *ilower , int *iupper ));
void ZZZ_AssembleStructGrid P((ZZZ_StructGrid grid ));
 
/* ZZZ_struct_matrix.c */
ZZZ_StructMatrix ZZZ_NewStructMatrix P((MPI_Comm *comm , ZZZ_StructGrid grid , ZZZ_StructStencil stencil ));
int ZZZ_FreeStructMatrix P((ZZZ_StructMatrix matrix ));
int ZZZ_InitializeStructMatrix P((ZZZ_StructMatrix matrix ));
int ZZZ_SetStructMatrixValues P((ZZZ_StructMatrix matrix , int *grid_index , int num_stencil_indices , int *stencil_indices , double *values ));
int ZZZ_SetStructMatrixCoeffs P((ZZZ_StructMatrix  matrix, int *grid_index , double *values ));
int ZZZ_SetStructMatrixBoxValues P((ZZZ_StructMatrix matrix , int *ilower , int *iupper , int num_stencil_indices , int *stencil_indices , double *values ));
int ZZZ_AssembleStructMatrix P((ZZZ_StructMatrix matrix ));
void ZZZ_SetStructMatrixNumGhost P((ZZZ_StructMatrix matrix , int *num_ghost ));
ZZZ_StructGrid ZZZ_StructMatrixGrid P(( ZZZ_StructMatrix matrix ));
void ZZZ_SetStructMatrixSymmetric P((ZZZ_StructMatrix matrix , int symmetric ));
void ZZZ_PrintStructMatrix P(( char *filename, ZZZ_StructMatrix matrix , int all ));
 
/* ZZZ_struct_stencil.c */
ZZZ_StructStencil ZZZ_NewStructStencil P((int dim , int size ));
void ZZZ_SetStructStencilElement P((ZZZ_StructStencil stencil , int element_index , int *offset ));
void ZZZ_FreeStructStencil P((ZZZ_StructStencil stencil ));
 
/* ZZZ_struct_vector.c */
ZZZ_StructVector ZZZ_NewStructVector P((MPI_Comm *comm , ZZZ_StructGrid grid , ZZZ_StructStencil stencil ));
int ZZZ_FreeStructVector P((ZZZ_StructVector struct_vector ));
int ZZZ_InitializeStructVector P((ZZZ_StructVector vector ));
int ZZZ_SetStructVectorValues P((ZZZ_StructVector vector , int *grid_index , double values ));
int ZZZ_GetStructVectorValues P((ZZZ_StructVector vector , int *grid_index , double *values ));
int ZZZ_SetStructVectorBoxValues P((ZZZ_StructVector vector , int *ilower , int *iupper , int num_stencil_indices , int *stencil_indices , double *values ));
int ZZZ_AssembleStructVector P((ZZZ_StructVector vector ));
void ZZZ_PrintStructVector P(( char *filename , ZZZ_StructVector vector , int all ));
void ZZZ_SetStructVectorNumGhost P((ZZZ_StructMatrix vector , int *num_ghost ));
int ZZZ_SetStructVectorConstantValues P((ZZZ_StructMatrix  vector , double values ));
 
#undef P

#endif
