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
 * Header info for the hypre_StructMatrix structures
 *
 *****************************************************************************/

#ifndef hypre_STRUCT_MATRIX_HEADER
#define hypre_STRUCT_MATRIX_HEADER


/*--------------------------------------------------------------------------
 * hypre_StructMatrix:
 *--------------------------------------------------------------------------*/

typedef struct
{
   MPI_Comm      context;

   hypre_StructGrid     *grid;
   hypre_StructStencil  *stencil;

   int      	 storage_type;
   void     	*translator;
   void     	*data;

} hypre_StructMatrix;

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_StructMatrix
 *--------------------------------------------------------------------------*/

#define hypre_StructMatrixContext(matrix)      ((matrix) -> context)
#define hypre_StructMatrixStructGrid(matrix)         ((matrix) -> grid)
#define hypre_StructMatrixStructStencil(matrix)      ((matrix) -> stencil)

#define hypre_StructMatrixStorageType(matrix)  ((matrix) -> storage_type)
#define hypre_StructMatrixTranslator(matrix)   ((matrix) -> translator)
#define hypre_StructMatrixData(matrix)         ((matrix) -> data)


#endif
