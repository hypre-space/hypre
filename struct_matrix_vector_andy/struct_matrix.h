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

#ifndef hypre_STRUCT_INTERFACE_MATRIX_HEADER
#define hypre_STRUCT_INTERFACE_MATRIX_HEADER


/*--------------------------------------------------------------------------
 * hypre_StructInterfaceMatrix:
 *--------------------------------------------------------------------------*/

typedef struct
{
   MPI_Comm      context;

   hypre_StructGrid     *grid;
   hypre_StructStencil  *stencil;

   int      	 storage_type;
   void     	*translator;
   void     	*data;

} hypre_StructInterfaceMatrix;

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_StructMatrix
 *--------------------------------------------------------------------------*/

#define hypre_StructInterfaceMatrixContext(matrix)      ((matrix) -> context)
#define hypre_StructInterfaceMatrixStructGrid(matrix)         ((matrix) -> grid)
#define hypre_StructInterfaceMatrixStructStencil(matrix)      ((matrix) -> stencil)

#define hypre_StructInterfaceMatrixStorageType(matrix)  ((matrix) -> storage_type)
#define hypre_StructInterfaceMatrixTranslator(matrix)   ((matrix) -> translator)
#define hypre_StructInterfaceMatrixData(matrix)         ((matrix) -> data)


#endif
