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
 * Header info for the zzz_StructMatrix structures
 *
 *****************************************************************************/

#ifndef zzz_STENCIL_MATRIX_HEADER
#define zzz_STENCIL_MATRIX_HEADER


/*--------------------------------------------------------------------------
 * zzz_StructMatrix:
 *--------------------------------------------------------------------------*/

typedef struct
{
   MPI_Comm      context;

   zzz_StructGrid     *grid;
   zzz_StructStencil  *stencil;

   int      	 storage_type;
   void     	*translator;
   void     	*data;

} zzz_StructMatrix;

/*--------------------------------------------------------------------------
 * Accessor macros: zzz_StructMatrix
 *--------------------------------------------------------------------------*/

#define zzz_StructMatrixContext(matrix)      ((matrix) -> context)
#define zzz_StructMatrixStructGrid(matrix)         ((matrix) -> grid)
#define zzz_StructMatrixStructStencil(matrix)      ((matrix) -> stencil)

#define zzz_StructMatrixStorageType(matrix)  ((matrix) -> storage_type)
#define zzz_StructMatrixTranslator(matrix)   ((matrix) -> translator)
#define zzz_StructMatrixData(matrix)         ((matrix) -> data)


#endif
