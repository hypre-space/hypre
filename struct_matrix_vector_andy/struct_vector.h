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
 * Header info for the hypre_StructVector structures
 *
 *****************************************************************************/

#ifndef hypre_STENCIL_VECTOR_HEADER
#define hypre_STENCIL_VECTOR_HEADER


/*--------------------------------------------------------------------------
 * hypre_StructVector:
 *--------------------------------------------------------------------------*/

typedef struct
{
   MPI_Comm      context;

   hypre_StructGrid     *grid;
   hypre_StructStencil  *stencil;

   int      	 storage_type;
   void     	*translator;
   void     	*data;

} hypre_StructVector;

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_StructVector
 *--------------------------------------------------------------------------*/

#define hypre_StructVectorContext(vector)      ((vector) -> context)
#define hypre_StructVectorStructGrid(vector)         ((vector) -> grid)
#define hypre_StructVectorStructStencil(vector)      ((vector) -> stencil)

#define hypre_StructVectorStorageType(vector)  ((vector) -> storage_type)
#define hypre_StructVectorTranslator(vector)   ((vector) -> translator)
#define hypre_StructVectorData(vector)         ((vector) -> data)


#endif
