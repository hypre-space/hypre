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

#ifndef hypre_STENCIL_INTERFACE_VECTOR_HEADER
#define hypre_STENCIL_INTERFACE_VECTOR_HEADER


/*--------------------------------------------------------------------------
 * hypre_StructInterfaceVector:
 *--------------------------------------------------------------------------*/

typedef struct
{
   MPI_Comm      context;

   hypre_StructGrid     *grid;
   hypre_StructStencil  *stencil;

   int      	 storage_type;
   void     	*translator;
   void     	*data;

} hypre_StructInterfaceVector;

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_StructInterfaceVector
 *--------------------------------------------------------------------------*/

#define hypre_StructInterfaceVectorContext(vector)      ((vector) -> context)
#define hypre_StructInterfaceVectorStructGrid(vector)         ((vector) -> grid)
#define hypre_StructInterfaceVectorStructStencil(vector)      ((vector) -> stencil)

#define hypre_StructInterfaceVectorStorageType(vector)  ((vector) -> storage_type)
#define hypre_StructInterfaceVectorTranslator(vector)   ((vector) -> translator)
#define hypre_StructInterfaceVectorData(vector)         ((vector) -> data)


#endif
