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
 * Header info for the zzz_StructVector structures
 *
 *****************************************************************************/

#ifndef zzz_STENCIL_VECTOR_HEADER
#define zzz_STENCIL_VECTOR_HEADER


/*--------------------------------------------------------------------------
 * zzz_StructVector:
 *--------------------------------------------------------------------------*/

typedef struct
{
   MPI_Comm      context;

   zzz_StructGrid     *grid;
   zzz_StructStencil  *stencil;

   int      	 storage_type;
   void     	*translator;
   void     	*data;

} zzz_StructVector;

/*--------------------------------------------------------------------------
 * Accessor macros: zzz_StructVector
 *--------------------------------------------------------------------------*/

#define zzz_StructVectorContext(vector)      ((vector) -> context)
#define zzz_StructVectorStructGrid(vector)         ((vector) -> grid)
#define zzz_StructVectorStructStencil(vector)      ((vector) -> stencil)

#define zzz_StructVectorStorageType(vector)  ((vector) -> storage_type)
#define zzz_StructVectorTranslator(vector)   ((vector) -> translator)
#define zzz_StructVectorData(vector)         ((vector) -> data)


#endif
