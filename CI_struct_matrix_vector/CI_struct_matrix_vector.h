
#include <HYPRE_config.h>

#include "HYPRE_CI_mv.h"

#ifndef hypre_CI_MV_HEADER
#define hypre_CI_MV_HEADER

#include "utilities.h"

#include "HYPRE.h"


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
 * Header info for the hypre_StructInterfaceMatrix structures
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
   int           symmetric;
   void     	*translator;
   void     	*data;

} hypre_StructInterfaceMatrix;

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_StructInterfaceMatrix
 *--------------------------------------------------------------------------*/

#define hypre_StructInterfaceMatrixContext(matrix)      ((matrix) -> context)
#define hypre_StructInterfaceMatrixStructGrid(matrix)         ((matrix) -> grid)
#define hypre_StructInterfaceMatrixStructStencil(matrix)      ((matrix) -> stencil)

#define hypre_StructInterfaceMatrixStorageType(matrix)  ((matrix) -> storage_type)
#define hypre_StructInterfaceMatrixSymmetric(matrix)  ((matrix) -> symmetric)
#define hypre_StructInterfaceMatrixTranslator(matrix)   ((matrix) -> translator)
#define hypre_StructInterfaceMatrixData(matrix)         ((matrix) -> data)


#endif

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
 * Header info for the hypre_StructInterfaceVector structures
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

   int           retrieval_on;

   int      	 storage_type;
   void     	*translator;
   void     	*data;
   void         *auxiliary_data;

} hypre_StructInterfaceVector;

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_StructInterfaceVector
 *--------------------------------------------------------------------------*/

#define hypre_StructInterfaceVectorContext(vector)      ((vector) -> context)
#define hypre_StructInterfaceVectorStructGrid(vector)         ((vector) -> grid)
#define hypre_StructInterfaceVectorStructStencil(vector)      ((vector) -> stencil)
#define hypre_StructInterfaceVectorRetrievalOn(vector)      ((vector) -> retrieval_on)

#define hypre_StructInterfaceVectorStorageType(vector)  ((vector) -> storage_type)
#define hypre_StructInterfaceVectorTranslator(vector)   ((vector) -> translator)
#define hypre_StructInterfaceVectorData(vector)         ((vector) -> data)
#define hypre_StructInterfaceVectorAuxData(vector)         ((vector) -> auxiliary_data)

/*--------------------------------------------------------------------------
 * Auxiliary Data Structure definitions
 *--------------------------------------------------------------------------*/

/* PETSc Matrix */

typedef struct
{
  double     *VecArray;
} hypre_StructInterfaceVectorPETScAD;

#define hypre_StructInterfaceVectorVecArray(vector) \
  (( (hypre_StructInterfaceVectorPETScAD *) hypre_StructInterfaceVectorAuxData(vector)) -> \
       VecArray)


#endif

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
 * Header info for the hypre_StructGridToCoord structures
 *
 *****************************************************************************/

#ifndef hypre_GRID_TO_COORD_HEADER
#define hypre_GRID_TO_COORD_HEADER


/*--------------------------------------------------------------------------
 * hypre_StructGridToCoordTable:
 *--------------------------------------------------------------------------*/

typedef struct
{
   int   offset;
   int   ni;
   int   nj;

} hypre_StructGridToCoordTableEntry;

typedef struct
{
   hypre_StructGridToCoordTableEntry  **entries;
   int           	       *indices[3];
   int           	        size[3];

   int                          last_index[3];

} hypre_StructGridToCoordTable;

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_StructGridToCoordTable
 *--------------------------------------------------------------------------*/

#define hypre_StructGridToCoordTableEntries(table)        ((table) -> entries)
#define hypre_StructGridToCoordTableIndices(table)        ((table) -> indices)
#define hypre_StructGridToCoordTableSize(table)           ((table) -> size)
#define hypre_StructGridToCoordTableLastIndex(table)      ((table) -> last_index)

#define hypre_StructGridToCoordTableIndexListD(table, d)  \
hypre_StructGridToCoordTableIndices(table)[d]
#define hypre_StructGridToCoordTableIndexD(table, d, i) \
hypre_StructGridToCoordTableIndices(table)[d][i]
#define hypre_StructGridToCoordTableSizeD(table, d) \
hypre_StructGridToCoordTableSize(table)[d]
#define hypre_StructGridToCoordTableLastIndexD(table, d) \
hypre_StructGridToCoordTableLastIndex(table)[d]

#define hypre_StructGridToCoordTableEntry(table, i, j, k) \
hypre_StructGridToCoordTableEntries(table)\
[((k*hypre_StructGridToCoordTableSizeD(table, 1) + j)*\
  hypre_StructGridToCoordTableSizeD(table, 0) + i)]

#define hypre_StructGridToCoordTableEntryOffset(entry)   ((entry) -> offset)
#define hypre_StructGridToCoordTableEntryNI(entry)       ((entry) -> ni)
#define hypre_StructGridToCoordTableEntryNJ(entry)       ((entry) -> nj)

/*--------------------------------------------------------------------------
 * Member macros for hypre_StructGridToCoord translator class
 *--------------------------------------------------------------------------*/

#define hypre_MapStructGridToCoord(index, entry) \
(hypre_StructGridToCoordTableEntryOffset(entry) + \
 ((index[2]*hypre_StructGridToCoordTableEntryNJ(entry) + \
   index[1])*hypre_StructGridToCoordTableEntryNI(entry) + index[0]))

#endif

# define	P(s) s

/* HYPRE_struct_matrix.c */
int HYPRE_StructInterfaceMatrixCreate P((MPI_Comm context , HYPRE_StructGrid grid , HYPRE_StructStencil stencil , HYPRE_StructInterfaceMatrix *matrix ));
int HYPRE_StructInterfaceMatrixDestroy P((HYPRE_StructInterfaceMatrix matrix ));
int HYPRE_SetStructInterfaceMatrixCoeffs P((HYPRE_StructInterfaceMatrix matrix , int *grid_index , double *coeffs ));
int HYPRE_StructInterfaceMatrixSetValues P((HYPRE_StructInterfaceMatrix matrix , int *grid_index , int num_stencil_indices , int *stencil_indices , double *coeffs ));
int HYPRE_StructInterfaceMatrixSetBoxValues P((HYPRE_StructInterfaceMatrix matrix , int *lower_grid_index , int *upper_grid_index , int num_stencil_indices , int *stencil_indices , double *coeffs ));
int HYPRE_StructInterfaceMatrixInitialize P((HYPRE_StructInterfaceMatrix matrix ));
int HYPRE_StructInterfaceMatrixAssemble P((HYPRE_StructInterfaceMatrix matrix ));
void *HYPRE_StructInterfaceMatrixGetData P((HYPRE_StructInterfaceMatrix matrix ));
int HYPRE_StructInterfaceMatrixPrint P((HYPRE_StructInterfaceMatrix matrix ));
int HYPRE_SetStructInterfaceMatrixStorageType P((HYPRE_StructInterfaceMatrix struct_matrix , int type ));
int HYPRE_StructInterfaceMatrixSetSymmetric P((HYPRE_StructInterfaceMatrix struct_matrix , int type ));
int HYPRE_StructInterfaceMatrixSetNumGhost P((HYPRE_StructInterfaceMatrix struct_matrix , int *num_ghost ));
int HYPRE_StructInterfaceMatrixGetGrid P((HYPRE_StructInterfaceMatrix matrix , HYPRE_StructGrid *grid ));

/* HYPRE_struct_vector.c */
int HYPRE_StructInterfaceVectorCreate P((MPI_Comm context , HYPRE_StructGrid grid , HYPRE_StructStencil stencil , HYPRE_StructInterfaceVector *vector ));
int HYPRE_StructInterfaceVectorDestroy P((HYPRE_StructInterfaceVector struct_vector ));
int HYPRE_SetStructInterfaceVectorCoeffs P((HYPRE_StructInterfaceVector vector , int *grid_index , double *coeffs ));
int HYPRE_StructInterfaceVectorSetValues P((HYPRE_StructInterfaceVector vector , int *grid_index , double *coeffs ));
int HYPRE_StructInterfaceVectorGetValues P((HYPRE_StructInterfaceVector vector , int *grid_index , double *values_ptr ));
int HYPRE_StructInterfaceVectorSetBoxValues P((HYPRE_StructInterfaceVector vector , int *lower_grid_index , int *upper_grid_index , double *coeffs ));
int HYPRE_StructInterfaceVectorGetBoxValues P((HYPRE_StructInterfaceVector vector , int *lower_grid_index , int *upper_grid_index , double *values_ptr ));
int HYPRE_SetStructInterfaceVector P((HYPRE_StructInterfaceVector vector , double *val ));
int HYPRE_StructInterfaceVectorInitialize P((HYPRE_StructInterfaceVector vector ));
int HYPRE_StructInterfaceVectorAssemble P((HYPRE_StructInterfaceVector vector ));
int HYPRE_SetStructInterfaceVectorStorageType P((HYPRE_StructInterfaceVector struct_vector , int type ));
void *HYPRE_StructInterfaceVectorGetData P((HYPRE_StructInterfaceVector vector ));
int HYPRE_StructInterfaceVectorPrint P((HYPRE_StructInterfaceVector vector ));
int HYPRE_RetrievalOnStructInterfaceVector P((HYPRE_StructInterfaceVector vector ));
int HYPRE_RetrievalOffStructInterfaceVector P((HYPRE_StructInterfaceVector vector ));
int HYPRE_GetStructInterfaceVectorValue P((HYPRE_StructInterfaceVector vector , int *index , double *value ));
int HYPRE_StructInterfaceVectorSetNumGhost P((HYPRE_StructInterfaceVector vector , int *num_ghost ));

/* grid_to_coord.c */
hypre_StructGridToCoordTable *hypre_NewStructGridToCoordTable P((hypre_StructGrid *grid , hypre_StructStencil *stencil ));
void hypre_FreeStructGridToCoordTable P((hypre_StructGridToCoordTable *table ));
hypre_StructGridToCoordTableEntry *hypre_FindStructGridToCoordTableEntry P((hypre_Index index , hypre_StructGridToCoordTable *table ));

/* struct_matrix.c */
hypre_StructInterfaceMatrix *hypre_NewStructInterfaceMatrix P((MPI_Comm context , hypre_StructGrid *grid , hypre_StructStencil *stencil ));
int hypre_FreeStructInterfaceMatrix P((hypre_StructInterfaceMatrix *matrix ));
int hypre_SetStructInterfaceMatrixCoeffs P((hypre_StructInterfaceMatrix *matrix , hypre_Index grid_index , double *coeffs ));
int hypre_SetStructInterfaceMatrixBoxValues P((hypre_StructInterfaceMatrix *matrix , hypre_Index lower_grid_index , hypre_Index upper_grid_index , int num_stencil_indices , int *stencil_indices , double *coeffs ));
int hypre_AssembleStructInterfaceMatrix P((hypre_StructInterfaceMatrix *matrix ));
int hypre_PrintStructInterfaceMatrix P((hypre_StructInterfaceMatrix *matrix ));
int hypre_SetStructInterfaceMatrixStorageType P((hypre_StructInterfaceMatrix *matrix , int type ));
int hypre_SetStructInterfaceMatrixSymmetric P((hypre_StructInterfaceMatrix *matrix , int type ));
int *hypre_FindBoxNeighborhood P((hypre_BoxArray *boxes , hypre_BoxArray *all_boxes , hypre_StructStencil *stencil ));
int *hypre_FindBoxApproxNeighborhood P((hypre_BoxArray *boxes , hypre_BoxArray *all_boxes , hypre_StructStencil *stencil ));

/* struct_matrix_PETSc.c */
int hypre_FreeStructInterfaceMatrixPETSc P((hypre_StructInterfaceMatrix *struct_matrix ));
int hypre_SetStructInterfaceMatrixPETScCoeffs P((hypre_StructInterfaceMatrix *struct_matrix , hypre_Index index , double *coeffs ));
int hypre_PrintStructInterfaceMatrixPETSc P((hypre_StructInterfaceMatrix *struct_matrix ));
int hypre_AssembleStructInterfaceMatrixPETSc P((hypre_StructInterfaceMatrix *struct_matrix ));

/* struct_vector.c */
hypre_StructInterfaceVector *hypre_NewStructInterfaceVector P((MPI_Comm context , hypre_StructGrid *grid , hypre_StructStencil *stencil ));
int hypre_FreeStructInterfaceVector P((hypre_StructInterfaceVector *vector ));
int hypre_SetStructInterfaceVectorCoeffs P((hypre_StructInterfaceVector *vector , hypre_Index grid_index , double *coeffs ));
int hypre_SetStructInterfaceVectorBoxValues P((hypre_StructInterfaceVector *vector , hypre_Index lower_grid_index , hypre_Index upper_grid_index , double *coeffs ));
int hypre_SetStructInterfaceVector P((hypre_StructInterfaceVector *vector , double *val ));
int hypre_AssembleStructInterfaceVector P((hypre_StructInterfaceVector *vector ));
int hypre_SetStructInterfaceVectorStorageType P((hypre_StructInterfaceVector *vector , int type ));
int hypre_PrintStructInterfaceVector P((hypre_StructInterfaceVector *vector ));
int hypre_RetrievalOnStructInterfaceVector P((hypre_StructInterfaceVector *vector ));
int hypre_RetrievalOffStructInterfaceVector P((hypre_StructInterfaceVector *vector ));
int hypre_GetStructInterfaceVectorValue P((hypre_StructInterfaceVector *vector , hypre_Index index , double *value ));

/* struct_vector_PETSc.c */
int hypre_FreeStructInterfaceVectorPETSc P((hypre_StructInterfaceVector *struct_vector ));
int hypre_SetStructInterfaceVectorPETScCoeffs P((hypre_StructInterfaceVector *struct_vector , hypre_Index index , double *coeffs ));
int hypre_SetStructInterfaceVectorPETSc P((hypre_StructInterfaceVector *struct_vector , double *val ));
int hypre_AssembleStructInterfaceVectorPETSc P((hypre_StructInterfaceVector *struct_vector ));
int hypre_PrintStructInterfaceVectorPETSc P((hypre_StructInterfaceVector *struct_vector ));
int hypre_RetrievalOnStructInterfaceVectorPETSc P((hypre_StructInterfaceVector *vector ));
int hypre_RetrievalOffStructInterfaceVectorPETSc P((hypre_StructInterfaceVector *vector ));
int hypre_GetStructInterfaceVectorPETScValue P((hypre_StructInterfaceVector *vector , hypre_Index index , double *value ));

#undef P

#endif
