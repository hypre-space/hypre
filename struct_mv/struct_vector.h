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

#ifndef zzz_STRUCT_VECTOR_HEADER
#define zzz_STRUCT_VECTOR_HEADER


/*--------------------------------------------------------------------------
 * zzz_StructVector:
 *--------------------------------------------------------------------------*/

typedef struct
{
   zzz_StructGrid     *grid;

   zzz_BoxArray       *data_space;

   double             *data;         /* Pointer to vector data */
   int                 data_size;    /* Size of vector data */
   int                *data_indices; /* num-boxes array of indices into
                                        the data array.  data_indices[b]
                                        is the starting index of vector
                                        data corresponding to box b. */

   int                 num_ghost[6]; /* Num ghost layers in each direction */

   int                 global_size;  /* Total number coefficients */

} zzz_StructVector;

/*--------------------------------------------------------------------------
 * Accessor macros: zzz_StructVector
 *--------------------------------------------------------------------------*/

#define zzz_StructVectorGrid(vector)          ((vector) -> grid)

#define zzz_StructVectorDataSpace(vector)     ((vector) -> data_space)
 
#define zzz_StructVectorData(vector)          ((vector) -> data)
#define zzz_StructVectorDataSize(vector)      ((vector) -> data_size)
#define zzz_StructVectorDataIndices(vector)   ((vector) -> data_indices)

#define zzz_StructVectorNumGhost(vector)      ((vector) -> num_ghost)
 
#define zzz_StructVectorGlobalSize(vector)    ((vector) -> global_size)
 
#define zzz_StructVectorBox(vector, b) \
zzz_BoxArrayBox(zzz_StructVectorDataSpace(vector), b)
 
#define zzz_StructVectorBoxData(vector, b) \
(zzz_StructVectorData(vector) + zzz_StructVectorDataIndices(vector)[b])
 
#define zzz_StructVectorBoxDataValue(vector, b, index) \
(zzz_StructVectorBoxData(vector, b) + \
 zzz_BoxIndexRank(zzz_StructVectorBox(vector, b), index))
 
#define zzz_StructVectorContext(vector) \
StructGridContext(StructVectorStructGrid(vector))


#endif
