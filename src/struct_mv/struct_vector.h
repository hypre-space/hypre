/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Header info for the hypre_StructVector structures
 *
 *****************************************************************************/

#ifndef hypre_STRUCT_VECTOR_HEADER
#define hypre_STRUCT_VECTOR_HEADER

/*--------------------------------------------------------------------------
 * hypre_StructVector:
 *--------------------------------------------------------------------------*/

typedef struct hypre_StructVector_struct
{
   MPI_Comm              comm;

   hypre_StructGrid     *grid;

   hypre_BoxArray       *data_space;

   HYPRE_MemoryLocation  memory_location;             /* memory location of data */
   HYPRE_Complex        *data;                        /* Pointer to vector data on device*/
   HYPRE_Int             data_alloced;                /* Boolean used for freeing data */
   HYPRE_Int             data_size;                   /* Size of vector data */
   HYPRE_Int            *data_indices;                /* num-boxes array of indices into
                                                         the data array.  data_indices[b]
                                                         is the starting index of vector
                                                         data corresponding to box b. */

   HYPRE_Int             num_ghost[2 * HYPRE_MAXDIM]; /* Num ghost layers in each
                                                       * direction */
   HYPRE_Int             bghost_not_clear;            /* Are boundary ghosts clear? */

   HYPRE_BigInt          global_size;                 /* Total number coefficients */

   HYPRE_Int             ref_count;

} hypre_StructVector;

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_StructVector
 *--------------------------------------------------------------------------*/

#define hypre_StructVectorComm(vector)           ((vector) -> comm)
#define hypre_StructVectorGrid(vector)           ((vector) -> grid)
#define hypre_StructVectorDataSpace(vector)      ((vector) -> data_space)
#define hypre_StructVectorMemoryLocation(vector) ((vector) -> memory_location)
#define hypre_StructVectorData(vector)           ((vector) -> data)
#define hypre_StructVectorDataAlloced(vector)    ((vector) -> data_alloced)
#define hypre_StructVectorDataSize(vector)       ((vector) -> data_size)
#define hypre_StructVectorDataIndices(vector)    ((vector) -> data_indices)
#define hypre_StructVectorNumGhost(vector)       ((vector) -> num_ghost)
#define hypre_StructVectorBGhostNotClear(vector) ((vector) -> bghost_not_clear)
#define hypre_StructVectorGlobalSize(vector)     ((vector) -> global_size)
#define hypre_StructVectorRefCount(vector)       ((vector) -> ref_count)

#define hypre_StructVectorNDim(vector) \
hypre_StructGridNDim(hypre_StructVectorGrid(vector))

#define hypre_StructVectorBox(vector, b) \
hypre_BoxArrayBox(hypre_StructVectorDataSpace(vector), b)

#define hypre_StructVectorBoxData(vector, b) \
(hypre_StructVectorData(vector) + hypre_StructVectorDataIndices(vector)[b])

#define hypre_StructVectorBoxDataValue(vector, b, index) \
(hypre_StructVectorBoxData(vector, b) + \
 hypre_BoxIndexRank(hypre_StructVectorBox(vector, b), index))

#endif
