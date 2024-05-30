/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Header file for hypre_IntArray struct for holding an array of integers
 *
 *****************************************************************************/

#ifndef hypre_INTARRAY_HEADER
#define hypre_INTARRAY_HEADER

/*--------------------------------------------------------------------------
 * hypre_IntArray
 *--------------------------------------------------------------------------*/

typedef struct
{
   /* pointer to data and size of data */
   HYPRE_Int            *data;
   HYPRE_Int             size;

   /* memory location of array data */
   HYPRE_MemoryLocation  memory_location;
} hypre_IntArray;

/*--------------------------------------------------------------------------
 * Accessor functions for the IntArray structure
 *--------------------------------------------------------------------------*/

#define hypre_IntArrayData(array)                  ((array) -> data)
#define hypre_IntArrayDataI(array, i)              ((array) -> data[i])
#define hypre_IntArraySize(array)                  ((array) -> size)
#define hypre_IntArrayMemoryLocation(array)        ((array) -> memory_location)

/******************************************************************************
 *
 * hypre_IntArrayArray: struct for holding an array of hypre_IntArray
 *
 *****************************************************************************/

/*--------------------------------------------------------------------------
 * hypre_IntArrayArray
 *--------------------------------------------------------------------------*/

typedef struct
{
   hypre_IntArray       **entries;
   HYPRE_Int              size;
} hypre_IntArrayArray;

/*--------------------------------------------------------------------------
 * Accessor functions for the IntArrayArray structure
 *--------------------------------------------------------------------------*/

#define hypre_IntArrayArrayEntries(array)           ((array) -> entries)
#define hypre_IntArrayArrayEntryI(array, i)         ((array) -> entries[i])
#define hypre_IntArrayArrayEntryIData(array, i)     ((array) -> entries[i] -> data)
#define hypre_IntArrayArrayEntryIDataJ(array, i, j) ((array) -> entries[i] -> data[j])
#define hypre_IntArrayArraySize(array)              ((array) -> size)

#endif
