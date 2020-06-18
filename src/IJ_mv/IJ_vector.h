/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#if 0
/******************************************************************************
 *
 * Header info for the hypre_IJMatrix structures
 *
 *****************************************************************************/

#ifndef hypre_IJ_VECTOR_HEADER
#define hypre_IJ_VECTOR_HEADER

/*--------------------------------------------------------------------------
 * hypre_IJVector:
 *--------------------------------------------------------------------------*/

typedef struct hypre_IJVector_struct
{
   MPI_Comm      comm;

   HYPRE_BigInt	*partitioning;      /* Indicates partitioning over tasks */

   HYPRE_Int     object_type;       /* Indicates the type of "local storage" */

   void         *object;            /* Structure for storing local portion */

   void         *translator;        /* Structure for storing off processor
				       information */

   void         *assumed_part;        /* IJ Vector assumed partition */

   HYPRE_BigInt  global_first_row;    /* these for data items are necessary */
   HYPRE_BigInt  global_num_rows;     /*   to be able to avoid using the global */
                                    /*    global partition */ 
   HYPRE_Int     print_level; 
   


} hypre_IJVector;

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_IJVector
 *--------------------------------------------------------------------------*/

#define hypre_IJVectorComm(vector)           ((vector) -> comm)

#define hypre_IJVectorPartitioning(vector)   ((vector) -> partitioning)

#define hypre_IJVectorObjectType(vector)     ((vector) -> object_type)

#define hypre_IJVectorObject(vector)         ((vector) -> object)

#define hypre_IJVectorTranslator(vector)     ((vector) -> translator)

#define hypre_IJVectorAssumedPart(vector)     ((vector) -> assumed_part)

#define hypre_IJVectorGlobalFirstRow(vector)  ((vector) -> global_first_row)

#define hypre_IJVectorGlobalNumRows(vector)  ((vector) -> global_num_rows)

#define hypre_IJVectorPrintLevel(vector)  ((vector) -> print_level)

/*--------------------------------------------------------------------------
 * prototypes for operations on local objects
 *--------------------------------------------------------------------------*/
/* #include "./internal_protos.h" */

#endif

#endif
