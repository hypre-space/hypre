/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Header info for Parallel Chord Matrix data structures
 *
 *****************************************************************************/

#ifndef hypre_PAR_CHORD_MATRIX_HEADER
#define hypre_PAR_CHORD_MATRIX_HEADER

/*--------------------------------------------------------------------------
 * Parallel Chord Matrix
 *--------------------------------------------------------------------------*/

typedef struct
{
   MPI_Comm comm;

   /*  A structure: -------------------------------------------------------- */
   HYPRE_Int num_inprocessors;
   HYPRE_Int *inprocessor;

   /* receiving in idof from different (in)processors; ---------------------- */
   HYPRE_Int *num_idofs_inprocessor;
   HYPRE_Int **idof_inprocessor;

   /* symmetric information: ----------------------------------------------- */
   /* this can be replaces by CSR format: ---------------------------------- */
   HYPRE_Int     *num_inchords;
   HYPRE_Int     **inchord_idof;
   HYPRE_Int     **inchord_rdof;
   HYPRE_Complex **inchord_data;

   HYPRE_Int num_idofs;
   HYPRE_Int num_rdofs;

   HYPRE_Int *firstindex_idof; /* not owned by my_id; ---------------------- */
   HYPRE_Int *firstindex_rdof; /* not owned by my_id; ---------------------- */

   /* --------------------------- mirror information: ---------------------- */
   /* participation of rdof in different processors; ----------------------- */

   HYPRE_Int num_toprocessors;
   HYPRE_Int *toprocessor;

   /* rdofs to be sentto toprocessors; -------------------------------------
      ---------------------------------------------------------------------- */
   HYPRE_Int *num_rdofs_toprocessor;
   HYPRE_Int **rdof_toprocessor;

} hypre_ParChordMatrix;

/*--------------------------------------------------------------------------
 * Accessor functions for the Parallel CSR Matrix structure
 *--------------------------------------------------------------------------*/

#define hypre_ParChordMatrixComm(matrix)                  ((matrix) -> comm)

/*  matrix structure: ----------------------------------------------------- */

#define hypre_ParChordMatrixNumInprocessors(matrix)  ((matrix) -> num_inprocessors)
#define hypre_ParChordMatrixInprocessor(matrix) ((matrix) -> inprocessor)
#define hypre_ParChordMatrixNumIdofsInprocessor(matrix) ((matrix) -> num_idofs_inprocessor)
#define hypre_ParChordMatrixIdofInprocessor(matrix) ((matrix) -> idof_inprocessor)

#define hypre_ParChordMatrixNumInchords(matrix) ((matrix) -> num_inchords)

#define hypre_ParChordMatrixInchordIdof(matrix) ((matrix) -> inchord_idof)
#define hypre_ParChordMatrixInchordRdof(matrix) ((matrix) -> inchord_rdof)
#define hypre_ParChordMatrixInchordData(matrix) ((matrix) -> inchord_data)
#define hypre_ParChordMatrixNumIdofs(matrix)    ((matrix) -> num_idofs)
#define hypre_ParChordMatrixNumRdofs(matrix)    ((matrix) -> num_rdofs)

#define hypre_ParChordMatrixFirstindexIdof(matrix) ((matrix) -> firstindex_idof)
#define hypre_ParChordMatrixFirstindexRdof(matrix) ((matrix) -> firstindex_rdof)

/* participation of rdof in different processors; ---------- */

#define hypre_ParChordMatrixNumToprocessors(matrix) ((matrix) -> num_toprocessors)
#define hypre_ParChordMatrixToprocessor(matrix)  ((matrix) -> toprocessor)
#define hypre_ParChordMatrixNumRdofsToprocessor(matrix) ((matrix) -> num_rdofs_toprocessor)
#define hypre_ParChordMatrixRdofToprocessor(matrix) ((matrix) -> rdof_toprocessor)

#endif

