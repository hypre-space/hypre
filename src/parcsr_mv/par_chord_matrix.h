/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.7 $
 ***********************************************************************EHEADER*/




/******************************************************************************
 *
 * Header info for Parallel Chord Matrix data structures
 *
 *
 *****************************************************************************/
#include <HYPRE_config.h>



#ifndef hypre_PAR_CHORD_MATRIX_HEADER
#define hypre_PAR_CHORD_MATRIX_HEADER

#include "_hypre_utilities.h"
#include "seq_mv.h"

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
  HYPRE_Int *num_inchords;
  HYPRE_Int **inchord_idof;
  HYPRE_Int **inchord_rdof;
  double **inchord_data;

  HYPRE_Int num_idofs;
  HYPRE_Int num_rdofs;

  HYPRE_Int *firstindex_idof; /* not owned by my_id; ----------------------------- */
  HYPRE_Int *firstindex_rdof; /* not owned by my_id; ----------------------------- */

  /* --------------------------- mirror information: ---------------------- */
  /* participation of rdof in different processors; ------------------------ */

  HYPRE_Int num_toprocessors;
  HYPRE_Int *toprocessor;

  /* rdofs to be sentto toprocessors; --------------------------------------
     ----------------------------------------------------------------------- */
  HYPRE_Int *num_rdofs_toprocessor;
  HYPRE_Int **rdof_toprocessor;


} hypre_ParChordMatrix;

/*--------------------------------------------------------------------------
 * Accessor functions for the Parallel CSR Matrix structure
 *--------------------------------------------------------------------------*/

#define hypre_ParChordMatrixComm(matrix)		  ((matrix) -> comm)

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
