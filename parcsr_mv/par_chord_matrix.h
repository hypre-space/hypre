/*BHEADER**********************************************************************
 * (c) 1996   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * Header info for Parallel Chord Matrix data structures
 *
 *
 *****************************************************************************/
#include <HYPRE_config.h>



#ifndef hypre_PAR_CHORD_MATRIX_HEADER
#define hypre_PAR_CHORD_MATRIX_HEADER

#include "utilities.h"
#include "seq_mv.h"

/*--------------------------------------------------------------------------
 * Parallel Chord Matrix
 *--------------------------------------------------------------------------*/

typedef struct
{
   MPI_Comm comm;

  /*  A structure: -------------------------------------------------------- */
  int num_inprocessors;
  int *inprocessor;

  /* receiving in idof from different (in)processors; ---------------------- */
  int *num_idofs_inprocessor; 
  int **idof_inprocessor; 


  /* symmetric information: ----------------------------------------------- */
  /* this can be replaces by CSR format: ---------------------------------- */
  int *num_inchords;
  int **inchord_idof;
  int **inchord_rdof;
  double **inchord_data;

  int num_idofs;
  int num_rdofs;

  int *firstindex_idof; /* not owned by my_id; ----------------------------- */
  int *firstindex_rdof; /* not owned by my_id; ----------------------------- */

  /* --------------------------- mirror information: ---------------------- */
  /* participation of rdof in different processors; ------------------------ */

  int num_toprocessors;
  int *toprocessor;

  /* rdofs to be sentto toprocessors; --------------------------------------
     ----------------------------------------------------------------------- */
  int *num_rdofs_toprocessor;
  int **rdof_toprocessor;


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
