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
 * Header info for Auxiliary Parallel Vector data structures
 *
 * Note: this vector currently uses 0-based indexing.
 *
 *****************************************************************************/

#ifndef hypre_AUX_PARCSR_VECTOR_HEADER
#define hypre_AUX_PARCSR_VECTOR_HEADER

/*--------------------------------------------------------------------------
 * Auxiliary Parallel Vector
 *--------------------------------------------------------------------------*/

typedef struct
{
   int      local_num_rows;   /* defines number of rows on this processor */

   int      need_aux; /* if need_aux = 1, aux_data are used to
			generate the ParVector (default),
			for need_aux = 0, data is put directly into
			ParVector (requires the knowledge of
			offd_i and diag_i ) */

   int    **aux_i;    /* contains collected column indices */
   double **aux_data; /* contains collected data */

} hypre_AuxParVector;

/*--------------------------------------------------------------------------
 * Accessor functions for the Parallel Vector structure
 *--------------------------------------------------------------------------*/

#define hypre_AuxParVectorLocalNumRows(vector)  ((vector) -> local_num_rows)
#define hypre_AuxParVectorNeedAux(vector)       ((vector) -> need_aux)
#define hypre_AuxParVectorAuxI(vector)          ((vector) -> aux_i)
#define hypre_AuxParVectorAuxData(vector)       ((vector) -> aux_data)

#endif
