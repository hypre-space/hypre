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

#ifndef hypre_AUX_PAR_VECTOR_HEADER
#define hypre_AUX_PAR_VECTOR_HEADER

/*--------------------------------------------------------------------------
 * Auxiliary Parallel Vector
 *--------------------------------------------------------------------------*/

typedef struct
{
   int	    max_off_proc_elmts_set; /* length of off processor stash for
					SetValues */
   int	    current_num_elmts_set; /* current no. of elements stored in stash */
   int     *off_proc_i_set; /* contains column indices */
   double  *off_proc_data_set; /* contains corresponding data */
   int	    max_off_proc_elmts_add; /* length of off processor stash for
					SetValues */
   int	    current_num_elmts_add; /* current no. of elements stored in stash */
   int     *off_proc_i_add; /* contains column indices */
   double  *off_proc_data_add; /* contains corresponding data */
} hypre_AuxParVector;

/*--------------------------------------------------------------------------
 * Accessor functions for the Parallel Vector structure
 *--------------------------------------------------------------------------*/

#define hypre_AuxParVectorMaxOffProcElmtsSet(matrix)  ((matrix) -> max_off_proc_elmts_set)
#define hypre_AuxParVectorCurrentNumElmtsSet(matrix)  ((matrix) -> current_num_elmts_set)
#define hypre_AuxParVectorOffProcISet(matrix)  ((matrix) -> off_proc_i_set)
#define hypre_AuxParVectorOffProcDataSet(matrix)  ((matrix) -> off_proc_data_set)
#define hypre_AuxParVectorMaxOffProcElmtsAdd(matrix)  ((matrix) -> max_off_proc_elmts_add)
#define hypre_AuxParVectorCurrentNumElmtsAdd(matrix)  ((matrix) -> current_num_elmts_add)
#define hypre_AuxParVectorOffProcIAdd(matrix)  ((matrix) -> off_proc_i_add)
#define hypre_AuxParVectorOffProcDataAdd(matrix)  ((matrix) -> off_proc_data_add)

#endif
