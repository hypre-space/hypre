/*BHEADER**********************************************************************
 * (c) 1999   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

// this is a C++ header file

#include "ParaSAILS.h"
#include "seq_matrix_vector.h"
#include "parcsr_matrix_vector.h"

#ifndef _HYPRE_PARASAILS_I_H
#define _HYPRE_PARASAILS_I_H

typedef struct
{
    MPI_Comm                 comm;
    HYPRE_DistributedMatrix  matrix;
    ParaSAILS               *obj;
    hypre_ParCSRMatrix      *par_matrix;

} hypre_ParaSAILS;

#endif /* _HYPRE_PARASAILS_I_H */
