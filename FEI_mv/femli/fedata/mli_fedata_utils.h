/*BHEADER**********************************************************************
 * (c) 2001   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

/**************************************************************************
 **************************************************************************
 * MLI_FEData utilities functions
 **************************************************************************
 **************************************************************************/

#include <stdio.h>
#include <string.h>
#include "HYPRE.h"
#include "parcsr_mv/parcsr_mv.h"
#include "fedata/mli_fedata.h"
#include "matrix/mli_matrix.h"

/**************************************************************************
 * functions 
 *-----------------------------------------------------------------------*/

void MLI_FEDataConstructElemNodeMatrix(MPI_Comm, MLI_FEData*, MLI_Matrix**);
void MLI_FEDataConstructElemFaceMatrix(MPI_Comm, MLI_FEData*, MLI_Matrix**);
void MLI_FEDataConstructFaceNodeMatrix(MPI_Comm, MLI_FEData*, MLI_Matrix**);
void MLI_FEDataConstructNodeElemMatrix(MPI_Comm, MLI_FEData*, MLI_Matrix**);
void MLI_FEDataConstructFaceElemMatrix(MPI_Comm, MLI_FEData*, MLI_Matrix**);
void MLI_FEDataConstructNodeFaceMatrix(MPI_Comm, MLI_FEData*, MLI_Matrix**);
void MLI_FEDataAgglomerateElemsLocal(MLI_Matrix *, int **macro_labels_out);
void MLI_FEDataAgglomerateElemsLocalOld(MLI_Matrix *, int **macro_labels_out);

