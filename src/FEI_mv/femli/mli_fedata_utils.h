/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/**************************************************************************
 **************************************************************************
 * MLI_FEData utilities functions
 **************************************************************************
 **************************************************************************/

#include <stdio.h>
#include <string.h>
#include "HYPRE.h"
#include "_hypre_parcsr_mv.h"
#include "mli_fedata.h"
#include "mli_matrix.h"

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

