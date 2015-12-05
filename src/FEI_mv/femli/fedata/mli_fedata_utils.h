/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 1.8 $
 ***********************************************************************EHEADER*/





/**************************************************************************
 **************************************************************************
 * MLI_FEData utilities functions
 **************************************************************************
 **************************************************************************/

#include <stdio.h>
#include <string.h>
#include "HYPRE.h"
#include "parcsr_mv/_hypre_parcsr_mv.h"
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

