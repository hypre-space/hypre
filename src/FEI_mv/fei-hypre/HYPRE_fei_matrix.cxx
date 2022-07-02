/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * HYPRE_fei_matrix functions
 *
 *****************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "IJ_mv/HYPRE_IJ_mv.h"
#include "parcsr_mv/HYPRE_parcsr_mv.h"
#include "fei_mv.h"
//NEW FEI 2.23.02
#include "fei_Data.hpp"

/*****************************************************************************/
/* HYPRE_FEMatrixCreate function                                             */
/*---------------------------------------------------------------------------*/

extern "C" int
HYPRE_FEMatrixCreate(MPI_Comm comm, HYPRE_FEMesh mesh, HYPRE_FEMatrix *matrix)
{
   HYPRE_FEMatrix myMatrix;
   myMatrix = (HYPRE_FEMatrix) hypre_TAlloc(HYPRE_FEMatrix, 1, HYPRE_MEMORY_HOST);
   myMatrix->comm_ = comm;
   myMatrix->mesh_ = mesh;
   (*matrix) = myMatrix;
   return 0;
}

/*****************************************************************************/
/* HYPRE_FEMatrixDestroy - Destroy a FEMatrix object.                        */
/*---------------------------------------------------------------------------*/

extern "C" int
HYPRE_FEMatrixDestroy(HYPRE_FEMatrix matrix)
{
   if (matrix)
   {
      free(matrix);
   }
   return 0;
}

/*****************************************************************************/
/* HYPRE_FEMatrixGetObject                                                   */
/*---------------------------------------------------------------------------*/

extern "C" int
HYPRE_FEMatrixGetObject(HYPRE_FEMatrix matrix, void **object)
{
   int                ierr=0;
   HYPRE_FEMesh       mesh;
   LinearSystemCore*  lsc;
   Data               dataObj;
   HYPRE_IJMatrix     A;
   HYPRE_ParCSRMatrix ACSR;

   if (matrix == NULL)
      ierr = 1;
   else
   {
      mesh = matrix->mesh_;
      if (mesh == NULL)
         ierr = 1;
      else
      {
         lsc = (LinearSystemCore *) mesh->linSys_;
         if (lsc != NULL)
         {
            lsc->copyOutMatrix(1.0e0, dataObj); 
            A = (HYPRE_IJMatrix) dataObj.getDataPtr();
            HYPRE_IJMatrixGetObject(A, (void **) &ACSR);
            (*object) = (void *) ACSR;
         }
         else
         {
            (*object) = NULL;
            ierr = 1;
         }
      }
   }
   return ierr;
}

