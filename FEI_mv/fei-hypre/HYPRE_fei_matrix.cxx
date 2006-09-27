/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team, UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer and the GNU Lesser General Public License.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the terms and conditions of the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
 ***********************************************************************EHEADER*/

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
#include "HYPRE_fei_mv.h"
#include "Data.h"

/*****************************************************************************/
/* HYPRE_FEMatrixCreate function                                             */
/*---------------------------------------------------------------------------*/

extern "C" int
HYPRE_FEMatrixCreate(MPI_Comm comm, HYPRE_FEMesh mesh, HYPRE_FEMatrix *matrix)
{
   HYPRE_FEMatrix myMatrix;
   myMatrix = (HYPRE_FEMatrix) malloc(sizeof(HYPRE_FEMatrix));
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
/* HYPRE_FEMatrixInitialize                                                  */
/*---------------------------------------------------------------------------*/

extern "C" int
HYPRE_FEMatrixInitialize(HYPRE_FEMatrix matrix)
{
   return 0;
}

/*****************************************************************************/
/* HYPRE_FEMatrixAssemble                                                    */
/*---------------------------------------------------------------------------*/

extern "C" int
HYPRE_FEMatrixAssemble(HYPRE_FEMatrix matrix)
{
   return 0;
}

/*****************************************************************************/
/* HYPRE_FEMatrixSetObjectType                                               */
/*---------------------------------------------------------------------------*/

extern "C" int
HYPRE_FEMatrixSetObjectType(HYPRE_FEMatrix matrix, int type)
{
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
         lsc->copyOutMatrix(1.0e0, dataObj); 
         A = (HYPRE_IJMatrix) dataObj.getDataPtr();
         HYPRE_IJMatrixGetObject(A, (void **) &ACSR);
         (*object) = (void *) ACSR;
      }
   }
   return 0;
}

