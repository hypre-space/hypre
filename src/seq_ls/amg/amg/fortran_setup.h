/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision$
 ***********************************************************************EHEADER*/





/******************************************************************************
 *
 * C/Fortran interface macros
 *
 *****************************************************************************/

#include "fortran.h"

/* setup */
#define hypre_CALL_SETUP(Setup_err_flag, A, amg_data) \
hypre_NAME_FORTRAN_FOR_C(setup)(&Setup_err_flag,\
			  &hypre_AMGDataNumLevels(amg_data),\
			  &hypre_AMGDataNSTR(amg_data),\
			  &hypre_AMGDataECG(amg_data),\
			  &hypre_AMGDataNCG(amg_data),\
			  &hypre_AMGDataEWT(amg_data),\
			  &hypre_AMGDataNWT(amg_data),\
			  hypre_AMGDataICDep(amg_data),\
			  &hypre_AMGDataIOutDat(amg_data),\
			  &hypre_AMGDataNumUnknowns(amg_data),\
			  hypre_AMGDataIMin(amg_data),\
			  hypre_AMGDataIMax(amg_data),\
			  hypre_MatrixData(A),\
			  hypre_MatrixIA(A),\
			  hypre_MatrixJA(A),\
			  hypre_AMGDataIU(amg_data),\
			  hypre_AMGDataIP(amg_data),\
			  hypre_AMGDataICG(amg_data),\
			  hypre_AMGDataIFG(amg_data),\
			  hypre_MatrixData(hypre_AMGDataP(amg_data)),\
			  hypre_MatrixIA(hypre_AMGDataP(amg_data)),\
			  hypre_MatrixJA(hypre_AMGDataP(amg_data)),\
			  hypre_AMGDataIPMN(amg_data),\
			  hypre_AMGDataIPMX(amg_data),\
			  hypre_AMGDataIV(amg_data),\
			  &hypre_AMGDataNDIMU(amg_data),\
			  &hypre_AMGDataNDIMP(amg_data),\
			  &hypre_AMGDataNDIMA(amg_data),\
			  &hypre_AMGDataNDIMB(amg_data),\
			  hypre_AMGDataLogFileName(amg_data),\
			  strlen(hypre_AMGDataLogFileName(amg_data)))

void hypre_NAME_FORTRAN_FOR_C(setup)(HYPRE_Int *, HYPRE_Int *, HYPRE_Int *,
			       HYPRE_Real *, HYPRE_Int *, HYPRE_Real *, HYPRE_Int *,
			       HYPRE_Int *, HYPRE_Int *, HYPRE_Int *, HYPRE_Int *, HYPRE_Int *,
			       HYPRE_Real *, HYPRE_Int *, HYPRE_Int *,
			       HYPRE_Int *, HYPRE_Int *, HYPRE_Int *, HYPRE_Int *,
			       HYPRE_Real *, HYPRE_Int *, HYPRE_Int *,
			       HYPRE_Int *, HYPRE_Int *, HYPRE_Int *,
			       HYPRE_Int *, HYPRE_Int *, HYPRE_Int *, HYPRE_Int *,
			       char *, hypre_longint);


/* idec */
void hypre_NAME_FORTRAN_FOR_C(idec)(HYPRE_Int *, HYPRE_Int *, HYPRE_Int *, HYPRE_Int *);

