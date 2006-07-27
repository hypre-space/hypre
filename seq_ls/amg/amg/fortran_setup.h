/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team <hypre-users@llnl.gov>, UCRL-CODE-222953.
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

void hypre_NAME_FORTRAN_FOR_C(setup)(int *, int *, int *,
			       double *, int *, double *, int *,
			       int *, int *, int *, int *, int *,
			       double *, int *, int *,
			       int *, int *, int *, int *,
			       double *, int *, int *,
			       int *, int *, int *,
			       int *, int *, int *, int *,
			       char *, long);


/* idec */
void hypre_NAME_FORTRAN_FOR_C(idec)(int *, int *, int *, int *);

