/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.5 $
 ***********************************************************************EHEADER*/




/******************************************************************************
 *
 * ConjGrad.h header file.
 *
 *****************************************************************************/

#ifndef _CONJGRAD_H
#define _CONJGRAD_H

void PCG_ParaSails(Matrix *mat, ParaSails *ps, double *b, double *x,
   double tol, HYPRE_Int max_iter);
void FGMRES_ParaSails(Matrix *mat, ParaSails *ps, double *b, double *x,
   HYPRE_Int dim, double tol, HYPRE_Int max_iter);

#endif /* _CONJGRAD_H */
