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

#include "headers.h"

int
hypre_ParVectorZeroBCValues(hypre_ParVector *v,
                            int             *rows,
                            int              nrows)
{
   int   ierr= 0;

   hypre_Vector *v_local = hypre_ParVectorLocalVector(v);

   hypre_SeqVectorZeroBCValues(v_local, rows, nrows);

   return ierr;
}

int
hypre_SeqVectorZeroBCValues(hypre_Vector *v,
                            int          *rows,
                            int           nrows)
{
   double  *vector_data = hypre_VectorData(v);
   int      i;
                                                                                                    
   int      ierr  = 0;
                                                                                                    
#define HYPRE_SMP_PRIVATE i
#include "../utilities/hypre_smp_forloop.h"
   for (i = 0; i < nrows; i++)
      vector_data[rows[i]]= 0.0;
                                                                                                    
   return ierr;
}

