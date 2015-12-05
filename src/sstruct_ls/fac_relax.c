/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team. UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer, contact information and the GNU Lesser General Public License.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the 
 * terms of the GNU General Public License (as published by the Free Software
 * Foundation) version 2.1 dated February 1999.
 *
 * HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
 * WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
 * FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision: 2.3 $
 ***********************************************************************EHEADER*/



/******************************************************************************
 *  FAC relaxation. Refinement patches are solved using system pfmg
 *  relaxation.
 ******************************************************************************/

#include "headers.h"

#define DEBUG 0

int
hypre_FacLocalRelax(void                 *relax_vdata,
                    hypre_SStructPMatrix *A,
                    hypre_SStructPVector *x,
                    hypre_SStructPVector *b,
                    int                   num_relax,
                    int                  *zero_guess)
{
   hypre_SysPFMGRelaxSetPreRelax(relax_vdata);
   hypre_SysPFMGRelaxSetMaxIter(relax_vdata, num_relax);
   hypre_SysPFMGRelaxSetZeroGuess(relax_vdata, *zero_guess);
   hypre_SysPFMGRelax(relax_vdata, A, b, x);
   zero_guess = 0;

   return 0;
}

