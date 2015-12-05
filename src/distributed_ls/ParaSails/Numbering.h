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
 * $Revision: 2.2 $
 ***********************************************************************EHEADER*/



/******************************************************************************
 *
 * Numbering.h header file.
 *
 *****************************************************************************/

#include <stdio.h>
#include "Common.h"
#include "Matrix.h"
#include "Hash.h"

#ifndef _NUMBERING_H
#define _NUMBERING_H

struct numbering
{
    int   size;    /* max number of indices that can be stored */
    int   beg_row;
    int   end_row;
    int   num_loc; /* number of local indices */
    int   num_ind; /* number of indices */

    int  *local_to_global;
    Hash *hash;
};

typedef struct numbering Numbering;

Numbering *NumberingCreate(Matrix *m, int size);
Numbering *NumberingCreateCopy(Numbering *orig);
void NumberingDestroy(Numbering *numb);
void NumberingLocalToGlobal(Numbering *numb, int len, int *local, int *global);
void NumberingGlobalToLocal(Numbering *numb, int len, int *global, int *local);

#endif /* _NUMBERING_H */
