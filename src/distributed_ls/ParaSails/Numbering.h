/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.4 $
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
