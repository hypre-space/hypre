/*BHEADER**********************************************************************
 * (c) 1999   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * Numbering.h header file.
 *
 *****************************************************************************/

#include <stdio.h>
#include "Common.h"
/*#include "Matrix.h"*/
#include "Hash.h"

#ifndef _NUMBERING_H
#define _NUMBERING_H

typedef struct
{
    int   size;    /* max number of indices that can be stored */
    int   beg_row;
    int   end_row;
    int   num_loc; /* number of local indices */
    int   num_ind; /* number of indices */
    int   num_ext; /* number of neighbor nodes */

    int  *local_to_global;
    int  *global_to_local;
    Hash *hash;
}
Numbering;

Numbering *NumberingCreate(int size, int beg_row, int end_row);
void NumberingDestroy(Numbering *numb);
void NumberingLocalToGlobal(Numbering *numb, int len, int *local, int *global);
void NumberingGlobalToLocal(Numbering *numb, int len, int *global, int *local);

#endif /* _NUMBERING_H */
