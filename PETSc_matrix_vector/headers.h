/*BHEADER**********************************************************************
 * (c) 1997   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

#include <stdio.h>
#include <math.h>

#include "mpi.h"
#include "ZZZ.h"

#include "general.h"

#include "matrix.h"

#include "box.h"
#include "struct_stencil.h"
#include "struct_grid.h"

#include "grid_to_coord.h"
#include "struct_matrix.h"
#include "struct_vector.h"
#include "struct_solver.h"

/* PETSc matrix and solver prototypes */
#include "PETSc_MV.h"

#include "PETSc_BP.h"

/* malloc debug stuff */
/* #include <gmalloc.h> */

