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
/*#include "ZZZ_mv.h"*/

#include "general.h"

#include "box.h"
#include "sbox.h"

#include "struct_stencil.h"
#include "struct_grid.h"

#include "communication.h"
#include "computation.h"

#include "struct_matrix.h"
#include "struct_vector.h"

#include "protos.h"

#ifdef ZZZ_MALLOC_DEBUG
#include <gmalloc.h>
#endif

