/*BHEADER**********************************************************************
 * (c) 1998   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "../../utilities/memory.h"
/* #include "../utilities/timing.h" */
#include "../../seq_matrix_vector/headers.h"

/* #include "HYPRE_amg.h" */

#include "general.h" 

#include "amg.h"

#include "protos.h"

#ifdef HYPRE_DEBUG
#include <gmalloc.h>
#endif

