/*BHEADER**********************************************************************
 * (c) 1997   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

#include <../HYPRE_config.h>

#include "../utilities/general.h"

#ifdef HYPRE_DEBUG
#include <gmalloc.h>
#endif

#include "../utilities/utilities.h"

#include "./box.h"
#include "./struct_stencil.h"
#include "./struct_grid.h"

#include "./grid_to_coord.h"
#include "./struct_matrix.h"
#include "./struct_vector.h"

#include "../HYPRE.h"

#include "./HYPRE_CI_struct_matrix_vector_types.h"

#include "./hypre_protos.h"
#include "./internal_protos.h"


