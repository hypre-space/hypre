/*BHEADER**********************************************************************
 * (c) 1997   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

#if HYPRE_USING_PGCC_SMP
#define HYPRE_SMP_PRIVATE HYPRE_BOX_SMP_PRIVATE,hypre_block
#include "../utilities/hypre_smp_forloop.h"
#else
#define HYPRE_SMP_PRIVATE HYPRE_BOX_SMP_PRIVATE
#include "../utilities/hypre_smp_forloop.h"
#endif
#undef HYPRE_BOX_SMP_PRIVATE

