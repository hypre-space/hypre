/*BHEADER**********************************************************************
 * (c) 1997   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * Header file for HYPRE library
 *
 *****************************************************************************/

#ifndef HYPRE_HEADER
#define HYPRE_HEADER


/*--------------------------------------------------------------------------
 * Structures
 *--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------
 * Constants
 *--------------------------------------------------------------------------*/

#define HYPRE_ISIS_MATRIX 11

#define HYPRE_PETSC_MATRIX 12

#define HYPRE_PARCSR_MATRIX 13

#define HYPRE_PETSC_VECTOR 33

#define HYPRE_PETSC_MAT_PARILUT_SOLVER 22

#define HYPRE_PARILUT      872

#define HYPRE_UNITIALIZED -47

/*--------------------------------------------------------------------------
 * Prototypes
 *--------------------------------------------------------------------------*/

#include "utilities/HYPRE_utilities.h"

#ifdef __STDC__
# define	P(s) s
#else
# define P(s) ()
#endif

#undef P

#endif
