/*BHEADER**********************************************************************
 * (c) 2000   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

/******************************************************************************
 *
 * krylov solver headers
 *
 *****************************************************************************/

#ifndef HYPRE_KRYLOV_HEADER
#define HYPRE_KRYLOV_HEADER

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#ifndef max
#define max(a,b)  (((a)<(b)) ? (b) : (a))
#endif

#define hypre_CTAllocF(type, count, funcs) \
( (type *)(*(funcs->CAlloc))\
((unsigned int)(count), (unsigned int)sizeof(type)) )

#define hypre_TFreeF( ptr, funcs ) \
( (*(funcs->Free))((char *)ptr), ptr = NULL )

/* A pointer to a type which is never defined, sort of works like void* ... */
#ifndef HYPRE_SOLVER_STRUCT
#define HYPRE_SOLVER_STRUCT
struct hypre_Solver_struct;
typedef struct hypre_Solver_struct *HYPRE_Solver;
#endif

#include "pcg.h"
#include "gmres.h"
#include "bicgstab.h"
#include "cgnr.h"

#endif
