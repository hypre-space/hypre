/*BHEADER**********************************************************************
 * (c) 1998   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

#include "headers.h"
#include "fortran.h"
#include "HYPRE_struct_int.h"

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorSetRandomValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structvectorsetrandomvalu, HYPRE_STRUCTVECTORSETRANDOMVALU)
               (long int *vector, int *seed, int *ierr)

{
   *ierr = (int) ( hypre_StructVectorSetRandomValues( (hypre_StructVector *) vector,
                                                      (int)                 *seed ));
}

int
hypre_StructSetRandomValues( void* v, int seed ) {

  return hypre_StructVectorSetRandomValues( (hypre_StructVector*)v, seed );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSetupInterpreter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsetupinterpreter, HYPRE_STRUCTSETUPINTERPRETER)
               (long int *i, int *ierr)

{
   *ierr = (int) ( HYPRE_StructSetupInterpreter( (mv_InterfaceInterpreter *) i ));
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSetupMatvec
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsetupmatvec, HYPRE_STRUCTSETUPMATVEC)
               (long int *mv, int *ierr)

{
   *ierr = (int) ( HYPRE_StructSetupMatvec( (HYPRE_MatvecFunctions *) mv));
}
