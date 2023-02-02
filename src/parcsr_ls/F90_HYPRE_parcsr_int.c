/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * HYPRE_ParCSRint Fortran interface
 *
 *****************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

HYPRE_Int hypre_ParVectorSize( void *x );
HYPRE_Int aux_maskCount( HYPRE_Int n, hypre_F90_Int *mask );
void aux_indexFromMask( HYPRE_Int n, hypre_F90_Int *mask, hypre_F90_Int *index );

/*--------------------------------------------------------------------------
 * hypre_ParSetRandomValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parsetrandomvalues, HYPRE_PARSETRANDOMVALUES)
(hypre_F90_Obj *v,
 hypre_F90_Int *seed,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ParVectorSetRandomValues(
                hypre_F90_PassObj (HYPRE_ParVector, v),
                hypre_F90_PassInt (seed)));
}

/*--------------------------------------------------------------------------
 * hypre_ParPrintVector
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parprintvector, HYPRE_PARPRINTVECTOR)
(hypre_F90_Obj *v,
 char *file,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( hypre_ParVectorPrint(
                (hypre_ParVector *) v,
                (char *)            file));
}

/*--------------------------------------------------------------------------
 * hypre_ParReadVector
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parreadvector, HYPRE_PARREADVECTOR)
(hypre_F90_Comm *comm,
 char *file,
 hypre_F90_Int *ierr)
{
   *ierr = 0;

   hypre_ParReadVector(
      hypre_F90_PassComm (comm),
      (char *) file );
}

/*--------------------------------------------------------------------------
 * hypre_ParVectorSize
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parvectorsize, HYPRE_PARVECTORSIZE)
(hypre_F90_Obj *x,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( hypre_ParVectorSize(
                (void *) x) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRMultiVectorPrint
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrmultivectorprint, HYPRE_PARCSRMULTIVECTORPRINT)
(hypre_F90_Obj *x,
 char *file,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ParCSRMultiVectorPrint(
                (void *)       x,
                (char *) file));
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRMultiVectorRead
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrmultivectorread, HYPRE_PARCSRMULTIVECTORREAD)
(hypre_F90_Comm *comm,
 hypre_F90_Obj *ii,
 char *file,
 hypre_F90_Int *ierr)
{
   *ierr = 0;

   HYPRE_ParCSRMultiVectorRead(
      hypre_F90_PassComm (comm),
      (void *)       ii,
      (char *) file );
}

/*--------------------------------------------------------------------------
 * HYPRE_TempParCSRSetupInterpreter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_tempparcsrsetupinterprete, HYPRE_TEMPPARCSRSETUPINTERPRETE)
(hypre_F90_Obj *i,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_TempParCSRSetupInterpreter(
                (mv_InterfaceInterpreter *) i ));
}

/*--------------------------------------------------------------------------
 * HYPRE_TempParCSRSetupInterpreter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrsetupinterpreter, HYPRE_PARCSRSETUPINTERPRETER)
(hypre_F90_Obj *i,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ParCSRSetupInterpreter(
                (mv_InterfaceInterpreter *) i ));
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRSetupMatvec
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrsetupmatvec, HYPRE_PARCSRSETUPMATVEC)
(hypre_F90_Obj *mv,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ParCSRSetupMatvec(
                hypre_F90_PassObjRef (HYPRE_MatvecFunctions, mv)));
}
#ifdef __cplusplus
}
#endif
