/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision$
 ***********************************************************************EHEADER*/




/******************************************************************************
 *
 * HYPRE_ParCSRint Fortran interface
 *
 *****************************************************************************/

#include "headers.h"
#include "fortran.h"

HYPRE_Int hypre_ParVectorSize( void *x );
HYPRE_Int aux_maskCount( HYPRE_Int n, HYPRE_Int *mask );
void aux_indexFromMask( HYPRE_Int n, HYPRE_Int *mask, HYPRE_Int *index );


/*--------------------------------------------------------------------------
 * hypre_ParSetRandomValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parsetrandomvalues, HYPRE_PARSETRANDOMVALUES)
               (hypre_F90_Obj *v, HYPRE_Int *seed, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) ( HYPRE_ParVectorSetRandomValues( (HYPRE_ParVector) *v,
                                                   (HYPRE_Int)             *seed));
}

/*--------------------------------------------------------------------------
 * hypre_ParPrintVector
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parprintvector, HYPRE_PARPRINTVECTOR)
               (hypre_F90_Obj *v, char *file, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) ( hypre_ParVectorPrint( (hypre_ParVector *) v,
                                         (char *)            file));
}

/*--------------------------------------------------------------------------
 * hypre_ParReadVector
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parreadvector, HYPRE_PARREADVECTOR)
               (hypre_F90_Comm *comm, char *file, HYPRE_Int *ierr)
{
   *ierr = 0;

   (void*) (hypre_ParReadVector( (MPI_Comm)    *comm, 
                                (const char *) file ));
}

/*--------------------------------------------------------------------------
 * hypre_ParVectorSize
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parvectorsize, HYPRE_PARVECTORSIZE)
               (hypre_F90_Obj *x, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) ( hypre_ParVectorSize( (void *) x) );
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMultiVectorPrint
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrmultivectorprint, HYPRE_PARCSRMULTIVECTORPRINT)
               (hypre_F90_Obj *x, char *file, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) ( hypre_ParCSRMultiVectorPrint( (void *)       x, 
                                                 (const char *) file));
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMultiVectorRead
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrmultivectorread, HYPRE_PARCSRMULTIVECTORREAD)
               (hypre_F90_Comm *comm, hypre_F90_Obj *ii, char *file, HYPRE_Int *ierr)
{
   *ierr = 0;

   (void *) hypre_ParCSRMultiVectorRead( (MPI_Comm)    *comm,
                                         (void *)       ii, 
                                         (const char *) file );
}

/*--------------------------------------------------------------------------
 * aux_maskCount
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(aux_maskcount, AUX_MASKCOUNT)
               (HYPRE_Int *n, HYPRE_Int *mask, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) ( aux_maskCount( (HYPRE_Int)   *n,
                                  (HYPRE_Int *)  mask ));
}

/*--------------------------------------------------------------------------
 * aux_indexFromMask
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(aux_indexfrommask, AUX_INDEXFROMMASK)
               (HYPRE_Int *n, HYPRE_Int *mask, HYPRE_Int *index, HYPRE_Int *ierr)
{
   *ierr = 0;

  (void) ( aux_indexFromMask( (HYPRE_Int)   *n, 
                              (HYPRE_Int *)  mask,
                              (HYPRE_Int *)  index ));
}

/*--------------------------------------------------------------------------
 * HYPRE_TempParCSRSetupInterpreter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_tempparcsrsetupinterprete, HYPRE_TEMPPARCSRSETUPINTERPRETE)
               (hypre_F90_Obj *i, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) ( HYPRE_TempParCSRSetupInterpreter( (mv_InterfaceInterpreter *) i ));
}

/*--------------------------------------------------------------------------
 * HYPRE_TempParCSRSetupInterpreter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrsetupinterpreter, HYPRE_PARCSRSETUPINTERPRETER)
               (hypre_F90_Obj *i, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) ( HYPRE_ParCSRSetupInterpreter( (mv_InterfaceInterpreter *) i ));
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRSetupMatvec
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrsetupmatvec, HYPRE_PARCSRSETUPMATVEC)
               (hypre_F90_Obj *mv, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) ( HYPRE_ParCSRSetupMatvec( (HYPRE_MatvecFunctions *) mv));
}
