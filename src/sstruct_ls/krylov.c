/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.5 $
 ***********************************************************************EHEADER*/

#include "_hypre_utilities.h"

/******************************************************************************
 *
 *
 *****************************************************************************/

HYPRE_Int hypre_SStructKrylovCopyVector( void *x, void *y );

/*--------------------------------------------------------------------------
 * hypre_SStructKrylovIdentitySetup
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructKrylovIdentitySetup( void *vdata,
                           void *A,
                           void *b,
                           void *x )

{
   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_SStructKrylovIdentity
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructKrylovIdentity( void *vdata,
                      void *A,
                      void *b,
                      void *x )

{
   return( hypre_SStructKrylovCopyVector(b, x) );
}

