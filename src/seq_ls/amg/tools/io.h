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
 * Header info for IO facilities
 *
 *****************************************************************************/

#ifndef _IO_HEADER
#define _IO_HEADER

#include <stdio.h>

#include "general.h"
#include "matrix.h"
#include "vector.h"

/*--------------------------------------------------------------------------
 * Prototypes
 *--------------------------------------------------------------------------*/

#ifdef __STDC__
# define        P(s) s
#else
# define P(s) ()
#endif


/* read.c */
hypre_Matrix *ReadYSMP P((char *file_name ));
hypre_Vector *ReadVec P((char *file_name ));

/* write.c */
void hypre_WriteYSMP P((char *file_name , hypre_Matrix *matrix ));
void hypre_WriteVec P((char *file_name , hypre_Vector *vector ));

#undef P


#endif
