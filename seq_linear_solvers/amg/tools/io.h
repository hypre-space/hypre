/*BHEADER**********************************************************************
 * (c) 1996   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

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
