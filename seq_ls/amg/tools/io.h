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
Matrix *ReadYSMP P((char *file_name ));
Vector *ReadVec P((char *file_name ));

/* write.c */
void WriteYSMP P((char *file_name , Matrix *matrix ));
void WriteVec P((char *file_name , Vector *vector ));

#undef P


#endif
