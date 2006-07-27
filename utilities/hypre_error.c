/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team <hypre-users@llnl.gov>, UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer and the GNU Lesser General Public License.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the terms and conditions of the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
 ***********************************************************************EHEADER*/


#include "utilities.h"

int hypre__global_error = 0;

/* Process the error with code ierr raised in the given line of the
   given source file. */
void hypre_error_handler(char *filename, int line, int ierr)
{
   hypre_error_flag |= ierr;

#ifdef HYPRE_PRINT_ERRORS
   fprintf(stderr,
           "hypre error in file \"%s\", line %d, error code = %d ",
           filename, line, ierr);
   HYPRE_DescribeError(ierr, stderr);
#endif
}

int HYPRE_GetError()
{
   return hypre_error_flag;
}

void HYPRE_DescribeError(int ierr, FILE *stream)
{
   if (ierr == 0)
      fprintf(stream,"[No error] ");

   if (ierr & HYPRE_ERROR_GENERIC)
      fprintf(stream,"[Generic error] ");

   if (ierr & HYPRE_ERROR_MEMORY)
      fprintf(stream,"[Memory error] ");

   if (ierr & HYPRE_ERROR_ARG)
      fprintf(stream,"[Error in argument %d] ", HYPRE_GetErrorArg());

   if (ierr & HYPRE_ERROR_CONV)
      fprintf(stream,"[Method did not converge] ");

   fprintf(stream,"\n");
}

int HYPRE_GetErrorArg()
{
   return (hypre_error_flag>>3 & 31);
}
