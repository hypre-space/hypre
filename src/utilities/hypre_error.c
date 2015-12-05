/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team, UCRL-CODE-222953.
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
 * $Revision: 2.5 $
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
#endif
}

int HYPRE_GetError()
{
   return hypre_error_flag;
}

int HYPRE_CheckError(int ierr, int hypre_error_code)
{
   return ierr & hypre_error_code;
}

void HYPRE_DescribeError(int ierr, char *msg)
{
   if (ierr == 0)
      sprintf(msg,"[No error] ");

   if (ierr & HYPRE_ERROR_GENERIC)
      sprintf(msg,"[Generic error] ");

   if (ierr & HYPRE_ERROR_MEMORY)
      sprintf(msg,"[Memory error] ");

   if (ierr & HYPRE_ERROR_ARG)
      sprintf(msg,"[Error in argument %d] ", HYPRE_GetErrorArg());

   if (ierr & HYPRE_ERROR_CONV)
      sprintf(msg,"[Method did not converge] ");
}

int HYPRE_GetErrorArg()
{
   return (hypre_error_flag>>3 & 31);
}
