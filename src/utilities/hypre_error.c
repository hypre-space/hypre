/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.14 $
 ***********************************************************************EHEADER*/



#include "_hypre_utilities.h"

HYPRE_Int hypre__global_error = 0;

/* Process the error with code ierr raised in the given line of the
   given source file. */
void hypre_error_handler(const char *filename, HYPRE_Int line, HYPRE_Int ierr, const char *msg)
{
   hypre_error_flag |= ierr;

#ifdef HYPRE_PRINT_ERRORS
   if (msg)
   {
      hypre_fprintf(
         stderr, "hypre error in file \"%s\", line %d, error code = %d - %s\n",
         filename, line, ierr, msg);
   }
   else
   {
      hypre_fprintf(
         stderr, "hypre error in file \"%s\", line %d, error code = %d\n",
         filename, line, ierr);
   }
#endif
}

HYPRE_Int HYPRE_GetError()
{
   return hypre_error_flag;
}

HYPRE_Int HYPRE_CheckError(HYPRE_Int ierr, HYPRE_Int hypre_error_code)
{
   return ierr & hypre_error_code;
}

void HYPRE_DescribeError(HYPRE_Int ierr, char *msg)
{
   if (ierr == 0)
      hypre_sprintf(msg,"[No error] ");

   if (ierr & HYPRE_ERROR_GENERIC)
      hypre_sprintf(msg,"[Generic error] ");

   if (ierr & HYPRE_ERROR_MEMORY)
      hypre_sprintf(msg,"[Memory error] ");

   if (ierr & HYPRE_ERROR_ARG)
      hypre_sprintf(msg,"[Error in argument %d] ", HYPRE_GetErrorArg());

   if (ierr & HYPRE_ERROR_CONV)
      hypre_sprintf(msg,"[Method did not converge] ");
}

HYPRE_Int HYPRE_GetErrorArg()
{
   return (hypre_error_flag>>3 & 31);
}

HYPRE_Int HYPRE_ClearAllErrors()
{
   hypre_error_flag = 0;
   return (hypre_error_flag != 0);
}

HYPRE_Int HYPRE_ClearError(HYPRE_Int hypre_error_code)
{
   hypre_error_flag &= ~hypre_error_code;
   return (hypre_error_flag & hypre_error_code);
}
