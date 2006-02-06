/*BHEADER**********************************************************************
 * (c) 1999   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

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
   return (hypre_error_flag>>3 & 7);
}
