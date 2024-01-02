/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_utilities.h"

/* Global variable for error handling */
hypre_Error hypre__global_error = {0, 0, 0, NULL, 0, 0};

/*--------------------------------------------------------------------------
 * Process the error raised on the given line of the given source file
 *--------------------------------------------------------------------------*/

void
hypre_error_handler(const char *filename, HYPRE_Int line, HYPRE_Int ierr, const char *msg)
{
   /* Copy global struct into a short name and copy changes back before exiting */
   hypre_Error err = hypre__global_error;

   /* Store the error code */
   err.error_flag |= ierr;

#if defined(HYPRE_PRINT_ERRORS)

   /* Error format strings without and with a message */
   const char  fmt_wo[] = "hypre error in file \"%s\", line %d, error code = %d\n";
   const char  fmt_wm[] = "hypre error in file \"%s\", line %d, error code = %d - %s\n";
   char       *buffer;
   HYPRE_Int   bufsz;

   /* Print error message to local buffer first */

   if (msg)
   {
      bufsz = hypre_snprintf(NULL, 0, fmt_wm, filename, line, ierr, msg);
   }
   else
   {
      bufsz = hypre_snprintf(NULL, 0, fmt_wo, filename, line, ierr);
   }

   bufsz += 1;
   buffer = hypre_TAlloc(char, bufsz, HYPRE_MEMORY_HOST);

   if (msg)
   {
      hypre_snprintf(buffer, bufsz, fmt_wm, filename, line, ierr, msg);
   }
   else
   {
      hypre_snprintf(buffer, bufsz, fmt_wo, filename, line, ierr);
   }

   /* Now print buffer to either memory or stderr */
   if (err.print_to_memory)
   {
      HYPRE_Int  msg_sz = err.msg_sz; /* Store msg_sz for snprintf below */

      /* Make sure there is enough memory for the new message */
      err.msg_sz += bufsz;
      if ( err.msg_sz > err.mem_sz )
      {
         err.mem_sz = err.msg_sz + 1024; /* Add some excess */
         err.memory = hypre_TReAlloc(err.memory, char, err.mem_sz, HYPRE_MEMORY_HOST);
      }

      hypre_snprintf((err.memory + msg_sz), bufsz, "%s", buffer);
   }
   else
   {
      hypre_fprintf(stderr, "%s", buffer);
   }

   /* Free buffer */
   hypre_TFree(buffer, HYPRE_MEMORY_HOST);
#else
   HYPRE_UNUSED_VAR(filename);
   HYPRE_UNUSED_VAR(line);
   HYPRE_UNUSED_VAR(msg);
#endif /* if defined(HYPRE_PRINT_ERRORS) */

   hypre__global_error = err;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void
hypre_error_code_save(void)
{
   /* Store the current error code in a temporary variable */
   hypre_error_temp_flag = hypre_error_flag;

   /* Reset current error code */
   HYPRE_ClearAllErrors();
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void
hypre_error_code_restore(void)
{
   /* Restore hypre's error code */
   hypre_error_flag = hypre_error_temp_flag;

   /* Reset temporary error code */
   hypre_error_temp_flag = 0;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_GetGlobalError(MPI_Comm comm)
{
   HYPRE_Int global_error_flag;

   hypre_MPI_Allreduce(&hypre_error_flag, &global_error_flag, 1,
                       HYPRE_MPI_INT, hypre_MPI_BOR, comm);

   return global_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_GetError(void)
{
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_CheckError(HYPRE_Int ierr, HYPRE_Int hypre_error_code)
{
   return ierr & hypre_error_code;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void
HYPRE_DescribeError(HYPRE_Int ierr, char *msg)
{
   if (ierr == 0)
   {
      hypre_sprintf(msg, "[No error] ");
   }

   if (ierr & HYPRE_ERROR_GENERIC)
   {
      hypre_sprintf(msg, "[Generic error] ");
   }

   if (ierr & HYPRE_ERROR_MEMORY)
   {
      hypre_sprintf(msg, "[Memory error] ");
   }

   if (ierr & HYPRE_ERROR_ARG)
   {
      hypre_sprintf(msg, "[Error in argument %d] ", HYPRE_GetErrorArg());
   }

   if (ierr & HYPRE_ERROR_CONV)
   {
      hypre_sprintf(msg, "[Method did not converge] ");
   }
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_GetErrorArg(void)
{
   return (hypre_error_flag >> 3 & 31);
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ClearAllErrors(void)
{
   hypre_error_flag = 0;
   return (hypre_error_flag != 0);
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ClearError(HYPRE_Int hypre_error_code)
{
   hypre_error_flag &= ~hypre_error_code;
   return (hypre_error_flag & hypre_error_code);
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SetPrintErrorMode(HYPRE_Int mode)
{
   hypre__global_error.print_to_memory = mode;
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_GetErrorMessages(char **buffer, HYPRE_Int *bufsz)
{
   hypre_Error err = hypre__global_error;

   *bufsz  = err.msg_sz;
   *buffer = hypre_CTAlloc(char, *bufsz, HYPRE_MEMORY_HOST);
   hypre_TMemcpy(*buffer, err.memory, char, *bufsz, HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);

   hypre_TFree(err.memory, HYPRE_MEMORY_HOST);
   err.mem_sz = 0;
   err.msg_sz = 0;

   hypre__global_error = err;
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_PrintErrorMessages(MPI_Comm comm)
{
   hypre_Error err = hypre__global_error;

   HYPRE_Int myid;
   char *msg;

   hypre_MPI_Barrier(comm);

   hypre_MPI_Comm_rank(comm, &myid);
   for (msg = err.memory; msg < (err.memory + err.msg_sz); msg += strlen(msg) + 1)
   {
      hypre_fprintf(stderr, "%d: %s", myid, msg);
   }

   hypre_TFree(err.memory, HYPRE_MEMORY_HOST);
   err.mem_sz = 0;
   err.msg_sz = 0;

   hypre__global_error = err;
   return hypre_error_flag;
}
