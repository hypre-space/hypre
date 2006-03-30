/*BHEADER**********************************************************************
 * (c) 1997   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

#ifndef hypre_ERROR_HEADER
#define hypre_ERROR_HEADER

/*--------------------------------------------------------------------------
 * HYPRE error codes
 *--------------------------------------------------------------------------*/

#define HYPRE_ERROR_GENERIC      1<<0   /* generic error */
#define HYPRE_ERROR_MEMORY       1<<1   /* unable to allocate memory */
#define HYPRE_ERROR_ARG          1<<2   /* argument error */
/* bits 4-8 are reserved for the index of the argument error */
#define HYPRE_ERROR_CONV         1<<8   /* method did not converge as expected */

/*--------------------------------------------------------------------------
 * Global variable used in hypre error checking
 *--------------------------------------------------------------------------*/

extern int hypre__global_error;
#define hypre_error_flag  hypre__global_error

/*--------------------------------------------------------------------------
 * HYPRE error macros
 *--------------------------------------------------------------------------*/

void hypre_error_handler(char *filename, int line, int ierr);
#define hypre_error(IERR)  hypre_error_handler(__FILE__, __LINE__, IERR)
#define hypre_error_in_arg(IARG)  hypre_error(HYPRE_ERROR_ARG | IARG<<3)
#ifdef NDEBUG
#define hypre_assert(EX)
#else
#define hypre_assert(EX) if (!(EX)) hypre_error(1)
#endif

#endif
