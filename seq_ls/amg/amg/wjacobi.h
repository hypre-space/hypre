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
 * Header info for WJacobi solver
 *
 *****************************************************************************/

#ifndef _WJACOBI_HEADER
#define _WJACOBI_HEADER


/*--------------------------------------------------------------------------
 * WJacobiData
 *--------------------------------------------------------------------------*/

typedef struct
{
   double   weight;
   int      max_iter;

   Matrix  *A;
   Vector  *t;

   char    *log_file_name;

} WJacobiData;

/*--------------------------------------------------------------------------
 * Accessor functions for the WJacobiData structure
 *--------------------------------------------------------------------------*/

#define WJacobiDataWeight(wjacobi_data)      ((wjacobi_data) -> weight)
#define WJacobiDataMaxIter(wjacobi_data)     ((wjacobi_data) -> max_iter)

#define WJacobiDataA(wjacobi_data)           ((wjacobi_data) -> A)
#define WJacobiDataT(wjacobi_data)           ((wjacobi_data) -> t)

#define WJacobiDataLogFileName(wjacobi_data) ((wjacobi_data) -> log_file_name)


#endif
