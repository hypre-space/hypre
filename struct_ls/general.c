/*BHEADER**********************************************************************
 * (c) 1997   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_Log2:
 *   This routine returns the integer, floor(log_2(p)).
 *   If p <= 0, it returns a -1.
 *--------------------------------------------------------------------------*/

int
hypre_Log2(int p)
{
   int  e;

   if (p <= 0)
      return -1;

   e = 0;
   while (p > 1)
   {
      e += 1;
      p /= 2;
   }
 
   return  e;
}
