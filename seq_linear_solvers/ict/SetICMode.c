#include "general.h"

#include "ic_data.h"

/*--------------------------------------------------------------------------
 * SetICMode
 *--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------
 * Purpose:       Communicate to factorization and solver routines choice of
 *                mode that routines should be used, in the context mainly
 *                of memory usage.

 * Author:        Andy Cleary
 *                Centre for Applied Scientific Computing
 *                Lawrence Livermore Labs
 * Revision History:
 *  5-22-97: AJC. Original Version.
 *--------------------------------------------------------------------------*/


int SetICMode( void * data, int mode )
     /* Mode =1: a copy of the input matrix will be made and into the 
        factorization routine because the
        original should not be overwritten.  
        At the end of ic_setup,
        the copy will be discarded. 
        For eaither mode=1 or mode=2, the original matrix should not be
        discarded or modified between ic_setup and ic_solve.

        Mode =2: the original
        input matrix can be overwritten.
        */
{
   ICData    *ic_data;

   ic_data = (ICData *) data;

   if( (mode != 1) && (mode != 2) )
     {
     return(-1);
     }

   ICDataMode(ic_data) = mode;

   return(0);
}

