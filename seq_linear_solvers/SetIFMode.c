#include "general.h"

#include "incfact_data.h"

/*--------------------------------------------------------------------------
 * SetINCFACTMode
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


int SetINCFACTMode( void * data, int mode )
     /* Mode =1: a copy of the input matrix will be made and into the 
        factorization routine because the
        original should not be overwritten.  
        At the end of incfact_setup,
        the copy will be discarded. 
        For eaither mode=1 or mode=2, the original matrix should not be
        discarded or modified between incfact_setup and incfact_solve.

        Mode =2: the original
        input matrix can be overwritten.
        */
{
   INCFACTData    *incfact_data;

   incfact_data = (INCFACTData *) data;

   if( (mode != 1) && (mode != 2) )
     {
     return(-1);
     }

   INCFACTDataMode(incfact_data) = mode;

   return(0);
}

