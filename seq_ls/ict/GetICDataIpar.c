#include "general.h"

#include "ic_data.h"

/*--------------------------------------------------------------------------
 * Purpose:       Routines for setting parameters to incomplete 
 *                factorization. Applications should use these routines
 *                to set the appropriate parameters so that they do not
 *                have to have any knowledge or dependence on the data
 *                structures internal to the factor and solve routines.

 * Author:        Andy Cleary
 *                Centre for Applied Scientific Computing
 *                Lawrence Livermore Labs
 * Revision History:
 *  5-22-97: AJC. Original Version.
 *--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------
 * GetICDataIpar
 *--------------------------------------------------------------------------*/

int *GetICDataIpar( void * data )
     /* Returns a pointer to the beginning of the integer vector that holds
        integer parameters (many optional) to the incomplete factorization
        and solve codes.
        For a description of each parameter and its default values, please
        see ic_initialize.
        This routine should be called *after* ict_initialize, but
        *before* ic_setup.
        Element i of the IPAR vector can be accessed as
        ICDataIpar(data)[i],
        where "data" is the void * pointer to the solver structure.
      */
{
   ICData    *ic_data;

   ic_data = (ICData *) data;

   return( ICDataIpar(ic_data) );
}

/*--------------------------------------------------------------------------
 * GetICDataRpar
 *--------------------------------------------------------------------------*/

double *GetICDataRpar( void * data )
     /* Returns a pointer to the beginning of the vector of doubles that holds
        floating point parameters (some optional) to the incomplete factorization
        and solve codes.
        For a description of each parameter and its default values, please
        see ict_initialize.
        This routine should be called *after* ict_initialize, but
        *before* ic_setup.
        Element i of the RPAR vector can be accessed as
        ICDataRpar(data)[i],
        where "data" is the void * pointer to the solver structure.
      */

{
   ICData    *ic_data;

   ic_data = (ICData *) data;

   return( ICDataRpar(ic_data) );
}

