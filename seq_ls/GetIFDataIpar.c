#include "general.h"

#include "incfact_data.h"

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
 * GetINCFACTDataIpar
 *--------------------------------------------------------------------------*/

int *GetINCFACTDataIpar( void * data )
     /* Returns a pointer to the beginning of the integer vector that holds
        integer parameters (many optional) to the incomplete factorization
        and solve codes.
        For a description of each parameter and its default values, please
        see incfact_initialize.
        This routine should be called *after* incfactt_initialize, but
        *before* incfact_setup.
        Element i of the IPAR vector can be accessed as
        INCFACTDataIpar(data)[i],
        where "data" is the void * pointer to the solver structure.
      */
{
   INCFACTData    *incfact_data;

   incfact_data = (INCFACTData *) data;

   return( INCFACTDataIpar(incfact_data) );
}

/*--------------------------------------------------------------------------
 * GetINCFACTDataRpar
 *--------------------------------------------------------------------------*/

double *GetINCFACTDataRpar( void * data )
     /* Returns a pointer to the beginning of the vector of doubles that holds
        floating point parameters (some optional) to the incomplete factorization
        and solve codes.
        For a description of each parameter and its default values, please
        see incfactt_initialize.
        This routine should be called *after* incfactt_initialize, but
        *before* incfact_setup.
        Element i of the RPAR vector can be accessed as
        INCFACTDataRpar(data)[i],
        where "data" is the void * pointer to the solver structure.
      */

{
   INCFACTData    *incfact_data;

   incfact_data = (INCFACTData *) data;

   return( INCFACTDataRpar(incfact_data) );
}

