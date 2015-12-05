/******************************************************************************
 *  FAC relaxation. Refinement patches are solved using system pfmg
 *  relaxation.
 ******************************************************************************/

#include "headers.h"

#define DEBUG 0

int
hypre_FacLocalRelax(void                 *relax_vdata,
                    hypre_SStructPMatrix *A,
                    hypre_SStructPVector *x,
                    hypre_SStructPVector *b,
                    int                   num_relax,
                    int                  *zero_guess)
{
   hypre_SysPFMGRelaxSetPreRelax(relax_vdata);
   hypre_SysPFMGRelaxSetMaxIter(relax_vdata, num_relax);
   hypre_SysPFMGRelaxSetZeroGuess(relax_vdata, *zero_guess);
   hypre_SysPFMGRelax(relax_vdata, A, b, x);
   zero_guess = 0;

   return 0;
}

