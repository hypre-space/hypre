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
 * Constructors and destructors for stencil structure.
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * zzz_NewStructStencil
 *--------------------------------------------------------------------------*/

zzz_StructStencil *
zzz_NewStructStencil( int         dim,
                      int         size,
                      zzz_Index **shape )
{
   zzz_StructStencil   *stencil;

   stencil = zzz_TAlloc(zzz_StructStencil, 1);

   zzz_StructStencilShape(stencil) = shape;
   zzz_StructStencilSize(stencil)  = size;
   zzz_StructStencilDim(stencil)   = dim;

   return stencil;
}

/*--------------------------------------------------------------------------
 * zzz_FreeStructStencil
 *--------------------------------------------------------------------------*/

void 
zzz_FreeStructStencil( zzz_StructStencil *stencil )
{
   int  i;

   if (stencil)
   {
      for (i = 0; i < zzz_StructStencilSize(stencil); i++)
         zzz_FreeIndex(zzz_StructStencilShape(stencil)[i]);
      zzz_TFree(zzz_StructStencilShape(stencil));
      zzz_TFree(stencil);
   }
}

/*--------------------------------------------------------------------------
 * zzz_StructStencilElementRank
 *    Returns the rank of the `stencil_element' in `stencil'.
 *    If the element is not found, a -1 is returned.
 *--------------------------------------------------------------------------*/

int
zzz_StructStencilElementRank( zzz_StructStencil *stencil,
                              zzz_Index         *stencil_element )
{
   zzz_Index **stencil_shape;
   int         rank;
   int         i;

   rank = -1;
   stencil_shape = zzz_StructStencilShape(stencil);
   for (i = 0; i < zzz_StructStencilSize(stencil); i++)
   {
      if ((zzz_IndexX(stencil_shape[i]) == zzz_IndexX(stencil_element)) &&
          (zzz_IndexY(stencil_shape[i]) == zzz_IndexY(stencil_element)) &&
          (zzz_IndexZ(stencil_shape[i]) == zzz_IndexZ(stencil_element))   )
      {
         rank = i;
         break;
      }
   }

   return rank;
}

/*--------------------------------------------------------------------------
 * zzz_SymmetrizeStructStencil:
 *    Computes a new "symmetrized" stencil.
 *
 *    An integer array called `symm_elements' is also set up.  A non-negative
 *    value of `symm_elements[i]' indicates that the `i'th stencil element
 *    is a "symmetric element".  That is, this stencil element is the
 *    transpose element of an element that is not a "symmetric element".
 *--------------------------------------------------------------------------*/

int
zzz_SymmetrizeStructStencil( zzz_StructStencil  *stencil,
                             zzz_StructStencil **symm_stencil_ptr,
                             zzz_Index         **symm_elements_ptr )
{
   zzz_Index         **stencil_shape = zzz_StructStencilShape(stencil);
   int                 stencil_size  = zzz_StructStencilSize(stencil); 

   zzz_StructStencil  *symm_stencil;
   zzz_Index         **symm_stencil_shape;
   int                 symm_stencil_size;
   int                *symm_elements;

   int                 no_symmetric_stencil_element;
   int                 i, j, d;

   int                 ierr = 0;

   /*------------------------------------------------------
    * Copy stencil elements into `symm_stencil_shape'
    *------------------------------------------------------*/

   symm_stencil_shape = zzz_CTAlloc(zzz_Index *, 2*stencil_size);
   for (i = 0; i < stencil_size; i++)
   {
      symm_stencil_shape[i] = zzz_NewIndex();
      zzz_CopyIndex(stencil_shape[i], symm_stencil_shape[i]);
   }

   /*------------------------------------------------------
    * Create symmetric stencil elements and `symm_elements'
    *------------------------------------------------------*/

   symm_elements = zzz_CTAlloc(int, 2*stencil_size);
   for (i = 0; i < 2*stencil_size; i++)
      symm_elements[i] = -1;

   symm_stencil_size = stencil_size;
   for (i = 0; i < stencil_size; i++)
   {
      if (symm_elements[i] < 0)
      {
         /* note: start at i to handle "center" element correctly */
         no_symmetric_stencil_element = 1;
         for (j = i; j < stencil_size; j++)
         {
            if ( (zzz_IndexX(symm_stencil_shape[j]) ==
                  -zzz_IndexX(symm_stencil_shape[i])  ) &&
                 (zzz_IndexY(symm_stencil_shape[j]) ==
                  -zzz_IndexY(symm_stencil_shape[i])  ) &&
                 (zzz_IndexZ(symm_stencil_shape[j]) ==
                  -zzz_IndexZ(symm_stencil_shape[i])  )   )
            {
               /* only "off-center" elements have symmetric entries */
               if (i != j)
                  symm_elements[j] = i;
               no_symmetric_stencil_element = 0;
            }
         }

         if (no_symmetric_stencil_element)
         {
            /* add symmetric stencil element to `symm_stencil' */
            symm_stencil_shape[symm_stencil_size] = zzz_NewIndex();
            for (d = 0; d < 3; d++)
            {
               zzz_IndexD(symm_stencil_shape[symm_stencil_size], d) =
                  -zzz_IndexD(symm_stencil_shape[i], d);
            }
	       
            symm_elements[symm_stencil_size] = i;
            symm_stencil_size++;
         }
      }
   }

   symm_stencil = zzz_NewStructStencil(zzz_StructStencilDim(stencil),
                                       symm_stencil_size, symm_stencil_shape);

   *symm_stencil_ptr  = symm_stencil;
   *symm_elements_ptr = symm_elements;

   return ierr;
}

