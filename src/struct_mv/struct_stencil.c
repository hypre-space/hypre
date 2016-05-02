/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision$
 ***********************************************************************EHEADER*/

/******************************************************************************
 *
 * Constructors and destructors for stencil structure.
 *
 *****************************************************************************/

#include "_hypre_struct_mv.h"

/*--------------------------------------------------------------------------
 * hypre_StructStencilCreate
 *--------------------------------------------------------------------------*/

hypre_StructStencil *
hypre_StructStencilCreate( HYPRE_Int     dim,
                           HYPRE_Int     size,
                           hypre_Index  *shape )
{
   hypre_StructStencil   *stencil;

   stencil = hypre_TAlloc(hypre_StructStencil, 1);

   hypre_StructStencilShape(stencil)    = shape;
   hypre_StructStencilSize(stencil)     = size;
   hypre_StructStencilNDim(stencil)      = dim;
   hypre_StructStencilRefCount(stencil) = 1;

   return stencil;
}

/*--------------------------------------------------------------------------
 * hypre_StructStencilRef
 *--------------------------------------------------------------------------*/

hypre_StructStencil *
hypre_StructStencilRef( hypre_StructStencil *stencil )
{
   hypre_StructStencilRefCount(stencil) ++;

   return stencil;
}

/*--------------------------------------------------------------------------
 * hypre_StructStencilDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructStencilDestroy( hypre_StructStencil *stencil )
{
   if (stencil)
   {
      hypre_StructStencilRefCount(stencil) --;
      if (hypre_StructStencilRefCount(stencil) == 0)
      {
         hypre_TFree(hypre_StructStencilShape(stencil));
         hypre_TFree(stencil);
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_StructStencilOffsetEntry
 *
 * Returns the entry number of the 'stencil_offset' in 'stencil'.  If the offset
 * is not found, a -1 is returned.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructStencilOffsetEntry( hypre_StructStencil *stencil,
                                hypre_Index          stencil_offset )
{
   hypre_Index  *stencil_shape;
   HYPRE_Int     entry;
   HYPRE_Int     i, ndim;

   entry = -1;
   ndim = hypre_StructStencilNDim(stencil);
   stencil_shape = hypre_StructStencilShape(stencil);
   for (i = 0; i < hypre_StructStencilSize(stencil); i++)
   {
      if (hypre_IndexesEqual(stencil_shape[i], stencil_offset, ndim))
      {
         entry = i;
         break;
      }
   }

   return entry;
}

/*--------------------------------------------------------------------------
 * hypre_StructStencilSymmetrize:
 *
 * Computes a new "symmetrized" stencil.  An integer array called 'symm_entries'
 * is also set up.  A non-negative value j = symm_entries[i] indicates that the
 * ith stencil entry is a "symmetric entry" of the jth stencil entry, that is,
 * stencil entry i is the transpose of stencil entry j (and symm_entries[j] is
 * assigned a negative value).
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructStencilSymmetrize( hypre_StructStencil  *stencil,
                               hypre_StructStencil **symm_stencil_ptr,
                               HYPRE_Int           **symm_entries_ptr )
{
   hypre_Index          *stencil_shape = hypre_StructStencilShape(stencil);
   HYPRE_Int             stencil_size  = hypre_StructStencilSize(stencil); 

   hypre_StructStencil  *symm_stencil;
   hypre_Index          *symm_stencil_shape;
   HYPRE_Int             symm_stencil_size;
   HYPRE_Int            *symm_entries;

   HYPRE_Int             no_symmetric_stencil_entry, symmetric;
   HYPRE_Int             i, j, d, ndim;
                       
   /*------------------------------------------------------
    * Copy stencil entrys into 'symm_stencil_shape'
    *------------------------------------------------------*/

   ndim = hypre_StructStencilNDim(stencil);
   symm_stencil_shape = hypre_CTAlloc(hypre_Index, 2*stencil_size);
   for (i = 0; i < stencil_size; i++)
   {
      hypre_CopyIndex(stencil_shape[i], symm_stencil_shape[i]);
   }

   /*------------------------------------------------------
    * Create symmetric stencil entries and 'symm_entries'
    *------------------------------------------------------*/

   symm_entries = hypre_CTAlloc(HYPRE_Int, 2*stencil_size);
   for (i = 0; i < 2*stencil_size; i++)
      symm_entries[i] = -1;

   symm_stencil_size = stencil_size;
   for (i = 0; i < stencil_size; i++)
   {
      if (symm_entries[i] < 0)
      {
         /* note: start at i to handle "center" entry correctly */
         no_symmetric_stencil_entry = 1;
         for (j = i; j < stencil_size; j++)
         {
            symmetric = 1;
            for (d = 0; d < ndim; d++)
            {
               if (hypre_IndexD(symm_stencil_shape[j], d) !=
                   -hypre_IndexD(symm_stencil_shape[i], d))
               {
                  symmetric = 0;
                  break;
               }
            }
            if (symmetric)
            {
               /* only "off-center" entries have symmetric entries */
               if (i != j)
               {
                  symm_entries[j] = i;
               }
               no_symmetric_stencil_entry = 0;
            }
         }

         if (no_symmetric_stencil_entry)
         {
            /* add symmetric stencil entry to 'symm_stencil' */
            for (d = 0; d < ndim; d++)
            {
               hypre_IndexD(symm_stencil_shape[symm_stencil_size], d) =
                  -hypre_IndexD(symm_stencil_shape[i], d);
            }
               
            symm_entries[symm_stencil_size] = i;
            symm_stencil_size++;
         }
      }
   }

   symm_stencil = hypre_StructStencilCreate(hypre_StructStencilNDim(stencil),
                                            symm_stencil_size,
                                            symm_stencil_shape);

   *symm_stencil_ptr = symm_stencil;
   *symm_entries_ptr = symm_entries;

   return hypre_error_flag;
}

