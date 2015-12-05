/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.7 $
 ***********************************************************************EHEADER*/




/*--------------------------------------------------------------------------
 * hypre_SStructOwnInfo data structure
 * This structure is for the coarsen fboxes that are on this processor,
 * and the cboxes of cgrid/(all coarsened fboxes) on this processor (i.e.,
 * the coarse boxes of the composite cgrid (no underlying) on this processor).
 *--------------------------------------------------------------------------*/
#ifndef hypre_OWNINFODATA_HEADER
#define hypre_OWNINFODATA_HEADER


typedef struct 
{
   HYPRE_Int             size;

   hypre_BoxArrayArray  *own_boxes;    /* size of fgrid */
   HYPRE_Int           **own_cboxnums; /* local cbox number- each fbox
                                          leads to an array of cboxes */

   hypre_BoxArrayArray  *own_composite_cboxes;  /* size of cgrid */
   HYPRE_Int             own_composite_size;
} hypre_SStructOwnInfoData;


/*--------------------------------------------------------------------------
 * Accessor macros: hypre_SStructOwnInfoData;
 *--------------------------------------------------------------------------*/

#define hypre_SStructOwnInfoDataSize(own_data)       ((own_data) -> size)
#define hypre_SStructOwnInfoDataOwnBoxes(own_data)   ((own_data) -> own_boxes)
#define hypre_SStructOwnInfoDataOwnBoxNums(own_data) \
((own_data) -> own_cboxnums)
#define hypre_SStructOwnInfoDataCompositeCBoxes(own_data) \
((own_data) -> own_composite_cboxes)
#define hypre_SStructOwnInfoDataCompositeSize(own_data) \
((own_data) -> own_composite_size)

#endif
