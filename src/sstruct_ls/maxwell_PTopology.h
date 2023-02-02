/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

typedef struct
{
   hypre_IJMatrix    *Face_iedge;
   hypre_IJMatrix    *Element_iedge;
   hypre_IJMatrix    *Edge_iedge;

   hypre_IJMatrix    *Element_Face;
   hypre_IJMatrix    *Element_Edge;

} hypre_PTopology;

