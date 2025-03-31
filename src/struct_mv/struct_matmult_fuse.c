
/*** This file should be included in struct_matmult.c and not compiled separately ***/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------
 * Macros used in the kernel loops below
 *--------------------------------------------------------------------------*/

//#include "struct_matmult_fuse.h"

#define HYPRE_FUSE_MAXDEPTH 15

#define HYPRE_SMMFUSE_1FFF(k) \
   mptrs[k][Mi] += cprod[k] * tptrs[k][0][fi] * tptrs[k][1][fi]* tptrs[k][2][fi]

#define HYPRE_SMMFUSE_1FFC(k) \
   mptrs[k][Mi] += cprod[k] * tptrs[k][0][fi] * tptrs[k][1][fi]* tptrs[k][2][ci]
//   locmp[k][Mi] += loccp[k] * loctp[k][0][fi] * loctp[k][1][fi]* loctp[k][2][ci]

#define HYPRE_SMMFUSE_1FCC(k) \
   mptrs[k][Mi] += cprod[k] * tptrs[k][0][fi] * tptrs[k][1][ci]* tptrs[k][2][ci]

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatmultCompute_fuse_1fff( hypre_StructMatmultDataMH *a,
                                      HYPRE_Int                  nprod,
                                      HYPRE_Complex             *cprod,
                                      HYPRE_Complex           ***tptrs,
                                      HYPRE_Complex            **mptrs,
                                      HYPRE_Int                  ndim,
                                      hypre_Index                loop_size,
                                      hypre_Box                 *fdbox,
                                      hypre_Index                fdstart,
                                      hypre_Index                fdstride,
                                      hypre_Box                 *Mdbox,
                                      hypre_Index                Mdstart,
                                      hypre_Index                Mdstride )
{
   HYPRE_Int     k;
   HYPRE_Int     depth;

   if (nprod < 1)
   {
      return hypre_error_flag;
   }

   HYPRE_ANNOTATE_FUNC_BEGIN;

   for (k = 0; k < nprod; k += HYPRE_FUSE_MAXDEPTH)
   {
      depth = hypre_min(HYPRE_FUSE_MAXDEPTH, (nprod - k));

      switch (depth)
      {
         case 15:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_SMMFUSE_1FFF(k + 0);
               HYPRE_SMMFUSE_1FFF(k + 0);
               HYPRE_SMMFUSE_1FFF(k + 1);
               HYPRE_SMMFUSE_1FFF(k + 2);
               HYPRE_SMMFUSE_1FFF(k + 3);
               HYPRE_SMMFUSE_1FFF(k + 4);
               HYPRE_SMMFUSE_1FFF(k + 5);
               HYPRE_SMMFUSE_1FFF(k + 6);
               HYPRE_SMMFUSE_1FFF(k + 7);
               HYPRE_SMMFUSE_1FFF(k + 8);
               HYPRE_SMMFUSE_1FFF(k + 9);
               HYPRE_SMMFUSE_1FFF(k + 10);
               HYPRE_SMMFUSE_1FFF(k + 11);
               HYPRE_SMMFUSE_1FFF(k + 12);
               HYPRE_SMMFUSE_1FFF(k + 13);
               HYPRE_SMMFUSE_1FFF(k + 14);
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 14:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_SMMFUSE_1FFF(k + 0);
               HYPRE_SMMFUSE_1FFF(k + 1);
               HYPRE_SMMFUSE_1FFF(k + 2);
               HYPRE_SMMFUSE_1FFF(k + 3);
               HYPRE_SMMFUSE_1FFF(k + 4);
               HYPRE_SMMFUSE_1FFF(k + 5);
               HYPRE_SMMFUSE_1FFF(k + 6);
               HYPRE_SMMFUSE_1FFF(k + 7);
               HYPRE_SMMFUSE_1FFF(k + 8);
               HYPRE_SMMFUSE_1FFF(k + 9);
               HYPRE_SMMFUSE_1FFF(k + 10);
               HYPRE_SMMFUSE_1FFF(k + 11);
               HYPRE_SMMFUSE_1FFF(k + 12);
               HYPRE_SMMFUSE_1FFF(k + 13);
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 13:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_SMMFUSE_1FFF(k + 0);
               HYPRE_SMMFUSE_1FFF(k + 1);
               HYPRE_SMMFUSE_1FFF(k + 2);
               HYPRE_SMMFUSE_1FFF(k + 3);
               HYPRE_SMMFUSE_1FFF(k + 4);
               HYPRE_SMMFUSE_1FFF(k + 5);
               HYPRE_SMMFUSE_1FFF(k + 6);
               HYPRE_SMMFUSE_1FFF(k + 7);
               HYPRE_SMMFUSE_1FFF(k + 8);
               HYPRE_SMMFUSE_1FFF(k + 9);
               HYPRE_SMMFUSE_1FFF(k + 10);
               HYPRE_SMMFUSE_1FFF(k + 11);
               HYPRE_SMMFUSE_1FFF(k + 12);
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 12:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_SMMFUSE_1FFF(k + 0);
               HYPRE_SMMFUSE_1FFF(k + 1);
               HYPRE_SMMFUSE_1FFF(k + 2);
               HYPRE_SMMFUSE_1FFF(k + 3);
               HYPRE_SMMFUSE_1FFF(k + 4);
               HYPRE_SMMFUSE_1FFF(k + 5);
               HYPRE_SMMFUSE_1FFF(k + 6);
               HYPRE_SMMFUSE_1FFF(k + 7);
               HYPRE_SMMFUSE_1FFF(k + 8);
               HYPRE_SMMFUSE_1FFF(k + 9);
               HYPRE_SMMFUSE_1FFF(k + 10);
               HYPRE_SMMFUSE_1FFF(k + 11);
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 11:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_SMMFUSE_1FFF(k + 0);
               HYPRE_SMMFUSE_1FFF(k + 1);
               HYPRE_SMMFUSE_1FFF(k + 2);
               HYPRE_SMMFUSE_1FFF(k + 3);
               HYPRE_SMMFUSE_1FFF(k + 4);
               HYPRE_SMMFUSE_1FFF(k + 5);
               HYPRE_SMMFUSE_1FFF(k + 6);
               HYPRE_SMMFUSE_1FFF(k + 7);
               HYPRE_SMMFUSE_1FFF(k + 8);
               HYPRE_SMMFUSE_1FFF(k + 9);
               HYPRE_SMMFUSE_1FFF(k + 10);
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 10:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_SMMFUSE_1FFF(k + 0);
               HYPRE_SMMFUSE_1FFF(k + 1);
               HYPRE_SMMFUSE_1FFF(k + 2);
               HYPRE_SMMFUSE_1FFF(k + 3);
               HYPRE_SMMFUSE_1FFF(k + 4);
               HYPRE_SMMFUSE_1FFF(k + 5);
               HYPRE_SMMFUSE_1FFF(k + 6);
               HYPRE_SMMFUSE_1FFF(k + 7);
               HYPRE_SMMFUSE_1FFF(k + 8);
               HYPRE_SMMFUSE_1FFF(k + 9);
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 9:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_SMMFUSE_1FFF(k + 0);
               HYPRE_SMMFUSE_1FFF(k + 1);
               HYPRE_SMMFUSE_1FFF(k + 2);
               HYPRE_SMMFUSE_1FFF(k + 3);
               HYPRE_SMMFUSE_1FFF(k + 4);
               HYPRE_SMMFUSE_1FFF(k + 5);
               HYPRE_SMMFUSE_1FFF(k + 6);
               HYPRE_SMMFUSE_1FFF(k + 7);
               HYPRE_SMMFUSE_1FFF(k + 8);
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 8:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_SMMFUSE_1FFF(k + 0);
               HYPRE_SMMFUSE_1FFF(k + 1);
               HYPRE_SMMFUSE_1FFF(k + 2);
               HYPRE_SMMFUSE_1FFF(k + 3);
               HYPRE_SMMFUSE_1FFF(k + 4);
               HYPRE_SMMFUSE_1FFF(k + 5);
               HYPRE_SMMFUSE_1FFF(k + 6);
               HYPRE_SMMFUSE_1FFF(k + 7);
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 7:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_SMMFUSE_1FFF(k + 0);
               HYPRE_SMMFUSE_1FFF(k + 1);
               HYPRE_SMMFUSE_1FFF(k + 2);
               HYPRE_SMMFUSE_1FFF(k + 3);
               HYPRE_SMMFUSE_1FFF(k + 4);
               HYPRE_SMMFUSE_1FFF(k + 5);
               HYPRE_SMMFUSE_1FFF(k + 6);
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 6:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_SMMFUSE_1FFF(k + 0);
               HYPRE_SMMFUSE_1FFF(k + 1);
               HYPRE_SMMFUSE_1FFF(k + 2);
               HYPRE_SMMFUSE_1FFF(k + 3);
               HYPRE_SMMFUSE_1FFF(k + 4);
               HYPRE_SMMFUSE_1FFF(k + 5);
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 5:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_SMMFUSE_1FFF(k + 0);
               HYPRE_SMMFUSE_1FFF(k + 1);
               HYPRE_SMMFUSE_1FFF(k + 2);
               HYPRE_SMMFUSE_1FFF(k + 3);
               HYPRE_SMMFUSE_1FFF(k + 4);
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 4:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_SMMFUSE_1FFF(k + 0);
               HYPRE_SMMFUSE_1FFF(k + 1);
               HYPRE_SMMFUSE_1FFF(k + 2);
               HYPRE_SMMFUSE_1FFF(k + 3);
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 3:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_SMMFUSE_1FFF(k + 0);
               HYPRE_SMMFUSE_1FFF(k + 1);
               HYPRE_SMMFUSE_1FFF(k + 2);
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 2:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_SMMFUSE_1FFF(k + 0);
               HYPRE_SMMFUSE_1FFF(k + 1);
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 1:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_SMMFUSE_1FFF(k + 0);
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         default:
            hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Unsupported loop fusion depth!");

            HYPRE_ANNOTATE_FUNC_END;
            return hypre_error_flag;
      }
   }

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatmultCompute_fuse_1ffc( hypre_StructMatmultDataMH *a,
                                      HYPRE_Int                  nprod,
                                      HYPRE_Complex             *cprod,
                                      HYPRE_Complex           ***tptrs,
                                      HYPRE_Complex            **mptrs,
                                      HYPRE_Int                  ndim,
                                      hypre_Index                loop_size,
                                      hypre_Box                 *fdbox,
                                      hypre_Index                fdstart,
                                      hypre_Index                fdstride,
                                      hypre_Box                 *cdbox,
                                      hypre_Index                cdstart,
                                      hypre_Index                cdstride,
                                      hypre_Box                 *Mdbox,
                                      hypre_Index                Mdstart,
                                      hypre_Index                Mdstride )
{
   HYPRE_Int     k;
   HYPRE_Int     depth;
   //HYPRE_Int       d;
   //HYPRE_Complex   loccp[38];
   //HYPRE_Complex  *loctp[38][3];
   //HYPRE_Complex  *locmp[38];
   ////HYPRE_Complex   loccp[HYPRE_FUSE_MAXDEPTH];
   ////HYPRE_Complex  *loctp[HYPRE_FUSE_MAXDEPTH][3];
   ////HYPRE_Complex  *locmp[HYPRE_FUSE_MAXDEPTH];

   if (nprod < 1)
   {
      return hypre_error_flag;
   }

   HYPRE_ANNOTATE_FUNC_BEGIN;

   //for (k = 0; k < nprod; k += HYPRE_FUSE_MAXDEPTH)
   for (k = 0; k < nprod; /* increment k at the end of the body below */ )
   {
      depth = hypre_min(HYPRE_FUSE_MAXDEPTH, (nprod - k));
      //depth = hypre_max( depth, hypre_min(38, (nprod - k)) );

      //// Try this later if k confuses the compiler
      //for (d = 0; d < depth; d++)
      //{
      //   loccp[d]  = cprod[k + d];
      //   loctp[d][0] = tptrs[k + d][0];
      //   loctp[d][1] = tptrs[k + d][1];
      //   loctp[d][2] = tptrs[k + d][2];
      //   locmp[d]  = mptrs[k + d];
      //}

      //hypre_ParPrintf(MPI_COMM_WORLD, "depth = %d\n", depth);

      switch (depth)
      {
         case 15:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_1FFC(k + 0);
               HYPRE_SMMFUSE_1FFC(k + 1);
               HYPRE_SMMFUSE_1FFC(k + 2);
               HYPRE_SMMFUSE_1FFC(k + 3);
               HYPRE_SMMFUSE_1FFC(k + 4);
               HYPRE_SMMFUSE_1FFC(k + 5);
               HYPRE_SMMFUSE_1FFC(k + 6);
               HYPRE_SMMFUSE_1FFC(k + 7);
               HYPRE_SMMFUSE_1FFC(k + 8);
               HYPRE_SMMFUSE_1FFC(k + 9);
               HYPRE_SMMFUSE_1FFC(k + 10);
               HYPRE_SMMFUSE_1FFC(k + 11);
               HYPRE_SMMFUSE_1FFC(k + 12);
               HYPRE_SMMFUSE_1FFC(k + 13);
               HYPRE_SMMFUSE_1FFC(k + 14);
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 14:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_1FFC(k + 0);
               HYPRE_SMMFUSE_1FFC(k + 1);
               HYPRE_SMMFUSE_1FFC(k + 2);
               HYPRE_SMMFUSE_1FFC(k + 3);
               HYPRE_SMMFUSE_1FFC(k + 4);
               HYPRE_SMMFUSE_1FFC(k + 5);
               HYPRE_SMMFUSE_1FFC(k + 6);
               HYPRE_SMMFUSE_1FFC(k + 7);
               HYPRE_SMMFUSE_1FFC(k + 8);
               HYPRE_SMMFUSE_1FFC(k + 9);
               HYPRE_SMMFUSE_1FFC(k + 10);
               HYPRE_SMMFUSE_1FFC(k + 11);
               HYPRE_SMMFUSE_1FFC(k + 12);
               HYPRE_SMMFUSE_1FFC(k + 13);
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 13:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_1FFC(k + 0);
               HYPRE_SMMFUSE_1FFC(k + 1);
               HYPRE_SMMFUSE_1FFC(k + 2);
               HYPRE_SMMFUSE_1FFC(k + 3);
               HYPRE_SMMFUSE_1FFC(k + 4);
               HYPRE_SMMFUSE_1FFC(k + 5);
               HYPRE_SMMFUSE_1FFC(k + 6);
               HYPRE_SMMFUSE_1FFC(k + 7);
               HYPRE_SMMFUSE_1FFC(k + 8);
               HYPRE_SMMFUSE_1FFC(k + 9);
               HYPRE_SMMFUSE_1FFC(k + 10);
               HYPRE_SMMFUSE_1FFC(k + 11);
               HYPRE_SMMFUSE_1FFC(k + 12);
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 12:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_1FFC(k + 0);
               HYPRE_SMMFUSE_1FFC(k + 1);
               HYPRE_SMMFUSE_1FFC(k + 2);
               HYPRE_SMMFUSE_1FFC(k + 3);
               HYPRE_SMMFUSE_1FFC(k + 4);
               HYPRE_SMMFUSE_1FFC(k + 5);
               HYPRE_SMMFUSE_1FFC(k + 6);
               HYPRE_SMMFUSE_1FFC(k + 7);
               HYPRE_SMMFUSE_1FFC(k + 8);
               HYPRE_SMMFUSE_1FFC(k + 9);
               HYPRE_SMMFUSE_1FFC(k + 10);
               HYPRE_SMMFUSE_1FFC(k + 11);
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 11:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_1FFC(k + 0);
               HYPRE_SMMFUSE_1FFC(k + 1);
               HYPRE_SMMFUSE_1FFC(k + 2);
               HYPRE_SMMFUSE_1FFC(k + 3);
               HYPRE_SMMFUSE_1FFC(k + 4);
               HYPRE_SMMFUSE_1FFC(k + 5);
               HYPRE_SMMFUSE_1FFC(k + 6);
               HYPRE_SMMFUSE_1FFC(k + 7);
               HYPRE_SMMFUSE_1FFC(k + 8);
               HYPRE_SMMFUSE_1FFC(k + 9);
               HYPRE_SMMFUSE_1FFC(k + 10);
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 10:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_1FFC(k + 0);
               HYPRE_SMMFUSE_1FFC(k + 1);
               HYPRE_SMMFUSE_1FFC(k + 2);
               HYPRE_SMMFUSE_1FFC(k + 3);
               HYPRE_SMMFUSE_1FFC(k + 4);
               HYPRE_SMMFUSE_1FFC(k + 5);
               HYPRE_SMMFUSE_1FFC(k + 6);
               HYPRE_SMMFUSE_1FFC(k + 7);
               HYPRE_SMMFUSE_1FFC(k + 8);
               HYPRE_SMMFUSE_1FFC(k + 9);
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 9:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_1FFC(k + 0);
               HYPRE_SMMFUSE_1FFC(k + 1);
               HYPRE_SMMFUSE_1FFC(k + 2);
               HYPRE_SMMFUSE_1FFC(k + 3);
               HYPRE_SMMFUSE_1FFC(k + 4);
               HYPRE_SMMFUSE_1FFC(k + 5);
               HYPRE_SMMFUSE_1FFC(k + 6);
               HYPRE_SMMFUSE_1FFC(k + 7);
               HYPRE_SMMFUSE_1FFC(k + 8);
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 8:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_1FFC(k + 0);
               HYPRE_SMMFUSE_1FFC(k + 1);
               HYPRE_SMMFUSE_1FFC(k + 2);
               HYPRE_SMMFUSE_1FFC(k + 3);
               HYPRE_SMMFUSE_1FFC(k + 4);
               HYPRE_SMMFUSE_1FFC(k + 5);
               HYPRE_SMMFUSE_1FFC(k + 6);
               HYPRE_SMMFUSE_1FFC(k + 7);
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 7:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_1FFC(k + 0);
               HYPRE_SMMFUSE_1FFC(k + 1);
               HYPRE_SMMFUSE_1FFC(k + 2);
               HYPRE_SMMFUSE_1FFC(k + 3);
               HYPRE_SMMFUSE_1FFC(k + 4);
               HYPRE_SMMFUSE_1FFC(k + 5);
               HYPRE_SMMFUSE_1FFC(k + 6);
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 6:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_1FFC(k + 0);
               HYPRE_SMMFUSE_1FFC(k + 1);
               HYPRE_SMMFUSE_1FFC(k + 2);
               HYPRE_SMMFUSE_1FFC(k + 3);
               HYPRE_SMMFUSE_1FFC(k + 4);
               HYPRE_SMMFUSE_1FFC(k + 5);
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 5:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_1FFC(k + 0);
               HYPRE_SMMFUSE_1FFC(k + 1);
               HYPRE_SMMFUSE_1FFC(k + 2);
               HYPRE_SMMFUSE_1FFC(k + 3);
               HYPRE_SMMFUSE_1FFC(k + 4);
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 4:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_1FFC(k + 0);
               HYPRE_SMMFUSE_1FFC(k + 1);
               HYPRE_SMMFUSE_1FFC(k + 2);
               HYPRE_SMMFUSE_1FFC(k + 3);
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 3:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_1FFC(k + 0);
               HYPRE_SMMFUSE_1FFC(k + 1);
               HYPRE_SMMFUSE_1FFC(k + 2);
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 2:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_1FFC(k + 0);
               HYPRE_SMMFUSE_1FFC(k + 1);
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 1:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_1FFC(k + 0);
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         default:
            hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Unsupported loop fusion depth!");

            HYPRE_ANNOTATE_FUNC_END;
            return hypre_error_flag;
      }

      k += depth;
   }

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatmultCompute_fuse_1fcc( hypre_StructMatmultDataMH *a,
                                      HYPRE_Int                  nprod,
                                      HYPRE_Complex             *cprod,
                                      HYPRE_Complex           ***tptrs,
                                      HYPRE_Complex            **mptrs,
                                      HYPRE_Int                  ndim,
                                      hypre_Index                loop_size,
                                      hypre_Box                 *fdbox,
                                      hypre_Index                fdstart,
                                      hypre_Index                fdstride,
                                      hypre_Box                 *cdbox,
                                      hypre_Index                cdstart,
                                      hypre_Index                cdstride,
                                      hypre_Box                 *Mdbox,
                                      hypre_Index                Mdstart,
                                      hypre_Index                Mdstride )
{
   HYPRE_Int     k;
   HYPRE_Int     depth;

   if (nprod < 1)
   {
      return hypre_error_flag;
   }

   HYPRE_ANNOTATE_FUNC_BEGIN;

   for (k = 0; k < nprod; k += HYPRE_FUSE_MAXDEPTH)
   {
      depth = hypre_min(HYPRE_FUSE_MAXDEPTH, (nprod - k));

      switch (depth)
      {
         case 15:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_1FCC(k + 0);
               HYPRE_SMMFUSE_1FCC(k + 1);
               HYPRE_SMMFUSE_1FCC(k + 2);
               HYPRE_SMMFUSE_1FCC(k + 3);
               HYPRE_SMMFUSE_1FCC(k + 4);
               HYPRE_SMMFUSE_1FCC(k + 5);
               HYPRE_SMMFUSE_1FCC(k + 6);
               HYPRE_SMMFUSE_1FCC(k + 7);
               HYPRE_SMMFUSE_1FCC(k + 8);
               HYPRE_SMMFUSE_1FCC(k + 9);
               HYPRE_SMMFUSE_1FCC(k + 10);
               HYPRE_SMMFUSE_1FCC(k + 11);
               HYPRE_SMMFUSE_1FCC(k + 12);
               HYPRE_SMMFUSE_1FCC(k + 13);
               HYPRE_SMMFUSE_1FCC(k + 14);
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 14:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_1FCC(k + 0);
               HYPRE_SMMFUSE_1FCC(k + 1);
               HYPRE_SMMFUSE_1FCC(k + 2);
               HYPRE_SMMFUSE_1FCC(k + 3);
               HYPRE_SMMFUSE_1FCC(k + 4);
               HYPRE_SMMFUSE_1FCC(k + 5);
               HYPRE_SMMFUSE_1FCC(k + 6);
               HYPRE_SMMFUSE_1FCC(k + 7);
               HYPRE_SMMFUSE_1FCC(k + 8);
               HYPRE_SMMFUSE_1FCC(k + 9);
               HYPRE_SMMFUSE_1FCC(k + 10);
               HYPRE_SMMFUSE_1FCC(k + 11);
               HYPRE_SMMFUSE_1FCC(k + 12);
               HYPRE_SMMFUSE_1FCC(k + 13);
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 13:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_1FCC(k + 0);
               HYPRE_SMMFUSE_1FCC(k + 1);
               HYPRE_SMMFUSE_1FCC(k + 2);
               HYPRE_SMMFUSE_1FCC(k + 3);
               HYPRE_SMMFUSE_1FCC(k + 4);
               HYPRE_SMMFUSE_1FCC(k + 5);
               HYPRE_SMMFUSE_1FCC(k + 6);
               HYPRE_SMMFUSE_1FCC(k + 7);
               HYPRE_SMMFUSE_1FCC(k + 8);
               HYPRE_SMMFUSE_1FCC(k + 9);
               HYPRE_SMMFUSE_1FCC(k + 10);
               HYPRE_SMMFUSE_1FCC(k + 11);
               HYPRE_SMMFUSE_1FCC(k + 12);
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 12:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_1FCC(k + 0);
               HYPRE_SMMFUSE_1FCC(k + 1);
               HYPRE_SMMFUSE_1FCC(k + 2);
               HYPRE_SMMFUSE_1FCC(k + 3);
               HYPRE_SMMFUSE_1FCC(k + 4);
               HYPRE_SMMFUSE_1FCC(k + 5);
               HYPRE_SMMFUSE_1FCC(k + 6);
               HYPRE_SMMFUSE_1FCC(k + 7);
               HYPRE_SMMFUSE_1FCC(k + 8);
               HYPRE_SMMFUSE_1FCC(k + 9);
               HYPRE_SMMFUSE_1FCC(k + 10);
               HYPRE_SMMFUSE_1FCC(k + 11);
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 11:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_1FCC(k + 0);
               HYPRE_SMMFUSE_1FCC(k + 1);
               HYPRE_SMMFUSE_1FCC(k + 2);
               HYPRE_SMMFUSE_1FCC(k + 3);
               HYPRE_SMMFUSE_1FCC(k + 4);
               HYPRE_SMMFUSE_1FCC(k + 5);
               HYPRE_SMMFUSE_1FCC(k + 6);
               HYPRE_SMMFUSE_1FCC(k + 7);
               HYPRE_SMMFUSE_1FCC(k + 8);
               HYPRE_SMMFUSE_1FCC(k + 9);
               HYPRE_SMMFUSE_1FCC(k + 10);
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 10:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_1FCC(k + 0);
               HYPRE_SMMFUSE_1FCC(k + 1);
               HYPRE_SMMFUSE_1FCC(k + 2);
               HYPRE_SMMFUSE_1FCC(k + 3);
               HYPRE_SMMFUSE_1FCC(k + 4);
               HYPRE_SMMFUSE_1FCC(k + 5);
               HYPRE_SMMFUSE_1FCC(k + 6);
               HYPRE_SMMFUSE_1FCC(k + 7);
               HYPRE_SMMFUSE_1FCC(k + 8);
               HYPRE_SMMFUSE_1FCC(k + 9);
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 9:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_1FCC(k + 0);
               HYPRE_SMMFUSE_1FCC(k + 1);
               HYPRE_SMMFUSE_1FCC(k + 2);
               HYPRE_SMMFUSE_1FCC(k + 3);
               HYPRE_SMMFUSE_1FCC(k + 4);
               HYPRE_SMMFUSE_1FCC(k + 5);
               HYPRE_SMMFUSE_1FCC(k + 6);
               HYPRE_SMMFUSE_1FCC(k + 7);
               HYPRE_SMMFUSE_1FCC(k + 8);
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 8:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_1FCC(k + 0);
               HYPRE_SMMFUSE_1FCC(k + 1);
               HYPRE_SMMFUSE_1FCC(k + 2);
               HYPRE_SMMFUSE_1FCC(k + 3);
               HYPRE_SMMFUSE_1FCC(k + 4);
               HYPRE_SMMFUSE_1FCC(k + 5);
               HYPRE_SMMFUSE_1FCC(k + 6);
               HYPRE_SMMFUSE_1FCC(k + 7);
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 7:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_1FCC(k + 0);
               HYPRE_SMMFUSE_1FCC(k + 1);
               HYPRE_SMMFUSE_1FCC(k + 2);
               HYPRE_SMMFUSE_1FCC(k + 3);
               HYPRE_SMMFUSE_1FCC(k + 4);
               HYPRE_SMMFUSE_1FCC(k + 5);
               HYPRE_SMMFUSE_1FCC(k + 6);
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 6:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_1FCC(k + 0);
               HYPRE_SMMFUSE_1FCC(k + 1);
               HYPRE_SMMFUSE_1FCC(k + 2);
               HYPRE_SMMFUSE_1FCC(k + 3);
               HYPRE_SMMFUSE_1FCC(k + 4);
               HYPRE_SMMFUSE_1FCC(k + 5);
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 5:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_1FCC(k + 0);
               HYPRE_SMMFUSE_1FCC(k + 1);
               HYPRE_SMMFUSE_1FCC(k + 2);
               HYPRE_SMMFUSE_1FCC(k + 3);
               HYPRE_SMMFUSE_1FCC(k + 4);
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 4:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_1FCC(k + 0);
               HYPRE_SMMFUSE_1FCC(k + 1);
               HYPRE_SMMFUSE_1FCC(k + 2);
               HYPRE_SMMFUSE_1FCC(k + 3);
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 3:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_1FCC(k + 0);
               HYPRE_SMMFUSE_1FCC(k + 1);
               HYPRE_SMMFUSE_1FCC(k + 2);
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 2:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_1FCC(k + 0);
               HYPRE_SMMFUSE_1FCC(k + 1);
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 1:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_1FCC(k + 0);
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         default:
            hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Unsupported loop fusion depth!");

            HYPRE_ANNOTATE_FUNC_END;
            return hypre_error_flag;
      }
   }

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatmultCompute_fuse_triple( hypre_StructMatmultDataMH *a,
                                        HYPRE_Int    na,
                                        HYPRE_Int    ndim,
                                        hypre_Index  loop_size,
                                        HYPRE_Int    stencil_size,
                                        hypre_Box   *fdbox,
                                        hypre_Index  fdstart,
                                        hypre_Index  fdstride,
                                        hypre_Box   *cdbox,
                                        hypre_Index  cdstart,
                                        hypre_Index  cdstride,
                                        hypre_Box   *Mdbox,
                                        hypre_Index  Mdstart,
                                        hypre_Index  Mdstride )
{
   HYPRE_Int         *nprod;
   HYPRE_Complex    **cprod;
   HYPRE_Complex  ****tptrs;
   HYPRE_Complex   ***mptrs;

   HYPRE_Int         ptype, nf, nc;
   HYPRE_Int         p, i, k, t;

   /* Allocate memory */
   nprod = hypre_CTAlloc(HYPRE_Int, 3, HYPRE_MEMORY_HOST);
   cprod = hypre_TAlloc(HYPRE_Complex *, 3, HYPRE_MEMORY_HOST);
   tptrs = hypre_TAlloc(HYPRE_Complex ***, 3, HYPRE_MEMORY_HOST);
   mptrs = hypre_TAlloc(HYPRE_Complex **, 3, HYPRE_MEMORY_HOST);
   for (ptype = 0; ptype < 3; ptype++)
   {
      cprod[ptype] = hypre_CTAlloc(HYPRE_Complex, na, HYPRE_MEMORY_HOST);
      tptrs[ptype] = hypre_TAlloc(HYPRE_Complex **, na, HYPRE_MEMORY_HOST);
      mptrs[ptype] = hypre_TAlloc(HYPRE_Complex *, na, HYPRE_MEMORY_HOST);
      for (p = 0; p < na; p++)
      {
         tptrs[ptype][p] = hypre_TAlloc(HYPRE_Complex *, 3, HYPRE_MEMORY_HOST);
      }
   }

   /* Initialize product counters */
   for (p = 0; p < 3; p++)
   {
      nprod[p] = 0;
   }

   /* Build product arrays */
   for (i = 0; i < na; i++)
   {
      /* Determine number of fine and coarse terms */
      nf = nc = 0;
      for (t = 0; t < 3; t++)
      {
         if (a[i].types[t] == 1)
         {
            /* Type 1 -> coarse data space */
            nc++;
         }
         else
         {
            /* Type 0 or 2 -> fine data space */
            nf++;
         }
      }

      /* Determine product type */
      switch (nc)
      {
         case 0: /* fff term (call core_1t) */
            ptype = 0;
            break;
         case 1: /* ffc term (call core_2tbb) */
            ptype = 1;
            break;
         case 2: /* fcc term (call core_2etb) */
            ptype = 2;
            break;
      }

      /* Set array values for product k of product type ptype */
      k = nprod[ptype];
      cprod[ptype][k] = a[i].cprod;
      nf = nc = 0;
      for (t = 0; t < 3; t++)
      {
         if (a[i].types[t] == 1)
         {
            /* Type 1 -> coarse data space */
            tptrs[ptype][k][2 - nc] = a[i].tptrs[t];  /* put last */
            nc++;
         }
         else
         {
            /* Type 0 or 2 -> fine data space */
            tptrs[ptype][k][nf] = a[i].tptrs[t];  /* put first */
            nf++;
         }
      }
      mptrs[ptype][k] = a[i].mptr;
      nprod[ptype]++;

   } /* loop i < na*/

   //hypre_ParPrintf(MPI_COMM_WORLD, "nprod = %d, %d, %d\n", nprod[0], nprod[1], nprod[2]);

   /* Call core functions */
   hypre_StructMatmultCompute_fuse_1fff(a, nprod[0], cprod[0], tptrs[0], mptrs[0],
                                        ndim, loop_size,
                                        fdbox, fdstart, fdstride,
                                        Mdbox, Mdstart, Mdstride);

   hypre_StructMatmultCompute_fuse_1ffc(a, nprod[1], cprod[1], tptrs[1], mptrs[1],
                                        ndim, loop_size,
                                        fdbox, fdstart, fdstride,
                                        cdbox, cdstart, cdstride,
                                        Mdbox, Mdstart, Mdstride);
   
   hypre_StructMatmultCompute_fuse_1fcc(a, nprod[2], cprod[2], tptrs[2], mptrs[2],
                                        ndim, loop_size,
                                        fdbox, fdstart, fdstride,
                                        cdbox, cdstart, cdstride,
                                        Mdbox, Mdstart, Mdstride);

   /* Free memory */
   for (ptype = 0; ptype < 3; ptype++)
   {
      for (p = 0; p < na; p++)
      {
         hypre_TFree(tptrs[ptype][p], HYPRE_MEMORY_HOST);
      }
      hypre_TFree(cprod[ptype], HYPRE_MEMORY_HOST);
      hypre_TFree(tptrs[ptype], HYPRE_MEMORY_HOST);
      hypre_TFree(mptrs[ptype], HYPRE_MEMORY_HOST);
   }
   hypre_TFree(nprod, HYPRE_MEMORY_HOST);
   hypre_TFree(cprod, HYPRE_MEMORY_HOST);
   hypre_TFree(tptrs, HYPRE_MEMORY_HOST);
   hypre_TFree(mptrs, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}

