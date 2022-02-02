
#include <HYPRE_config.h>

#include "HYPRE_sstruct_ls.h"

#ifndef hypre_SSTRUCT_LS_HEADER
#define hypre_SSTRUCT_LS_HEADER

#include "_hypre_utilities.h"
#include "krylov.h"
#include "_hypre_struct_ls.h"
#include "_hypre_sstruct_mv.h"
#include "_hypre_parcsr_ls.h"
#include "multivector.h"

#ifdef __cplusplus
extern "C" {
#endif

/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

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
/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/*--------------------------------------------------------------------------
 * hypre_SStructRecvInfo data structure
 *--------------------------------------------------------------------------*/
#ifndef hypre_RECVINFODATA_HEADER
#define hypre_RECVINFODATA_HEADER


typedef struct
{
   HYPRE_Int             size;

   hypre_BoxArrayArray  *recv_boxes;
   HYPRE_Int           **recv_procs;

} hypre_SStructRecvInfoData;

#endif
/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/*--------------------------------------------------------------------------
 * hypre_SStructSendInfo data structure
 *--------------------------------------------------------------------------*/
#ifndef hypre_SENDINFODATA_HEADER
#define hypre_SENDINFODATA_HEADER


typedef struct
{
   HYPRE_Int             size;

   hypre_BoxArrayArray  *send_boxes;
   HYPRE_Int           **send_procs;
   HYPRE_Int           **send_remote_boxnums;

} hypre_SStructSendInfoData;

#endif
/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
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

/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Header info for the Maxwell solver
 *
 *****************************************************************************/

#ifndef hypre_MAXWELL_HEADER
#define hypre_MAXWELL_HEADER

/*--------------------------------------------------------------------------
 * hypre_MaxwellData:
 *--------------------------------------------------------------------------*/

typedef struct
{
   MPI_Comm                comm;

   HYPRE_Real              tol;
   HYPRE_Int               max_iter;
   HYPRE_Int               rel_change;
   HYPRE_Int               zero_guess;
   HYPRE_Int               ndim;

   HYPRE_Int               num_pre_relax;  /* number of pre relaxation sweeps */
   HYPRE_Int               num_post_relax; /* number of post relaxation sweeps */

   HYPRE_Int               constant_coef;

   hypre_Index            *rfactor;

   hypre_SStructGrid     **egrid_l;

   HYPRE_IJMatrix          Aen;
   hypre_ParCSRMatrix    **Aen_l;

   /* these will be extracted from the amg_data structure. Note that there is no grid
      underlying these matrices and vectors if they are generated by the amg_setup.
      So, will be stored as Parcsr_matrices and Par_vectors. */
   hypre_SStructMatrix    *Ann;
   hypre_SStructVector    *bn;
   hypre_SStructVector    *xn;

   void                   *amg_vdata;
   hypre_ParCSRMatrix    **Ann_l;
   hypre_SStructStencil  **Ann_stencils;
   hypre_ParCSRMatrix    **Pn_l;
   hypre_ParCSRMatrix    **RnT_l;
   hypre_ParVector       **bn_l;
   hypre_ParVector       **xn_l;
   hypre_ParVector       **resn_l;
   hypre_ParVector       **en_l;
   hypre_ParVector       **nVtemp_l;
   hypre_ParVector       **nVtemp2_l;
   HYPRE_Int             **nCF_marker_l;
   HYPRE_Real             *nrelax_weight;
   HYPRE_Real             *nomega;
   HYPRE_Int               nrelax_type;
   HYPRE_Int               node_numlevels;

   hypre_ParCSRMatrix     *Tgrad;
   hypre_ParCSRMatrix     *T_transpose;

   /* edge data structure. These will have grids. */
   HYPRE_Int               edge_maxlevels;
   HYPRE_Int               edge_numlevels;
   hypre_ParCSRMatrix    **Aee_l;
   hypre_ParVector       **be_l;
   hypre_ParVector       **xe_l;
   hypre_ParVector       **rese_l;
   hypre_ParVector       **ee_l;
   hypre_ParVector       **eVtemp_l;
   hypre_ParVector       **eVtemp2_l;
   HYPRE_Int             **eCF_marker_l;
   HYPRE_Real             *erelax_weight;
   HYPRE_Real             *eomega;
   HYPRE_Int               erelax_type;

   /* edge data structure. These will have no grid. */
   hypre_IJMatrix        **Pe_l;
   hypre_IJMatrix        **ReT_l;
   HYPRE_Int             **BdryRanks_l;
   HYPRE_Int              *BdryRanksCnts_l;

   /* edge-node data structure. These will have grids. */
   HYPRE_Int               en_numlevels;

   /* log info (always logged) */
   HYPRE_Int               num_iterations;
   HYPRE_Int               time_index;

   /* additional log info (logged when `logging' > 0) */
   HYPRE_Int               print_level;
   HYPRE_Int               logging;
   HYPRE_Real             *norms;
   HYPRE_Real             *rel_norms;

} hypre_MaxwellData;

#endif
/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

typedef struct
{
   HYPRE_BigInt row;

   HYPRE_Int ncols;
   HYPRE_BigInt      *cols;
   HYPRE_Real   *data;

} hypre_MaxwellOffProcRow;


/* HYPRE_sstruct_InterFAC.c */

/* HYPRE_sstruct_bicgstab.c */

/* HYPRE_sstruct_flexgmres.c */

/* HYPRE_sstruct_gmres.c */

/* HYPRE_sstruct_int.c */

/* HYPRE_sstruct_lgmres.c */

/* HYPRE_sstruct_maxwell.c */

/* HYPRE_sstruct_pcg.c */

/* HYPRE_sstruct_split.c */

/* HYPRE_sstruct_sys_pfmg.c */

/* bsearch.c */

/* eliminate_rowscols.c */

/* fac.c */

/* fac_CFInterfaceExtents.c */

/* fac_amr_fcoarsen.c */

/* fac_amr_rap.c */

/* fac_amr_zero_data.c */

/* fac_cf_coarsen.c */

/* fac_cfstencil_box.c */

/* fac_interp2.c */

/* fac_relax.c */

/* fac_restrict2.c */

/* fac_setup2.c */

/* fac_solve3.c */

/* fac_zero_cdata.c */

/* fac_zero_stencilcoef.c */

/* krylov.c */

/* krylov_sstruct.c */

/* maxwell_PNedelec.c */

/* maxwell_PNedelec_bdy.c */

/* maxwell_TV.c */

/* maxwell_TV_setup.c */

/* maxwell_grad.c */

/* maxwell_physbdy.c */

/* maxwell_semi_interp.c */

/* maxwell_solve.c */

/* maxwell_solve2.c */

/* maxwell_zeroBC.c */

/* nd1_amge_interpolation.c */

/* node_relax.c */

/* sstruct_amr_intercommunication.c */

/* sstruct_owninfo.c */

/* sstruct_recvinfo.c */

/* sstruct_sendinfo.c */

/* sstruct_sharedDOFComm.c */

/* sys_pfmg.c */

/* sys_pfmg_relax.c */

/* sys_pfmg_setup.c */

/* sys_pfmg_setup_interp.c */

/* sys_pfmg_setup_rap.c */

/* sys_pfmg_solve.c */

/* sys_semi_interp.c */

/* sys_semi_restrict.c */

#ifdef __cplusplus
}
#endif

#endif

