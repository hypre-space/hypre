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
 * Header info for zzz_StructStencil data structures
 *
 *****************************************************************************/

#ifndef zzz_STRUCT_STENCIL_HEADER
#define zzz_STRUCT_STENCIL_HEADER


/*--------------------------------------------------------------------------
 * zzz_StructStencil
 *--------------------------------------------------------------------------*/

typedef struct
{
   zzz_Index   *shape;   /* Description of a stencil's shape */
   int          size;    /* Number of stencil coefficients */
                
   int          dim;     /* Number of dimensions */

} zzz_StructStencil;

/*--------------------------------------------------------------------------
 * Accessor functions for the zzz_StructStencil structure
 *--------------------------------------------------------------------------*/

#define zzz_StructStencilShape(stencil) ((stencil) -> shape)
#define zzz_StructStencilSize(stencil)  ((stencil) -> size)

#define zzz_StructStencilDim(stencil)   ((stencil) -> dim)

#define zzz_StructStencilElement(stencil, i) \
zzz_StructStencilShape(stencil)[i]

#endif
