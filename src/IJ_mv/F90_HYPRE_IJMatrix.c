/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.14 $
 ***********************************************************************EHEADER*/

/******************************************************************************
 *
 * HYPRE_IJMatrix Fortran interface
 *
 *****************************************************************************/

#include "./_hypre_IJ_mv.h"
#include "fortran.h"

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ijmatrixcreate, HYPRE_IJMATRIXCREATE)
   ( hypre_F90_Comm *comm,
     hypre_F90_Int *ilower,
     hypre_F90_Int *iupper,
     hypre_F90_Int *jlower,
     hypre_F90_Int *jupper,
     hypre_F90_Obj *matrix,
     hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_IJMatrixCreate(
           hypre_F90_PassComm (comm),
           hypre_F90_PassInt (ilower),
           hypre_F90_PassInt (iupper),
           hypre_F90_PassInt (jlower),
           hypre_F90_PassInt (jupper),
           hypre_F90_PassObjRef (HYPRE_IJMatrix, matrix) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixDestroy
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_ijmatrixdestroy, HYPRE_IJMATRIXDESTROY)
   ( hypre_F90_Obj *matrix,
     hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_IJMatrixDestroy(
           hypre_F90_PassObj (HYPRE_IJMatrix, matrix) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixInitialize
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ijmatrixinitialize, HYPRE_IJMATRIXINITIALIZE)
   ( hypre_F90_Obj *matrix,
     hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_IJMatrixInitialize(
           hypre_F90_PassObj (HYPRE_IJMatrix, matrix) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixSetValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ijmatrixsetvalues, HYPRE_IJMATRIXSETVALUES)
   ( hypre_F90_Obj *matrix,
     hypre_F90_Int *nrows,
     hypre_F90_IntArray *ncols,
     hypre_F90_IntArray *rows,
     hypre_F90_IntArray *cols,
     hypre_F90_DblArray *values,
     hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_IJMatrixSetValues(
           hypre_F90_PassObj (HYPRE_IJMatrix, matrix),
           hypre_F90_PassInt (nrows),
           hypre_F90_PassIntArray (ncols),
           hypre_F90_PassIntArray (rows),
           hypre_F90_PassIntArray (cols),
           hypre_F90_PassDblArray (values)  ) );

}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixAddToValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ijmatrixaddtovalues, HYPRE_IJMATRIXADDTOVALUES)
   ( hypre_F90_Obj *matrix,
     hypre_F90_Int *nrows,
     hypre_F90_IntArray *ncols,
     hypre_F90_IntArray *rows,
     hypre_F90_IntArray *cols,
     hypre_F90_DblArray *values,
     hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_IJMatrixAddToValues(
           hypre_F90_PassObj (HYPRE_IJMatrix, matrix),
           hypre_F90_PassInt (nrows),
           hypre_F90_PassIntArray (ncols),
           hypre_F90_PassIntArray (rows),
           hypre_F90_PassIntArray (cols),
           hypre_F90_PassDblArray (values)  ) );

}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixAssemble
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_ijmatrixassemble, HYPRE_IJMATRIXASSEMBLE)
   ( hypre_F90_Obj *matrix,
     hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_IJMatrixAssemble(
           hypre_F90_PassObj (HYPRE_IJMatrix, matrix) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixGetRowCounts
 *--------------------------------------------------------------------------*/
 
void
hypre_F90_IFACE(hypre_ijmatrixgetrowcounts, HYPRE_IJMATRIXGETROWCOUNTS)
   ( hypre_F90_Obj *matrix,
     hypre_F90_Int *nrows,
     hypre_F90_IntArray *rows,
     hypre_F90_IntArray *ncols,
     hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_IJMatrixGetRowCounts(
           hypre_F90_PassObj (HYPRE_IJMatrix, matrix),
           hypre_F90_PassInt (nrows),
           hypre_F90_PassIntArray (rows),
           hypre_F90_PassIntArray (ncols) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixGetValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ijmatrixgetvalues, HYPRE_IJMATRIXGETVALUES)
   ( hypre_F90_Obj *matrix,
     hypre_F90_Int *nrows,
     hypre_F90_IntArray *ncols,
     hypre_F90_IntArray *rows,
     hypre_F90_IntArray *cols,
     hypre_F90_DblArray *values,
     hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_IJMatrixGetValues(
           hypre_F90_PassObj (HYPRE_IJMatrix, matrix),
           hypre_F90_PassInt (nrows),
           hypre_F90_PassIntArray (ncols),
           hypre_F90_PassIntArray (rows),
           hypre_F90_PassIntArray (cols),
           hypre_F90_PassDblArray (values)  ) );

}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixSetObjectType
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_ijmatrixsetobjecttype, HYPRE_IJMATRIXSETOBJECTTYPE)
   ( hypre_F90_Obj *matrix,
     hypre_F90_Int *type,
     hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_IJMatrixSetObjectType(
           hypre_F90_PassObj (HYPRE_IJMatrix, matrix),
           hypre_F90_PassInt (type)    ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixGetObjectType
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_ijmatrixgetobjecttype, HYPRE_IJMATRIXGETOBJECTTYPE)
   ( hypre_F90_Obj *matrix,
     hypre_F90_Int *type,
     hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_IJMatrixGetObjectType(
           hypre_F90_PassObj (HYPRE_IJMatrix, matrix),
           hypre_F90_PassIntRef (type)    ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixGetLocalRange
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_ijmatrixgetlocalrange, HYPRE_IJMATRIXGETLOCALRANGE)
   ( hypre_F90_Obj *matrix,
     hypre_F90_Int *ilower,
     hypre_F90_Int *iupper,
     hypre_F90_Int *jlower,
     hypre_F90_Int *jupper,
     hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_IJMatrixGetLocalRange(
           hypre_F90_PassObj (HYPRE_IJMatrix, matrix),
           hypre_F90_PassIntRef (ilower),
           hypre_F90_PassIntRef (iupper),
           hypre_F90_PassIntRef (jlower),
           hypre_F90_PassIntRef (jupper) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixGetObject
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_ijmatrixgetobject, HYPRE_IJMATRIXGETOBJECT)
   ( hypre_F90_Obj *matrix,
     hypre_F90_Obj *object,
     hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_IJMatrixGetObject(
           hypre_F90_PassObj (HYPRE_IJMatrix, matrix),
           (void **)         object  ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixSetRowSizes
 *--------------------------------------------------------------------------*/
 
void
hypre_F90_IFACE(hypre_ijmatrixsetrowsizes, HYPRE_IJMATRIXSETROWSIZES)
   ( hypre_F90_Obj *matrix,
     hypre_F90_IntArray *sizes,
     hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_IJMatrixSetRowSizes(
           hypre_F90_PassObj (HYPRE_IJMatrix, matrix),
           hypre_F90_PassIntArray (sizes)   ) );

}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixSetDiagOffdSizes
 *--------------------------------------------------------------------------*/
 
void
hypre_F90_IFACE(hypre_ijmatrixsetdiagoffdsizes, HYPRE_IJMATRIXSETDIAGOFFDSIZES)
   ( hypre_F90_Obj *matrix,
     hypre_F90_IntArray *diag_sizes,
     hypre_F90_IntArray *offd_sizes,
     hypre_F90_Int *ierr        )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_IJMatrixSetDiagOffdSizes(
           hypre_F90_PassObj (HYPRE_IJMatrix, matrix),
           hypre_F90_PassIntArray (diag_sizes),
           hypre_F90_PassIntArray (offd_sizes) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixSetMaxOffProcElmts
 *--------------------------------------------------------------------------*/
 
void
hypre_F90_IFACE(hypre_ijmatrixsetmaxoffprocelmt, HYPRE_IJMATRIXSETMAXOFFPROCELMT)
   ( hypre_F90_Obj *matrix,
     hypre_F90_Int *max_off_proc_elmts,
     hypre_F90_Int *ierr        )
{
   *ierr = (hypre_F90_Int) 
      ( HYPRE_IJMatrixSetMaxOffProcElmts(
           hypre_F90_PassObj (HYPRE_IJMatrix, matrix),
           hypre_F90_PassInt (max_off_proc_elmts) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixRead
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_ijmatrixread, HYPRE_IJMATRIXREAD)
   ( char     *filename,
     hypre_F90_Comm *comm,
     hypre_F90_Int *object_type,
     hypre_F90_Obj *matrix,
     hypre_F90_Int *ierr      )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_IJMatrixRead(
           (char *)            filename,
           hypre_F90_PassComm (comm),
           hypre_F90_PassInt (object_type),
           hypre_F90_PassObjRef (HYPRE_IJMatrix, matrix)    ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixPrint
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_ijmatrixprint, HYPRE_IJMATRIXPRINT)
   ( hypre_F90_Obj *matrix,
     char     *filename,
     hypre_F90_Int *ierr      )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_IJMatrixPrint(
           hypre_F90_PassObj (HYPRE_IJMatrix, matrix),
           (char *)          filename ) );
}
