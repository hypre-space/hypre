/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * HYPRE_IJVector interface
 *
 *****************************************************************************/

#include "./_hypre_IJ_mv.h"

#include "../HYPRE.h"

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorCreate
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_IJVectorCreate( MPI_Comm        comm,
                      HYPRE_BigInt    jlower,
                      HYPRE_BigInt    jupper,
                      HYPRE_IJVector *vector )
{
   hypre_IJVector *vec;
   HYPRE_Int       num_procs, my_id;
   HYPRE_BigInt    row0, rowN;

   vec = hypre_CTAlloc(hypre_IJVector,  1, HYPRE_MEMORY_HOST);

   if (!vec)
   {
      hypre_error(HYPRE_ERROR_MEMORY);
      return hypre_error_flag;
   }

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   if (jlower > jupper + 1 || jlower < 0)
   {
      hypre_error_in_arg(2);
      hypre_TFree(vec, HYPRE_MEMORY_HOST);
      return hypre_error_flag;
   }
   if (jupper < -1)
   {
      hypre_error_in_arg(3);
      return hypre_error_flag;
   }

   /* now we need the global number of rows as well
      as the global first row index */

   /* proc 0 has the first row  */
   if (my_id == 0)
   {
      row0 = jlower;
   }
   hypre_MPI_Bcast(&row0, 1, HYPRE_MPI_BIG_INT, 0, comm);
   /* proc (num_procs-1) has the last row  */
   if (my_id == (num_procs - 1))
   {
      rowN = jupper;
   }
   hypre_MPI_Bcast(&rowN, 1, HYPRE_MPI_BIG_INT, num_procs - 1, comm);

   hypre_IJVectorGlobalFirstRow(vec) = row0;
   hypre_IJVectorGlobalNumRows(vec) = rowN - row0 + 1;

   hypre_IJVectorComm(vec)            = comm;
   hypre_IJVectorNumComponents(vec)   = 1;
   hypre_IJVectorObjectType(vec)      = HYPRE_UNITIALIZED;
   hypre_IJVectorObject(vec)          = NULL;
   hypre_IJVectorTranslator(vec)      = NULL;
   hypre_IJVectorAssumedPart(vec)     = NULL;
   hypre_IJVectorPrintLevel(vec)      = 0;
   hypre_IJVectorPartitioning(vec)[0] = jlower;
   hypre_IJVectorPartitioning(vec)[1] = jupper + 1;

   *vector = (HYPRE_IJVector) vec;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorSetNumComponents
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_IJVectorSetNumComponents( HYPRE_IJVector vector,
                                HYPRE_Int      num_components )
{
   hypre_IJVector *vec = (hypre_IJVector *) vector;

   if (!vec)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (num_components < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   hypre_IJVectorNumComponents(vector) = num_components;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorSetComponent
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_IJVectorSetComponent( HYPRE_IJVector vector,
                            HYPRE_Int      component )
{
   hypre_IJVector *vec = (hypre_IJVector *) vector;

   if (!vec)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (hypre_IJVectorObjectType(vec) == HYPRE_PARCSR)
   {
      hypre_IJVectorSetComponentPar(vector, component);
   }
   else
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_IJVectorDestroy( HYPRE_IJVector vector )
{
   hypre_IJVector *vec = (hypre_IJVector *) vector;

   if (!vec)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (hypre_IJVectorAssumedPart(vec))
   {
      hypre_AssumedPartitionDestroy((hypre_IJAssumedPart*)hypre_IJVectorAssumedPart(vec));
   }

   if ( hypre_IJVectorObjectType(vec) == HYPRE_PARCSR )
   {
      hypre_IJVectorDestroyPar(vec);
      if (hypre_IJVectorTranslator(vec))
      {
         hypre_AuxParVectorDestroy((hypre_AuxParVector *)
                                   (hypre_IJVectorTranslator(vec)));
      }
   }
   else if ( hypre_IJVectorObjectType(vec) != -1 )
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   hypre_TFree(vec, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorInitialize
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_IJVectorInitialize( HYPRE_IJVector vector )
{
   hypre_IJVector *vec = (hypre_IJVector *) vector;

   if (!vec)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if ( hypre_IJVectorObjectType(vec) == HYPRE_PARCSR )
   {
      if (!hypre_IJVectorObject(vec))
      {
         hypre_IJVectorCreatePar(vec, hypre_IJVectorPartitioning(vec));
      }

      hypre_IJVectorInitializePar(vec);
   }
   else
   {
      hypre_error_in_arg(1);
   }

   return hypre_error_flag;
}

HYPRE_Int
HYPRE_IJVectorInitialize_v2( HYPRE_IJVector vector, HYPRE_MemoryLocation memory_location )
{
   hypre_IJVector *vec = (hypre_IJVector *) vector;

   if (!vec)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if ( hypre_IJVectorObjectType(vec) == HYPRE_PARCSR )
   {
      if (!hypre_IJVectorObject(vec))
      {
         hypre_IJVectorCreatePar(vec, hypre_IJVectorPartitioning(vec));
      }

      hypre_IJVectorInitializePar_v2(vec, memory_location);
   }
   else
   {
      hypre_error_in_arg(1);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorSetPrintLevel
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_IJVectorSetPrintLevel( HYPRE_IJVector vector,
                             HYPRE_Int print_level )
{
   hypre_IJVector *ijvector = (hypre_IJVector *) vector;

   if (!ijvector)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   hypre_IJVectorPrintLevel(ijvector) = (print_level > 0) ? print_level : 0;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorSetValues
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_IJVectorSetValues( HYPRE_IJVector        vector,
                         HYPRE_Int             nvalues,
                         const HYPRE_BigInt   *indices,
                         const HYPRE_Complex  *values   )
{
   hypre_IJVector *vec = (hypre_IJVector *) vector;

   if (nvalues == 0) { return hypre_error_flag; }

   if (!vec)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (nvalues < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   if (!values)
   {
      hypre_error_in_arg(4);
      return hypre_error_flag;
   }

   if ( hypre_IJVectorObjectType(vec) == HYPRE_PARCSR )
   {
#if defined(HYPRE_USING_GPU)
      HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1( hypre_IJVectorMemoryLocation(vector) );

      if (exec == HYPRE_EXEC_DEVICE)
      {
         return ( hypre_IJVectorSetAddValuesParDevice(vec, nvalues, indices, values, "set") );
      }
      else
#endif
      {
         return ( hypre_IJVectorSetValuesPar(vec, nvalues, indices, values) );
      }
   }
   else
   {
      hypre_error_in_arg(1);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorAddToValues
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_IJVectorAddToValues( HYPRE_IJVector        vector,
                           HYPRE_Int             nvalues,
                           const HYPRE_BigInt   *indices,
                           const HYPRE_Complex  *values )
{
   hypre_IJVector *vec = (hypre_IJVector *) vector;

   if (nvalues == 0) { return hypre_error_flag; }

   if (!vec)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (nvalues < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   if (!values)
   {
      hypre_error_in_arg(4);
      return hypre_error_flag;
   }

   if ( hypre_IJVectorObjectType(vec) == HYPRE_PARCSR )
   {
#if defined(HYPRE_USING_GPU)
      HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1( hypre_IJVectorMemoryLocation(vector) );

      if (exec == HYPRE_EXEC_DEVICE)
      {
         return ( hypre_IJVectorSetAddValuesParDevice(vec, nvalues, indices, values, "add") );
      }
      else
#endif
      {
         return ( hypre_IJVectorAddToValuesPar(vec, nvalues, indices, values) );
      }
   }
   else
   {
      hypre_error_in_arg(1);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorAssemble
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_IJVectorAssemble( HYPRE_IJVector vector )
{
   hypre_IJVector *vec = (hypre_IJVector *) vector;

   if (!vec)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if ( hypre_IJVectorObjectType(vec) == HYPRE_PARCSR )
   {
#if defined(HYPRE_USING_GPU)
      HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1( hypre_IJVectorMemoryLocation(vector) );

      if (exec == HYPRE_EXEC_DEVICE)
      {
         return ( hypre_IJVectorAssembleParDevice(vec) );
      }
      else
#endif
      {
         return ( hypre_IJVectorAssemblePar(vec) );
      }
   }
   else
   {
      hypre_error_in_arg(1);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorUpdateValues
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_IJVectorUpdateValues( HYPRE_IJVector        vector,
                            HYPRE_Int             nvalues,
                            const HYPRE_BigInt   *indices,
                            const HYPRE_Complex  *values,
                            HYPRE_Int             action )
{
   hypre_IJVector *vec = (hypre_IJVector *) vector;

   if (nvalues == 0) { return hypre_error_flag; }

   if (!vec)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (nvalues < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   if (!values)
   {
      hypre_error_in_arg(4);
      return hypre_error_flag;
   }

   if ( hypre_IJVectorObjectType(vec) == HYPRE_PARCSR )
   {
#if defined(HYPRE_USING_GPU)
      HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1( hypre_IJVectorMemoryLocation(vector) );

      if (exec == HYPRE_EXEC_DEVICE)
      {
         return ( hypre_IJVectorUpdateValuesDevice(vec, nvalues, indices, values, action) );
      }
      else
#endif
      {
         if (action == 1)
         {
            return ( hypre_IJVectorSetValuesPar(vec, nvalues, indices, values) );
         }
         else
         {
            return ( hypre_IJVectorAddToValuesPar(vec, nvalues, indices, values) );
         }
      }
   }
   else
   {
      hypre_error_in_arg(1);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorGetValues
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_IJVectorGetValues( HYPRE_IJVector      vector,
                         HYPRE_Int           nvalues,
                         const HYPRE_BigInt *indices,
                         HYPRE_Complex      *values )
{
   hypre_IJVector *vec = (hypre_IJVector *) vector;

   if (nvalues == 0) { return hypre_error_flag; }

   if (!vec)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (nvalues < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   if (!values)
   {
      hypre_error_in_arg(4);
      return hypre_error_flag;
   }

   if ( hypre_IJVectorObjectType(vec) == HYPRE_PARCSR )
   {
      return ( hypre_IJVectorGetValuesPar(vec, nvalues, indices, values) );
   }
   else
   {
      hypre_error_in_arg(1);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorSetMaxOffProcElmts
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_IJVectorSetMaxOffProcElmts( HYPRE_IJVector vector,
                                  HYPRE_Int      max_off_proc_elmts )
{
   hypre_IJVector *vec = (hypre_IJVector *) vector;

   if (!vec)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if ( hypre_IJVectorObjectType(vec) == HYPRE_PARCSR )
   {
      return ( hypre_IJVectorSetMaxOffProcElmtsPar(vec, max_off_proc_elmts));
   }
   else
   {
      hypre_error_in_arg(1);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorSetObjectType
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_IJVectorSetObjectType( HYPRE_IJVector vector,
                             HYPRE_Int      type )
{
   hypre_IJVector *vec = (hypre_IJVector *) vector;

   if (!vec)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   hypre_IJVectorObjectType(vec) = type;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorGetObjectType
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_IJVectorGetObjectType( HYPRE_IJVector  vector,
                             HYPRE_Int      *type )
{
   hypre_IJVector *vec = (hypre_IJVector *) vector;

   if (!vec)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *type = hypre_IJVectorObjectType(vec);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorGetLocalRange
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_IJVectorGetLocalRange( HYPRE_IJVector  vector,
                             HYPRE_BigInt   *jlower,
                             HYPRE_BigInt   *jupper )
{
   hypre_IJVector *vec = (hypre_IJVector *) vector;

   if (!vec)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *jlower = hypre_IJVectorPartitioning(vec)[0];
   *jupper = hypre_IJVectorPartitioning(vec)[1] - 1;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorGetObject
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_IJVectorGetObject( HYPRE_IJVector   vector,
                         void           **object )
{
   hypre_IJVector *vec = (hypre_IJVector *) vector;

   if (!vec)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *object = hypre_IJVectorObject(vec);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorRead
 * create IJVector on host memory
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_IJVectorRead( const char     *filename,
                    MPI_Comm        comm,
                    HYPRE_Int       type,
                    HYPRE_IJVector *vector_ptr )
{
   HYPRE_IJVector  vector;
   HYPRE_BigInt    jlower, jupper, j;
   HYPRE_Complex   value;
   HYPRE_Int       myid, ret;
   char            new_filename[255];
   FILE           *file;

   hypre_MPI_Comm_rank(comm, &myid);

   hypre_sprintf(new_filename, "%s.%05d", filename, myid);

   if ((file = fopen(new_filename, "r")) == NULL)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   hypre_fscanf(file, "%b %b", &jlower, &jupper);
   HYPRE_IJVectorCreate(comm, jlower, jupper, &vector);

   HYPRE_IJVectorSetObjectType(vector, type);

   HYPRE_IJVectorInitialize_v2(vector, HYPRE_MEMORY_HOST);

   /* It is important to ensure that whitespace follows the index value to help
    * catch mistakes in the input file.  This is done with %*[ \t].  Using a
    * space here causes an input line with a single decimal value on it to be
    * read as if it were an integer followed by a decimal value. */
   while ( (ret = hypre_fscanf(file, "%b%*[ \t]%le", &j, &value)) != EOF )
   {
      if (ret != 2)
      {
         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Error in IJ vector input file.");
         return hypre_error_flag;
      }
      if (j < jlower || j > jupper)
      {
         HYPRE_IJVectorAddToValues(vector, 1, &j, &value);
      }
      else
      {
         HYPRE_IJVectorSetValues(vector, 1, &j, &value);
      }
   }

   HYPRE_IJVectorAssemble(vector);

   fclose(file);

   *vector_ptr = vector;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorReadBinary
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_IJVectorReadBinary( const char     *filename,
                          MPI_Comm        comm,
                          HYPRE_Int       type,
                          HYPRE_IJVector *vector_ptr )
{
   return hypre_IJVectorReadBinary(comm, filename, type, vector_ptr);
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorPrint
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_IJVectorPrint( HYPRE_IJVector  vector,
                     const char     *filename )
{
   MPI_Comm        comm;
   HYPRE_BigInt   *partitioning;
   HYPRE_BigInt    jlower, jupper, j;
   HYPRE_Complex  *h_values = NULL, *d_values = NULL, *values = NULL;
   HYPRE_Int       myid, n_local;
   char            new_filename[255];
   FILE           *file;

   if (!vector)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   comm = hypre_IJVectorComm(vector);
   hypre_MPI_Comm_rank(comm, &myid);

   hypre_sprintf(new_filename, "%s.%05d", filename, myid);

   if ((file = fopen(new_filename, "w")) == NULL)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   partitioning = hypre_IJVectorPartitioning(vector);
   jlower = partitioning[0];
   jupper = partitioning[1] - 1;
   n_local = jupper - jlower + 1;

   hypre_fprintf(file, "%b %b\n", jlower, jupper);

   HYPRE_MemoryLocation memory_location = hypre_IJVectorMemoryLocation(vector);

   d_values = hypre_TAlloc(HYPRE_Complex, n_local, memory_location);

   HYPRE_IJVectorGetValues(vector, n_local, NULL, d_values);

   if ( hypre_GetActualMemLocation(memory_location) == hypre_MEMORY_HOST )
   {
      values = d_values;
   }
   else
   {
      h_values = hypre_TAlloc(HYPRE_Complex, n_local, HYPRE_MEMORY_HOST);
      hypre_TMemcpy(h_values, d_values, HYPRE_Complex, n_local, HYPRE_MEMORY_HOST, memory_location);
      values = h_values;
   }

   for (j = jlower; j <= jupper; j++)
   {
      hypre_fprintf(file, "%b %.14e\n", j, values[j - jlower]);
   }

   hypre_TFree(d_values, memory_location);
   hypre_TFree(h_values, HYPRE_MEMORY_HOST);

   fclose(file);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorPrintBinary
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_IJVectorPrintBinary( HYPRE_IJVector  vector,
                           const char     *filename )
{
   if (!vector)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (hypre_IJVectorObjectType(vector) == HYPRE_PARCSR)
   {
      hypre_ParVectorPrintBinaryIJ((hypre_ParVector*) hypre_IJVectorObject(vector),
                                   filename);
   }
   else
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorInnerProd
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_IJVectorInnerProd( HYPRE_IJVector  x,
                         HYPRE_IJVector  y,
                         HYPRE_Real     *prod )
{
   hypre_IJVector *xvec = (hypre_IJVector *) x;
   hypre_IJVector *yvec = (hypre_IJVector *) y;

   if (!xvec)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (!yvec)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   if (hypre_IJVectorObjectType(xvec) != hypre_IJVectorObjectType(yvec))
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Input vectors don't have the same object type!");
      return hypre_error_flag;
   }

   if (hypre_IJVectorObjectType(xvec) == HYPRE_PARCSR)
   {
      hypre_ParVector *par_x = (hypre_ParVector*) hypre_IJVectorObject(xvec);
      hypre_ParVector *par_y = (hypre_ParVector*) hypre_IJVectorObject(yvec);

      HYPRE_ParVectorInnerProd(par_x, par_y, prod);
   }
   else
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   return hypre_error_flag;
}
