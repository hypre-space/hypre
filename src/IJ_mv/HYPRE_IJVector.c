/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
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
   HYPRE_Int num_procs, my_id;
   HYPRE_BigInt *partitioning;

#ifdef HYPRE_NO_GLOBAL_PARTITION
   HYPRE_BigInt  row0, rowN;
#else
   HYPRE_BigInt *recv_buf;
   HYPRE_BigInt *info;
   HYPRE_Int i, i2;
#endif

   vec = hypre_CTAlloc(hypre_IJVector,  1, HYPRE_MEMORY_HOST);

   if (!vec)
   {
      hypre_error(HYPRE_ERROR_MEMORY);
      return hypre_error_flag;
   }

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   if (jlower > jupper+1 || jlower < 0)
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


#ifdef HYPRE_NO_GLOBAL_PARTITION

   partitioning = hypre_CTAlloc(HYPRE_BigInt,  2, HYPRE_MEMORY_HOST);

   partitioning[0] = jlower;
   partitioning[1] = jupper+1;


   /* now we need the global number of rows as well
      as the global first row index */

   /* proc 0 has the first row  */
   if (my_id==0)
   {
      row0 = jlower;
   }
   hypre_MPI_Bcast(&row0, 1, HYPRE_MPI_BIG_INT, 0, comm);
   /* proc (num_procs-1) has the last row  */
   if (my_id == (num_procs-1))
   {
      rowN = jupper;
   }
   hypre_MPI_Bcast(&rowN, 1, HYPRE_MPI_BIG_INT, num_procs-1, comm);

   hypre_IJVectorGlobalFirstRow(vec) = row0;
   hypre_IJVectorGlobalNumRows(vec) = rowN - row0 + 1;

#else

   info = hypre_CTAlloc(HYPRE_BigInt, 2, HYPRE_MEMORY_HOST);
   recv_buf = hypre_CTAlloc(HYPRE_BigInt,  2*num_procs, HYPRE_MEMORY_HOST);
   partitioning = hypre_CTAlloc(HYPRE_BigInt,  num_procs+1, HYPRE_MEMORY_HOST);

   info[0] = jlower;
   info[1] = jupper;

   hypre_MPI_Allgather(info, 2, HYPRE_MPI_BIG_INT, recv_buf, 2, HYPRE_MPI_BIG_INT, comm);

   partitioning[0] = recv_buf[0];
   for (i=0; i < num_procs-1; i++)
   {
      i2 = i+i;
      if (recv_buf[i2+1] != (recv_buf[i2+2]-1))
      {
         /*hypre_printf("Inconsistent partitioning -- HYPRE_IJVectorCreate\n");  */
         hypre_error(HYPRE_ERROR_GENERIC);
         hypre_TFree(info, HYPRE_MEMORY_HOST);
         hypre_TFree(recv_buf, HYPRE_MEMORY_HOST);
         hypre_TFree(partitioning, HYPRE_MEMORY_HOST);
         hypre_TFree(vec, HYPRE_MEMORY_HOST);
         return hypre_error_flag;
      }
      else
      {
         partitioning[i+1] = recv_buf[i2+2];
      }
   }
   i2 = (num_procs-1)*2;
   partitioning[num_procs] = recv_buf[i2+1]+1;

   hypre_TFree(info, HYPRE_MEMORY_HOST);
   hypre_TFree(recv_buf, HYPRE_MEMORY_HOST);


   hypre_IJVectorGlobalFirstRow(vec) = partitioning[0];
   hypre_IJVectorGlobalNumRows(vec)= partitioning[num_procs]-partitioning[0];

#endif

   hypre_IJVectorComm(vec)         = comm;
   hypre_IJVectorPartitioning(vec) = partitioning;
   hypre_IJVectorObjectType(vec)   = HYPRE_UNITIALIZED;
   hypre_IJVectorObject(vec)       = NULL;
   hypre_IJVectorTranslator(vec)   = NULL;
   hypre_IJVectorAssumedPart(vec)   = NULL;
   hypre_IJVectorPrintLevel(vec)   = 0;

   *vector = (HYPRE_IJVector) vec;

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

   if (hypre_IJVectorPartitioning(vec))
   {
      hypre_TFree(hypre_IJVectorPartitioning(vec), HYPRE_MEMORY_HOST);
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

   hypre_IJVectorPrintLevel(ijvector) = 1;
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

   if (nvalues == 0) return hypre_error_flag;

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
#if defined(HYPRE_USING_CUDA)
      HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1( hypre_IJVectorMemoryLocation(vector) );

      if (exec == HYPRE_EXEC_DEVICE)
      {
         return ( hypre_IJVectorSetAddValuesParDevice(vec, nvalues, indices, values, "set") );
      }
      else
#endif
      {
         return( hypre_IJVectorSetValuesPar(vec, nvalues, indices, values) );
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

   if (nvalues == 0) return hypre_error_flag;

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
#if defined(HYPRE_USING_CUDA)
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
#if defined(HYPRE_USING_CUDA)
      HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1( hypre_IJVectorMemoryLocation(vector) );

      if (exec == HYPRE_EXEC_DEVICE)
      {
         return( hypre_IJVectorAssembleParDevice(vec) );
      }
      else
#endif
      {
         return( hypre_IJVectorAssemblePar(vec) );
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

   if (nvalues == 0) return hypre_error_flag;

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
      return( hypre_IJVectorGetValuesPar(vec, nvalues, indices, values) );
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
      return( hypre_IJVectorSetMaxOffProcElmtsPar(vec, max_off_proc_elmts));
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
   MPI_Comm comm;
   HYPRE_BigInt *partitioning;
   HYPRE_Int my_id;

   if (!vec)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   comm = hypre_IJVectorComm(vec);
   partitioning = hypre_IJVectorPartitioning(vec);
   hypre_MPI_Comm_rank(comm, &my_id);

#ifdef HYPRE_NO_GLOBAL_PARTITION
   *jlower = partitioning[0];
   *jupper = partitioning[1]-1;
#else
   *jlower = partitioning[my_id];
   *jupper = partitioning[my_id+1]-1;
#endif
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

   hypre_sprintf(new_filename,"%s.%05d", filename, myid);

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
 * HYPRE_IJVectorPrint
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_IJVectorPrint( HYPRE_IJVector  vector,
                     const char     *filename )
{
   MPI_Comm        comm;
   HYPRE_BigInt   *partitioning;
   HYPRE_BigInt    jlower, jupper, j;
   HYPRE_Complex   value;
   HYPRE_Int       myid;
   char            new_filename[255];
   FILE           *file;

   if (!vector)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   comm = hypre_IJVectorComm(vector);
   hypre_MPI_Comm_rank(comm, &myid);

   hypre_sprintf(new_filename,"%s.%05d", filename, myid);

   if ((file = fopen(new_filename, "w")) == NULL)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   partitioning = hypre_IJVectorPartitioning(vector);
#ifdef HYPRE_NO_GLOBAL_PARTITION
   jlower = partitioning[0];
   jupper = partitioning[1] - 1;
#else
   jlower = partitioning[myid];
   jupper = partitioning[myid+1] - 1;
#endif
   hypre_fprintf(file, "%b %b\n", jlower, jupper);

   for (j = jlower; j <= jupper; j++)
   {
      HYPRE_IJVectorGetValues(vector, 1, &j, &value);

      hypre_fprintf(file, "%b %.14e\n", j, value);
   }

   fclose(file);

   return hypre_error_flag;
}
