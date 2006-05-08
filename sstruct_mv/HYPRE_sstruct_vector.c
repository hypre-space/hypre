/*BHEADER**********************************************************************
 * (c) 2000   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * HYPRE_SStructVector interface
 *
 *****************************************************************************/

#include "headers.h"
#include "sstruct_mv.h"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructVectorCreate( MPI_Comm              comm,
                           HYPRE_SStructGrid     grid,
                           HYPRE_SStructVector  *vector_ptr )
{
   int ierr = 0;

   hypre_SStructVector   *vector;
   int                    nparts;
   hypre_SStructPVector **pvectors;
   MPI_Comm               pcomm;
   hypre_SStructPGrid    *pgrid;
   int                    part;

   vector = hypre_TAlloc(hypre_SStructVector, 1);

   hypre_SStructVectorComm(vector) = comm;
   hypre_SStructVectorNDim(vector) = hypre_SStructGridNDim(grid);
   hypre_SStructGridRef(grid, &hypre_SStructVectorGrid(vector));
   hypre_SStructVectorObjectType(vector) = HYPRE_SSTRUCT;
   nparts = hypre_SStructGridNParts(grid);
   hypre_SStructVectorNParts(vector) = nparts;
   pvectors = hypre_TAlloc(hypre_SStructPVector *, nparts);
   for (part = 0; part < nparts; part++)
   {
      pgrid = hypre_SStructGridPGrid(grid, part);
      pcomm = hypre_SStructPGridComm(pgrid);
      ierr = hypre_SStructPVectorCreate(pcomm, pgrid, &pvectors[part]);
   }
   hypre_SStructVectorPVectors(vector)   = pvectors;
   hypre_SStructVectorIJVector(vector)   = NULL;

   /* GEC1002 initializing to NULL */

    hypre_SStructVectorDataIndices(vector) = NULL;
    hypre_SStructVectorData(vector)        = NULL;

   /* GEC1002 moving the creation of the ijvector the the initialize part  
    *   ilower = hypre_SStructGridStartRank(grid); 
    *   iupper = ilower + hypre_SStructGridLocalSize(grid) - 1;
    *  HYPRE_IJVectorCreate(comm, ilowergh, iuppergh,
    *                  &hypre_SStructVectorIJVector(vector)); */

   hypre_SStructVectorIJVector(vector)   = NULL;
   hypre_SStructVectorParVector(vector)  = NULL;
   hypre_SStructVectorGlobalSize(vector) = 0;
   hypre_SStructVectorRefCount(vector)   = 1;
   hypre_SStructVectorDataSize(vector)   = 0;
   hypre_SStructVectorObjectType(vector) = HYPRE_SSTRUCT;
 
   *vector_ptr = vector;

   return ierr;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

int 
HYPRE_SStructVectorDestroy( HYPRE_SStructVector vector )
{
   int ierr = 0;

   int                    nparts;
   hypre_SStructPVector **pvectors;
   int                    part;
   int                   vector_type = hypre_SStructVectorObjectType(vector);

   /* GEC1002 destroying dataindices and data in vector  */

   if (vector)
   {
      hypre_SStructVectorRefCount(vector) --;
      if (hypre_SStructVectorRefCount(vector) == 0)
      {
         HYPRE_SStructGridDestroy(hypre_SStructVectorGrid(vector));
         nparts   = hypre_SStructVectorNParts(vector);
         pvectors = hypre_SStructVectorPVectors(vector);
         for (part = 0; part < nparts; part++)
         {
            hypre_SStructPVectorDestroy(pvectors[part]);
         }
         hypre_TFree(pvectors);
         HYPRE_IJVectorDestroy(hypre_SStructVectorIJVector(vector));
         /* GEC1002 the ijdestroy takes care of the data when the
          *  vector is type HYPRE_SSTRUCT. This is a result that the
          * ijvector does not use the owndata flag in the data structure
          * unlike the structvector                               */                      

	 /* GEC if data has been allocated then free the pointer */
          hypre_TFree(hypre_SStructVectorDataIndices(vector));
         
         if (hypre_SStructVectorData(vector) && (vector_type == HYPRE_PARCSR))
	 {
             hypre_TFree(hypre_SStructVectorData(vector));
         }

         hypre_TFree(vector);
      }
   }

   return ierr;
}

/*---------------------------------------------------------
 * GEC1002 changes to initialize the vector with a data chunk
 * that includes all the part,var pieces instead of just svector-var
 * pieces. In case of pure unstruct-variables (ucvar), which are at the
 * end of each part, we might need to modify initialize shell vector
 * ----------------------------------------------------------*/

int 
HYPRE_SStructVectorInitialize( HYPRE_SStructVector vector )
{
  int                      ierr = 0;
  int                      datasize;
  int                      nvars ;
  int                      nparts = hypre_SStructVectorNParts(vector) ;
  int                      var,part  ;
  double                  *data ;
  double                  *pdata ;
  double                  *sdata  ;
  hypre_SStructPVector    *pvector;
  hypre_StructVector      *svector;
  int                     *dataindices;
  int                     *pdataindices;
  int                     vector_type = hypre_SStructVectorObjectType(vector);
  hypre_SStructGrid      *grid =  hypre_SStructVectorGrid(vector);
  MPI_Comm                comm = hypre_SStructVectorComm(vector);
  HYPRE_IJVector          ijvector;

 /* GEC0902 addition of variables for ilower and iupper   */
   int                     ilower, iupper;
   hypre_ParVector          *par_vector;
   hypre_Vector             *parlocal_vector;
 

   /* GEC0902 getting the datasizes and indices we need  */

   hypre_SStructVectorInitializeShell(vector);

   datasize = hypre_SStructVectorDataSize(vector);

   data = hypre_CTAlloc(double, datasize);

   dataindices = hypre_SStructVectorDataIndices(vector);

   hypre_SStructVectorData(vector)  = data;
  
  for (part = 0; part < nparts; part++)
  {
     pvector = hypre_SStructVectorPVector(vector,part);
     pdataindices = hypre_SStructPVectorDataIndices(pvector);
     /* shift-num   = dataindices[part]; */
     pdata = data + dataindices[part];
     nvars = hypre_SStructPVectorNVars(pvector);

     for (var = 0; var < nvars; var++)
     {     
       svector = hypre_SStructPVectorSVector(pvector, var);
       /*  shift-pnum    = pdataindices[var]; */ 
       sdata   = pdata + pdataindices[var];

       /* GEC1002 initialization of inside data pointer of a svector
        * because no data is alloced, we make sure the flag is zero. This
        * affects the destroy */
       hypre_StructVectorInitializeData(svector, sdata);
       hypre_StructVectorDataAlloced(svector) = 0;
     }
 
  }
   
  /* GEC1002 this is now the creation of the ijmatrix and the initialization  
   * by checking the type of the vector */

   if(vector_type == HYPRE_PARCSR )
   {
     ilower = hypre_SStructGridStartRank(grid);
     iupper = ilower + hypre_SStructGridLocalSize(grid) - 1;
   }
   
   if(vector_type == HYPRE_SSTRUCT || vector_type == HYPRE_STRUCT)
   {
     ilower = hypre_SStructGridGhstartRank(grid);
     iupper = ilower + hypre_SStructGridGhlocalSize(grid) - 1;
   }    
 
   HYPRE_IJVectorCreate(comm, ilower, iupper,
                      &hypre_SStructVectorIJVector(vector)); 

   /* GEC1002, once the partitioning is done, it is time for the actual
    * initialization                                                 */


   /* u-vector: the type is for the parvector inside the ijvector */

   ijvector = hypre_SStructVectorIJVector(vector);
 
   ierr = HYPRE_IJVectorSetObjectType(ijvector, HYPRE_PARCSR);

   ierr += HYPRE_IJVectorInitialize(ijvector);

   
   /* GEC1002  for HYPRE_SSTRUCT type of vector, we do not need data allocated inside
    * the parvector piece of the structure. We make that pointer within the localvector
    * to point to the outside "data". Before redirecting the 
    * local pointer to point to the true data chunk for HYPRE_SSTRUCT: we destroy and assign.  
    * We now have two entries of the data structure  pointing to the same chunk
    * if we have a HYPRE_SSTRUCT vector  
    * We do not need the IJVectorInitializePar, we have to undoit for the SStruct case
    * in a sense it is a desinitializepar */

   if (vector_type == HYPRE_SSTRUCT || vector_type == HYPRE_STRUCT)
   {
     par_vector = hypre_IJVectorObject(ijvector);
     parlocal_vector = hypre_ParVectorLocalVector(par_vector);
     hypre_TFree(hypre_VectorData(parlocal_vector));
     hypre_VectorData(parlocal_vector) = data ;
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructVectorSetValues( HYPRE_SStructVector  vector,
                              int                  part,
                              int                 *index,
                              int                  var,
                              double              *value )
{
   int ierr = 0;
   int                   ndim    = hypre_SStructVectorNDim(vector);
   hypre_SStructPVector *pvector = hypre_SStructVectorPVector(vector, part);
   hypre_Index           cindex;

   hypre_CopyToCleanIndex(index, ndim, cindex);

   if (var < hypre_SStructPVectorNVars(pvector))
   {
      ierr = hypre_SStructPVectorSetValues(pvector, cindex, var, value, 0);
   }
   else
   {
      /* TODO */
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructVectorSetBoxValues( HYPRE_SStructVector  vector,
                                 int                  part,
                                 int                 *ilower,
                                 int                 *iupper,
                                 int                  var,
                                 double              *values )
{
   int ierr = 0;
   int                   ndim    = hypre_SStructVectorNDim(vector);
   hypre_SStructPVector *pvector = hypre_SStructVectorPVector(vector, part);
   hypre_Index           cilower;
   hypre_Index           ciupper;

   hypre_CopyToCleanIndex(ilower, ndim, cilower);
   hypre_CopyToCleanIndex(iupper, ndim, ciupper);

   ierr = hypre_SStructPVectorSetBoxValues(pvector, cilower, ciupper,
                                           var, values, 0);

   return ierr;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructVectorAddToValues( HYPRE_SStructVector  vector,
                                int                  part,
                                int                 *index,
                                int                  var,
                                double              *value )
{
   int ierr = 0;
   int                   ndim    = hypre_SStructVectorNDim(vector);
   hypre_SStructPVector *pvector = hypre_SStructVectorPVector(vector, part);
   hypre_Index           cindex;

   hypre_CopyToCleanIndex(index, ndim, cindex);

   if (var < hypre_SStructPVectorNVars(pvector))
   {
      ierr = hypre_SStructPVectorSetValues(pvector, cindex, var, value, 1);
   }
   else
   {
      /* TODO */
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructVectorAddToBoxValues( HYPRE_SStructVector  vector,
                                   int                  part,
                                   int                 *ilower,
                                   int                 *iupper,
                                   int                  var,
                                   double              *values )
{
   int ierr = 0;
   int                   ndim    = hypre_SStructVectorNDim(vector);
   hypre_SStructPVector *pvector = hypre_SStructVectorPVector(vector, part);
   hypre_Index           cilower;
   hypre_Index           ciupper;

   hypre_CopyToCleanIndex(ilower, ndim, cilower);
   hypre_CopyToCleanIndex(iupper, ndim, ciupper);

   ierr = hypre_SStructPVectorSetBoxValues(pvector, cilower, ciupper,
                                           var, values, 1);

   return ierr;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

int 
HYPRE_SStructVectorAssemble( HYPRE_SStructVector vector )
{
   int ierr = 0;
   int            nparts   = hypre_SStructVectorNParts(vector);
   HYPRE_IJVector ijvector = hypre_SStructVectorIJVector(vector);
   int            part;

   for (part = 0; part < nparts; part++)
   {
      hypre_SStructPVectorAssemble(hypre_SStructVectorPVector(vector, part));
   }

#if 0
   /* ZTODO: construct comm_pkg for communications between parts */
   hypre_CommPkgDestroy(comm_pkgs[var]);
   comm_pkgs[var] =
      hypre_CommPkgCreate(send_boxes, recv_boxes,
                          unit_stride, unit_stride,
                          hypre_StructVectorDataSpace(svectors[var]),
                          hypre_StructVectorDataSpace(svectors[var]),
                          send_processes, recv_processes, 1,
                          hypre_StructVectorComm(svectors[var]),
                          hypre_StructGridPeriodic(sgrid));
#endif

   /* u-vector */
   ierr = HYPRE_IJVectorAssemble(ijvector);

   HYPRE_IJVectorGetObject(ijvector, 
                          (void **) &hypre_SStructVectorParVector(vector));

   /* if the object type is parcsr, then convert the sstruct vector which has ghost
      layers to a parcsr vector without ghostlayers. */
   if (hypre_SStructVectorObjectType(vector) == HYPRE_PARCSR)
   {
      hypre_SStructVectorParConvert(vector, 
                                   &hypre_SStructVectorParVector(vector));
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

int 
HYPRE_SStructVectorGather( HYPRE_SStructVector vector )
{
   int ierr = 0;
   int            nparts   = hypre_SStructVectorNParts(vector);
   int            part;

   /* GEC1102 we change the name of the restore-->parrestore  */

   if (hypre_SStructVectorObjectType(vector) == HYPRE_PARCSR)
   {
      hypre_SStructVectorParRestore(vector, hypre_SStructVectorParVector(vector));
   }

   for (part = 0; part < nparts; part++)
   {
      hypre_SStructPVectorGather(hypre_SStructVectorPVector(vector, part));
   }

#if 0
   /* ZTODO: gather data from other parts */
   {
      int                    nvars     = hypre_SStructPVectorNVars(pvector);
      hypre_StructVector   **svectors  = hypre_SStructPVectorSVectors(pvector);
      hypre_CommPkg        **comm_pkgs = hypre_SStructPVectorCommPkgs(pvector);
      hypre_CommHandle      *comm_handle;
      int                    var;

      for (var = 0; var < nvars; var++)
      {
         if (comm_pkgs[var] != NULL)
         {
            hypre_InitializeCommunication(comm_pkgs[var],
                                          hypre_StructVectorData(svectors[var]),
                                          hypre_StructVectorData(svectors[var]),
                                          &comm_handle);
            hypre_FinalizeCommunication(comm_handle);
         }
      }
   }
#endif

   return ierr;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructVectorGetValues( HYPRE_SStructVector  vector,
                              int                  part,
                              int                 *index,
                              int                  var,
                              double              *value )
{
   int ierr = 0;
   int                   ndim    = hypre_SStructVectorNDim(vector);
   hypre_SStructPVector *pvector = hypre_SStructVectorPVector(vector, part);
   hypre_Index           cindex;

   hypre_CopyToCleanIndex(index, ndim, cindex);

   if (var < hypre_SStructPVectorNVars(pvector))
   {
      ierr = hypre_SStructPVectorGetValues(pvector, cindex, var, value);
   }
   else
   {
      /* TODO */
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructVectorGetBoxValues(HYPRE_SStructVector  vector,
                                int                  part,
                                int                 *ilower,
                                int                 *iupper,
                                int                  var,
                                double              *values )
{
   int ierr = 0;
   int                   ndim    = hypre_SStructVectorNDim(vector);
   hypre_SStructPVector *pvector = hypre_SStructVectorPVector(vector, part);
   hypre_Index           cilower;
   hypre_Index           ciupper;

   hypre_CopyToCleanIndex(ilower, ndim, cilower);
   hypre_CopyToCleanIndex(iupper, ndim, ciupper);

   ierr = hypre_SStructPVectorGetBoxValues(pvector, cilower, ciupper,
                                           var, values);

   return ierr;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

int 
HYPRE_SStructVectorSetConstantValues( HYPRE_SStructVector vector,
                                      double              value )
{
   int ierr = 0;
   hypre_SStructPVector *pvector;
   int part;
   int nparts   = hypre_SStructVectorNParts(vector);

   for ( part = 0; part < nparts; part++ )
   {
      pvector = hypre_SStructVectorPVector( vector, part );
      ierr += hypre_SStructPVectorSetConstantValues( pvector, value );
   };

   return ierr;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructVectorSetObjectType( HYPRE_SStructVector  vector,
                                  int                  type )
{
   int ierr = 0;

   /* this implements only HYPRE_PARCSR, which is always available */
   hypre_SStructVectorObjectType(vector) = type;

   return ierr;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructVectorGetObject( HYPRE_SStructVector   vector,
                              void                **object )
{
   int ierr = 0;
   int vector_type = hypre_SStructVectorObjectType(vector);
   hypre_SStructPVector *pvector;
   hypre_StructVector   *svector;

   int part, var;

   /* GEC1102 in case the vector is HYPRE_SSTRUCT  */

   if (vector_type == HYPRE_SSTRUCT)
   {
      hypre_SStructVectorConvert(vector, (hypre_ParVector **) object);
   }
   else if (vector_type == HYPRE_PARCSR)
   {
     *object= hypre_SStructVectorParVector(vector);
   }
   else if (vector_type == HYPRE_STRUCT)
   {
      /* only one part & one variable */
      part= 0;
      var = 0;
      pvector= hypre_SStructVectorPVector(vector, part);
      svector= hypre_SStructPVectorSVector(pvector, var);
     *object = svector;
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructVectorPrint( const char          *filename,
                          HYPRE_SStructVector  vector,
                          int                  all )
{
   int  ierr = 0;
   int  nparts = hypre_SStructVectorNParts(vector);
   int  part;
   char new_filename[255];

   for (part = 0; part < nparts; part++)
   {
      sprintf(new_filename, "%s.%02d", filename, part);
      hypre_SStructPVectorPrint(new_filename,
                                hypre_SStructVectorPVector(vector, part),
                                all);
   }

   return ierr;
}

/******************************************************************************
 * copy x to y, y should already exist and be the same size
 *****************************************************************************/
int
HYPRE_SStructVectorCopy( HYPRE_SStructVector x,
                         HYPRE_SStructVector y )
{
   return hypre_SStructCopy( (hypre_SStructVector *)x,
                             (hypre_SStructVector *)y );
}

/******************************************************************************
 * y = a*y, for vector y and scalar a
 *****************************************************************************/
int
HYPRE_SStructVectorScale( double alpha, HYPRE_SStructVector y )
{
   return hypre_SStructScale( alpha, (hypre_SStructVector *)y );
}

/******************************************************************************
 * inner or dot product, result = < x, y >
 *****************************************************************************/
int
HYPRE_SStructInnerProd( HYPRE_SStructVector x,
                        HYPRE_SStructVector y,
                        double *result )
{
   return hypre_SStructInnerProd( (hypre_SStructVector *)x,
                                  (hypre_SStructVector *)y,
                                  result );
}

/******************************************************************************
 * y = y + alpha*x for vectors y, x and scalar alpha
 *****************************************************************************/
int
HYPRE_SStructAxpy( double alpha,
                        HYPRE_SStructVector x,
                        HYPRE_SStructVector y )
{
   return hypre_SStructAxpy( alpha,
                             (hypre_SStructVector *)x,
                             (hypre_SStructVector *)y );
}
