/*BHEADER**********************************************************************
 * (c) 2000   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

#ifndef HYPRE_SSTRUCT_MV_HEADER
#define HYPRE_SSTRUCT_MV_HEADER

#include "HYPRE_config.h"
#include "HYPRE_utilities.h"
#include "HYPRE_mv.h"
#include "HYPRE_IJ_mv.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name SStruct Linear Solvers Interface
 *
 * A general description of the interface goes here...
 *
 * @memo A linear solver interface for semi-structured grids
 * @version 0.2
 * @author Robert D. Falgout
 **/
/*@{*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name SStruct Grids
 *
 * A grid object is constructed out of several structured ``parts''
 * and an optional unstructured ``part''.  Each structured part has
 * its own abstract index space. 
 **/
/*@{*/

/**
 * The {\tt HYPRE\_SStructGrid} object ...
 **/
struct hypre_SStructGrid_struct;
typedef struct hypre_SStructGrid_struct *HYPRE_SStructGrid;

/**
 * An enumerated type that supports cell centered, node centered, face
 * centered, and edge centered variables.  Face centered variables are
 * split into x-face, y-face, and z-face variables, and edge centered
 * variables are split into x-edge, y-edge, and z-edge variables.  The
 * edge centered variable types are only used in 3D.  In 2D, edge
 * centered variables are handled by the face centered types.
 *
 * Variables are referenced relative to an abstract (cell centered)
 * index in the following way:
 * \begin{itemize}
 * \item cell centered variables are aligned with the index;
 * \item node centered variables are aligned with the cell corner
 *       at relative index (1/2, 1/2, 1/2);
 * \item x-face, y-face, and z-face centered variables are aligned
 *       with the faces at relative indexes (1/2, 0, 0), (0, 1/2, 0),
 *       and (0, 0, 1/2), respectively;
 * \item x-edge, y-edge, and z-edge centered variables are aligned
 *       with the edges at relative indexes (0, 1/2, 1/2), (1/2, 0, 1/2),
 *       and (1/2, 1/2, 0), respectively.
 * \end{itemize}
 *
 * The supported identifiers are:
 * \begin{itemize}
 * \item {\tt HYPRE\_SSTRUCT\_VARIABLE\_CELL}
 * \item {\tt HYPRE\_SSTRUCT\_VARIABLE\_NODE}
 * \item {\tt HYPRE\_SSTRUCT\_VARIABLE\_XFACE}
 * \item {\tt HYPRE\_SSTRUCT\_VARIABLE\_YFACE}
 * \item {\tt HYPRE\_SSTRUCT\_VARIABLE\_ZFACE}
 * \item {\tt HYPRE\_SSTRUCT\_VARIABLE\_XEDGE}
 * \item {\tt HYPRE\_SSTRUCT\_VARIABLE\_YEDGE}
 * \item {\tt HYPRE\_SSTRUCT\_VARIABLE\_ZEDGE}
 * \end{itemize}
 *
 * NOTE: Although variables are referenced relative to a unique
 * abstract cell-centered index, some variables are associated with
 * multiple grid cells.  For example, node centered variables in 3D
 * are associated with 8 cells (away from boundaries).  Although grid
 * cells are distributed uniquely to different processes, variables
 * may be owned by multiple processes because they may be associated
 * with multiple cells.
 *
 * @param param [IN] ... 
 **/
enum hypre_SStructVariable_enum
{
   HYPRE_SSTRUCT_VARIABLE_UNDEFINED = -1,
   HYPRE_SSTRUCT_VARIABLE_CELL      =  0,
   HYPRE_SSTRUCT_VARIABLE_NODE      =  1,
   HYPRE_SSTRUCT_VARIABLE_XFACE     =  2,
   HYPRE_SSTRUCT_VARIABLE_YFACE     =  3,
   HYPRE_SSTRUCT_VARIABLE_ZFACE     =  4,
   HYPRE_SSTRUCT_VARIABLE_XEDGE     =  5,
   HYPRE_SSTRUCT_VARIABLE_YEDGE     =  6,
   HYPRE_SSTRUCT_VARIABLE_ZEDGE     =  7
};
typedef enum hypre_SStructVariable_enum HYPRE_SStructVariable;

/**
 * Create an {\tt ndim}-dimensional grid object with {\tt nparts}
 * structured parts.
 *
 * @param param [IN] ...
 **/
int HYPRE_SStructGridCreate(MPI_Comm           comm,
                            int                ndim,
                            int                nparts,
                            HYPRE_SStructGrid *grid);

/**
 * Destroy a grid object.  A grid should be explicitly destroyed using
 * this destructor when the user's code no longer needs direct access
 * to the grid description.  Once destroyed, the object must not be
 * referenced again.  Note that the grid description may not be
 * deallocated at the completion of this call, since there may be
 * internal package references to the object.  The grid will then be
 * destroyed when all internal reference counts go to zero.
 *
 * @param param [IN] ...
 **/
int HYPRE_SStructGridDestroy(HYPRE_SStructGrid grid);

/**
 * Set the extents for a box on a structured part of the grid.
 *
 * @param param [IN] ...
 **/
int HYPRE_SStructGridSetExtents(HYPRE_SStructGrid  grid,
                                int                part,
                                int               *ilower,
                                int               *iupper);

/**
 * Describe the variables that live on a structured part of the grid.
 *
 * @param param [IN] ...
 **/
int HYPRE_SStructGridSetVariables(HYPRE_SStructGrid      grid,
                                  int                    part,
                                  int                    nvars,
                                  HYPRE_SStructVariable *vartypes);

/**
 * Describe additional variables that live at a particular index.
 * These variables are appended to the array of variables set in {\tt
 * HYPRE\_SStructGridSetVariables}, and are referenced as such.
 *
 * @param param [IN] ... 
 **/
int HYPRE_SStructGridAddVariables(HYPRE_SStructGrid      grid,
                                  int                    part,
                                  int                   *index,
                                  int                    nvars,
                                  HYPRE_SStructVariable *vartypes);

/**
 * Add an unstructured part to the grid.  The variables in the
 * unstructured part of the grid are referenced by a global rank
 * between 0 and the total number of unstructured variables minus one.
 * Each process owns some unique consecutive range of variables,
 * defined by {\tt ilower} and {\tt iupper}.
 *
 * NOTE: This is just a placeholder.  This part of the interface
 * is not finished.
 *
 * @param param [IN] ... 
 **/
int HYPRE_SStructGridAddUnstructuredPart(HYPRE_SStructGrid grid,
                                         int               ilower,
                                         int               iupper);

/**
 * Finalize the construction of the grid before using.
 *
 * @param param [IN] ...
 **/
int HYPRE_SStructGridAssemble(HYPRE_SStructGrid grid);

/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name SStruct Stencils
 *
 * Description...
 **/
/*@{*/

/**
 * The {\tt HYPRE\_SStructStencil} object ...
 **/
struct hypre_SStructStencil_struct;
typedef struct hypre_SStructStencil_struct *HYPRE_SStructStencil;

/**
 * Create a stencil object for the specified number of spatial dimensions
 * and stencil entries.
 *
 * @param param [IN] ...
 **/
int HYPRE_SStructStencilCreate(int                   ndim,
                               int                   size,
                               HYPRE_SStructStencil *stencil);

/**
 * Destroy the stencil object.  The stencil object should be explicitly
 * destroyed using this destructor when the user's code no longer needs to
 * directly access the stencil description.  Once destroyed, the object
 * must not be referenced again.
 *
 * @param param [IN] ...
 **/
int HYPRE_SStructStencilDestroy(HYPRE_SStructStencil stencil);

/**
 * Set a stencil entry ...
 *
 * @param param [IN] ...
 **/
int HYPRE_SStructStencilSetEntry(HYPRE_SStructStencil  stencil,
                                 int                   entry,
                                 int                  *offset,
                                 int                   var);

/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name SStruct Graphs
 *
 * Description...
 **/
/*@{*/

/**
 * The {\tt HYPRE\_SStructGraph} object ...
 **/
struct hypre_SStructGraph_struct;
typedef struct hypre_SStructGraph_struct *HYPRE_SStructGraph;

/**
 * Create a graph object, such as the nonzero structure of a matrix.
 *
 * @param param [IN] ...
 **/
int HYPRE_SStructGraphCreate(MPI_Comm             comm,
                             HYPRE_SStructGrid    grid,
                             HYPRE_SStructGraph  *graph);

/**
 * Destroy a graph object.  A graph should be explicitly destroyed
 * using this destructor when the user's code no longer needs to
 * directly access the graph description.  Once destroyed, the object
 * must not be referenced again.
 *
 * @param param [IN] ...
 **/
int HYPRE_SStructGraphDestroy(HYPRE_SStructGraph graph);

/**
 * Set the stencil for a variable on a structured part of the grid.
 *
 * @param param [IN] ...
 **/
int HYPRE_SStructGraphSetStencil(HYPRE_SStructGraph   graph,
                                 int                  part,
                                 int                  var,
                                 HYPRE_SStructStencil stencil);

/**
 * Add non-stencil graph entries at a particular index.  These graph
 * entries are appended to the existing graph entries, and are
 * referenced as such.
 *
 * NOTE: Users are required to set graph entries on all processes that
 * own the associated variables.  This means that some data will be
 * multiply defined.
 *
 * @param param [IN] ... 
 **/
int HYPRE_SStructGraphAddEntries(HYPRE_SStructGraph   graph,
                                 int                  part,
                                 int                 *index,
                                 int                  var,
                                 int                  nentries,
                                 int                  to_part,
                                 int                **to_indexes,
                                 int                  to_var);

/**
 * Finalize the construction of the graph before using.
 *
 * @param param [IN] ...
 **/
int HYPRE_SStructGraphAssemble(HYPRE_SStructGraph graph);

/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name SStruct Matrices
 *
 * Description...
 **/
/*@{*/

/**
 * The {\tt HYPRE\_SStructMatrix} object ...
 **/
struct hypre_SStructMatrix_struct;
typedef struct hypre_SStructMatrix_struct *HYPRE_SStructMatrix;

/**
 * Create a matrix object.
 *
 * @param param [IN] ...
 **/
int HYPRE_SStructMatrixCreate(MPI_Comm              comm,
                              HYPRE_SStructGraph    graph,
                              HYPRE_SStructMatrix  *matrix);

/**
 * Destroy a matrix object.
 *
 * @param param [IN] ...
 **/
int HYPRE_SStructMatrixDestroy(HYPRE_SStructMatrix matrix);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_SStructMatrixInitialize(HYPRE_SStructMatrix matrix);

/**
 * Set matrix coefficients index by index.
 *
 * NOTE: Users are required to set values on all processes that own
 * the associated variables.  This means that some data will be
 * multiply defined.
 *
 * NOTE: The entries in this routine must all be of the same type:
 * either stencil or non-stencil, but not both.  Also, if they are
 * stencil entries, they must all represent couplings to the same
 * variable type.
 *
 * @param param [IN] ... 
 **/
int HYPRE_SStructMatrixSetValues(HYPRE_SStructMatrix  matrix,
                                 int                  part,
                                 int                 *index,
                                 int                  var,
                                 int                  nentries,
                                 int                 *entries,
                                 double              *values);

/**
 * Set matrix coefficients a box at a time.
 *
 * NOTE: Users are required to set values on all processes that own
 * the associated variables.  This means that some data will be
 * multiply defined.
 *
 * NOTE: The entries in this routine must all be of stencil type.
 * Also, they must all represent couplings to the same variable type.
 *
 * @param param [IN] ...
 **/
int HYPRE_SStructMatrixSetBoxValues(HYPRE_SStructMatrix  matrix,
                                    int                  part,
                                    int                 *ilower,
                                    int                 *iupper,
                                    int                  var,
                                    int                  nentries,
                                    int                 *entries,
                                    double              *values);
/**
 * Add to matrix coefficients index by index.
 *
 * NOTE: Users are required to set values on all processes that own
 * the associated variables.  This means that some data will be
 * multiply defined.
 *
 * NOTE: The entries in this routine must all be of the same type:
 * either stencil or non-stencil, but not both.  Also, if they are
 * stencil entries, they must all represent couplings to the same
 * variable type.
 *
 * @param param [IN] ...
 **/
int HYPRE_SStructMatrixAddToValues(HYPRE_SStructMatrix  matrix,
                                   int                  part,
                                   int                 *index,
                                   int                  var,
                                   int                  nentries,
                                   int                 *entries,
                                   double              *values);

/**
 * Add to matrix coefficients a box at a time.
 *
 * NOTE: Users are required to set values on all processes that own
 * the associated variables.  This means that some data will be
 * multiply defined.
 *
 * NOTE: The entries in this routine must all be of stencil type.
 * Also, they must all represent couplings to the same variable type.
 *
 * @param param [IN] ...
 **/
int HYPRE_SStructMatrixAddToBoxValues(HYPRE_SStructMatrix  matrix,
                                      int                  part,
                                      int                 *ilower,
                                      int                 *iupper,
                                      int                  var,
                                      int                  nentries,
                                      int                 *entries,
                                      double              *values);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_SStructMatrixAssemble(HYPRE_SStructMatrix matrix);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_SStructMatrixSetSymmetric(HYPRE_SStructMatrix  matrix,
                                    int                  symmetric);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_SStructMatrixPrint(char                *filename,
                             HYPRE_SStructMatrix  matrix,
                             int                  all);

/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name SStruct Vectors
 *
 * Description...
 **/
/*@{*/

/**
 * The {\tt HYPRE\_SStructVector} object ...
 **/
struct hypre_SStructVector_struct;
typedef struct hypre_SStructVector_struct *HYPRE_SStructVector;

/**
 * Create a vector object.
 *
 * @param param [IN] ...
 **/
int HYPRE_SStructVectorCreate(MPI_Comm              comm,
                              HYPRE_SStructGrid     grid,
                              HYPRE_SStructVector  *vector);

/**
 * Destroy a vector object.
 *
 * @param param [IN] ...
 **/
int HYPRE_SStructVectorDestroy(HYPRE_SStructVector vector);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_SStructVectorInitialize(HYPRE_SStructVector vector);

/**
 * Set vector coefficients index by index.
 *
 * NOTE: Users are required to set values on all processes that own
 * the associated variables.  This means that some data will be
 * multiply defined.
 *
 * @param param [IN] ...
 **/
int HYPRE_SStructVectorSetValues(HYPRE_SStructVector  vector,
                                 int                  part,
                                 int                 *index,
                                 int                  var,
                                 double               value);

/**
 * Set vector coefficients a box at a time.
 *
 * NOTE: Users are required to set values on all processes that own
 * the associated variables.  This means that some data will be
 * multiply defined.
 *
 * @param param [IN] ...
 **/
int HYPRE_SStructVectorSetBoxValues(HYPRE_SStructVector  vector,
                                    int                  part,
                                    int                 *ilower,
                                    int                 *iupper,
                                    int                  var,
                                    double              *values);
/**
 * Set vector coefficients index by index.
 *
 * NOTE: Users are required to set values on all processes that own
 * the associated variables.  This means that some data will be
 * multiply defined.
 *
 * @param param [IN] ...
 **/
int HYPRE_SStructVectorAddToValues(HYPRE_SStructVector  vector,
                                   int                  part,
                                   int                 *index,
                                   int                  var,
                                   double               value);

/**
 * Set vector coefficients a box at a time.
 *
 * NOTE: Users are required to set values on all processes that own
 * the associated variables.  This means that some data will be
 * multiply defined.
 *
 * @param param [IN] ...
 **/
int HYPRE_SStructVectorAddToBoxValues(HYPRE_SStructVector  vector,
                                      int                  part,
                                      int                 *ilower,
                                      int                 *iupper,
                                      int                  var,
                                      double              *values);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_SStructVectorAssemble(HYPRE_SStructVector vector);


/**
 * Gather vector data so that efficient GetValues can be done.
 *
 * @param param [IN] ...
 **/
int HYPRE_SStructVectorGather(HYPRE_SStructVector vector);


/**
 * Description...
 *
 * NOTE: Users may only get values on processes that own the
 * associated variables.
 *
 * @param param [IN] ... 
 **/
int HYPRE_SStructVectorGetValues(HYPRE_SStructVector  vector,
                                 int                  part,
                                 int                 *index,
                                 int                  var,
                                 double              *value);

/**
 * Description...
 *
 * NOTE: Users may only get values on processes that own the
 * associated variables.
 *
 * @param param [IN] ...
 **/
int HYPRE_SStructVectorGetBoxValues(HYPRE_SStructVector  vector,
                                    int                  part,
                                    int                 *ilower,
                                    int                 *iupper,
                                    int                  var,
                                    double              *values);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_SStructVectorPrint(char                *filename,
                             HYPRE_SStructVector  vector,
                             int                  all);

/*@}*/
/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

#ifdef __cplusplus
}
#endif

#endif

