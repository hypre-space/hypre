C       
C       File:          bHYPRE_SStructVariable.inc
C       Symbol:        bHYPRE.SStructVariable-v1.0.0
C       Symbol Type:   enumeration
C       Babel Version: 0.8.2
C       SIDL Created:  20030401 14:47:46 PST
C       Generated:     20030401 14:48:09 PST
C       Description:   Client-side glue code for bHYPRE.SStructVariable
C       
C       WARNING: Automatically generated; changes will be lost
C       
C       babel-version = 0.8.2
C       source-line   = 888
C       source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
C       


C       
C       Symbol "bHYPRE.SStructVariable" (version 1.0.0)
C       
C       The SStructVariable enumerated type.
C       
C       An enumerated type that supports cell centered, node
C       centered, face centered, and edge centered variables.  Face
C       centered variables are split into x-face, y-face, and z-face
C       variables, and edge centered variables are split into x-edge,
C       y-edge, and z-edge variables.  The edge centered variable
C       types are only used in 3D.  In 2D, edge centered variables
C       are handled by the face centered types.
C       
C       Variables are referenced relative to an abstract (cell centered)
C       index in the following way:
C       \begin{itemize}
C       \item cell centered variables are aligned with the index;
C       \item node centered variables are aligned with the cell corner
C             at relative index (1/2, 1/2, 1/2);
C       \item x-face, y-face, and z-face centered variables are aligned
C             with the faces at relative indexes (1/2, 0, 0), (0, 1/2, 0),
C             and (0, 0, 1/2), respectively;
C       \item x-edge, y-edge, and z-edge centered variables are aligned
C             with the edges at relative indexes (0, 1/2, 1/2), (1/2, 0, 1/2),
C             and (1/2, 1/2, 0), respectively.
C       \end{itemize}
C       
C       The supported identifiers are:
C       \begin{itemize}
C       \item {\tt HYPRE\_SSTRUCT\_VARIABLE\_CELL}
C       \item {\tt HYPRE\_SSTRUCT\_VARIABLE\_NODE}
C       \item {\tt HYPRE\_SSTRUCT\_VARIABLE\_XFACE}
C       \item {\tt HYPRE\_SSTRUCT\_VARIABLE\_YFACE}
C       \item {\tt HYPRE\_SSTRUCT\_VARIABLE\_ZFACE}
C       \item {\tt HYPRE\_SSTRUCT\_VARIABLE\_XEDGE}
C       \item {\tt HYPRE\_SSTRUCT\_VARIABLE\_YEDGE}
C       \item {\tt HYPRE\_SSTRUCT\_VARIABLE\_ZEDGE}
C       \end{itemize}
C       
C       NOTE: Although variables are referenced relative to a unique
C       abstract cell-centered index, some variables are associated
C       with multiple grid cells.  For example, node centered
C       variables in 3D are associated with 8 cells (away from
C       boundaries).  Although grid cells are distributed uniquely to
C       different processes, variables may be owned by multiple
C       processes because they may be associated with multiple cells.
C       
C       


        integer*4 UNDEFINED
        parameter (UNDEFINED = -1)
        integer*4 CELL
        parameter (CELL = 0)
        integer*4 NODE
        parameter (NODE = 1)
        integer*4 XFACE
        parameter (XFACE = 2)
        integer*4 YFACE
        parameter (YFACE = 3)
        integer*4 ZFACE
        parameter (ZFACE = 4)
        integer*4 XEDGE
        parameter (XEDGE = 5)
        integer*4 YEDGE
        parameter (YEDGE = 6)
        integer*4 ZEDGE
        parameter (ZEDGE = 7)
