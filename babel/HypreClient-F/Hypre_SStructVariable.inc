C       
C       File:          Hypre_SStructVariable.inc
C       Symbol:        Hypre.SStructVariable-v0.1.7
C       Symbol Type:   enumeration
C       Babel Version: 0.8.0
C       SIDL Created:  20030306 17:05:23 PST
C       Generated:     20030306 17:05:27 PST
C       Description:   Client-side glue code for Hypre.SStructVariable
C       
C       WARNING: Automatically generated; changes will be lost
C       
C       babel-version = 0.8.0
C       source-line   = 898
C       source-url    = file:/home/falgout/linear_solvers/babel/Interfaces.idl
C       


C       
C       Symbol "Hypre.SStructVariable" (version 0.1.7)
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


        integer UNDEFINED
        parameter (UNDEFINED = -1)
        integer CELL
        parameter (CELL = 0)
        integer NODE
        parameter (NODE = 1)
        integer XFACE
        parameter (XFACE = 2)
        integer YFACE
        parameter (YFACE = 3)
        integer ZFACE
        parameter (ZFACE = 4)
        integer XEDGE
        parameter (XEDGE = 5)
        integer YEDGE
        parameter (YEDGE = 6)
        integer ZEDGE
        parameter (ZEDGE = 7)
