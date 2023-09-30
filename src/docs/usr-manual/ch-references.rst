.. Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
   HYPRE Project Developers. See the top-level COPYRIGHT file for details.

   SPDX-License-Identifier: (Apache-2.0 OR MIT)


.. _references:

.. only:: html or text

   **********
   References
   **********

.. [AsFa1996] S. F. Ashby and R. D. Falgout.  A parallel multigrid
   preconditioned conjugate gradient algorithm for groundwater flow simulations.
   *Nuclear Science and Engineering*, 124(1):145--159, September 1996.  Also
   available as LLNL Technical Report UCRL-JC-122359.

.. [BFKY2011] A. Baker, R. Falgout, T. Kolev, and U. M. Yang.  Multigrid
   smoothers for ultra-parallel computing.  *SIAM J. on Sci. Comp.*,
   33:2864--2887, 2011.  Also available as LLNL technical report
   LLLNL-JRNL-473191.

.. [BaFY2006] A.H. Baker, R.D. Falgout, and U.M. Yang.  An assumed partition
   algorithm for determining processor inter-communication.  *Parallel
   Computing*, 32:394--414, 2006.

.. [BaKY2010] A. Baker, T. Kolev, and U. M. Yang.  Improving algebraic multigrid
   interpolation operators for linear elasticity problems.  *Numer. Linear
   Algebra Appl.*, 17:495--517, 2010.  Also available as LLNL technical report
   LLLNL-JRNL-412928.

.. [BKRHSMTY2021] Luc Berger-Vergiat, Brian Kelley, Sivasankaran Rajamanickam,
   Jonathan Hu, Katarzyna Swirydowicz, Paul Mullowney, Stephen Thomas, Ichitaro
   Yamazaki. Two-Stage Gauss--Seidel Preconditioners and Smoothers for Krylov
   Solvers on a GPU cluster.
   `https://arxiv.org/abs/2104.01196 <https://arxiv.org/abs/2104.01196>`_.

.. [BLOPEWeb] BLOPEX, parallel preconditioned eigenvalue solvers.
   `http://code.google.com/p/blopex/ <http://code.google.com/p/blopex/>`_.

.. [BrFJ2000] P. N. Brown, R. D. Falgout, and J. E. Jones.  Semicoarsening
   multigrid on distributed memory machines.  *SIAM J. Sci. Comput.*,
   21(5):1823--1834, 2000.  Special issue on the Fifth Copper Mountain
   Conference on Iterative Methods. Also available as LLNL technical report
   UCRL-JC-130720.

.. [Chow2000] E. Chow.  A priori sparsity patterns for parallel sparse
   approximate inverse preconditioners.  *SIAM J. Sci. Comput.*,
   21:1804--1822, 2000.

.. [ClEA1999] R. L. Clay et al.  An annotated reference guide to the Finite
   Element Interface (FEI) specification, Version 1.0.  Technical Report
   SAND99-8229, Sandia National Laboratories, Livermore, CA, 1999.

.. [CMakeWeb] CMake, a cross-platform open-source build system.
   `http://www.cmake.org/ <http://www.cmake.org/>`_.

.. [DFNY2008] H. De Sterck, R. Falgout, J. Nolting, and U. M. Yang.
   Distance-two interpolation for parallel algebraic multigrid.  *Numer. Linear
   Algebra Appl.*, 15:115--139, 2008.  Also available as LLNL technical report
   UCRL-JRNL-230844.

.. [DeYH2004] H. De Sterck, U. M. Yang, and J. Heys.  Reducing complexity in
   parallel algebraic multigrid preconditioners.  *SIAM Journal on Matrix
   Analysis and Applications*, 27:1019--1039, 2006.  Also available as LLNL
   technical report UCRL-JRNL-206780.

.. [FaJo2000] R. D. Falgout and J. E. Jones.  Multigrid on massively parallel
   architectures.  In E. Dick, K. Riemslagh, and J. Vierendeels, editors,
   *Multigrid Methods VI*, volume 14 of *Lecture Notes in Computational Science
   and Engineering*, pages 101--107, Berlin, 2000. Springer.  Proc. of the Sixth
   European Multigrid Conference held in Gent, Belgium, September
   27-30, 1999. Also available as LLNL technical report UCRL-JC-133948.

.. [FaJY2004] R. D. Falgout, J. E. Jones, and U. M. Yang.  The design and
   implementation of hypre, a library of parallel high performance
   preconditioners.  In A. M. Bruaset and A. Tveito, editors, *Numerical
   Solution of Partial Differential Equations on Parallel Computers*, pages
   267--294.  Springer--Verlag, 2006.  Also available as LLNL technical report
   UCRL-JRNL-205459.

.. [FaJY2005] R. D. Falgout, J. E. Jones, and U. M. Yang.  Conceptual interfaces
   in hypre.  *Future Generation Computer Systems*, 22:239--251, 2006.  Special
   issue on PDE software. Also available as LLNL technical report
   UCRL-JC-148957.

.. [FaSc2014] Robert D. Falgout and Jacob B. Schroder.  Non-galerkin coarse
   grids for algebraic multigrid.  *SIAM J. Sci. Comput.*, 36(3):309--334, 2014.

.. [GrKo2015] A. Grayver and Tz. Kolev.  Large-scale 3D geoelectromagnetic
   modeling using parallel adaptive high-order finite element method.
   *Geophysics*, 80(6):E277–E291, 2015.  Also available as LLNL technical report
   LLNL-JRNL-665742.

.. [GrMS2006a] M. Griebel, B. Metsch, and M. A. Schweitzer.  Coarse grid
   classification: A parallel coarsening scheme for algebraic multigrid methods.
   *Numerical Linear Algebra with Applications*, 13(2--3):193--214, 2006.  Also
   available as SFB 611 preprint No. 225, Universität Bonn, 2005.

.. [GrMS2006b] M. Griebel, B. Metsch, and M. A. Schweitzer.  Coarse grid
   classification - Part II: Automatic coarse grid agglomeration for parallel
   AMG.  Preprint No. 271, Sonderforschungsbereich 611, Universität Bonn, 2006.

.. [HeYa2002] V. E. Henson and U. M. Yang.  BoomerAMG: a parallel algebraic
   multigrid solver and preconditioner.  *Applied Numerical Mathematics*,
   41(5):155--177, 2002.  Also available as LLNL technical report
   UCRL-JC-141495.

.. [HiXu2006] R. Hiptmair and J. Xu.  Nodal auxiliary space preconditioning in
   :math:`H(curl)` and :math:`H(div)` spaces.  *Numer. Math.*, 2006.

.. [HyPo1999] D. Hysom and A. Pothen.  Efficient parallel computation of ILU(k)
   preconditioners.  In *Proceedings of Supercomputing '99*. ACM, November 1999.
   Published on CDROM, ISBN \#1-58113-091-0, ACM Order \#415990, IEEE Computer
   Society Press Order \# RS00197.

.. [HyPo2001] D. Hysom and A. Pothen.  A scalable parallel algorithm for
   incomplete factor preconditioning.  *SIAM J. Sci. Comput.*,
   22(6):2194--2215, 2001.

.. [KaKu1998] G. Karypis and V. Kumar.  Parallel threshold-based ILU
   factorization.  Technical Report 061, University of Minnesota, Department of
   Computer Science/Army HPC Research Center, Minneapolis, MN 5455, 1998.

.. [Knya2001] A. Knyazev.  Toward the optimal preconditioned eigensolver:
   Locally optimal block preconditioned conjugate gradient method.
   *SIAM J. Sci. Comput.*, 23(2):517--541, 2001.

.. [KLAO2007] A. Knyazev, I. Lashuk, M. Argentati, and E. Ovchinnikov.  Block
   locally optimal preconditioned eigenvalue xolvers (blopex) in hypre and
   petsc.  *SIAM J. Sci. Comput.*, 25(5):2224--2239, 2007.

.. [KoVa2009] Tz. Kolev and P. Vassilevski.  Parallel auxiliary space AMG for
   :math:`H(curl)` problems.  *J. Comput. Math.*, 27:604--623, 2009.  Special
   issue on Adaptive and Multilevel Methods in Electromagnetics.
   UCRL-JRNL-237306.

.. [KoYe1993] L. Yu. Kolotilina and A. Yu. Yeremin. Factorized Sparse
   Approximate Inverse Preconditionings I. Theory. *SIAM J. Matrix Anal. A.*, 14(1):45--58, 1993.
   `https://doi.org/10.1137/0614004 <https://doi.org/10.1137/0614004>`_.

.. [LiSY2021] R. Li, B. Sjogreen and U. M. Yang. A new class of AMG interpolation
   methods based on matrix-matrix multiplications. *SIAM J. Sci. Comput.*, 43(5), 
   S540--S564.

.. [JaFe2015] C. Janna, M. Ferronato, F. Sartoretto and G. Gambolati.
   FSAIPACK: A Software Package for High-Performance Factored Sparse Approximate Inverse
   Preconditioning. *ACM T. Math. Software*, 41(2):1–-26, 2015.
   `https://doi.org/10.1145/2629475 <https://doi.org/10.1145/2629475>`_.

.. [JoLe2006] J. Jones and B. Lee.  A multigrid method for variable coefficient
   maxwell's equations.  *SIAM J. Sci. Comput.*, 27:1689--1708, 2006.

.. [McCo1989] S. F. McCormick.  *Multilevel Adaptive Methods for Partial
   Differential Equations*, volume 6 of *Frontiers in Applied Mathematics*.
   SIAM Books, Philadelphia, 1989.

.. [MoRS1998] J.E. Morel, Randy M. Roberts, and Mikhail J. Shashkov.  A local
   support-operators diffusion discretization scheme for quadrilateral r-z
   meshes.  *J. Comp. Physics*, 144:17--51, 1998.

.. [PaFa2019] V. A. Paludetto Magri, A. Franceschini and C. Janna. A novel algebraic
   multigrid approach based on adaptive smoothing and prolongation for ill-conditioned
   systems. *SIAM J. Sci. Comput.*, 41(1):A190--A219, 2019.
   `https://doi.org/10.1137/17M1161178 <https://doi.org/10.1137/17M1161178>`_.

.. [RuSt1987] J. W. Ruge and K. Stüben.  Algebraic multigrid (AMG).
   In S. F. McCormick, editor, *Multigrid Methods*, volume 3 of *Frontiers in
   Applied Mathematics*, pages 73--130. SIAM, Philadelphia, PA, 1987.

.. [Scha1998] S. Schaffer.  A semi-coarsening multigrid method for elliptic
   partial differential equations with highly discontinuous and anisotropic
   coefficients.  *SIAM J. Sci. Comput.*, 20(1):228--242, 1998.

.. [Stue1999] K. Stüben.  Algebraic multigrid (AMG): an introduction with
   applications.  In U. Trottenberg, C. Oosterlee, and A. Schüller, editors,
   *Multigrid*. Academic Press, 2001.

.. [Umpire] Umpire: Managing Heterogeneous Memory Resources.
   `https://github.com/LLNL/Umpire <https://github.com/LLNL/Umpire>`_.

.. [VaMB1996] P. Vaněk, J. Mandel, and M. Brezina.  Algebraic multigrid based on
   smoothed aggregation for second and fourth order problems.  *Computing*,
   56:179--196, 1996.

.. [VaBM2001] P. Vaněk, M. Brezina, and J. Mandel.  Convergence of algebraic
   multigrid based on smoothed aggregation.  *Numerische Mathematik*,
   88:559--579, 2001.

.. [VaYa2014] P. Vassilevski and U. M. Yang.  Reducing communication in
   algebraic multigrid using additive variants.  *Numer. Linear Algebra Appl.*,
   21:275--296, 2014.  Also available as LLNL technical report
   LLLNL-JRNL-637872.

.. [Yang2004] U. M. Yang.  On the use of relaxation parameters in hybrid
   smoothers.  *Numerical Linear Algebra with Applications*, 11:155--172, 2004.

.. [Yang2005] U. M. Yang.  Parallel algebraic multigrid methods - high
   performance preconditioners.  In A. M. Bruaset and A. Tveito, editors,
   *Numerical Solution of Partial Differential Equations on Parallel Computers*,
   pages 209--236.  Springer-Verlag, 2006.  Also available as LLNL technical
   report UCRL-BOOK-208032.

.. [Yang2010] U. M. Yang.  On long range interpolation operators for aggressive
   coarsening.  *Numer. Linear Algebra Appl.*, 17:453--472, 2010.  Also
   available as LLNL technical report LLLNL-JRNL-417371.
