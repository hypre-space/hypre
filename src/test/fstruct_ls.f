!     Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
!     HYPRE Project Developers. See the top-level COPYRIGHT file for details.
!
!     SPDX-License-Identifier: (Apache-2.0 OR MIT)

!***********************************************************************
!     Routines to test struct_ls fortran interfaces
!***********************************************************************


!***********************************************************************
!             HYPRE_StructBiCGSTAB routines
!***********************************************************************

!***********************************************************************
!     fhypre_structbicgstabcreate
!***********************************************************************
      subroutine fhypre_structbicgstabcreate(fcomm, fsolver)
      integer ierr
      integer fcomm
      integer*8 fsolver

      call HYPRE_StructBiCGSTABCreate(fcomm, fsolver, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structbicgstabcreate: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structbicgstabdestroy
!***********************************************************************
      subroutine fhypre_structbicgstabdestroy(fsolver)
      integer ierr
      integer*8 fsolver

      call HYPRE_StructBiCGSTABDestroy(fsolver, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structbicgstabdestroy: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structbicgstabsetup
!***********************************************************************
      subroutine fhypre_structbicgstabsetup(fsolver, fA, fb, fx)
      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call HYPRE_StructBiCGSTABSetup(fsolver, fA, fb, fx, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structbicgstabsetup: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structbicgstabsolve
!***********************************************************************
      subroutine fhypre_structbicgstabsolve(fsolver, fA, fb, fx)
      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call HYPRE_StructBiCGSTABSolve(fsolver, fA, fb, fx, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structbicgstabsolve: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structbicgstabsettol
!***********************************************************************
      subroutine fhypre_structbicgstabsettol(fsolver, ftol)
      integer ierr
      integer*8 fsolver
      double precision ftol

      call HYPRE_StructBiCGSTABSetTol(fsolver, ftol, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structbicgstabsettol: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structbicgstabsetmaxiter
!***********************************************************************
      subroutine fhypre_structbicgstabsetmaxiter(fsolver, fmaxiter)
      integer ierr
      integer fmaxiter
      integer*8 fsolver

      call HYPRE_StructBiCGSTABSetMaxIter(fsolver, fmaxiter, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structbicgstabsetmaxiter: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structbicgstabsetprecond
!***********************************************************************
      subroutine fhypre_structbicgstabsetprecond(fsolver, fprecond_id,
     1                                           fprecond_solver)
      integer ierr
      integer*8 fsolver
      integer*8 fprecond_id
      integer*8 fprecond_solver

      call HYPRE_StructBiCGSTABSetPrecond(fsolver, fprecond_id,
     1                                    fprecond_solver, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structbicgstabsetprecond: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structbicgstabsetlogging
!***********************************************************************
      subroutine fhypre_structbicgstabsetlogging(fsolver, flogging)
      integer ierr
      integer flogging
      integer*8 fsolver

      call HYPRE_StructBiCGSTABSetLogging(fsolver, flogging, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structbicgstabsetlogging: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structbicgstabsetprintlevel
!***********************************************************************
      subroutine fhypre_structbicgstabsetprintle(fsolver, fprintlev)
      integer ierr
      integer fprintlev
      integer*8 fsolver

      call HYPRE_StructBiCGSTABSetPrintLev(fsolver, fprintlev, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structbicgstabsetprintle: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structbicgstabgetnumiterations
!***********************************************************************
      subroutine fhypre_structbicgstabgetnumiter(fsolver, fnumiter)
      integer ierr
      integer fnumiter
      integer*8 fsolver

      call HYPRE_StructBiCGSTABGetNumItera(fsolver, fnumiter, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structbicgstabgetnumiter: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structbicgstabgetresidual
!***********************************************************************
      subroutine fhypre_structbicgstabgetresidua(fsolver, fresidual)
      integer ierr
      integer*8 fsolver
      double precision fresidual

      call HYPRE_StructBiCGSTABGetResidual(fsolver, fresidual, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structbicgstabgetresidua: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structbicgstabgetfinalrelativeresidualnorm
!***********************************************************************
      subroutine fhypre_structbicgstabgetfinalre(fsolver, fnorm)
      integer ierr
      integer*8 fsolver
      double precision fnorm

      call HYPRE_StructBiCGSTABGetFinalRel(fsolver, fnorm)
      if (ierr .ne. 0) then
         print *, 'fhypre_structbicgstabgetfinalre: err = ', ierr
      endif

      return
      end





!***********************************************************************
!             HYPRE_StructGMRES routines
!***********************************************************************

!***********************************************************************
!     fhypre_structgmrescreate
!***********************************************************************
      subroutine fhypre_structgmrescreate(fcomm, fsolver)
      integer ierr
      integer fcomm
      integer*8 fsolver

      call HYPRE_StructGMRESCreate(fcomm, fsolver, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structgmrescreate: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structgmresdestroy
!***********************************************************************
      subroutine fhypre_structgmresdestroy(fsolver)
      integer ierr
      integer*8 fsolver

      call HYPRE_StructGMRESDestroy(fsolver, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structgmresdestroy: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structgmressetup
!***********************************************************************
      subroutine fhypre_structgmressetup(fsolver, fA, fb, fx)
      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call HYPRE_StructGMRESSetup(fsolver, fA, fb, fx, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structgmressetup: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structgmressolve
!***********************************************************************
      subroutine fhypre_structgmressolve(fsolver, fA, fb, fx)
      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call HYPRE_StructGMRESSolve(fsolver, fA, fb, fx, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structgmressolve: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structgmressettol
!***********************************************************************
      subroutine fhypre_structgmressettol(fsolver, ftol)
      integer ierr
      integer*8 fsolver
      double precision ftol

      call HYPRE_StructGMRESSetTol(fsolver, ftol, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structgmressettol: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structgmressetmaxiter
!***********************************************************************
      subroutine fhypre_structgmressetmaxiter(fsolver, fmaxiter)
      integer ierr
      integer fmaxiter
      integer*8 fsolver

      call HYPRE_StructGMRESSetMaxIter(fsolver, fmaxiter, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structgmressetmaxiter: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structgmressetprecond
!***********************************************************************
      subroutine fhypre_structgmressetprecond(fsolver, fprecond_id,
     1                                        fprecond_solver)
      integer ierr
      integer fprecond_id
      integer*8 fsolver
      integer*8 fprecond_solver

      call HYPRE_StructGMRESSetPrecond(fsolver, fprecond_id,
     1                                 fprecond_solver, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structgmressetprecond: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structgmressetlogging
!***********************************************************************
      subroutine fhypre_structgmressetlogging(fsolver, flogging)
      integer ierr
      integer flogging
      integer*8 fsolver

      call HYPRE_StructGMRESSetLogging(fsolver, flogging, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structgmressetlogging: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structgmressetprintlevel
!***********************************************************************
      subroutine fhypre_structgmressetprintlevel(fsolver, fprintlevel)
      integer ierr
      integer fprintlevel
      integer*8 fsolver

      call HYPRE_StructGMRESSetPrintLevel(fsolver, fprint_level, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structgmressetprintlevel: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structgmresgetnumiterations
!***********************************************************************
      subroutine fhypre_structgmresgetnumiterati(fsolver, fnumiters)
      integer ierr
      integer fnumiters
      integer*8 fsolver

      call HYPRE_StructGMRESGetNumIteratio(fsolver, fnumiters, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structgmresgetnumiterati: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structgmresgetfinalrelativeresidualnorm
!***********************************************************************
      subroutine fhypre_structgmresgetfinalrelat(fsolver, fnorm)
      integer ierr
      integer*8 fsolver
      double precision fnorm

      call HYPRE_StructGMRESGetFinalRelati(fsolver, fnorm, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structgmresgetfinalrelat: err = ', ierr
      endif

      return
      end





!***********************************************************************
!             HYPRE_StructHybrid routines
!***********************************************************************

!***********************************************************************
!     fhypre_structhybridcreate
!***********************************************************************
      subroutine fhypre_structhybridcreate(fcomm, fsolver)
      integer ierr
      integer fcomm
      integer*8 fsolver

      call HYPRE_StructHybridCreate(fcomm, fsolver, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structhybridcreate: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structhybriddestroy
!***********************************************************************
      subroutine fhypre_structhybriddestroy(fsolver)
      integer ierr
      integer*8 fsolver

      call HYPRE_StructHybridDestroy(fsolver, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structhybriddestroy: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structhybridsetup
!***********************************************************************
      subroutine fhypre_structhybridsetup(fsolver, fA, fb, fx)
      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call HYPRE_StructHybridSetup(fsolver, fA, fb, fx, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structhybridsetup: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structhybridsolve
!***********************************************************************
      subroutine fhypre_structhybridsolve(fsolver, fA, fb, fx)
      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call HYPRE_StructHybridSolve(fsolver, fA, fb, fx, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structhybridsolve: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structhybridsetsolvertype
!***********************************************************************
      subroutine fhypre_structhybridsetsolvertyp(fsolver, fsolver_typ)
      integer ierr
      integer fsolver_typ
      integer*8 fsolver

      call HYPRE_StructHybridSetSolverType(fsolver, fsolver_typ, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structhybridsetsolvertyp: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structhybridsetstopcrit
!***********************************************************************
      subroutine fhypre_structhybridsetstopcrit(fsolver, fstop_crit)
      integer ierr
      integer fstop_crit
      integer*8 fsolver

      call HYPRE_StructHybridSetStopCrit(fsolver, fstop_crit, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structhybridsetstopcrit: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structhybridsetkdim
!***********************************************************************
      subroutine fhypre_structhybridsetkdim(fsolver, fkdim)
      integer ierr
      integer fkdim
      integer*8 fsolver

      call HYPRE_StructHybridSetKDim(fsolver, fkdim, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structhybridsetkdim: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structhybridsettol
!***********************************************************************
      subroutine fhypre_structhybridsettol(fsolver, ftol)
      integer ierr
      integer*8 fsolver
      double precision ftol

      call HYPRE_StructHybridSetTol(fsolver, ftol, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structhybridsettol: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structhybridsetconvergencetol
!***********************************************************************
      subroutine fhypre_structhybridsetconvergen(fsolver, fcftol)
      integer ierr
      integer*8 fsolver
      double precision fcftol

      call HYPRE_StructHybridSetConvergenc(fsolver, fcftol, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structhybridsetconvergen: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structhybridsetpcgabsolutetolfactor
!***********************************************************************
      subroutine fhypre_structhybridsetpcgabsolu(fsolver, fpcgtol)
      integer ierr
      integer*8 fsolver
      double precision fpcgtol

      call HYPRE_StructHybridSetPCGAbsolut(fsolver, fpcgtol, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structhybridsetpcgabsolu: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structhybridsetdscgmaxiter
!***********************************************************************
      subroutine fhypre_structhybridsetdscgmaxit(fsolver, fdscgmaxitr)
      integer ierr
      integer fdscgmaxitr
      integer*8 fsolver

      call HYPRE_StructHybridSetDSCGMaxIte(fsolver, fdscgmaxitr, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structhybridsetdscgmaxit: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structhybridsetpcgmaxiter
!***********************************************************************
      subroutine fhypre_structhybridsetpcgmaxite(fsolver, fpcgmaxitr)
      integer ierr
      integer fpcgmaxitr
      integer*8 fsolver

      call HYPRE_StructHybridSetPCGMaxIter(fsolver, fpcgmaxitr, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structhybridsetpcgmaxite: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structhybridsettwonorm
!***********************************************************************
      subroutine fhypre_structhybridsettwonorm(fsolver, ftwonorm)
      integer ierr
      integer ftwonorm
      integer*8 fsolver

      call HYPRE_StructHybridSetTwoNorm(fsolver, ftwonorm, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structhybridsettwonorm: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structhybridsetrelchange
!***********************************************************************
      subroutine fhypre_structhybridsetrelchange(fsolver, frelchng)
      integer ierr
      integer frelchng
      integer*8 fsolver

      call HYPRE_StructHybridSetRelChange(fsolver, frelchng, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structhybridsetrelchange: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structhybridsetprecond
!***********************************************************************
      subroutine fhypre_structhybridsetprecond(fsolver, fprecond_id,
     1                                         fprecond)
      integer ierr
      integer fprecond_id
      integer*8 fsolver
      integer*8 fprecond

      call HYPRE_StructHybridSetPrecond(fsolver, fprecond_id, fprecond,
     1                                  ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structhybridsetprecond: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structhybridsetlogging
!***********************************************************************
      subroutine fhypre_structhybridsetlogging(fsolver, flogging)
      integer ierr
      integer flogging
      integer*8 fsolver

      call HYPRE_StructHybridSetLogging(fsolver, flogging, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structhybridsetlogging: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structhybridsetprintlevel
!***********************************************************************
      subroutine fhypre_structhybridsetprintleve(fsolver, fprntlvl)
      integer ierr
      integer fprntlvl
      integer*8 fsolver

      call HYPRE_StructHybridSetPrintLevel(fsolver, fprntlvl, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structhybridsetprintleve: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structhybridgetnumiterations
!***********************************************************************
      subroutine fhypre_structhybridgetnumiterat(fsolver, fnumits)
      integer ierr
      integer fnumits
      integer*8 fsolver

      call HYPRE_StructHybridGetNumIterati(fsolver, fnumits, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structhybridgetnumiterat: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structhybridgetdscgnumiterations
!***********************************************************************
      subroutine fhypre_structhybridgetdscgnumit(fsolver, fdscgnumits)
      integer ierr
      integer fdscgnumits
      integer*8 fsolver

      call HYPRE_StructHybridGetDSCGNumIte(fsolver, fdscgnumits, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structhybridgetdscgnumit: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structhybridgetpcgnumiterations
!***********************************************************************
      subroutine fhypre_structhybridgetpcgnumite(fsolver, fpcgnumits)
      integer ierr
      integer fpcgnumits
      integer*8 fsolver

      call HYPRE_StructHybridGetPCGNumIter(fsolver, fpcgnumits, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structhybridgetpcgnumite: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structhybridgetfinalrelativeresidualnorm
!***********************************************************************
      subroutine fhypre_structhybridgetfinalrela(fsolver, fnorm)
      integer ierr
      integer*8 fsolver
      double precision fnorm

      call HYPRE_StructHybridGetFinalRelat(fsolver, fnorm, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structhybridgetfinalrela: err = ', ierr
      endif

      return
      end





!***********************************************************************
!             HYPRE_StructInterpreter routines
!***********************************************************************

!***********************************************************************
!     fhypre_structvectorsetrandomvalues
!***********************************************************************
      subroutine fhypre_structvectorsetrandomvalu(fvector, fseed)
      integer ierr
      integer fseed
      integer*8 fvector

      call hypre_StructVectorSetRandomValu(fvector, fseed, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structvectorsetrandomvalues: err = ', ierr
      endif

      return
      end


!***********************************************************************
!     fhypre_structsetrandomvalues
!***********************************************************************
      subroutine fhypre_structsetrandomvalues(fvector, fseed)
      integer ierr
      integer fseed
      integer*8 fvector

      call hypre_StructSetRandomValues(fvector, fseed, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structsetrandomvalues: err = ', ierr
      endif

      return
      end


!***********************************************************************
!     fhypre_structsetupinterpreter
!***********************************************************************
      subroutine fhypre_structsetupinterpreter(fi)
      integer ierr
      integer*8 fi

      call HYPRE_StructSetupInterpreter(fi, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structsetupinterpreter: err = ', ierr
      endif

      return
      end


!***********************************************************************
!     fhypre_structsetupmatvec
!***********************************************************************
      subroutine fhypre_structsetupmatvec(fmv)
      integer ierr
      integer*8 fmv

      call HYPRE_StructSetupMatvec(fmv, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structsetupmatvec: err = ', ierr
      endif

      return
      end




!***********************************************************************
!             HYPRE_StructJacobi routines
!***********************************************************************

!***********************************************************************
!     fhypre_structjacobicreate
!***********************************************************************
      subroutine fhypre_structjacobicreate(fcomm, fsolver)
      integer ierr
      integer fcomm
      integer*8 fsolver

      call HYPRE_StructJacobiCreate(fcomm, fsolver, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structjacobicreate: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structjacobidestroy
!***********************************************************************
      subroutine fhypre_structjacobidestroy(fsolver)
      integer ierr
      integer*8 fsolver

      call HYPRE_StructJacobiDestroy(fsolver, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structjacobidestroy: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structjacobisetup
!***********************************************************************
      subroutine fhypre_structjacobisetup(fsolver, fA, fb, fx)
      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call HYPRE_StructJacobiSetup(fsolver, fA, fb, fx, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structjacobisetup: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structjacobisolve
!***********************************************************************
      subroutine fhypre_structjacobisolve(fsolver, fA, fb, fx)
      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call HYPRE_StructJacobiSolve(fsolver, fA, fb, fx, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structjacobisolve: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structjacobisettol
!***********************************************************************
      subroutine fhypre_structjacobisettol(fsolver, ftol)
      integer ierr
      integer*8 fsolver
      double precision ftol

      call HYPRE_StructJacobiSetTol(fsolver, ftol, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structjacobisettol: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structjacobigettol
!***********************************************************************
      subroutine fhypre_structjacobigettol(fsolver, ftol)
      integer ierr
      integer*8 fsolver
      double precision ftol

      call HYPRE_StructJacobiGetTol(fsolver, ftol, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structjacobigettol: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structjacobisetmaxiter
!***********************************************************************
      subroutine fhypre_structjacobisetmaxiter(fsolver, fmaxiter)
      integer ierr
      integer fmaxiter
      integer*8 fsolver

      call HYPRE_StructJacobiSetMaxIter(fsolver, fmaxiter, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structjacobisetmaxiter: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structjacobigetmaxiter
!***********************************************************************
      subroutine fhypre_structjacobigetmaxiter(fsolver, fmaxiter)
      integer ierr
      integer fmaxiter
      integer*8 fsolver

      call HYPRE_StructJacobiGetMaxIter(fsolver, fmaxiter, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structjacobigetmaxiter: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structjacobisetzeroguess
!***********************************************************************
      subroutine fhypre_structjacobisetzeroguess(fsolver)
      integer ierr
      integer*8 fsolver

      call HYPRE_StructJacobiSetZeroGuess(fsolver, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structjacobisetzeroguess: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structjacobigetzeroguess
!***********************************************************************
      subroutine fhypre_structjacobigetzeroguess(fsolver)
      integer ierr
      integer*8 fsolver

      call HYPRE_StructJacobiGetZeroGuess(fsolver, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structjacobigetzeroguess: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structjacobisetnonzeroguess
!***********************************************************************
      subroutine fhypre_structjacobisetnonzerogu(fsolver)
      integer ierr
      integer*8 fsolver

      call HYPRE_StructJacobiSetNonZeroGue(fsolver, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structjacobisetnonzerogu: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structjacobigetnumiterations
!***********************************************************************
      subroutine fhypre_structjacobigetnumiterat(fsolver, fnumiters)
      integer ierr
      integer fnumiters
      integer*8 fsolver

      call HYPRE_StructJacobiGetNumIterati(fsolver, fnumiters, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structjacobigetnumiterat: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structjacobigetfinalrelativeresidualnorm
!***********************************************************************
      subroutine fhypre_structjacobigetfinalrela(fsolver, fnorm)
      integer ierr
      integer*8 fsolver
      double precision fnorm

      call HYPRE_StructJacobiGetFinalRelat(fsolver, fnorm, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structjacobigetfinalrela: err = ', ierr
      endif

      return
      end





!***********************************************************************
!             HYPRE_StructPCG routines
!***********************************************************************

!***********************************************************************
!     fhypre_structpcgcreate
!***********************************************************************
      subroutine fhypre_structpcgcreate(fcomm, fsolver)
      integer ierr
      integer fcomm
      integer*8 fsolver

      call HYPRE_StructPCGCreate(fcomm, fsolver, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structpcgcreate: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structpcgdestroy
!***********************************************************************
      subroutine fhypre_structpcgdestroy(fsolver)
      integer ierr
      integer*8 fsolver

      call HYPRE_StructPCGDestroy(fsolver, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structpcgdestroy: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structpcgsetup
!***********************************************************************
      subroutine fhypre_structpcgsetup(fsolver, fA, fb, fx)
      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call HYPRE_StructPCGSetup(fsolver, fA, fb, fx, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structpcgsetup: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structpcgsolve
!***********************************************************************
      subroutine fhypre_structpcgsolve(fsolver, fA, fb, fx)
      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call HYPRE_StructPCGSolve(fsolver, fA, fb, fx, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structpcgsolve: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structpcgsettol
!***********************************************************************
      subroutine fhypre_structpcgsettol(fsolver, ftol)
      integer ierr
      integer*8 fsolver
      double precision ftol

      call HYPRE_StructPCGSetTol(fsolver, ftol, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structpcgsettol: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structpcgsetmaxiter
!***********************************************************************
      subroutine fhypre_structpcgsetmaxiter(fsolver, fmaxiter)
      integer ierr
      integer fmaxiter
      integer*8 fsolver

      call HYPRE_StructPCGSetMaxIter(fsolver, fmaxiter, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structpcgsetmaxiter: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structpcgsettwonorm
!***********************************************************************
      subroutine fhypre_structpcgsettwonorm(fsolver, ftwonorm)
      integer ierr
      integer ftwonorm
      integer*8 fsolver

      call HYPRE_StructPCGSetTwoNorm(fsolver, ftwonorm, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structpcgsettwonorm: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structpcgsetrelchange
!***********************************************************************
      subroutine fhypre_structpcgsetrelchange(fsolver, frelchng)
      integer ierr
      integer frelchng
      integer*8 fsolver

      call HYPRE_StructPCGSetRelChange(fsolver, frelchng, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structpcgsetrelchange: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structpcgsetprecond
!***********************************************************************
      subroutine fhypre_structpcgsetprecond(fsolver, fprecond_id, 
     1                                      fprecond)
      integer ierr
      integer fprecond_id
      integer*8 fsolver
      integer*8 fprecond

      call HYPRE_StructPCGSetPrecond(fsolver, fprecond_id, fprecond,
     1                               ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structpcgsetprecond: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structpcgsetlogging
!***********************************************************************
      subroutine fhypre_structpcgsetlogging(fsolver, flogging) 
      integer ierr
      integer flogging
      integer*8 fsolver

      call HYPRE_StructPCGSetLogging(fsolver, flogging, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structpcgsetlogging: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structpcgsetprintlevel
!***********************************************************************
      subroutine fhypre_structpcgsetprintlevel(fsolver, fprntlvl) 
      integer ierr
      integer fprntlvl
      integer*8 fsolver

      call HYPRE_StructPCGSetPrintLevel(fsolver, fprntlvl, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structpcgsetprintlevel: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structpcggetnumiterations
!***********************************************************************
      subroutine fhypre_structpcggetnumiteration(fsolver, fnumiters)
      integer ierr
      integer fnumiters
      integer*8 fsolver

      call HYPRE_StructPCGGetNumIterations(fsolver, fnumiters, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structpcggetnumiteration: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structpcggetfinalrelativeresidualnorm
!***********************************************************************
      subroutine fhypre_structpcggetfinalrelativ(fsolver, fnorm)
      integer ierr
      integer*8 fsolver
      double precision fnorm

      call HYPRE_StructPCGGetFinalRelative(fsolver, fnorm, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structjacobigetfinalrelativ: err = ', ierr
      endif

      return
      end



!***********************************************************************
!     fhypre_structdiagscalesetup
!***********************************************************************
      subroutine fhypre_structdiagscalesetup(fsolver, fA, fb, fx)
      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call HYPRE_StructDiagScaleSetup(fsolver, fA, fb, fx, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structdiagscalesetup: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structdiagscale
!***********************************************************************
      subroutine fhypre_structdiagscale(fsolver, fA, fb, fx)
      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call HYPRE_StructDiagScale(fsolver, fA, fb, fx, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structdiagscale: err = ', ierr
      endif

      return
      end





!***********************************************************************
!             HYPRE_StructPFMG routines
!***********************************************************************

!***********************************************************************
!     fhypre_structpfmgcreate
!***********************************************************************
      subroutine fhypre_structpfmgcreate(fcomm, fsolver)
      integer ierr
      integer fcomm
      integer*8 fsolver

      call HYPRE_StructPFMGCreate(fcomm, fsolver, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structpfmgcreate: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structpfmgdestroy
!***********************************************************************
      subroutine fhypre_structpfmgdestroy(fsolver)
      integer ierr
      integer*8 fsolver

      call HYPRE_StructPFMGDestroy(fsolver, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structpfmgdestroy: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structpfmgsetup
!***********************************************************************
      subroutine fhypre_structpfmgsetup(fsolver, fA, fb, fx)
      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call HYPRE_StructPFMGSetup(fsolver, fA, fb, fx, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structpfmgsetup: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structpfmgsolve
!***********************************************************************
      subroutine fhypre_structpfmgsolve(fsolver, fA, fb, fx)
      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call HYPRE_StructPFMGSolve(fsolver, fA, fb, fx, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structpfmgsolve: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structpfmgsettol
!***********************************************************************
      subroutine fhypre_structpfmgsettol(fsolver, ftol)
      integer ierr
      integer*8 fsolver
      double precision ftol

      call HYPRE_StructPFMGSetTol(fsolver, ftol, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structpfmgsettol: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structpfmggettol
!***********************************************************************
      subroutine fhypre_structpfmggettol(fsolver, ftol)
      integer ierr
      integer*8 fsolver
      double precision ftol

      call HYPRE_StructPFMGGetTol(fsolver, ftol, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structpfmggettol: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structpfmgsetmaxiter
!***********************************************************************
      subroutine fhypre_structpfmgsetmaxiter(fsolver, fmaxiter)
      integer ierr
      integer fmaxiter
      integer*8 fsolver

      call HYPRE_StructPFMGSetMaxIter(fsolver, fmaxiter, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structpfmgsetmaxiter: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structpfmggetmaxiter
!***********************************************************************
      subroutine fhypre_structpfmggetmaxiter(fsolver, fmaxiter)
      integer ierr
      integer fmaxiter
      integer*8 fsolver

      call HYPRE_StructPFMGGetMaxIter(fsolver, fmaxiter, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structpfmggetmaxiter: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structpfmgsetmaxlevels
!***********************************************************************
      subroutine fhypre_structpfmgsetmaxlevels(fsolver, fmaxlevels)
      integer ierr
      integer fmaxlevels
      integer*8 fsolver

      call HYPRE_StructPFMGSetMaxLevels(fsolver, fmaxlevels, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structpfmgsetmaxlevels: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structpfmggetmaxlevels
!***********************************************************************
      subroutine fhypre_structpfmggetmaxlevels(fsolver, fmaxlevels)
      integer ierr
      integer fmaxlevels
      integer*8 fsolver

      call HYPRE_StructPFMGGetMaxLevels(fsolver, fmaxlevels, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structpfmggetmaxlevels: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structpfmgsetrelchange
!***********************************************************************
      subroutine fhypre_structpfmgsetrelchange(fsolver, frelchange)
      integer ierr
      integer frelchange
      integer*8 fsolver

      call HYPRE_StructPFMGSetRelChange(fsolver, frelchange, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structpfmgsetrelchange: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structpfmggetrelchange
!***********************************************************************
      subroutine fhypre_structpfmggetrelchange(fsolver, frelchange)
      integer ierr
      integer frelchange
      integer*8 fsolver

      call HYPRE_StructPFMGGetRelChange(fsolver, frelchange, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structpfmggetrelchange: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structpfmgsetzeroguess
!***********************************************************************
      subroutine fhypre_structpfmgsetzeroguess(fsolver)
      integer ierr
      integer*8 fsolver

      call HYPRE_StructPFMGSetZeroGuess(fsolver, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structpfmgsetzeroguess: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structpfmggetzeroguess
!***********************************************************************
      subroutine fhypre_structpfmggetzeroguess(fsolver)
      integer ierr
      integer*8 fsolver

      call HYPRE_StructPFMGGetZeroGuess(fsolver, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structpfmggetzeroguess: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structpfmgsetnonzeroguess
!***********************************************************************
      subroutine fhypre_structpfmgsetnonzerogues(fsolver)
      integer ierr
      integer*8 fsolver

      call HYPRE_StructPFMGSetNonZeroGuess(fsolver, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structpfmgsetnonzerogues: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structpfmggetnumiterations
!***********************************************************************
      subroutine fhypre_structpfmggetnumiteratio(fsolver, fnumiters)
      integer ierr
      integer fnumiters
      integer*8 fsolver

      call HYPRE_StructPFMGGetNumIteration(fsolver, fnumiters, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structpfmggetnumiteratio: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structpfmggetfinalrelativeresidualnorm
!***********************************************************************
      subroutine fhypre_structpfmggetfinalrelati(fsolver, fnorm)
      integer ierr
      integer*8 fsolver
      double precision fnorm

      call HYPRE_StructPFMGGetFinalRelativ(fsolver, fnorm, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structpfmggetfinalrelati: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structpfmgsetskiprelax
!***********************************************************************
      subroutine fhypre_structpfmgsetskiprelax(fsolver, fskiprelax)
      integer ierr
      integer fskiprelax
      integer*8 fsolver

      call HYPRE_StructPFMGSetSkipRelax(fsolver, fskiprelax, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structpfmgsetskiprelax: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structpfmggetskiprelax
!***********************************************************************
      subroutine fhypre_structpfmggetskiprelax(fsolver, fskiprelax)
      integer ierr
      integer fskiprelax
      integer*8 fsolver

      call HYPRE_StructPFMGGetSkipRelax(fsolver, fskiprelax, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structpfmggetskiprelax: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structpfmgsetrelaxtype
!***********************************************************************
      subroutine fhypre_structpfmgsetrelaxtype(fsolver, frelaxtype)
      integer ierr
      integer frelaxtype
      integer*8 fsolver

      call HYPRE_StructPFMGSetRelaxType(fsolver, frelaxtype, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structpfmgsetrelaxtype: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structpfmggetrelaxtype
!***********************************************************************
      subroutine fhypre_structpfmggetrelaxtype(fsolver, frelaxtype)
      integer ierr
      integer frelaxtype
      integer*8 fsolver

      call HYPRE_StructPFMGGetRelaxType(fsolver, frelaxtype, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structpfmggetrelaxtype: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structpfmgsetraptype
!***********************************************************************
      subroutine fhypre_structpfmgsetraptype(fsolver, fraptype)
      integer ierr
      integer fraptype
      integer*8 fsolver

      call HYPRE_StructPFMGSetRAPType(fsolver, fraptype, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structpfmgsetraptype: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structpfmggetraptype
!***********************************************************************
      subroutine fhypre_structpfmggetraptype(fsolver, fraptype)
      integer ierr
      integer fraptype
      integer*8 fsolver

      call HYPRE_StructPFMGGetRAPType(fsolver, fraptype, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structpfmggetraptype: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structpfmgsetnumprerelax
!***********************************************************************
      subroutine fhypre_structpfmgsetnumprerelax(fsolver,
     1                                             fnumprerelax)
      integer ierr
      integer fnumprerelax
      integer*8 fsolver

      call HYPRE_StructPFMGSetNumPreRelax(fsolver, fnumprerelax, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structpfmgsetnumprerelax: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structpfmggetnumprerelax
!***********************************************************************
      subroutine fhypre_structpfmggetnumprerelax(fsolver,
     1                                             fnumprerelax)
      integer ierr
      integer fnumprerelax
      integer*8 fsolver

      call HYPRE_StructPFMGGetNumPreRelax(fsolver, fnumprerelax, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structpfmggetnumprerelax: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structpfmgsetnumpostrelax
!***********************************************************************
      subroutine fhypre_structpfmgsetnumpostrela(fsolver,
     1                                             fnumpostrelax)
      integer ierr
      integer fnumpostrelax
      integer*8 fsolver

      call HYPRE_StructPFMGSetNumPostRelax(fsolver, fnumpostrelax, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structpfmgsetnumpostrela: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structpfmggetnumpostrelax
!***********************************************************************
      subroutine fhypre_structpfmggetnumpostrela(fsolver,
     1                                             fnumpostrelax)
      integer ierr
      integer fnumpostrelax
      integer*8 fsolver

      call HYPRE_StructPFMGGetNumPostRelax(fsolver, fnumpostrelax, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structpfmggetnumpostrela: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structpfmgsetdxyz
!***********************************************************************
      subroutine fhypre_structpfmgsetdxyz(fsolver, fdxyz)
      integer ierr
      integer*8 fsolver
      double precision fdxyz

      call HYPRE_StructPFMGSetDxyz(fsolver, fdxyz, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structpfmgsetdxyz: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structpfmgsetlogging
!***********************************************************************
      subroutine fhypre_structpfmgsetlogging(fsolver, flogging)
      integer ierr
      integer flogging
      integer*8 fsolver

      call HYPRE_StructPFMGSetLogging(fsolver, flogging, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structpfmgsetlogging: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structpfmggetlogging
!***********************************************************************
      subroutine fhypre_structpfmggetlogging(fsolver, flogging)
      integer ierr
      integer flogging
      integer*8 fsolver

      call HYPRE_StructPFMGGetLogging(fsolver, flogging, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structpfmggetlogging: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structpfmgsetprintlevel
!***********************************************************************
      subroutine fhypre_structpfmgsetprintlevel(fsolver, fprintlevel)
      integer ierr
      integer fprintlevel
      integer*8 fsolver

      call HYPRE_StructPFMGSetPrintLevel(fsolver, fprintlevel, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structpfmgsetprintlevel: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structpfmggetprintlevel
!***********************************************************************
      subroutine fhypre_structpfmggetprintlevel(fsolver, fprintlevel)
      integer ierr
      integer fprintlevel
      integer*8 fsolver

      call HYPRE_StructPFMGGetPrintLevel(fsolver, fprintlevel, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structpfmggetprintlevel: err = ', ierr
      endif

      return
      end





!***********************************************************************
!             HYPRE_StructSMG routines
!***********************************************************************

!***********************************************************************
!     fhypre_structsmgcreate
!***********************************************************************
      subroutine fhypre_structsmgcreate(fcomm, fsolver)
      integer ierr
      integer fcomm
      integer*8 fsolver

      call HYPRE_StructSMGCreate(fcomm, fsolver, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structsmgcreate: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structsmgdestroy
!***********************************************************************
      subroutine fhypre_structsmgdestroy(fsolver)
      integer ierr
      integer*8 fsolver

      call HYPRE_StructSMGDestroy(fsolver, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structsmgdestroy: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structsmgsetup
!***********************************************************************
      subroutine fhypre_structsmgsetup(fsolver, fA, fb, fx)
      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call HYPRE_StructSMGSetup(fsolver, fA, fb, fx, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structsmgsetup: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structsmgsolve
!***********************************************************************
      subroutine fhypre_structsmgsolve(fsolver, fA, fb, fx)
      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call HYPRE_StructSMGSolve(fsolver, fA, fb, fx, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structsmgsolve: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structsmgsetmemoryuse
!***********************************************************************
      subroutine fhypre_structsmgsetmemoryuse(fsolver, fmemuse)
      integer ierr
      integer fmemuse
      integer*8 fsolver

      call HYPRE_StructSMGSetMemoryUse(fsolver, fmemuse, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structsmgsetmemoryuse: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structsmggetmemoryuse
!***********************************************************************
      subroutine fhypre_structsmggetmemoryuse(fsolver, fmemuse)
      integer ierr
      integer fmemuse
      integer*8 fsolver

      call HYPRE_StructSMGGetMemoryUse(fsolver, fmemuse, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structsmggetmemoryuse: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structsmgsettol
!***********************************************************************
      subroutine fhypre_structsmgsettol(fsolver, ftol)
      integer ierr
      integer*8 fsolver
      double precision ftol

      call HYPRE_StructSMGSetTol(fsolver, ftol, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structsmgsettol: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structsmggettol
!***********************************************************************
      subroutine fhypre_structsmggettol(fsolver, ftol)
      integer ierr
      integer*8 fsolver
      double precision ftol

      call HYPRE_StructSMGGetTol(fsolver, ftol, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structsmggettol: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structsmgsetmaxiter
!***********************************************************************
      subroutine fhypre_structsmgsetmaxiter(fsolver, fmaxiter)
      integer ierr
      integer fmaxiter
      integer*8 fsolver

      call HYPRE_StructSMGSetMaxIter(fsolver, fmaxiter, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structsmgsetmaxiter: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structsmggetmaxiter
!***********************************************************************
      subroutine fhypre_structsmggetmaxiter(fsolver, fmaxiter)
      integer ierr
      integer fmaxiter
      integer*8 fsolver

      call HYPRE_StructSMGGetMaxIter(fsolver, fmaxiter, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structsmggetmaxiter: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structsmgsetrelchange
!***********************************************************************
      subroutine fhypre_structsmgsetrelchange(fsolver, frelchange)
      integer ierr
      integer frelchange
      integer*8 fsolver

      call HYPRE_StructSMGSetRelChange(fsolver, frelchange, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structsmgsetrelchange: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structsmggetrelchange
!***********************************************************************
      subroutine fhypre_structsmggetrelchange(fsolver, frelchange)
      integer ierr
      integer frelchange
      integer*8 fsolver

      call HYPRE_StructSMGGetRelChange(fsolver, frelchange, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structsmggetrelchange: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structsmgsetzeroguess
!***********************************************************************
      subroutine fhypre_structsmgsetzeroguess(fsolver)
      integer ierr
      integer*8 fsolver

      call HYPRE_StructSMGSetZeroGuess(fsolver, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structsmgsetzeroguess: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structsmggetzeroguess
!***********************************************************************
      subroutine fhypre_structsmggetzeroguess(fsolver)
      integer ierr
      integer*8 fsolver

      call HYPRE_StructSMGGetZeroGuess(fsolver, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structsmggetzeroguess: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structsmgsetnonzeroguess
!***********************************************************************
      subroutine fhypre_structsmgsetnonzeroguess(fsolver)
      integer ierr
      integer*8 fsolver

      call HYPRE_StructSMGSetNonZeroGuess(fsolver, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structsmgsetnonzeroguess: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structsmggetnumiterations
!***********************************************************************
      subroutine fhypre_structsmggetnumiteration(fsolver, fnumiters)
      integer ierr
      integer fnumiters
      integer*8 fsolver

      call HYPRE_StructSMGGetNumIterations(fsolver, fnumiters, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structsmggetnumiteration: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structsmggetfinalrelativeresidualnorm
!***********************************************************************
      subroutine fhypre_structsmggetfinalrelativ(fsolver, fnorm)
      integer ierr
      integer*8 fsolver
      double precision fnorm

      call HYPRE_StructSMGGetFinalRelative(fsolver, fnorm, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structsmggetfinalrelativ: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structsmgsetnumprerelax
!***********************************************************************
      subroutine fhypre_structsmgsetnumprerelax(fsolver, fnumprerelax)
      integer ierr
      integer fnumprerelax
      integer*8 fsolver

      call HYPRE_StructSMGSetNumPreRelax(fsolver, fnumprerelax, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structsmgsetnumprerelax: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structsmggetnumprerelax
!***********************************************************************
      subroutine fhypre_structsmggetnumprerelax(fsolver, fnumprerelax)
      integer ierr
      integer fnumprerelax
      integer*8 fsolver

      call HYPRE_StructSMGGetNumPreRelax(fsolver, fnumprerelax, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structsmggetnumprerelax: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structsmgsetnumpostrelax
!***********************************************************************
      subroutine fhypre_structsmgsetnumpostrelax(fsolver, fnumpstrlx)
      integer ierr
      integer fnumpstrlx
      integer*8 fsolver

      call HYPRE_StructSMGSetNumPostRelax(fsolver, fnumpstrlx, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structsmgsetnumpostrelax: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structsmggetnumpostrelax
!***********************************************************************
      subroutine fhypre_structsmggetnumpostrelax(fsolver, fnumpstrlx)
      integer ierr
      integer fnumpstrlx
      integer*8 fsolver

      call HYPRE_StructSMGGetNumPostRelax(fsolver, fnumpstrlx, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structsmggetnumpostrelax: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structsmgsetlogging
!***********************************************************************
      subroutine fhypre_structsmgsetlogging(fsolver, flogging)
      integer ierr
      integer flogging
      integer*8 fsolver

      call HYPRE_StructSMGSetLogging(fsolver, flogging, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structsmgsetlogging: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structsmggetlogging
!***********************************************************************
      subroutine fhypre_structsmggetlogging(fsolver, flogging)
      integer ierr
      integer flogging
      integer*8 fsolver

      call HYPRE_StructSMGGetLogging(fsolver, flogging, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structsmggetlogging: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structsmgsetprintlevel
!***********************************************************************
      subroutine fhypre_structsmgsetprintlevel(fsolver, fprintlevel)
      integer ierr
      integer fprintlevel
      integer*8 fsolver

      call HYPRE_StructSMGSetPrintLevel(fsolver, fprintlevel, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structsmgsetprintlevel: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structsmggetprintlevel
!***********************************************************************
      subroutine fhypre_structsmggetprintlevel(fsolver, fprintlevel)
      integer ierr
      integer fprintlevel
      integer*8 fsolver

      call HYPRE_StructSMGGetPrintLevel(fsolver, fprintlevel, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structsmggetprintlevel: err = ', ierr
      endif

      return
      end





!***********************************************************************
!             HYPRE_StructSparseMSG routines
!***********************************************************************

!***********************************************************************
!     fhypre_structsparsemsgcreate
!***********************************************************************
      subroutine fhypre_structsparsemsgcreate(fcomm, fsolver)
      integer ierr
      integer fcomm
      integer*8 fsolver

      call HYPRE_StructSparseMSGCreate(fcomm, fsolver, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structsparsemsgcreate: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structsparsemsgdestroy
!***********************************************************************
      subroutine fhypre_structsparsemsgdestroy(fsolver)
      integer ierr
      integer*8 fsolver

      call HYPRE_StructSparseMSGDestroy(fsolver, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structsparsemsgdestroy: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structsparsemsgsetup
!***********************************************************************
      subroutine fhypre_structsparsemsgsetup(fsolver, fA, fb, fx)
      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call HYPRE_StructSparseMSGSetup(fsolver, fA, fb, fx, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structsparsemsgsetup: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structsparsemsgsolve
!***********************************************************************
      subroutine fhypre_structsparsemsgsolve(fsolver, fA, fb, fx)
      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call HYPRE_StructSparseMSGSolve(fsolver, fA, fb, fx, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structsparsemsgsolve: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structsparsemsgsetjump
!***********************************************************************
      subroutine fhypre_structsparsemsgsetjump(fsolver, fjump)
      integer ierr
      integer fjump
      integer*8 fsolver

      call HYPRE_StructSparseMSGSetJump(fsolver, fjump, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structsparsemsgsetjump: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structsparsemsgsettol
!***********************************************************************
      subroutine fhypre_structsparsemsgsettol(fsolver, ftol)
      integer ierr
      integer*8 fsolver
      double precision ftol

      call HYPRE_StructSparseMSGSetTol(fsolver, ftol, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structsparsemsgsettol: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structsparsemsgsetmaxiter
!***********************************************************************
      subroutine fhypre_structsparsemsgsetmaxite(fsolver, fmaxiter)
      integer ierr
      integer fmaxiter
      integer*8 fsolver

      call HYPRE_StructSparseMSGSetMaxIter(fsolver, fmaxiter, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structsparsemsgsetmaxite: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structsparsemsgsetrelchange
!***********************************************************************
      subroutine fhypre_structsparsemsgsetrelcha(fsolver, frelchange)
      integer ierr
      integer frelchange
      integer*8 fsolver

      call HYPRE_StructSparseMSGSetRelChan(fsolver, frelchange, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structsparsemsgsetrelcha: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structsparsemsgsetzeroguess
!***********************************************************************
      subroutine fhypre_structsparsemsgsetzerogu(fsolver)
      integer ierr
      integer*8 fsolver

      call HYPRE_StructSparseMSGSetZeroGue(fsolver, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structsparsemsgsetzerogu: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structsparsemsgsetnonzeroguess
!***********************************************************************
      subroutine fhypre_structsparsemsgsetnonzer(fsolver)
      integer ierr
      integer*8 fsolver

      call HYPRE_StructSparseMSGSetNonZero(fsolver, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structsparsemsgsetnonzer: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structsparsemsggetnumiterations
!***********************************************************************
      subroutine fhypre_structsparsemsggetnumite(fsolver, fniters)
      integer ierr
      integer fniters
      integer*8 fsolver

      call HYPRE_StructSparseMSGGetNumIter(fsolver, fniters, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structsparsemsggetnumite: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structsparsemsggetfinalrelativeresidualnorm
!***********************************************************************
      subroutine fhypre_structsparsemsggetfinalr(fsolver, fnorm)
      integer ierr
      integer*8 fsolver
      double precision fnorm

      call HYPRE_StructSparseMSGGetFinalRe(fsolver, fnorm, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structsparsemsggetfinalr: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structsparsemsgsetrelaxtype
!***********************************************************************
      subroutine fhypre_structsparsemsgsetrelaxt(fsolver, frelaxtype)
      integer ierr
      integer frelaxtype
      integer*8 fsolver

      call HYPRE_StructSparseMSGSetRelaxTy(fsolver, frelaxtype, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structsparsemsgsetrelaxt: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structsparsemsgsetnumprerelax
!***********************************************************************
      subroutine fhypre_structsparsemsgsetnumpre(fsolver, fnprelax)
      integer ierr
      integer fnprelax
      integer*8 fsolver

      call HYPRE_StructSparseMSGSetNumPreR(fsolver, fnprelax, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structsparsemsgsetnumpre: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structsparsemsgsetnumpostrelax
!***********************************************************************
      subroutine fhypre_structsparsemsgsetnumpos(fsolver, fnpstrlx)
      integer ierr
      integer fnpstrlx
      integer*8 fsolver

      call HYPRE_StructSparseMSGSetNumPost(fsolver, fnpstrlx, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structsparsemsgsetnumpos: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structsparsemsgsetnumfinerelax
!***********************************************************************
      subroutine fhypre_structsparsemsgsetnumfin(fsolver, fnfine)
      integer ierr
      integer fnfine
      integer*8 fsolver

      call HYPRE_StructSparseMSGSetNumFine(fsolver, fnfine, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structsparsemsgsetnumfin: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structsparsemsgsetlogging
!***********************************************************************
      subroutine fhypre_structsparsemsgsetloggin(fsolver, flogging)
      integer ierr
      integer flogging
      integer*8 fsolver

      call HYPRE_StructSparseMSGSetLogging(fsolver, flogging, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structsparsemsgsetloggin: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fhypre_structsparsemsgsetprintlevel
!***********************************************************************
      subroutine fhypre_structsparsemsgsetprintl(fsolver, fprntlvl)
      integer ierr
      integer fprntlvl
      integer*8 fsolver

      call HYPRE_StructSparseMSGSetPrintLe(fsolver, fprntlvl, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structsparsemsgsetprintl: err = ', ierr
      endif

      return
      end
