cBHEADER**********************************************************************
c Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
c Produced at the Lawrence Livermore National Laboratory.
c This file is part of HYPRE.  See file COPYRIGHT for details.
c
c HYPRE is free software; you can redistribute it and/or modify it under the
c terms of the GNU Lesser General Public License (as published by the Free
c Software Foundation) version 2.1 dated February 1999.
c
c $Revision: 1.4 $
cEHEADER**********************************************************************

c--------------------------------------------------------------------------
c GenerateLaplacian
c--------------------------------------------------------------------------
      subroutine fhypre_generatelaplacian(fcom, fnx, fny, fnz,
     1                                    fcapp, fcapq, fcapr, 
     1                                    fp, fq, fr,
     2                                    fvalue, fmatrix)

      integer ierr
      integer fcomm
      integer fnx, fny, fnz
      integer fcapp, fcapq, fcapr
      integer fp, fq, fr
      double precision fvalue
      integer*8 fmatrix

      call hypre_GenerateLaplacian(fcomm, fnx, fny, fnz, fcapp, fcapq,
     1                       fcapr, fp, fq, fr, fvalue, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_generatelaplacian error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGCreate
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramgcreate(fsolver)

      integer ierr
      integer*8 fsolver

      call HYPRE_BoomerAMGCreate(fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramgcreate error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGDestroy
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramgdestroy(fsolver)

      integer ierr
      integer*8 fsolver

      call HYPRE_BoomerAMGDestroy(fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramgdestroy error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGSetup
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramgsetup(fsolver, fA, fb, fx)

      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call HYPRE_BoomerAMGSetup(fsolver, fA, fb, fx, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramgsetup error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGSolve
c-------------------------------------------------------------------------- 
      subroutine fhypre_boomeramgsolve(fsolver, fA, fb, fx)

      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call HYPRE_BoomerAMGSolve(fsolver, fA, fb, fx, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramgsolve error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGSolveT
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramgsolvet(fsolver, fA, fb, fx)

      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call HYPRE_BoomerAMGSolveT(fsolver, fA, fb, fx, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramgsolvet error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGSetRestriction
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramgsetrestriction(fsolver, frestr_par)

      integer ierr
      integer frestr_par
      integer*8 fsolver

      call HYPRE_BoomerAMGSetRestriction(fsolver, frestr_par, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramgsetrestriction error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGSetMaxLevels
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramgsetmaxlevels(fsolver, fmaxlvl)

      integer ierr
      integer fmaxlvl
      integer*8 fsolver

      call HYPRE_BoomerAMGSetMaxLevels(fsolver, fmaxlvl, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramgsetmaxlevels error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGGetMaxLevels
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramggetmaxlevels(fsolver, fmaxlvl)

      integer ierr
      integer fmaxlvl
      integer*8 fsolver

      call HYPRE_BoomerAMGGetMaxLevels(fsolver, fmaxlvl, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramggetmaxlevels error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGSetStrongThreshold
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramgsetstrongthrshl(fsolver, fstrong)

      integer ierr
      double precision fstrong
      integer*8 fsolver

      call HYPRE_BoomerAMGSetStrongThrshld(fsolver, fstrong, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramgsetstrongthreshold error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGGetStrongThreshold
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramggetstrongthrshl(fsolver, fstrong)

      integer ierr
      double precision fstrong
      integer*8 fsolver

      call HYPRE_BoomerAMGGetStrongThrshld(fsolver, fstrong, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramggetstrongthreshold error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGSetMaxRowSum
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramgsetmaxrowsum(fsolver, fmaxrowsum)

      integer ierr
      double precision fmaxrowsum
      integer*8 fsolver

      call HYPRE_BoomerAMGSetMaxRowSum(fsolver, fmaxrowsum, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramgsetmaxrowsum error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGGetMaxRowSum
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramggetmaxrowsum(fsolver, fmaxrowsum)

      integer ierr
      double precision fmaxrowsum
      integer*8 fsolver

      call HYPRE_BoomerAMGGetMaxRowSum(fsolver, fmaxrowsum, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramggetmaxrowsum error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGSetTruncFactor
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramgsettruncfactor(fsolver, ftrunc_factor)

      integer ierr
      double precision ftrunc_factor
      integer*8 fsolver

      call HYPRE_BoomerAMGSetTruncFactor(fsolver, ftrunc_factor, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramgsettruncfactor error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGGetTruncFactor
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramggettruncfactor(fsolver, ftrunc_factor)

      integer ierr
      double precision ftrunc_factor
      integer*8 fsolver

      call HYPRE_BoomerAMGGetTruncFactor(fsolver, ftrunc_factor, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramggettruncfactor error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGSetSCommPkgSwitch
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramgsetscommpkgswit(fsolver, fcommswtch)

      integer ierr
      integer fcommswtch
      integer*8 fsolver

      call HYPRE_BoomerAMGSetSCommPkgSwitc(fsolver, fcommswtch, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramgsetscommpkgswitch error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGSetInterpType
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramgsetinterptype(fsolver, finterp)

      integer ierr
      integer finterp
      integer*8 fsolver

      call HYPRE_BoomerAMGSetInterpType(fsolver, finterp, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramgsetinterptype error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGSetMinIter
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramgsetminiter(fsolver, fminiter)

      integer ierr
      integer fminiter  
      integer*8 fsolver

      call HYPRE_BoomerAMGSetMinIter(fsolver, fminiter, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramgsetminiter error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGSetMaxIter
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramgsetmaxiter(fsolver, fmaxiter)

      integer ierr
      integer fmaxiter  
      integer*8 fsolver

      call HYPRE_BoomerAMGSetMaxIter(fsolver, fmaxiter, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramgsetmaxiter error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGGetMaxIter
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramggetmaxiter(fsolver, fmaxiter)

      integer ierr
      integer fmaxiter  
      integer*8 fsolver

      call HYPRE_BoomerAMGGetMaxIter(fsolver, fmaxiter, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramggetmaxiter error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGSetCoarsenType
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramgsetcoarsentype(fsolver, fcoarsen)

      integer ierr
      integer fcoarsen
      integer*8 fsolver

      call HYPRE_BoomerAMGSetCoarsenType(fsolver, fcoarsen, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramgsetcoarsentype error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGGetCoarsenType
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramggetcoarsentype(fsolver, fcoarsen)

      integer ierr
      integer fcoarsen
      integer*8 fsolver

      call HYPRE_BoomerAMGGetCoarsenType(fsolver, fcoarsen, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramggetcoarsentype error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGSetMeasureType
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramgsetmeasuretype(fsolver, fmeasure)

      integer ierr
      integer fmeasure
      integer*8 fsolver

      call HYPRE_BoomerAMGSetMeasureType(fsolver, fmeasure, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramgsetmeasuretype error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGGetMeasureType
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramggetmeasuretype(fsolver, fmeasure)

      integer ierr
      integer fmeasure
      integer*8 fsolver

      call HYPRE_BoomerAMGGetMeasureType(fsolver, fmeasure, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramggetmeasuretype error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGSetSetupType
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramgsetsetuptype(fsolver, fsetup)

      integer ierr
      integer fsetup
      integer*8 fsolver

      call HYPRE_BoomerAMGSetSetupType(fsolver, fsetup, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramgsetuptype error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGSetCycleType
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramgsetcycletype(fsolver, fcycle)

      integer ierr
      integer fcycle
      integer*8 fsolver

      call HYPRE_BoomerAMGSetCycleType(fsolver, fcycle, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramgsetcycletype error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGGetCycleType
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramggetcycletype(fsolver, fcycle)

      integer ierr
      integer fcycle
      integer*8 fsolver

      call HYPRE_BoomerAMGGetCycleType(fsolver, fcycle, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramggetcycletype error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGSetTol
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramgsettol(fsolver, ftol)

      integer ierr
      double precision ftol
      integer*8 fsolver

      call HYPRE_BoomerAMGSetTol(fsolver, ftol, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramgsettol error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGGetTol
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramggettol(fsolver, ftol)

      integer ierr
      double precision ftol
      integer*8 fsolver

      call HYPRE_BoomerAMGGetTol(fsolver, ftol, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramggettol error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGSetNumSweeps
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramgsetnumsweeps(fsolver, fnumsweeps)

      integer ierr
      integer fnumsweeps
      integer*8 fsolver

      call HYPRE_BoomerAMGSetNumSweeps(fsolver, fnumsweeps, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramgsetnumsweeps error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGSetCycleNumSweeps
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramgsetcyclenumswee(fsolver, fnumsweeps,
     1                                             fk)

      integer ierr
      integer fnumsweeps
      integer fk
      integer*8 fsolver

      call HYPRE_BoomerAMGSetCycleNumSweeps(fsolver, fnumsweeps, fk,
     1                                      ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramgsetcyclenumsweeps error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGGetCycleNumSweeps
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramggetcyclenumswee(fsolver, fnumsweeps,
     1                                             fk)

      integer ierr
      integer fnumsweeps
      integer fk
      integer*8 fsolver

      call HYPRE_BoomerAMGGetCycleNumSweeps(fsolver, fnumsweeps, fk,
     1                                      ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramggetcyclenumsweeps error: ', ierr
      endif

      return
      end
c--------------------------------------------------------------------------
c HYPRE_BoomerAMGInitGridRelaxation
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramginitgridrelaxat(fnumsweeps, fgridtype,
     1                                            fgridrelax, fcoarsen,
     2                                            frelaxwt, fmaxlvl)

      integer ierr
      integer fcoarsen
      integer fmaxlvl
      integer*8 fnumsweeps
      integer*8 fgridtype
      integer*8 fgridrelax
      integer*8 frelaxwt

      call HYPRE_BoomerAMGInitGridRelaxatn(fnumsweeps, fgridtype, 
     1                                       fgridrelax, fcoarsen, 
     2                                       frelaxwt, fmaxlvl, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramginitgridrelaxation error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGFinalizeGridRelaxation
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramgfingridrelaxatn(fnumsweeps, fgridtype,
     1                                            fgridrelax, frelaxwt)

      integer ierr
      integer*8 fnumsweeps
      integer*8 fgridtype
      integer*8 fgridrelax
      integer*8 frelaxwt

c     hypre_TFree(num_grid_sweeps);
c     hypre_TFree(grid_relax_type);
c     hypre_TFree(grid_relax_points);
c     hypre_TFree(relax_weights);

      ierr = 0

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGSetRelaxType
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramgsetrelaxtype(fsolver, frelaxtype)

      integer ierr
      integer frelaxtype
      integer*8 fsolver

      call HYPRE_BoomerAMGSetRelaxType(fsolver, frelaxtype, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramgsetrelaxtype error: ', ierr
      endif
 
      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGSetCycleRelaxType
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramgsetcyclerelaxty(fsolver, frelaxtype,
     1                                             fk)

      integer ierr
      integer frelaxtype
      integer fk
      integer*8 fsolver

      call HYPRE_BoomerAMGSetCycleRelaxType(fsolver, fk, frelaxtype, 
     1                                      ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramgsetcyclerelaxtype error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGGetCycleRelaxType
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramggetcyclerelaxty(fsolver, frelaxtype,
     1                                             fk)

      integer ierr
      integer frelaxtype
      integer fk
      integer*8 fsolver

      call HYPRE_BoomerAMGGetCycleRelaxType(fsolver, fk, frelaxtype, 
     1                                      ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramggetcyclerelaxtype error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGSetRelaxOrder
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramgsetrelaxorder(fsolver, frlxorder)
     
      integer ierr
      integer frlxorder
      integer*8 fsolver

      call HYPRE_BoomerAMGSetRelaxOrder(fsolver, frlxorder, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramgsetrelaxorder error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGSetRelaxWt
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramgsetrelaxwt(fsolver, frelaxwt)
     
      integer ierr
      integer*8 fsolver
      double precision frelaxwt

      call HYPRE_BoomerAMGSetRelaxWt(fsolver, frelaxwt, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramgsetrelaxwt error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGSetLevelRelaxWt
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramgsetlevelrelaxwt(fsolver, frelaxwt, 
     1                                           flevel)

      integer ierr
      integer flevel
      integer*8 fsolver
      double precision frelaxwt

      call HYPRE_BoomerAMGSetLevelRelaxWt(fsolver, frelaxwt, flevel,
     1                                    ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramgsetlevelrelaxwt error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGSetOuterWt
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramgsetouterwt(fsolver, fouterwt)
     
      integer ierr
      integer*8 fsolver
      double precision fouterwt

      call HYPRE_BoomerAMGSetOuterWt(fsolver, fouterwt, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramgsetouterwt error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGSetLevelOuterWt
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramgsetlevelouterwt(fsolver, fouterwt, 
     1                                           flevel)

      integer ierr
      integer flevel
      integer*8 fsolver
      double precision fouterwt

      call HYPRE_BoomerAMGSetLevelOuterWt(fsolver, fouterwt, flevel,
     1                                    ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramgsetlevelouterwt error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGSetSmoothType
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramgsetsmoothtype(fsolver, fsmooth) 

      integer ierr
      integer fsmooth
      integer*8 fsolver

      call HYPRE_BoomerAMGSetSmoothType(fsolver, fsmooth, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramgsetsmoothtype error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGGetSmoothType
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramggetsmoothtype(fsolver, fsmooth) 

      integer ierr
      integer fsmooth
      integer*8 fsolver

      call HYPRE_BoomerAMGGetSmoothType(fsolver, fsmooth, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramggetsmoothtype error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGSetSmoothNumLvls
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramgsetsmoothnumlvl(fsolver, fsmooth) 

      integer ierr
      integer fsmooth
      integer*8 fsolver

      call HYPRE_BoomerAMGSetSmoothNumLvls(fsolver, fsmooth, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramgsetsmoothnumlvls error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGGetSmoothNumLvls
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramggetsmoothnumlvl(fsolver, fsmooth) 

      integer ierr
      integer fsmooth
      integer*8 fsolver

      call HYPRE_BoomerAMGGetSmoothNumLvls(fsolver, fsmooth, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramggetsmoothnumlvls error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGSetSmoothNumSwps
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramgsetsmoothnumswp(fsolver, fsmooth) 

      integer ierr
      integer fsmooth
      integer*8 fsolver

      call HYPRE_BoomerAMGSetSmoothNumSwps(fsolver, fsmooth, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramgsetsmoothnumswps error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGGetSmoothNumSwps
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramggetsmoothnumswp(fsolver, fsmooth) 

      integer ierr
      integer fsmooth
      integer*8 fsolver

      call HYPRE_BoomerAMGGetSmoothNumSwps(fsolver, fsmooth, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramggetsmoothnumswps error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGSetLogging
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramgsetlogging(fsolver, flogging)

      integer ierr
      integer flogging
      integer*8 fsolver

      call HYPRE_BoomerAMGSetLogging(fsolver, flogging, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramgsetlogging error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGGetLogging
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramggetlogging(fsolver, flogging)

      integer ierr
      integer flogging
      integer*8 fsolver

      call HYPRE_BoomerAMGGetLogging(fsolver, flogging, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramggetlogging error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGSetPrintLevel
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramgsetprintlevel(fsolver, fprintlevel)

      integer ierr
      integer fprintlevel
      integer*8 fsolver

      call HYPRE_BoomerAMGSetPrintLevel(fsolver, fprintlevel, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramgsetprintlevel error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGGetPrintLevel
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramggetprintlevel(fsolver, fprintlevel)

      integer ierr
      integer fprintlevel
      integer*8 fsolver

      call HYPRE_BoomerAMGGetPrintLevel(fsolver, fprintlevel, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramggetprintlevel error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGSetPrintFileName
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramgsetprintfilenam(fsolver, fname)

      integer ierr
      integer*8 fsolver
      character*(*) fname

      call HYPRE_BoomerAMGSetPrintFileName(fsolver, fname, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramgsetprintfilename error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGSetDebugFlag
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramgsetdebugflag(fsolver, fdebug)

      integer ierr
      integer fdebug
      integer*8 fsolver

      call HYPRE_BoomerAMGSetDebugFlag(fsolver, fdebug, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramgsetdebugflag error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGGetDebugFlag
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramggetdebugflag(fsolver, fdebug)

      integer ierr
      integer fdebug
      integer*8 fsolver

      call HYPRE_BoomerAMGGetDebugFlag(fsolver, fdebug, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramggetdebugflag error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGGetNumIterations
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramggetnumiteration(fsolver, fnumiter)

      integer ierr
      integer fnumiter
      integer*8 fsolver

      call HYPRE_BoomerAMGGetNumIterations(fsolver, fnumiter, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramggetnumiterations error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGGetCumNumIterations
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramggetcumnumiterat(fsolver, fnumiter)

      integer ierr
      integer fnumiter
      integer*8 fsolver

      call HYPRE_BoomerAMGGetNumIterations(fsolver, fnumiter, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramggetnumiterations error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGGetResidual
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramggetresidual(fsolver, fresid)

      integer ierr
      integer*8 fsolver
      double precision fresid

      call HYPRE_BoomerAMGGetResidual(fsolver, fresid, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramggetresidual error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGGetFinalRelativeResidual
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramggetfinalreltvre(fsolver, frelresid)

      integer ierr
      integer*8 fsolver
      double precision frelresid

      call HYPRE_BoomerAMGGetFinalReltvRes(fsolver, frelresid, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramggetfinalrelativeres error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGSetVariant
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramgsetvariant(fsolver, fvariant)

      integer ierr
      integer fvariant
      integer*8 fsolver

      call HYPRE_BoomerAMGSetVariant(fsolver, fvariant, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramgsetvariant error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGGetVariant
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramggetvariant(fsolver, fvariant)

      integer ierr
      integer fvariant
      integer*8 fsolver

      call HYPRE_BoomerAMGGetVariant(fsolver, fvariant, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramggetvariant error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGSetOverlap
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramgsetoverlap(fsolver, foverlap)

      integer ierr
      integer foverlap
      integer*8 fsolver

      call HYPRE_BoomerAMGSetOverlap(fsolver, foverlap, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramgsetoverlap error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGGetOverlap
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramggetoverlap(fsolver, foverlap)

      integer ierr
      integer foverlap
      integer*8 fsolver

      call HYPRE_BoomerAMGGetOverlap(fsolver, foverlap, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramggetoverlap error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGSetDomainType
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramgsetdomaintype(fsolver, fdomain)

      integer ierr
      integer fdomain
      integer*8 fsolver

      call HYPRE_BoomerAMGSetDomainType(fsolver, fdomain, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramgsetdomaintype error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGGetDomainType
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramggetdomaintype(fsolver, fdomain)

      integer ierr
      integer fdomain
      integer*8 fsolver

      call HYPRE_BoomerAMGGetDomainType(fsolver, fdomain, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramggetdomaintype error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGSetSchwarzRlxWt
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramgsetschwarzrlxwt(fsolver, fschwarz)

      integer ierr
      integer fschwarz
      integer*8 fsolver

      call HYPRE_BoomerAMGSetSchwarzRlxWt(fsolver, fschwarz, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramgsetschwarzrlxwt error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGGetSchwarzRlxWt
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramggetschwarzrlxwt(fsolver, fschwarz)

      integer ierr
      integer fschwarz
      integer*8 fsolver

      call HYPRE_BoomerAMGGetSchwarzRlxWt(fsolver, fschwarz, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramggetschwarzrlxwt error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGSetSym
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramgsetsym(fsolver, fsym)

      integer ierr
      integer fsym
      integer*8 fsolver

      call HYPRE_BoomerAMGSetSym(fsolver, fsym, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramgsetsym error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGSetLevel
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramgsetlevel(fsolver, flevel)

      integer ierr
      integer flevel
      integer*8 fsolver

      call HYPRE_BoomerAMGSetLevel(fsolver, flevel, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramgsetlevel error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGSetFilter
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramgsetfilter(fsolver, ffilter)

      integer ierr
      integer ffilter
      integer*8 fsolver

      call HYPRE_BoomerAMGSetFilter(fsolver, ffilter, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramgsetfilter error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGSetDropTol
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramgsetdroptol(fsolver, fdroptol)

      integer ierr
      integer fdroptol
      integer*8 fsolver

      call HYPRE_BoomerAMGSetDropTol(fsolver, fdroptol, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramgsetdroptol error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGSetMaxNzPerRow
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramgsetmaxnzperrow(fsolver, fmaxnzperrow)

      integer ierr
      integer fmaxnzperrow
      integer*8 fsolver

      call HYPRE_BoomerAMGSetMaxNzPerRow(fsolver, fmaxnzperrow, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramgsetmaxnzperrow error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGSetEuclidFile
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramgseteuclidfile(fsolver, ffile)

      integer ierr
      integer*8 fsolver
      character*(*) ffile

      call HYPRE_BoomerAMGSetEuclidFile(fsolver, ffile, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramgseteuclidfile error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGSetNumFunctions
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramgsetnumfunctions(fsolver, fnfncs)

      integer ierr
      integer fnfncs
      integer*8 fsolver

      call HYPRE_BoomerAMGSetNumFunctions(fsolver, fnfncs, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramgsetnumfunctions error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGGetNumFunctions
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramggetnumfunctions(fsolver, fnfncs)

      integer ierr
      integer fnfncs
      integer*8 fsolver

      call HYPRE_BoomerAMGGetNumFunctions(fsolver, fnfncs, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramggetnumfunctions error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGSetNodal
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramgsetnodal(fsolver, fnodal)

      integer ierr
      integer fnodal
      integer*8 fsolver

      call HYPRE_BoomerAMGSetNodal(fsolver, fnodal, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramgsetnodal error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGSetDofFunc
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramgsetdoffunc(fsolver, fdoffunc)

      integer ierr
      integer fdoffunc
      integer*8 fsolver

      call HYPRE_BoomerAMGSetDofFunc(fsolver, fdoffunc, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramgsetdoffunc error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGSetNumPaths
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramgsetnumpaths(fsolver, fnumpaths)

      integer ierr
      integer fnumpaths
      integer*8 fsolver

      call HYPRE_BoomerAMGSetNumPaths(fsolver, fnumpaths, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramgsetnumpaths error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGSetAggNumLevels
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramgsetaggnumlevels(fsolver, fagglvl)

      integer ierr
      integer fagglvl
      integer*8 fsolver

      call HYPRE_BoomerAMGSetAggNumLevels(fsolver, fagglvl, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramgsetaggnumlevels error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGSetGSMG
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramgsetgsmg(fsolver, fgsmg)

      integer ierr
      integer fgsmg
      integer*8 fsolver

      call HYPRE_BoomerAMGSetGSMG(fsolver, fgsmg, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramgsetgsmg error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGSetNumSamples
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramgsetnumsamples(fsolver, fsamples)

      integer ierr
      integer fsamples
      integer*8 fsolver

      call HYPRE_BoomerAMGSetNumSamples(fsolver, fsamples, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramgsetnumsamples error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_ParCSRBiCGSTABCreate
c--------------------------------------------------------------------------
      subroutine fhypre_parcsrbicgstabcreate(fcomm, fsolver)

      integer fcomm
      integer ierr
      integer*8 fsolver

      call HYPRE_ParCSRBiCGSTABCreate(fcomm, fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrbicgstabcreate error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_ParCSRBiCGSTABDestroy
c-------------------------------------------------------------------------- 
      subroutine fhypre_parcsrbicgstabdestroy(fsolver)

      integer ierr
      integer*8 fsolver

      call HYPRE_ParCSRBiCGSTABDestroy(fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrbicgstabdestroy error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_ParCSRBiCGSTABSetup
c------------------------------------------------------------------------- 
      subroutine fhypre_parcsrbicgstabsetup(fsolver, fA, fb, fx)

      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call HYPRE_ParCSRBiCGSTABSetup(fsolver, fA, fb, fx, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrbicgstabsetup error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_ParCSRBiCGSTABSolve
c-------------------------------------------------------------------------- 
      subroutine fhypre_parcsrbicgstabsolve(fsolver, fA, fb, fx)
     
      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call HYPRE_ParCSRBiCGSTABSolve(fsolver, fA, fb, fx)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrbicgstabsolve error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRBiCGSTABSetTol
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrbicgstabsettol(fsolver, ftol)

      integer ierr
      integer*8 fsolver
      double precision ftol

      call HYPRE_ParCSRBiCGSTABSetTol(fsolver, ftol, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrbicgstabsettol error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRBiCGSTABSetMinIter
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrbicgstabsetminiter(fsolver, fminiter)

      integer ierr
      integer fminiter
      integer*8 fsolver

      call HYPRE_ParCSRBiCGSTABSetMinIter(fsolver, fminiter, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrparcsrbicgstabsetminiter error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRBiCGSTABSetMaxIter
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrbicgstabsetmaxiter(fsolver, fmaxiter)

      integer ierr
      integer fmaxiter
      integer*8 fsolver

      call HYPRE_ParCSRBiCGSTABSetMaxIter(fsolver, fmaxiter, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrbicgstabsetmaxiter error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRBiCGSTABSetStopCrit
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrbicgstabsetstopcrit(fsolver, fstopcrit)

      integer ierr
      integer fstopcrit
      integer*8 fsolver

      call HYPRE_ParCSRBiCGSTABSetStopCrit(fsolver, fstopcrit, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrbicgstabsetstopcrit error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRBiCGSTABSetPrecond
c--------------------------------------------------------------------------
      subroutine fhypre_parcsrbicgstabsetprecond(fsolver, fprecond_id, 
     1                                           fprecond)       

      integer ierr
      integer fprecond_id
      integer*8 fsolver
      integer*8 fprecond

      call HYPRE_ParCSRBiCGSTABSetPrecond(fsolver, fprecond_id, 
     1                                    fprecond, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrbicgstabsetprecond error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRBiCGSTABGetPrecond
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrbicgstabgetprecond(fsolver, fprecond)
      
      integer ierr
      integer*8 fsolver
      integer*8 fprecond

      call HYPRE_ParCSRBiCGSTABGetPrecond(fsolver, fprecond)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrbicgstabgetprecond error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRBiCGSTABSetLogging
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrbicgstabsetlogging(fsolver, flog)

      integer ierr
      integer flog
      integer*8 fsolver

      call HYPRE_ParCSRBiCGSTABSetLogging(fsolver, flog, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrbicgstabsetlogging error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRBiCGSTABSetPrintLevel
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrbicgstabsetprintle(fsolver, fprntlvl)

      integer ierr
      integer fprntlvl
      integer*8 fsolver

      call HYPRE_ParCSRBiCGSTABSetPrintLev(fsolver, fprntlvl, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrbicgstabsetprintlevel error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRBiCGSTABGetNumIter
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrbicgstabgetnumiter(fsolver, fnumiter)

      integer ierr
      integer fnumiter
      integer*8 fsolver

      call HYPRE_ParCSRBiCGSTABGetNumIter(fsolver, fnumiter, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrbicgstabgetnumiterations error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRBiCGSTABGetFinalRelativeResidualNorm
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrbicgstabgetfinalre(fsolver, fnorm)

      integer ierr
      integer*8 fsolver
      double precision fnorm

      call HYPRE_ParCSRBiCGSTABGetFinalRel(fsolver, fnorm, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrbicgstabgetfinalrel error: ', ierr
      endif
      return
      end



c-------------------------------------------------------------------------
c HYPRE_BlockTridiagCreate
c-------------------------------------------------------------------------
      subroutine fhypre_blocktridiagcreate(fsolver)

      integer ierr
      integer*8 fsolver

      call HYPRE_BlockTridiagCreate(fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_blocktridiagcreate error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_BlockTridiagDestroy
c-------------------------------------------------------------------------
      subroutine fhypre_blocktridiagdestroy(fsolver)

      integer ierr
      integer*8 fsolver

      call HYPRE_BlockTridiagDestroy(fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_blocktridiagdestroy error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_BlockTridiagSetup
c-------------------------------------------------------------------------
      subroutine fhypre_blocktridiagsetup(fsolver, fA, fb, fx)

      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call HYPRE_BlockTridiagSetup(fsolver, fA, fb, fx, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_blocktridiagsetup error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_BlockTridiagSolve
c-------------------------------------------------------------------------
      subroutine fhypre_blocktridiagsolve(fsolver, fA, fb, fx)

      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call HYPRE_BlockTridiagSolve(fsolver, fA, fb, fx, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_blocktridiagsolve error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_BlockTridiagSetIndexSet
c-------------------------------------------------------------------------
      subroutine fhypre_blocktridiagsetindexset(fsolver, fn, finds)

      integer ierr
      integer fn
      integer finds
      integer*8 fsolver

      call HYPRE_BlockTridiagSetIndexSet(fsolver, fn, finds, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_blocktridiagsetindexset error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_BlockTridiagSetAMGStrengthThreshold
c-------------------------------------------------------------------------
      subroutine fhypre_blocktridiagsetamgstreng(fsolver, fthresh)

      integer ierr
      integer*8 fsolver
      double precision fthresh

      call HYPRE_BlockTridiagSetAMGStrengt(fsolver, fthresh,
     1                                     ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_blocktridiagsetamgstrengththreshold error: ',
     1                                            ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_BlockTridiagSetAMGNumSweeps
c-------------------------------------------------------------------------
      subroutine fhypre_blocktridiagsetamgnumswe(fsolver, fnumsweep)

      integer ierr
      integer fnumsweep
      integer*8 fsolver

      call HYPRE_BlockTridiagSetAMGNumSwee(fsolver, fnumsweep, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_blocktridiagsetamgnumsweeps error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_BlockTridiagSetAMGRelaxType
c-------------------------------------------------------------------------
      subroutine fhypre_blocktridiagsetamgrelaxt(fsolver, frlxtyp)

      integer ierr
      integer frlxtyp
      integer*8 fsolver

      call HYPRE_BlockTridiagSetAMGRelaxTy(fsolver, frlxtyp, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_blocktridiagsetamgrelaxype error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_BlockTridiagSetPrintLevel
c-------------------------------------------------------------------------
      subroutine fhypre_blocktridiagsetprintleve(fsolver, fprntlvl)

      integer ierr
      integer fprntlvl
      integer*8 fsolver

      call HYPRE_BlockTridiagSetPrintLevel(fsolver, fprntlvl, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_blocktridiagsetprintlevel error: ', ierr
      endif

      return
      end



c-------------------------------------------------------------------------
c HYPRE_ParCSRCGNRCreate
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrcgnrcreate(fcomm, fsolver)

      integer ierr
      integer fcomm
      integer*8 fsolver

      call HYPRE_ParCSRCGNRCreate(fcomm, fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrcgnrcreate error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRCGNRDestroy
c------------------------------------------------------------------------- 
      subroutine fhypre_parcsrcgnrdestroy(fsolver)

      integer ierr
      integer*8 fsolver

      call HYPRE_ParCSRCGNRDestroy(fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrcgnrdestroy error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRCGNRSetup
c------------------------------------------------------------------------- 
      subroutine fhypre_parcsrcgnrsetup(fsolver, fA, fb, fx)

      integer ierr
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call HYPRE_ParCSRCGNRSetup(fsolver, fA, fb, fx, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrcgnrsetup error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRCGNRSolve
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrcgnrsolve(fsolver, fA, fb, fx)

      integer ierr
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call HYPRE_ParCSRCGNRSolve(fsolver, fA, fb, fx, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrcgnrsolve error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRCGNRSetTol
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrcgnrsettol(fsolver, ftol)

      integer ierr
      integer*8 fsolver
      double precision ftol

      call HYPRE_ParCSRCGNRSetTol(fsolver, ftol, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrcgnrsettol error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRCGNRSetMinIter
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrcgnrsetminiter(fsolver, fminiter)
 
      integer ierr
      integer fminiter
      integer*8 fsolver

      call HYPRE_ParCSRCGNRSetMinIter(fsolver, fminiter, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrcgnrsetminiter error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRCGNRSetMaxIter
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrcgnrsetmaxiter(fsolver, fmaxiter)
 
      integer ierr
      integer fmaxiter
      integer*8 fsolver

      call HYPRE_ParCSRCGNRSetMaxIter(fsolver, fmaxiter, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrcgnrsetmaxiter error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRCGNRSetStopCrit
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrcgnrsetstopcri(fsolver, fstopcrit)
 
      integer ierr
      integer fstopcrit
      integer*8 fsolver

      call HYPRE_ParCSRCGNRSetStopCrit(fsolver, fstopcrit, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrcgnrsetstopcrit error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRCGNRSetPrecond
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrcgnrsetprecond(fsolver, fprecond_id, 
     1                                       fprecond)

      integer ierr
      integer fprecond_id
      integer*8 fsolver
      integer*8 fprecond

      call HYPRE_ParCSRCGNRSetPrecond(fsolver, fprecond_id, fprecond, 
     1                                ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrcgnrsetprecond error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRCGNRGetPrecond
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrcgnrgetprecond(fsolver, fprecond)

      integer ierr
      integer*8 fsolver
      integer*8 fprecond

      call HYPRE_ParCSRCGNRGetPrecond(fsolver, fprecond, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrcgnrgetprecond error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRCGNRSetLogging
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrcgnrsetlogging(fsolver, flog)

      integer ierr
      integer flog
      integer*8 fsolver

      call HYPRE_ParCSRCGNRSetLogging(fsolver, flog, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrcgnrsetlogging error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRCGNRGetNumIteration
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrcgnrgetnumiteratio(fsolver, fnumiter)

      integer ierr
      integer fnumiter
      integer*8 fsolver

      call HYPRE_ParCSRCGNRGetNumIteration(fsolver, fnumiter, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrcgnrgetnumiterations error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRCGNRGetFinalRelativeResidualNorm
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrcgnrgetfinalrelati(fsolver, fnorm)

      integer ierr
      integer*8 fsolver
      double precision fnorm

      call HYPRE_ParCSRCGNRGetFinalRelativ(fsolver, fnorm, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrcgnrgetfinalrelativ error: ', ierr
      endif

      return
      end



c-------------------------------------------------------------------------
c HYPRE_EuclidCreate
c-------------------------------------------------------------------------
      subroutine fhypre_euclidcreate(fcomm, fsolver)

      integer ierr
      integer fcomm
      integer*8 fsolver

      call HYPRE_EuclidCreate(fcomm, fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_euclidcreate error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_EuclidDestroy
c-------------------------------------------------------------------------
      subroutine fhypre_eucliddestroy(fsolver)

      integer ierr
      integer*8 fsolver

      call HYPRE_EuclidDestroy(fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_eucliddestroy error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_EuclidSetup
c-------------------------------------------------------------------------
      subroutine fhypre_euclidsetup(fsolver, fA, fb, fx)

      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call HYPRE_EuclidSetup(fsolver, fA, fb, fx, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_euclidsetup error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_EuclidSolve
c-------------------------------------------------------------------------
      subroutine fhypre_euclidsolve(fsolver, fA, fb, fx)

      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call HYPRE_EuclidSolve(fsolver, fA, fb, fx, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_euclidsolve error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_EuclidSetParams
c-------------------------------------------------------------------------
      subroutine fhypre_euclidsetparams(fsolver, fargc, fargv)

      integer ierr
      integer fargc
      integer*8 fsolver
      character*(*) fargv

      call HYPRE_EuclidSetParams(fsolver, fargc, fargv, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_euclidsetparams error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_EuclidSetParamsFromFile
c-------------------------------------------------------------------------
      subroutine fhypre_euclidsetparamsfromfile(fsolver, ffile)

      integer ierr
      integer*8 fsolver
      character*(*) ffile

      call HYPRE_EuclidSetParamsFromFile(fsolver, ffile, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_euclidsetparamsfromfile error: ', ierr
      endif

      return
      end



c-------------------------------------------------------------------------
c HYPRE_ParCSRGMRESCreate
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrgmrescreate(fcomm, fsolver)

      integer ierr
      integer fcomm
      integer*8 fsolver

      call HYPRE_ParCSRGMRESCreate(fcomm, fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrgmrescreate error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRGMRESDestroy
c------------------------------------------------------------------------- 
      subroutine fhypre_parcsrgmresdestroy(fsolver)

      integer ierr
      integer*8 fsolver

      call HYPRE_ParCSRGMRESDestroy(fsolver)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrgmresdestroy error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRGMRESSetup
c------------------------------------------------------------------------- 
      subroutine fhypre_parcsrgmressetup(fsolver, fA, fb, fx)

      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call HYPRE_ParCSRGMRESSetup(fsolver, fA, fb, fx, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrgmressetup error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRGMRESSolve
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrgmressolve(fsolver, fA, fb, fx)

      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call HYPRE_ParCSRGMRESSolve(fsolver, fA, fb, fx, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrgmressolve error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRGMRESSetKDim
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrgmressetkdim(fsolver, fkdim)

      integer ierr
      integer fkdim
      integer*8 fsolver

      call HYPRE_ParCSRGMRESSetKDim(fsolver, fkdim, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrgmressetkdim error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRGMRESSetTol
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrgmressettol(fsolver, ftol)

      integer ierr
      integer*8 fsolver
      double precision ftol

      call HYPRE_ParCSRGMRESSetTol(fsolver, ftol, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrgmressettol error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRGMRESSetMinIter
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrgmressetminiter(fsolver, fminiter)

      integer ierr
      integer fminiter
      integer*8 fsolver

      call HYPRE_ParCSRGMRESSetMinIter(fsolver, fminiter, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrgmressetminiter error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRGMRESSetMaxIter
c--------------------------------------------------------------------------
      subroutine fhypre_parcsrgmressetmaxiter(fsolver, fmaxiter)

      integer ierr
      integer fmaxiter
      integer*8 fsolver

      call HYPRE_ParCSRGMRESSetMaxIter(fsolver, fmaxiter, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrgmressetmaxiter error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRGMRESSetStopCrit
c--------------------------------------------------------------------------
      subroutine fhypre_parcsrgmressetstopcrit(fsolver, fstopcrit)

      integer ierr
      integer fstopcrit
      integer*8 fsolver

      call HYPRE_ParCSRGMRESSetStopCrit(fsolver, fstopcrit, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrgmressetstopcrit error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRGMRESSetPrecond
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrgmressetprecond(fsolver, fprecond_id, 
     1                                        fprecond)

      integer ierr
      integer fprecond_id
      integer*8 fsolver
      integer*8 fprecond

      call HYPRE_ParCSRGMRESSetPrecond(fsolver, fprecond_id, fprecond,
     1                                 ierr)
     
      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrgmressetprecond error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRGMRESGetPrecond
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrgmresgetprecond(fsolver, fprecond)

      integer ierr
      integer*8 fsolver
      integer*8 fprecond

      call HYPRE_ParCSRGMRESGetPrecond(fsolver, fprecond, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrgmresgetprecond error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRGMRESSetLogging
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrgmressetlogging(fsolver, flog)

      integer ierr
      integer flog
      integer*8 fsolver

      call HYPRE_ParCSRGMRESSetLogging(fsolver, flog, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrgmressetlogging error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRGMRESSetPrintLevel
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrgmressetprintlevel(fsolver, fprntlvl)

      integer ierr
      integer fprntlvl
      integer*8 fsolver

      call HYPRE_ParCSRGMRESSetPrintLevel(fsolver, fprntlvl, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrgmressetprintlevel error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRGMRESGetNumIterations
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrgmresgetnumiterati(fsolver, fnumiter)

      integer ierr
      integer fnumiter
      integer*8 fsolver

      call HYPRE_ParCSRGMRESGetNumIteratio(fsolver, fnumiter, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrgmresgetnumiterations error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRGMRESGetFinalRelativeResidualNorm
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrgmresgetfinalrelat(fsolver, fnorm)

      integer ierr
      integer*8 fsolver
      double precision fnorm

      call HYPRE_ParCSRGMRESGetFinalRelati(fsolver, fnorm, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrgmresgetfinalrelative error: ', ierr
      endif

      return
      end



c-------------------------------------------------------------------------
c HYPRE_ParCSRHybridCreate
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrhybridcreate(fsolver)

      integer ierr
      integer*8 fsolver

      call HYPRE_ParCSRHybridCreate(fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrhybridcreate error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRHybridDestroy
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrhybriddestroy(fsolver)

      integer ierr
      integer*8 fsolver

      call HYPRE_ParCSRHybridDestroy(fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrhybriddestroy error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRHybridSetup
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrhybridsetup(fsolver, fA, fb, fx)

      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call HYPRE_ParCSRHybridSetup(fsolver, fA, fb, fx, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrhybridsetup error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRHybridSolve
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrhybridsolve(fsolver, fA, fb, fx)

      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call HYPRE_ParCSRHybridSolve(fsolver, fA, fb, fx, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrhybridsolve error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRHybridSetTol
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrhybridsettol(fsolver, ftol)

      integer ierr
      integer*8 fsolver
      double precision ftol

      call HYPRE_ParCSRHybridSetTol(fsolver, ftol, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrhybridsettol error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRHybridSetConvergenceTol
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrhybridsetconvergenc(fsolver, fcftol)

      integer ierr
      integer*8 fsolver
      double precision fcftol

      call HYPRE_ParCSRHybridSetConvergenc(fsolver, fcftol, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrhybridsetconvergencetol error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRHybridSetDSCGMaxIter
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrhybridsetdscgmaxit(fsolver, fmaxiter)

      integer ierr
      integer fmaxiter
      integer*8 fsolver

      call HYPRE_ParCSRHybridSetDSCGMaxIte(fsolver, fmaxiter, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrhybridsetdscgmaxiter error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRHybridSetPCGMaxIter
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrhybridsetpcgmaxite(fsolver, fmaxiter)

      integer ierr
      integer fmaxiter
      integer*8 fsolver

      call HYPRE_ParCSRHybridSetPCGMaxIter(fsolver, fmaxiter, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrhybridsetpcgmaxiter error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRHybridSetSolverType
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrhybridsetsolvertyp(fsolver, ftype)

      integer ierr
      integer ftype
      integer*8 fsolver

      call HYPRE_ParCSRHybridSetSolverType(fsolver, ftype, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrhybridsetsolvertype error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRHybridSetKDim
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrhybridsetkdim(fsolver, fkdim)

      integer ierr
      integer fkdim
      integer*8 fsolver

      call HYPRE_ParCSRHybridSetKDim(fsolver, fkdim, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrhybridsetkdim error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRHybridSetTwoNorm
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrhybridsettwonorm(fsolver, f2norm)

      integer ierr
      integer f2norm
      integer*8 fsolver

      call HYPRE_ParCSRHybridSetTwoNorm(fsolver, f2norm, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrhybridsettwonorm error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRHybridSetStopCrit
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrhybridsetstopcrit(fsolver, fstopcrit)

      integer ierr
      integer fstopcrit
      integer*8 fsolver

      call HYPRE_ParCSRHybridSetStopCrit(fsolver, fstopcrit, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrhybridsetstopcrit error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRHybridSetRelChange
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrhybridsetrelchange(fsolver, frelchg)

      integer ierr
      integer  frelchg
      integer*8 fsolver

      call HYPRE_ParCSRHybridSetRelChange(fsolver, frelchg, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrhybridsetrelchange error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRHybridSetPrecond
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrhybridsetprecond(fsolver, fpreid, 
     1                                         fpresolver)

      integer ierr
      integer  fpreid
      integer*8 fsolver
      integer*8 fpresolver

      call HYPRE_ParCSRHybridSetPrecond(fsolver, fpreid, fpresolver,
     1                                  ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrhybridsetprecond error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRHybridSetLogging
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrhybridsetlogging(fsolver, flogging)

      integer ierr
      integer  flogging
      integer*8 fsolver

      call HYPRE_ParCSRHybridSetLogging(fsolver, flogging, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrhybridsetlogging error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRHybridSetPrintLevel
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrhybridsetprintlevel(fsolver, fprntlvl)

      integer ierr
      integer  fprntlvl
      integer*8 fsolver

      call HYPRE_ParCSRHybridSetPrintLevel(fsolver, fprntlvl, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrhybridsetprintlevel error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRHybridSetStrongThreshold
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrhybridsetstrongthr(fsolver, fthresh)

      integer ierr
      integer  fthresh
      integer*8 fsolver

      call HYPRE_ParCSRHybridSetStrongThre(fsolver, fthresh, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrhybridsetstrongthreshold error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRHybridSetMaxRowSum
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrhybridsetmaxrowsum(fsolver, fsum)

      integer ierr
      integer  fsum
      integer*8 fsolver

      call HYPRE_ParCSRHybridSetMaxRowSum(fsolver, fsum, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrhybridsetmaxrowsum error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRHybridSetTruncFactor
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrhybridsettruncfact(fsolver, ftfact)

      integer ierr
      integer  ftfact
      integer*8 fsolver

      call HYPRE_ParCSRHybridSetTruncFacto(fsolver, ftfact, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrhybridsettruncfactor error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRHybridSetMaxLevels
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrhybridsetmaxlevels(fsolver, fmaxlvl)

      integer ierr
      integer  fmaxlvl
      integer*8 fsolver

      call HYPRE_ParCSRHybridSetMaxLevels(fsolver, fmaxlvl, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrhybridsetmaxlevels error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRHybridSetMeasureType
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrhybridsetmeasurety(fsolver, fmtype)

      integer ierr
      integer  fmtype
      integer*8 fsolver

      call HYPRE_ParCSRHybridSetMeasureTyp(fsolver, fmtype, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrhybridsetmeasuretype error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRHybridSetCoarsenType
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrhybridsetcoarsenty(fsolver, fcoarse)

      integer ierr
      integer  fcoarse
      integer*8 fsolver

      call HYPRE_ParCSRHybridSetCoarsenTyp(fsolver, fcoarse, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrhybridsetcoarsentype error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRHybridSetCycleType
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrhybridsetcycletype(fsolver, fcycle)

      integer ierr
      integer  fcycle
      integer*8 fsolver

      call HYPRE_ParCSRHybridSetCycleType(fsolver, fcycle, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrhybridsetcycletype error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRHybridSetNumGridSweeps
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrhybridsetnumgridsw(fsolver, fsweep)

      integer ierr
      integer  fsweep
      integer*8 fsolver

      call HYPRE_ParCSRHybridSetNumGridSwe(fsolver, fsweep, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrhybridsetnumgridsweeps error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRHybridSetGridRelaxType
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrhybridsetgridrlxtyp(fsolver, frlxt)

      integer ierr
      integer  frlxt
      integer*8 fsolver

      call HYPRE_ParCSRHybridSetGridRelaxT(fsolver, frlxt, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrhybridsetgridrelaxtype error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRHybridSetGridRelaxPoints
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrhybridsetgridrlxpts(fsolver, frlxp)

      integer ierr
      integer  frlxp
      integer*8 fsolver

      call HYPRE_ParCSRHybridSetGridRelaxP(fsolver, frlxp, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrhybridsetgridrelaxpoints error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRHybridSetNumSweeps
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrhybridsetnumsweeps(fsolver, fsweep)

      integer ierr
      integer  fsweep
      integer*8 fsolver

      call HYPRE_ParCSRHybridSetNumSweeps(fsolver, fsweep, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrhybridsetnumsweeps error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRHybridSetCycleNumSweeps
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrhybridsetcyclenums(fsolver, fsweep)

      integer ierr
      integer  fsweep
      integer*8 fsolver

      call HYPRE_ParCSRHybridSetCycleNumSw(fsolver, fsweep, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrhybridsetcyclenumsweeps error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRHybridSetRelaxType
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrhybridsetrelaxtype(fsolver, frlxt)

      integer ierr
      integer  frlxt
      integer*8 fsolver

      call HYPRE_ParCSRHybridSetRelaxType(fsolver, frlxt, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrhybridsetrelaxtype error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRHybridSetCycleRelaxType
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrhybridsetcyclerela(fsolver, frlxt)

      integer ierr
      integer  frlxt
      integer*8 fsolver

      call HYPRE_ParCSRHybridSetCycleRelax(fsolver, frlxt, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrhybridsetcyclerelaxtype error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRHybridSetRelaxOrder
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrhybridsetrelaxorde(fsolver, frlx)

      integer ierr
      integer  frlx
      integer*8 fsolver

      call HYPRE_ParCSRHybridSetRelaxOrder(fsolver, frlx, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrhybridsetrelaxorder error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRHybridSetRelaxWt
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrhybridsetrelaxwt(fsolver, frlx)

      integer ierr
      integer  frlx
      integer*8 fsolver

      call HYPRE_ParCSRHybridSetRelaxWt(fsolver, frlx, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrhybridsetrelaxwt error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRHybridSetLevelRelaxWt
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrhybridsetlevelrela(fsolver, frlx)

      integer ierr
      integer  frlx
      integer*8 fsolver

      call HYPRE_ParCSRHybridSetLevelRelax(fsolver, frlx, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrhybridsetlevelrelaxwt error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRHybridSetOuterWt
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrhybridsetouterwt(fsolver, fout)

      integer ierr
      integer  fout
      integer*8 fsolver

      call HYPRE_ParCSRHybridSetOuterWt(fsolver, fout, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrhybridsetouterwt error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRHybridSetLevelOuterWt
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrhybridsetleveloute(fsolver, fout)

      integer ierr
      integer  fout
      integer*8 fsolver

      call HYPRE_ParCSRHybridSetLevelOuter(fsolver, fout ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrhybridsetlevelouterwt error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRHybridSetRelaxWeight
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrhybridsetrelaxweig(fsolver, frlx)

      integer ierr
      integer  frlx
      integer*8 fsolver

      call HYPRE_ParCSRHybridSetRelaxWeigh(fsolver, frlx, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrhybridsetrelaxweight error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRHybridSetOmega
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrhybridsetomega(fsolver, fomega)

      integer ierr
      integer  fomega
      integer*8 fsolver

      call HYPRE_ParCSRHybridSetOmega(fsolver, fomega, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrhybridsetomega error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRHybridGetNumIterations
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrhybridgetnumiterat(fsolver, fiters)

      integer ierr
      integer  fiters
      integer*8 fsolver

      call HYPRE_ParCSRHybridGetNumIterati(fsolver, fiters, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrhybridgetnumiterations error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRHybridGetDSCGNumIterations
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrhybridgetdscgnumit(fsolver, fiters)

      integer ierr
      integer  fiters
      integer*8 fsolver

      call HYPRE_ParCSRHybridGetDSCGNumIte(fsolver, fiters, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrhybridgetdscgnumiterations error: ',
     1                                                    ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRHybridGetPCGNumIterations
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrhybridgetpcgnumite(fsolver, fiters)

      integer ierr
      integer  fiters
      integer*8 fsolver

      call HYPRE_ParCSRHybridGetPCGNumIter(fsolver, fiters, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrhybridgetpcgnumiterations error: ',
     1                                                    ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRHybridGetFinalRelativeResidualNorm
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrhybridgetfinalrela(fsolver, fnorm)

      integer ierr
      integer  fnorm
      integer*8 fsolver

      call HYPRE_ParCSRHybridGetFinalRelat(fsolver, 
     1                                     fnorm, ierr)

      if(ierr .ne. 0) then
         print *, 
     1        'fhypre_parcsrhybridgetfinalrelativeresidualnorm error: ',
     1                                                    ierr
      endif

      return
      end



c-------------------------------------------------------------------------
c HYPRE_ParSetRandomValues
c-------------------------------------------------------------------------
      subroutine fhypre_parsetrandomvalues(fv, fseed)

      integer ierr
      integer fseed
      integer*8 fv

      call HYPRE_ParVectorSetRandomValues(fv, fseed, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parsetrandomvalues error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParPrintVector
c-------------------------------------------------------------------------
      subroutine fhypre_parprintvector(fv, ffile)

      integer ierr
      integer*8 fv
      character*(*) ffile

      call hypre_ParVectorPrint(fv, ffile, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parprintvector error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParReadVector
c-------------------------------------------------------------------------
      subroutine fhypre_parreadvector(fcomm, ffile)

      integer ierr
      integer fcomm
      character*(*) ffile

      call hypre_ParReadVector(fcomm, ffile, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parreadvector error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParVectorSize
c-------------------------------------------------------------------------
      subroutine fhypre_parvectorsize(fx)

      integer ierr
      integer*8 fx

      call hypre_ParVectorSize(fx, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parvectorsize error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRMultiVectorPrint
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrmultivectorprint(fx, ffile)

      integer ierr
      integer*8 fx
      character*(*) ffile

      call hypre_ParCSRMultiVectorPrint(fx, ffile, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrmultivectorprint error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRMultiVectorRead
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrmultivectorread(fcomm, fii, ffile)

      integer ierr
      integer fcomm
      integer*8 fii
      character*(*) ffile

      call hypre_ParCSRMultiVectorRead(fcomm, fii, ffile, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrmultivectorread error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c aux_maskCount
c-------------------------------------------------------------------------
      subroutine fhypre_aux_maskcount(fn, fmask)

      integer ierr
      integer fn
      integer fmask

      call aux_maskCount(fn, fmask, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_aux_maskcount error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c aux_indexFromMask
c-------------------------------------------------------------------------
      subroutine fhypre_auxindexfrommask(fn, fmask, findex)

      integer ierr
      integer fn
      integer fmask
      integer findex

      call aux_indexFromMask(fn, fmask, findex, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_aux_indexfrommask error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_TempParCSRSetupInterpreter
c-------------------------------------------------------------------------
      subroutine fhypre_tempparcsrsetupinterpret(fi)

      integer ierr
      integer*8 fi

      call HYPRE_TempParCSRSetupInterprete(fi, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_tempparcsrsetupinterpreter error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRSetupInterpreter
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrsetupinterpreter(fi)

      integer ierr
      integer*8 fi

      call HYPRE_ParCSRSetupInterpreter(fi, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrsetupinterpreter error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRSetupMatvec
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrsetupmatvec(fmv)

      integer ierr
      integer*8 fmv

      call HYPRE_ParCSRSetupMatvec(fmv, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrsetupmatvec error: ', ierr
      endif

      return
      end



c-------------------------------------------------------------------------
c HYPRE_ParaSailsCreate
c-------------------------------------------------------------------------
      subroutine fhypre_parasailscreate(fcomm, fsolver)

      integer ierr
      integer fcomm
      integer*8 fsolver

      call HYPRE_ParaSailsCreate(fcomm, fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parasailscreate error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParaSailsDestroy
c-------------------------------------------------------------------------
      subroutine fhypre_parasailsdestroy(fsolver)

      integer ierr
      integer*8 fsolver

      call HYPRE_ParaSailsDestroy(fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parasailsdestroy error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParaSailsSetup
c------------------------------------------------------------------------- 
      subroutine fhypre_parasailssetup(fsolver, fA, fb, fx)

      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call HYPRE_ParaSailsSetup(fsolver, fA, fb, fx, ierr) 

      if(ierr .ne. 0) then
         print *, 'fhypre_parasailssetup error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParaSailsSolve
c------------------------------------------------------------------------- 
      subroutine fhypre_parasailssolve(fsolver, fA, fb, fx)

      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call HYPRE_ParaSailsSolve(fsolver, fA, fb, fx, ierr) 

      if(ierr .ne. 0) then
         print *, 'fhypre_parasailssolve error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParaSailsSetParams
c-------------------------------------------------------------------------
      subroutine fhypre_parasailssetparams(fsolver, fthresh, fnlevels)

      integer ierr
      integer fnlevels
      integer*8 fsolver
      double precision fthresh

      call HYPRE_ParaSailsSetParams(fsolver, fthresh, fnlevels, ierr) 

      if(ierr .ne. 0) then
         print *, 'fhypre_parasailssetparams error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParaSailsSetThresh
c-------------------------------------------------------------------------
      subroutine fhypre_parasailssetthresh(fsolver, fthresh)

      integer ierr
      integer*8 fsolver
      double precision fthresh

      call HYPRE_ParaSailsSetThresh(fsolver, fthresh, ierr) 

      if(ierr .ne. 0) then
         print *, 'fhypre_parasailssetthresh error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParaSailsGetThresh
c-------------------------------------------------------------------------
      subroutine fhypre_parasailsgetthresh(fsolver, fthresh)

      integer ierr
      integer*8 fsolver
      double precision fthresh

      call HYPRE_ParaSailsGetThresh(fsolver, fthresh, ierr) 

      if(ierr .ne. 0) then
         print *, 'fhypre_parasailsgetthresh error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParaSailsSetNlevels
c-------------------------------------------------------------------------
      subroutine fhypre_parasailssetnlevels(fsolver, fnlevels)

      integer ierr
      integer fnlevels
      integer*8 fsolver

      call HYPRE_ParaSailsSetNlevels(fsolver, fnlevels, ierr) 

      if(ierr .ne. 0) then
         print *, 'fhypre_parasailssetnlevels error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParaSailsGetNlevels
c-------------------------------------------------------------------------
      subroutine fhypre_parasailsgetnlevels(fsolver, fnlevels)

      integer ierr
      integer fnlevels
      integer*8 fsolver

      call HYPRE_ParaSailsGetNlevels(fsolver, fnlevels, ierr) 

      if(ierr .ne. 0) then
         print *, 'fhypre_parasailsgetnlevels error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParaSailsSetFilter
c-------------------------------------------------------------------------
      subroutine fhypre_parasailssetfilter(fsolver, ffilter)

      integer ierr
      integer*8 fsolver
      double precision ffilter

      call HYPRE_ParaSailsSetFilter(fsolver, ffilter, ierr) 

      if(ierr .ne. 0) then
         print *, 'fhypre_parasailssetfilter error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParaSailsGetFilter
c-------------------------------------------------------------------------
      subroutine fhypre_parasailsgetfilter(fsolver, ffilter)

      integer ierr
      integer*8 fsolver
      double precision ffilter

      call HYPRE_ParaSailsGetFilter(fsolver, ffilter, ierr) 

      if(ierr .ne. 0) then
         print *, 'fhypre_parasailsgetfilter error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParaSailsSetSym
c-------------------------------------------------------------------------
      subroutine fhypre_parasailssetsym(fsolver, fsym)

      integer ierr
      integer fsym
      integer*8 fsolver

      call HYPRE_ParaSailsSetSym(fsolver, fsym, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parasailssetsym error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParaSailsGetSym
c-------------------------------------------------------------------------
      subroutine fhypre_parasailsgetsym(fsolver, fsym)

      integer ierr
      integer fsym
      integer*8 fsolver

      call HYPRE_ParaSailsGetSym(fsolver, fsym, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parasailsgetsym error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParaSailsSetLoadbal
c-------------------------------------------------------------------------
      subroutine fhypre_parasailssetloadbal(fsolver, floadbal)

      integer ierr
      integer*8 fsolver
      double precision floadbal

      call HYPRE_ParaSailsSetLoadbal(fsolver, floadbal, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parasailssetloadbal error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParaSailsGetLoadbal
c-------------------------------------------------------------------------
      subroutine fhypre_parasailsgetloadbal(fsolver, floadbal)

      integer ierr
      integer*8 fsolver
      double precision floadbal

      call HYPRE_ParaSailsGetLoadbal(fsolver, floadbal, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parasailsgetloadbal error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParaSailsSetReuse
c-------------------------------------------------------------------------
      subroutine fhypre_parasailssetreuse(fsolver, freuse)

      integer ierr
      integer freuse
      integer*8 fsolver

      call HYPRE_ParaSailsSetReuse(fsolver, freuse, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parasailssetreuse error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParaSailsGetReuse
c-------------------------------------------------------------------------
      subroutine fhypre_parasailsgetreuse(fsolver, freuse)

      integer ierr
      integer freuse
      integer*8 fsolver

      call HYPRE_ParaSailsGetReuse(fsolver, freuse, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parasailsgetreuse error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParaSailsSetLogging
c-------------------------------------------------------------------------
      subroutine fhypre_parasailssetlogging(fsolver, flog)

      integer ierr
      integer flog
      integer*8 fsolver

      call HYPRE_ParaSailsSetLogging(fsolver, flog, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parasailssetlogging error: ', ierr
      endif
      
      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParaSailsGetLogging
c-------------------------------------------------------------------------
      subroutine fhypre_parasailsgetlogging(fsolver, flog)

      integer ierr
      integer flog
      integer*8 fsolver

      call HYPRE_ParaSailsGetLogging(fsolver, flog, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parasailsgetlogging error: ', ierr
      endif
      
      return
      end



c-------------------------------------------------------------------------
c HYPRE_ParCSRPCGCreate
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrpcgcreate(fcomm, fsolver)

      integer ierr
      integer fcomm
      integer*8 fsolver

      call HYPRE_ParCSRPCGCreate(fcomm, fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrpcgcreate error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRPCGDestroy
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrpcgdestroy(fsolver)

      integer ierr
      integer*8 fsolver

      call HYPRE_ParCSRPCGDestroy(fsolver)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrpcgdestroy error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRPCGSetup
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrpcgsetup(fsolver, fA, fb, fx)

      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call HYPRE_ParCSRPCGSetup(fsolver, fA, fb, fx, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrpcgcreate error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRPCGSolve
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrpcgsolve(fsolver, fA, fb, fx)

      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call HYPRE_ParCSRPCGSolve(fsolver, fA, fb, fx, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrpcgsolve error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRPCGSetTol
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrpcgsettol(fsolver, ftol)

      integer ierr
      integer*8 fsolver
      double precision ftol

      call HYPRE_ParCSRPCGSetTol(fsolver, ftol, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrpcgsettol error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRPCGSetMaxIter
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrpcgsetmaxiter(fsolver, fmaxiter)

      integer ierr
      integer fmaxiter
      integer*8 fsolver

      call HYPRE_ParCSRPCGSetMaxIter(fsolver, fmaxiter, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrpcgsetmaxiter error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRPCGSetStopCrit
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrpcgsetstopcrit(fsolver, ftol)

      integer ierr
      integer*8 fsolver
      double precision ftol

      call HYPRE_ParCSRPCGSetStopCrit(fsolver, ftol, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrpcgsetstopcrit error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRPCGSetTwoNorm
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrpcgsettwonorm(fsolver, ftwonorm)

      integer ierr
      integer ftwonorm
      integer*8 fsolver

      call HYPRE_ParCSRPCGSetTwoNorm(fsolver, ftwonorm, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrpcgsettwonorm error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRPCGSetRelChange
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrpcgsetrelchange(fsolver, frelchange)

      integer ierr
      integer frelchange
      integer*8 fsolver

      call HYPRE_ParCSRPCGSetRelChange(fsolver, frelchange, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrpcgsetrelchange error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRPCGSetPrecond
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrpcgsetprecond(fsolver, fprecond_id, 
     1                                      fprecond)

      integer ierr
      integer fprecond_id
      integer*8 fsolver
      integer*8 fprecond

      call HYPRE_ParCSRPCGSetPrecond(fsolver, fprecond_id, fprecond, 
     1                              ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrpcgsetprecond error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRPCGGetPrecond
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrpcggetprecond(fsolver, fprecond)

      integer ierr
      integer*8 fsolver
      integer*8 fprecond

      call HYPRE_ParCSRPCGGetPrecond(fsolver, fprecond, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrpcggetprecond error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRPCGSetPrintLevel
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrpcgsetprintlevel(fsolver, flevel)

      integer ierr
      integer flevel
      integer*8 fsolver

      call HYPRE_ParCSRPCGSetPrintLevel(fsolver, flevel, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrpcgsetprintlevel error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRPCGGetNumIterations
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrpcggetnumiteration(fsolver, fnumiter)

      integer ierr
      integer fnumiter
      integer*8 fsolver

      call HYPRE_ParCSRPCGGetNumIterations(fsolver, fnumiter, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrpcggetnumiteration error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRPCGGetFinalRelativeResidualNorm
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrpcggetfinalrelativ(fsolver, fnorm)

      integer ierr
      integer*8 fsolver
      double precision fnorm

      call HYPRE_ParCSRPCGGetFinalRelative(fsolver, fnorm, ierr)
      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrpcggetfinalrelative error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRDiagScaleSetup
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrdiagscalesetup(fsolver, fA, fb, fx)

      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call HYPRE_ParCSRDiagScaleSetup(fsolver, fA, fb, fx, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrdiagscalesetup error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRDiagScale
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrdiagscale(fsolver, fHA, fHy, fHx)

      integer ierr
      integer*8 fsolver
      integer*8 fHA
      integer*8 fHy
      integer*8 fHx

      call HYPRE_ParCSRDiagScale(fsolver, fHA, fHy, fHx, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrdiagscale error: ', ierr
      endif

      return
      end



c-------------------------------------------------------------------------
c HYPRE_ParCSRPilutCreate
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrpilutcreate(fcomm, fsolver)

      integer ierr
      integer fcomm
      integer*8 fsolver

      call HYPRE_ParCSRPilutCreate(fcomm, fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrpilutcreate error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRPilutDestroy
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrpilutdestroy(fsolver)

      integer ierr
      integer*8 fsolver

      call HYPRE_ParCSRPilutDestroy(fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrpilutdestroy error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRPilutSetup
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrpilutsetup(fsolver, fA, fb, fx)

      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call HYPRE_ParCSRPilutSetup(fsolver, fA, fb, fx, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrpilutsetup error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRPilutSolve
c------------------------------------------------------------------------- 
      subroutine fhypre_parcsrpilutsolve(fsolver, fA, fb, fx)

      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call HYPRE_ParCSRPilutSolve(fsolver, fA, fb, fx, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrpilutsolve error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRPilutSetMaxIter
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrpilutsetmaxiter(fsolver, fmaxiter)

      integer ierr
      integer fmaxiter
      integer*8 fsolver

      call HYPRE_ParCSRPilutSetMaxIter(fsolver, fmaxiter, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrpilutsetmaxiter error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRPilutSetDropToleran
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrpilutsetdroptolera(fsolver, ftol)

      integer ierr
      integer*8 fsolver
      double precision ftol

      call HYPRE_ParCSRPilutSetDropToleran(fsolver, ftol, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrpilutsetdroptol error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_ParCSRPilutSetFacRowSize
c--------------------------------------------------------------------------
      subroutine fhypre_parcsrpilutsetfacrowsize(fsolver, fsize)
      
      integer ierr
      integer fsize
      integer*8 fsolver

      call HYPRE_ParCSRPilutSetFacRowSize(fsolver, fsize, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrpilutsetfacrowsize error: ', ierr
      endif

      return
      end



c-------------------------------------------------------------------------
c HYPRE_SchwarzCreate
c-------------------------------------------------------------------------
      subroutine fhypre_schwarzcreate(fsolver)

      integer ierr
      integer*8 fsolver

      call HYPRE_SchwarzCreate(fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_schwarzcreate error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_SchwarzDestroy
c-------------------------------------------------------------------------
      subroutine fhypre_schwarzdestroy(fsolver)

      integer ierr
      integer*8 fsolver

      call HYPRE_SchwarzDestroy(fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_schwarzdestroy error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_SchwarzSetup
c-------------------------------------------------------------------------
      subroutine fhypre_schwarzsetup(fsolver, fA, fb, fx)

      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call HYPRE_SchwarzSetup(fsolver, fA, fb, fx, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_schwarzsetup error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_SchwarzSolve
c------------------------------------------------------------------------- 
      subroutine fhypre_schwarzsolve(fsolver, fA, fb, fx)

      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call HYPRE_SchwarzSolve(fsolver, fA, fb, fx, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_schwarzsolve error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_SchwarzSetVariant
c------------------------------------------------------------------------- 
      subroutine fhypre_schwarzsetvariant(fsolver, fvariant)

      integer ierr
      integer fvariant
      integer*8 fsolver

      call HYPRE_SchwarzSetVariant(fsolver, fvariant, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_schwarzsetvariant error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_SchwarzSetOverlap
c------------------------------------------------------------------------- 
      subroutine fhypre_schwarzsetoverlap(fsolver, foverlap)

      integer ierr
      integer foverlap
      integer*8 fsolver

      call HYPRE_SchwarzSetOverlap(fsolver, foverlap, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_schwarzsetoverlap error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_SchwarzSetDomainType
c------------------------------------------------------------------------- 
      subroutine fhypre_schwarzsetdomaintype(fsolver, fdomaint)

      integer ierr
      integer fdomaint
      integer*8 fsolver

      call HYPRE_SchwarzSetDomainType(fsolver, fdomaint, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_schwarzsetdomaintype error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_SchwarzSetDomainStructure
c------------------------------------------------------------------------- 
      subroutine fhypre_schwarzsetdomainstructur(fsolver, fdomains)

      integer ierr
      integer fdomains
      integer*8 fsolver

      call HYPRE_SchwarzSetDomainStructure(fsolver, fdomains, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_schwarzsetdomainstructure error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_SchwarzSetNumFunctions
c------------------------------------------------------------------------- 
      subroutine fhypre_schwarzsetnumfunctions(fsolver, fnumfncs)

      integer ierr
      integer fnumfncs
      integer*8 fsolver

      call HYPRE_SchwarzSetNumFunctions(fsolver, fnumfncs, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_schwarzsetnumfunctions error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_SchwarzSetRelaxWeight
c------------------------------------------------------------------------- 
      subroutine fhypre_schwarzsetrelaxweight(fsolver, frlxwt)

      integer ierr
      integer*8 fsolver
      double precision frlxwt

      call HYPRE_SchwarzSetRelaxWeight(fsolver, frlxwt, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_schwarzsetrelaxweight error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_SchwarzSetDofFunc
c------------------------------------------------------------------------- 
      subroutine fhypre_schwarzsetdoffunc(fsolver, fdofnc)

      integer ierr
      integer fdofnc
      integer*8 fsolver

      call HYPRE_SchwarzSetDofFunc(fsolver, fdofnc, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_schwarzsetdoffunc error: ', ierr
      endif

      return
      end
