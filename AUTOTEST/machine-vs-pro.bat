@echo off
rem Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
rem HYPRE Project Developers. See the top-level COPYRIGHT file for details.
rem
rem SPDX-License-Identifier: (Apache-2.0 OR MIT)

setlocal

rem This script can be run from anywhere, but must live in the same directory
rem (e.g., AUTOTEST) as supporting batch scripts like 'cmake.bat'

rem directory where script is being run
set rundir=%cd%

rem directory where script is located
cd %~dp0
set scriptdir=%cd%

rem source directory passed in as argument 1
cd %rundir%
cd %1
set srcdir=%cd%

rem output directory is a subdirectory of rundir
set outdir=%rundir%\machine-vs-pro.dir

rem create clean output directory
if exist %outdir% rmdir /s /q %outdir%
mkdir %outdir%

rem run the cmake.bat script from rundir in a subshell to avoid overwriting variables
rem (i.e., use 'cmd /c')
cd %rundir%

cmd /c %scriptdir%\cmake.bat %srcdir% " "
move %rundir%\cmake.dir %outdir%\cmake-default.dir
move %rundir%\cmake.err %outdir%\cmake-default.err

cmd /c %scriptdir%\cmake.bat %srcdir% "-DHYPRE_USING_OPENMP=ON"
move %rundir%\cmake.dir %outdir%\cmake-openmp.dir
move %rundir%\cmake.err %outdir%\cmake-openmp.err

cmd /c %scriptdir%\cmake.bat %srcdir% "-DHYPRE_SEQUENTIAL=ON"
move %rundir%\cmake.dir %outdir%\cmake-sequential.dir
move %rundir%\cmake.err %outdir%\cmake-sequential.err

cmd /c %scriptdir%\cmake.bat %srcdir% "-DHYPRE_BIGINT=ON"
move %rundir%\cmake.dir %outdir%\cmake-bigint.dir
move %rundir%\cmake.err %outdir%\cmake-bigint.err

rem create error file - check file size of cmake error files
cd %rundir%
type NUL > machine-vs-pro.err
for %%f in (%outdir%\*.err) do (
    if %%~zf gtr 0 @echo %%f >> machine-vs-pro.err
)

cd %rundir%

endlocal
