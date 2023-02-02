@echo off
rem Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
rem HYPRE Project Developers. See the top-level COPYRIGHT file for details.
rem
rem SPDX-License-Identifier: (Apache-2.0 OR MIT)

setlocal

rem This script can be run from anywhere
rem Example usage: cmake.bat ..\src "-DHYPRE_SEQUENTIAL=ON"

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
set outdir=%rundir%\cmake.dir

rem cmake options passed in as argument 2
set cmakeopts=%2

rem set location of cmake and msbuild programs
set CMAKE="C:\Program Files\CMake\bin\cmake.exe"
set MSBUILD="C:\Program Files (x86)\Microsoft Visual Studio\2017\Professional\MSBuild\15.0\Bin\msbuild"

rem create clean output directory
if exist %outdir% rmdir /s /q %outdir%
mkdir %outdir%

rem clean out 'cmbuild' directories
cd %srcdir%      && rmdir /s /q cmbuild & mkdir cmbuild

rem run cmake for hypre library and test directory
cd %srcdir%\cmbuild
%CMAKE% -G "Visual Studio 15 2017" %cmakeopts% "-DHYPRE_BUILD_TESTS=ON" ..               >  %outdir%\lib-cmake.out

rem build release version
cd %srcdir%\cmbuild
%MSBUILD% HYPRE.vcxproj /t:Rebuild /p:Configuration=Release     >  %outdir%\lib-release.out
%MSBUILD% INSTALL.vcxproj /p:Configuration=Release              >> %outdir%\lib-release.out

rem build debug version
cd %srcdir%\cmbuild
%MSBUILD% HYPRE.vcxproj /t:Rebuild /p:Configuration=Debug       >  %outdir%\lib-debug.out
%MSBUILD% INSTALL.vcxproj /p:Configuration=Debug                >> %outdir%\lib-debug.out

rem create error file - inspect output file lines with "Error(s)" substring
cd %rundir%
type NUL > cmake.err
for %%f in (%outdir%\*.out) do (
    set sum=0
    for /f "tokens=1" %%i in ('findstr "Error(s)" %%f') do set /a sum+=%%i
    if %sum% gtr 0 @echo %%f >> cmake.err
)

cd %rundir%

endlocal
