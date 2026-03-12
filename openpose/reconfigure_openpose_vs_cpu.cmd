@echo off
setlocal

set "BUILD_DIR=d:\VITON\openpose\build"
set "SRC_DIR=d:\VITON\openpose"
set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set "CMAKE_EXE=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"

if exist "%BUILD_DIR%" rmdir /s /q "%BUILD_DIR%"
mkdir "%BUILD_DIR%"

call "%VCVARS%"

"%CMAKE_EXE%" -S "%SRC_DIR%" -B "%BUILD_DIR%" -G "Visual Studio 17 2022" -A x64 -DGPU_MODE:STRING=CPU_ONLY -DDOWNLOAD_BODY_25_MODEL:BOOL=OFF -DDOWNLOAD_BODY_COCO_MODEL:BOOL=OFF -DDOWNLOAD_BODY_MPI_MODEL:BOOL=OFF -DDOWNLOAD_FACE_MODEL:BOOL=OFF -DDOWNLOAD_HAND_MODEL:BOOL=OFF -Wno-dev
if errorlevel 1 exit /b 1

echo RECONFIGURE_DONE
exit /b 0
