@echo off
REM Record Space Invaders gameplay videos from trained checkpoints.
REM Activates the project venv, then runs record_video_space_invaders.py.
REM
REM Usage:
REM   record_video_space_invaders.bat                  <- all algos, all checkpoints
REM   record_video_space_invaders.bat dqn              <- DQN only
REM   record_video_space_invaders.bat ppo videos\si 60 <- PPO, custom output dir, 60 fps

setlocal

REM Resolve the directory this bat file lives in
set "ROOT=%~dp0"

REM Activate virtual environment
if exist "%ROOT%venv\Scripts\activate.bat" (
    call "%ROOT%venv\Scripts\activate.bat"
) else (
    echo [WARNING] venv not found at %ROOT%venv — using system Python
)

REM Optional positional arguments:
REM   %1 = algo filter   (dqn / a2c / ppo, default: all)
REM   %2 = output dir    (default: videos\space_invaders)
REM   %3 = fps           (default: 30)

set "ALGO_ARG="
set "OUTPUT_ARG=--output_dir videos\space_invaders"
set "FPS_ARG=--fps 30"

if not "%~1"=="" set "ALGO_ARG=--algo %~1"
if not "%~2"=="" set "OUTPUT_ARG=--output_dir %~2"
if not "%~3"=="" set "FPS_ARG=--fps %~3"

echo.
echo =========================================================
echo  Space Invaders Video Recorder
echo  Output : %OUTPUT_ARG:~13%
echo  FPS    : %FPS_ARG:~6%
if not "%ALGO_ARG%"=="" echo  Algo   : %ALGO_ARG:~7%
echo =========================================================
echo.

python "%ROOT%scripts\record_video_space_invaders.py" ^
    %ALGO_ARG% ^
    %OUTPUT_ARG% ^
    %FPS_ARG%

if errorlevel 1 (
    echo.
    echo [ERROR] Script exited with an error. Check the output above.
    pause
    exit /b 1
)

echo.
echo Done. Videos saved to %OUTPUT_ARG:~13%
pause
endlocal
