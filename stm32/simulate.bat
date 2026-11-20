@echo off
rem simulate.bat : runs the built firmware on a SIMULATED STM32F4 Discovery board (Renode) - no board needed
rem   .\simulate.bat           a Renode window shows the board's serial output (USART2) live
rem   .\simulate.bat log 10    no window: run 10 s of board time (1-59), serial output saved to
rem                            build\renode_uart.txt and printed
rem Run build.bat first. Renode (free, renode.io) is looked for in ..\..\tools\renode\ (portable zip),
rem on the PATH and in C:\Program Files\Renode. The board model runs the SAME rlpid.elf as the real board;
rem its timing is simulated, so the "network decision" cycle count it prints is not a hardware number.
setlocal
cd /d "%~dp0"

set "RENODE="
for /d %%D in ("%~dp0..\..\tools\renode\renode_*") do if exist "%%D\renode.exe" set "RENODE=%%D\renode.exe"
if not defined RENODE for %%R in (renode.exe) do if not "%%~$PATH:R"=="" set "RENODE=%%~$PATH:R"
if not defined RENODE if exist "C:\Program Files\Renode\bin\Renode.exe" set "RENODE=C:\Program Files\Renode\bin\Renode.exe"
if not defined RENODE goto no_renode
if not exist build\firmware\rlpid.elf ( echo Run build.bat first & exit /b 1 )

set "HERE=%CD:\=/%"
set "SCRIPT=build\simulate.resc"
> "%SCRIPT%" echo using sysbus
>> "%SCRIPT%" echo mach create "rlpid"
>> "%SCRIPT%" echo machine LoadPlatformDescription @platforms/boards/stm32f4_discovery-kit.repl
>> "%SCRIPT%" echo cpu PerformanceInMips 168
>> "%SCRIPT%" echo sysbus LoadELF @%HERE%/build/firmware/rlpid.elf
if /i "%1"=="log" goto log

>> "%SCRIPT%" echo showAnalyzer usart2
>> "%SCRIPT%" echo start
echo Starting Renode: the window "usart2" shows the board's output. Close Renode to stop.
"%RENODE%" "%SCRIPT%"
exit /b

:log
set "SECS=%2"
if "%SECS%"=="" set "SECS=10"
if %SECS% LSS 10 set "SECS=0%SECS%"
if exist build\renode_uart.txt del build\renode_uart.txt
>> "%SCRIPT%" echo usart2 CreateFileBackend @%HERE%/build/renode_uart.txt true
>> "%SCRIPT%" echo emulation RunFor "00:00:%SECS%"
>> "%SCRIPT%" echo quit
echo Running %SECS% s of simulated board time ...
"%RENODE%" --console --disable-gui "%SCRIPT%" > build\renode_log.txt 2>&1
if not exist build\renode_uart.txt ( echo No serial output - see build\renode_log.txt & exit /b 1 )
type build\renode_uart.txt
exit /b

:no_renode
echo Renode not found. Get the portable zip "renode-1.17.0.windows-portable.zip" from
echo https://github.com/renode/renode/releases and unpack it into rlpid_step1\tools\renode\
exit /b 1
