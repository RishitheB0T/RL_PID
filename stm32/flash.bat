@echo off
rem flash.bat : writes the built firmware to the STM32F4 Discovery board over its ST-LINK (USB)
rem Run build.bat first. Uses STM32CubeProgrammer, which comes with STM32CubeIDE.
setlocal
cd /d "%~dp0"
set "PROG="
for /d %%D in ("C:\ST\STM32CubeIDE_*") do (
    for /d %%P in ("%%D\STM32CubeIDE\plugins\com.st.stm32cube.ide.mcu.externaltools.cubeprogrammer*") do set "PROG=%%P\tools\bin\STM32_Programmer_CLI.exe"
)
if not defined PROG ( echo STM32CubeProgrammer not found - install STM32CubeIDE in C:\ST & exit /b 1 )
if not exist build\firmware\rlpid.bin ( echo Run build.bat first & exit /b 1 )
"%PROG%" -c port=SWD -w build\firmware\rlpid.bin 0x08000000 -v -rst
