@echo off
rem build.bat : builds the STM32F407 firmware - no IDE project needed
rem   .\build.bat          simulated process on the chip (default)
rem   .\build.bat real     real process through DAC (PA4) and ADC (PA1)
rem Uses the ARM compiler inside STM32CubeIDE and ST's STM32CubeF4 package (HAL, CMSIS, startup,
rem linker script). Run export_agent.py first to choose the agent (default: cs1 improved_v4n2_seed0).
setlocal enabledelayedexpansion
cd /d "%~dp0"

set "CUBE_F4=%USERPROFILE%\STM32Cube\Repository\STM32Cube_FW_F4_V1.28.3"
set "GCC_BIN="
for /d %%D in ("C:\ST\STM32CubeIDE_*") do (
    for /d %%P in ("%%D\STM32CubeIDE\plugins\com.st.stm32cube.ide.mcu.externaltools.gnu-tools-for-stm32*") do set "GCC_BIN=%%P\tools\bin"
)
if not defined GCC_BIN ( echo ARM compiler not found - install STM32CubeIDE in C:\ST & exit /b 1 )
if not exist "%CUBE_F4%\Drivers" ( echo STM32CubeF4 package not found at %CUBE_F4% & exit /b 1 )

set REAL=0
if /i "%1"=="real" set REAL=1
set "HAL=%CUBE_F4%\Drivers\STM32F4xx_HAL_Driver"
set "CMSIS=%CUBE_F4%\Drivers\CMSIS"
set "OUT=build\firmware"
if not exist "%OUT%" mkdir "%OUT%"

set "SRC=firmware\Src\main.c firmware\Src\rl_pid.c firmware\Src\syscalls.c "%CMSIS%\Device\ST\STM32F4xx\Source\Templates\system_stm32f4xx.c" "%CMSIS%\Device\ST\STM32F4xx\Source\Templates\gcc\startup_stm32f407xx.s""
for %%M in (hal hal_cortex hal_rcc hal_rcc_ex hal_gpio hal_dma hal_uart hal_pwr hal_pwr_ex hal_flash hal_flash_ex hal_adc hal_dac) do set "SRC=!SRC! "%HAL%\Src\stm32f4xx_%%M.c""

set "CPU=-mcpu=cortex-m4 -mthumb -mfpu=fpv4-sp-d16 -mfloat-abi=hard"
set "INC=-Ifirmware\Inc -I"%HAL%\Inc" -I"%CMSIS%\Device\ST\STM32F4xx\Include" -I"%CMSIS%\Include""
set "LD=%CUBE_F4%\Projects\STM32F4-Discovery\Templates\STM32CubeIDE\STM32F407VGTX_FLASH.ld"

echo Building the firmware (real process: %REAL%) ...
"%GCC_BIN%\arm-none-eabi-gcc.exe" %CPU% -DSTM32F407xx -DUSE_HAL_DRIVER -DRLPID_USE_REAL_PLANT=%REAL% %INC% ^
    -O2 -std=gnu11 -Wall -ffunction-sections -fdata-sections %SRC% ^
    -T"%LD%" --specs=nano.specs --specs=nosys.specs -u _printf_float -Wl,--gc-sections ^
    -Wl,-Map=%OUT%\rlpid.map -lm -o %OUT%\rlpid.elf
if errorlevel 1 ( echo BUILD FAILED & exit /b 1 )
"%GCC_BIN%\arm-none-eabi-objcopy.exe" -O binary %OUT%\rlpid.elf %OUT%\rlpid.bin
"%GCC_BIN%\arm-none-eabi-size.exe" %OUT%\rlpid.elf
echo Built %OUT%\rlpid.elf and %OUT%\rlpid.bin
