/* stm32f4xx_hal_conf.h : which parts of ST's HAL library the firmware uses
 * (a trimmed version of ST's stm32f4xx_hal_conf_template.h for the STM32F4 Discovery board)
 */
#ifndef STM32F4XX_HAL_CONF_H
#define STM32F4XX_HAL_CONF_H

/* ----- HAL modules used ----- */
#define HAL_MODULE_ENABLED
#define HAL_RCC_MODULE_ENABLED
#define HAL_GPIO_MODULE_ENABLED
#define HAL_DMA_MODULE_ENABLED
#define HAL_CORTEX_MODULE_ENABLED
#define HAL_FLASH_MODULE_ENABLED
#define HAL_PWR_MODULE_ENABLED
#define HAL_UART_MODULE_ENABLED
#define HAL_ADC_MODULE_ENABLED     /* only used with a real process (RLPID_USE_REAL_PLANT = 1) */
#define HAL_DAC_MODULE_ENABLED     /* only used with a real process (RLPID_USE_REAL_PLANT = 1) */

/* ----- clocks: the Discovery board has an 8 MHz crystal ----- */
#define HSE_VALUE              8000000U
#define HSE_STARTUP_TIMEOUT    100U
#define HSI_VALUE              16000000U
#define LSI_VALUE              32000U
#define LSE_VALUE              32768U
#define LSE_STARTUP_TIMEOUT    5000U
#define EXTERNAL_CLOCK_VALUE   12288000U

/* ----- system ----- */
#define VDD_VALUE                  3300U
#define TICK_INT_PRIORITY          0x0FU
#define USE_RTOS                   0U
#define PREFETCH_ENABLE            1U
#define INSTRUCTION_CACHE_ENABLE   1U
#define DATA_CACHE_ENABLE          1U
#define USE_HAL_ADC_REGISTER_CALLBACKS   0U
#define USE_HAL_DAC_REGISTER_CALLBACKS   0U
#define USE_HAL_UART_REGISTER_CALLBACKS  0U
#define USE_SPI_CRC                0U
#define assert_param(expr)         ((void)0U)

/* ----- module headers ----- */
#include "stm32f4xx_hal_rcc.h"
#include "stm32f4xx_hal_gpio.h"
#include "stm32f4xx_hal_dma.h"
#include "stm32f4xx_hal_cortex.h"
#include "stm32f4xx_hal_flash.h"
#include "stm32f4xx_hal_pwr.h"
#include "stm32f4xx_hal_uart.h"
#include "stm32f4xx_hal_adc.h"
#include "stm32f4xx_hal_dac.h"

#endif /* STM32F4XX_HAL_CONF_H */
