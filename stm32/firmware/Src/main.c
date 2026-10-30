/* main.c : the RL-tuned PID on the STM32F407G-DISC1 (STM32F4 Discovery, Cortex-M4F at 168 MHz)
 * -----------------------------------------------------------------------------
 * 1. Step test on the chip: the same test as the Python evaluate.py, printed next to the
 *    numbers Python measured (from agent_params.h).
 * 2. Live control at real time (one sample every RLPID_DT seconds). The trained network picks
 *    the PID gains every RLPID_ACTION_REPEAT samples; the PID controls the process.
 *    The process is simulated on the chip (RLPID_USE_REAL_PLANT = 0) or a real one is connected
 *    through the DAC and ADC (RLPID_USE_REAL_PLANT = 1, see README.md).
 *
 * Board connections:
 *   USART2  PA2 = TX, PA3 = RX, 115200 baud  (the ST-LINK virtual COM port on the DISC1 board)
 *   LEDs    PD12 green  = blinks with every sample (running)
 *           PD13 orange = step test running
 *           PD14 red    = output ran away (the run restarts)
 *           PD15 blue   = output within 2 % of the setpoint
 *   Button  PA0 (blue USER button) = next setpoint
 *   Real process only: DAC PA4 = control input u, ADC PA1 = measured output y
 * -----------------------------------------------------------------------------
 */
#include "stm32f4xx_hal.h"
#include "rl_pid.h"
#include <math.h>
#include <stdarg.h>
#include <stdio.h>

#ifndef RLPID_USE_REAL_PLANT
#define RLPID_USE_REAL_PLANT 0
#endif

/* Real process only: how volts map to the controller's numbers (adapt to your circuit). */
#define REAL_Y_AT_0V     (-0.5f)                 /* ADC 0 V    -> output -0.5 */
#define REAL_Y_AT_3V3    (2.5f)                  /* ADC 3.3 V  -> output  2.5 */

static UART_HandleTypeDef huart2;
#if RLPID_USE_REAL_PLANT
static ADC_HandleTypeDef hadc1;
static DAC_HandleTypeDef hdac;
#endif

/* ===== small helpers ===== */
static void say(const char *fmt, ...)
{
    char buf[200];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    if (n > (int)sizeof buf - 1) n = (int)sizeof buf - 1;
    if (n > 0) HAL_UART_Transmit(&huart2, (uint8_t *)buf, (uint16_t)n, 100);
}

static void led(uint16_t pin, int on) { HAL_GPIO_WritePin(GPIOD, pin, on ? GPIO_PIN_SET : GPIO_PIN_RESET); }

static void fatal(void)                          /* setup failed: blink the red LED forever */
{
    for (;;) { HAL_GPIO_TogglePin(GPIOD, GPIO_PIN_14); HAL_Delay(100); }
}

/* ===== setup ===== */
static void clock_168mhz(void)                   /* 8 MHz crystal -> PLL -> 168 MHz */
{
    RCC_OscInitTypeDef osc = {0};
    RCC_ClkInitTypeDef clk = {0};
    __HAL_RCC_PWR_CLK_ENABLE();
    __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);
    osc.OscillatorType = RCC_OSCILLATORTYPE_HSE;
    osc.HSEState = RCC_HSE_ON;
    osc.PLL.PLLState = RCC_PLL_ON;
    osc.PLL.PLLSource = RCC_PLLSOURCE_HSE;
    osc.PLL.PLLM = 8;
    osc.PLL.PLLN = 336;
    osc.PLL.PLLP = RCC_PLLP_DIV2;
    osc.PLL.PLLQ = 7;
    if (HAL_RCC_OscConfig(&osc) != HAL_OK) fatal();
    clk.ClockType = RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_SYSCLK | RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2;
    clk.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
    clk.AHBCLKDivider = RCC_SYSCLK_DIV1;
    clk.APB1CLKDivider = RCC_HCLK_DIV4;
    clk.APB2CLKDivider = RCC_HCLK_DIV2;
    if (HAL_RCC_ClockConfig(&clk, FLASH_LATENCY_5) != HAL_OK) fatal();
}

static void gpio_init(void)                      /* LEDs PD12-PD15, USER button PA0 */
{
    GPIO_InitTypeDef g = {0};
    __HAL_RCC_GPIOA_CLK_ENABLE();
    __HAL_RCC_GPIOD_CLK_ENABLE();
    g.Pin = GPIO_PIN_12 | GPIO_PIN_13 | GPIO_PIN_14 | GPIO_PIN_15;
    g.Mode = GPIO_MODE_OUTPUT_PP;
    g.Speed = GPIO_SPEED_FREQ_LOW;
    HAL_GPIO_Init(GPIOD, &g);
    g.Pin = GPIO_PIN_0;
    g.Mode = GPIO_MODE_INPUT;
    g.Pull = GPIO_NOPULL;                        /* the board has its own pull-down */
    HAL_GPIO_Init(GPIOA, &g);
}

static void uart2_init(void)                     /* PA2 TX, PA3 RX, 115200 8N1 */
{
    GPIO_InitTypeDef g = {0};
    __HAL_RCC_USART2_CLK_ENABLE();
    g.Pin = GPIO_PIN_2 | GPIO_PIN_3;
    g.Mode = GPIO_MODE_AF_PP;
    g.Pull = GPIO_PULLUP;
    g.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
    g.Alternate = GPIO_AF7_USART2;
    HAL_GPIO_Init(GPIOA, &g);
    huart2.Instance = USART2;
    huart2.Init.BaudRate = 115200;
    huart2.Init.WordLength = UART_WORDLENGTH_8B;
    huart2.Init.StopBits = UART_STOPBITS_1;
    huart2.Init.Parity = UART_PARITY_NONE;
    huart2.Init.Mode = UART_MODE_TX_RX;
    huart2.Init.HwFlowCtl = UART_HWCONTROL_NONE;
    huart2.Init.OverSampling = UART_OVERSAMPLING_16;
    if (HAL_UART_Init(&huart2) != HAL_OK) fatal();
}

static void cycle_counter_init(void)             /* DWT cycle counter: measures the network's time */
{
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}

#if RLPID_USE_REAL_PLANT
static void real_plant_init(void)                /* DAC1 channel 1 on PA4, ADC1 channel 1 on PA1 */
{
    GPIO_InitTypeDef g = {0};
    ADC_ChannelConfTypeDef ch = {0};
    DAC_ChannelConfTypeDef dch = {0};
    __HAL_RCC_DAC_CLK_ENABLE();
    __HAL_RCC_ADC1_CLK_ENABLE();
    g.Pin = GPIO_PIN_1 | GPIO_PIN_4;
    g.Mode = GPIO_MODE_ANALOG;
    HAL_GPIO_Init(GPIOA, &g);
    hdac.Instance = DAC;
    if (HAL_DAC_Init(&hdac) != HAL_OK) fatal();
    dch.DAC_Trigger = DAC_TRIGGER_NONE;
    dch.DAC_OutputBuffer = DAC_OUTPUTBUFFER_ENABLE;
    if (HAL_DAC_ConfigChannel(&hdac, &dch, DAC_CHANNEL_1) != HAL_OK) fatal();
    HAL_DAC_Start(&hdac, DAC_CHANNEL_1);
    hadc1.Instance = ADC1;
    hadc1.Init.ClockPrescaler = ADC_CLOCK_SYNC_PCLK_DIV4;
    hadc1.Init.Resolution = ADC_RESOLUTION_12B;
    hadc1.Init.DataAlign = ADC_DATAALIGN_RIGHT;
    hadc1.Init.NbrOfConversion = 1;
    hadc1.Init.ExternalTrigConv = ADC_SOFTWARE_START;
    if (HAL_ADC_Init(&hadc1) != HAL_OK) fatal();
    ch.Channel = ADC_CHANNEL_1;
    ch.Rank = 1;
    ch.SamplingTime = ADC_SAMPLETIME_84CYCLES;
    if (HAL_ADC_ConfigChannel(&hadc1, &ch) != HAL_OK) fatal();
}

static float real_plant_read(void)               /* ADC volts -> output y */
{
    HAL_ADC_Start(&hadc1);
    HAL_ADC_PollForConversion(&hadc1, 10);
    float v = (float)HAL_ADC_GetValue(&hadc1) / 4095.0f;
    return REAL_Y_AT_0V + v * (REAL_Y_AT_3V3 - REAL_Y_AT_0V);
}

static void real_plant_write(float u)            /* control input -U_LIMIT..+U_LIMIT -> DAC 0..3.3 V */
{
    float v = (u + RLPID_U_LIMIT) / (2.0f * RLPID_U_LIMIT);
    HAL_DAC_SetValue(&hdac, DAC_CHANNEL_1, DAC_ALIGN_12B_R, (uint32_t)(v * 4095.0f + 0.5f));
}
#endif

/* ===== interrupts ===== */
void SysTick_Handler(void) { HAL_IncTick(); }    /* 1 ms tick for HAL_GetTick / HAL_Delay */

/* ===== the program ===== */
int main(void)
{
    static const float setpoints[] = {RLPID_TEST_SETPOINT,
                                      RLPID_START_SETPOINT + 0.5f * (RLPID_TEST_SETPOINT - RLPID_START_SETPOINT),
                                      RLPID_START_SETPOINT + 1.5f * (RLPID_TEST_SETPOINT - RLPID_START_SETPOINT),
                                      RLPID_START_SETPOINT};
    const int n_setpoints = (int)(sizeof setpoints / sizeof setpoints[0]);
    rlpid_controller_t ctrl;
    rlpid_plant_t plant;
    rlpid_metrics_t m;
    uint32_t t0, cycles, next_tick;
    int which = 0, button_was = 0;
    float r, y, u, obs[3] = {0}, action[AGENT_N_ACTIONS];

    HAL_Init();
    gpio_init();
    clock_168mhz();
    uart2_init();
    cycle_counter_init();
#if RLPID_USE_REAL_PLANT
    real_plant_init();
#endif

    say("\r\n=== RL-tuned PID on STM32F407 (%lu MHz) ===\r\n", HAL_RCC_GetSysClockFreq() / 1000000UL);
    say("agent %s - %s\r\n", AGENT_NAME, AGENT_CASE_NAME);

    /* 1. step test on the chip */
    led(GPIO_PIN_13, 1);
    m = rlpid_step_test(NULL);
    led(GPIO_PIN_13, 0);
    say("step test on the chip: overshoot %.1f %%  settling %.2f s  ISE %.3f  IAE %.3f  completed %s\r\n",
        m.overshoot, m.settling, m.ise, m.iae, m.completed ? "yes" : "no");
    say("measured in Python   : overshoot %.1f %%  settling %.2f s  ISE %.3f  IAE %.3f  completed %s\r\n",
        EXPECTED_OVERSHOOT, EXPECTED_SETTLING, EXPECTED_ISE, EXPECTED_IAE, EXPECTED_COMPLETED ? "yes" : "no");

    /* time of one network decision, in CPU cycles */
    t0 = DWT->CYCCNT;
    rlpid_policy(obs, action);
    cycles = DWT->CYCCNT - t0;
    say("one network decision: %lu cycles = %.1f us\r\n", cycles,
        (double)cycles * 1e6 / (double)HAL_RCC_GetSysClockFreq());

    /* 2. live control, one sample every RLPID_DT_MS milliseconds */
    say("live control (%s process), blue button = next setpoint\r\n",
        RLPID_USE_REAL_PLANT ? "real" : "simulated");
    say("t_s,setpoint,output,control_input,Kp,tau_i,tau_d\r\n");
    rlpid_plant_reset(&plant);
    y = plant.y;
    rlpid_controller_reset(&ctrl, y);
    r = setpoints[0];
    next_tick = HAL_GetTick();
    for (uint32_t k = 0;; k++) {
        while ((int32_t)(HAL_GetTick() - next_tick) < 0) {
            int pressed = HAL_GPIO_ReadPin(GPIOA, GPIO_PIN_0) == GPIO_PIN_SET;
            if (pressed && !button_was) { which = (which + 1) % n_setpoints; r = setpoints[which]; }
            button_was = pressed;
        }
        next_tick += RLPID_DT_MS;

#if RLPID_USE_REAL_PLANT
        y = real_plant_read();
        u = rlpid_controller_step(&ctrl, r, y);
        real_plant_write(u);
#else
        u = rlpid_controller_step(&ctrl, r, y);
        y = rlpid_plant_step(&plant, u);
        if (fabsf(y) > RLPID_Y_LIMIT) {           /* ran away: restart the run */
            led(GPIO_PIN_14, 1);
            rlpid_plant_reset(&plant);
            y = plant.y;
            rlpid_controller_reset(&ctrl, y);
        }
#endif
        led(GPIO_PIN_12, (int)(k & 1u));
        led(GPIO_PIN_15, fabsf(r - y) <= 0.02f * fabsf(RLPID_TEST_SETPOINT - RLPID_START_SETPOINT));
        say("%.1f,%.3f,%.4f,%.3f,%.3f,%.3f,%.3f\r\n", (double)k * RLPID_DT, r, y, u,
            ctrl.gains.Kp, ctrl.gains.tau_i, ctrl.gains.tau_d);
    }
}
