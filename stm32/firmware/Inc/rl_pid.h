/* rl_pid.h : the trained RL-tuned PID controller in plain C (no hardware code)
 * -----------------------------------------------------------------------------
 * The same calculations as the Python project (core/env.py, core/pid.py, core/versions.py):
 *   - every RLPID_ACTION_REPEAT samples the neural network looks at
 *     (setpoint, output now, output one sample ago) and chooses the PID gains;
 *   - every sample the PID computes the control input u.
 * The network weights and all settings come from agent_params.h, which is written by
 * stm32/export_agent.py. The same file compiles on the STM32 (Cortex-M4F, single-precision
 * FPU) and on a PC (stm32/host_test), so the PC can check it against Python.
 * -----------------------------------------------------------------------------
 */
#ifndef RL_PID_H
#define RL_PID_H

#include "agent_params.h"

#define RLPID_MAX_DELAY 32                 /* longest process dead time, in samples */
#define RLPID_TEST_LENGTH (RLPID_WARMUP_STEPS + RLPID_HOLD_STEPS)   /* samples in the step test */

typedef struct {            /* PID gains chosen by the agent */
    float Kp, tau_i, tau_d, b;
} rlpid_gains_t;

typedef struct {            /* memory of the PID controller */
    float integral;
    float y_prev;
    int   has_prev;
} rlpid_pid_t;

typedef struct {            /* the controller: agent + PID */
    rlpid_pid_t   pid;
    rlpid_gains_t gains;
    float         y_prev;   /* output one sample ago (part of what the agent sees) */
    unsigned int  sample;   /* samples since the start; the agent decides every RLPID_ACTION_REPEAT */
} rlpid_controller_t;

typedef struct {            /* simulated process: y[k+1] = A*y[k] + B*u[k - delay] */
    float y;
    float pipe[RLPID_MAX_DELAY];
    int   head;
} rlpid_plant_t;

typedef struct {            /* step-test results, as in the Python evaluate.py */
    float overshoot;        /* % */
    float settling;         /* s, 2 % band */
    float ise;
    float iae;
    int   completed;        /* 1 = reached the end, 0 = output ran away */
    int   samples;          /* index of the last simulated sample */
} rlpid_metrics_t;

/* The neural network: what the agent sees -> sliders in -1..+1 (input scaling included). */
void rlpid_policy(const float obs[3], float action[AGENT_N_ACTIONS]);

/* Sliders -> PID gains, using the version's gain ranges. */
rlpid_gains_t rlpid_action_to_gains(const float action[AGENT_N_ACTIONS]);

/* The controller: call rlpid_controller_step once per sample with the setpoint r and the
 * measured output y; it returns the control input u. */
void  rlpid_controller_reset(rlpid_controller_t *c, float y0);
float rlpid_controller_step(rlpid_controller_t *c, float r, float y);

/* The simulated process (used when no real process is connected). */
void  rlpid_plant_reset(rlpid_plant_t *p);
float rlpid_plant_step(rlpid_plant_t *p, float u);
void  rlpid_plant_bump(rlpid_plant_t *p, float amount);

/* The step test of evaluate.py: setpoint START -> TEST, scored with the same four metrics.
 * y_log (optional, may be NULL) receives the output of every sample (RLPID_TEST_LENGTH values). */
rlpid_metrics_t rlpid_step_test(float *y_log);

#endif /* RL_PID_H */
