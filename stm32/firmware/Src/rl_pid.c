/* rl_pid.c : the trained RL-tuned PID controller in plain C - see rl_pid.h
 * -----------------------------------------------------------------------------
 * Every function mirrors one piece of the Python project:
 *   rlpid_policy            stable-baselines3 MlpPolicy (2 x 64, tanh) + VecNormalize scaling
 *   rlpid_action_to_gains   core/versions.py  Version.to_gains
 *   rlpid_controller_step   core/env.py       step_gains (agent every ACTION_REPEAT samples)
 *                           core/pid.py       PID.control
 *   rlpid_plant_step        cases/csN.py      Process.step
 *   rlpid_step_test         core/tools.py     run_test + score
 * -----------------------------------------------------------------------------
 */
#define RLPID_DEFINE_WEIGHTS          /* agent_params.h then also defines the weight arrays */
#include "rl_pid.h"
#include <math.h>
#include <stddef.h>

static float clampf(float x, float lo, float hi) { return x < lo ? lo : (x > hi ? hi : x); }

/* ===== the neural network ===== */
void rlpid_policy(const float obs[3], float action[AGENT_N_ACTIONS])
{
    float x[3], h1[AGENT_HIDDEN], h2[AGENT_HIDDEN];
    int i, j;

    /* input scaling (VecNormalize): (obs - mean) / sqrt(var + eps), clipped to +-clip.
     * Done in double, exactly like numpy, then rounded to float like PyTorch. */
    for (i = 0; i < 3; i++) {
        double z = ((double)obs[i] - OBS_MEAN[i]) / sqrt(OBS_VAR[i] + OBS_EPS);
        if (z > OBS_CLIP) z = OBS_CLIP;
        if (z < -OBS_CLIP) z = -OBS_CLIP;
        x[i] = (float)z;
    }
    /* hidden layer 1: tanh(W1 x + b1) */
    for (i = 0; i < AGENT_HIDDEN; i++) {
        float s = B1[i];
        for (j = 0; j < 3; j++) s += W1[i][j] * x[j];
        h1[i] = tanhf(s);
    }
    /* hidden layer 2: tanh(W2 h1 + b2) */
    for (i = 0; i < AGENT_HIDDEN; i++) {
        float s = B2[i];
        for (j = 0; j < AGENT_HIDDEN; j++) s += W2[i][j] * h1[j];
        h2[i] = tanhf(s);
    }
    /* output layer (the mean action), clipped to the slider range -1..+1 */
    for (i = 0; i < AGENT_N_ACTIONS; i++) {
        float s = B3[i];
        for (j = 0; j < AGENT_HIDDEN; j++) s += W3[i][j] * h2[j];
        action[i] = clampf(s, -1.0f, 1.0f);
    }
}

/* ===== sliders -> gains ===== */
rlpid_gains_t rlpid_action_to_gains(const float action[AGENT_N_ACTIONS])
{
    rlpid_gains_t g;
    float a[3];
    int i;
    for (i = 0; i < 3; i++) a[i] = clampf(action[i], -1.0f, 1.0f);
    g.Kp    = GAIN_LOW[0] + (a[0] + 1.0f) / 2.0f * (GAIN_HIGH[0] - GAIN_LOW[0]);
    g.tau_i = GAIN_LOW[1] + (a[1] + 1.0f) / 2.0f * (GAIN_HIGH[1] - GAIN_LOW[1]);
    g.tau_d = GAIN_LOW[2] + (a[2] + 1.0f) / 2.0f * (GAIN_HIGH[2] - GAIN_LOW[2]);
#if AGENT_WEIGHT_B
    g.b = (clampf(action[3], -1.0f, 1.0f) + 1.0f) / 2.0f;   /* setpoint weight (improved_v3) */
#else
    g.b = 1.0f;
#endif
    return g;
}

/* ===== the controller ===== */
void rlpid_controller_reset(rlpid_controller_t *c, float y0)
{
    c->pid.integral = 0.0f;
    c->pid.has_prev = 0;
    c->pid.y_prev = 0.0f;
    c->y_prev = y0;
    c->sample = 0;
    c->gains.Kp = c->gains.tau_i = c->gains.tau_d = 0.0f;
    c->gains.b = 1.0f;
}

float rlpid_controller_step(rlpid_controller_t *c, float r, float y)
{
    float Ki, Kd, e, dy, u;

    /* 1. every ACTION_REPEAT samples the agent chooses new gains from what it sees */
    if (c->sample % RLPID_ACTION_REPEAT == 0) {
        float obs[3], action[AGENT_N_ACTIONS];
        obs[0] = r;
        obs[1] = y;
        obs[2] = c->y_prev;
        rlpid_policy(obs, action);
        c->gains = rlpid_action_to_gains(action);
    }
    c->sample++;
    c->y_prev = y;

    /* 2. the PID: P on the weighted error, I with anti-windup, D on the measurement */
    Ki = c->gains.Kp / (c->gains.tau_i + 0.01f);
    Kd = c->gains.Kp * c->gains.tau_d;
    e = r - y;
    dy = c->pid.has_prev ? y - c->pid.y_prev : 0.0f;
    c->pid.integral = clampf(c->pid.integral + Ki * e * RLPID_DT, -RLPID_U_LIMIT, RLPID_U_LIMIT);
    u = c->gains.Kp * (c->gains.b * r - y) + c->pid.integral - Kd * dy / RLPID_DT;
    c->pid.y_prev = y;
    c->pid.has_prev = 1;
    return clampf(u, -RLPID_U_LIMIT, RLPID_U_LIMIT);
}

/* ===== the simulated process ===== */
void rlpid_plant_reset(rlpid_plant_t *p)
{
    int i;
    p->y = 0.0f;
    p->head = 0;
    for (i = 0; i < RLPID_MAX_DELAY; i++) p->pipe[i] = 0.0f;
}

float rlpid_plant_step(rlpid_plant_t *p, float u)
{
    float u_delayed = u;
#if PLANT_DELAY > 0
    u_delayed = p->pipe[p->head];                  /* the input from PLANT_DELAY samples ago */
    p->pipe[p->head] = u;                          /* the new input joins the delay line */
    p->head = (p->head + 1) % PLANT_DELAY;
#endif
    p->y = PLANT_A * p->y + PLANT_B * u_delayed;
    return p->y;
}

void rlpid_plant_bump(rlpid_plant_t *p, float amount) { p->y += amount; }

/* ===== the step test ===== */
rlpid_metrics_t rlpid_step_test(float *y_log)
{
    static float r[RLPID_TEST_LENGTH], y[RLPID_TEST_LENGTH];
    rlpid_controller_t c;
    rlpid_plant_t p;
    rlpid_metrics_t m;
    const float size = RLPID_TEST_SETPOINT - RLPID_START_SETPOINT;
    float peak, band;
    int k, last_outside = -1;

    rlpid_plant_reset(&p);
    for (k = 0; k < RLPID_TEST_LENGTH; k++) {
        r[k] = (k < RLPID_WARMUP_STEPS) ? RLPID_START_SETPOINT : RLPID_TEST_SETPOINT;
        y[k] = p.y;
    }
    k = RLPID_WARMUP_STEPS - 1;                    /* "now" = the last warm-up sample */
    rlpid_controller_reset(&c, y[k - 1]);
    m.completed = 1;
    for (;;) {
        float u = rlpid_controller_step(&c, r[k], y[k]);
        y[k + 1] = rlpid_plant_step(&p, u);
        k++;
        if (fabsf(y[k]) > RLPID_Y_LIMIT) { m.completed = 0; break; }   /* ran away */
        if (k >= RLPID_TEST_LENGTH - 1) break;                          /* end of the test */
    }
    m.samples = k;

    /* the four metrics, from the setpoint step on */
    band = 0.02f * fabsf(size);
    peak = y[RLPID_WARMUP_STEPS];
    m.ise = m.iae = 0.0f;
    for (int i = RLPID_WARMUP_STEPS; i <= k; i++) {
        float e = r[i] - y[i];
        if (fabsf(e) > band) last_outside = i - RLPID_WARMUP_STEPS;
        if ((size > 0 && y[i] > peak) || (size < 0 && y[i] < peak)) peak = y[i];
        m.ise += e * e * RLPID_DT;
        m.iae += fabsf(e) * RLPID_DT;
    }
    m.overshoot = (peak - RLPID_TEST_SETPOINT) / size * 100.0f;
    if (m.overshoot < 0.0f) m.overshoot = 0.0f;
    m.settling = (last_outside >= 0) ? (float)(last_outside + 1) * RLPID_DT : 0.0f;

    if (y_log != NULL)
        for (int i = 0; i < RLPID_TEST_LENGTH; i++) y_log[i] = (i <= k) ? y[i] : y[k];
    return m;
}
