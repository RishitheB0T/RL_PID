/* hil.c : the hardware-in-the-loop protocol (see hil.h) - plain C, no hardware code */
#include "hil.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Reads one number after position *p; returns 1 if there was one. */
static int read_number(const char **p, float *value)
{
    char *end;
    *value = strtof(*p, &end);
    if (end == *p) return 0;
    *p = end;
    return 1;
}

void hil_handle_line(const char *line, char *reply, int reply_size, rlpid_controller_t *ctrl,
                     uint32_t (*cycles)(void))
{
    const char *p = line + 1;
    float r, y;

    switch (line[0]) {
    case 'I':                                     /* which agent and settings */
        snprintf(reply, reply_size, "I %s %.9g %.9g %.9g %d", AGENT_NAME, (double)RLPID_DT,
                 (double)RLPID_U_LIMIT, (double)RLPID_Y_LIMIT, RLPID_ACTION_REPEAT);
        return;
    case 'Z':                                     /* reset the controller */
        if (!read_number(&p, &y)) break;
        rlpid_controller_reset(ctrl, y);
        snprintf(reply, reply_size, "Z OK");
        return;
    case 'S': {                                   /* one sample: r, y in -> u out */
        uint32_t t0, used;
        float u;
        if (!read_number(&p, &r) || !read_number(&p, &y)) break;
        t0 = cycles ? cycles() : 0u;
        u = rlpid_controller_step(ctrl, r, y);
        used = cycles ? cycles() - t0 : 0u;
        snprintf(reply, reply_size, "U %.9g %.9g %.9g %.9g %lu", (double)u, (double)ctrl->gains.Kp,
                 (double)ctrl->gains.tau_i, (double)ctrl->gains.tau_d, (unsigned long)used);
        return;
    }
    default:
        break;
    }
    snprintf(reply, reply_size, "E cannot read '%.40s' (send I, Z <y0> or S <r> <y>)", line);
}
