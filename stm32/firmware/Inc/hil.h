/* hil.h : hardware-in-the-loop protocol - the STM32 is the controller, the process runs on the PC
 * -----------------------------------------------------------------------------
 * One text line per message, ending with '\n'. The PC asks, the controller answers:
 *
 *   PC sends            controller answers
 *   I                   I <agent> <dt> <u_limit> <y_limit> <action_repeat>     (which agent is on it)
 *   Z <y0>              Z OK                    reset: empty integral, previous output = y0
 *   S <r> <y>           U <u> <Kp> <tau_i> <tau_d> <cycles>
 *                       one sample: setpoint r and measured output y in; control input u and
 *                       the gains the agent is using out; cycles = CPU cycles of this step (0 on a PC)
 *   anything else       E <message>
 *
 * The controller knows nothing about the process: it only sees r and y, exactly as with a real plant.
 * The same code runs on the STM32 (main.c, build.bat pc) and on a PC (host_test/host_hil.c).
 * -----------------------------------------------------------------------------
 */
#ifndef HIL_H
#define HIL_H

#include <stdint.h>
#include "rl_pid.h"

/* Handles one received line (without the '\n'). Writes the answer (without '\n') into reply.
 * cycles: a function returning the CPU cycle counter, or NULL (then 0 is reported). */
void hil_handle_line(const char *line, char *reply, int reply_size, rlpid_controller_t *ctrl,
                     uint32_t (*cycles)(void));

#endif /* HIL_H */
