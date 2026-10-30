/* host_main.c : runs the firmware's controller code on a PC (no STM32 needed)
 * -----------------------------------------------------------------------------
 * Built and run by stm32/verify_on_pc.py with the PC's gcc. It runs the same step test as the
 * board and prints the metrics and every output sample, so Python can compare them.
 * -----------------------------------------------------------------------------
 */
#include <stdio.h>
#include "rl_pid.h"

int main(void)
{
    static float y[RLPID_TEST_LENGTH];
    rlpid_metrics_t m = rlpid_step_test(y);
    int k;

    printf("agent %s\n", AGENT_NAME);
    printf("metrics %.6f %.6f %.6f %.6f %d\n", m.overshoot, m.settling, m.ise, m.iae, m.completed);
    for (k = 0; k < RLPID_TEST_LENGTH; k++) printf("y %d %.9g\n", k, y[k]);
    return 0;
}
