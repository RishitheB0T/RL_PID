/* host_hil.c : the firmware's controller + hil protocol, on a PC (no STM32 needed)
 * -----------------------------------------------------------------------------
 * Reads protocol lines on standard input and answers on standard output, exactly like the board
 * in "pc" mode does over its serial port. Used by stm32/pc_plant.py --sim to test the whole
 * loop (process on the PC <-> controller) without a board.
 * -----------------------------------------------------------------------------
 */
#include <stdio.h>
#include <string.h>
#include "hil.h"

int main(void)
{
    static char line[200], reply[200];
    rlpid_controller_t ctrl;

    rlpid_controller_reset(&ctrl, 0.0f);
    printf("READY\n");
    fflush(stdout);
    while (fgets(line, sizeof line, stdin) != NULL) {
        line[strcspn(line, "\r\n")] = '\0';
        if (line[0] == '\0') continue;
        hil_handle_line(line, reply, sizeof reply, &ctrl, NULL);
        printf("%s\n", reply);
        fflush(stdout);
    }
    return 0;
}
