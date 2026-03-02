i have connected to it using ssh.  the orin talks to the real x3plus master board through pyserial. all the details are defined in the @.py_install_dl/py_install/Rosmaster_Lib/Rosmaster_Lib.py . now, do

## Fix: Controller update_rate (100→20 Hz) and servo run_time (2000→60 ms)

### Problem
The robot arm was not reaching trajectory waypoints accurately. When sending a
multi-point trajectory (e.g. joint1 to pi/2 then to -pi/2), the arm would fall
short of the first waypoint before moving to the second.

### Root cause
The joint_trajectory_controller is purely time-based. It samples a spline at
each control tick and forwards the commanded position to hardware. It does NOT
wait for the robot to physically arrive at a waypoint before moving on to the
next trajectory segment.

The real bottleneck was in lqtech_zmq_service.py: the servo command
`set_uart_servo_angle_array()` was called with `run_time=2000`, meaning each
position command told the servo "take 2 seconds to reach this angle." But the
controller sent new commands at 100 Hz (every 10 ms). So every 10 ms the servo
restarted a 2-second ramp to a slightly different target — it never finished
any move and was perpetually lagging behind.

### Fix: align all rates
- controller_manager update_rate: 100 → 20 Hz (one command every 50 ms)
- servo run_time: 2000 → 60 ms (servo motion duration per command)
- state_publish_rate: 50 → 20 Hz (match control rate)
- zmq_bridge_node state timer: 0.02s → 0.05s (20 Hz, match control rate)
- opus_x3plus_real_bridge servo_run_time_ms default: 50 → 60

### Why these values
- 20 Hz is appropriate for the x3plus hardware: single half-duplex UART at
  115200 baud, 2 ms delay per write, plus reads share the same bus.
- run_time=60 ms is slightly longer than the 50 ms command period. This creates
  a small overlap: the servo is still finishing its current move when the next
  command arrives, so it smoothly redirects without stopping. This avoids the
  accelerate-stop-accelerate-stop jerkiness that would occur if run_time exactly
  equaled the command period (50 ms).
- The rule: run_time ≈ 1.0-1.2x the command period for smooth streaming.

### Files changed
- opus_x3plus_controllers.yaml: update_rate, state_publish_rate
- lqtech_zmq_service.py: run_time in set_joint_position_array and set_joint_position_single
- zmq_bridge_node.py: state timer period
- opus_x3plus_real_bridge.py: servo_run_time_ms default
