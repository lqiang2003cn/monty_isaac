#!/usr/bin/env python3
# coding: utf-8
"""
Slim serial driver for the x3plus arm servos only.

Replaces the 970-line Rosmaster car+arm driver with ~180 lines that implement
only the functions needed for arm control over ZMQ:

  - open / close serial
  - read all 6 joint angles  (FUNC_ARM_CTRL request)
  - write all 6 joint angles (FUNC_ARM_CTRL command)
  - write single servo angle (FUNC_UART_SERVO command)
  - enable / disable servo torque
  - read battery voltage      (on-demand, from auto-report cache)
  - disable MCU auto-report   (stops ~100 pkts/s of unused car telemetry)

Serial protocol is the same binary framing as the Rosmaster board firmware:
  TX: [0xFF] [DEVICE_ID] [length] [function] [data...] [checksum]
  RX: [0xFF] [DEVICE_ID-1] [length] [function] [data...] [checksum]

All timing values (delays, timeouts) match the original Rosmaster driver so
the MCU firmware sees identical traffic patterns.
"""

import struct
import threading
import time

import serial as pyserial

_HEAD = 0xFF
_DEVICE_ID = 0xFC
_COMPLEMENT = 257 - _DEVICE_ID

FUNC_AUTO_REPORT = 0x01
FUNC_UART_SERVO = 0x20
FUNC_UART_SERVO_TORQUE = 0x22
FUNC_ARM_CTRL = 0x23
FUNC_REQUEST_DATA = 0x50
FUNC_VERSION = 0x51

# Only parse these function codes; silently discard everything else.
_ARM_PARSE_SET = frozenset((
    FUNC_ARM_CTRL,
    FUNC_UART_SERVO,
    FUNC_VERSION,
    0x0A,  # FUNC_REPORT_SPEED — only for battery voltage byte
))


def _checksum(cmd):
    return sum(cmd, _COMPLEMENT) & 0xFF


# ── angle ↔ pulse conversion (matches Rosmaster firmware) ────────────────

def _angle_to_pulse(servo_id, angle):
    """Degrees → pulse value for servos 1-6."""
    if 1 <= servo_id <= 4:
        return int((3100 - 900) * (angle - 180) / (0 - 180) + 900)
    if servo_id == 5:
        return int((3700 - 380) * (angle - 0) / (270 - 0) + 380)
    if servo_id == 6:
        return int((3100 - 900) * (angle - 0) / (180 - 0) + 900)
    return -1


def _pulse_to_angle(servo_id, pulse):
    """Pulse value → degrees for servos 1-6."""
    if 1 <= servo_id <= 4:
        return int((pulse - 900) * (0 - 180) / (3100 - 900) + 180 + 0.5)
    if servo_id == 5:
        return int((270 - 0) * (pulse - 380) / (3700 - 380) + 0 + 0.5)
    if servo_id == 6:
        return int((180 - 0) * (pulse - 900) / (3100 - 900) + 0 + 0.5)
    return -1


_ANGLE_RANGES = {1: (0, 180), 2: (0, 180), 3: (0, 180),
                 4: (0, 180), 5: (0, 270), 6: (0, 180)}

# Minimum plausible pulse values.  The firmware returns 0 when a servo doesn't
# respond on the bus.  Any pulse below these thresholds is treated as an error.
_MIN_PULSE = {1: 800, 2: 800, 3: 800, 4: 800, 5: 300, 6: 800}
_MAX_PULSE = {1: 3200, 2: 3200, 3: 3200, 4: 3200, 5: 3800, 6: 3200}


class ArmSerial:
    """Minimal serial interface for the x3plus 6-DOF arm."""

    def __init__(self, port="/dev/ttyUSB0", delay=0.002):
        self._ser = pyserial.Serial(port, 115200)
        self._delay = delay

        # Receive state
        self._read_arm = [-1] * 6
        self._read_arm_ok = False
        self._read_servo_id = 0
        self._read_servo_val = 0
        self._battery_raw = 0  # 0-255, divide by 10 for volts
        self._version_h = 0
        self._version_l = 0

        self._lock = threading.Lock()
        self._rx_thread = None

        # Enable torque so servo 6 is readable on first connect.
        self._send(FUNC_UART_SERVO_TORQUE, [1])
        time.sleep(delay)

    # ── low-level serial I/O ─────────────────────────────────────────────

    def _send(self, func, data_bytes=None):
        if data_bytes is None:
            data_bytes = []
        length = len(data_bytes) + 2  # func + data + checksum counted
        cmd = [_HEAD, _DEVICE_ID, length + 1, func] + data_bytes
        cmd.append(_checksum(cmd))
        self._ser.write(bytearray(cmd))
        time.sleep(self._delay)

    def _request(self, func, param=0):
        cmd = [_HEAD, _DEVICE_ID, 0x05, FUNC_REQUEST_DATA, func & 0xFF, param & 0xFF]
        cmd.append(_checksum(cmd))
        self._ser.write(bytearray(cmd))
        time.sleep(self._delay)

    def _rx_loop(self):
        self._ser.flushInput()
        while True:
            head1 = bytearray(self._ser.read())[0]
            if head1 != _HEAD:
                continue
            head2 = bytearray(self._ser.read())[0]
            if head2 != _DEVICE_ID - 1:
                continue
            ext_len = bytearray(self._ser.read())[0]
            ext_type = bytearray(self._ser.read())[0]

            data_len = ext_len - 2
            ext_data = []
            check_sum = ext_len + ext_type
            while len(ext_data) < data_len:
                val = bytearray(self._ser.read())[0]
                ext_data.append(val)
                if len(ext_data) < data_len:
                    check_sum += val
            if data_len > 0 and check_sum % 256 != ext_data[-1]:
                continue

            if ext_type not in _ARM_PARSE_SET:
                continue
            self._parse(ext_type, ext_data)

    def _parse(self, ext_type, data):
        if ext_type == FUNC_ARM_CTRL:
            with self._lock:
                for i in range(6):
                    self._read_arm[i] = struct.unpack('h', bytearray(data[i*2:i*2+2]))[0]
                self._read_arm_ok = True
        elif ext_type == FUNC_UART_SERVO:
            self._read_servo_id = struct.unpack('B', bytearray(data[0:1]))[0]
            self._read_servo_val = struct.unpack('h', bytearray(data[1:3]))[0]
        elif ext_type == FUNC_VERSION:
            self._version_h = struct.unpack('B', bytearray(data[0:1]))[0]
            self._version_l = struct.unpack('B', bytearray(data[1:2]))[0]
        elif ext_type == 0x0A:  # REPORT_SPEED — extract battery only
            if len(data) >= 7:
                self._battery_raw = struct.unpack('B', bytearray(data[6:7]))[0]

    # ── public API ───────────────────────────────────────────────────────

    def start(self):
        """Start the background receive thread."""
        if self._rx_thread is not None:
            return
        t = threading.Thread(target=self._rx_loop, daemon=True)
        t.start()
        self._rx_thread = t
        time.sleep(0.05)

    def close(self):
        if self._ser and self._ser.is_open:
            self._ser.close()

    # ── auto-report control ──────────────────────────────────────────────

    def set_auto_report(self, enable, forever=False):
        """Enable/disable MCU auto-report (IMU, encoders, speed, battery).

        When disabled, eliminates ~100 packets/sec of unused car telemetry.
        Battery voltage will no longer update passively; use get_battery_voltage()
        with an explicit request instead.
        """
        state = 1 if enable else 0
        perm = 0x5F if forever else 0
        self._send(FUNC_AUTO_REPORT, [state, perm])

    # ── arm joint read / write ───────────────────────────────────────────

    def get_angles(self):
        """Read all 6 servo angles (degrees). Returns list of 6 ints (-1 = error).

        Validates raw pulse values before conversion: pulses outside the
        plausible range (e.g. 0 when a servo doesn't respond) are returned
        as -1 rather than being converted to a bogus angle.
        """
        with self._lock:
            self._read_arm = [-1] * 6
            self._read_arm_ok = False

        self._request(FUNC_ARM_CTRL, 1)

        for _ in range(30):
            with self._lock:
                if self._read_arm_ok:
                    result = []
                    for i in range(6):
                        pulse = self._read_arm[i]
                        sid = i + 1
                        if pulse <= 0 or pulse < _MIN_PULSE[sid] or pulse > _MAX_PULSE[sid]:
                            result.append(-1)
                            continue
                        angle = _pulse_to_angle(sid, pulse)
                        lo, hi = _ANGLE_RANGES[sid]
                        if angle < lo or angle > hi:
                            result.append(-1)
                            continue
                        result.append(angle)
                    return result
            time.sleep(0.001)
        return [-1] * 6

    def set_angles(self, angles, run_time=500):
        """Set all 6 servo angles (degrees). run_time in ms [0, 2000]."""
        if len(angles) != 6:
            raise ValueError("Expected 6 angles, got %d" % len(angles))
        for i, a in enumerate(angles):
            lo, hi = _ANGLE_RANGES[i + 1]
            if not (lo <= a <= hi):
                raise ValueError("Joint %d angle %s outside [%d, %d]" % (i + 1, a, lo, hi))

        run_time = max(0, min(2000, int(run_time)))
        pulses = [_angle_to_pulse(i + 1, angles[i]) for i in range(6)]

        data = []
        for p in pulses:
            data.extend(struct.pack('h', int(p)))
        data.extend(struct.pack('h', run_time))

        cmd = [_HEAD, _DEVICE_ID, 0x00, FUNC_ARM_CTRL] + list(data)
        cmd[2] = len(cmd) - 1
        cmd.append(_checksum(cmd))
        self._ser.write(bytearray(cmd))
        time.sleep(self._delay)

    def set_single_angle(self, servo_id, angle, run_time=500):
        """Set one servo angle by ID (1-6)."""
        if servo_id < 1 or servo_id > 6:
            raise ValueError("servo_id must be 1-6")
        lo, hi = _ANGLE_RANGES[servo_id]
        if not (lo <= angle <= hi):
            raise ValueError("Angle %s outside [%d, %d]" % (angle, lo, hi))

        pulse = _angle_to_pulse(servo_id, angle)
        run_time = max(0, min(2000, int(run_time)))
        val_bytes = struct.pack('h', int(pulse))
        rt_bytes = struct.pack('h', run_time)
        self._send(FUNC_UART_SERVO,
                   [servo_id & 0xFF, val_bytes[0], val_bytes[1],
                    rt_bytes[0], rt_bytes[1]])

    # ── torque ───────────────────────────────────────────────────────────

    def set_torque(self, enable):
        """Enable (True) or disable (False) servo torque for all bus servos."""
        self._send(FUNC_UART_SERVO_TORQUE, [1 if enable else 0])

    # ── battery ──────────────────────────────────────────────────────────

    def get_battery_voltage(self):
        """Return battery voltage in volts (from last auto-report frame).

        If auto-report is disabled, the value may be stale. The caller can
        re-enable auto-report briefly or accept the cached value.
        """
        return self._battery_raw / 10.0

    # ── version ──────────────────────────────────────────────────────────

    def get_version(self):
        """Request and return firmware version (e.g. 3.9)."""
        self._version_h = 0
        self._request(FUNC_VERSION)
        for _ in range(20):
            if self._version_h != 0:
                return self._version_h + self._version_l / 10.0
            time.sleep(0.001)
        return -1
