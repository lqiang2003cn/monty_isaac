# coding: utf-8
"""
ZMQ server for x3plus arm control. Runs on the Orin next to the serial port.

Uses the slim ArmSerial driver (arm_serial.py) instead of the full Rosmaster
car+arm driver, which eliminates ~100 packets/sec of unused car telemetry
from the serial bus and reduces CPU overhead.

Protocol (JSON) — see zmq_protocol.py for full spec.
  Core:    getJointPositionArray, setJointPositionArray, setJointPositionSingle
  Optional: getBatteryVoltage, setAutoReport, getStatus

Serial device: set SERIAL_DEVICE env or pass --device; default /dev/ttyUSB0.
"""

import json
import logging
import os
import sys
import time

import zmq

from arm_serial import ArmSerial
from zmq_protocol import (
    GET_JOINT_POSITION_ARRAY,
    SET_JOINT_POSITION_ARRAY,
    SET_JOINT_POSITION_SINGLE,
    GET_BATTERY_VOLTAGE,
    SET_AUTO_REPORT,
    GET_STATUS,
)

try:
    from serial.serialutil import SerialException
except ImportError:
    SerialException = Exception

DEFAULT_PORT = 5555
DEFAULT_SERIAL = "/dev/ttyUSB0"

log = logging.getLogger("zmq_service")


def get_serial_device():
    """Serial device from SERIAL_DEVICE env or --device arg or default."""
    dev = os.environ.get("SERIAL_DEVICE")
    if dev:
        return dev
    for i, arg in enumerate(sys.argv):
        if arg == "--device" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return DEFAULT_SERIAL


def get_port():
    port = os.environ.get("ZMQ_PORT")
    if port is not None:
        return int(port)
    for i, arg in enumerate(sys.argv):
        if arg == "--port" and i + 1 < len(sys.argv):
            return int(sys.argv[i + 1])
    return DEFAULT_PORT


class ArmZMQHandler:
    def __init__(self):
        serial_dev = get_serial_device()
        print("service initializing (serial=%s)" % serial_dev)
        try:
            self.arm = ArmSerial(port=serial_dev)
        except SerialException as e:
            if "Permission denied" in str(e) or "Errno 13" in str(e):
                print(
                    "Permission denied opening serial port. Run with: "
                    "sudo chmod a+rw %s" % serial_dev
                )
                print("Or run container with --privileged or ensure device is readable.")
            else:
                print("Serial open failed: %s" % e)
            raise

        self.arm.start()

        # Disable MCU auto-report: stops ~100 pkts/s of IMU/encoder/speed data
        # that the arm never uses. Battery voltage cache goes stale, but we can
        # request it on-demand through getStatus / getBatteryVoltage.
        self.arm.set_auto_report(False)
        self._auto_report_enabled = False

        voltage = self.arm.get_battery_voltage()
        version = self.arm.get_version()
        print("battery=%.1fV  firmware=v%.1f  auto_report=off" % (voltage, version))
        print("service initialized")

    def get_joint_position_array(self):
        angles = self.arm.get_angles()
        retries = 0
        while any(a == -1 for a in angles):
            retries += 1
            if retries == 1:
                bad = [i + 1 for i, a in enumerate(angles) if a == -1]
                log.warning("Bad pulse on servo(s) %s, retrying...", bad)
            if retries > 50:
                bad = [i + 1 for i, a in enumerate(angles) if a == -1]
                log.error("Servo read timeout (50 retries), bad servos: %s", bad)
                return {"error": "Servo read timeout (50 retries), bad servos: %s" % bad}
            time.sleep(0.001)
            angles = self.arm.get_angles()
        return {"joint_array": list(angles)}

    def set_joint_position_array(self, joint_array):
        self.arm.set_angles(joint_array, run_time=100)
        return {"result": "OK"}

    def set_joint_position_single(self, joint_name, joint_value):
        name_to_sid = {"arm_joint1": 1, "arm_joint2": 2, "arm_joint3": 3,
                       "arm_joint4": 4, "arm_joint5": 5, "grip_joint": 6}
        if joint_name not in name_to_sid:
            return {"error": "Joint name not recognized: %s" % joint_name}
        self.arm.set_single_angle(name_to_sid[joint_name], int(joint_value), run_time=100)
        return {"result": "OK"}

    def get_battery_voltage(self):
        return {"voltage": self.arm.get_battery_voltage()}

    def set_auto_report(self, enable):
        self.arm.set_auto_report(enable)
        self._auto_report_enabled = enable
        return {"result": "OK", "auto_report": enable}

    def get_status(self):
        return {
            "voltage": self.arm.get_battery_voltage(),
            "version": self.arm.get_version(),
            "auto_report": self._auto_report_enabled,
        }

    def handle_request(self, req):
        if not isinstance(req, dict) or "method" not in req:
            return {"error": "Missing method"}
        method = req["method"]

        if method == GET_JOINT_POSITION_ARRAY:
            return self.get_joint_position_array()

        if method == SET_JOINT_POSITION_ARRAY:
            arr = req.get("joint_array")
            if arr is None or len(arr) != 6:
                return {"error": "joint_array must be 6 integers"}
            return self.set_joint_position_array(list(arr))

        if method == SET_JOINT_POSITION_SINGLE:
            name = req.get("joint_name")
            value = req.get("joint_value")
            if name is None or value is None:
                return {"error": "joint_name and joint_value required"}
            return self.set_joint_position_single(name, value)

        if method == GET_BATTERY_VOLTAGE:
            return self.get_battery_voltage()

        if method == SET_AUTO_REPORT:
            enable = req.get("enable")
            if enable is None:
                return {"error": "'enable' (bool) required for setAutoReport"}
            return self.set_auto_report(bool(enable))

        if method == GET_STATUS:
            return self.get_status()

        return {"error": "Unknown method: %s" % method}


def serve():
    port = get_port()
    handler = ArmZMQHandler()
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:%d" % port)
    print("ZMQ server listening on tcp://*:%d" % port)

    while True:
        try:
            raw = socket.recv_string()
            req = json.loads(raw)
            resp = handler.handle_request(req)
            socket.send_string(json.dumps(resp))
        except json.JSONDecodeError as e:
            socket.send_string(json.dumps({"error": "Invalid JSON: %s" % e}))
        except Exception as e:
            log.exception("request failed")
            socket.send_string(json.dumps({"error": str(e)}))


if __name__ == "__main__":
    logging.basicConfig()
    serve()
