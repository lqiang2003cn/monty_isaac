# coding: utf-8
"""
ZMQ server for x3plus robot read/write. Replaces gRPC transport with JSON over ZMQ.

Protocol (JSON):
  Request:  {"method": "<name>", ...}
    getJointPositionArray: no extra fields
    setJointPositionArray:  "joint_array": [a1,a2,a3,a4,a5,a6]
    setJointPositionSingle: "joint_name": str, "joint_value": int
  Response: {"joint_array": [...]} | {"result": "OK"} | {"error": "message"}

Serial permission: run with sudo chmod a+rw /dev/ttyUSB0 first, or sudo python ...
"""

import json
import logging
import os
import sys
import time

import numpy as np
import zmq

from x3plus_serial import Rosmaster
from zmq_protocol import (
    GET_JOINT_POSITION_ARRAY,
    SET_JOINT_POSITION_ARRAY,
    SET_JOINT_POSITION_SINGLE,
)

try:
    from serial.serialutil import SerialException
except ImportError:
    SerialException = Exception

DEFAULT_PORT = 5555


def get_port():
    port = os.environ.get("ZMQ_PORT")
    if port is not None:
        return int(port)
    for i, arg in enumerate(sys.argv):
        if arg == "--port" and i + 1 < len(sys.argv):
            return int(sys.argv[i + 1])
    return DEFAULT_PORT


class RosmasterZMQHandler:
    def __init__(self):
        self.joint_name_to_sid_map = {"arm_joint1": 1}
        print("service initializing")
        try:
            self.serial = Rosmaster(car_type=2, com="/dev/ttyUSB0", debug=False)
        except SerialException as e:
            if "Permission denied" in str(e) or "Errno 13" in str(e):
                print(
                    "Permission denied opening serial port. Run with: "
                    "sudo chmod a+rw /dev/ttyUSB0 && python lqtech_zmq_service.py"
                )
                print("Or: sudo python lqtech_zmq_service.py")
            else:
                print("Serial open failed: %s" % e)
            raise
        self.serial.create_receive_threading()
        print("current battery voltage is:", self.serial.get_battery_voltage())
        print("service initialized")

    def get_joint_position_array(self):
        angles = self.serial.get_uart_servo_angle_array()
        while np.any(np.array(angles) == -1):
            angles = self.serial.get_uart_servo_angle_array()
            time.sleep(0.0001)
        return {"joint_array": list(angles)}

    def set_joint_position_array(self, joint_array):
        self.serial.set_uart_servo_angle_array(angle_s=joint_array, run_time=2000)
        return {"result": "OK"}

    def set_joint_position_single(self, joint_name, joint_value):
        if joint_name not in self.joint_name_to_sid_map:
            return {"error": "Joint name not recognized"}
        sid = self.joint_name_to_sid_map[joint_name]
        self.serial.set_uart_servo_angle(sid, int(joint_value), run_time=2000)
        return {"result": "OK"}

    def handle_request(self, req):
        if not isinstance(req, dict) or "method" not in req:
            return {"error": "Missing method"}
        method = req.get("method")
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
        return {"error": "Unknown method: %s" % method}


def serve():
    port = get_port()
    handler = RosmasterZMQHandler()
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
            logging.exception("request failed")
            socket.send_string(json.dumps({"error": str(e)}))


if __name__ == "__main__":
    logging.basicConfig()
    serve()
