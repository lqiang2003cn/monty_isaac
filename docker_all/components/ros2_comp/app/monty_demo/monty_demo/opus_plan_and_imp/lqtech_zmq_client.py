# coding: utf-8
"""
ZMQ client for x3plus robot read/write. Connect to robot over the network.

Usage:
  from monty_demo.opus_plan_and_imp.lqtech_zmq_client import LqtechZMQClient
  client = LqtechZMQClient(host="192.168.1.10", port=5555)
  angles = client.get_joint_position_array()
  client.set_joint_position_single("arm_joint1", 92)

  # Optional diagnostics (only available with ArmSerial-based service)
  voltage = client.get_battery_voltage()
  status = client.get_status()
"""

import json
import os

import zmq

from monty_demo.opus_plan_and_imp.zmq_protocol import (
    GET_JOINT_POSITION_ARRAY,
    SET_JOINT_POSITION_ARRAY,
    SET_JOINT_POSITION_SINGLE,
    GET_BATTERY_VOLTAGE,
    SET_AUTO_REPORT,
    GET_STATUS,
)

DEFAULT_HOST = "192.168.31.142"
DEFAULT_PORT = 5555


class LqtechZMQClient:
    def __init__(self, host=None, port=None):
        self.host = host or os.environ.get("ZMQ_HOST", DEFAULT_HOST)
        self.port = int(port or os.environ.get("ZMQ_PORT", DEFAULT_PORT))
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.connect("tcp://%s:%d" % (self.host, self.port))

    def _request(self, req):
        self._socket.send_string(json.dumps(req))
        raw = self._socket.recv_string()
        resp = json.loads(raw)
        if "error" in resp:
            raise RuntimeError(resp["error"])
        return resp

    # ── Core: arm joint control ──────────────────────────────────────────

    def get_joint_position_array(self):
        resp = self._request({"method": GET_JOINT_POSITION_ARRAY})
        return resp["joint_array"]

    def set_joint_position_array(self, joint_array):
        if len(joint_array) != 6:
            raise ValueError("joint_array must have 6 elements")
        resp = self._request({"method": SET_JOINT_POSITION_ARRAY, "joint_array": list(joint_array)})
        return resp.get("result", "OK")

    def set_joint_position_single(self, joint_name, joint_value):
        resp = self._request({
            "method": SET_JOINT_POSITION_SINGLE,
            "joint_name": joint_name,
            "joint_value": int(joint_value),
        })
        return resp.get("result", "OK")

    # ── Optional: diagnostics & configuration ────────────────────────────

    def get_battery_voltage(self):
        """Return battery voltage in volts."""
        resp = self._request({"method": GET_BATTERY_VOLTAGE})
        return resp["voltage"]

    def set_auto_report(self, enable):
        """Enable/disable MCU auto-report on the Orin side."""
        resp = self._request({"method": SET_AUTO_REPORT, "enable": bool(enable)})
        return resp.get("auto_report", enable)

    def get_status(self):
        """Return dict with voltage, firmware version, and auto_report state."""
        return self._request({"method": GET_STATUS})

    # ── lifecycle ────────────────────────────────────────────────────────

    def close(self):
        self._socket.close()
        self._context.term()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
