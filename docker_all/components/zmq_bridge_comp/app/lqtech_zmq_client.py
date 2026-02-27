# coding: utf-8
"""
ZMQ client for x3plus robot read/write. Connect to robot over the network.

Usage:
  from lqtech_zmq_client import LqtechZMQClient
  client = LqtechZMQClient(host="192.168.1.10", port=5555)
  angles = client.get_joint_position_array()
  client.set_joint_position_single("arm_joint1", 92)

Or run as script: python lqtech_zmq_client.py [--host HOST] [--port PORT]
  Demo: get angles, set joint 1 to 92, print response.
"""

import argparse
import json
import logging
import os
import sys
import time

import zmq

from zmq_protocol import (
    GET_JOINT_POSITION_ARRAY,
    SET_JOINT_POSITION_ARRAY,
    SET_JOINT_POSITION_SINGLE,
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

    def close(self):
        self._socket.close()
        self._context.term()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def run_demo(host, port):
    with LqtechZMQClient(host=host, port=port) as client:
        print("---- iteration 1 / 1 ----")
        angles = client.get_joint_position_array()
        print("client received:", angles)
        target_angles = list(angles)
        target_angles[0] = 0
        result = client.set_joint_position_single("arm_joint1", target_angles[0])
        print("Set joint position response:", result)


def main():
    parser = argparse.ArgumentParser(description="x3plus ZMQ client demo")
    parser.add_argument("--host", default=os.environ.get("ZMQ_HOST", DEFAULT_HOST), help="Robot host (default: localhost)")
    parser.add_argument("--port", type=int, default=int(os.environ.get("ZMQ_PORT", DEFAULT_PORT)), help="ZMQ port (default: 5555)")
    args = parser.parse_args()
    logging.basicConfig()
    start = time.perf_counter()
    run_demo(args.host, args.port)
    elapsed = time.perf_counter() - start
    print("Execution took %.6f seconds" % elapsed)


if __name__ == "__main__":
    main()
