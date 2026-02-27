# coding: utf-8
"""
ZMQ JSON protocol for x3plus robot read/write.

Request: {"method": "<name>", ...}
  - getJointPositionArray: no extra fields
  - setJointPositionArray: "joint_array": [a1, a2, a3, a4, a5, a6]
  - setJointPositionSingle: "joint_name": str, "joint_value": int

Response: either result payload or error
  - getJointPositionArray -> {"joint_array": [int, ...]}
  - setJointPositionArray / setJointPositionSingle -> {"result": "OK"}
  - error -> {"error": "message"}
"""

GET_JOINT_POSITION_ARRAY = "getJointPositionArray"
SET_JOINT_POSITION_ARRAY = "setJointPositionArray"
SET_JOINT_POSITION_SINGLE = "setJointPositionSingle"
