# coding: utf-8
"""
ZMQ JSON protocol for x3plus robot read/write.

Request: {"method": "<name>", ...}
  Core (arm control):
  - getJointPositionArray: no extra fields
  - setJointPositionArray: "joint_array": [a1, a2, a3, a4, a5, a6]
  - setJointPositionSingle: "joint_name": str, "joint_value": int

  Optional (diagnostics):
  - getBatteryVoltage: no extra fields
  - setAutoReport: "enable": bool
  - getStatus: no extra fields  (returns version + voltage + auto_report state)

Response: either result payload or error
  - getJointPositionArray -> {"joint_array": [int, ...]}
  - setJointPositionArray / setJointPositionSingle -> {"result": "OK"}
  - getBatteryVoltage -> {"voltage": float}
  - setAutoReport -> {"result": "OK", "auto_report": bool}
  - getStatus -> {"voltage": float, "version": float, "auto_report": bool}
  - error -> {"error": "message"}

Keep this file in sync with the ROS-side copy at:
  monty_demo/opus_plan_and_imp/zmq_protocol.py
"""

# ── Core: arm joint control ──────────────────────────────────────────────
GET_JOINT_POSITION_ARRAY = "getJointPositionArray"
SET_JOINT_POSITION_ARRAY = "setJointPositionArray"
SET_JOINT_POSITION_SINGLE = "setJointPositionSingle"

# ── Optional: diagnostics & configuration ────────────────────────────────
GET_BATTERY_VOLTAGE = "getBatteryVoltage"
SET_AUTO_REPORT = "setAutoReport"
GET_STATUS = "getStatus"
