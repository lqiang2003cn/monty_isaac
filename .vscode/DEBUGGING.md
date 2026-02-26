# Debugging Isaac Sim ROS2 Demo in VSCode

This guide explains how to debug the Isaac Sim ROS2 demo script in VSCode.

## Prerequisites

1. **Python Debugger Extension**: Install the "Python" extension by Microsoft in VSCode
2. **debugpy**: Install in your Isaac Sim environment:
   ```bash
   pip install debugpy
   ```

## Debugging Methods

### Method 1: Local Debugging (Recommended)

If you have Isaac Sim installed locally with a virtual environment:

1. **Set up your Python interpreter**:
   - Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
   - Type "Python: Select Interpreter"
   - Choose your Isaac Sim Python interpreter (usually `~/isaacsim_venv/bin/python`)

2. **Set breakpoints**:
   - Open `demos/isaac_sim_ros2_demo.py`
   - Click in the left margin to set breakpoints (red dots)

3. **Start debugging**:
   - Press `F5` or go to Run and Debug panel
   - Select "Debug Isaac Sim ROS2 Demo (Local)"
   - Click the green play button

4. **Debug controls**:
   - `F5`: Continue
   - `F10`: Step Over
   - `F11`: Step Into
   - `Shift+F11`: Step Out
   - `Shift+F5`: Stop

## Environment Variables

The launch configurations automatically set:
- `ACCEPT_EULA=Y`
- `ROS_DOMAIN_ID=0`
- `RMW_IMPLEMENTATION=rmw_fastrtps_cpp`
- `ROS_DISTRO=jazzy`

You can modify these in `.vscode/launch.json` if needed.

## Troubleshooting

### "Python interpreter not found"
- Make sure your Isaac Sim virtual environment path is correct
- Update `.vscode/settings.json` with the correct path
- Or use "Debug Isaac Sim ROS2 Demo (Local - Custom Python)" and select interpreter manually

### "Cannot attach to debugger"
- Check that port 5678 is not blocked by firewall
- Try changing `DEBUG_PORT` environment variable

### "Module 'isaacsim' not found"
- Make sure you're using the correct Python interpreter (Isaac Sim's Python, e.g. `~/isaacsim_venv/bin/python`)

### Breakpoints not hitting
- Check that `justMyCode: false` is set in launch.json (it is by default)
- Make sure you're debugging the correct file
- Try adding `import pdb; pdb.set_trace()` as a temporary breakpoint

## Tips

1. **Use conditional breakpoints**: Right-click on a breakpoint to add conditions
2. **Watch variables**: Add variables to the Watch panel while debugging
3. **Debug console**: Use the Debug Console to evaluate expressions
4. **Call stack**: Inspect the call stack to understand execution flow

## Configuration Files

- `.vscode/launch.json`: Debug configurations
- `.vscode/settings.json`: VSCode Python settings
- `demos/debug_isaac_sim_ros2_demo.py`: Debug-enabled version of the demo
