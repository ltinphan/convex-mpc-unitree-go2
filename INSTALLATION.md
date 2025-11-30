# Installation Guide

This guide provides step-by-step instructions for setting up the Convex MPC Unitree Go2 project environment.

## Prerequisites

- macOS (tested on macOS with Apple Silicon)
- Python 3.10+ (tested with Python 3.13.1)
- Git
- `wget` or `curl` for downloading files

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/ltinphan/convex-mpc-unitree-go2.git
cd convex-mpc-unitree-go2
```

### 2. Create Python Virtual Environment

This project uses a Python virtual environment instead of conda:

```bash
python3 -m venv venv
source venv/bin/activate
```

**Note:** Always activate the virtual environment before running the project:

```bash
source venv/bin/activate
```

### 3. Install Python Dependencies

Install all required Python packages:

```bash
pip install --upgrade pip
pip install mujoco casadi numpy scipy matplotlib pin
```

**Installed versions:**

- MuJoCo: 3.3.7
- CasADi: 3.7.2
- Pinocchio (pin): 3.8.0
- NumPy: 2.3.5
- SciPy: 1.16.3
- Matplotlib: 3.10.7

### 4. Download and Setup Unitree MuJoCo Assets

Clone the Unitree MuJoCo repository and copy it to the third_party directory:

```bash
git clone https://github.com/unitreerobotics/unitree_mujoco.git
mkdir -p third_party
cp -r unitree_mujoco third_party/
```

#### Update Foot Friction

Edit the MuJoCo model to update the foot friction coefficient:

```bash
# Open the file: third_party/unitree_mujoco/unitree_robots/go2/go2.xml
# Find line 33 (the foot friction definition)
# Change: friction="0.4 0.02 0.01"
# To:     friction="0.8 0.02 0.01"
```

Or use this command:

```bash
sed -i '' 's/friction="0.4 0.02 0.01"/friction="0.8 0.02 0.01"/' third_party/unitree_mujoco/unitree_robots/go2/go2.xml
```

### 5. Download and Setup Unitree GO2 URDF

Download the GO2 URDF package from Unitree:

```bash
wget https://oss-global-cdn.unitree.com/static/Go2_URDF.zip
unzip Go2_URDF.zip
cp -r GO2_URDF third_party/go2_description
```

**Important:** The directory must be named `go2_description` (not `unitree_go2_description`) to match the package name referenced in the URDF files.

### 6. Clean Up Temporary Files

Remove the downloaded files to keep the repository clean:

```bash
rm -rf unitree_mujoco GO2_URDF Go2_URDF.zip
```

### 7. Verify Installation

Run a quick verification:

```bash
python -c "import mujoco, casadi, pin; print('All packages imported successfully!')"
```

## Directory Structure

After installation, your directory structure should look like this:

```
convex-mpc-unitree-go2/
├── convex_mpc/
│   ├── __init__.py
│   ├── centroidal_mpc.py
│   ├── com_trajectory.py
│   ├── gait.py
│   ├── go2_robot_data.py
│   ├── leg_controller.py
│   ├── mujoco_model.py
│   ├── plot_helper.py
│   └── test_MPC.py
├── third_party/
│   ├── go2_description/          # GO2 URDF files
│   └── unitree_mujoco/           # Unitree MuJoCo assets
├── venv/                         # Python virtual environment
├── simulation_results/           # Generated plots (created at runtime)
├── README.md
├── INSTALLATION.md
└── LICENSE
```

## Running the Demo

To run the MPC locomotion demo:

```bash
source venv/bin/activate
mjpython convex_mpc/test_MPC.py
```

**What to expect:**

- The simulation runs for 10 seconds
- MuJoCo viewer window shows the robot performing locomotion
- Plots are saved to `simulation_results/` directory:
  - `swing_foot_trajectory.png` - Foot trajectory during swing phase
  - `mpc_results.png` - MPC optimization results
  - `solve_time.png` - QP solver performance

## Troubleshooting

### Issue: "No module named '_tkinter'"

This is expected on macOS with mjpython. The project uses the Agg backend to save plots as files instead of displaying them interactively.

### Issue: "Mesh package://go2_description/... could not be found"

Make sure the URDF directory is named exactly `go2_description` (not `unitree_go2_description`).

### Issue: "NSWindow should only be instantiated on the main thread"

This occurs if matplotlib tries to display plots interactively. The project is configured to save plots to files to avoid this issue.

### Issue: Missing Python packages

Ensure the virtual environment is activated:

```bash
source venv/bin/activate
pip list | grep -E "(mujoco|casadi|numpy|scipy|matplotlib|pin)"
```

## Configuration

You can modify simulation parameters in `convex_mpc/test_MPC.py`:

- **Simulation duration:** `RUN_SIM_LENGTH_S` (default: 10 seconds)
- **Motion commands:** `CMD_SCHEDULE` (velocity, yaw rate, etc.)
- **Gait parameters:** In `gait.py` (frequency, duty cycle, swing height)
- **MPC cost matrices:** In `centroidal_mpc.py`
- **Friction coefficient:** In `centroidal_mpc.py` and MuJoCo XML file

## System Requirements

- **Python:** 3.10 or higher (recommended: 3.10.15)
- **OS:** macOS (tested on Apple Silicon)
- **RAM:** Minimum 8GB recommended
- **Storage:** ~500MB for dependencies and assets

## Notes

- The project uses `mjpython` (bundled with MuJoCo) instead of regular Python for proper MuJoCo viewer integration on macOS
- Plots are saved to files to avoid threading conflicts with the macOS GUI framework
- The virtual environment approach is used instead of conda for better compatibility

## References

- [MuJoCo Documentation](https://mujoco.readthedocs.io/)
- [Pinocchio Documentation](https://stack-of-tasks.github.io/pinocchio/)
- [CasADi Documentation](https://web.casadi.org/)
- [Unitree Robotics](https://www.unitree.com/)
