## Micromouse Maze Solver: D\* Lite, Flood Fill, and Hybrid Algorithms

Welcome to the **Micromouse Maze Solver** repository! This project implements multiple maze-solving algorithms including **D\* Lite**, **Flood Fill**, and a **hybrid approach (D\* Lite + Flood Fill)**. These algorithms are designed to solve grid-based mazes and are tested within the Micromouse Simulator.

---

### How to Run the Project

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/vb277/micromousee.git
   cd micromousee
   ```

2. **Install the Micromouse Simulator**:
   The algorithms are designed to be run within the [MMS - Micromouse Simulator](https://github.com/mackorone/mms). You can download the simulator by following these steps:

   - **Windows**: Download `windows.zip` from the [MMS Releases Page](https://github.com/mackorone/mms/releases), unzip, and run `mms.exe`.
   - **macOS**: Download `macos.zip` from the [MMS Releases Page](https://github.com/mackorone/mms/releases), unzip, and run the `mms` app.

   _Note_: You may encounter security warnings. Follow the steps in the simulator's README to bypass them:

   - Windows: Click "More info" and then "Run anyway".
   - macOS: Control-click the app and select "Open" to bypass security warnings for unidentified developers.

3. **Configure the Simulator**:

   - In the simulator interface, click the "Browse" button next to the `Run Command` field.
   - Navigate to the directory where you cloned the repository and select `main.py`.
   - Fill in the `Run Command` field with the following (modify it to reflect your local Python version and file path):
     ```bash
     /path/to/python3 /path/to/micromousee/main.py
     ```
   - Leave the `Build Command` blank unless you are compiling specific algorithms.

4. **Choose the Algorithm**:
   In `main.py`, update the `algorithm` variable to run the algorithm of your choice:

   ```python
   algorithm = "dliteff32"  # Example: Running the D* Lite with Flood Fill hybrid for 32x32 mazes
   ```

### Here are the available options for the `algorithm` variable:

#### **Flood Fill Algorithms**:

- `flood_fill`: Classic Flood Fill for 16x16 mazes.
- `flood_fill_6`: Flood Fill for 6x6 mazes.
- `flood_fill_32`: Flood Fill for 32x32 mazes.

#### **D\* Lite Algorithms**:

- `dlite`: Classic D\* Lite for 16x16 mazes.
- `dlite6`: D\* Lite for 6x6 mazes.
- `dlite32`: D\* Lite for 32x32 mazes.
- `dlite4`: D\* Lite for 4x4 mazes (useful for smaller test cases).

#### **Hybrid D\* Lite + Flood Fill**:

- `dliteff`: Hybrid D\* Lite with Flood Fill for 16x16 mazes.
- `dliteff6`: Hybrid D\* Lite with Flood Fill for 6x6 mazes.
- `dliteff32`: Hybrid D\* Lite with Flood Fill for 32x32 mazes.

---

### 5. Run the Simulator

After selecting the algorithm in the `main.py` file, press **Run** in the simulator interface. Make sure the correct maze is loaded, and the path to your Python environment and `main.py` is correctly set.
