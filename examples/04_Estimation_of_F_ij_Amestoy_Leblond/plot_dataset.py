import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
# Replace this with the path to your actual CSV file
csv_file = "FIJ.csv"

# --- Reading Data ---
# Read the CSV file using numpy
# header=True will use the first row as column names
data = np.genfromtxt(csv_file, delimiter=",", usecols=(0, 1, 2, 3, 4, 5), names=True)

# Extract columns based on your description:
# phi is column 0
# The Fij elements are columns 1, 2, 3, 4, 5
phi = data["phi"]
F11 = data["F11"]
F12 = data["F12"]
F21 = data["F21"]
F22 = data["F22"]
F33 = data["F33"]

# --- Plotting ---
plt.figure(figsize=(10, 6))

# Plotting the Fij components as a line plot
plt.plot(phi, F11, label="F11", linestyle="-", linewidth=1.5)
plt.plot(phi, F12, label="F12", linestyle="--", linewidth=1.5)
plt.plot(phi, F21, label="F21", linestyle="-.", linewidth=1.5)
plt.plot(phi, F22, label="F22", linestyle="-", linewidth=1.5)
plt.plot(phi, F33, label="F33", linestyle="-", linewidth=1.5)

plt.xlabel(r"Phi ($\phi$)", fontsize=12)
plt.ylabel("F$_{ij}$", fontsize=12)
plt.title("F$_{ij}$ Components as a function of Phi", fontsize=14)
plt.legend(loc="best")
plt.grid(True, linestyle=":", alpha=0.6)

# Tight layout to prevent label cutoff
plt.tight_layout()

# Save or display the plot
plt.savefig("Fij_vs_phi.png")
plt.show()
