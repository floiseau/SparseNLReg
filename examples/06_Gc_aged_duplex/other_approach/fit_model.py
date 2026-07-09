import csv
from time import perf_counter

from jax import jit
import jax.numpy as jnp
from scipy.optimize import minimize_scalar, differential_evolution

from gcrack.lefm import G_star


def load_data(data_file):
    # Initialize the values
    phis = []
    theta0s = []
    gcs = []
    # Read the CSV file
    with open(data_file, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            phis.append(jnp.deg2rad(float(row["phi_deg"])))
            theta0s.append(jnp.deg2rad(float(row["Theta0"])))
            gcs.append(float(row["Gc_J_m2"]))
    # Convert to jnp array
    phis = jnp.asarray(phis)
    theta0s = jnp.asarray(theta0s)
    gcs = jnp.asarray(gcs)
    # Return the data
    return phis, theta0s, gcs


@jit
def Gc_model(phi, p, theta0):
    # Define alpha
    alpha = phi - theta0
    # Generate a polynomial
    return (
        p[0]
        + p[1] * jnp.cos(2 * alpha)
        + p[2] * jnp.cos(4 * alpha)
        + p[3] * jnp.cos(6 * alpha)
    )


@jit
def gmeer_obj(phi, p, theta0, phi0, SIFs, Ep):
    Gs = G_star(phi, phi0, SIFs["KI"], SIFs["KII"], SIFs["T"], Ep, 0.0)
    Gc = Gc_model(phi, p=p, theta0=theta0)
    return Gc / Gs


def phi_model(p, theta0):
    # Create the elastic model (from gcrack)
    Ep = 1.0 / (1 - 0.3**2)
    # Asssume mode I sollicitation
    SIFs = {"KI": 50e6, "KII": 0.0, "T": 0.0}
    # Initial angle
    phi0 = 0.0
    # Solve for the predicted crack propagation angle
    res = minimize_scalar(lambda phi: gmeer_obj(phi, p, theta0, phi0, SIFs, Ep))
    return res.x


def calibration_objective(p, phis_data, theta0s_data, Gcs_data):
    print(f"=== Calculation of objective function for {p=}")
    # Calculate Gc from the model
    Gcs_model = jnp.asarray(
        [Gc_model(phi, p, theta0) for phi, theta0 in zip(phis_data, theta0s_data)]
    )
    # Calculate phi from the model (only unique theta0 values)
    utheta0s_data = jnp.unique(theta0s_data)
    uphis_model = {float(theta0): phi_model(p, theta0) for theta0 in utheta0s_data}
    phis_model = jnp.asarray([uphis_model[float(theta0)] for theta0 in theta0s_data])
    # Calculate the objective function
    obj = jnp.sum((Gcs_model - Gcs_data) ** 2) / jnp.sum(Gcs_data**2)
    obj += jnp.sum((phis_model - phis_data) ** 2) / jnp.sum(phis_data**2)
    # Divide by two to compare error to unit value
    obj /= 2
    return obj


def main():
    start = perf_counter()
    # Parameters
    data_file = "dataset.csv"
    # Load the data
    phis_data, theta0s_data, Gcs_data = load_data(data_file)
    # Define the bounds in the parametric space
    bounds = [(10_000, 30_000), (-5_000, 5_000), (-5_000, 5_000), (-5_000, 5_000)]
    # Minimize the objective function
    res = differential_evolution(
        calibration_objective,
        bounds,
        args=(phis_data, theta0s_data, Gcs_data),
    )
    end = perf_counter()

    print(f"Results : {res}")
    print(f"Calibration in {end - start}s.")


if __name__ == "__main__":
    main()
