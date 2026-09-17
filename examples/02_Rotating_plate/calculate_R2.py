import numpy as np
from sklearn.metrics import r2_score


def main():
    # Load data
    data = np.loadtxt("dataset_plane_stress_L=400mm.csv", delimiter=",", skiprows=1)

    # Get the parameters
    a_W = data[:, 0]
    Lc_L = data[:, 1]
    # Get the output
    KIe_dat = data[:, 2]
    KIc_dat = data[:, 6]

    # Get the model prediction
    KIe_num = KIe_exp(a_W, Lc_L)
    KIc_num = KIc_exp(a_W, Lc_L)

    # Compute the R2 score
    print(f"{r2_score(KIe_dat, KIe_num)=}")
    print(f"{r2_score(KIc_dat, KIc_num)=}")


def KIe_exp(a_W, Lc_L):
    return np.exp(
        +7
        + 0.5956399254252611 * np.exp(a_W) ** 2
        - 0.27462777897111884 * np.exp(Lc_L) ** 3
    )


def KIc_exp(a_W, Lc_L):
    return np.exp(
        +2.37 + 2.724969871424484 * np.exp(a_W) - 1.026445231266538 * np.exp(Lc_L)
    )


if __name__ == "__main__":
    main()
