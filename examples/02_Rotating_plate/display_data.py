import pandas as pd
import matplotlib.pyplot as plt


def plot_data(df: pd.DataFrame, mode: str, load: str):
    # Generate the labels
    if load == "centrifugal":
        load_label = r"\Omega"
        # title = r"Centrifugal load ($\Omega=1, \dot{\Omega} = 0$)"
    elif load == "euler":
        load_label = r"\dot{\Omega}"
        # title = r"Euler load ($\Omega=0, \dot{\Omega} = 1$)"
    title = f"$\hat{{K}}_{{{mode}}}^{{{load_label}}}$ ({load.title()})"
    # Display the SIFs for the centrifugal load
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(projection="3d")
    surf = ax.plot_trisurf(
        df["Lc_L"],
        df["a_W"],
        df[f"K{mode}_{load}"],
        cmap="viridis",
        alpha=0.9,
    )
    # # Adjust the colorbar position and size
    # fig.colorbar(
    #     surf,
    #     ax=ax,
    #     shrink=0.6,  # Reduce the size of the colorbar
    #     aspect=10,
    #     pad=0.15,  # Add padding to move the colorbar away from the axes
    #     label=zlabel,
    # )
    ax.set_xlabel(r"$L_c / L$")  # , fontsize=24)  # "Crack position $L_c / L$")
    ax.set_ylabel(r"$a / W$")  # , fontsize=24)  # "Crack length $a / W$")
    # ax.set_zlabel(zlabel)
    # ax.set_title(title, fontsize=24)
    plt.tight_layout()  # Automatically adjust subplot parameters to give specified padding
    plt.savefig(f"K{mode}_{load}_data.pdf")
    return fig


def main():
    # Read the dataset
    df = pd.read_csv("dataset.csv")
    # Generate the plots
    plot_data(df, mode="I", load="centrifugal")
    plot_data(df, mode="II", load="centrifugal")
    plot_data(df, mode="III", load="centrifugal")
    plot_data(df, mode="I", load="euler")
    plot_data(df, mode="II", load="euler")
    plot_data(df, mode="III", load="euler")


if __name__ == "__main__":
    main()
