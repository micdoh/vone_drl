import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
from matplotlib import rc


if __name__ == "__main__":
    rc('font', size=12)
    rc('legend', fontsize=10)
    fig, ax = plt.subplots()
    colors = {
        "calrc": "green",
        "nsc": "blue",
        "tmr": "orange",
        "Random Masked": "black",
        "Agent": "red"
    }

    linestyles = {
        "msp_ef": "--",
        "ff": ":",
        "fdl": "-.",
        "Random Masked": "solid",
        "Agent": "solid"
    }

    nh = ["calrc", "nsc", "tmr"]
    ph = ["ff", "fdl", "msp_ef"]

    heur_eval = Path("./eval/ecoc_100slots")

    for nodeh in nh:
        if nodeh == "nsc" or nodeh == "tmr":
            nname = nodeh.upper()
        else:
            nname = "CaLRC"

        for pathh in ph:
            if pathh[0] == 'f':
                pname = f"kSP-{pathh.upper()}"
            else:
                pname = "MSP-EF"

            filename = heur_eval / f"{nodeh}_{pathh}_100slots.csv"
            df = pd.read_csv(filename)

            heuristic = f"{nname}+{pname}"
            color = colors[nodeh]
            linestyle = linestyles[pathh]

            ax.plot(df["load"], df["blocking"], label=heuristic, c=color, linestyle=linestyle)
            ax.fill_between(df["load"], df["blocking"] - df["blocking_std"],
                            df["blocking"] + df["blocking_std"], alpha=0.3, facecolor=color)

    # Plot random masked
    random_file = heur_eval / "eval_random_masked.csv"
    df = pd.read_csv(random_file)
    heuristic = "Random Masked"
    color = colors[heuristic]
    linestyle = linestyles[heuristic]

    ax.plot(df["load"], df["blocking"], label=heuristic, c=color, linestyle=linestyle)
    ax.fill_between(df["load"], df["blocking"] - df["blocking_std"],
                    df["blocking"] + df["blocking_std"], alpha=0.3, facecolor=color)

    combined_eval = heur_eval / "chnwvsml_100slots.csv"
    combined_eval_df = pd.read_csv(combined_eval)
    heuristic = "Agent"
    color = colors[heuristic]
    linestyle = linestyles[heuristic]

    ax.plot(combined_eval_df["load"], combined_eval_df["blocking"], label=heuristic, c=color, linestyle=linestyle)
    ax.fill_between(combined_eval_df["load"], combined_eval_df["blocking"] - combined_eval_df["blocking_std"],
                    combined_eval_df["blocking"] + combined_eval_df["blocking_std"], alpha=0.3, facecolor=color)

    ax.legend(loc="lower right")
    # Increase legend size
    ax.legend()
    ax.set_yscale('log')

    plt.ylabel(r"Blocking Probability")
    plt.xlabel("Traffic Load [Erlangs]")
    plt.yscale("log")
    plt.xlim(40, 100)
    plt.grid(True)
    plt.show()
