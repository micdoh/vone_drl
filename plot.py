import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from matplotlib import rc
import argparse


def get_blocking(col):
    blocking = (10 - col) * 10
    return blocking


def get_blocking_std(col):
    return col * 10


def replace_blocking_cols(df):
    df["blocking"] = get_blocking(df["reward"])
    df["blocking_std"] = get_blocking_std(df["std"])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval", action="store_true", help="Plot evaluation data"
    )
    parser.add_argument(
        "--train", action="store_true", help="Plot training data"
    )
    parser.add_argument(
        "--heur_eval_file", default="", type=str, help="Path to heuristics csv file"
    )
    parser.add_argument(
        "--node_eval_file", default="", type=str, help="Path to node agent csv file"
    )
    parser.add_argument(
        "--path_eval_file", default="", type=str, help="Path to path agent csv file"
    )
    parser.add_argument(
        "--combined_eval_file", default="", type=str, help="Path to combined agent csv file"
    )
    parser.add_argument(
        "--node_train_file", default="", type=str, help="Path to node agent training csv file"
    )
    parser.add_argument(
        "--path_train_file", default="", type=str, help="Path to path agent training csv file"
    )
    parser.add_argument(
        "--combined_train_file", default="", type=str, help="Path to combined agent training csv file"
    )
    parser.add_argument(
        "--use_tex", action="store_true", help="Use tex for plotting"
    )
    args = parser.parse_args()

    if args.eval:

        node_eval = Path(args.node_eval_file)
        path_eval = Path(args.path_eval_file)
        combined_eval = Path(args.combined_eval_file)
        heur_eval = Path(args.heur_eval_file)

        path_eval_df = pd.read_csv(path_eval)
        node_eval_df = pd.read_csv(node_eval)
        combined_eval_df = pd.read_csv(combined_eval)
        heur_eval_df = pd.read_csv(heur_eval)

        replace_blocking_cols(path_eval_df)
        replace_blocking_cols(node_eval_df)
        replace_blocking_cols(combined_eval_df)
        replace_blocking_cols(heur_eval_df)

        if args.use_tex:
            rc('text', usetex=True)
            rc('font', size=14)
            rc('legend', fontsize=13)
            rc('text.latex', preamble=r'\usepackage{cmbright}')

        plt.plot(path_eval_df["load"], path_eval_df["blocking"], label="Path Agent", marker='v')
        plt.plot(node_eval_df["load"], node_eval_df["blocking"], label="Node Agent", marker='o')
        plt.plot(combined_eval_df["load"], combined_eval_df["blocking"], label="Combined Agent", marker="d")
        plt.plot(heur_eval_df["load"], heur_eval_df["blocking"], label="NSC-kSP-FDL", marker="s")

        plt.legend()
        plt.ylabel(r"Blocking Probability [\%]")
        plt.xlabel("Traffic Load [Erlangs]")
        plt.yscale("log")
        plt.grid(True)
        plt.show()

    if args.train:

        node_train = Path(args.node_train_file)
        path_train = Path(args.path_train_file)
        combined_train = Path(args.combined_train_file)

        node_train_df = pd.read_csv(node_train)
        path_train_df = pd.read_csv(path_train)
        combined_train_df = pd.read_csv(combined_train)

        if args.use_tex:
            rc('text', usetex=True)
            rc('font', size=14)
            rc('legend', fontsize=13)
            rc('text.latex', preamble=r'\usepackage{cmbright}')

        plt.plot(path_train_df["Step"], path_train_df["acceptance_ratio"], label="Path Agent", marker='v')
        plt.plot(node_train_df["Step"], node_train_df["acceptance_ratio"], label="Node Agent", marker="o")
        plt.plot(combined_train_df["Step"][:40], combined_train_df["acceptance_ratio"][:40], label="Combined Agent", marker='d')

        plt.legend()
        plt.ylabel("Acceptance Ratio")
        plt.xlabel("Training Episode")
        plt.ylim(0, 1)
        plt.xlim(0, 40)
        plt.grid(True)
        plt.show()
