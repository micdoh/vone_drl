import pandas as pd
from pathlib import Path
from matplotlib import rc
import argparse
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, Circle
from matplotlib.collections import PatchCollection
import matplotlib.colors as mcolors
import seaborn as sns


class GridVisualizer:
    def __init__(self, G):
        self.G = G
        self._initialize_representation()

    def _initialize_representation(self):
        self.array_repr = []
        self.edge_order = []
        for edge in self.G.edges:
            self.edge_order.append(edge)
            self.array_repr.append(self.G.edges[edge]["slots"])

        self.array_repr = np.array(self.array_repr)

    def update(self, G_new):
        self.G = G_new
        self._initialize_representation()

    def draw_figure(self):
        nrows, ncols = self.array_repr.shape
        colors = list(plt.cm.tab10.colors)
        colors[0] = (1, 1, 1, 1)  # Set the first color (for unoccupied slots) to white

        fig, ax = plt.subplots(figsize=(10, 10 * nrows / ncols))

        rect_width = ax.get_window_extent().width / ncols
        rect_height = ax.get_window_extent().height / nrows

        for row in range(nrows):
            for col in range(ncols):
                value = self.array_repr[row, col]
                rect = Rectangle((col * rect_width, row * rect_height), rect_width, rect_height,
                                 facecolor=colors[int(value)], edgecolor='black')
                ax.add_patch(rect)

        fontsize = int(5 + nrows * 1.5)
        circle_radius = fontsize / 4
        for row, edge in enumerate(self.edge_order):
            node1_capacity = self.G.nodes[edge[0]]["capacity"]
            node2_capacity = self.G.nodes[edge[1]]["capacity"]

            circle1 = Circle((-1 * rect_width, row * rect_height + 0.5 * rect_height), circle_radius, facecolor='white',
                             edgecolor='black')
            circle2 = Circle((ncols * rect_width, row * rect_height + 0.5 * rect_height), circle_radius,
                             facecolor='white', edgecolor='black')
            ax.add_patch(circle1)
            ax.add_patch(circle2)

            ax.annotate(f"{node1_capacity}", (-1 * rect_width, row * rect_height + 0.5 * rect_height),
                        fontsize=fontsize, ha='center', va='center')
            ax.annotate(f"{node2_capacity}", (ncols * rect_width, row * rect_height + 0.5 * rect_height),
                        fontsize=fontsize, ha='center', va='center')

        ax.set_xlim(-1 * rect_width, ncols * rect_width)
        ax.set_ylim(0, nrows * rect_height)
        ax.set_aspect('equal')
        ax.axis('off')
        plt.gca().set_position([0.1, 0.1, 0.8, 0.8])  # Center the grid
        plt.show()


class EdgeRectangles:
    def __init__(self, binary_array, edge_unit_vector, start_pos, edge_length, square_size, angle):
        self.rectangles = []

        for i, bit in enumerate(binary_array):
            if bit == 1:
                square_color = 'black'
            else:
                square_color = 'white'

            start_offset = (edge_length - len(binary_array) * square_size) / 2
            rect_center = start_pos + (start_offset + i * square_size) * edge_unit_vector
            half_square_size = square_size / 2

            rotation_matrix = np.array([
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)]
            ])

            corners = np.array([
                [half_square_size, half_square_size],
                [half_square_size, -half_square_size],
                [-half_square_size, -half_square_size],
                [-half_square_size, half_square_size]
            ])

            rotated_corners = np.dot(corners, rotation_matrix) + rect_center

            rect = Polygon(rotated_corners, closed=True, linewidth=1, edgecolor='black', facecolor=square_color)
            self.rectangles.append(rect)

    def add_to_axes(self, ax):
        patch_collection = PatchCollection(self.rectangles, match_original=True)
        ax.add_collection(patch_collection)
        return patch_collection


# Function to visualize the graph with binary array rectangles
def visualize_graph(G, node_pos, node_size=500):
    # Create a figure and axis for plotting
    fig, ax = plt.subplots()

    # Draw nodes and labels
    nx.draw_networkx_nodes(G, node_pos, ax=ax, node_size=node_size, node_color="white", edgecolors="black", linewidths=2)
    nx.draw_networkx_labels(G, node_pos, labels={n: G.nodes[n]['capacity'] for n in G.nodes}, ax=ax, font_size=10, font_weight='bold')

    # Draw edges and rectangles
    for edge in G.edges:
        path = nx.draw_networkx_edges(G, node_pos, edgelist=[edge], ax=ax)
        binary_array = G.edges[edge]['slots']
        edge_vector = np.array(node_pos[edge[1]]) - np.array(node_pos[edge[0]])
        edge_length = np.linalg.norm(edge_vector)
        square_size = min(edge_length / len(binary_array), node_size)
        edge_unit_vector = (np.array(node_pos[edge[1]]) - np.array(node_pos[edge[0]])) / edge_length

        angle = np.abs(np.arctan2(edge_vector[1], edge_vector[0]))
        if (node_pos[edge[0]][0] < node_pos[edge[1]][0]) and \
                (node_pos[edge[0]][1] < node_pos[edge[1]][1]):
            angle = np.pi - angle

        edge_rectangles = EdgeRectangles(binary_array, edge_unit_vector, node_pos[edge[0]], edge_length,
                                         square_size, angle)
        edge_rectangles.add_to_axes(ax)
        path.set_zorder(-1)

    # Set the axis to equal aspect ratio
    ax.axis('equal')

    # Draw the figure
    fig.canvas.draw()
    fig.show()

    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # plt.close(fig)

    return img


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
        "--heur_eval_file_1", default="", type=str, help="Path to heuristics csv file"
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

    def calc_blocking_std(df):
        df["blocking_std"] = (df["reward_std"] / 10) * df["blocking"]

    combined_eval = Path(args.combined_eval_file)
    heur_eval = Path(args.heur_eval_file)
    heur_eval_1 = Path(args.heur_eval_file_1)

    combined_eval_df = pd.read_csv(combined_eval)
    heur_eval_df = pd.read_csv(heur_eval)
    heur_eval_1_df = pd.read_csv(heur_eval_1)

    calc_blocking_std(heur_eval_df)
    calc_blocking_std(heur_eval_1_df)

    if args.use_tex:
        rc('text', usetex=True)
        rc('font', size=14)
        rc('legend', fontsize=13)
        rc('text.latex', preamble=r'\usepackage{cmbright}')

    fig, ax = plt.subplots()
    clrs = sns.color_palette("husl", 5)
    labels = ["Combined Agent", "NSC-kSP-FDL", "CaLRC-kSP-FF"]
    with sns.axes_style("darkgrid"):
        for i, df in enumerate([combined_eval_df, heur_eval_df, heur_eval_1_df]):
            ax.plot(df["load"], df["blocking"], label=labels[i], c=clrs[i])
            ax.fill_between(df["load"], df["blocking"] - df["blocking_std"], df["blocking"] + df["blocking_std"], alpha=0.3, facecolor=clrs[i])
            ax.legend()
            ax.set_yscale('log')

    #plt.plot(combined_eval_df["load"], combined_eval_df["blocking"], label="Agent", marker="d", color='g')
    #plt.plot(heur_eval_df["load"], heur_eval_df["blocking"], label="CaLRC-kSP-FF", marker="s", color='r')
    #plt.plot(heur_eval_1_df["load"], heur_eval_1_df["blocking"], label="NSC-kSP-FDL", marker="s", color='b')

    plt.legend()
    plt.ylabel(r"Blocking Probability")# [\%]")
    plt.xlabel("Traffic Load [Erlangs]")
    plt.yscale("log")
    plt.xlim(40, 100)
    plt.grid(True)
    plt.show()
