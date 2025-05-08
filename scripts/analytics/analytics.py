import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd

# def plot_cum_success_against_steps(df, methods, map_name):
#     for method in methods:
#         df_method = df[(df["method"] == method) & (df["map"] == map_name)]
#         steps = np.concatenate(([0], df_method["steps_sum"].sort_values()))
#         y_vals = np.arange(0, len(steps)) / len(steps)

#         plt.plot(steps, y_vals, label=method)
#     plt.xlabel("Steps")
#     plt.ylabel("Cumulative Success Rate %")
#     plt.title(f"Cumulative Success Rate vs Steps for {method} on {map_name}")
#     plt.legend()
#     plt.grid()
#     plt.show()
LINE_WIDHT = 4


def plot_cum_success_against_x_all(df, methods: list[str], x_col: str, maps: list[str], x_col_label=None, color_dict=None):
    rows = math.ceil(len(maps) / 2)
    cols = 2
    fig, axs = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    # fig.suptitle("Cumulative Success Rate vs Steps for maps")
    for i, map_name in enumerate(maps):
        if rows > 1:
            ax = axs[i // cols, i % cols]
        else:
            ax = axs[i]
        for method in methods:
            df_method = df[(df["method"] == method) & (df["map"] == map_name)]
            steps = np.concatenate(([0], df_method[x_col].sort_values()))
            y_vals = np.arange(0, len(steps)) / (len(steps) - 1)
            if color_dict is not None:
                if method in color_dict:
                    ax.plot(steps, y_vals, label=method,
                            color=color_dict[method][0], ls=color_dict[method][1], linewidth=LINE_WIDHT)
                else:
                    raise ValueError(
                        f"Method {method} not found in color_dict")
            else:
                ax.plot(steps, y_vals, label=method, linewidth=LINE_WIDHT)
        ax.set_title(map_name)
        ax.ticklabel_format(style='scientific', axis='both', scilimits=(0, 0))
        if x_col_label is None:
            ax.set_xlabel(x_col)
        else:
            ax.set_xlabel(x_col_label)
        ax.set_ylabel("Success Rate")
        # ax.legend()
        ax.grid()
        # handles, labels = ax.get_legend_handles_labels()
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center',
               bbox_to_anchor=(0.5, -0.05), ncol=min(len(methods), 4))

    # Adjust layout to make space for the legend

    # Hide any unused subplots
    for j in range(i + 1, rows * cols):
        fig.delaxes(axs.flatten()[j])
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()
    return fig


def plot_cum_success_against_steps_all(df, methods: list[str], maps: list[str], color_dict=None):
    return plot_cum_success_against_x_all(df, methods, "steps_sum", maps, "Steps", color_dict)


def merge_by_method(view_table: pd.DataFrame, series: pd.Series, old_col_name: str, new_col_name: str = None) -> pd.DataFrame:
    """
    Merge a series into a dataframe by method
    """
    if new_col_name is None:
        new_col_name = old_col_name
    # Create a new dataframe with the same index as the view_table
    wanted = series.to_frame().reset_index()
    wanted.rename(columns={old_col_name: new_col_name}, inplace=True)

    view_table = pd.merge(view_table, wanted, on=["method"])
    return view_table


def merge_by_map_method(view_table: pd.DataFrame, series: pd.Series, old_col_name: str, new_col_name: str = None) -> pd.DataFrame:
    if new_col_name is None:
        new_col_name = old_col_name
    # Create a new dataframe with the same index as the view_table
    wanted = series.to_frame().reset_index()
    wanted.rename(columns={old_col_name: new_col_name}, inplace=True)

    view_table = pd.merge(view_table, wanted, on=["method", "map"])
    return view_table


def create_viewTable(df: pd.DataFrame):
    view_table = pd.DataFrame()
    view_table["method"] = df["method"].unique()
    view_table.sort_values(by="method", inplace=True)
    finished = df.groupby("method")["finished"].mean()
    view_table = merge_by_method(
        view_table, finished, "finished", "Success Rate")
    run_time = df.groupby("method")["tot_time"].mean()
    view_table = merge_by_method(
        view_table, run_time, "tot_time", "Mean Time (s)")
    run_time_finished = df[df["finished"] == True].groupby("method")[
        "tot_time"].mean()
    view_table = merge_by_method(
        view_table, run_time_finished, "tot_time", "Mean Time FO (s)")
    node_cnt = df.groupby("method")["node_cnt"].mean()
    view_table = merge_by_method(
        view_table, node_cnt, "node_cnt", "Mean Node Count")
    node_cnt_finished = df[df["finished"] == True].groupby("method")[
        "node_cnt"].mean()
    view_table = merge_by_method(
        view_table, node_cnt_finished, "node_cnt", "Mean Node Count FO")
    iterations = df.groupby("method")["iterations"].mean()
    view_table = merge_by_method(
        view_table, iterations, "iterations", "Mean Iterations")
    iterations_finished = df[df["finished"] == True].groupby("method")[
        "iterations"].mean()
    view_table = merge_by_method(
        view_table, iterations_finished, "iterations", "Mean Iterations FO")
    view_table = view_table.set_index("method")
    return view_table
