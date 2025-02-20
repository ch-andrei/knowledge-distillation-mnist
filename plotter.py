from pprint import pprint

import numpy as np
from matplotlib import pyplot as plt
from trainer import load_checkpoint


def get_colors_from_temperatures(temperatures):
    if 1 < len(temperatures):
        temperatures_n = np.array(temperatures, float)
        # normalize to 0-1
        temperatures_n = np.log(temperatures_n)
        temperatures_n -= temperatures_n.min()
        temperatures_n /= temperatures_n.max()
        temperatures_n = temperatures_n
        # apply color map
        colors = plt.cm.viridis(temperatures_n)
        colors = {temperature: color for temperature, color in zip(temperatures, colors)}
        return colors
    else:
        return {temperatures[0]: 'g'}


def do_plots(hidden_dims, temperatures, run_tag, plot_tag):
    colors = get_colors_from_temperatures(temperatures)

    for plot_accuracy in [False, True]:
        plot_tag_ = plot_tag + ("acc" if plot_accuracy else "err")

        def get_test_metric(stat):
            # stat format: loss, correct_train, incorrect_train, correct_test, incorrect_teste
            return stat[3] / (stat[3] + stat[4]) if plot_accuracy else stat[4]

        checkpoint_teacher = load_checkpoint(f"{run_tag}teacher/ckpt_teacher.pth")
        acc_teacher = [get_test_metric(stat) for stat in checkpoint_teacher["epoch_stats"]]

        print(f"Teacher final {('acc' if plot_accuracy else 'err')}: ",
              get_test_metric(checkpoint_teacher["epoch_stats"][-1]))

        for hidden_dim in hidden_dims:
            print(f'Plotting for hidden_dim={hidden_dim}...')

            plt.figure()

            # get result no KD
            result_path = f"student-{hidden_dim}"
            checkpoint = load_checkpoint(f"{run_tag}{result_path}/ckpt_student.pth")
            acc_student = [get_test_metric(stat) for stat in checkpoint["epoch_stats"]]

            # plot accuracy teacher and student no KD
            plt.plot(list(range(len(acc_teacher))), acc_teacher, "r--", label=f"Teacher")
            plt.plot(list(range(len(acc_student))), acc_student, "k--", label=f"No KD")

            print(f"Student no KD final {('acc' if plot_accuracy else 'err')}: ",
                  get_test_metric(checkpoint["epoch_stats"][-1]))

            for temperature in temperatures:
                print(f'Plotting for hidden_dim={hidden_dim}, temperature={temperature}...')

                result_path = f"student-{hidden_dim}-{temperature}"
                checkpoint = load_checkpoint(f"{run_tag}{result_path}/ckpt_student.pth")
                acc_student = [get_test_metric(stat) for stat in checkpoint["epoch_stats"]]

                print(f"Student KD T={temperature} final {('acc' if plot_accuracy else 'err')}: ",
                      get_test_metric(checkpoint["epoch_stats"][-1]))

                color = colors[temperature]

                # plot acc student with KD
                plt.plot(list(range(len(acc_student))), acc_student, "-", color=color, label=f"KD T={temperature}")

            plt.xlabel("Epochs")
            plt.ylabel("Accuracy" if plot_accuracy else "Num errors")
            plt.legend()
            plt.grid()
            plt.draw()

            figname = f"{hidden_dim}-{plot_tag_}.pdf"
            print(f'Saving figure {figname}.')
            plt.savefig(figname)


def do_plots_all():
    # hidden_dims = [800]
    # temperatures = [20]
    hidden_dims = [30, 300, 800]
    temperatures = [1, 2, 4, 10, 20, 50]
    run_tag = "run3-"
    # run_tag = ""
    do_plots(hidden_dims, temperatures, run_tag, plot_tag="")


def do_plots_800_20():
    hidden_dims = [800]
    temperatures = [20]
    run_tag = "run3-"
    do_plots(hidden_dims, temperatures, run_tag, plot_tag="T20")


if __name__ == "__main__":
    do_plots_800_20()
    do_plots_all()
