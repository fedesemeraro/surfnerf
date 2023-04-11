import numpy as np
import os.path
from matplotlib import pyplot as plt
from scipy import interpolate


## Importing data
# 4: datasets: 004, 068, 214, 260
# 10: cases: 0 base, 1 aug, 2 viewdirs, 3 aug_viewdirs, 4 aug_viewdirs_low-netdepth, 5 aug_viewdirs_high-netdepth,
#            6 aug_viewdirs_low-netwidth, 7 aug_viewdirs_high-netwidth, 8 aug_viewdirs_low-posenc, 9 aug_viewdirs_high-posenc
# 10: iterations
# 3: iterations number, PSNR, SSIM
results = np.zeros((4, 10, 100, 3), dtype=float)

datasets = ["4", "68", "214", "260"]
cases = ["", "_aug", "_viewdirs", "_aug_viewdirs", "_aug_viewdirs_low-netdepth", "_aug_viewdirs_high-netdepth",
         "_aug_viewdirs_low-netwidth", "_aug_viewdirs_high-netwidth", "_aug_viewdirs_low-posenc", "_aug_viewdirs_high-posenc"]

filepath = "/Users/fsemerar/Downloads/hyper/"

for d, dataset in enumerate(datasets):
    for c, case in enumerate(cases):
        filename = filepath + "sat" + dataset + case + "/training_log.txt"

        if not os.path.isfile(filename):
            continue

        with open(filename) as f:
            lines = f.readlines()

        counter = 0
        for line in lines:
            if "TEST" not in line:
                numbers = line[:-2].split(",")

                for i, n in enumerate(numbers):
                    results[d, c, counter, i] = float(n)
                counter += 1

## Plotting
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
          'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

dataset_names = ["004", "068", "214", "260"]
case_names = ["Base", "Augmented", "View Dirs", "Aug. + View Dirs",
              "Aug. + View Dirs + Net Depth=4", "Aug. + View Dirs + Net Depth=16",
              "Aug. + View Dirs + Net Width=128", "Aug. + View Dirs + Net Width=512",
              "Aug. + View Dirs + Pos. Enc.=(5,2)",
              "Aug. + View Dirs + Pos. Enc.=(20,8)"]

fig, axs = plt.subplots(nrows=4, ncols=1, constrained_layout=True, figsize=(7.5, 10))
for ax in axs:
    ax.remove()

gridspec = axs[0].get_subplotspec().get_gridspec()
subfigs = [fig.add_subfigure(gs) for gs in gridspec]

for d, subfig in enumerate(subfigs):
    subfig.suptitle(f"Scene {dataset_names[d]}")

    axs = subfig.subplots(nrows=1, ncols=2)
    for c in range(results.shape[1]):
        if results[d, c, :, 1].mean() < 10:
            continue
        axs[0].plot(results[d, c, :, 0], results[d, c, :, 1], c=colors[c])
        axs[1].plot(results[d, c, :, 0], results[d, c, :, 2], c=colors[c])

    axs[0].set_xlabel("Iteration #")
    axs[0].set_ylabel("PSNR")
    axs[0].grid(visible=True)

    axs[1].set_xlabel("Iteration #")
    axs[1].set_ylabel("SSIM")
    axs[1].grid(visible=True)

plt.savefig("/Users/fsemerar/Downloads/hyper_study.png", dpi=500)
# plt.show()
plt.close('all')

# This is a dummy plot only for the legend
for c in range(results.shape[1]):
    plt.scatter(results[0, c, :, 0], results[0, c, :, 1], c=colors[c], label=case_names[c])
plt.legend(framealpha=1)
plt.savefig("/Users/fsemerar/Downloads/hyper_legend.png", dpi=500)
# plt.show()



## Processing production runs for NeRF and S-NeRF
results = np.zeros((4, 2, 1000, 3), dtype=float)

datasets = ["4", "68", "214", "260"]

filepath = ["/Users/fsemerar/Downloads/optimal/", "/Users/fsemerar/Downloads/snerf/"]
file_end = ["_optimal2", "_snerf"]

for c, case in enumerate(file_end):
    for d, dataset in enumerate(datasets):
        filename = filepath[c] + "sat" + dataset + case + "/training_log.txt"

        if not os.path.isfile(filename):
            print(f"Dataset {dataset} and case {case} not found!")
            continue

        with open(filename) as f:
            lines = f.readlines()

        counter = 0
        last_epoch = False
        prev_it = 0
        for line in lines:
            if "TEST" not in line:
                numbers = line[:-2].split(",")
                it = float(numbers[0])

                if it > prev_it:
                    for i, n in enumerate(numbers):
                        results[d, c, counter, i] = float(n)
                        if float(n) >= 100000:
                            last_epoch = True
                    counter += 1
                    prev_it = it
                else:
                    break

            if last_epoch:
                break

## Plotting
dataset_names = ["004", "068", "214", "260"]
case_names = ["NeRF", "S-NeRF"]
colors = ["g", "b"]

fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(7.5, 10))
for d, dataset_name in enumerate(dataset_names):
    for c, case_name in enumerate(case_names):
        mask = results[d, c, :, 1] > 0
        x_int = np.linspace(results[d, c, :, 0][mask].min(), results[d, c, :, 0][mask].max(), 500)
        tck = interpolate.splrep(results[d, c, :, 0][mask], results[d, c, :, 1][mask], k=3, s=800)
        y_int = interpolate.splev(x_int, tck, der=0)
        axs[d].plot(x_int, y_int, colors[c], label=case_name)
        axs[d].scatter(results[d, c, :, 0][mask], results[d, c, :, 1][mask], marker='.', s=0.5, c=colors[c])  # psnr
        axs[d].set_xlabel("Iteration #")
        axs[d].set_ylabel("PSNR")
        axs[d].grid(visible=True)
        axs[d].set_title(f"Scene {dataset_names[d]}")
        axs[d].legend(loc="lower right")
plt.tight_layout()
plt.savefig("/Users/fsemerar/Downloads/nerf_snerf.png", dpi=500)
# plt.show()
