import numpy as np
import matplotlib
import matplotlib.pyplot as plt 

matplotlib.rcParams["axes.linewidth"] = 1.6

colors = ["#9403fc", "#fcba03", "#db2727"]

h_accs_path = "data/h_accs_4.npy"
q_accs_path = "data/q_accs_4.npy"
pc_accs_path = "data/pc_accs_4.npy"
fig_path = "imgs/accuracy_hybrid.png"
n_steps = 50


if __name__ == "__main__":
    h_accs = np.load(h_accs_path)[0:n_steps]
    q_accs = np.load(q_accs_path)[0:n_steps]
    pc_accs = np.load(pc_accs_path)[0:n_steps]

    plt.plot(h_accs, label="Hybrid", color=colors[0], lw=3)
    plt.plot(q_accs, label="Amortised", color=colors[1], lw=3)
    plt.plot(pc_accs, label="Predictive Coding", color=colors[2], lw=3)
    plt.xlabel("Number of epochs")
    plt.ylabel("MNIST accuracy (%)")
    plt.title("MNIST learning")
    plt.legend()
    plt.savefig(fig_path)
    plt.close()

