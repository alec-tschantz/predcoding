import numpy as np
import matplotlib.pyplot as plt 

h_accs_path = "data/h_accs_3.npy"
q_accs_path = "data/q_accs_3.npy"
pc_accs_path = "data/pc_accs_2.npy"
fig_path = "imgs/accuracy_hybrid.png"

if __name__ == "__main__":
    h_accs = np.load(h_accs_path)
    q_accs = np.load(q_accs_path)
    pc_accs = np.load(pc_accs_path)

    plt.plot(h_accs, label="hybrid")
    plt.plot(q_accs, label="amortised")
    plt.plot(pc_accs, label="pc")
    plt.legend()
    plt.savefig(fig_path)
    plt.close()