import matplotlib.pyplot as plt
import json

plt.style.use("seaborn")  # pretty matplotlib plots

programs = ["curl", "openssl", "httpd", "sqlite3", "libcrypto", "libcrypto-xarch", "libc"]

fig, ax1 = plt.subplots()
ax1.set_xlabel("epochs")
ax1.set_ylabel("hits score (%)")  # we already handled the x-label with ax1
ax1.tick_params(axis="y")

for target in programs:
    f = open(f"history-{target}-bow-seed50-epoch15000-D200.json")
    data = json.load(f)

    epoch_history = data["epoch"]
    loss_history = [float(x) for x in data["loss"]]
    hits_history = [float(x) for x in data["hits"]]

    epoch_history = [x for x in epoch_history if x <= 4000]
    loss_history = loss_history[: len(epoch_history)]
    hits_history = hits_history[: len(epoch_history)]

    ax1.plot(
        epoch_history,
        hits_history,
        lw=2,
        ls="-",
        alpha=0.5,
        label=f"{target}",
    )


ax1.legend()
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(f"GCN-layer2-200D-epochs-test-hits.png")
