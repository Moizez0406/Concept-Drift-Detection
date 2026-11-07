from capymoa.stream.drift import DriftStream, AbruptDrift
from capymoa.stream.generator import SEA
from capymoa.classifier import HoeffdingTree
from capymoa.drift.detectors import ADWIN
import matplotlib.pyplot as plt
import numpy as np

stream = DriftStream(stream=[
    SEA(function=1), AbruptDrift(position=2500),
    SEA(function=2), AbruptDrift(position=5000),
    SEA(function=3), AbruptDrift(position=7500),
    SEA(function=4),
])

learner = HoeffdingTree(stream.get_schema())
detector = ADWIN()

errors, detections = [], []
X_all, y_all = [], []
for t in range(10000):
    inst = stream.next_instance()
    pred = learner.predict(inst)
    err = int(pred != inst.y_index)
    errors.append(err)
    X_all.append(inst.x)
    y_all.append(inst.y_index)

    detector.add_element(err)
    if detector.detected_change():
        detections.append(t)

    learner.train(inst)
X_all = np.array(X_all); y_all = np.array(y_all); errors = np.array(errors)
w = 200
rolling_err = np.convolve(errors, np.ones(w) / w, mode="valid")
plt.plot(rolling_err)
for d in (2500, 5000, 7500):
    plt.axvline(d, color="r", ls="--")
for d in detections:
    plt.axvline(d, color="g", alpha=0.5)
plt.ylabel("rolling error rate"); plt.xlabel("instance")
plt.show()

THRESH = [8, 9, 7, 9.5]
xs = np.linspace(0, 10, 100)

fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), sharex=True, sharey=True)
for k, ax in enumerate(axes):
    sl = slice(k * 2500, (k + 1) * 2500)
    X, y, e = X_all[sl], y_all[sl], errors[sl].astype(bool)

    ax.scatter(X[y == 0, 0], X[y == 0, 1], s=6, alpha=0.5, c="tab:blue")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], s=6, alpha=0.5, c="tab:orange")
    ax.scatter(X[e, 0], X[e, 1], s=25, facecolors="none", edgecolors="k", lw=0.6)
    ax.plot(xs, THRESH[k] - xs, "k--", lw=1.5)

    ax.set_title(f"Concept {k + 1}: t = {k * 2500}-{(k + 1) * 2500}")
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.set_xlabel("x1")
axes[0].set_ylabel("x2")
plt.tight_layout(); plt.show()
plt.figure(figsize=(12, 3.5))
plt.scatter(np.arange(10000), X_all[:, 0] + X_all[:, 1], c=y_all, cmap="coolwarm", s=3, alpha=0.5)
for d, th in zip((0, 2500, 5000, 7500), THRESH):
    plt.hlines(th, d, d + 2500, color="k", ls="--")
for d in detections:
    plt.axvline(d, color="g", alpha=0.5)
plt.xlabel("instance"); plt.ylabel("x1 + x2")
plt.show()
