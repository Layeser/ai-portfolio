"""Regenerate the figures and numbers of the deep-learning pages of docs/.

    PYTHONPATH=src python scripts/make_dl_figures.py                 # everything
    PYTHONPATH=src python scripts/make_dl_figures.py vae lenet       # some sections
    PYTHONPATH=src python scripts/make_dl_figures.py resnet --retrain

Sections: translation, autoencoder, vae, lenet, resnet. Needs MNIST and CIFAR-10 under ./data (downloaded on first
use). Timings on an RTX 5060 laptop GPU: translation ~5 min, autoencoder, VAE and LeNet ~1 min each, resnet ~50 min.
The ResNet runs are cached in figures/resnet/runs.json; without --retrain the figures are redrawn from that file.

GPU kernels are not bit-for-bit deterministic: a rerun can move the neural-network numbers in the last digit.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_figures import COLORS, FIG, save  # noqa: E402
from mlscratch.classical.pca import PCA  # noqa: E402
from mlscratch.deep_learning.autoencoder import load_mnist, reconstruction_mse, train_autoencoder  # noqa: E402
from mlscratch.deep_learning.lenet import normalize, predict, train_lenet  # noqa: E402
from mlscratch.deep_learning.resnet import ResNet18, ResNet34, load_cifar10, train_resnet  # noqa: E402
from mlscratch.deep_learning.vae import evaluate as vae_evaluate  # noqa: E402
from mlscratch.deep_learning.vae import train_vae  # noqa: E402
from mlscratch.nlp.en_fr_translation.data import encode, train_test_split  # noqa: E402
from mlscratch.nlp.en_fr_translation.train import exact_match, train  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def r(x: float, n: int = 4) -> float:
    return round(float(x), n)


def digits_row(axes, images, titles=None):
    for i, ax in enumerate(axes):
        ax.imshow(images[i].reshape(28, 28), cmap="gray_r", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        if titles is not None:
            ax.set_title(titles[i], fontsize=8)


# ------------------------------------------------------------------------------------------- translation
def translation_figures() -> dict:
    m: dict = {}
    # 1. How much of the grid must be seen? Held-out exact match against the share of templated pairs in training.
    fractions, seeds = [0.2, 0.4, 0.6, 0.8, 0.9], [0, 1, 2]
    cache = FIG / "transformer" / "sweep.json"  # 15 trainings: cached, delete the file to retrain
    done = json.loads(cache.read_text()) if cache.exists() else {}
    em = np.zeros((len(fractions), len(seeds)))
    for i, frac in enumerate(fractions):
        for j, seed in enumerate(seeds):
            key = f"{frac}/{seed}"
            if key not in done:
                tr, te = train_test_split(test_fraction=frac, seed=seed)
                t = train(tr, te, epochs=60, eval_every=60, seed=seed, verbose=False)
                done[key] = t.history["test_em"][-1]
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps(done, indent=1) + "\n")
            em[i, j] = done[key]
            print(f"  held out {frac:.0%}, seed {seed}: EM {em[i, j]:.3f}")
    seen = [round(100 * (1 - f)) for f in fractions]
    m["random_split"] = {"share_of_templates_seen_pct": seen, "heldout_em_mean": [r(x, 3) for x in em.mean(1)],
                         "heldout_em_min": [r(x, 3) for x in em.min(1)], "heldout_em_max": [r(x, 3) for x in em.max(1)]}

    # 2. Systematic split: "door" and "window" never appear with an adjective in training.
    tr, te = train_test_split(hold_out=lambda p: p[0].split()[1] in ("door", "window") and " is " in p[0])
    t = train(tr, te, epochs=60, eval_every=60, verbose=False)
    m["systematic_split"] = {"n_test": len(te), "heldout_em": r(exact_match(t, te), 3),
                             "examples": {e: t.translate(e) for e, _ in te[:3]}}

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.fill_between(seen, em.min(1), em.max(1), color=COLORS[0], alpha=0.15, linewidth=0, label="min–max, 3 seeds")
    ax.plot(seen, em.mean(1), "o-", color=COLORS[0], linewidth=1.8, label="random held-out combinations (mean)")
    ax.axhline(m["systematic_split"]["heldout_em"], color=COLORS[1], linestyle="--", linewidth=1.4,
               label='systematic: "the door / window is ADJ" never seen')
    ax.set_xlabel("share of the templated sentences seen in training (%)")
    ax.set_ylabel("held-out exact match")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Unseen combinations of seen words: easy when the grid is well covered;\n"
                 "an unseen use of a word: not at all")
    ax.legend(frameon=False, fontsize=8, loc="center right")
    save(fig, FIG / "transformer" / "generalisation.png")

    # 3. Cross-attention of the model trained on the default split, on two held-out sentences.
    tr, te = train_test_split()
    t = train(tr, te, epochs=60, eval_every=60, verbose=False)
    m["default_split"] = {"n_train": len(tr), "n_test": len(te), "heldout_em": r(exact_match(t, te), 3),
                          "examples": [{"en": e, "output": t.translate(e), "reference": f} for e, f in te[:8]]}
    picks = [next(p for p in te if " is " in p[0]), next(p for p in te if p[0].startswith(("i ", "we ", "they ")))]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6))
    m["attention_examples"] = []
    for ax, (en, fr) in zip(axes, picks, strict=True):
        out = t.translate(en)
        src = torch.tensor([encode(en, t.en_stoi, add_bos_eos=False)], device=DEVICE)
        tgt = torch.tensor([encode(out, t.fr_stoi)[:-1]], device=DEVICE)  # BOS + generated words
        att = t.model.cross_attention(src, tgt)[0].cpu().numpy()  # (T, S), last decoder layer
        ax.imshow(att, cmap="Greys", vmin=0, vmax=1)
        ax.set_xticks(range(len(en.split())), en.split(), rotation=30, ha="right")
        ax.set_yticks(range(att.shape[0]), [f"→ {w}" for w in out.split() + ["<eos>"]])  # last step predicts EOS
        ax.set_title(f'"{en}" (held out)', fontsize=9)
        m["attention_examples"].append({"en": en, "output": out, "reference": fr})
    fig.suptitle("Cross-attention, last decoder layer, mean over 4 heads: row = French word being predicted",
                 fontsize=9)
    save(fig, FIG / "transformer" / "attention.png")
    return m


# ------------------------------------------------------------------------------------------- autoencoder
def autoencoder_figures() -> dict:
    m: dict = {}
    X_tr, _, X_te, _ = load_mnist(device=DEVICE)
    ks = [2, 4, 8, 16, 32, 64]
    Xtr_np, Xte_np = X_tr.flatten(1).cpu().numpy(), X_te.flatten(1).cpu().numpy()
    pca = PCA(n_components=max(ks), solver="eigh").fit(Xtr_np)
    pca_mse, ae_mse, models = [], [], {}
    for k in ks:
        Z = (Xte_np - pca.mean_) @ pca.components_[:, :k]
        pca_mse.append(float(((Z @ pca.components_[:, :k].T + pca.mean_ - Xte_np) ** 2).mean()))
        model, _ = train_autoencoder(X_tr, latent_dim=k, epochs=10)
        ae_mse.append(reconstruction_mse(model, X_te))
        models[k] = model
        print(f"  k={k:2d}  test MSE  PCA {pca_mse[-1]:.4f}  autoencoder {ae_mse[-1]:.4f}")
    m.update({"latent_dims": ks, "test_mse_pca": [r(x) for x in pca_mse], "test_mse_autoencoder": [r(x) for x in ae_mse]})

    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(ks, pca_mse, "s--", color=COLORS[1], linewidth=1.6, label="PCA (linear)")
    ax.plot(ks, ae_mse, "o-", color=COLORS[0], linewidth=1.8, label="autoencoder 784-256-k (ReLU)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks, [str(k) for k in ks])
    ax.set_xlabel("code size k")
    ax.set_ylabel("test reconstruction MSE per pixel")
    ratio = ae_mse[ks.index(16)] / pca_mse[ks.index(16)]
    ax.set_title(f"MNIST, same code size: the non-linear encoder reconstructs better at every k\n"
                 f"(k = 16: {ratio:.0%} of the PCA error)", fontsize=10)
    ax.legend(frameon=False)
    save(fig, FIG / "autoencoder" / "vs_pca.png")

    k, n = 8, 10
    x = X_te[:n]
    with torch.no_grad():
        ae = models[k](x).flatten(1).cpu().numpy()
    xp = x.flatten(1).cpu().numpy()
    pc = ((xp - pca.mean_) @ pca.components_[:, :k]) @ pca.components_[:, :k].T + pca.mean_
    fig, axes = plt.subplots(3, n, figsize=(8.5, 2.9))
    for row, (imgs, name) in enumerate([(xp, "original"), (np.clip(pc, 0, 1), f"PCA, k={k}"),
                                        (ae, f"autoencoder, k={k}")]):
        digits_row(axes[row], imgs)
        axes[row, 0].set_ylabel(name, fontsize=8, rotation=0, ha="right", va="center")
    fig.suptitle(f"Ten test digits squeezed through {k} numbers", fontsize=10)
    save(fig, FIG / "autoencoder" / "reconstructions.png")
    return m


# ------------------------------------------------------------------------------------------------- VAE
def vae_figures() -> dict:
    m: dict = {}
    X_tr, y_tr, X_te, y_te = load_mnist(device=DEVICE)

    model, _ = train_vae(X_tr, latent_dim=16, epochs=20)
    m["latent16_beta1_test"] = {k: r(v, 2) for k, v in vae_evaluate(model, X_te).items()}
    samples = model.sample(40, torch.Generator().manual_seed(0)).flatten(1).cpu().numpy()
    with torch.no_grad():  # interpolate between the codes (means) of two test digits
        mu, _ = model.encode(X_te[[3, 2]])
        alphas = torch.linspace(0, 1, 10, device=DEVICE)[:, None]
        line = model.decode((1 - alphas) * mu[0] + alphas * mu[1]).flatten(1).cpu().numpy()
    fig, axes = plt.subplots(5, 10, figsize=(8.5, 4.6))
    for i in range(4):
        digits_row(axes[i], samples[10 * i: 10 * i + 10])
    digits_row(axes[4], line)
    axes[0, 0].set_ylabel("z ~ N(0, I)", fontsize=8)
    axes[4, 0].set_ylabel("from 0 to 1", fontsize=8)
    fig.suptitle("Top: 40 digits decoded from random codes. Bottom: a straight line between the codes of two test "
                 "digits", fontsize=9)
    save(fig, FIG / "vae" / "samples.png")

    model2, _ = train_vae(X_tr, latent_dim=2, epochs=20)
    with torch.no_grad():
        mu, _ = model2.encode(X_te)
        g = torch.linspace(-2.5, 2.5, 12)
        zz = torch.stack(torch.meshgrid(g, -g, indexing="xy"), -1).reshape(-1, 2).to(DEVICE)
        grid = model2.decode(zz).cpu().numpy().reshape(12, 12, 28, 28).transpose(0, 2, 1, 3).reshape(12 * 28, 12 * 28)
    mu, y = mu.cpu().numpy(), y_te.cpu().numpy()
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6))
    cmap = plt.get_cmap("tab10")
    for d in range(10):
        s = (y == d)
        axes[0].scatter(mu[s, 0], mu[s, 1], s=2, color=cmap(d), alpha=0.5, rasterized=True)
        cx, cy = np.median(mu[s], 0)
        axes[0].text(cx, cy, str(d), fontsize=12, weight="bold", ha="center", va="center",
                     bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": cmap(d), "alpha": 0.85})
    axes[0].set_xlim(-3.5, 3.5)
    axes[0].set_ylim(-3.5, 3.5)
    axes[0].set_aspect("equal")
    axes[0].set_title("10 000 test digits, mean of q(z|x)", fontsize=9)
    axes[1].imshow(grid, cmap="gray_r", extent=(-2.5 - 0.23, 2.5 + 0.23, -2.5 - 0.23, 2.5 + 0.23))
    axes[1].set_title("decoder output over the same plane", fontsize=9)
    for ax in axes:
        ax.set_xlabel("$z_1$")
        ax.set_ylabel("$z_2$")
    fig.suptitle("A VAE with a 2-D code: digits are arranged by shape, and every point of the plane decodes to a digit",
                 fontsize=10)
    save(fig, FIG / "vae" / "latent_2d.png")

    betas, rec, kl = [0.25, 0.5, 1, 2, 4, 8], [], []
    for b in betas:
        mb, _ = train_vae(X_tr, latent_dim=16, beta=b, epochs=10)
        e = vae_evaluate(mb, X_te)
        rec.append(e["recon"])
        kl.append(e["kl"])
        print(f"  beta={b}: recon {e['recon']:.1f}  KL {e['kl']:.1f}")
    m["beta_sweep"] = {"beta": betas, "test_recon": [r(x, 1) for x in rec], "test_kl": [r(x, 1) for x in kl]}
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(betas, rec, "o-", color=COLORS[0], linewidth=1.8, label="reconstruction term (BCE)")
    ax2 = ax.twinx()
    ax2.plot(betas, kl, "s--", color=COLORS[1], linewidth=1.6, label="KL term")
    ax.set_xscale("log", base=2)
    ax.set_xticks(betas, [str(b) for b in betas])
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel("reconstruction, nats per image", color=COLORS[0])
    ax2.set_ylabel("KL, nats per image", color=COLORS[1])
    ax2.spines["right"].set_visible(True)
    ax.set_title(r"Test set, 16-D code: $\beta$ moves the balance between the two terms", fontsize=10)
    fig.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 0.82))
    save(fig, FIG / "vae" / "beta.png")
    return m


# ----------------------------------------------------------------------------------------------- LeNet
def lenet_figures() -> dict:
    m: dict = {}
    X_tr, y_tr, X_te, y_te = load_mnist(device=DEVICE)
    model, h = train_lenet(normalize(X_tr), y_tr, normalize(X_te), y_te, epochs=10)
    m["test_acc_per_epoch"] = [r(a) for a in h["test_acc"]]
    _, h_const = train_lenet(normalize(X_tr), y_tr, normalize(X_te), y_te, epochs=10, cosine=False)
    m["test_acc_per_epoch_constant_lr"] = [r(a) for a in h_const["test_acc"]]
    ep = np.arange(1, 11)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(ep, 100 * np.array(h_const["test_acc"]), "s--", color=COLORS[1], linewidth=1.5, markersize=4,
            label="constant learning rate 1e-3")
    ax.plot(ep, 100 * np.array(h["test_acc"]), "o-", color=COLORS[0], linewidth=1.8, markersize=4,
            label="cosine decay from 1e-3 to 0")
    ax.set_xlabel("epoch")
    ax.set_ylabel("test accuracy (%)")
    ax.set_xticks(ep)
    ax.set_title("LeNet-5 on MNIST, same seed and data order:\nthe schedule turns a noisy plateau into a steady finish",
                 fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    save(fig, FIG / "lenet" / "schedule.png")
    m["params"] = sum(p.numel() for p in model.parameters())
    pred = predict(model, normalize(X_te))
    wrong = (pred != y_te).nonzero().flatten().cpu().numpy()
    m["test_errors"] = int(len(wrong))
    conf = np.zeros((10, 10), int)
    np.add.at(conf, (y_te.cpu().numpy(), pred.cpu().numpy()), 1)
    np.fill_diagonal(conf, 0)
    i, j = np.unravel_index(conf.argmax(), conf.shape)
    m["most_common_confusion"] = {"true": int(i), "predicted": int(j), "count": int(conf[i, j])}

    # What the first layer computes on one digit
    x = X_te[:1]
    with torch.no_grad():
        a1, _ = model.features(normalize(x))
    w = model.conv1.weight.detach().cpu().numpy()[:, 0]
    fig, axes = plt.subplots(2, 7, figsize=(8.5, 2.9))
    digits_row([axes[0, 0]], x.flatten(1).cpu().numpy(), ["input"])
    axes[1, 0].axis("off")
    for c in range(6):
        ax = axes[0, c + 1]
        lim = np.abs(w[c]).max()
        ax.imshow(w[c], cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax.set_title(f"filter {c + 1}", fontsize=8)
        axes[1, c + 1].imshow(a1[0, c].cpu().numpy(), cmap="gray_r")
        for a in (ax, axes[1, c + 1]):
            a.set_xticks([])
            a.set_yticks([])
            for sp in a.spines.values():
                sp.set_visible(False)
    axes[1, 1].set_ylabel("response\n(ReLU, pool)", fontsize=8)
    fig.suptitle("conv1: six 5×5 filters learned from data (red +, blue −) and their maps on one test digit",
                 fontsize=9)
    save(fig, FIG / "lenet" / "first_layer.png")

    show = wrong[:12]
    fig, axes = plt.subplots(1, 12, figsize=(9, 1.4))
    digits_row(axes, X_te[show].flatten(1).cpu().numpy(),
               [f"{int(y_te[k])}→{int(pred[k])}" for k in show])
    fig.suptitle(f"The first 12 of the {len(wrong)} test errors (true → predicted)", fontsize=9)
    save(fig, FIG / "lenet" / "errors.png")
    return m


# ---------------------------------------------------------------------------------------------- ResNet
RUNS = {  # name: (builder, residual, augment)
    "resnet18_aug": (ResNet18, True, True),
    "resnet18_noaug": (ResNet18, True, False),
    "plain18_aug": (ResNet18, False, True),
    "resnet34_aug": (ResNet34, True, True),
    "plain34_aug": (ResNet34, False, True),
}


def resnet_figures(retrain: bool = False, epochs: int = 20) -> dict:
    cache = FIG / "resnet" / "runs.json"
    runs = json.loads(cache.read_text()) if cache.exists() else {}
    if retrain or set(RUNS) - set(runs):
        X_tr, y_tr, X_te, y_te = load_cifar10(device=DEVICE)
        for name, (build, residual, aug) in RUNS.items():
            if name in runs and not retrain:
                continue
            print(f"  training {name}")
            model = build(residual=residual).to(DEVICE).to(memory_format=torch.channels_last)
            t0 = time.perf_counter()
            h = train_resnet(model, X_tr, y_tr, X_te, y_te, epochs=epochs, use_augment=aug, verbose=False)
            h["minutes"] = r((time.perf_counter() - t0) / 60, 1)
            h["params"] = sum(p.numel() for p in model.parameters())
            runs[name] = {k: ([r(x) for x in v] if isinstance(v, list) else v) for k, v in h.items()}
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(runs, indent=1) + "\n")  # saved after each run: an interruption loses one
            print(f"    final test acc {h['test_acc'][-1]:.4f} in {h['minutes']} min")

    m = {name: {"final_test_acc": v["test_acc"][-1], "final_train_acc": v["train_acc"][-1],
                "final_train_loss": v["loss"][-1], "params": v["params"], "minutes": v["minutes"]}
         for name, v in runs.items()}
    ep = np.arange(1, len(runs["resnet18_aug"]["test_acc"]) + 1)

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for name, color, label in [("resnet18_aug", COLORS[0], "with augmentation"),
                               ("resnet18_noaug", COLORS[1], "without")]:
        ax.plot(ep, runs[name]["train_acc"], ":", color=color, linewidth=1.4)
        ax.plot(ep, runs[name]["test_acc"], "-", color=color, linewidth=1.8, label=f"{label}: test")
    ax.plot([], [], ":", color="grey", label="dotted: train (on augmented images when on)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0.4, 1.0)
    a, n = runs["resnet18_aug"]["test_acc"][-1], runs["resnet18_noaug"]["test_acc"][-1]
    ax.set_title(f"ResNet-18, CIFAR-10: without augmentation the training set is memorised (100 %);\n"
                 f"random crops and flips raise test accuracy from {100 * n:.2f} % to {100 * a:.2f} %", fontsize=10)
    ax.set_xticks(ep[1::2])
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    save(fig, FIG / "resnet" / "augmentation.png")

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8))
    for name, color, style, label in [("plain18_aug", COLORS[1], "--", "plain-18"),
                                      ("plain34_aug", COLORS[1], "-", "plain-34"),
                                      ("resnet18_aug", COLORS[0], "--", "ResNet-18"),
                                      ("resnet34_aug", COLORS[0], "-", "ResNet-34")]:
        axes[0].plot(ep, runs[name]["loss"], style, color=color, linewidth=1.6, label=label)
        axes[1].plot(ep, runs[name]["test_acc"], style, color=color, linewidth=1.6, label=label)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("training loss (log scale)")
    axes[1].set_ylabel("test accuracy")
    axes[1].set_ylim(0.5, 1.0)
    for ax in axes:
        ax.set_xlabel("epoch")
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")
    for ax in axes:
        ax.set_xticks(ep[1::2])
    fig.suptitle("Without skip connections, 34 layers train worse than 18; with them, depth stops hurting\n"
                 "(CIFAR-10, same layers, same 20-epoch schedule)", fontsize=10)
    save(fig, FIG / "resnet" / "skip_connections.png")
    return m


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sections", nargs="*", default=["translation", "autoencoder", "vae", "lenet", "resnet"])
    parser.add_argument("--retrain", action="store_true", help="retrain the ResNet runs instead of using the cache")
    args = parser.parse_args()
    fns = {"translation": ("transformer", translation_figures), "autoencoder": ("autoencoder", autoencoder_figures),
           "vae": ("vae", vae_figures), "lenet": ("lenet", lenet_figures),
           "resnet": ("resnet", lambda: resnet_figures(args.retrain))}
    print("device:", DEVICE)
    for section in args.sections:
        folder, fn = fns[section]
        print(section)
        t0 = time.perf_counter()
        m = fn()
        (FIG / folder).mkdir(parents=True, exist_ok=True)
        (FIG / folder / "measurements.json").write_text(json.dumps(m, indent=2, ensure_ascii=False) + "\n")
        for k, v in m.items():
            print(f"  {k}: {v}")
        print(f"  ({time.perf_counter() - t0:.0f} s)")


if __name__ == "__main__":
    main()
