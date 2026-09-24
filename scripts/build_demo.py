#!/usr/bin/env python3
"""Export six paired test scenes from the existing experiment workspace.

Run in the look2hear environment. Inference stages use separate processes
because DAMSEP and SPMamba have different packages named look2hear.
The website itself has no Python or model dependency.
"""
import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf
import yaml
from scipy.signal import resample_poly

REPO = Path(__file__).resolve().parents[1]
METHODS = {
    "recrir": "Rec-RIR",
    "vinp_tcnsas_epoch120": "VINP",
    "fins_local_epoch250": "FiNS",
    "buddy": "BUDDy",
    "speech2rir": "Speech2RIR",
}
PARAMETERS = SimpleNamespace(
    pre_direct_ms=2.5, direct_half_ms=2.5, drr_analysis_pre_ms=5.0,
    drr_sensitivity_half_ms=(1.25, 2.5, 5.0), effective_response_seconds=1.0,
    tail_seconds=1.0, rir50_ms=50.0, edc_limit_db=-35.0,
    lsd_floor_db=-80.0, clarity_min_late_db=-80.0, min_decay_r2=0.9,
)


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False,
                               separators=(",", ":")) + "\n")


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare(workspace, cache):
    metadata = {r["utterance_id"]: r for r in json.loads(
        (REPO / "data/metadata/distance_test.json").read_text())}
    manifest = read_csv(workspace / "evaluation/rir_baselines/artifacts/dars_p10_xsep_16k/manifest.csv")
    near = [r for r in manifest if r["source"] == "1"]
    lookup = {(r["utterance"], r["source"]): r for r in manifest}
    near.sort(key=lambda r: (metadata[r["utterance"]]["s2"]["distance_to_left_mic"],
                             metadata[r["utterance"]]["s1"]["distance_to_left_mic"],
                             r["utterance"]))
    selected = []
    for index, row in enumerate(near[:6]):
        sources = [lookup[(row["utterance"], str(s))] for s in (1, 2)]
        assert len({r["crop_start"] for r in sources}) == 1
        selected.append({"id": "scene-{:02d}".format(index + 1),
                         "selectionRank": index + 1, "geometry": metadata[row["utterance"]],
                         "rows": sources})
    write_json(cache / "selection.json", selected)
    print("Selected the six closest pairs by farther-source distance, independently of model scores.", flush=True)


def load_experiment(workspace, model_name, device):
    if model_name == "damsep":
        root = workspace / "distant-Separation"
        experiment = root / "Experiments/checkpoint/mix_rir0_p10_0719"
    else:
        root = workspace / "SPMamba-distant"
        experiment = root / "Experiments/checkpoint/mixed_g3090_p5_0227"
    sys.path.insert(0, str(root))
    import torch
    import look2hear.models
    conf = yaml.safe_load((experiment / "conf.yml").read_text())
    cls = getattr(look2hear.models, conf["audionet"]["audionet_name"])
    model = cls.from_pretrain(str(experiment / "best_model.pth"), sample_rate=8000,
                             **conf["audionet"]["audionet_config"])
    model.eval().to(device)
    return model, experiment


def infer(workspace, cache, model_name, device):
    import torch
    torch.set_num_threads(4)
    torch.manual_seed(0)
    model, experiment = load_experiment(workspace, model_name, device)
    if model_name == "damsep":
        import evaluate_rir_metrics as metrics
        decoder = metrics.SweepRIRDecoder(8000, torch.device(device))
    selected = json.loads((cache / "selection.json").read_text())
    for scene in selected:
        row = scene["rows"][0]
        item = dict(key=row["utterance"], dataset_index=int(row["dataset_index"]),
                    crop_start=int(row["crop_start"]))
        mix_path = Path(row["anechoic_path"]).parent.parent / "mix_both_reverb" / row["utterance"]
        mixture, rate = sf.read(mix_path, start=item["crop_start"],
                               stop=item["crop_start"] + 32000, dtype="float32")
        assert rate == 8000 and mixture.shape == (32000,)
        with torch.inference_mode():
            output = model(torch.from_numpy(mixture).unsqueeze(0).to(device))
            if model_name == "damsep":
                raw = decoder.decode(metrics.decode_complex_ctf(output["rir"]))
                arrays = {"clean": output["x_derev"].reshape(2, -1).cpu().numpy(),
                          "reverberant": output["x_sep"].reshape(2, -1).cpu().numpy()}
                dr = []
                references = []
                target_drr, estimate_drr = [], []
                for source, source_row in enumerate(scene["rows"]):
                    full, full_rate = sf.read(source_row["full_rir_path"], always_2d=True)
                    direct, direct_rate = sf.read(source_row["direct_rir_path"], always_2d=True)
                    assert full_rate == direct_rate == 8000
                    record, _, estimated, target = metrics.evaluate_source(
                        item, source, raw[source], full[:, source], direct[:, source],
                        {n: float("nan") for n in metrics.RECONSTRUCTION_METRICS},
                        8000, PARAMETERS, metrics.acoustic_parameter_names(PARAMETERS.drr_sensitivity_half_ms))
                    dr.append(estimated); references.append(target)
                    target_drr.append(record["target_drr_db"])
                    estimate_drr.append(record["est_drr_db"])
                    # Assert that the rerun uses the same crop and source identity as the baseline inputs.
                    previous, old_rate = sf.read(source_row["input_path"], dtype="float32")
                    expected = resample_poly(arrays["reverberant"][source], 2, 1)
                    error = float(np.max(np.abs(expected - previous)))
                    assert old_rate == 16000 and error < 0.002, (scene["id"], error)
                arrays.update(rir=np.stack(dr), target_rir=np.stack(references),
                              target_drr=target_drr, estimate_drr=estimate_drr)
            else:
                arrays = {"clean": output.reshape(2, -1).cpu().numpy()}
        assert all(np.isfinite(value).all() for value in arrays.values())
        np.savez(cache / (scene["id"] + "_" + model_name + ".npz"), **arrays)
        print(model_name, scene["id"], "exported", flush=True)
    write_json(cache / (model_name + "_provenance.json"), {
        "checkpoint": str(experiment.relative_to(workspace) / "best_model.pth"),
        "sha256": sha256(experiment / "best_model.pth"), "assignment": "fixed source identity"})


def crop_anchor(wave, anchor):
    start = anchor - 20
    a = np.pad(wave[max(start, 0):max(start, 0) + 8020 - max(-start, 0)],
               (max(-start, 0), 8020))
    return a[:8020]


def series(rir, target):
    rir = np.asarray(rir, dtype=np.float64)
    assert rir.shape == (8020,) and np.isfinite(rir).all() and np.max(np.abs(rir)) > 0
    normalized = rir / np.max(np.abs(rir))
    if np.dot(rir, target) < 0:
        normalized = -normalized
    full = []
    for start in range(0, len(rir), 16):
        block = normalized[start:start + 16]
        for offset in sorted(set((int(np.argmin(block)), int(np.argmax(block))))):
            i = start + offset
            full.append([round((i - 20) / 8, 3), round(float(normalized[i]), 6)])
    power = rir ** 2
    edc = np.clip(10 * np.log10(np.maximum(np.cumsum(power[::-1])[::-1] / power.sum(), 1e-12)), -60, 0)
    return {"early": np.round(normalized[:421], 6).tolist(), "full": full,
            "edc": [[round((i - 20) / 8, 3), round(float(edc[i]), 4)]
                    for i in range(20, 8020, 16)]}


def package(workspace, cache):
    docs = REPO / "docs"
    selected = json.loads((cache / "selection.json").read_text())
    metric_maps = {}
    for method in METHODS:
        metric_maps[method] = {(r["utterance"], r["source"]): r for r in read_csv(
            workspace / "evaluation/rir_baselines/results" / method / "per_source_metrics.csv")}
    scenes = []
    for scene in selected:
        id_ = scene["id"]
        damsep = np.load(cache / (id_ + "_damsep.npz"))
        baseline = np.load(cache / (id_ + "_spmamba.npz"))
        row = scene["rows"][0]
        start = int(row["crop_start"])
        mix_path = Path(row["anechoic_path"]).parent.parent / "mix_both_reverb" / row["utterance"]
        audio = {"mixture": sf.read(mix_path, start=start, stop=start + 32000)[0]}
        for s, r in enumerate(scene["rows"]):
            for field, label in (("anechoic_path", "reference_clean"), ("reverberant_target_path", "reference_reverberant")):
                a, rate = sf.read(r[field], start=start, stop=start + 32000)
                assert rate == 8000
                audio[label + "_" + str(s)] = a
            audio["damsep_clean_" + str(s)] = damsep["clean"][s]
            audio["damsep_reverberant_" + str(s)] = damsep["reverberant"][s]
            audio["spmamba_clean_" + str(s)] = baseline["clean"][s]
        gain = .94 / max(float(np.max(np.abs(a))) for a in audio.values())
        audio_dir = docs / "audio" / id_
        audio_dir.mkdir(parents=True, exist_ok=True)
        audio_data = {}
        for label, a in audio.items():
            assert a.shape == (32000,) and np.isfinite(a).all()
            path = audio_dir / (label + ".wav")
            sf.write(path, a * gain, 8000, subtype="PCM_16")
            envelope = [float(np.max(np.abs(v))) for v in np.array_split(a * gain, 96)]
            audio_data[label] = {"src": str(path.relative_to(docs)),
                                 "peaks": np.round(envelope, 5).tolist()}
        responses = []
        for s, source_row in enumerate(scene["rows"]):
            target = damsep["target_rir"][s]
            curves = {
                "reference": dict(series(target, target), drr=float(damsep["target_drr"][s])),
                "damsep": dict(series(damsep["rir"][s], target), drr=float(damsep["estimate_drr"][s])),
            }
            for method in METHODS:
                path = workspace / "evaluation/rir_baselines/predictions" / method / "rir" / Path(source_row["input_path"]).name
                waveform, rate = sf.read(path)
                assert waveform.ndim == 1
                if rate != 8000:
                    gcd = int(np.gcd(rate, 8000))
                    waveform = resample_poly(waveform, 8000 // gcd, rate // gcd)
                aligned = crop_anchor(waveform, int(np.argmax(np.abs(waveform))))
                metric = metric_maps[method][(source_row["utterance"], source_row["source"])]
                assert int(metric["crop_start"]) == start
                assert abs(float(metric["target_drr_db"]) - float(damsep["target_drr"][s])) < 1e-4
                curves[method] = dict(series(aligned, target), drr=float(metric["est_drr_db"]))
            responses.append(curves)
        scenes.append({"id": id_, "utterance": row["utterance"],
                       "selectionRank": scene["selectionRank"],
                       "cropStartSamples": start, "geometry": scene["geometry"],
                       "audio": audio_data, "playbackGain": gain,
                       "responses": responses})
        print("Packaged", id_, flush=True)
    result = {"sampleRate": 8000, "duration": 4, "methods": METHODS,
              "selection": "The six eligible test mixtures with the smallest farther-source distance to the reference microphone, ordered by that distance. Ties use nearer-source distance, then utterance ID. Selection is independent of model scores.",
              "provenance": {k: json.loads((cache / (k + "_provenance.json")).read_text()) for k in ("damsep", "spmamba")},
              "scenes": scenes}
    (docs / "data/demo-data.js").write_text("window.DAMSEP_DEMO=" + json.dumps(
        result, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + ";\n")
    print("Finished:", len(list((docs / "audio").glob("*/*.wav"))), "audio clips.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--stage", choices=("all", "prepare", "damsep", "spmamba", "package"), default="all")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    cache = REPO / ".demo-cache"
    cache.mkdir(exist_ok=True)
    if args.stage == "all":
        for stage in ("prepare", "damsep", "spmamba", "package"):
            subprocess.run([sys.executable, __file__, "--workspace", str(workspace),
                            "--stage", stage, "--device", args.device], check=True)
    elif args.stage == "prepare":
        prepare(workspace, cache)
    elif args.stage == "package":
        package(workspace, cache)
    else:
        infer(workspace, cache, args.stage, args.device)


if __name__ == "__main__":
    main()
