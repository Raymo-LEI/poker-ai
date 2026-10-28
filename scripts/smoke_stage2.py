"""Reproducible CPU-only full-size synthetic bridge -> 2 matches -> export smoke.

Creates untrained teachers in a new directory. Never opens a real run or GPU.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import torch
from alphanl.cfr.config import CFRConfig
from alphanl.cfr.model import RegretNetwork
from alphanl.transformer.config import ModelConfig
from alphanl.transformer.model import PokerTransformer
from alphanl.transformer.hunl import RULES_VERSION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1); torch.manual_seed(17)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    archive = args.out_dir / "strategies"; archive.mkdir()
    config = CFRConfig(equity_features=True, device="cpu", model=ModelConfig(
        d_model=192, nhead=6, num_layers=4, dim_feedforward=512))
    network = RegretNetwork(PokerTransformer(**config.to_dict()["model"]), fallback=config.fallback)
    for t in (100,101,102):
        with torch.no_grad():
            network.regret_head.bias.copy_(torch.tensor([.1,.5,.2,.15,.05]) * (t/100))
        torch.save({"format":"sd_cfr_strategy_v2", "iteration":t, "weight":t, "config":config.to_dict(),
                    "rules_version":RULES_VERSION, "numeric_version":config.numeric_version,
                    "models":{"shared":network.state_dict()}}, archive / f"iteration_{t:07d}.pt")
    (archive / "index.json").write_text(json.dumps({"format":"sd_cfr_archive_v1","iterations":[100,101,102]}))
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    def run(script, *arguments):
        subprocess.run([sys.executable, "scripts/"+script, *map(str, arguments)], env=env, check=True)
    distill = args.out_dir / "distillation"
    run("distill_cfr_policy.py", "--strategies",archive,"--out-dir",distill,"--burn-in",100,"--end-iter",102,
        "--weight-power",2,"--train-hands",12,"--validation-hands",4,"--steps",2,"--batch-size",8,"--device","cpu")
    settings = json.loads(Path("configs/stage2/botzone_50hand.json").read_text())
    settings.update(reference=str(distill / "stage1_distilled_ref.pt"), strategies=str(archive))
    settings["ppo"] = {"epochs":1,"minibatch_size":64,"gamma":1.,"gae_lambda":1.}
    path = args.out_dir / "smoke_config.json"; path.write_text(json.dumps(settings))
    run("train_stage2.py", "--config",path,"--out-dir",args.out_dir/"ppo","--device","cpu","--updates",2)
    run("export_stage2_agent.py","--checkpoint",args.out_dir/"ppo/train.pt","--output",args.out_dir/"agent.py","--decisions",300)
    (args.out_dir / "SMOKE_ONLY.txt").write_text("Synthetic untrained teachers; validates interfaces/CPU execution only. No strength or final Stage 1 acceptance.\n")

if __name__ == "__main__":
    main()
