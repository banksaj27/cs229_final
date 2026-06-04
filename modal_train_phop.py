import modal
import os
import subprocess

# =========================
# modal setup
# =========================

app = modal.App("phop")
vol = modal.Volume.from_name("phop-checkpoints", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "triton", "numpy", "transformers")
    .add_local_dir(".", remote_path="/root/project")
)

@app.function(
    image=image,
    gpu="A100",
    timeout=3600 * 12,
    volumes={"/root/checkpoints": vol},
)
def train(K: int, L: int):
    env = os.environ.copy()
    env["PHOP_K"] = str(K)
    env["PHOP_L"] = str(L)
    subprocess.run(["python", "/root/project/train_phop.py"], check=True, env=env)

@app.local_entrypoint()
def main(k: int = 6, l: int = 1):
    print(f"Training ({k}⊗{l}) on p-hop")
    call = train.spawn(k, l)
    print(f"Job spawned: {call.object_id}")