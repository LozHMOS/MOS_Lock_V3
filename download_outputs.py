import anthropic
import os

client = anthropic.Anthropic()

with open("session_info.txt") as f:
    for line in f:
        if line.startswith("SESSION_ID="):
            SESSION_ID = line.strip().split("=", 1)[1]

OUTPUT_DIR = "/Users/lozhemmings/moslock_v3_build/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Downloading from session: {SESSION_ID}\n")

files = client.beta.files.list(
    session_id=SESSION_ID,
    betas=["files-api-2025-04-14"],
)

for f in files.data:
    name = getattr(f, "filename", f.id)
    print(f"  Downloading {name}...")
    resp = client.beta.files.download(f.id, betas=["files-api-2025-04-14"])
    out_path = os.path.join(OUTPUT_DIR, name)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as out:
        out.write(resp.read())
    print(f"     Saved: {out_path}")

print(f"\nAll files downloaded to {OUTPUT_DIR}")