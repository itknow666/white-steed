from pathlib import Path
import hashlib, urllib.request
URL = "https://github.com/moabitcoin/ig65m-pytorch/releases/download/v1.0.0/r2plus1d_34_clip32_ft_kinetics_from_ig65m-ade133f1.pth"
SHA = "ade133f18548b59da3a36d6a70d44a4a7cc4b90460654c338cb8a023e9a3e0e1"
def digest(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(4*1024*1024), b""): h.update(b)
    return h.hexdigest()
def main():
    p=Path("weights/pretrained.pth");p.parent.mkdir(exist_ok=True)
    if p.exists() and digest(p)==SHA:
        print(p, "verified");return
    temp=p.with_suffix(".pth.partial")
    with urllib.request.urlopen(URL,timeout=120) as response, temp.open("wb") as out:
        while True:
            block=response.read(4*1024*1024)
            if not block:break
            out.write(block)
    if digest(temp)!=SHA:raise ValueError("Pretrained weight SHA-256 mismatch")
    temp.replace(p);print(p,"verified")
if __name__=="__main__":main()
