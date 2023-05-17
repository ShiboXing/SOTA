import os
import subprocess as sp

DATA_HOME = os.path.expanduser("~")
IS_UNIX = os.path.exists("/etc/os-release")

def download_file(uri, out_file):
    out_file = os.path.expanduser(out_file)
    if os.path.exists(out_file):
        print("download path exists: ", out_file)
        return
    
    if IS_UNIX:
        sp.run(f"curl '{uri}' -o '{out_file}'", shell=True, check=True)
    else:
        sp.run(f"powershell \"Invoke-WebRequest -Uri '{uri}' -OutFile '{out_file}'\"", shell=True, check=True)

def decompress_file(pth, out_dir):
    pth = os.path.expanduser(pth)
    out_dir = os.path.expanduser(out_dir)
    if os.path.exists(out_dir):
        print("download path exists: ", pth)
        return
    
    dir_pth = os.path.dirname(pth)
    if IS_UNIX:
        sp.run(f"unzip {pth}", shell=True, check=True)
    else:
        sp.run(f"powershell \"Expand-Archive -Path '{pth}' -DestinationPath '{out_dir}'", shell=True, check=True)
