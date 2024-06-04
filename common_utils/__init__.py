import os, sys
import subprocess as sp

IS_UNIX = sys.platform.startswith("linux") or sys.platform.startswith("darwin")
DATA_HOME = (
    os.path.expanduser("~/datasets")
    if IS_UNIX
    else os.path.expanduser("$HOME").join("datasets")
)


def join(pth1, pth2):
    if IS_UNIX:
        return pth1 + "/" + pth2
    else:
        return pth1 + "\\\\" + pth2


def set_data_home(pth):
    global DATA_HOME
    DATA_HOME = pth


def kaggle_download(cmd):
    # windows
    print(sp.run(cmd, shell=True, check=True, text=True))


def download_file(uri, out_file):
    out_file = os.path.expanduser(out_file)
    if os.path.exists(out_file):
        print("download path exists: ", out_file)
        return

    if IS_UNIX:
        sp.run(f"curl '{uri}' -o '{out_file}'", shell=True, check=True)
    else:
        sp.run(
            f"powershell \"Invoke-WebRequest -Uri '{uri}' -OutFile '{out_file}'\"",
            shell=True,
            check=True,
        )


def decompress_file(pth, out_dir):
    pth = os.path.join(DATA_HOME, pth)
    out_dir = os.path.join(DATA_HOME, out_dir)
    if os.path.exists(out_dir):
        print("extraction path exists: ", out_dir)
        return

    if IS_UNIX:
        sp.run(f"unzip {pth} -d {out_dir}", shell=True, check=True)
    else:
        print(
            sp.run(
                f"powershell \"Expand-Archive -Path '{pth}' -DestinationPath '{out_dir}'\"",
                shell=True,
                # check=True,
                capture_output=True,
            )
        )
