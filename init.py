import os
import sys
import platform

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

if platform.system() == "Windows":
    INIT_FILE = os.path.join(os.environ["userprofile"], ".init.cmd") if len(sys.argv) == 1 else sys.argv[1]

    with open(INIT_FILE, "a") as init_file:
        init_file.write(f"\nset PYTHONPATH={ROOT_DIR};%PYTHONPATH%\n")

    os.system(f"reg add \"HKCU\Software\Microsoft\Command Processor\" /v AutoRun ^ \
      /t REG_EXPAND_SZ /d \"{INIT_FILE}\" /f")
else:
    os.system(f"echo \"export PYTHONPATH={ROOT_DIR}:$PYTHONPATH\" >> $HOME/.bashrc")
