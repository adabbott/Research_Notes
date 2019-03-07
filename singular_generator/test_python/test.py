#import os
#import signal
#import subprocess
#
#proc = subprocess.Popen("Singular singular.inp >> output", stdout=subprocess.PIPE, shell=True, preexec_fn=os.setsid)
#os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
#stuff = subprocess.run("Singular singular.inp", shell=False)
#os.kill(stuff)
#subprocess.call("exit;")

#subprocess.kill(stuff)

import subprocess
import psutil


def kill(proc_pid):
    process = psutil.Process(proc_pid)
    for proc in process.children(recursive=True):
        proc.kill()
    process.kill()


proc = subprocess.Popen(["Singular singular.inp >> output"], shell=True)
try:
    proc.wait(timeout=10)
except subprocess.TimeoutExpired:
    kill(proc.pid)


proc = subprocess.Popen(["Singular singular.inp >> output1"], shell=True)
try:
    proc.wait(timeout=10)
except subprocess.TimeoutExpired:
    kill(proc.pid)
