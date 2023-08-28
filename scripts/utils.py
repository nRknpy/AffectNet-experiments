from typing import List
import os
import re
import signal
import subprocess
import threading
import wandb
import numpy as np

from torchaffectnet.const import ID2EXPRESSION


def save_token_and_target(tokens, targets, destination):
    data = np.column_stack((tokens, targets))
    np.savetxt(destination, data, delimiter=',')
    return tokens, targets


def exclude_id(exclude_labels: List[int]):
    base_id2label = ID2EXPRESSION
    label_list = []
    for k, v in base_id2label.items():
        if k in exclude_labels:
            continue
        label_list.append(v)
    id2label = {}
    for i, v in enumerate(label_list):
        id2label[i] = v
    label2id = {v: k for k, v in id2label.items()}
    return id2label, label2id


# 2023/6/18: wandb hangs when you try to finish wandb run.
# https://github.com/wandb/wandb/issues/5601
# This function forcibly kills the remaining wandb process.
def force_finish_wandb():
    with open(os.path.join(os.path.dirname(__file__), '../wandb/latest-run/logs/debug-internal.log'), 'r') as f:
        last_line = f.readlines()[-1]
    match = re.search(r'(HandlerThread:|SenderThread:)\s*(\d+)', last_line)
    if match:
        pid = int(match.group(2))
        print(f'wandb pid: {pid}')
    else:
        print('Cannot find wandb process-id.')
        return

    print('Trying to kill wandb process...')
    try:
        os.kill(pid, signal.SIGKILL)
        print(f"Process with PID {pid} killed successfully.")
    except OSError:
        print(f"Failed to kill process with PID {pid}.")


def try_finish_wandb():
    threading.Timer(60, force_finish_wandb).start()
    wandb.finish()
