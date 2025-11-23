import os
from datetime import datetime

def create_experiment_dir(base_dir, exp_name):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base_dir, f"{exp_name}_{ts}")
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir
