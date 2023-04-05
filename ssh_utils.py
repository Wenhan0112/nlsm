import os
import shutil

def move_py_files_to_ssh(dest_ssh, dest_folder):
    files = [f for f in os.listdir() if f[-3:] == ".py"]
    temp_dir = "__temp_py_dir"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    [shutil.copy(f, os.path.join(temp_dir, f)) for f in files]
    origin = os.path.join(temp_dir, "*")
    dest = f"{dest_ssh}:{dest_folder}"
    os.system(f"scp {origin} {dest}")
    shutil.rmtree(temp_dir)

if __name__ == "__main__":
    move_py_files_to_ssh("wenhan@ai88uno.dhcp.lbl.gov", "nlsm")
