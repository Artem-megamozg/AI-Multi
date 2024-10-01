import subprocess

def run_command(command):
  """Выполняет команду в терминале."""
  process = subprocess.run(command.split(), capture_output=True, text=True)
  if process.returncode == 0:
    print(f"Команда '{command}' успешно выполнена.")
  else:
    print(f"Ошибка при выполнении команды '{command}':\n{process.stderr}")


run_command("pip3 install torch torchvision torchaudio")

# run_command("pip install nvidia-pyindex")

# run_command("pip install nvidia-nccl")
run_command("sudo apt install gpustat")
