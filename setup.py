import subprocess

def run_command(command):
  """Выполняет команду в терминале."""
  process = subprocess.run(command.split(), capture_output=True, text=True)
  if process.returncode == 0:
    print(f"Команда '{command}' успешно выполнена.")
  else:
    print(f"Ошибка при выполнении команды '{command}':\n{process.stderr}")

run_command("sudo apt update")

run_command("sudo apt install htop")

run_command("sudo apt install nvidia-cuda-toolkit")

run_command("sudo apt install git python3-pip python3-venv")

run_command("python3 -m venv venv")








