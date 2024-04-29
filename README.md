![OpenRASE](docs/assets/open_rase.svg##gh-light-mode-only)
![OpenRASE](docs/assets/open_rase_white.svg##gh-dark-mode-only)
---
# OpenRASE
This is an emulator built on top of Containernet to benchmark and evaluate solutions to the NFV-RA problem.

# Requirements
- Python 3.9
- Ubuntu 20.04
- Docker

# Installation Instructions
1. Install the required packages.
```bash
sudo apt-get install -y build-essential  zlib1g-dev libffi-dev libssl-dev liblzma-dev libbz2-dev libreadline-dev libsqlite3-dev curl git tk-dev gcc python-dev libxml2-dev libxslt1-dev zlib1g-dev python-setuptools python3-venv
```
2. Install pyenv.
```bash
curl https://pyenv.run | bash
```
Add the following lines to your .bashrc or .zshrc and .bash-profile or .profile file.
```bash
export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
```
Source your .bashrc or .zshrc file.
```bash
source ~/.bashrc
```
Add pyenv to the root user. This is important because mininet has to be run as the root user.
Run the following command.
```bash
which pyenv
```
Copy the output leaving the trailing `/pyenv` part.
Run the following command.
```bash
sudo visudo
```
Append the copied output to `secure_path` as shown below.
```bash
Defaults        secure_path="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/snap/bin:/home/username/.pyenv/bin"
```
3. Install Python 3.9.
```bash
pyenv install 3.9
sudo pyenv install 3.9
```
4. Install ansible.
```bash
sudo apt-get install ansible
```
5. Clone containernet.
```bash
git clone https://github.com/Project-Kelvin/containernet
```
6. Install containernet.
```bash
sudo ansible-playbook -i "localhost," -c local containernet/ansible/install.yml
```
7. Add user to docker group.
```bash
sudo usermod -aG docker $USER
newgrp docker
```
8. Install pipx.
```bash
python3 -m pip install --user pipx
python3 -m pipx ensurepath
```
Source your .bashrc or .zshrc file.
```bash
source ~/.bashrc
```
9. Install poetry.
```bash
pipx install poetry
```
Add `poetry` to the root user.
Run the following command.
```bash
which poetry
```
Copy the output leaving the trailing `/poetry` part.
Run the following command.
```bash
sudo visudo
```
Append the copied output to `secure_path` as shown below.
```bash
Defaults        secure_path="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/snap/bin:/home/username/.pyenv/bin:/home/username/.local/bin"
```
10. Clone OpenRASE.
```bash
git clone https://github.com/Project-Kelvin/OpenRASE
```
11. Install OpenRASE.
Move into the OpenRASE directory.
```bash
cd OpenRASE
```
Get the path of the Python 3.9 executable.
```bash
pyenv which python
```
Copy the output.
Run the following command replacing the placehodler with the copied output.
```bash
poetry env use <path to python 3.9 executable>
```
Repeat for the root user.
Get the path of the Python 3.9 executable.
```bash
sudo pyenv which python
```
Copy the output.
Run the following command replacing the placehodler with the copied output.
```bash
sudo poetry env use <path to python 3.9 executable>
```
Install the dependencies.
```bash
sudo poetry install
```
12. Install Ryu.
```bash
sudo poetry run python -m pip install ryu
```
Install eventlet.
```bash
sudo poetry run python -m pip install eventlet==0.30.2
```
13. Initialize OpenRASE
```bash
sudo poetry run init
```
Restart the Docker service.
```bash
sudo service docker restart
```
Start the private Docker registry.
```bash
sudo poetry run init --registry
```
14. Run OpenRASE test.
```bash
sudo poetry run test
```
