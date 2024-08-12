# Install dependencies
sudo apt update
sudo apt install -y make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev \
liblzma-dev python-openssl git

# Clone pyenv repository
curl https://pyenv.run | bash

echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init --path)"' >> ~/.bashrc
source ~/.bashrc

pyenv install 3.7.12
pyenv global 3.7.12
/home/codespace/.pyenv/versions/3.7.12/bin/python3.7 -m pip install --upgrade pip
pip install virtualenv

python -m venv work
source work/bin/activate
pip install tensorflow==1.15.0
pip install protobuf==3.20.3
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
pip install pandas matplotlib scipy scikit-learn