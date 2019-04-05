FROM nvcr.io/nvidia/tensorflow:19.01-py3

# Install GIT
RUN apt-get update && apt-get install git libgtk2.0-dev libsm6 libxext6 xvfb -y

# Upgrade PIP
RUN python3 -m pip install --upgrade pip

WORKDIR /root/

# Clone Code
RUN git clone https://github.com/perara/deep_logistics_ml.git deep_logistics_ml_git
RUN git clone https://github.com/perara/deep_logistics.git deep_logistics_git

# Install dependencies
RUN pip3 install -r deep_logistics_git/requirements.txt
RUN pip3 install -r deep_logistics_ml_git/requirements.txt
RUN mkdir -p /root/ray_results
RUN pip3 install tensorboard setproctitle

RUN chmod +x /root/deep_logistics_ml_git/runner.sh

CMD ["/root/deep_logistics_ml/runner.sh"]




