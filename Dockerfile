FROM pytorch/pytorch@sha256:5ffed6c836956ef474d369d9dfe7b3d52263e93b51c7a864b068f98e02ea8c51
RUN apt-get update && \
    apt-get install -y  python3-dev \
                        build-essential \
                        automake \
                        bison \
                        libtool \
                        byacc \
                        swig \
                        pkg-config \
                        g++ \
                        gcc \
                        wget \
                        libgtk-3-dev \
                        libcairo2-dev \
                        ghostscript \
                        expat \
                        libpng-dev \
                        zlib1g-dev \
                        libgts-dev \
                        libperl-dev \
                        vim \
                        openssh-server
RUN python -m pip install --upgrade pip setuptools wheel
RUN python -m pip install --upgrade tables \
                    scipy \
                    matplotlib \
                    jupyter \
                    pandas \
                    sympy \
                    nose \
                    networkx \
                    sparsemax \
                    h5py \
                    ray \
                    tabulate \
                    sklearn \
                    tensorboard
# For some reason has to be installed separately
RUN python -m pip install --upgrade ray[tune]
RUN python -m pip install --upgrade ray[rllib]

#RUN wget https://www2.graphviz.org/Packages/stable/portable_source/graphviz-2.44.1.tar.gz
#RUN tar -xzvf graphviz-2.44.1.tar.gz
#RUN cd graphviz-2.44.1 && ./configure && make && make install
#RUN python -m pip install --upgrade pygraphviz

RUN apt-get install texlive-full -y

RUN useradd -rm -d /home/developer -s /bin/bash -g root -G sudo -u 1000 developer 
RUN echo 'developer:AJas12!$' | chpasswd
RUN service ssh start
EXPOSE 22
ENTRYPOINT sleep 1 && service ssh start && bash
