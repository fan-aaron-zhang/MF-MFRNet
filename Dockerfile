FROM tensorflow/tensorflow:1.14.0-gpu-py3


# RUN apt-get update && apt-get install -y python3-opencv


# Python packages
RUN pip install --upgrade pip
RUN pip install --upgrade opencv-python
RUN pip install --upgrade imageio
RUN pip install --upgrade scikit-image
RUN pip install --upgrade scipy
RUN pip install --upgrade numpy==1.15.0

# Move code
COPY ./CNN ./CNN
RUN true
COPY ./delete_db/ ./delete_db
RUN true
