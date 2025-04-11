FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

# Create the application directory
RUN mkdir -p /app

# Copy the source code
COPY /requirements /app/requirements

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /app/requirements/requirements.txt

# Copy the rest of the application code
COPY /src /app/src

# Set the working directory
WORKDIR /app

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=1234", "--no-browser", "--allow-root", "--NotebookApp.token=''"]