FROM python:3.8

#expose port 8501 for app to be run on
EXPOSE 8501

# Make a directory for our application
WORKDIR /app

# Install dependencies
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
RUN apt-get update
RUN apt install ffmpeg -y

# Copy our source code
COPY . /app/

# Run the application
ENTRYPOINT ["streamlit", "run"]

CMD ["app.py"]