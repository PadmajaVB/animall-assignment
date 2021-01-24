FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

RUN pip install virtualenv
ENV VIRTUAL_ENV=/venv
RUN virtualenv venv -p python3
ENV PATH="VIRTUAL_ENV/bin:$PATH"

WORKDIR /app
ADD . /app

# Install dependencies
RUN pip install -r requirements.txt

# Expose port 
EXPOSE 8000

# copy contents of project into docker
COPY ./ /app

# Run the application:
CMD ["python", "code/test.py"]