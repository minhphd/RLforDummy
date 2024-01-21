# Use an official Python runtime as a parent image
FROM python:3.10

# Copy the requirements file into the container at /usr/src/app
COPY requirements.txt ./

# Install any needed packages specified in requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# install ubuntu dependencies
COPY . .
ENV NAME RLforDummy