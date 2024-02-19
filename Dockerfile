FROM openfabric/tee-python-cpu:latest

RUN mkdir openfabric-ai-software-engineer
WORKDIR /openfabric-ai-software-engineer
COPY main.py .
RUN poetry install -vvv --no-dev
EXPOSE 5500
CMD ["sh","start.sh"]