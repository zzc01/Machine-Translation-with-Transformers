FROM python:3.10.6-slim
COPY ./app.py /deploy/
COPY ./requirements.txt /deploy/
COPY ./translator/variables/variables.data-00000-of-00001 /deploy/metadata/translator/variables/
COPY ./translator/variables/variables.index /deploy/metadata/translator/variables/
COPY ./translator/assets/deu_vocab.txt /deploy/metadata/translator/assets/
COPY ./translator/assets/eng_vocab.txt /deploy/metadata/translator/assets/
COPY ./translator/saved_model.pb /deploy/metadata/translator/
WORKDIR /deploy/
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 80
ENTRYPOINT ["python", "app.py"]
