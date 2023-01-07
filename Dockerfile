FROM continuumio/miniconda3

RUN conda install -y -c conda-forge \ 
    onnxruntime \
    fastapi \ 
    uvicorn \
    python-multipart \
    numpy \
    scipy

COPY ./models /models
COPY ./main.py /main.py

CMD uvicorn main:app --host=0.0.0.0 --port=$PORT