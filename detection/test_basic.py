import requests
import pytest
import os

ML_BACKEND = os.getenv('ML_BACKEND', 'dit-backend')


@pytest.mark.first
def test_basic_health_check():
    print(ML_BACKEND)
    response = requests.get("http://super-bo.bologna.enea.it:9090/")
    assert response.status_code == 200

    response = requests.get("http://super-bo.bologna.enea.it:9090/health")
    assert response.status_code == 200


@pytest.mark.second
@pytest.mark.skipif(ML_BACKEND!='dit-backend', reason="Test for text detection backend")
def test_setup():
    data = {
        "project": "1.1654592272",
        'schema': '<View>  <Image name="image" value="$ocr" zoom="true"/>  <Labels name="label" toName="image">        <Label value="Handwriting" background="blue"/>  <Label value="Other" background="#FFA39E"/></Labels>  <Rectangle name="bbox" toName="image" strokeWidth="3"/>  <Polygon name="poly" toName="image" strokeWidth="3"/>  <TextArea name="transcription" toName="image" editable="true" perRegion="true" required="true" maxSubmissions="1" rows="5" placeholder="Recognized Text" displayMode="region-list"/></View>',
        'hostname': "http://localhost:8080",
        'access_token': '1234567890123456789012345678901234567890'
    }
    response = requests.post("http://super-bo.bologna.enea.it:9090/setup", json=data)
    assert response.status_code == 200


@pytest.mark.skipif(ML_BACKEND!='dit-backend', reason="Test for text detection backend")
def test_predict_detection():
    data = {"tasks":[{"data":{"ocr":"s3://navyledger-imgs/CUST-3-17_1-000010.png"}}]    }
    response = requests.post("http://super-bo.bologna.enea.it:9090/predict", json=data)
    assert response.status_code == 200


@pytest.mark.skipif(ML_BACKEND!='dit-backend', reason="Test for text detection backend")
def test_webhook_predict_detection():
    #TO DO
    pass


@pytest.mark.skipif(ML_BACKEND != "dit-backend", reason="Test for text detection backend")
def test_train_detection():
    #TO DO
    pass