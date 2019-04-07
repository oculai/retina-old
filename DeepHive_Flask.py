from flask import Flask, request
from DeepHive import *
import json
import pika

"""
RABBIT_HOST=mosquito.rmq.cloudamqp.com
RABBIT_USER=	fqwvdgit
RABBIT_PASSWORD=fWsaLjs2Cpr5EXoAvHzLOqWHTd-TrK3A
RABBIT_VHOST=	fqwvdgit
RABBIT_EXCHANGE=hive
RABBIT_EXCHANGE_TYPE=topic
RABBIT_EXCHANGE_DURABLE=true
RABBIT_EXCHANGE_CONFIRM=true
RABBIT_EXCHANGE_AUTODELETE=false
RABBIT_QUEUE_NAME=hive-sensors
BIND_KEY=hive
"""

print(' [*] Waiting for logs. To exit press CTRL+C')
def callback(ch, method, properties, body):
    print(" [x] %r" % body)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # credentials = pika.PlainCredentials('fqwvdgit', 'fWsaLjs2Cpr5EXoAvHzLOqWHTd-TrK3A')
    # connection = pika.BlockingConnection(pika.ConnectionParameters('mosquito.rmq.cloudamqp.com', 5672, 'fqwvdgit', credentials))
    # channel = connection.channel()
    #
    # channel.exchange_declare(exchange = 'hive', exchange_type='topic', durable=True)
    # result = channel.queue_declare(exclusive=True)
    #
    # channel.queue_bind(exchange='hive',
                       #queue='hive-sensors')

    print('Prepping ->')
    data =  request.get_json()
    #get data
    sensors = data['sensors']
    input_scaled = scale_data(sensors)
    input_scaled = input_scaled.reshape(1, 32, 5)
    #input_win = make_windows(input_scaled, 16)
    #input_ready = prep_for_model(input_win, 16, 5)

    #build model
    model = build_model('C:/Users/Adrian/Desktop/Hackathons/HackABull2019/Models/deep_hive_weights.h5')

    #get predictions
    prediction = get_prediction(input_scaled, model, get_scaler())
    #next_ten = get_next_ten(input_scaled, model, get_scaler())

    # channel.basic_consume(prediction,
    #                       queue='hive-sensors',
    #                       no_ack=True)
    print(prediction)
    # channel.start_consuming()

    return json.dumps({"sensors": prediction})

# listen for messages

if __name__ == '__main__':
    app.run(debug=False, port = 3001)
