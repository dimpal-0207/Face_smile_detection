++++run script++++

pip3 install virtualenv
python3 venv venv 

++install dependencies ++
pip3 install -r requirements.txt


+++run project on local server +++
flask run --host:0.0.0.0

++run project on gunicorn on production server +++
gunicorn -k gevent -b 0.0.0.0:5000 app:app
