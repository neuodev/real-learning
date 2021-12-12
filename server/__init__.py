import os
import click
from flask import Flask
from .ara_eng.train_lstm import train_lstm


def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite'),
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    @app.route('/ping')
    def ping():
        return 'Pong!'

    # CLI
    @app.cli.command('nmt_lstm')
    @click.option('--summary', default=False)
    @click.option('--sample', default=False)
    @click.option('--train', default=False)
    @click.option('--epochs', default=10)
    @click.option('--save', default='local', help='local|replace')
    @click.option('--test', default=False, help='Test a model on the test data')
    def _train_lstm(summary, sample, train, epochs, save, test):
      train_lstm(summary, sample, train, epochs, save, test, app.instance_path, app.root_path)

    return app