from flask import Flask, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from config import Config
from flask_login import LoginManager, UserMixin
import os
from logging.handlers import RotatingFileHandler

## DEBUG
from logging.config import dictConfig
import logging
dictConfig({
    'version': 1,
    'formatters': {'default': {
        'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
    }},
    'handlers': {'wsgi': {
        'class': 'logging.StreamHandler',
        'stream': 'ext://flask.logging.wsgi_errors_stream',
        'formatter': 'default'
    }},
    'root': {
        'level': 'INFO',
        'handlers': ['wsgi']
    }
})
## end DEBUG



db = SQLAlchemy()
#migrate = Migrate(app, db, compare_type=True)
migrate = Migrate(compare_type=True)

#View module (view functions) must be imported after the application object is created.
login = LoginManager()
login.login_view = 'auth.login'
from daily import models 

def create_app(config_class=Config):

    app = Flask(__name__)
    app.config.from_object(Config)

    db.init_app(app)
    migrate.init_app(app, db)
    login.init_app(app)

    #if not app.debug and not app.testing:
    if not app.debug:
        if not os.path.exists('logs'):
            os.mkdir('logs')
        file_handler = RotatingFileHandler('logs/daily_info.log', maxBytes=10240,
                        backupCount=10)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'))
        file_handler.setLevel(logging.INFO)

        file_handler_error = RotatingFileHandler('logs/daily_error.log', maxBytes=10240,
                        backupCount=10)
        file_handler_error.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'))
        file_handler_error.setLevel(logging.ERROR)

        app.logger.addHandler(file_handler)
        app.logger.addHandler(file_handler_error)
        app.logger.setLevel(logging.INFO)
        app.logger.info('Daily startup')
        

    # Register blueprints
    from daily.main import bp as main_bp
    from daily.auth import bp as auth_bp
    from daily.errors import bp as errors_bp
    from daily.data_analysis import bp as data_bp 
    app.register_blueprint(auth_bp, url_prefix='/auth')
    app.register_blueprint(errors_bp)
    app.register_blueprint(data_bp)
    app.register_blueprint(main_bp)

    return app
