import os
basedir = os.path.abspath(os.path.dirname(__file__))

class Config(object):

    SECRET_KEY = os.environ.get('SECRET_KEY')

    # SQLAlchemy
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL')
    SQLALCHEMY_POOL_RECYCLE = 280
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Pagination
    DAYS_PER_PAGE = 7
    EVENTS_PER_PAGE = 4

#     if os.environ.get('FLASK_ENV') == 'development':
    if os.environ.get('FLASK_DEBUG') == True:
        #SQLALCHEMY_DATABASE_URI =  \
                #'mysql+mysqldb://dev:test@localhost/dailydb'
        #SQLALCHEMY_DATABASE_URI = \
                #'mysql+mysqldb://test:test@localhost/dailyapp$daily' # Home
        #SQLALCHEMY_DATABASE_URI = \
        #        'mysql+mysqldb://test:test@localhost/dailydev' # Home
        SQLALCHEMY_DATABASE_URI = \
                'mysql+mysqldb://dailydev:dailydev@localhost/daily' # Home
        SECRET_KEY = 'asd'
        EXPLAIN_TEMPLATE_LOADING = True;

import pdb; pdb.set_trace()
