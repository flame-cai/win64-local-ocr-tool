import os
from dotenv import load_dotenv

load_dotenv('.env')

class Config(object):
    SQLALCHEMY_DATABASE_URI = "sqlite:///project.db"
    DATA_PATH = os.environ.get('DATA_PATH', default='instance')
    