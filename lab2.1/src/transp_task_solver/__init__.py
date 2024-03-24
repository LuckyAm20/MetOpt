import webview
from dotenv import find_dotenv, load_dotenv

from .web_gui import app

load_dotenv(find_dotenv())
webview.create_window("Метод потенциалов", app, maximized=True)
