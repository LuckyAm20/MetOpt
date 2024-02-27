from .web_gui import app
import webview
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())
webview.create_window("Симлекс-метод", app, maximized=True)
