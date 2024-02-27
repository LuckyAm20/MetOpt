import os
import sys

from flask import Flask

if getattr(sys, "frozen", False):
    template_folder = os.path.join(sys._MEIPASS, "templates")
    static_folder = os.path.join(sys._MEIPASS, "static")
    app = Flask(__name__, static_folder=static_folder, template_folder=template_folder)
else:
    app = Flask(__name__, static_folder="./static", template_folder="./templates")

# etc
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY") or "secret"

from .routers import index, input_eqs
