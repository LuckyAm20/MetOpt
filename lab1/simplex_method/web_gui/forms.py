from flask_wtf import FlaskForm
from wtforms import IntegerField, SubmitField
from wtforms.validators import DataRequired, NumberRange


class NumVarConstForm(FlaskForm):
    number_of_variables = IntegerField(
        "Введите количество переменных СЛАУ",
        validators=[
            DataRequired(),
            NumberRange(1, 1000),
        ],
        default=5,
    )
    number_of_constraints = IntegerField(
        "Введите количество ограничений",
        validators=[DataRequired(), NumberRange(0)],
        default=6,
    )

    submit = SubmitField("Далее")
