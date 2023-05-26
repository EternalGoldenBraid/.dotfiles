from flask_wtf import FlaskForm
from wtforms import (BooleanField, StringField, PasswordField, FormField,
    FieldList, SubmitField, TextAreaField, IntegerField,
    SelectField, SelectMultipleField, RadioField)
from wtforms.validators import DataRequired, length, NumberRange
from wtforms.fields import DateField
from wtforms.widgets import ListWidget, CheckboxInput

class MultiCheckboxField(SelectMultipleField):
    widget = ListWidget(prefix_label=False)
    option_widget = CheckboxInput()


class KPrototypes_network_form(FlaskForm):
    k=IntegerField('K', 
            render_kw={'placeholder': 'k'},
            validators=[NumberRange(min=0, max=24, 
                message='Clusters exceeding timespan is undefined'),
                DataRequired(),
                ],
            default = 5)
    timespan=IntegerField("Days",
            render_kw={'placeholder': 'dT'},
            validators=[NumberRange(min=0, max=999, message='Must be positive'),
                DataRequired(),
                ],
            default=14)
    freq_threshold=IntegerField("Threshold",
            render_kw={'placeholder': 'f'}, validators=[NumberRange(min=0, max=999, 
                message='Must be positive'), DataRequired()], default=4)
    version = RadioField('Version',
                coerce=int,
                choices = [ (1, u'Join clusters on mutual nodes'), 
                            (2, u'Keep clusters separate'),], default=1)
    fit = BooleanField('Check to retrain and not load from file.')
    d_set = RadioField('Dataset',
                choices=[
                    ('count', 'Count.'), ('binary', 'Binary.')],
                default=1, validators=[DataRequired()])
    init = RadioField('Initialization method.',
                choices=[ ('huang', u'Huang'), ('cao', u'Cao')],
                default='huang',
                validators=[DataRequired()])
    submit=SubmitField('OK')

class Kmodes_elbow_form(FlaskForm):
    timespan = MultiCheckboxField('Days',
                coerce=int,
                choices = [
                    (7, 7), (30, 30), (120, 120), (0, 'all')],
                default=('7', '7')
                )
    freq_threshold=IntegerField( 
            render_kw={'placeholder': 'f'}, validators=[NumberRange(min=0, max=999, 
                message='Must be positive'), DataRequired(), ],
            default=4)
    fit = BooleanField('Check to retrain and not load from file.')
    d_set = RadioField('Dataset',
                choices=[
                    ('count', 'Count.'), ('binary', 'Binary.')],
                default=1, validators=[DataRequired()])
    init = RadioField('Initialization method.',
                choices=[ ('huang', u'Huang'), ('cao', u'Cao')],
                default='huang',
                validators=[DataRequired()])
    submit=SubmitField('OK')

class Tag_network_form(FlaskForm):
    timespan=IntegerField("Days",
            render_kw={'placeholder': 'dT'},
            validators=[NumberRange(min=0, max=999, message='Must be positive'),
                DataRequired(),
                ],
            default=10)
    freq_threshold=IntegerField("Threshold",
            render_kw={'placeholder': 'f'}, validators=[NumberRange(min=0, max=999, 
                message='Must be positive'), DataRequired()], default=4)
    version = RadioField('Version',
                coerce=int,
                choices = [ (1, u'Date based'), 
                            (2, u'Event based'),], default=2)
    fit = BooleanField('Check to retrain and not load from file.')
    submit=SubmitField('OK')
