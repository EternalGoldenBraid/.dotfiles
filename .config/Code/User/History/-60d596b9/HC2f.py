from flask_wtf import FlaskForm
from wtforms import (BooleanField, StringField, PasswordField, FormField,
    FieldList, SubmitField, TextAreaField, IntegerField)
from wtforms.validators import DataRequired, length, NumberRange
from wtforms.fields import DateField
from wtforms.widgets import ListWidget


class DescriptionForm(FlaskForm):
    event=TextAreaField('Event', 
                validators=[length(max=200, 
                message='Keep event under 200 characters')])
    #event=StringField('Event', 
    #            validators=[length(max=200, 
    #            message='Keep event under 200 characters')])
    duration_event_hours=IntegerField('Duration', 
            render_kw={'placeholder': 'Hours'},
            validators=[NumberRange(min=0, max=24, 
                message='Event duration exceeds 24 hours')])
    duration_event_minutes=IntegerField( 
            render_kw={'placeholder': 'Minutes'},
            validators=[NumberRange(min=0, max=59, 
                message='Event duration minutes exceed 59 minutes')])
    submit=SubmitField('Submit Events')

class EventsForm(FlaskForm):
    description=FieldList(FormField(DescriptionForm), 
                min_entries=1)
    submit=SubmitField('Submit Events')


class EntryForm(FlaskForm):
    date=DateField('Date', format='%Y-%m-%d', 
                validators=[DataRequired()], 
                render_kw={"placeholder": "Date"})

    sleep_rating=StringField('Sleep', validators=[DataRequired()],
                render_kw={"placeholder": "Sleep"}) 

    meditation_hours=StringField('Meditation (H:M)',
                validators=[DataRequired(),
                length(max=2, 
                    message="Meditation contains too many digits")],
                    render_kw={"placeholder": "Hours"})

    meditation_minutes=StringField('Minutes', validators=[DataRequired(),
                length(max=2, 
                    message="Meditation contains too many digits")],
                render_kw={"placeholder": "Minutes"})

    day_rating=StringField('Rating', validators=[DataRequired(), 
                length(max=2, 
                    message='Rating for the day contains too many digits')],
                render_kw={"placeholder": "Rating"})

    lights=StringField('Lights', validators=[DataRequired(), 
                length(max=15,
                message='Your lights input contains too many digits')],
                render_kw={"placeholder": "Lights"})
    cw_hours=StringField('Creative work (Hours & Minutes)',
                validators=[DataRequired(),
                length(max=2, 
                    message= 'Creative work input field contains too many digits')],
                render_kw={"placeholder": "Hours"})

    cw_minutes=StringField('CW', validators=[DataRequired(),
                length(max=2, 
                    message= 'Creative work input field contains too many digits')],
                render_kw={"placeholder": "Minutes"})
    submit=SubmitField('Submit For the day')

class BacklogForm(FlaskForm):
    api_key=StringField('API key', validators=[DataRequired()])
    remember_me=BooleanField('Remember Me')
    submit=SubmitField('Fetch')

