from daily import db
from daily.data_analysis.gnn.get_data import get_event_tag_data


engine = db.get_engine()
get_event_tag_data(engine, timespan = 0, freq_threshold=2)
