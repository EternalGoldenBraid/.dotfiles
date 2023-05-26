from pathlib import Path
import pandas as pd
import numpy as np
from collections import Counter
import pickle
import os
# import h5py

def get_user_id_():
    # Allow demoers to view plots generated from my data.
    if current_user.is_anonymous or current_user.id == 2:
        # TODO: Change this to query by my username.
        user_id = 1
    else:
        user_id = current_user.id
        #user_id_= current_user.get_id()

    return user_id

def get_event_tag_data(engine, timespan=0, freq_threshold=1,
        to_save=True, filename='data_dump', out_dir: Path = Path('data_dumps')):
    """
    This function merges together three SQL tables consiting of:
    -Rating: attributes of a given day.
    -Events: posts of all the days.
    -Tags: Tags of all the posts.

    This is achieved by utilizing the association tables associating
    Rating id's with event id's and event id's with tag id's
    """

    # Allows demoers to view plots generated from my data.
    user_id_ = 1

    # Merge ratings, events and tags to get a df of dates with combinations of tags.
    
    # Read ratings for user
    #ratings = pd.read_sql('rating',engine,index_col='id')
    if timespan != 0:
        #query = f'SELECT * FROM rating WHERE user_id={user_id_} and date > CURRENT_DATE()-{timespan}'
        query = f'SELECT * FROM rating WHERE user_id={user_id_} ORDER BY date desc LIMIT {timespan}'

    else: 
        query = f'SELECT * FROM rating WHERE user_id={user_id_}'
    ratings = pd.read_sql(query, engine,index_col='id')
    ratings['rating_id'] = ratings.index

    # Read all tags.
    # TODO: Inefficient, fetch only for user, or set lazy=true?.
    # TODO: Add manual sql query.
    tags = pd.read_sql('tag',engine,index_col='id')
    #tags.columns = tags.columns.str.replace('id','tag_id')

    # Read all rating event associations.
    # TODO: Add manual sql query.
    # TODO: Inefficient, fetch only for user, or lazy=dynamic.
    re_m2m = pd.read_sql('rating_events',engine,index_col=False)
    ### Improvement
    #query = 'SELECT * FROM rating_events WHERE rating_id IN '+str(tuple(ratings.index))
    #query = 'SELECT * FROM rating_events WHERE rating_id IN '+str((414,1))
    #query = 'SELECT * FROM rating_events'
    #query = 'SELECT * FROM rating'
    #re_m2m = pd.read_sql(sql=query, con=engine,index_col=False,
                #columns=['rating_id', 'event_id'],
    #)
    ### END Improvement

    re_m2m = re_m2m[re_m2m['rating_id'].isin(ratings.index)]

    # Read event stories.
    ### TODO: Improvement
    #query = 'SELECT * FROM event WHERE id IN '+str(tuple(re_m2m['event_id']))
    #stories = pd.read_sql(sql=query, con=engine,index_col=False)
    stories = pd.read_sql(sql='event', con=engine, index_col='id')\
            .filter(items=re_m2m['event_id'].values, axis=0)['story']
    stories.index.names=['event_id']

    # TODO: DEBUG THIS UNDER. HOW DO M2M RELATIONS GET CREATED
    # UNDER THREE PERMUTATIONS:
    # EVENT-NO_TAGS, EVENT_TAGS, ...
    # Read all event tag associations.
    # Not all events have tags, thus merge innner vs. left
    # dictates whether null tags are included.
    # TODO: Inefficient, fetch only for user, or lazy=dynamic.
    et_m2m = pd.read_sql('event_tags',engine, index_col=False)
    et_m2m = re_m2m \
            .merge(et_m2m,how='inner',on='event_id')[['event_id', 'tag_id']]

    # Clean data. Remove rows there tag_id doesn't occur >= n times
    et_m2m = et_m2m[et_m2m.groupby('tag_id')['tag_id'] \
            .transform('count').ge(freq_threshold)]

    data = re_m2m \
            .merge(et_m2m,how='inner',on='event_id')[['event_id', 'tag_id','rating_id']]

    # Reduce redundant rating_id entries
    data = (data.groupby('event_id').agg(list))
    # Hack around index switching after merge
    data.index.names=['event_id']
    data['event_id'] = data.index
    data['rating_id'] = data['rating_id'].apply(lambda x: x[0])
    data = data.merge(ratings,how='inner',left_on='rating_id', right_on='rating_id')
    data = data.merge(stories,how='inner', left_on='event_id', right_on='event_id')
    data.index = data['event_id']
    data['event_id'] = data.index

    
    # Get tag names
    tag_list = et_m2m.groupby('event_id').agg(list)
    all_tags = np.concatenate([np.array(i) for i in tag_list['tag_id'].values])
    tag_id_columns = np.unique(all_tags)
    tag_names = tags.loc[tag_id_columns]['tag_name']
    data['tag_name'] = data['tag_id'].apply(lambda x: tag_names.loc[x].values)
    
    #features = ['tag_id', 'meditation', 'rating_sleep', 'rating_day', 'cw', 'screen']
    features = ['tag_id', 'tag_name', 'meditation', 'rating_sleep', 'rating_day', 'cw', 'screen', 'date', 'story']
    event_features = data[features]

    # if to_save:
    if True:
        # Dump data to file
        # THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))

        # path = os.path.join(THIS_FOLDER, filename+'.h5')
        path = out_dir/(filename+'.h5')
        out_dir.mkdir(parents=True, exist_ok=True)
        data.to_hdf(path, key='event_features', mode='w')
        print("Data saved to file in ", path)
        # stories.to_hdf(path+"test", key='stories', mode='a')

        # THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
        # path = os.path.join(THIS_FOLDER, filename+'.h5')
        # features.to_hdf(path, key='features', mode='w')
        # dates.to_hdf(path, key='dates', mode='a')
        # tag_names.to_hdf(path, key='tag_names', mode='a')
        # stories.to_hdf(path, key='stories', mode='a')

    print("Data packed")
    return event_features, tag_names
