from __future__ import print_function
from __future__ import division

import sys
import time
import argparse
import numpy as np
import cPickle as pickle
from collections import Counter, defaultdict


def entropy_spatial(sessions):
    locations = {}
    days = sorted(sessions.keys())
    for d in days:
        session = sessions[d]
        for s in session:
            if s[0] not in locations:
                locations[s[0]] = 1
            else:
                locations[s[0]] += 1
    frequency = np.array([locations[loc] for loc in locations])
    frequency = frequency / np.sum(frequency)
    entropy = - np.sum(frequency * np.log(frequency))
    return entropy

def load_venues_from_tweets(path):
    poi = {}
    cnt = 0
    with open(path) as v:
        for line in v:
            _, uid, lon, lat, tim, _, _, tweet, _, _, _, _, _ = line.strip('\r\n').split('')
            poi[(lon, lat)] = cnt
            cnt += 1
    return poi

class DataFoursquareLA(object):
    def __init__(self, trace_min=10, global_visit=10, hour_gap=72, min_gap=10, session_min=2, session_max=10,
                 sessions_min=2, train_split=0.8, embedding_len=50):
        tmp_path = "../data/LA/"
        self.TWITTER_PATH = tmp_path + 'la_tweets1100000_cikm.txt'
        self.VENUES_PATH = tmp_path + 'venues_all.txt'
        self.SAVE_PATH = tmp_path
        self.save_name = 'foursquare_la2'

        self.trace_len_min = trace_min
        self.location_global_visit_min = global_visit
        self.hour_gap = hour_gap
        self.min_gap = min_gap
        self.session_max = session_max
        self.filter_short_session = session_min
        self.sessions_count_min = sessions_min
        self.words_embeddings_len = embedding_len

        self.train_split = train_split

        self.data = {}
        self.venues = {}
        self.pois = load_venues_from_tweets(self.TWITTER_PATH)
        self.words_original = []
        self.words_lens = []
        self.dictionary = dict()
        self.words_dict = None
        self.data_filter = {}
        self.user_filter3 = None
        self.uid_list = {}
        self.vid_list = {'unk': [0, -1]}
        self.vid_list_lookup = {}
        self.vid_lookup = {}
        self.pid_loc_lat = {}
        self.data_neural = {}


    # ############# 1. read trajectory data from twitters
    def load_trajectory_from_tweets(self):
        with open(self.TWITTER_PATH) as fid:
            for i, line in enumerate(fid):
                pid, uid, lon, lat, tim, _, _, tweet, _, _, _, _, _ = line.strip('\r\n').split('')
                # pid = self.pois[(lon, lat)]
                # Mon Sep 29 20:29:55 CDT 2014
                tim = tim.replace('CDT ', '').replace('CST ', '')
                tim = time.strptime(tim, "%a %b %d %H:%M:%S %Y")
                tim = time.strftime("%Y-%m-%d %H:%M:%S", tim)
                if uid not in self.data:
                    self.data[uid] = [[pid, tim]]
                else:
                    self.data[uid].append([pid, tim])
                if pid not in self.venues:
                    self.venues[pid] = 1
                else:
                    self.venues[pid] += 1

    # ########### 3.0 basically filter users based on visit length and other statistics
    def filter_users_by_length(self):
        # average number of records per user
        import numpy as np
        avg = np.mean([len(self.data[x]) for x in self.data])
        mx = np.max([len(self.data[x]) for x in self.data])
        #print('Average number of records per user: %f' %avg)
        #print('Max number of records per user: %f' %mx)
        # filter out uses with less than 10 records
        uid_3 = [x for x in self.data if len(self.data[x]) > self.trace_len_min]
        #print('Number of users with >= 10 records: %d' %len(uid_3))
        # sort users by the number of records, descending order
        pick3 = sorted([(x, len(self.data[x])) for x in uid_3], key=lambda x: x[1], reverse=True)
        # average number of visits per venue
        avg2 = np.mean([self.venues[x] for x in self.venues])
        mx2 = np.max([self.venues[x] for x in self.venues])
        #print('Average number of visits per venue: %f' %avg2)
        #print('Max number of visits per venue: %f' %mx2)
        # filter out venues with less than 10 visits
        pid_3 = [x for x in self.venues if self.venues[x] > self.location_global_visit_min]
        #print('Number of venues with >= 10 visits: %d' % len(pid_3))
        # sort venues by the number of visits, descending order
        pid_pic3 = sorted([(x, self.venues[x]) for x in pid_3], key=lambda x: x[1], reverse=True)
        pid_3 = dict(pid_pic3)

        session_len_list = []
        # hr_gap = []
        # mn_gap = []
        for u in pick3:
            uid = u[0]
            info = self.data[uid]  # [[pid, tim]]
            topk = Counter([x[0] for x in info]).most_common()  # pid, number of visits (descending order)
            # pid of locations visited more than once
            topk1 = [x[0] for x in topk if x[1] > 1]
            sessions = {}
            for i, record in enumerate(info):
                poi, tmd = record
                try:
                    tid = int(time.mktime(time.strptime(tmd, "%Y-%m-%d %H:%M:%S")))
                except Exception as e:
                    print('error:{}'.format(e))
                    continue
                sid = len(sessions)  # session id, 0 index
                if poi not in pid_3 and poi not in topk1:  # visited less than 10 times
                    # if poi not in topk1:
                    continue
                # else, add this [pid, tmd] as session
                if i == 0 or len(sessions) == 0:
                    sessions[sid] = [record]
                else:
                    # hr_gap.append((tid - last_tid) / 3600)
                    # mn_gap.append((tid - last_tid) / 60)
                    # if hour gap since last record > 72 | last session has > 10 records, start new session
                    if (tid - last_tid) / 3600 > self.hour_gap or len(sessions[sid - 1]) > self.session_max:
                        sessions[sid] = [record]
                    # if the record is apart from the last record for
                    # <= 72 hours, and
                    # > 10 minutes,
                    # then append record to the last session
                    elif (tid - last_tid) / 60 > self.min_gap:
                        sessions[sid - 1].append(record)
                    # if the record is apart from the last record for <= 10 minutes
                    else:
                        pass
                last_tid = tid
            sessions_filter = {}
            # sess_cnt = [len(sessions[s]) for s in sessions]
            # print('Average number of records per session: %f' %np.mean(sess_cnt))
            #print('Max number of records per session: %f' %np.max(sess_cnt))
            for s in sessions:
                # sessions with records >= 5
                if len(sessions[s]) >= self.filter_short_session:
                    sessions_filter[len(sessions_filter)] = sessions[s]
                    session_len_list.append(len(sessions[s]))
            # if the filtered sessions(sessions with >= 5 records) are >= 5
            if len(sessions_filter) >= self.sessions_count_min:
                # print('Number of filtered sessions: %d' %len(sessions_filter))
                self.data_filter[uid] = {'sessions_count': len(sessions_filter), 'topk_count': len(topk), 'topk': topk,
                                         'sessions': sessions_filter, 'raw_sessions': sessions}
            # else:
            #     print('No filtered sessions left')
        # print('Average hour gap: %f' %np.mean(hr_gap))
        # print('Max hour gap: %f' %np.max(hr_gap))
        # print('Average minute gap: %f' %np.mean(mn_gap))
        # print('Max minute gap: %f' %np.max(mn_gap))

        # list of uid in filtered sessions
        self.user_filter3 = [x for x in self.data_filter if
                             self.data_filter[x]['sessions_count'] >= self.sessions_count_min]

    # ########### 4. build dictionary for users and location
    def build_users_locations_dict(self):
        for u in self.user_filter3:
            sessions = self.data_filter[u]['sessions']
            if u not in self.uid_list:
                self.uid_list[u] = [len(self.uid_list), len(sessions)]
            for sid in sessions:
                poi = [p[0] for p in sessions[sid]]
                for p in poi:
                    if p not in self.vid_list:
                        self.vid_list_lookup[len(self.vid_list)] = p
                        self.vid_list[p] = [len(self.vid_list), 1]
                    else:
                        self.vid_list[p][1] += 1

    # support for radius of gyration
    def load_venues(self):
        with open(self.TWITTER_PATH, 'r') as fid:
            for line in fid:
                pid, uid, lon, lat, tim, _, _, tweet, _, _, _, _, _ = line.strip('\r\n').split('')
                #pid = self.pois[(lon, lat)]
                self.pid_loc_lat[pid] = [float(lon), float(lat)]

    def venues_lookup(self):
        for vid in self.vid_list_lookup:
            pid = self.vid_list_lookup[vid]
            lon_lat = self.pid_loc_lat[pid]
            self.vid_lookup[vid] = lon_lat

    # ########## 5.0 prepare training data for neural network
    @staticmethod
    def tid_list(tmd):
        tm = time.strptime(tmd, "%Y-%m-%d %H:%M:%S")
        tid = tm.tm_wday * 24 + tm.tm_hour
        return tid

    @staticmethod
    def tid_list_48(tmd):
        tm = time.strptime(tmd, "%Y-%m-%d %H:%M:%S")
        if tm.tm_wday in [0, 1, 2, 3, 4]:
            tid = tm.tm_hour
        else:
            tid = tm.tm_hour + 24
        return tid

    def prepare_neural_data(self):
        for u in self.uid_list:
            sessions = self.data_filter[u]['sessions']
            sessions_tran = {}
            sessions_id = []
            for sid in sessions:
                sessions_tran[sid] = [[self.vid_list[p[0]][0], self.tid_list_48(p[1])] for p in
                                      sessions[sid]]
                sessions_id.append(sid)
            split_id = int(np.floor(self.train_split * len(sessions_id)))
            train_id = sessions_id[:split_id]
            test_id = sessions_id[split_id:]
            pred_len = sum([len(sessions_tran[i]) - 1 for i in train_id])
            valid_len = sum([len(sessions_tran[i]) - 1 for i in test_id])
            train_loc = {}
            for i in train_id:
                for sess in sessions_tran[i]:
                    if sess[0] in train_loc:
                        train_loc[sess[0]] += 1
                    else:
                        train_loc[sess[0]] = 1
            # calculate entropy
            entropy = entropy_spatial(sessions)

            # calculate location ratio
            train_location = []
            for i in train_id:
                train_location.extend([s[0] for s in sessions[i]])
            train_location_set = set(train_location)
            test_location = []
            for i in test_id:
                test_location.extend([s[0] for s in sessions[i]])
            test_location_set = set(test_location)
            whole_location = train_location_set | test_location_set
            test_unique = whole_location - train_location_set
            location_ratio = len(test_unique) / len(whole_location)

            # calculate radius of gyration
            lon_lat = []
            for pid in train_location:
                try:
                    lon_lat.append(self.pid_loc_lat[pid])
                except:
                    print(pid)
                    print('error')
            lon_lat = np.array(lon_lat)
            center = np.mean(lon_lat, axis=0, keepdims=True)
            center = np.repeat(center, axis=0, repeats=len(lon_lat))
            rg = np.sqrt(np.mean(np.sum((lon_lat - center) ** 2, axis=1, keepdims=True), axis=0))[0]

            self.data_neural[self.uid_list[u][0]] = {'sessions': sessions_tran, 'train': train_id, 'test': test_id,
                                                     'pred_len': pred_len, 'valid_len': valid_len,
                                                     'train_loc': train_loc, 'explore': location_ratio,
                                                     'entropy': entropy, 'rg': rg}

    # ############# 6. save variables
    def get_parameters(self):
        parameters = {}
        parameters['TWITTER_PATH'] = self.TWITTER_PATH
        parameters['SAVE_PATH'] = self.SAVE_PATH

        parameters['trace_len_min'] = self.trace_len_min
        parameters['location_global_visit_min'] = self.location_global_visit_min
        parameters['hour_gap'] = self.hour_gap
        parameters['min_gap'] = self.min_gap
        parameters['session_max'] = self.session_max
        parameters['filter_short_session'] = self.filter_short_session
        parameters['sessions_min'] = self.sessions_count_min
        parameters['train_split'] = self.train_split

        return parameters

    def save_variables(self):
        foursquare_dataset = {'data_neural': self.data_neural, 'vid_list': self.vid_list, 'uid_list': self.uid_list,
                              'parameters': self.get_parameters(), 'data_filter': self.data_filter,
                              'vid_lookup': self.vid_lookup}
        pickle.dump(foursquare_dataset, open(self.SAVE_PATH + self.save_name + '.pk', 'wb'))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace_min', type=int, default=10, help="raw trace length filter threshold")
    # global_visit
    # 0: final users:2665 final locations:247188
    # >=1: final users:0 final locations:1
    parser.add_argument('--global_visit', type=int, default=10, help="location global visit threshold")
    parser.add_argument('--hour_gap', type=int, default=72, help="maximum interval of two trajectory points")
    parser.add_argument('--min_gap', type=int, default=10, help="minimum interval of two trajectory points")
    parser.add_argument('--session_max', type=int, default=10, help="control the length of session not too long")
    parser.add_argument('--session_min', type=int, default=5, help="control the length of session not too short")
    parser.add_argument('--sessions_min', type=int, default=5, help="the minimum amount of the good user's sessions")
    parser.add_argument('--train_split', type=float, default=0.8, help="train/test ratio")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    data_generator = DataFoursquareLA(trace_min=args.trace_min, global_visit=args.global_visit,
                                    hour_gap=args.hour_gap, min_gap=args.min_gap,
                                    session_min=args.session_min, session_max=args.session_max,
                                    sessions_min=args.sessions_min, train_split=args.train_split)
    parameters = data_generator.get_parameters()
    print('############PARAMETER SETTINGS:\n' + '\n'.join([p + ':' + str(parameters[p]) for p in parameters]))
    print('############START PROCESSING:')
    print('load trajectory from {}'.format(data_generator.TWITTER_PATH))
    data_generator.load_trajectory_from_tweets()
    print('filter users')
    data_generator.filter_users_by_length()
    print('build users/locations dictionary')
    data_generator.build_users_locations_dict()
    data_generator.load_venues()
    data_generator.venues_lookup()
    print('prepare data for neural network')
    data_generator.prepare_neural_data()
    print('save prepared data')
    data_generator.save_variables()
    # raw users:153626 raw locations:734559
    print('raw users:{} raw locations:{}'.format(
        len(data_generator.data), len(data_generator.venues)))
    # final users:602 final locations:7621
    print('final users:{} final locations:{}'.format(
        len(data_generator.data_neural), len(data_generator.vid_list)))
