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


def load_venues_from_tweets(path, header):
    poi = {}
    cnt = 0
    with open(path) as v:
        if header:
            v.readline()
        for line in v:
            taxi_seq, year, month, day, hour, minute, second, lat, lon = line.strip('\r\n').split('\t')
            poi[(round(float(lon), 6), round(float(lat), 6))] = cnt
            cnt += 1
    return poi


GRID_COUNT = 100
def geo_grade(index, x, y, m_nGridCount=GRID_COUNT):  # index: [pids], x: [lon], y: [lat]. 100 by 100
    dXMax, dXMin, dYMax, dYMin = max(x), min(x), max(y), min(y)
    # print dXMax, dXMin, dYMax, dYMin
    m_dOriginX = dXMin
    m_dOriginY = dYMin
    dSizeX = (dXMax - dXMin) / m_nGridCount
    dSizeY = (dYMax - dYMin) / m_nGridCount
    m_vIndexCells = []  # list of lists
    center_location_list = []
    for i in range(0, m_nGridCount * m_nGridCount + 1):
        m_vIndexCells.append([])
        y_ind = int(i / m_nGridCount)
        x_ind = i - y_ind * m_nGridCount
        center_location_list.append((dXMin + x_ind * dSizeX + 0.5 * dSizeX, dYMin + y_ind * dSizeY + 0.5 * dSizeY))
    # print (m_nGridCount, m_dOriginX, m_dOriginY, \
    #        dSizeX, dSizeY, len(m_vIndexCells), len(index))
    poi_index_dict = {}
    _poi_index_dict = defaultdict(list)
    for i in range(len(x)):
        nXCol = int((x[i] - m_dOriginX) / dSizeX)
        nYCol = int((y[i] - m_dOriginY) / dSizeY)
        if nXCol >= m_nGridCount:
            # print 'max X'
            nXCol = m_nGridCount - 1

        if nYCol >= m_nGridCount:
            # print 'max Y'
            nYCol = m_nGridCount - 1

        iIndex = nYCol * m_nGridCount + nXCol
        poi_index_dict[index[i]] = iIndex  # key: raw poi, val: grid id
        _poi_index_dict[iIndex].append(index[i])  # key: grid id, val: raw pid
        m_vIndexCells[iIndex].append([index[i], x[i], y[i]])

    # normalize grid center location
    lon = np.array([l[0] for l in center_location_list])
    lon_m = np.mean(lon)
    lon_s = np.std(lon)
    lon_norm = [(l-lon_m)/lon_s for l in lon]

    lat = np.array([l[1] for l in center_location_list])
    lat_m = np.mean(lat)
    lat_s = np.std(lat)
    lat_norm = [(l-lat_m)/lat_s for l in lat]
    import pdb
    pdb.set_trace()

    center_location_list_norm = zip(lon_norm, lat_norm)

    # round 6
    center_location_list = [(round(x[0], 6), round(x[1], 6)) for x in center_location_list]
    center_location_list_norm = [(round(x[0], 6), round(x[1], 6)) for x in center_location_list_norm]

    return poi_index_dict, center_location_list, center_location_list_norm
    # return poi_index_dict, _poi_index_dict


class DataTaxi(object):
    def __init__(self, trace_min=10, global_visit=10, minute_gap=5, min_gap=4, session_min=2, session_max=10,
                 sessions_min=2, train_split=0.8, embedding_len=50, header=True, grid=True):
        tmp_path = "../../data/Taxi/"
        self.TWITTER_PATH = tmp_path + 'merged_cleaned_taxi_data.txt'
        # self.VENUES_PATH = tmp_path + 'venues_all.txt'
        self.SAVE_PATH = tmp_path
        self.save_name = 'taxi'
        self.header = header
        self.grid = grid

        self.trace_len_min = trace_min
        self.location_global_visit_min = global_visit
        # self.hour_gap = hour_gap
        self.minute_gap = minute_gap
        self.min_gap = min_gap
        self.session_max = session_max
        self.filter_short_session = session_min
        self.sessions_count_min = sessions_min
        self.words_embeddings_len = embedding_len

        self.train_split = train_split

        self.data = {}
        self.venues = {}
        self.pois = load_venues_from_tweets(self.TWITTER_PATH, self.header)  # key: (lon, lat), val: pid
        _raw_xy = self.pois.keys()
        _raw_x = [i[0] for i in _raw_xy]
        _raw_y = [i[1] for i in _raw_xy]
        _raw_pid = [self.pois[k] for k in _raw_xy]
        self.poi_index_dict, self.center_location_list, self.center_location_list_norm = geo_grade(_raw_pid, _raw_x, _raw_y)
        self.words_original = []
        self.words_lens = []
        self.dictionary = dict()
        self.words_dict = None
        self.data_filter = {}
        self.user_filter3 = None
        self.uid_list = {}
        self.vid_list = {'unk': [0, -1]}
        self.vid_list_lookup = {}  # key: int vid, val: raw pid
        self.vid_lookup = {}  # key: int vid, val: grid center [lon, lat]
        self.pid_loc_lat = {}  # key: raw vid, val: grid center [lon, lat]
        self.data_neural = {}


    # ############# 1. read trajectory data from twitters
    def load_trajectory_from_tweets(self):
        with open(self.TWITTER_PATH) as fid:
            if self.header:
                fid.readline()
            for i, line in enumerate(fid):
                taxi_seq, year, month, day, hour, minute, second, lat, lon = line.strip('\r\n').split('\t')
                pid = self.pois[(round(float(lon), 6), round(float(lat), 6))]  # pid
                if self.grid:
                    pid = self.poi_index_dict[pid]  # key: raw pid, val: grid id 
                uid = taxi_seq.split('.')[0]
                year = year.split('.')[0]
                month = month.split('.')[0].zfill(2)
                day = day.split('.')[0].zfill(2)
                hour = hour.split('.')[0].zfill(2)
                minute = minute.split('.')[0].zfill(2)
                second = second.split('.')[0].zfill(2)
                tim = '%s-%s-%s %s:%s:%s' %(year, month, day, hour, minute, second)
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
        # filter out uses with <= 10 records
        uid_3 = [x for x in self.data if len(self.data[x]) > self.trace_len_min]
        # sort users by the number of records, descending order
        pick3 = sorted([(x, len(self.data[x])) for x in uid_3], key=lambda x: x[1], reverse=True)
        # filter out venues with less than or equal to 10 visits
        pid_3 = [x for x in self.venues if self.venues[x] > self.location_global_visit_min]
        # sort venues by the number of visits, descending order
        pid_pic3 = sorted([(x, self.venues[x]) for x in pid_3], key=lambda x: x[1], reverse=True)
        pid_3 = dict(pid_pic3)

        session_len_list = []
        for u in pick3:  # users with > 10 records. [(uid, number of records)]
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
                if poi not in pid_3 and poi not in topk1:  # filter out poi if visited <= 1
                    continue
                # else, add this [pid, tmd] as session
                if i == 0 or len(sessions) == 0:
                    sessions[sid] = [record]
                else:
                    # if minute gap since last record > 5 | last session has > 10 records, start new session
                    if (tid - last_tid) / 60 > self.minute_gap or len(sessions[sid - 1]) > self.session_max:
                        sessions[sid] = [record]
                    # if the record is apart from the last record for
                    # <= 72 hours, and
                    # > 10 minutes,
                    # then append record to the last session
                    elif (tid - last_tid) / 60 > self.min_gap:
                        sessions[sid - 1].append(record)
                    # if the record is apart from the last record for <= 4 minutes
                    else:
                        pass
                last_tid = tid

            sessions_filter = {}  # key: filtered sid, val: [[raw pid, raw tim]]
            for s in sessions:  # sid
                # sessions with records >= 5
                if len(sessions[s]) >= self.filter_short_session:
                    sessions_filter[len(sessions_filter)] = sessions[s]
                    session_len_list.append(len(sessions[s]))
            # if the filtered sessions(sessions with >= 5 records) are >= 5
            if len(sessions_filter) >= self.sessions_count_min:
                self.data_filter[uid] = {'sessions_count': len(sessions_filter), 'topk_count': len(topk), 'topk': topk,
                                         'sessions': sessions_filter, 'raw_sessions': sessions}

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
                poi = [p[0] for p in sessions[sid]]  # [[raw pid, tim]]
                for p in poi:
                    if p not in self.vid_list:
                        self.vid_list_lookup[len(self.vid_list)] = p
                        self.vid_list[p] = [len(self.vid_list), 1]
                    else:
                        self.vid_list[p][1] += 1

    # support for radius of gyration
    def load_venues(self):
        with open(self.TWITTER_PATH, 'r') as fid:
            if self.header:
                fid.readline()  # header
            for line in fid:
                taxi_seq, year, month, day, hour, minute, second, lat, lon = line.strip('\r\n').split('\t')
                pid = self.pois[(round(float(lon), 6), round(float(lat), 6))]
                if self.grid:
                    pid = self.poi_index_dict[pid]
                # self.pid_loc_lat[pid] = [round(float(lon), 6), round(float(lat), 6)]
                self.pid_loc_lat[pid] = self.center_location_list[pid]
                # self.pid_loc_lat[pid] = self.center_location_list_norm[pid]

    def venues_lookup(self):
        for vid in self.vid_list_lookup:  # int vid
            pid = self.vid_list_lookup[vid]  # raw pid
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
            sessions = self.data_filter[u]['sessions']  # key: sid, val: [[raw pid, raw tim]]
            sessions_tran = {}  # key: sid, val: [[int vid, int tid]]
            sessions_id = []
            for sid in sessions:
                sessions_tran[sid] = [[self.vid_list[p[0]][0], self.tid_list_48(p[1])] for p in
                                      sessions[sid]]  # [[int vid, int tid]]
                sessions_id.append(sid)
            split_id = int(np.floor(self.train_split * len(sessions_id)))  # 0.8
            split_valid_id = int(split_id + np.floor((len(sessions_id) - split_id) / 2))

            train_id = sessions_id[:split_id]  # 0.8 from the beginning
            valid_id = sessions_id[split_id:split_valid_id]
            test_id = sessions_id[split_valid_id:]
            # test_id = sessions_id[split_id:]  # the rest
            pred_len = sum([len(sessions_tran[i]) - 1 for i in train_id])  # train sid
            valid_len = sum([len(sessions_tran[i]) - 1 for i in test_id])  # test sid

            train_loc = {}  # key: int vid, val: # of visits
            for i in train_id:
                for sess in sessions_tran[i]:  # [int vid, int tid]
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

            # self.data_neural[self.uid_list[u][0]] = {'sessions': sessions_tran, 'train': train_id, 'test': test_id,
            self.data_neural[self.uid_list[u][0]] = {'sessions': sessions_tran,
                                                     'train': train_id, 'valid': valid_id, 'test': test_id,
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
        # parameters['hour_gap'] = self.hour_gap
        parameters['min_gap'] = self.min_gap
        parameters['minute_gap'] = self.minute_gap
        parameters['session_max'] = self.session_max
        parameters['filter_short_session'] = self.filter_short_session
        parameters['sessions_min'] = self.sessions_count_min
        parameters['train_split'] = self.train_split

        return parameters

    def save_variables(self):
        taxi_dataset = {'data_neural': self.data_neural, 'vid_list': self.vid_list, 'uid_list': self.uid_list,
                              'parameters': self.get_parameters(), 'data_filter': self.data_filter,
                              'vid_lookup': self.vid_lookup}
        pickle.dump(taxi_dataset, open(self.SAVE_PATH + self.save_name + '.pk', 'wb'))

        dataset = {'grid_dictionary': self.vid_lookup, 'center_location_list': self.center_location_list, 'center_location_list_norm': self.center_location_list_norm}
        pickle.dump(dataset, open(self.save_name + '_dictionary' + '.pk', 'wb'))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace_min', type=int, default=10, help="raw trace length filter threshold")
    # global_visit
    parser.add_argument('--global_visit', type=int, default=10, help="location global visit threshold")
    # parser.add_argument('--hour_gap', type=int, default=72, help="maximum interval of two trajectory points")
    parser.add_argument('--min_gap', type=int, default=4, help="minimum interval of two trajectory points")
    parser.add_argument('--minute_gap', type=int, default=5, help="maximum interval of two trajectory points")
    parser.add_argument('--session_max', type=int, default=10, help="control the length of session not too long")
    parser.add_argument('--session_min', type=int, default=5, help="control the length of session not too short")
    parser.add_argument('--sessions_min', type=int, default=5, help="the minimum amount of the good user's sessions")
    parser.add_argument('--train_split', type=float, default=0.7, help="train/test ratio")
    parser.add_argument('--header', type=bool, default=True, help="data has a header line")
    parser.add_argument('--grid', type=bool, default=True, help="use grid id instead of pid")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    data_generator = DataTaxi(trace_min=args.trace_min, global_visit=args.global_visit,
                                    minute_gap=args.minute_gap, min_gap=args.min_gap,
                                    session_min=args.session_min, session_max=args.session_max,
                                    sessions_min=args.sessions_min, train_split=args.train_split,
                                    grid=args.grid, header=args.header)
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
    print('raw users:{} raw locations:{}'.format(
        len(data_generator.data), len(data_generator.venues)))
    print('final users:{} final locations:{}'.format(
        len(data_generator.data_neural), len(data_generator.vid_list)))
