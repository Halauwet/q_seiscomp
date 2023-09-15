import sys
import shutil
import numpy as np
import requests
import subprocess
import pandas as pd
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits.basemap import Basemap
from glob import glob
from six import string_types
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from datetime import datetime as dt
from datetime import timezone as tz
# from obspy import read_inventory
# from obspy.clients.fdsn.client import Client
from scipy.spatial import Delaunay
from os import getcwd, mkdir, environ, listdir
from os.path import join, exists


def set_latency_unit(data_latency):
    """
    Function to set latency y units based on max value
    :param data_latency:
    :return:
    """
    if max(data_latency) < 600:
        units = 'seconds'
        series = data_latency
    elif max(data_latency) < 36000:
        units = 'minutes'
        series = data_latency / 60
    elif max(data_latency) < 864000:
        units = 'hours'
        series = data_latency / 3600
    elif max(data_latency) < 6048000:
        units = 'days'
        series = data_latency / 86400
    else:
        units = 'weeks'
        series = data_latency / 604800
    return series, units


def set_plot_nrows_ncols(num_plots, max_row=8):

    if num_plots > 24:
        print("Maximal plot is 24 stations. Only first 24 stations were plotted")
        num_plots = 24
    if num_plots > 2 * max_row:
        num_cols = 3
        num_rows = num_plots - int(2 * num_plots / num_cols)
    elif num_plots > max_row:
        num_cols = 2
        num_rows = num_plots - int(num_plots / num_cols)
    else:
        num_cols = 1
        num_rows = num_plots
    return num_rows, num_cols


def parse_inventory(inv_path):
    """
    Function to parse the XML Inventory file
    :return: root element
    """
    try:
        tree = ET.parse(inv_path)
        root = tree.getroot()
        return root
    except ET.ParseError as e:
        print(f"Error parsing {inv_path}: {e}")
        return None


def get_element(root, elem_name):
    # root = parse_inventory(inv_path)

    for elem in root[-1][-1][-1]:
        if elem_name in elem.tag:
            return elem.text


def compare_xml_elements(elem1, elem2):
    """
    Function to compare two XML elements recursively
    :param elem1:
    :param elem2:
    :return:
    """
    if elem1.tag != elem2.tag:
        return False
    if elem1.text != elem2.text:
        return False
    if elem1.tail != elem2.tail:
        return False
    if elem1.attrib != elem2.attrib:
        return False
    if len(elem1) != len(elem2):
        return False

    for child1, child2 in zip(elem1, elem2):
        if child1.tag != child2.tag:
            return False
        if child1.text != child2.text:
            return False
        if child1.tail != child2.tail:
            return False
        if child1.attrib != child2.attrib:
            return False
        if len(child1) != len(child2):
            return False

    return True


def assign_channel_priority(channel):
    """
    Function to set channel priority to be added in scproc
    :param channel:
    :return:
    """
    if 'SH' in channel:
        return 'High'
    elif 'BH' in channel:
        return 'Medium'
    elif 'HH' in channel:
        return 'low'
    else:
        return 'lower'


def alpha_shape(points, alpha, only_outer=True):
    """
    Function to compute the alpha shape (concave hull) of a set of points.

    :param points: np.array of shape (n,2) points.
    :param alpha: alpha value.
    :param only_outer: boolean value to specify if we keep only the outer border
    or also inner edges.
    :return: set of (i,j) pairs representing edges of the alpha-shape. (i,j) are
    the indices in the points array.
    """
    assert points.shape[0] > 3, "Need at least four points"

    def add_edge(edgess, i, j):
        """
        Add an edge between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edgess or (j, i) in edgess:
            # already added
            assert (j, i) in edgess, "Can't go twice over same directed edge right?"
            if only_outer:
                # if both neighboring triangles are in shape, it's not a boundary edge
                edgess.remove((j, i))
            return
        edgess.add((i, j))

    tri = Delaunay(points)
    edges = set()
    # Loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.simplices:
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]
        # Computing radius of triangle circumcircle
        # www.mathalino.com/reviewer/derivation-of-formulas/derivation-of-formula-for-radius-of-circumcircle
        a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
        s = (a + b + c) / 2.0
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)
        if circum_r < alpha:
            add_edge(edges, ia, ib)
            add_edge(edges, ib, ic)
            add_edge(edges, ic, ia)
    return edges


def find_edges_with(i, edge_set):
    i_first = [j for (x, j) in edge_set if x == i]
    i_second = [j for (j, x) in edge_set if x == i]
    return i_first, i_second


def stitch_boundaries(edges):
    edge_set = edges.copy()
    boundary_lst = []
    while len(edge_set) > 0:
        boundary = []
        edge0 = edge_set.pop()
        boundary.append(edge0)
        last_edge = edge0
        while len(edge_set) > 0:
            i, j = last_edge
            j_first, j_second = find_edges_with(j, edge_set)
            if j_first:
                edge_set.remove((j, j_first[0]))
                edge_with_j = (j, j_first[0])
                boundary.append(edge_with_j)
                last_edge = edge_with_j
            elif j_second:
                edge_set.remove((j_second[0], j))
                edge_with_j = (j, j_second[0])  # flip edge rep
                boundary.append(edge_with_j)
                last_edge = edge_with_j

            if edge0[0] == last_edge[1]:
                break

        boundary_lst.append(boundary)
    return boundary_lst


class QSeisComP:
    """
    Additional SeisComP module for optimizing the monitoring and processing system
    by: eQ_Halauwet (yehezkiel.halauwet@bmkg.go.id)

    :usages: copy q_seiscomp dir to seiscomp/lib/python

            register ts_latency to crontab:
                crontab -e
                */5 * * * * python /home/sysop/seiscomp/lib/python/q_seiscomp/ts_latency.py > q_seiscomp.log 2>&1

            run 3 main function after load the class instance:
                from q_seiscomp.QSeisComP import Q_SC

                Q_SC.plot_ts_latency("STATION_CODE") --> to plot time series after register ts_latency to crontab
                Q_SC.check_existing_configuration() --> to check and fix mismatch station configuration
                Q_SC.check_unexists_sts() --> to check and add unexists station on scproc observation area

    need additional package basemap and shapely: pip install pkgs
    """

    def __init__(self):
        self.sc_version = 4
        self.sc_schema = "0.11"

        self.etc_dir = join(environ['HOME'], "seiscomp", "etc")
        self.slmon_ts_dir = join(environ['HOME'], "seiscomp", "var", "lib", "slmon_ts")
        self.lib_dir = join(environ['HOME'], "seiscomp", "lib", "python", "q_seiscomp")
        self.check_sc_version()
        # self.etc_dir = join(getcwd(), 'etc')
        # self.slmon_ts_dir = join(getcwd(), 'slmon_ts')
        # self.lib_dir = getcwd()

        self.key_dir = join(self.etc_dir, 'key')
        self.inv_dir = join(self.etc_dir, 'inventory')
        self.sl_profile_dir = join(self.key_dir, 'seedlink')
        self.tmp_dir = join(self.lib_dir, "tmp")

        self.df_local_sts = pd.DataFrame()
        self.df_seedlink_servers = pd.DataFrame()
        self.df_seedlink_responses = pd.DataFrame()
        self.sl_profiles = []

        # cfg = seiscomp.config.Config()
        # cfg.readConfig(join(self.env.SEISCOMP_ROOT, "etc", "defaults", self.name + ".cfg"))
        # cfg.readConfig(join(self.env.SEISCOMP_ROOT, "etc", self.name + ".cfg"))

        self.current_time = dt.now(tz.utc).replace(tzinfo=None)

        if not exists(self.slmon_ts_dir):
            mkdir(self.slmon_ts_dir)
        if not exists(self.tmp_dir):
            mkdir(self.tmp_dir)

    def check_sc_version(self):
        """
        Method to get seiscomp version and data schema
        """

        current_env = environ.copy()
        additional_paths = ['/home/sysop/seiscomp/bin/']
        current_env['PATH'] = ":".join(additional_paths + [current_env.get('PATH', '')])

        completed_process = subprocess.run("seiscomp exec scrttv -V", shell=True, env=current_env,
                                           text=True, capture_output=True)

        if completed_process.returncode == 0:
            output = completed_process.stdout
            lines = output.splitlines()
            for l in lines:
                if "Framework" in l:
                    self.sc_version = int(l.split(" ")[1].split(".")[0])
                if "schema" in l:
                    self.sc_schema = l.split(" ")[3].strip()
            print(f"Seiscomp Major Version = {self.sc_version}")
            print(f"Seiscomp Data Schema = {self.sc_schema}")
        else:
            print("Error get seiscomp configuration")

    def get_existing_stations(self):
        """
        Method to obtain existing stations on scproc (through read the stream_key)
        """
        sts_key_list = [file for file in listdir(self.key_dir) if file.startswith("station")]

        # for stn_key in sts_key_list:
        #     if stn_key == "station_IA_RKPI": # "station_IA_SAUI"
        #         lc, ch, ad = self.read_stream_key(stn_key)
        sts_list = []
        for key in sts_key_list:
            s, net, stn = key.split("_")
            sts_list.append({
                'Stream_key': key.strip(),
                'Network_code': net.strip(),
                'Station_code': stn.strip(),
            })
        df = pd.DataFrame(sts_list)

        df[['Location_code', 'Channel', 'Ignore_amp']] = df.apply(lambda x: self.read_stream_key(x['Stream_key']),
                                                                  axis=1, result_type='expand')

        self.df_local_sts = df.dropna(subset=['Location_code', 'Channel'])

        self.df_local_sts = self.df_local_sts.reset_index(drop=True)

    def get_configured_hosts(self):
        """
        Method to obtain configured host, port, channel filter (through the seedlink profile key)
        """
        data_list = []
        for profile in self.sl_profiles:
            with open(join(self.sl_profile_dir, f"profile_{profile}"), "r") as sl_profile:
                p = sl_profile.readlines()
                for l in p:
                    if "sources.chain.address" in l:
                        l = l.strip().split("=")
                        host = l[1].strip()
                    if "sources.chain.port" in l:
                        l = l.strip().split("=")
                        port = l[1].strip()
                    if "sources.chain.selectors" in l:
                        l = l.strip().split("=")
                        filt = l[1].strip()
                    else:
                        filt = ""
                data_list.append({
                    'Host': host,
                    'Port': port,
                    'Filter': filt,
                })
        self.df_seedlink_servers = pd.DataFrame(data_list).drop_duplicates(subset=['Host', 'Port'])
        self.df_seedlink_servers = self.df_seedlink_servers.reset_index(drop=True)

    def read_stream_key(self, key):
        """
        Method to read the stream key
        """
        with open(join(self.key_dir, key), "r") as stream_key:
            keys = stream_key.readlines()
            if keys:
                for l in keys:
                    if "global" in l:
                        l = l.strip().split(":")
                        if len(l[1]) == 2:
                            lc = ''
                            ch = f"{l[1]}Z"
                            ad = ''
                        elif len(l[1]) == 4:
                            if 'AD' in l[1]:
                                lc = ''
                                ch = f"{l[1][:2]}Z"
                                ad = l[1][2:]
                            else:
                                lc = l[1][:2]
                                ch = f"{l[1][2:]}Z"
                                ad = ''
                        else:
                            lc = l[1][:2]
                            ch = f"{l[1][2:4]}Z"
                            ad = l[1][4:]
                    if "seedlink" in l:
                        if len(l.strip().split(":")) > 1:
                            if l.strip().split(":")[1] not in self.sl_profiles:
                                self.sl_profiles.append(l.strip().split(":")[1])
                        else:
                            self.sl_profiles.append(key)
                        break
            else:
                return None, None, ""

        return lc, ch, ad

    def stream_latency(self, save=True):
        """
        Method to obtain latency of the existing station on the scproc
        """
        # if self.df_local_sts.empty:
        self.get_existing_stations()
        self.get_configured_hosts()
        self.run_slinktool()

        self.df_seedlink_responses[['Latency', 'Latency_secs']] = self.df_seedlink_responses.apply(
            lambda x: self.calc_latency(x['Buffer_end']), axis=1, result_type='expand')

        self.df_local_sts = self.df_local_sts.merge(self.df_seedlink_responses,
                                                    on=['Network_code', 'Station_code', 'Location_code', 'Channel'],
                                                    how='left')
        if save:
            print(self.df_local_sts[['Network_code', 'Station_code', 'Location_code', 'Channel', 'Latency']])
            self.write_ts_latency()
        else:
            print(self.df_local_sts[['Network_code', 'Station_code', 'Location_code', 'Channel', 'Latency']])

    def run_slinktool(self, filt_station=False):
        """
        Method to fetch the response from the slinktool
        """

        current_env = environ.copy()
        additional_paths = ['/home/eq/slinktool', '/home/sysop/seiscomp/bin/']
        current_env['PATH'] = ":".join(additional_paths + [current_env.get('PATH', '')])

        self.df_seedlink_responses = pd.DataFrame()

        for i, r in self.df_seedlink_servers.iterrows():
            if filt_station:
                completed_process = subprocess.run(f"slinktool -Q {r['Host']}:{r['Port']} | grep {filt_station}",
                                                   shell=True, env=current_env, text=True, capture_output=True)
            else:
                completed_process = subprocess.run(f"slinktool -Q {r['Host']}:{r['Port']}",
                                                   shell=True, env=current_env, text=True, capture_output=True)

            if completed_process.returncode == 0:
                output = completed_process.stdout
                lines = output.splitlines()

                data_list = []
                for l in lines:
                    if l.strip():
                        data_list.append({
                            'Network_code': l[:3].strip(),
                            'Station_code': l[3:9].strip(),
                            'Location_code': l[9:12].strip(),
                            'Channel': l[12:16].strip(),
                            'Record_type': l[16:18].strip(),
                            'Buffer_start': l[18:44].strip(),
                            'Buffer_end': l[45:].strip()
                        })

                df2 = pd.DataFrame(data_list)
                self.df_seedlink_responses = pd.concat([self.df_seedlink_responses, df2], ignore_index=True)

            else:
                if r['Filter']:
                    print(f"Error fetch seedlink from {r['Host']}:{r['Port']} | grep {filt_station}")
                else:
                    print(f"Error fetch seedlink from {r['Host']}:{r['Port']}")

        self.df_seedlink_responses = self.df_seedlink_responses.drop_duplicates(subset=['Network_code', 'Station_code',
                                                                                        'Location_code', 'Channel'])
        self.df_seedlink_responses = self.df_seedlink_responses.reset_index(drop=True)

    def calc_latency(self, end_buffer):
        """
        Method to calculate the latency data

        :return: latency in datetime format and seconds unit
        """

        self.current_time = current_time = dt.now(tz.utc).replace(tzinfo=None)
        end_buffer = pd.to_datetime(end_buffer).to_pydatetime()
        return (current_time - end_buffer), (current_time - end_buffer).total_seconds()

    def write_ts_latency(self):
        """
        Method to store calculated latencies to time series data in self.slmon_ts_dir directory

        :output: time series latency data
        """
        for i, r in self.df_local_sts.iterrows():
            sts_ts_latency = join(self.slmon_ts_dir, f'{r["Network_code"]}.{r["Station_code"]}.'
                                                     f'{r["Location_code"]}.{r["Channel"][:2]}')

            latency_data = (f'{self.current_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]}\t'
                            f'{self.current_time.timestamp()}\t'
                            f'{r["Latency"]}\t{r["Latency_secs"]}')
            # latency_data = f'{self.current_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]}\t{r["Latency_secs"]}'

            if exists(sts_ts_latency):  # write finished event to finished catalog
                with open(sts_ts_latency, 'a') as f:
                    f.write(f'{latency_data}\n')
            else:
                with open(sts_ts_latency, 'w') as f:
                    f.write(f'Timestamp\tTimestamp(sec)\tLatency\tLatency(sec)\n{latency_data}\n')
                    # f.write(f'Timestamp\tLatency(sec)\n{latency_data}\n')
        print(f'Latency data written to: {self.slmon_ts_dir}')

    def plot_ts_latency(self, station, dt_from=None, dt_to=None):
        """
        Method to plot time series of latency data

        :param station: station code
        :param dt_from: date from (YYYY-M-D H:m:s) UTC timezone
        :param dt_to: date to (YYYY-M-D H:m:s) UTC timezone
        :return: plot of latency time series

        :usages:   Q_SC.plot_ts_latency("STATION_CODE", from_datetime, to_datetime)

        :example: Q_SC.plot_ts_latency(["AAI", "WSTMM", "MSAI", "KRAI"]) --> plot several stations data
                  Q_SC.plot_ts_latency("AAI", "2023-8-29 00:00:00", "2023-8-30 00:01:00")  --> plot an hour of AAI data
                  Q_SC.plot_ts_latency("AAI", "2023-8-20", "2023-8-30")  --> plot 10 days of AAI data
                  Q_SC.plot_ts_latency("AAI")  --> plot all of AAI data
        """
        group = None
        if isinstance(station, string_types):
            station = [station]
        elif isinstance(station, list):
            pass
        if dt_from is None:
            dt_from = dt(1970, 1, 1, 0, 0, 0)
        else:
            dt_from = pd.to_datetime(dt_from)
        if dt_to is None:
            dt_to = dt.now(tz.utc).replace(tzinfo=None)
        else:
            dt_to = pd.to_datetime(dt_to)

        files = listdir(self.slmon_ts_dir)
        dict_data = {sta: next((f for f in files if f.split(".")[1] == sta), None) for sta in station}
        filt_dict = {key: value for key, value in dict_data.items() if value is not None}
        unuse_dict = {key: value for key, value in dict_data.items() if value is None}

        rows, cols = set_plot_nrows_ncols(len(filt_dict))
        if rows > 1:
            fig, ax = plt.subplots(rows, cols, figsize=(6 * cols, 2 * rows), sharex='all')
        else:
            fig, ax = plt.subplots(figsize=(8, 5))
        plt.rcParams.update({'font.size': 7})
        if unuse_dict:
            for sta in unuse_dict.keys():
                print(f"Station {sta} data not found on {self.slmon_ts_dir}")
        if filt_dict:
            min_xthick = dt(3000, 1, 1, 0, 0, 0)
            max_xthick = dt(1970, 1, 1, 0, 0, 0)
            i = j = 0
            for sta, file in filt_dict.items():
                if i >= rows:
                    # plt.tick_params(axis='both', which='both', labelsize=5)
                    j += 1
                    i = 0
                data = pd.read_csv(join(self.slmon_ts_dir, file), parse_dates=['Timestamp'], sep="\t")
                df = data.set_index('Timestamp')
                df = df[(df.index >= dt_from) & (df.index <= dt_to)]
                if group:
                    df['Latency(sec)'].resample(group).sum().plot(kind='bar', ax=ax[i])
                    if group == 'D':
                        if cols > 1:
                            ax[i, j].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%M-%d'))
                            ax[i, j].set_xlabel('Day', fontsize=6)
                        else:
                            if rows > 1:
                                ax[i].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%M-%d'))
                                ax[i].set_xlabel('Day', fontsize=6)
                            else:
                                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%M-%d'))
                                ax.set_xlabel('Day', fontsize=6)
                    else:
                        if cols > 1:
                            ax[i, j].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                            ax[i, j].set_xlabel('Month', fontsize=6)
                        else:
                            if rows > 1:
                                ax[i].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                                ax[i].set_xlabel('Month', fontsize=6)
                            else:
                                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                                ax.set_xlabel('Month', fontsize=6)
                    if rows > 1:
                        ax[i, j].set_ylabel('Latency (s)', fontsize=6)
                    else:
                        ax.set_ylabel('Latency (s)', fontsize=6)
                else:
                    latency_plot, unit = set_latency_unit(df['Latency(sec)'])
                    if df.index[-1] > max_xthick:
                        max_xthick = df.index[-1]
                    if df.index[0] < min_xthick:
                        min_xthick = df.index[0]
                    if cols > 1:
                        latency_plot.plot(ax=ax[i, j], label=f"Station {sta}")
                        ax[i, j].set_ylabel(f'Latency ({unit})', fontsize=6)
                        ax[i, j].legend(loc="upper right")
                    else:
                        if rows > 1:
                            latency_plot.plot(ax=ax[i], label=f"Station {sta}")
                            ax[i].set_ylabel(f'Latency ({unit})', fontsize=6)
                            ax[i].legend(loc="upper right")
                        else:
                            latency_plot.plot(ax=ax, label=f"Station {sta}")
                            ax.set_ylabel(f'Latency ({unit})', fontsize=6)
                            ax.legend(loc="upper right")
                # if cols > 1:
                #     ax[i, j].set_title(f"Station {sta}", fontsize=7)
                # else:
                #     if rows > 1:
                #         ax[i].set_title(f"Station {sta}", fontsize=7)
                #     else:
                #         ax.set_title(f"Station {sta}", fontsize=7)
                i += 1
            if group:
                fig.suptitle('Daily Bar Chart', fontsize=8)
            else:
                td_hours = (max_xthick - min_xthick).total_seconds() / 3600
                td_mins = (max_xthick - min_xthick).total_seconds() / 60
                if td_hours / 12 < 1:
                    if cols > 1:
                        ax[0, 0].xaxis.set_major_locator(mdates.MinuteLocator(interval=int(td_mins / 12)))
                        ax[0, 0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                    else:
                        if rows > 1:
                            ax[-1].xaxis.set_major_locator(mdates.MinuteLocator(interval=int(td_mins / 12)))
                            ax[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                        else:
                            ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=int(td_mins / 12)))
                            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                else:
                    if cols > 1:
                        ax[0, 0].xaxis.set_major_locator(mdates.HourLocator(interval=int(td_hours / 12)))
                        ax[0, 0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
                    else:
                        if rows > 1:
                            ax[-1].xaxis.set_major_locator(mdates.HourLocator(interval=int(td_hours / 12)))
                            ax[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
                            ax[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
                        else:
                            ax.xaxis.set_major_locator(mdates.HourLocator(interval=int(td_hours / 12)))
                            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
                            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
                for j in range(cols):
                    plt.xticks(rotation=30)
                    if cols > 1:
                        ax[-1, j].set_xlabel('Time', fontsize=6)
                        ax[-1, j].tick_params(axis='both', labelsize=5)
                    else:
                        if rows > 1:
                            ax[-1].set_xlabel('Time', fontsize=6)
                            ax[-1].tick_params(axis='both', labelsize=5)
                        else:
                            ax.set_xlabel('Time', fontsize=6)
                            ax.tick_params(axis='both', labelsize=5)
                fig.suptitle('Latency Time Series', fontsize=8)
            plt.tight_layout()
            plt.ion()
            plt.show(block=False)
        else:
            print(f"Stations data not found on {self.slmon_ts_dir}")
        plt.ioff()

    def check_existing_configuration(self):
        """
        Method to match configured stations on scproc with active stream on the seedlink

        :return: info about mismatch station, recommended configuration, and option to fix the configuration

        :usages:  Q_SC.check_existing_configuration()
        """
        # if self.df_local_sts.empty:
        self.get_existing_stations()
        if self.df_seedlink_responses.empty:
            self.get_configured_hosts()
            self.run_slinktool()
        error, list_recom = self.check_station()
        print(error)
        add_sts = input("Fix current configuration? ([Y]/N)") or "Y"
        if add_sts == 'Y' or add_sts == 'y':
            self.add_station(list_recom, checked=True)

    def check_station(self):
        """
        Method to check a station configuration is match with seedlink

        :return: str details of error configuration and list of recommended configuration
        """
        pd.options.mode.chained_assignment = None
        merged_df = self.df_local_sts.merge(self.df_seedlink_responses,
                                            on=['Network_code', 'Station_code', 'Location_code', 'Channel'],
                                            how='left')

        error_sta = merged_df[merged_df['Buffer_start'].isna()]

        self.df_seedlink_responses['Init_Channel'] = self.df_seedlink_responses['Channel'].str[:2]

        error_detail = ""
        recomm_sts = []
        for i, r in error_sta.iterrows():
            err_sta = f'{r["Network_code"]}.{r["Station_code"]}.{r["Location_code"]}.{r["Channel"][:2]}'
            recomend_detail = ""
            recomend_sts = self.df_seedlink_responses[self.df_seedlink_responses['Station_code'] == r['Station_code']]
            recomend_sts["Priority"] = (recomend_sts['Channel'].apply(assign_channel_priority))
            recomend_sts = recomend_sts[~recomend_sts['Init_Channel'].duplicated()]
            recomend_sts = recomend_sts.sort_values(by='Priority', ascending=True)
            recomend_sts = recomend_sts[recomend_sts['Priority'] != 'lower']
            found = False
            recom_sts = ""
            for ii, rr in recomend_sts.iterrows():
                recomend_detail += (f'"{rr["Network_code"]}.{rr["Station_code"]}.{rr["Location_code"]}.'
                                    f'{rr["Init_Channel"]}"\n                    ')
                if rr['Init_Channel'] == "SH" or rr['Init_Channel'] == "BH":
                    for item in recomm_sts:
                        if rr['Station_code'] in item:
                            found = True
                            break
                    if not found:
                        recom_sts = (f"{rr['Network_code']}.{rr['Station_code']}."
                                     f"{rr['Location_code']}.{rr['Init_Channel']}")

            if err_sta not in recomend_detail:
                error_detail += (f'\nStation "{r["Network_code"]}.{r["Station_code"]}.{r["Location_code"]}.'
                                 f'{r["Channel"][:2]}" is not available in seedlink stream. \n'
                                 f'Recommended stream: {recomend_detail}')
                recomm_sts.append(recom_sts)

        return error_detail, recomm_sts

    def add_station(self, full_id, use_amplitude=True, checked=False, iters=False):
        """
        Method to add station/s to scproc

        :param full_id: <list/string> full station ID separated by a dot: NET.STA.LOC.CH
        :param use_amplitude: <bool> using amplitude in magnitude calculation
        :param checked: <bool> already checked for seedlink availability
        :param iters: <bool> set true if added simultaneously (update-config on last station add)

        :usages:  Q_SC.add_station(["STATION_CODE"])

        :example: Q_SC.add_station("IA.AAI..BH")  --> add a station
                  Q_SC.add_station(["IA.AAI..BH", "IA.BNDI..BH"])  --> add several stations
        """
        if isinstance(full_id, string_types):
            full_id = [full_id]
        elif isinstance(full_id, list):
            pass

        amp = "" if use_amplitude else "AD"
        for stn in full_id:
            try:
                net, sta, loc, cha = stn.split('.')
            except ValueError:
                print('Wrong full id. Full id format: "NET.STA.LOC.CH". example: "IA.AAI..BH"')
                sys.exit(1)

            if not checked:
                self.df_local_sts = pd.DataFrame([{'Stream_key': f"station_{net}_{sta}", 'Network_code': net,
                                                   'Station_code': sta, 'Location_code': loc, 'Channel': cha,
                                                   'Ignore_amp': amp}])
                self.df_seedlink_servers = pd.DataFrame([{'Host': 'geof.bmkg.go.id', 'Port': '18000'}])
                self.run_slinktool(filt_station=sta)
                error, list_recom = self.check_station()
                if error:
                    print(error)
                    print("Canceling add station...")
                    sys.exit(1)

            self.get_inventory(net, sta)
            # self.compare_inventory(net, sta)
            self.update_inventory(net, sta)
            self.write_key(net, sta, loc, cha, use_amplitude, sta_SL_profile=False)

        if not iters:
            self.update_config()

    def get_inventory(self, network, station):
        """
        Method to get updated metadata

        :param network:
        :param station:
        """
        inv_url = (f"https://geof.bmkg.go.id/fdsnws/station/1/query?network={network}&"
                   f"station={station}&level=response&format=sc3ml&nodata=404")

        update_inv_dir = join(self.inv_dir, "update")

        local_inv = join(update_inv_dir, f"{network}.{station}.xml")

        if not exists(update_inv_dir):
            mkdir(update_inv_dir)

        response = requests.get(inv_url)

        if response.status_code == 200:
            if self.sc_version <= 4:
                original_content = response.text
                modified_content = original_content.replace('version="0.12"', f'version="{self.sc_schema}"')
                modified_content = modified_content.replace('gfz-potsdam.de/ns/seiscomp3-schema/0.12"',
                                                            f'gfz-potsdam.de/ns/seiscomp3-schema/{self.sc_schema}"')

                with open(local_inv, "w") as output_inv:
                    output_inv.write(modified_content)
                print(f"\nInventory is downloaded and modified to seiscomp {self.sc_version},"
                      f" scheme saved at {local_inv}")
            else:
                with open(local_inv, 'wb') as file:
                    file.write(response.content)
                print(f"\nInventory is downloaded to {local_inv}")
        else:
            print(f"Failed to download the {network}.{station} inventory. Status code: {response.status_code}")

    def compare_inventory(self, network, station):
        """
        Method to check if two inventory is match or there is an update

        :param network:
        :param station:
        """

        inv1_path = join(self.inv_dir, f"{network}.{station}.xml")
        inv2_path = join(self.inv_dir, "update", f"{network}.{station}.xml")

        root1 = parse_inventory(inv1_path)
        root2 = parse_inventory(inv2_path)

        if root1 is not None and root2 is not None:
            if compare_xml_elements(root1, root2):
                print("The XML inventories are the same.")
            else:
                self.update_inventory(network, station)
                print("The XML inventories are different.")
        else:
            print("One or both XML files could not be parsed.")

    def update_inventory(self, network, station):
        """
        Method to update scproc inventory (preserve old inventory)

        :param network:
        :param station:
        """
        old_inv_dir = join(self.inv_dir, "old")

        if not exists(old_inv_dir):
            mkdir(old_inv_dir)

        old_inv = join(self.inv_dir, f"{network}.{station}.xml")

        if exists(old_inv):
            if exists(join(old_inv_dir, f"{network}.{station}.xml")):
                for i in range(99):
                    renamed_inv = f'{network}.{station}_{i + 1:02}.xml'
                    if not glob(join(old_inv_dir, renamed_inv)):
                        try:
                            shutil.move(old_inv, join(old_inv_dir, renamed_inv))
                        except (IOError, OSError, shutil.Error):
                            print(f'Not fully moving inventory "{network}.{station}"')
                        break
            else:
                try:
                    shutil.move(old_inv, old_inv_dir)
                except FileNotFoundError:
                    print("Source inventory not found.")
                except PermissionError:
                    print("Permission denied. Make sure you have the necessary permissions.")
                except Exception as e:
                    print(f"An error occurred: {e}")

        new_inv = join(self.inv_dir, "update", f"{network}.{station}.xml")

        try:
            shutil.move(new_inv, join(self.inv_dir))
            print(f"Local inventory updated")
        except FileNotFoundError:
            print("Source inventory not found.")
        except PermissionError:
            print("Permission denied. Make sure you have the necessary permissions.")
        except Exception as e:
            print(f"An error occurred: {e}")

    def write_key(self, network, station, location, channel, use_amplitude, sta_SL_profile=False):
        """
        Method to write key to scproc

        :param network:
        :param station:
        :param location:
        :param channel:
        :param use_amplitude:
        :param sta_SL_profile:
        """
        key_path = join(self.key_dir, f"station_{network}_{station}")

        amp = "" if use_amplitude else "AD"
        if exists(key_path):
            old_key_dir = join(self.key_dir, "old")
            if not exists(old_key_dir):
                mkdir(old_key_dir)
            if exists(join(old_key_dir, f"station_{network}_{station}")):
                for i in range(99):
                    renamed_key = f"station_{network}_{station}_{i + 1:02}"
                    if not glob(join(old_key_dir, renamed_key)):
                        try:
                            shutil.move(key_path, join(old_key_dir, renamed_key))
                        except (IOError, OSError, shutil.Error):
                            print(f'Not fully moving inventory "{network}.{station}"')
                        break
            else:
                try:
                    shutil.move(key_path, old_key_dir)
                except FileNotFoundError:
                    print("Source inventory not found.")
                except PermissionError:
                    print("Permission denied. Make sure you have the necessary permissions.")
                except Exception as e:
                    print(f"An error occurred: {e}")

        with open(key_path, 'w') as f:
            f.write("# Binding references\n")
            f.write(f"global:{location}{channel}{amp}\n")
            f.write("scautopick:default\n")
            if sta_SL_profile:
                f.write("seedlink\n")
                # todo: self.write_seedlink_profile()
            else:
                f.write(f"seedlink:pst_{location}{channel}\n")
            f.write("slarchive:default_30d\n")
            f.write("access\n")
        print(f'Key "station_{network}_{station}" is written\n')

    def check_unexists_sts(self, alpha=6):
        """
        Method to detect unexisted station on scproc and automatically configured it

        :param alpha: concave parameter to estimate searching area based on alpha shape of existing stations

        :usages:  Q_SC.check_existing_configuration()

        """
        pd.options.mode.chained_assignment = None

        local_coord = self.read_local_coordinate()
        server_coord = self.read_server_coordinate()

        server_coord_filt = server_coord[~server_coord['Station_code'].isin(local_coord['Station_code'])]
        server_coord_filt["Location_code"] = server_coord_filt["Location_code"].copy().fillna('')

        sts_locs = local_coord[['Station_lon', 'Station_lat']].values

        # min_x = local_coord['Station_lon'].min()
        # max_x = local_coord['Station_lon'].max()
        # min_y = local_coord['Station_lat'].min()
        # max_y = local_coord['Station_lat'].max()

        llon = 90
        rlon = 150
        blat = -18
        ulat = 12
        margin_area = []
        fig, ax = plt.subplots(figsize=(8, 6))

        while alpha != 'Y':

            if alpha == 'y':
                break
            try:
                alpha = float(alpha)
            except ValueError:
                sys.exit(1)

            margin_area = []

            area_edges = alpha_shape(sts_locs, alpha=alpha, only_outer=True)
            area_edges = stitch_boundaries(area_edges)

            for i, j in area_edges[0]:
                margin_area.append((sts_locs[[i, j], 0][0], sts_locs[[i, j], 1][0]))
                margin_area.append((sts_locs[[i, j], 0][1], sts_locs[[i, j], 1][1]))

            lons = [item[0] for item in margin_area]
            lats = [item[1] for item in margin_area]

            llon = min(lons)-((max(lons)-min(lons))/15)
            rlon = max(lons)+((max(lons)-min(lons))/15)
            blat = min(lats)-((max(lats)-min(lats))/15)
            ulat = max(lats)+((max(lats)-min(lats))/15)

            # Create a Basemap instance with a specific projection (e.g., Mercator)
            m = Basemap(projection='merc', llcrnrlat=blat, urcrnrlat=ulat,
                        llcrnrlon=llon, urcrnrlon=rlon, resolution='l')
            m.drawcoastlines()
            m.fillcontinents(color='lightgray', lake_color='aqua')
            m.scatter(sts_locs[:, 0], sts_locs[:, 1], latlon=True, marker='v', color='red',
                      label='Existing Stations')

            x, y = m(lons, lats)
            m.plot(x, y, marker=None, linewidth=0.7, color='blue', linestyle='dashed', label="Searching area")
            m.drawparallels(range(-90, 91, 5), labels=[1, 0, 0, 1], fontsize=12, fontweight='bold')
            m.drawmeridians(range(-180, 181, 5), labels=[1, 0, 0, 1], fontsize=12, fontweight='bold')

            # ax.set_xlabel("Longitude")
            # ax.set_ylabel("Latitude")
            plt.legend()
            plt.ion()
            plt.show(block=False)

            alpha = input("Check on current area? (Y/new_alpha)\n"
                          f"Set smaller alpha to ignore outlier far station and vice versa (current alpha = {alpha})\n")

            fig.clf()

        # Create a Polygon object from the local stst area
        polygon_area = Polygon(margin_area)

        # Check if the server remaining sts is inside the polygon
        recomm_sts = []
        recomm_sts_loc = np.empty([0, 2])
        for i, r in server_coord_filt.iterrows():
            point = Point((float(r["Station_lon"]), float(r["Station_lat"])))
            is_inside = polygon_area.contains(point)

            if is_inside:
                recomm_sts.append(f"{r['Network_code']}.{r['Station_code']}.{r['Location_code']}.{r['Init_Channel']}")
                recomm_sts_loc = np.append(recomm_sts_loc,
                                           [np.array([float(r["Station_lon"]), float(r["Station_lat"])])], axis=0)

        m = Basemap(projection='merc', llcrnrlat=blat, urcrnrlat=ulat,
                    llcrnrlon=llon, urcrnrlon=rlon, resolution='l')
        m.drawcoastlines()
        m.fillcontinents(color='lightgray', lake_color='aqua')
        m.scatter(sts_locs[:, 0], sts_locs[:, 1], latlon=True, marker='v', color='red', label='Existing Stations')
        m.scatter(recomm_sts_loc[:, 0], recomm_sts_loc[:, 1], latlon=True,
                  marker='v', color='blue', label='New Stations')
        m.drawparallels(range(-90, 91, 5), labels=[1, 0, 0, 1], fontsize=12, fontweight='bold')
        m.drawmeridians(range(-180, 181, 5), labels=[1, 0, 0, 1], fontsize=12, fontweight='bold')

        xs, ys = m(recomm_sts_loc[:, 0], recomm_sts_loc[:, 1])
        for sts, s_lon, s_lat in zip(recomm_sts, xs, ys):
            plt.text(s_lon, s_lat, sts.split('.')[1], fontsize=7.5, ha='center', va='bottom', color='k', weight='bold')
        # ax.set_xlabel("Longitude")
        # ax.set_ylabel("Latitude")
        plt.legend()
        plt.show(block=False)

        print(f"There are {len(recomm_sts)} new stations that can be added:")
        division = int(len(recomm_sts)/10) if len(recomm_sts) > 10 else 1
        for i, item in enumerate(recomm_sts):
            print(item.ljust(15), end='\n' if (i + 1) % division == 0 else ' ')

        print("\nHow do you want to add the new stations:")
        print("1. Check and add station one by one [default]")
        print("2. Add all stations directly")
        print("3. Cancel")

        choice = input("") or '1'

        fig.clf()

        if choice == '1':
            for sta, loc in zip(recomm_sts, recomm_sts_loc):

                m = Basemap(projection='merc', llcrnrlat=loc[1] - 5, urcrnrlat=loc[1] + 5,
                            llcrnrlon=loc[0] - 6, urcrnrlon=loc[0] + 6, resolution='l')
                m.drawcoastlines()
                m.fillcontinents(color='lightgray', lake_color='aqua')
                m.scatter(sts_locs[:, 0], sts_locs[:, 1], latlon=True, marker='v', color='red',
                          label='Existing Stations')
                m.scatter(loc[0], loc[1], latlon=True,
                          marker='v', color='blue', label=f'New Station "{sta}"')
                xx, yy = m(loc[0], loc[1])
                plt.text(xx, yy, sta.split('.')[1], fontsize=6, ha='center', va='bottom', color='k',
                         weight='bold')
                m.drawparallels(range(-90, 91, 5), labels=[1, 0, 0, 1], fontsize=12, fontweight='bold')
                m.drawmeridians(range(-180, 181, 5), labels=[1, 0, 0, 1], fontsize=12, fontweight='bold')
                # ax.set_xlabel("Longitude")
                # ax.set_ylabel("Latitude")
                plt.legend(loc='upper right')
                plt.show(block=False)

                add_sts = input(f"Add station {sta}? ([Y]/N/X for cancel)") or "Y"
                if add_sts == 'Y' or add_sts == 'y':
                    self.add_station(sta, checked=True, iters=True)
                if add_sts == 'X' or add_sts == 'x':
                    plt.ioff()
                    plt.close()
                    sys.exit(1)
                fig.clf()
            plt.ioff()
            plt.close()
            self.update_config()
        elif choice == '2':
            self.add_station(recomm_sts, checked=True)
        elif choice == '3':
            print("Canceling...")
            plt.ioff()
            plt.close()
            # update_config()
        else:
            print("Invalid choice. Canceling...")
            plt.ioff()
            plt.close()

    def read_local_coordinate(self):
        """
        Method to get existing scproc station location information

        :return: list of stations with the coordinates
        """
        if exists(join(self.tmp_dir, 'local_coord.csv')):
            sts_list = pd.read_csv(join(self.tmp_dir, 'local_coord.csv'))
        else:
            sts_list = pd.DataFrame(columns=['Network_code', 'Station_code', 'Station_lat', 'Station_lon'])

        update = False
        sts_inv_list = [file for file in listdir(self.inv_dir) if file.endswith(".xml")]
        for inv in sts_inv_list:
            net, stn, ext = inv.split(".")
            if stn not in sts_list['Station_code'].values:
                tree = ET.parse(join(self.inv_dir, inv))
                root = tree.getroot()
                lon = get_element(root, 'longitude')
                lat = get_element(root, 'latitude')
                new_sts_data = {
                    'Network_code': net,
                    'Station_code': stn,
                    'Station_lat': lat,
                    'Station_lon': lon}
                sts_list = pd.concat([sts_list, pd.DataFrame([new_sts_data])], ignore_index=True)
                update = True

        if update:
            sts_list = sts_list.drop_duplicates(subset=['Network_code', 'Station_code'])
            sts_list = sts_list.reset_index(drop=True)
            sts_list.to_csv(join(self.tmp_dir, 'local_coord.csv'), index=False)

        return sts_list

    def read_server_coordinate(self):
        """
        Method to get seedlink station location information

        :return: list of stations with the coordinates
        """
        if exists(join(self.tmp_dir, 'server_coord.csv')):
            sts_list = pd.read_csv(join(self.tmp_dir, 'server_coord.csv'))
        # elif exists(join(self.tmp_dir, 'local_coord.csv')):
        #     sts_list = pd.read_csv(join(self.tmp_dir, 'local_coord.csv'))
        else:
            sts_list = pd.DataFrame(columns=['Network_code', 'Station_code', 'Location_code',
                                             'Init_Channel', 'Station_lat', 'Station_lon'])

        self.df_seedlink_servers = pd.DataFrame([{'Host': 'geof.bmkg.go.id', 'Port': '18000'}])
        self.run_slinktool()
        self.df_seedlink_responses["Priority"] = (self.df_seedlink_responses['Channel'].
                                                  apply(assign_channel_priority))
        self.df_seedlink_responses = self.df_seedlink_responses.sort_values(by='Priority', ascending=True)
        self.df_seedlink_responses = self.df_seedlink_responses.drop_duplicates(subset=['Network_code', 'Station_code'],
                                                                                keep='first')
        self.df_seedlink_responses = self.df_seedlink_responses[self.df_seedlink_responses['Priority'] != 'lower']
        self.df_seedlink_responses = self.df_seedlink_responses.sort_values(by=['Network_code', 'Station_code'],
                                                                            ascending=True)

        self.df_seedlink_responses = self.df_seedlink_responses.reset_index(drop=True)
        self.df_seedlink_responses['Init_Channel'] = self.df_seedlink_responses['Channel'].str[:2]

        update = False
        for i, r in self.df_seedlink_responses.iterrows():
            if r['Station_code'] not in sts_list['Station_code'].values:
                inv_url = (f"https://geof.bmkg.go.id/fdsnws/station/1/query?network={r['Network_code']}&"
                           f"station={r['Station_code']}&level=response&format=sc3ml&nodata=404")
                response = requests.get(inv_url)
                if response.status_code == 200:
                    xml_data = response.text
                    root = ET.fromstring(xml_data)
                else:
                    continue

                lon = get_element(root, 'longitude')
                lat = get_element(root, 'latitude')

                new_sts_data = {
                    'Network_code': r['Network_code'],
                    'Station_code': r['Station_code'],
                    'Location_code': r['Location_code'],
                    'Init_Channel': r['Init_Channel'],
                    'Station_lat': lat,
                    'Station_lon': lon}
                sts_list = pd.concat([sts_list, pd.DataFrame([new_sts_data])], ignore_index=True)
                update = True
        if update:
            sts_list = sts_list.drop_duplicates(subset=['Network_code', 'Station_code'])
            sts_list = sts_list.reset_index(drop=True)
            sts_list.to_csv(join(self.tmp_dir, 'server_coord.csv'), index=False)

        return sts_list

    @staticmethod
    def update_config():
        """
        Function to update the configurations and restart seiscomp services
        """
        current_env = environ.copy()
        additional_paths = ['/home/sysop/seiscomp/bin/']
        current_env['PATH'] = ":".join(additional_paths + [current_env.get('PATH', '')])

        completed_process = subprocess.run("seiscomp update-config", shell=True, env=current_env,
                                           text=True, capture_output=True)

        if completed_process.returncode == 0:
            completed_process = subprocess.run("seiscomp restart", shell=True, env=current_env,
                                               text=True, capture_output=True)
            if completed_process.returncode == 0:
                print("Seiscomp configuration updated. Please restart the GUIs")
            else:
                print("Seiscomp configuration updated. Please restart the seiscomp service and the the GUIs")
        else:
            print("Error update seiscomp configuration. "
                  "Please run 'seiscomp update-config' manually then restart the seiscomp services and the GUIs")


if __name__ == "__main__":
    Q_SC = QSeisComP()
    help(Q_SC)

Q_SC = QSeisComP()
PGRIX = ["AAI", "AAII", "TAMI", "KRAI", "MSAI", "NLAI", "SRMI", "NBMI", "SEMI", "BNDI", "BSMI", "SSMI",
         "TLE2", "KTMI", "KKMI", "SAUI", "ARMI", "TMTMM", "WSTMM", "NSBMM", "TTSMI", "PBMMI", "MLMMI"]
# TESTING
# Q_SC.plot_ts_latency("KRAI")
# Q_SC.plot_ts_latency(PGRIX)
# Q_SC.plot_ts_latency(["WSTMM", "DGF", "KRAI", "ABC"], "2023-8-29 00:00:00", "2023-8-30 00:01:00")
# Q_SC.check_existing_configuration()
# Q_SC.check_unexists_sts()
# Q_SC.get_inventory("IA", "AAI")
# Q_SC.compare_inventory("IA", "AAI")
# Q_SC.update_inventory("IA", "AAI")
# Q_SC.add_station("IA.AAI..SH")
