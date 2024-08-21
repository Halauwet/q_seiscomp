import pandas as pd
import matplotlib.pyplot as plt
import os, inspect, timeit, sys, numpy as np
from obspy import read, UTCDateTime, Trace
from obspy.core.inventory.inventory import read_inventory
from obspy.signal.spectral_estimation import PPSD  # import load_npz
from obspy.signal.spectral_estimation import get_nhnm
from obspy.signal.spectral_estimation import get_nlnm
from datetime import datetime as dt
from datetime import timedelta as td
from six import string_types
from obspy.imaging.cm import pqlx
from datetime import timezone as tz
from os.path import join, exists
from os import getcwd, mkdir, environ, listdir


class QSeisComP:
    """
    Additional SeisComP module for optimizing the monitoring and processing system
    by: eQ_Halauwet (yehezkiel.halauwet@bmkg.go.id)

    :usages: copy q_seiscomp dir to seiscomp/lib/python

            register ts_latency to crontab:
                crontab -e
                */5 * * * * python /home/sysop/seiscomp/lib/python/q_seiscomp/ts_latency.py > q_seiscomp.log 2>&1

            run 3 main method after load the class instance:
                from q_seiscomp import q_seiscomp

                q_seiscomp.plot_ts_latency("STATION_CODE") --> to plot time series after register ts_latency to crontab
                q_seiscomp.check_existing_configuration() --> to check and fix mismatch station configuration
                q_seiscomp.check_unexists_sts() --> to check and add unexists station on scproc observation area

            using help(method) to see more detail: example help(q_seiscomp.plot_ts_latency)

    dependecies: numpy, pandas, scipy, basemap and shapely
    """

    def __init__(self):
        self.sc_version = 4
        self.sc_schema = "0.11"

        self.current_time = dt.now(tz.utc).replace(tzinfo=None)

        self.df_local_sts = pd.DataFrame()
        self.sl_profiles = []
        self.etc_dir = join(environ['HOME'], "seiscomp", "etc")
        # self.key_dir = join(self.etc_dir, 'key')
        # self.inv_dir = join(self.etc_dir, 'inventory')
        # self.arc_dir = join(environ['HOME'], "seiscomp", 'var', 'lib', 'archive')

        # self.key_dir = join("E:/", 'PGR_Backup', 'key')
        # self.inv_dir = join("E:/", 'PGR_Backup', 'inventory')
        # self.arc_dir = join("E:/", 'PGR_Backup', 'archive')

        exe_dir = os.getcwd()
        self.key_dir = join(exe_dir, 'etc', 'key')
        # self.inv_dir = join(exe_dir, 'etc', 'inventory')
        self.inv_dir = join(exe_dir, 'inventory_fsdn')
        self.arc_dir = join(exe_dir, 'archive_sample')

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

    # def archive_PPSD(self, station=["AAI"], dt_from=None, dt_to=None):
    def archive_PPSD(self, station=None, dt_from=None, dt_to=None):
        self.get_existing_stations()
        # if station is None:
        stations_df = self.df_local_sts
        if isinstance(station, string_types):
            station = [station]
        if isinstance(station, list):
            stations_df = self.df_local_sts[self.df_local_sts['Station_code'].isin(station)]

        if dt_from is None:
            dt_from = dt.now(tz.utc).replace(tzinfo=None) - td(minutes=30)
        else:
            dt_from = pd.to_datetime(dt_from)
            # self.dt_from = pd.to_datetime(dt_from)
        if dt_to is None:
            dt_to = dt.now(tz.utc).replace(tzinfo=None)
        else:
            dt_to = pd.to_datetime(dt_to)
            # self.dt_to = pd.to_datetime(dt_to)

        # files = []
        for i, r in stations_df.iterrows():

            net = r['Network_code']
            sta = r['Station_code']
            ch = r['Channel'][0:2]
            inv_file = f'{net}.{sta}.xml'
            try:
                inv = read_inventory(os.path.join(self.inv_dir, inv_file))
            except Exception:
                print('Cannot find responses metadata(s) for station {0:s}:{1:s}:{2:s}.'.format(net, sta, ch))
                continue

            for comp in ["ZNE"]:
                datadir = os.path.join(str(dt_from.year), net, sta, f'{ch}{comp}.D')
                try:
                    names = [waveform for waveform in os.listdir(join(self.arc_dir, datadir)) if os.path.isfile(join(self.arc_dir, datadir, waveform)) and f'{dt_from.year}.{UTCDateTime(dt_from).julday:03d}' in waveform or os.path.isfile(join(self.arc_dir, datadir, waveform)) and f'{dt_to.year}.{UTCDateTime(dt_to).julday:03d}' in waveform]
                    # for name in names:
                    #     files.append(join(self.arc_dir, datadir, name))
                    #     break
                    trace = read(join(self.arc_dir, datadir, names[0]))
                    trace.trim(starttime=UTCDateTime(dt_from), endtime=UTCDateTime(dt_to))
                    # st = st.select(channel="SH*")
                    # st = st.merge()
                    # st.detrend()
                    # st.filter('bandpass', freqmin=0.2, freqmax=25)
                    if len(trace) == 0:
                        # if there is no data at specified time range, marked as blank data (-20)
                        ppsd_perc = -20
                        print('There is no data for station {:s}.{:s}.{:s}{:s} at specified time range {:s}-{:s}'.
                              format(net, sta, ch, comp, dt_from, dt_to))
                        continue
                    elif len(trace) > 1:
                        # if there is gap, take the longer trace;
                        # if longest trace duration less than 120 sec, marked as gap data (-10)
                        L_duration = 0
                        # L_trace = Trace()
                        for trc in trace:
                            duration = trc.stats.endtime - trc.stats.starttime
                            if duration > L_duration:
                                L_duration = duration
                                L_trace = trc
                        if L_duration < 120:
                            ppsd_perc = -10
                            print('The data is gap, there is not enough data for station {:s}.{:s}.{:s}{:s} '
                                  'at specified time range {:s}-{:s}'.format(net, sta, ch, comp, dt_from, dt_to))
                            continue
                        else:
                            print('The data is gap, using longest data for station {:s}.{:s}.{:s}{:s} at {:s}-{:s}'.
                                  format(net, sta, ch, comp, L_trace.stats.starttime, L_trace.stats.endtime))
                            trace = L_trace
                    ppsd_perc = self.calc_PPSDcoverage(trace[0], inv)

                except Exception:
                    print('There is no data for station {:s}.{:s}.{:s}{:s}'.format(net, sta, ch, comp))
                    break

    def calc_PPSDcoverage(self, trace, inventory, low_freq=0.01, high_freq=6, save_img=True):
        """
        Calculate the percentage of data that are between the noise model (Peterson, 1993)
        low_freq, high_freq: Lower and upper frequencies to calculate (Hz)
        """

        # calculate the PPSD
        # ppsd = PPSD(trace.stats, metadata=inventory, ppsd_length=200)
        ppsd = PPSD(trace.stats, metadata=inventory)
        ppsd.add(trace)

        if save_img:
            # ppsd.plot(cmap=pqlx)
            ppsd.plot(cmap=pqlx, show=False, period_lim=(0.05, 500), show_coverage=True, show_histogram=True,
                      filename=f"{trace.stats.network}.{trace.stats.station}.{trace.stats.channel}.jpg")

        precision = 5
        np.set_printoptions(precision=precision, suppress=True, threshold=sys.maxsize)

        # retrieve periods and amplitudes calculated by PPSD then convert to freqs
        periods = ppsd.period_bin_centers
        all_psd_values = np.array(ppsd.psd_values)  # shape (n_segments, n_periods)
        freqs = 1.0 / periods

        # Filter frequencies and PSD values to the desired range
        valid_idx = np.where((freqs >= low_freq) & (freqs <= high_freq))[0]
        filtered_freqs = freqs[valid_idx]
        filtered_psd_values = all_psd_values[:, valid_idx]

        # retrieve periods, low and high amplitudes based on noise model (Peterson, 1993) then convert to freqs
        (mperiods, mlampls), mhampls = get_nlnm(), get_nhnm()[1]
        mfreqs = 1.0 / mperiods

        # count how many freq points are between Peterson Noise Model
        inside_count = 0
        total_count = 0
        for i, freq in enumerate(filtered_freqs):
            midx = (np.abs(mfreqs - freq)).argmin()
            for psd_value in filtered_psd_values[:, i]:
                # plt.figure(figsize=(10, 6))
                # plt.hist(filtered_psd_values[:, i], bins=10, edgecolor='black', alpha=0.75)
                # plt.grid(True)
                # plt.show()
                # plt.close()
                # total_count += 1
                if mlampls[midx] <= psd_value <= mhampls[midx]:
                    inside_count += 1

        percentage = inside_count * 100 / total_count

        return percentage


# q_seiscomp = QSeisComP()
# q_seiscomp.archive_PPSD()
