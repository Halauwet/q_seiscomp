import sys
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
# import seiscomp.kernel, seiscomp.config
# from crontab import CronTab
from six import string_types
from datetime import datetime as dt
from datetime import timezone as tz
from obspy.clients.fdsn.client import Client
from os import getcwd, listdir, mkdir, environ, listdir
from os.path import join, exists


class QSeisComP:
    """
    Additional SeisComP module for monitoring and processing system
    by: eQ_Halauwet (yehezkiel.halauwet@bmkg.go.id)

    using: copy q_seiscomp dir to seiscomp/lib/python
           register ts_latency to crontab:
                crontab -e
                */5 * * * * /home/sysop/anaconda3/bin/python /home/sysop/seiscomp/lib/python/q_seiscomp/ts_latency.py > q_seiscomp.log 2>&1
    """
    def __init__(self):
        self.etc_dir = join(environ['HOME'], "seiscomp", "etc")
        self.slmon_ts_dir = join(environ['HOME'], "seiscomp", "var", "lib", "slmon_ts")
        # self.etc_dir = join(getcwd(), 'etc')
        # self.slmon_ts_dir = join(getcwd(), 'slmon_ts')
        self.key_dir = join(self.etc_dir, 'key')
        self.sl_profile_dir = join(self.key_dir, 'seedlink')

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

    def get_active_stations(self):
        """
        function to obtain active station on scproc (through read the stream_key)
        :return:
        """
        sts_key_list = [file for file in listdir(self.key_dir) if file.startswith("station")]

        # for stn_key in sts_key_list:
        #     if stn_key == "station_IA_RKPI": # "station_IA_SAUI"
        #         lc, ch, ad = self.read_stream_key(stn_key)
        data_list = []
        for key in sts_key_list:
            s, net, stn = key.split("_")
            data_list.append({
                'Stream_key': key.strip(),
                'Network_code': net.strip(),
                'Station_code': stn.strip(),
            })
        df = pd.DataFrame(data_list)

        df[['Location_code', 'Channel', 'Ignore_amp']] = df.apply(lambda x: self.read_stream_key(x['Stream_key']),
                                                                  axis=1, result_type='expand')

        self.df_local_sts = df.dropna(subset=['Location_code', 'Channel'])

        self.df_local_sts = self.df_local_sts.reset_index(drop=True)

    def get_configured_hosts(self):
        """
        function to obtain configured host, port, channel filter (through the seedlink profile key)
        :return:
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

    def run_slinktool(self):
        """
        function to fetch the response from the slinktool
        :return:
        """

        current_env = environ.copy()
        additional_paths = ['/home/eq/slinktool', '/home/sysop/seiscomp/bin/']
        current_env['PATH'] = ":".join(additional_paths + [current_env.get('PATH', '')])

        self.df_seedlink_responses = pd.DataFrame()

        for i, r in self.df_seedlink_servers.iterrows():
            # if r['Filter']:
            #     completed_process = subprocess.run(f"slinktool -Q {r['Host']}:{r['Port']} | grep {r['Filter']}",
            #                                        shell=True, env=current_env, text=True, capture_output=True)
            # else:
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
                # if r['Filter']:
                #     print(f"Error fetch seedlink from {r['Host']}:{r['Port']} grep {r['Port']}")
                # else:
                print(f"Error fetch seedlink from {r['Host']}:{r['Port']}")

        self.df_seedlink_responses = self.df_seedlink_responses.drop_duplicates(subset=['Network_code', 'Station_code',
                                                                                        'Location_code', 'Channel'])
        self.df_seedlink_responses = self.df_seedlink_responses.reset_index(drop=True)

    def read_stream_key(self, key):
        """
        function to read the stream key
        :return:
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
        function to obtain latency of the active station on the scproc
        :return:
        """
        # if self.df_local_sts.empty:
        self.get_active_stations()
        self.get_configured_hosts()
        self.run_slinktool()

        self.df_seedlink_responses[['Latency', 'Latency_secs']] = self.df_seedlink_responses.apply(
            lambda x: self.calc_latency(x['Buffer_end']), axis=1, result_type='expand')

        self.df_local_sts = self.df_local_sts.merge(self.df_seedlink_responses,
                                                    on=['Network_code', 'Station_code', 'Location_code', 'Channel'],
                                                    how='left')
        if save:
            print(self.df_local_sts[['Network_code', 'Station_code', 'Location_code', 'Channel', 'Latency']])
            self.write_TS_latency()
        else:
            print(self.df_local_sts[['Network_code', 'Station_code', 'Location_code', 'Channel', 'Latency']])

    def check_existing_configuration(self):
        """
        function to match configured stations on scproc with active stream on the seedlink
        :print: mismatch station and recommended configuration
        """
        # if self.df_local_sts.empty:
        self.get_active_stations()
        if self.df_seedlink_responses.empty:
            self.get_configured_hosts()
            self.run_slinktool()

        merged_df = self.df_local_sts.merge(self.df_seedlink_responses,
                                            on=['Network_code', 'Station_code', 'Location_code', 'Channel'],
                                            how='left')

        error_sta = merged_df[merged_df['Buffer_start'].isna()]

        self.df_seedlink_responses['Init_Channel'] = self.df_seedlink_responses['Channel'].str[:2]

        error_detail = ""
        for i, r in error_sta.iterrows():
            recomend_detail = ""
            recomend_sts = self.df_seedlink_responses[self.df_seedlink_responses['Station_code'] == r['Station_code']]
            recomend_sts = recomend_sts[~recomend_sts['Init_Channel'].duplicated()]
            for ii, rr in recomend_sts.iterrows():
                recomend_detail += (f"{rr['Network_code']}.{rr['Station_code']}.{rr['Location_code']}."
                                    f"{rr['Init_Channel']}\n                    ")
            error_detail += (f'Station "{r["Network_code"]}.{r["Station_code"]}.{r["Location_code"]}.'
                             f'{r["Channel"][:2]}" is not available in seedlink stream. \n'
                             f'Recommended stream: {recomend_detail}')

        print(error_detail)

    def calc_latency(self, end_buffer):
        """
        function to calculate the latency data
        :return: latency in datetime format and seconds unit
        """
        self.current_time = current_time = dt.now(tz.utc).replace(tzinfo=None)
        end_buffer = pd.to_datetime(end_buffer).to_pydatetime()
        return (current_time - end_buffer), (current_time - end_buffer).total_seconds()

    def write_TS_latency(self):
        """
        function to store calculated latencies to each station time series data in self.slmon_ts_dir directory
        :output: time series latency data
        """
        for i, r in self.df_local_sts.iterrows():
            sts_ts_latency = join(self.slmon_ts_dir, f'{r["Network_code"]}.{r["Station_code"]}.'
                                                     f'{r["Location_code"]}.{r["Channel"][:2]}')

            latency_data = (f'{self.current_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]}\t'
                            f'{self.current_time.timestamp()}\t'
                            f'{r["Latency"]}\t{r["Latency_secs"]}')

            if exists(sts_ts_latency):  # write finished event to finished catalog
                with open(sts_ts_latency, 'a') as f:
                    f.write(f'{latency_data}\n')
            else:
                with open(sts_ts_latency, 'w') as f:
                    f.write(f'Timestamp\tTimestamp(sec)\tLatency\tLatency(sec)\n{latency_data}\n')
        print(f'Latency data written to: {self.slmon_ts_dir}')

    def plot_ts_latency(self, station, dt_from=None, dt_to=None, group=""):
        """
        plot time series of latency data

        usages: from q_seiscomp.QSeisComP import QSeisComP
                QSeisComP.plot_ts_latency("station_name")

        :param station: station code
        :param dt_from: date from (YYYY-M-D)
        :param dt_to: date to (YYYY-M-D)
        :param group: barchart group D/M
        :return: plot of latency time series
        """
        if dt_from is None:
            dt_from = dt(1970, 1, 1, 0, 0, 0)
        else:
            dt_from = pd.to_datetime(dt_from)
        if dt_to is None:
            dt_to = dt.now(tz.utc).replace(tzinfo=None)
        else:
            dt_to = pd.to_datetime(dt_to)
        files = listdir(self.slmon_ts_dir)
        filtered_files = [filename for filename in files if filename.split(".")[1] == station]
        if filtered_files:
            data = pd.read_csv(join(self.slmon_ts_dir, filtered_files[0]), parse_dates=['Timestamp'], sep="\t")
            df = data.set_index('Timestamp')

            df = df[(df.index >= dt_from) & (df.index <= dt_to)]

            fig, ax = plt.subplots(figsize=(10, 6))
            # plt.figure(figsize=(10, 6))
            if group:
                df['Latency(sec)'].resample(group).sum().plot(kind='bar')
                ax.set_title('Daily Bar Chart')
                ax.set_xlabel('Day')
                ax.set_ylabel('Latency (s)')
                plt.show()
            else:
                td_hours = (df.index[-1] - df.index[0]).total_seconds() / 3600
                td_mins = (df.index[-1] - df.index[0]).total_seconds() / 60
                df['Latency(sec)'].plot()
                ax.set_title(f'Station {station} Latency Time Series')
                ax.set_xlabel('Time')
                # ax.plot(df.index, [time.strftime('%Y-%m-%d %H:%M:%S') for time in df.index], rotation=90)
                if td_hours/10 < 1:
                    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=int(td_mins/10)))
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))  # Format as desired
                else:
                    ax.xaxis.set_major_locator(mdates.HourLocator(interval=int(td_hours/10)))
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))  # Format as desired
                ax.set_ylabel('Latency (s)')
                plt.show()

        else:
            print(f"Station data not found on {self.slmon_ts_dir}")

    def add_station(self, full_id):
        """
        Add station to scproc
            download inventory from geof
            store inventory
            configure key
            update-config
            restart

        :param full_id: list of full station ID separated by a dot: NET.STA.LOC.CH
        """
        if isinstance(full_id, string_types):
            full_id = [full_id]
        elif isinstance(full_id, list):
            pass
        for stn in full_id:
            net, sta, loc, cha = stn.split('.')
            fdsn_bmkg = Client(base_url='https://geof.bmkg.go.id', user='pgn', password='InfoPgn!&#',
                               force_redirect=False)
            inv = fdsn_bmkg.get_stations(network=net, sta=sta, loc="*", channel=cha,
                                         level="response")


QSeisComP = QSeisComP()
QSeisComP.plot_ts_latency("KRAI")
