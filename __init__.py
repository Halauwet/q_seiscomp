from QSeisComP import QSeisComP

print(
    """
    Additional SeisComP module for optimizing the monitoring and processing system
    by: eQ_Halauwet (yehezkiel.halauwet@bmkg.go.id)

    :usages: copy q_seiscomp dir to seiscomp/lib/python

            register ts_latency to crontab:
                crontab -e
                */5 * * * * python /home/sysop/seiscomp/lib/python/q_seiscomp/ts_latency.py > q_seiscomp.log 2>&1

            run 3 main method after load the class instance:

                q_seiscomp.plot_ts_latency("STATION_CODE") --> to plot time series after register ts_latency to crontab
                q_seiscomp.check_existing_configuration() --> to check and fix mismatch station configuration
                q_seiscomp.check_unexists_sts() --> to check and add unexists station on scproc observation area

            using help(method) to see more detail. Example: help(q_seiscomp.plot_ts_latency)

    """)

q_seiscomp = QSeisComP()
Q_SC = q_seiscomp
PGRIX = ["AAI", "AAII", "TAMI", "KRAI", "MSAI", "NLAI", "SRMI", "NBMI", "SEMI", "BNDI", "BSMI", "SSMI",
         "TLE2", "KTMI", "KKMI", "SAUI", "ARMI", "TMTMM", "WSTMM", "NSBMM", "TTSMI", "PBMMI", "MLMMI"]
# help(q_seiscomp)
