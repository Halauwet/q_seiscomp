from QSeisComP import QSeisComP

print(
    """
    Additional SeisComP module for optimizing the monitoring and processing system
    by: eQ_Halauwet (yehezkiel.halauwet@bmkg.go.id)

    :usages: copy q_seiscomp dir to seiscomp/lib/python

            register ts_latency to crontab:
                crontab -e
                */5 * * * * python /home/sysop/seiscomp/lib/python/q_seiscomp/ts_latency.py > q_seiscompL.log 2>&1
                0 3,7,11,15,19,23 * * * python /home/sysop/seiscomp/lib/python/q_seiscomp/ts_quality.py > q_seiscompQ.log 2>&1"
                0 3,7,11,15,19,23 * * * /bin/bash -c "source /home/sysop/anaconda3/etc/profile.d/conda.sh && conda activate RMT && python /home/sysop/seiscomp/lib/python/q_seiscomp/ts_quality.py > q_seiscompQ.log 2>&1"


            run 3 main method after load the class instance:
                from q_seiscomp import *

                q_seiscomp.plot_ts_latency("STATION_CODE") --> to plot time series after register ts_latency to crontab
                q_seiscomp.plot_ts_quality("STATION_CODE") --> to plot time series after register ts_quality to crontab
                q_seiscomp.check_existing_configuration() --> to check and fix mismatch station configuration
                q_seiscomp.check_unexists_sts() --> to check and add unexists station on scproc observation area

            using help(method) to see more detail: example help(q_seiscomp.plot_ts_latency)

    """)

q_seiscomp = QSeisComP()
PGR_IX = ["AAI", "AAII", "MSAI", "NLAI", "BNDI", "SAUI", "KRAI", "TLE2", "BSMI", "TAMI", "SRMI", "NBMI", "SEMI", "SSMI", "KTMI", "KKMI","ARMI", "TMTMM", "NSBMM", "WSTMM", "TTSMI", "PBMMI", "MLMMI", "PAMI", "BDMI", "TKMN"]
PGRIX = ["AAI", "AAII", "MSAI", "NLAI", "BNDI", "SAUI", "KRAI", "TLE2", "BSMI", "TAMI", "SRMI", "NBMI", "SEMI", "SSMI", "TMTMM", "NSBMM", "WSTMM", "TTSMI", "PBMMI", "MLMMI", "PAMI", "BDMI", "TKMN"]
# help(q_seiscomp)
