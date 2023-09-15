# q_seiscomp
Additional SeisComP module for optimizing the monitoring and processing system
by: eQ_Halauwet (yehezkiel.halauwet@bmkg.go.id)

usages: copy q_seiscomp dir to seiscomp/lib/python
    
        register ts_latency to crontab:
            crontab -e
            */5 * * * * python /home/sysop/seiscomp/lib/python/q_seiscomp/ts_latency.py > q_seiscomp.log 2>&1
    
        run 3 main function after load the class instance <from q_seiscomp.QSeisComP import Q_SC>
    
            Q_SC.plot_ts_latency("STATION_CODE") --> to plot time series after register ts_latency to crontab
            Q_SC.check_existing_configuration() --> to check and fix mismatch station configuration
            Q_SC.check_unexists_sts() --> to check and add unexists station on scproc observation area

dependecies: numpy, pandas, scipy, basemap and shapely
