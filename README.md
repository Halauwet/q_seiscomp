# q_seiscomp
Additional SeisComP module for optimizing the monitoring and processing system
by: eQ_Halauwet (yehezkiel.halauwet@bmkg.go.id)

usages: copy q_seiscomp dir to seiscomp/lib/python
    
        register ts_latency to crontab:
            crontab -e
            */5 * * * * python /home/sysop/seiscomp/lib/python/q_seiscomp/ts_latency.py > q_seiscomp.log 2>&1
    
run 3 main function after load the class instance <from q_seiscomp import q_seiscomp>

q_seiscomp.plot_ts_latency("STATION_CODE") --> to plot time series after register ts_latency to crontab
![latency_plot_single_sta](images/TS_1.png?raw=true "Title")
![latency_plot_multi_sts](images/TS_2.png?raw=true "Title")
            
q_seiscomp.check_existing_configuration() --> to check and fix mismatch station configuration

q_seiscomp.check_unexists_sts() --> to check and add unexists station on scproc observation area

[Searching area]
![latency_plot_multi_sts](images/Set_searching_area.png?raw=true "Title")


[Detected new stations]
![latency_plot_single_sta](images/Detected_new_sts.png?raw=true "Title")


[Interactively ask to add each station]
![latency_plot_multi_sts](images/Interactive_ask_to_be_added.png?raw=true "Title")


dependecies: numpy, pandas, scipy, basemap and shapely
