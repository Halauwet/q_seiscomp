# q_seiscomp
Additional SeisComP module for optimizing the monitoring and processing system
by: eQ_Halauwet (yehezkiel.halauwet@bmkg.go.id)

usages: copy q_seiscomp dir to seiscomp/lib/python
    
        register ts_latency to crontab:
            crontab -e
            */5 * * * * python /home/sysop/seiscomp/lib/python/q_seiscomp/ts_latency.py > q_seiscompL.log 2>&1
            0 0,4,8,12,16,20 * * * python /home/sysop/seiscomp/lib/python/q_seiscomp/ts_quality.py > q_seiscompQ.log 2>&1"

Demo video: https://youtu.be/FVi7mRas8M8. 
    
run 3 main function after load the class instance <from q_seiscomp import q_seiscomp>

**q_seiscomp.plot_ts_latency("STATION_CODE")** --> to plot latency time series after register ts_latency to crontab
![latency_plot_multi_sts](images/TS_2.png?raw=true "latency_plot_multi_sts")
![latency_plot_single_sta](images/TS_1.png?raw=true "latency_plot_single_sta")

**q_seiscomp.plot_ts_quality("STATION_CODE")** --> to plot PPSD coverage time series after register ts_quality to crontab. The PPSD coverage are calculated on specified freq range set on ts_quality.py (default 0.05-5Hz)
![quality_plot_multi_sts](images/QTS_2.png?raw=true "quality_plot_multi_sts")
![quality_plot_single_sta](images/QTS_1.png?raw=true "quality_plot_single_sta")
![psd](images/IA.KRAI.SHZ.jpg?raw=true "psd")
![psd](images/IA.KRAI.SHN.jpg?raw=true "psd") 
![psd](images/IA.KRAI.SHE.jpg?raw=true "psd")

**q_seiscomp.check_existing_configuration()** --> to check and fix mismatch station configuration

**q_seiscomp.check_unexists_sts()** --> to check and add unexists station on scproc observation area

[Searching area]
![searching_area](images/Set_searching_area.png?raw=true "searching_area")


[Detected new stations]
![detected_sta](images/Detected_new_sts.png?raw=true "detected_sta")


[Interactively ask to add each station]
![interactive](images/Interactive_ask_to_be_added.png?raw=true "interactive")


dependecies: obspy, numpy, pandas, scipy, basemap and shapely
