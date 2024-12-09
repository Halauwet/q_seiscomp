from QSeisComP import QSeisComP
q_seiscomp = QSeisComP()

q_seiscomp.archive_PPSD(qcmt_station="~/q_repo/Q_CMT/analysis/AutoQCMT/data/stations.dat",
                        low_freq=0.005, high_freq=0.2, fsdn=True, duration=20, save_quality=False)
