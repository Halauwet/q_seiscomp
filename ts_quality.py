from QSeisComP import QSeisComP

q_seiscomp = QSeisComP()
PGRIX = ["AAI", "AAII", "TAMI", "KRAI", "MSAI", "NLAI", "SRMI", "NBMI", "SEMI", "BNDI", "BSMI", "SSMI",
         "TLE2", "KTMI", "KKMI", "SAUI", "ARMI", "TMTMM", "WSTMM", "NSBMM", "TTSMI", "PBMMI", "MLMMI"]
q_seiscomp.archive_PPSD(station=PGRIX, low_freq=0.05, high_freq=5)
