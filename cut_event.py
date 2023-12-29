import glob
from obspy import read, UTCDateTime, Stream, Trace
from os import listdir, mkdir
from os.path import join, isdir, isfile, exists


def add_SC3arc2(t_start, t_end, filenames=None):
    """
    Reads data from seiscomp archive file. Can read 6 continuous mseed files (3 components, from 2 respective days)
    for overlapping waveform on a station to produce three component stream.
    Append the stream to ``self.data_raw``.
    If its sampling is not contained in ``self.data_deltas``, add it there.
    """
    if len(filenames) % 3 == 0:
        st = Stream()
        for f in filenames:
            st1 = read(f)
            st += st1
        st.trim(t_start, t_end)
        # st1 = read(filenames[0]); st2 = read(filenames[1]); st3 = read(filenames[2])
        # st4 = read(filenames[3]); st5 = read(filenames[4]); st6 = read(filenames[5])

        # st1 += st2
        # st1.trim(t_start, t_end)

        # st3 += st4
        # st3.trim(t_start, t_end)

        # st5 += st6
        # st5.trim(t_start, t_end)
        # st = Stream(traces=[st1[0], st3[0], st5[0]])
        # st1 += st3
        # st1 += st5
    else:
        raise ValueError('Read six files (Z, N, E components from several days))')

    return st


def add_SC3arc(t_start, t_end, filename, filename2=None, filename3=None):
    """
    Reads data from seiscomp archive file. Can read either one mseed continuous file,
    or three mseed continuous files simultaneously to produce three component stream.
    Append the stream to ``self.data_raw``.
    If its sampling is not contained in ``self.data_deltas``, add it there.
    """
    if filename3:
        st1 = read(filename, starttime=t_start, endtime=t_end)
        st2 = read(filename2, starttime=t_start, endtime=t_end)
        st3 = read(filename3, starttime=t_start, endtime=t_end)
        st1 += st2
        st1 += st3
        # st = Stream(traces=[st1[0], st2[0], st3[0]])
    else:
        raise ValueError('Must read three files (Z, N, E components)')
    return st1


archive_dir = "/home/sysop/seiscomp/var/lib/archive"
# archive_dir = "archive_sample"

out_dir = "Mainshock"
if not exists(out_dir):
    mkdir(out_dir)

event_ot = UTCDateTime(2023, 11, 8, 4, 52, 0)

t_before = 1200
t_after = 1800
t0 = event_ot - t_before
# t1 = event_ot + t_after
t1 = UTCDateTime(2023, 11, 10, 6, 52, 0)

recording_net = ['IA']
recording_sts = ["PBMMI", "BNDI", "KTMI", "SAUI", "MLMMI", "WSTMM", "TMTMM", "AAI", "AAII", "KKMI", "MSAI", "TLE2",
                 "TAMI", "KRAI", "BSMI", "SEMI", "SRMI", "NSMBMM", "TTSMI", "NBMI", "NLAI", "FAKI", "ARMI", "KMPI",
                 "ALTI", "BATPI", "TSPI", "AAFM", "OBMI", "ALKI", "ATNI", "FKSPI", "SIJI", "SWI", "SANI", "AMPM",
                 "MBPI", "GARPI", "IEPI", "PAFM", "RAPI", "LBMI", "WECI", "LMNI", "KTTI", "ATKTI", "SOEI", "NBPI",
                 "RKPI", "ASTTI", "WLFM", "SWPM", "MTN", "IBTI", "MWPI", "WBMI", "LRTI", "ERPI", "PMMI", "BBSI",
                 "BATI", "WFTFM", "MIPI", "TNTI", "KDI", "WHMI", "PKCI", "JHMI", "BBLSI", "MMRI", "RONI", "UKCM",
                 "LKUCM", "IHMI", "WEFM", "BDMUI", "BGCI", "KKSI", "BHCM", "GLMI", "EDFI", "BTCM", "TBMCM", "LUWI",
                 "WKCM", "PBCI", "RKCM", "SBNI", "RNFM", "KMDI", "TMSI", "BNTI", "WMCM", "TOCM", "PBMSI", "PSJCM",
                 "MNI", "BSSI", "LKCI", "BUBSI", "BBBCM", "MTCM", "BUMSI", "LMTI", "TBCM", "TGCM", "SRSI", "BKSI",
                 "KMBFM", "PWCM", "TLCM", "APSI", "BBCM", "KGCM", "BNSI", "UTUSI", "GTOI", "BASI", "WSI", "LBFI",
                 "KAPI", "WUPCM", "OMBFM", "MSCM", "POCI", "MRSI", "SMSI", "TTSI", "MGAI", "WBSI", "MMPI", "LBNFM",
                 "WBNI", "DBNFM", "PBNI", "TSNI", "ESNI", "DAV", "WRAB", "COEN", "CTAO"]
channels = ["SH", "BH"]

for net in recording_net:
    if not recording_sts:
        recording_sts = [d for d in listdir(join(archive_dir, str(event_ot.year), net))
                         if isdir(join(archive_dir, str(event_ot.year), net, d))]
    for sta in recording_sts:
        files = []
        for comp in "ZNE":
            datadir = []
            datadirs = []
            if t0.year == t1.year:
                for ch in channels:
                    if exists(join(archive_dir, str(event_ot.year), net, sta, f"{ch}{comp}.D")):
                        datadir = join(archive_dir, str(event_ot.year), net, sta, f"{ch}{comp}.D")
                        break
                if not datadir:
                    continue
                try:
                    # names = [waveform for waveform in listdir(join(datadir))
                    #          if isfile(join(datadir, waveform))
                    #          and f'{t0.year}.{t0.julday:03d}' in waveform or
                    #          isfile(join(datadir, waveform))
                    #          and f'{t1.year}.{t1.julday:03d}' in waveform]
                    part_of_filenames = [f'{t1.year}.{i:03d}' for i in range(t0.julday, t1.julday+1)]

                    names = [waveform for waveform in listdir(datadir)
                             if isfile(join(datadir, waveform))
                             and any(part in waveform for part in part_of_filenames)]

                    for name in names:
                        files.append(join(datadir, name))
                        # break
                except Exception:
                    # print('There is no data for station {:s}.{:s}.{:s}{:s}'.format(net, sta, ch, comp))
                    break
            else:
                for ch in channels:
                    if exists(join(archive_dir, str(t0.year), net, sta, f"{ch}{comp}.D")):
                        datadirs = [join(archive_dir, str(t0.year), net, sta, f"{ch}{comp}.D"),
                                    join(archive_dir, str(t1.year), net, sta, f"{ch}{comp}.D")]
                        break
                if not datadirs:
                    continue
                for datadir in datadirs:
                    try:
                        names = [waveform for waveform in listdir(datadir)
                                 if isfile(join(datadir, waveform))
                                 and f'{t0.year}.{t0.julday:03d}' in waveform or
                                 isfile(join(datadir, waveform))
                                 and f'{t1.year}.{t1.julday:03d}' in waveform]
                        # todo: fix get list of file based on the julday range
                        for name in names:
                            files.append(join(datadir, name))
                            # break
                    except Exception:
                        # print('There is no data for station {:s}.{:s}.{:s}{:s}'.format(net, sta, ch, comp))
                        pass
        files = sorted(files, key=lambda x: x.split('/')[-1])
        if len(files) == 0:
            datadir = join(archive_dir, str(event_ot.year), net, sta, f"{ch}{comp}.D")
            datadirs = [join(archive_dir, str(t0.year), net, sta, f"{ch}{comp}.D"),
                        join(archive_dir, str(t1.year), net, sta, f"{ch}{comp}.D")]
            print('Cannot find data file(s) for station {0:s}:{1:s}:{2:s}.'.format(net, sta, ch))
            if t0.year == t1.year:
                print(f'\tExpected file location: {datadir}')
            else:
                print(f'\tExpected file location: {datadirs[0]}\n'
                      f'\t\t\t\t\t\t\t\t\t\tand\t\t{datadirs[1]}')
            continue
        if len(files) == 3:
            try:
                st = add_SC3arc(t0, t1, files[0], files[1], files[2])
                st.write(join(out_dir, f'{sta}.mseed'))
            except Exception:
                print('Error read archive data(s) for station {0:s}:{1:s}:{2:s}.'.format(net, sta, ch))
                continue
        elif len(files) % 3 == 0:
            try:
                st = add_SC3arc2(t0, t1, files)
                st.write(join(out_dir, f'{sta}.mseed'))
            except Exception:
                print('Error read archive data(s) for station {0:s}:{1:s}:{2:s}.'.format(net, sta, ch))
                continue
        else:
            datadir = join(archive_dir, str(event_ot.year), net, sta, f"{ch}{comp}.D")
            datadirs = [join(archive_dir, str(t0.year), net, sta, f"{ch}{comp}.D"),
                        join(archive_dir, str(t1.year), net, sta, f"{ch}{comp}.D")]
            print('Problem in each component file(s) for station {0:s}:{1:s}:{2:s}.'.format(net, sta, ch))
            if t0.year == t1.year:
                print(f'\tFile location: {datadir}')
            else:
                print(f'\tFile location: {datadirs[0]}\n'
                      f'\t\t\t\t\t\t\t\t\t\tand\t\t{datadirs[1]}')
            continue
