#!/usr/bin/env python
"""
DMLC submission script, MPI version
"""
import argparse
import sys
import os
import subprocess
import tracker
from threading import Thread

parser = argparse.ArgumentParser(description='DMLC script to submit dmlc job using MPI')
parser.add_argument('-n', '--nworker', required=True, type=int,
                    help = 'number of worker proccess to be launched')
parser.add_argument('-s', '--server-nodes', default = 0, type=int,
                    help = 'number of server nodes to be launched')
parser.add_argument('--log-level', default='INFO', type=str,
                    choices=['INFO', 'DEBUG'],
                    help = 'logging level')
parser.add_argument('--log-file', type=str,
                    help = 'output log to the specific log file')
parser.add_argument('-S', '--serverhost', type=str,
                    help = 'the hostfile of mpi server')
parser.add_argument('-W', '--workerhost', type=str,
                    help = 'the hostfile of mpi worker')
parser.add_argument('-e', '--eval', type=str,
                    help = 'the eval file')
parser.add_argument('-t', '--traindir', type=str,
                    help = 'the eval file')
parser.add_argument('command', nargs='+',
                    help = 'command for dmlc program')
parser.add_argument('--host-ip', type=str,
                    help = 'the scheduler ip', default='ip')
args, unknown = parser.parse_known_args()
#
# submission script using MPI
#

def get_mpi_env(envs):
    """get the mpirun command for setting the envornment

    support both openmpi and mpich2
    """
    outfile="/tmp/mpiver"
    os.system("mpirun -version 1>/tmp/mpiver 2>/tmp/mpiver")
    with open (outfile, "r") as infile:
        mpi_ver = infile.read()
    cmd = ''
    if 'Open MPI' in mpi_ver:
        for k, v in envs.items():
            cmd += ' -x %s=%s' % (k, str(v))
    elif 'mpich' in mpi_ver:
        for k, v in envs.items():
            cmd += ' -env %s %s' % (k, str(v))
    else:
        raise Exception('unknow mpi version %s' % (mpi_ver))

    return cmd

def mpi_submit(nworker, nserver, pass_envs):
    """
      customized submit script, that submit nslave jobs, each must contain args as parameter
      note this can be a lambda function containing additional parameters in input
      Parameters
         nworker number of slave process to start up
         nserver number of server nodes to start up
         pass_envs enviroment variables to be added to the starting programs
    """
    def run(prog):
        """"""
        subprocess.check_call(prog, shell = True)

    scmd = ''
    if args.serverhost is not None:
        scmd = '--hostfile %s' % (args.serverhost)
    scmd += ' ' + ' '.join(args.command) + ' ' + ' '.join(unknown)

    wcmd = ''
    if args.workerhost is not None:
        wcmd = '--hostfile %s' % (args.workerhost)
    wcmd += ' ' + ' '.join(args.command) + ' ' + ' '.join(unknown)

    """
        SYNC_MODE:
            1:  sync
            2:  semi-sync without vr
            3:  semi-sync with vr
    """

    # pass_envs['PS_VERBOSE'] = 1
    # pass_envs['DMLC_PS_ROOT_URI'] = '192.17.176.172'

    pass_envs['EVAL'] = 1
    pass_envs['EVAL_FILE'] = args.eval

    pass_envs['SYNC_MODE'] = 3
    # pass_envs['TRAIN_DIR'] = '/home/ubuntu/cx2/data/a9a_data/part-'
    pass_envs['TRAIN_DIR'] = args.traindir
    # pass_envs['TEST_FILE'] = '/home/cx2/ClionProjects/ps-lite-new/examples/LR_proximal/script/a9a-data/test/part-001'
    pass_envs['NSAMPLES'] = 32561
    pass_envs['NUM_FEATURE_DIM'] = 123
    pass_envs['NUM_ITERATION'] = 20

    pass_envs['SAVE_PREFIX'] = '/home/ubuntu/cx2/data/results/a9a/weight_track_'

    pass_envs['PROXIMAL'] = 'l1'
    pass_envs['LAMBDA'] = 0.01

    pass_envs['TAU'] = 1000
    pass_envs['GD_DELAY_MSG'] = 0
    pass_envs['GD_RESEND_DELAY'] = 1000

    # start servers
    if nserver > 0:
        pass_envs['LEARNING_RATE'] = 0.1
        pass_envs['DMLC_ROLE'] = 'server'
        prog = 'mpirun -n %d %s %s' % (nserver, get_mpi_env(pass_envs), scmd)
        thread = Thread(target = run, args=(prog,))
        thread.setDaemon(True)
        thread.start()

    if nworker > 0:
        pass_envs['BATCH_SIZE'] = 100
        pass_envs['DMLC_ROLE'] = 'worker'
        prog = 'mpirun -n %d %s %s' % (nworker, get_mpi_env(pass_envs), wcmd)
        thread = Thread(target = run, args=(prog,))
        thread.setDaemon(True)
        thread.start()

tracker.config_logger(args)

tracker.submit(args.nworker, args.server_nodes, fun_submit = mpi_submit,
               hostIP=args.host_ip,
               pscmd=(' '.join(args.command) + ' ' + ' '.join(unknown)))