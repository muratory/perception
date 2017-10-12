#!/usr/bin/env python
# Basic Bluetooth Serial server using RFCOMM protocol
# Client device must be paired in advanced with host running this script

import bluetooth
import signal
import sys

client_sock = None
server_sock = None

def signal_handler(signal, frame):

    print('Exiting due to Ctrl+C')
    bluetooth_stop()
    sys.exit(0)

def bluetooth_stop():
    if client_sock:
        client_sock.close()
    if server_sock:
        server_sock.close()

def run_bluetooth_echo_server():

    print('Starting Bluetooth deep-drive test server (press Ctrl+C to stop)...')

    server_sock=bluetooth.BluetoothSocket( bluetooth.RFCOMM )
    port = 1
    server_sock.bind(("", port))
    server_sock.listen(1)

    client_sock, address = server_sock.accept()
    print "Accepted connection from ", address

    try:
        while True:

            print "..."
            data = client_sock.recv(1024)
            if len(data) != 0:
                for reply in data.splitlines():
                    print "<%s" % reply
                    client_sock.send("OK\n")
                    print ">OK"

    except bluetooth.btcommon.BluetoothError as err:
        print('Bluetooth client has disconnected ' + str(err) + ', exiting...')
        bluetooth_stop()
        sys.exit(0)

if __name__ == '__main__':

    signal.signal(signal.SIGINT, signal_handler)
    run_bluetooth_echo_server()
