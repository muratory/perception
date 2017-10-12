#!/usr/bin/env python
# Basic Bluetooth Serial server using RFCOMM protocol
# Client device must be paired in advanced with host running this script

import bluetooth
from bluetooth.btcommon import BluetoothError
import signal
import sys
import argparse
import logging
import logging.handlers
from importlib import import_module # dynamic module import
import os

logger = None
server = None

def signal_handler(signal, frame):
    print('Exiting due to Ctrl+C')
    if server:
        server.stop()
    sys.exit(0)

def get_config_path():
    return os.path.dirname(os.path.realpath(__file__))

class DriveConfigError(Exception):
    pass

class DriveConfig(object):

    CONFIG_FILE = 'config'
    config = {'offset': 0}

    def __init__(self, logger):
        self.logger = logger
        self.config_file = get_config_path() + '/' + self.CONFIG_FILE
        self.load_config()

    def load_config(self):
        # Get offset from config file
        try:
            for line in open(self.config_file):
                if line[0:8] == 'offset =':
                    self.config['offset'] = int(line[9:-1])
            self.logger.info("Successfully parsed configuration file %s" % self.config_file)
        except Exception as err:
            self.logger.error("Failed to open configuration file %s [%s]" % (self.config_file, str(err)))
            raise DriveConfigError("Failed to open configuration file")

    def get(self, param):
        if param in self.config:
            return self.config[param]
        else:
            raise DriveConfigError("Parameter %s not found in configuration" % param)

class DriveServerError(Exception):
    pass

class DriveServer(object):

    drive_uuid = "1e0ca4ea-299d-4335-93eb-deeddeeddeed" # deed for DEEp Drive

    def __init__(self, logger, test_bluetooth=False, drive_request_mod=None):
        self.test_only = test_bluetooth
        self.client_sock = None
        self.server_sock = None
        self.logger = logger
        self.client_address = None
        self.config = DriveConfig(logger)
        if not test_bluetooth and drive_request_mod is None:
            self.fail("Missing module to execute drive requests")
        if not test_bluetooth:
            self.drive_request_mod = drive_request_mod
            setup = getattr(self.drive_request_mod, 'drive_request_setup')
            setup()

    def stop(self):
        if self.client_sock:
            self.client_sock.close()
        if self.server_sock:
            self.server_sock.close()

    def output(self, msg):
        print(msg)
        self.logger.info(msg)

    def output_error(self, err):
        print(err)
        self.logger.error(err)

    def process_request(self, request):
        if self.test_only:
            if 'details' in request:
                reply = "Deep Drive Bluetooth Test Server"
            else:
                reply = "OK\n"
            self.output("<%s" % request)
            self.client_sock.send(reply + '\n')
            self.output(">%s" % reply)
            print "..."
        else:
            self.logger.debug("Execute request '%s'" % request)
            DriveRequest = getattr(self.drive_request_mod, 'DriveRequest')
            DriveRequestError = getattr(self.drive_request_mod, 'DriveRequestError')
            if DriveRequest.supported(request):
                try:
                    cmd = DriveRequest(request)
                    out = cmd.execute()
                    self.client_sock.send(out)
                except DriveRequestError as err:
                    self.output_error("Failed to process request '%s' [%s]" % (request, str(err)))
            else:
                self.output_error("Unsupported request %s" % request)
                self.client_sock.send('NOK\n')

    def fail(self, err):
        self.stop()
        raise DriveServerError(err)

    def run(self):
        self.logger.info('Starting Bluetooth deep-drive server (press Ctrl+C to stop)...')
        if self.test_only:
            self.logger.info('Test mode enabled')

        try:
            self.server_sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
            self.server_sock.bind(("", bluetooth.PORT_ANY))
            self.server_sock.listen(1) # argument stands for max number of queued connections
            port = self.server_sock.getsockname()[1]
            # Advertising service UUID would require this change
            # https://www.raspberrypi.org/forums/viewtopic.php?t=132470&p=883017
            #bluetooth.advertise_service(self.server_sock,
            #                            "Deep Drive Service",
            #                            self.drive_uuid)
        except BluetoothError as err:
            self.output_error('Failed to start bluetooth server ' + str(err))
            self.fail(err)

        self.logger.info("Listening on port %d" % port)
        self.client_sock, self.client_address = self.server_sock.accept()
        self.output("Accepted connection from %s" % str(self.client_address))

        try:
            while True:
                data = self.client_sock.recv(1024)
                if len(data) != 0:
                    for request in data.splitlines():
                        self.process_request(request)
        except BluetoothError as err:
            if '104' not in str(err):
                self.output_error('Bluetooth client has disconnected ' + str(err) + ', exiting...')
                self.fail(err)
            else:
                self.output('Disconnected' + str(err))

if __name__ == '__main__':

    argparser = argparse.ArgumentParser(description='Deep Drive embedded Bluetooth server')
    argparser.add_argument('-t', '--test', dest='test_bluetooth', action='store_true',
                            help='test Bluetooth communication only (do not use car motors)')
    argparser.add_argument('-v', '--verbose', dest="verbose", action="store_true",
                            help='enable verbose mode')
    args = argparser.parse_args()

    # Logging configuration
    logger = logging.getLogger('DeepDrive')
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(name)s: [%(levelname)s] %(message)s')

    # Stdout logging handler
    stdoutHandler = logging.StreamHandler(sys.stdout)
    stdoutHandler.setFormatter(formatter)
    logger.addHandler(stdoutHandler)

    # Syslog logging handler
    syslogHandler = logging.handlers.SysLogHandler(address = '/dev/log')
    syslogHandler.setFormatter(formatter)
    syslogHandler.setLevel(logging.INFO) # avoid verbose logging on filesystem
    logger.addHandler(syslogHandler)

    signal.signal(signal.SIGINT, signal_handler)

    if not args.test_bluetooth:
        drive_request_mod = import_module('drive_request')
    else:
        drive_request_mod = None

    attempts = 10
    while attempts > 0:
        try:
            server = DriveServer(logger, args.test_bluetooth, drive_request_mod)
            server.run()
        except Exception as err:
            logger.error("Server failed with error " + str(err))
            attempts = attempts - 1
        finally:
            logger.info("Restarting deep drive server (attempts remaining : %d)" % attempts)
