#!/usb/bin/env python
# Drive request processing, to be called from a drive server

import car_dir_deep_drive
import motor

def drive_request_setup():
    car_dir_deep_drive.setup()
    motor.setup()     # Initialize the Raspberry Pi GPIO connected to the DC motor.
    car_dir_deep_drive.home()

class DriveRequestError(Exception):
    pass

class DriveRequest(object):

    #COMMANDS = ['forward', 'backward', 'left', 'right', 'stop', 'read cpu_temp', 'home', 'distance', 'x+', 'x-', 'y+', 'y-', 'xy_home']
    COMMANDS = {'forward':'forward',
                'backward':'backward',
                'left':'left',
                'right':'right',
                'turn':'turn',
                'stop':'stop',
                'speed':'speed',
                'details':'details',
                'home':'home',
                'RIGHT_TURN':'nn_turn_right',
                'LEFT_TURN':'nn_turn_left',
                'STRAIGHT':'nn_straight',
                'IDLE':'nn_idle'
                }

    ACK="\n"

    def __init__(self, request):
        if not self.supported(request):
            raise DriveRequestError("Unsupported command %s" % self.request)
        (req, args) = self.splitreq(request)
        self.full = request
        self.request = req
        if len(args):
            self.arg = args[0]
        else:
            self.arg = None

    @classmethod
    def splitreq(cls, request):
        req_arg = request.split('=')
        return (req_arg[0], req_arg[1:])

    @classmethod
    def supported(cls, request):
        (req, args) = cls.splitreq(request)
        return req in cls.COMMANDS

    def execute(self):
        function = getattr(self, self.COMMANDS[self.request])
        if self.arg:
            return function(self.arg)
        else:
            return function()

    def ack(self):
        return self.ACK

    def details(self):
        return 'Deep Drive Car'

    def forward(self, speed = None):
        if speed is None:
            motor.forward()
        else:
            motor.forwardWithSpeed(int(speed))
        return self.ack()

    def backward(self, speed = None):
        if speed is None:
            motor.backward()
        else:
            motor.backwardWithSpeed(int(speed))
        return self.ack()

    def turn(self, angle):
        car_dir_deep_drive.turn(int(angle))
        return self.ack()

    def left(self, angle = None):
        car_dir_deep_drive.turn_left()
        return self.ack()

    def right(self, angle = None):
        car_dir_deep_drive.turn_right()
        return self.ack()

    def stop(self):
        motor.ctrl(0)
        return self.ack()

    def speed(self, value):
        motor.setSpeed(int(value))
        return self.ack()

    def home(self):
        car_dir_deep_drive.home()
        return self.ack()

    def nn_straight(self):
        return self.ack()

    def nn_turn_left(self):
        return self.ack()

    def nn_turn_right(self):
        return self.ack()

    def nn_idle(self):
        return self.ack()


        # elif data == ctrl_cmd[5]:
        #     print 'read cpu temp...'
        #     temp = cpu_temp.read()
        #     tcpCliSock.send('[%s] %0.2f' % (ctime(), temp))
        # elif data == ctrl_cmd[8]:
        #     print 'recv x+ cmd'
        #     video_dir.move_increase_x()
        # elif data == ctrl_cmd[9]:
        #     print 'recv x- cmd'
        #     video_dir.move_decrease_x()
        # elif data == ctrl_cmd[10]:
        #     print 'recv y+ cmd'
        #     video_dir.move_increase_y()
        # elif data == ctrl_cmd[11]:
        #     print 'recv y- cmd'
        #     video_dir.move_decrease_y()
        # elif data == ctrl_cmd[12]:
        #     print 'home_x_y'
        #     video_dir.home_x_y()
        # elif data[0:7] == 'offset+':
        #     offset = offset + int(data[7:])
        #     print 'Turning offset', offset
        #     car_dir_deep_drive.calibrate(offset)
        #
        # elif data[0:7] == 'offset-':
        #     offset = offset - int(data[7:])
        #     print 'Turning offset', offset
        #     car_dir_deep_drive.calibrate(offset)
