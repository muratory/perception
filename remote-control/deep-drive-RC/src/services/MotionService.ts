import { DeviceMotion, DeviceMotionAccelerationData } from 'ionic-native';

import { Observable } from 'rxjs/Observable';
import { Injectable } from '@angular/core';
import { Subscription } from 'rxjs/Subscription';

import { ConfigService } from './ConfigService';

export class Acceleration {

    public xg: number;
    public yg: number;
    public zg: number;

    public G: number = 9.81;

    constructor(private x: number, private y: number, private z: number) {
      this.xg = x / this.G;
      this.yg = y / this.G;
      this.zg = z / this.G;
    }

}

export class Motion {

    public LRAngle: number;
    public FBAngle: number;
    public forward: boolean;
    public backward: boolean;
    public left: boolean;
    public right: boolean;

    constructor(public acc: Acceleration, public timestamp: number) {
        this.LRAngle = 0;
        this.FBAngle = 0;
        this.left = false;
        this.right = false;
        this.forward = false;
        this.backward = false;
    }

    getAccel() {
        return this.acc;
    }

    getTimestamp() {
        return this.timestamp;
    }

    setLeftRight(direction, angle) {
        if (direction == "right") {
            this.right = true;
            this.left = false;
        }
        else if (direction == "left") {
            this.left = true;
            this.right = false;
        }
        else {
            this.right = false;
            this.left = false;
        }
        this.LRAngle = angle;
    }

    setForwardBackward(direction, angle) {
        if (direction == "forward") {
            this.forward = true;
            this.backward = false;
        }
        else if (direction == "backward") {
            this.backward = true;
            this.forward = false;
        }
        else {
            this.backward = false;
            this.forward = false;
        }
        this.FBAngle = angle;
    }

}

@Injectable()
export class MotionService {

    private lastMotion: Motion;
    public motionSubscription: any;
    /* Supported values : portrait-primary, landscape-primary or landscape-secondary */
    private screenOrientation: string;

    private rcOrientationSubscription: Subscription;
    private stopAngleFB: number = 0;

    /* observable for parent class calling the start() method */
    public motionObserver: any;
    public motion: Observable<Motion>;

    private MOTION_PERIOD_MS: number = 100;

    private FLAT_MAX_ANGLE: number = 5; // degrees, consider that below this angle, phone is flat
    private LEFT_RIGHT_MIN_DEG: number = 5; // degrees, below this angle, no left/right direction
    private FORW_BACK_MIN_DEG: number = 5; // degrees, below this angle, no forward/backward direction

    constructor(private config: ConfigService) {
      console.log("MOTION: new instance");
      let initAccel: Acceleration = new Acceleration(0, 0, 0);
      let initTimestamp: number = 0;
      this.lastMotion = new Motion(initAccel, initTimestamp);
      this.motionSubscription = null;
      this.motionObserver = null;
      this.rcOrientationSubscription = this.config.getRCOrientation().subscribe((value) => {
          console.log("MOTION: new rcOrientation detected " + value);
          if (value == 'fortyFiveDegStop') {
              /* Device is hold 45° from flat position, screen facing user */
              this.stopAngleFB = 45;
          }
          else {
              /* 'flatStop' and default values */
              this.stopAngleFB = 0;
          }
      });
      this.motion = new Observable(observer => {
            this.motionObserver = observer;
        });
    }

    start(screenOrientation: string) {
        this.motionSubscription = DeviceMotion.watchAcceleration(
                {frequency: this.MOTION_PERIOD_MS}
            ).subscribe((devAcc: DeviceMotionAccelerationData) => {
                /* In fat arrow function, 'this' will stand for MotionService */
                let newAccel = new Acceleration(devAcc.x, devAcc.y, devAcc.z);
                if (this.lastMotion.getTimestamp() == 0) {
                  /* It is too early to compute a motion, since we only have one sample */
                  this.lastMotion = new Motion(newAccel, devAcc.timestamp);
                }
                else {
                  let deltaMs: number = devAcc.timestamp - this.lastMotion.getTimestamp();
                  if (deltaMs > this.MOTION_PERIOD_MS) {
                      this.lastMotion = this.computeMotion(this.lastMotion, newAccel, devAcc.timestamp);
                      console.log("MOTION: " + JSON.stringify(this.lastMotion) + " since " + deltaMs + " ms");
                  }
                }
                this.screenOrientation = screenOrientation;
                /* update observable so that subscribers get notified */
                this.motionObserver.next(this.lastMotion);
            });
    }

    /*
     * Compute new motion
     * forward: y << 0
     * backward: y >> 0
     * left: x >> 0
     * right: x << 0
     */
    computeMotion(oldMotion, newAcc, timestamp) {
        let newMotion = new Motion(newAcc, timestamp);

        let xg: number = (newAcc.xg + oldMotion.getAccel().xg) / 2;
        let yg: number = (newAcc.yg + oldMotion.getAccel().yg) / 2;
        let zg: number = (newAcc.zg + oldMotion.getAccel().zg) / 2;
        let norm: number = Math.sqrt(xg*xg + yg*yg + zg*zg);
        let nXg = xg / norm;
        let nYg = yg / norm;
        let nZg = zg / norm;
        let inclinationDeg: number = Math.round((Math.acos(nZg))/(Math.PI / 180));
        let rotationFBDeg: number = 0;
        let rotationLRDeg: number = 0;
        let LRDirection: string = "none";
        let FBDirection: string = "none";
        /* get angles on X and Y axis against Z (between 0 and 180 degrees)
         * Note: since z is positive, inverting the angles, so that right/forward are positive angles
         */
        if (this.screenOrientation == "portrait-primary") {
            rotationFBDeg = -Math.round((Math.atan2(nYg, nZg)/(Math.PI / 180)));
        }
        else if (this.screenOrientation == "landscape-primary") {
            rotationFBDeg = -Math.round(Math.atan2(nXg, nZg)/(Math.PI / 180));
        }
        else /* landscape-secondary */ {
            rotationFBDeg = Math.round(Math.atan2(nXg, nZg)/(Math.PI / 180));
        }
        if (this.stopAngleFB == 0) {
            if (rotationFBDeg < -90) {
                rotationFBDeg = -90;
            }
            if (rotationFBDeg > 90) {
                rotationFBDeg = 90;
            }
            if (Math.abs(rotationFBDeg) >= this.FORW_BACK_MIN_DEG) {
              if (rotationFBDeg >= 0) {
                FBDirection = "forward";
              }
              else {
                FBDirection = "backward";
              }
            }
        }
        else if (this.stopAngleFB == 45) {
            if (rotationFBDeg < -90) {
                rotationFBDeg = -90;
            }
            if (rotationFBDeg > 0) {
                rotationFBDeg = 0;
            }
            if (Math.abs(rotationFBDeg) >= (this.stopAngleFB + this.FORW_BACK_MIN_DEG)) {
                FBDirection = "backward";
            }
            else if (Math.abs(rotationFBDeg) <= (this.stopAngleFB - this.FORW_BACK_MIN_DEG)) {
                FBDirection = "forward";
            }
            rotationFBDeg = rotationFBDeg + this.stopAngleFB;
        }
        newMotion.setForwardBackward(FBDirection, rotationFBDeg);

        if (this.screenOrientation == "portrait-primary") {
            rotationLRDeg = -Math.round(Math.atan2(nXg, nZg)/(Math.PI / 180));
        }
        else if (this.screenOrientation == "landscape-primary") {
            rotationLRDeg = Math.round(Math.atan2(nYg, nZg)/(Math.PI/180));
        }
        else /* landscape-secondary */ {
            rotationLRDeg = -Math.round(Math.atan2(nYg, nZg)/(Math.PI/180));
        }
        if (rotationLRDeg > -90 && rotationLRDeg < 90) {
          if (Math.abs(rotationLRDeg) >= this.LEFT_RIGHT_MIN_DEG) {
            if (rotationLRDeg > 0) {
              LRDirection = "right";
            }
            else {
              LRDirection = "left";
            }
          }
        }
        newMotion.setLeftRight(LRDirection, rotationLRDeg);

        return newMotion;
    }

    getMotion() {
        return this.lastMotion;
    }

    stop() {
        if (this.motionSubscription) {
          this.motionSubscription.unsubscribe();
          this.lastMotion = new Motion(new Acceleration(0, 0, 0), 0);
        }
    }
}
