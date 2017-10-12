import { Injectable } from '@angular/core';

import { Subscription } from 'rxjs/Subscription';

import { Motion } from './MotionService';
import { ConfigService } from './ConfigService';

export abstract class DriveCommand {

    constructor() {

    }

    abstract getCommand():string;

    convertAngleToParam(angle: number, min_in: number, max_in: number,
                            min_out: number, max_out: number): number {
        /* param = offset + slope * abs(angle) */
        angle = Math.abs(angle);
        if (angle > max_in) angle = max_in;
        else if (angle < min_in) angle = min_in;
        let slope: number = (max_out - min_out) / (max_in - min_in);
        let offset: number = min_out - min_in * slope;
        return Math.round(offset + slope * angle);
    }
}

class CmdStop extends DriveCommand {

    constructor() {
        super();
    }

    getCommand(): string {
        return "stop";
    }

}

class CmdHome extends DriveCommand {

    constructor() {
        super();
    }

    getCommand(): string {
        return "home";
    }
}

class CmdForwardBackward extends DriveCommand {

    private MAX_ANGLE_IN: number = 45;
    private MIN_ANGLE_IN: number = 5;

    private speed: number = 0;

    constructor(private isForward: boolean, angle: number,
                private speedRange: any) {
          super();
          this.angleToSpeed(angle);
    }

    angleToSpeed(angle: number) {
        this.speed = super.convertAngleToParam(angle,
                        this.MIN_ANGLE_IN, this.MAX_ANGLE_IN,
                        this.speedRange.lower, this.speedRange.upper);
    }

    getCommand(): string {
        return (this.isForward?'forward=':'backward=') + String(this.speed);
    }

}

class CmdLeftRight extends DriveCommand {

    private MAX_ANGLE_OUT: number = 50;
    private MIN_ANGLE_OUT: number = -50;
    private MAX_ANGLE_IN: number = 45;
    private MIN_ANGLE_IN: number = -45;

    private angleOut: number = 0;

    constructor(private isRight: boolean, angle: number) {
        super();
        this.angleScale(angle);
    }

    angleScale(angle: number) {
        this.angleOut = super.convertAngleToParam(angle, this.MIN_ANGLE_IN,
                this.MAX_ANGLE_IN, this.MIN_ANGLE_OUT, this.MAX_ANGLE_OUT);
    }

    getCommand(): string {
        if (this.isRight) {
            return "turn=" + String(this.angleOut);
        }
        else {
            return "turn=" + String(-this.angleOut);
        }
    }

}

class NeuralNetworkCommand extends DriveCommand {

    private MAPPING = {'left':'LEFT_TURN',
                      'right':'RIGHT_TURN',
                      'straight':'STRAIGHT',
                      'idle':'IDLE'}

    constructor(private nnSelected: string) {
        super();
        if (!this.MAPPING.hasOwnProperty(nnSelected)) {
            this.nnSelected = 'idle';
        }
    }

    getCommand(): string {
        return this.MAPPING[this.nnSelected];
    }

}

@Injectable()
export class DriveControlService {

    private hasStopped: boolean = false;
    private speedRange: any;
    private speedRangeSubscription: Subscription;

    constructor(private config: ConfigService) {
        console.log("DRIVECTRL: new instance");
        this.speedRangeSubscription = this.config.getSpeedRange().subscribe((range) => {
            console.log("DRIVECTRL: new speed range detected " + JSON.stringify(range));
            this.speedRange = range;
        });
    }

    getInitCommands(): Array<DriveCommand> {
        let initCmds: Array<DriveCommand> = [];
        initCmds.push(new CmdLeftRight(true, 0));
        initCmds.push(new CmdStop());
        initCmds.push(new CmdHome());
        return initCmds;
    }

    getNeuralNetworkCommands(nnSelected: string) {
       let nnCommands: Array<DriveCommand> = [];
       nnCommands.push(new NeuralNetworkCommand(nnSelected));
       return nnCommands;
    }

    getCommandsFromMotion(motion: Motion): Array<DriveCommand> {
        let driveCmds: Array<DriveCommand> = [];
        if (!motion) return driveCmds;
        if (motion.forward || motion.backward || motion.right || motion.left) {
            this.hasStopped = false;
            if (motion.forward) {
                driveCmds.push(new CmdForwardBackward(true, motion.FBAngle,
                    this.speedRange));
            }
            else if (motion.backward) {
                driveCmds.push(new CmdForwardBackward(false, motion.FBAngle,
                    this.speedRange));
            }
            if (motion.right) {
                driveCmds.push(new CmdLeftRight(true, motion.LRAngle));
            }
            else if (motion.left) {
                driveCmds.push(new CmdLeftRight(false, motion.LRAngle));
            }
        }
        else {
            if (!this.hasStopped) {
                driveCmds.push(new CmdStop());
                this.hasStopped = true;
            };
        }
        return driveCmds;
    }

}
