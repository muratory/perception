import { Component, ChangeDetectorRef } from '@angular/core';

// Platform required to know when access is ready to HW feature
import { NavController, Platform } from 'ionic-angular';
import { ScreenOrientation } from 'ionic-native';
import { TextToSpeech } from '@ionic-native/text-to-speech';
import { Insomnia } from '@ionic-native/insomnia';

import { BluetoothService } from '../../services/BluetoothService';
import { Acceleration, Motion, MotionService } from '../../services/MotionService';
import { DriveControlService, DriveCommand } from '../../services/DriveControlService';
import { ConfigService } from '../../services/ConfigService';

declare var SpeechRecognition: any;

@Component({
  selector: 'page-drive',
  templateUrl: 'drive.html',
  providers: [TextToSpeech, Insomnia],
})
export class DrivePage {

  private motion: Motion;
  private isActive: boolean;

  private DRIVE_CMD_INTERVAL: number = 100; /* ms */
  private cmdIntervalId: any = null;

  private bluetoothIsConnected: boolean = false;

  private neuralNetworks = {'left':false,
                            'straight':false,
                            'right':false,
                            'idle':false};
  private nnColors = {'left':'light',
                      'straight':'light',
                      'right':'light'}

  private speechRecognition: any = null;

  constructor(public navCtrl: NavController, private platform: Platform,
              private bluetoothService: BluetoothService,
              private ref: ChangeDetectorRef,
              private motionService: MotionService,
              private configService: ConfigService,
              private driveService: DriveControlService,
              private tts: TextToSpeech,
              private insomnia: Insomnia) {
    this.motion = new Motion(new Acceleration(0, 0, 0), 0);
    this.isActive = false;
    platform.ready().then(() => {
      console.log("DRIVE: motion can now be used");
      this.motionService.motion.subscribe((newMotion) => {
          this.motion = newMotion;
      })
      this.bluetoothService.isConnected().subscribe((isConnected) => {
          let state = 'connected';
          if (!isConnected) state = 'disconnected';
          console.log("DRIVE: connection state change to " + state);
          this.bluetoothIsConnected = isConnected;
          if (this.isActive && !isConnected) {
              this.stopMotion();
          }
          /* reset neural network selection to idle */
          for (var nn in this.neuralNetworks) {
              this.neuralNetworks[nn] = false;
          }
          this.selectNN('idle');

          /* forcing change detection as it does not work well in observable */
          this.ref.detectChanges();
      })
    })
  }

  /* Return portrait-primary, landscape-primary or landscape-secondary */
  getScreenOrientation(): string {
      /* need to change plugins/screen-orientation.d.ts so that :
       * static orientation: any; (instead of string) to access type
       */
      let currentOrientation: string = ScreenOrientation.orientation.type;
      console.log("MOTION: current orientation is " + currentOrientation);
      if (currentOrientation == "portrait" || currentOrientation == "portrait-primary"
        || currentOrientation == "portrait-secondary" || currentOrientation == "any") {
          /* note: value "any" is not expected */
          return "portrait-primary";
      }
      else if (currentOrientation == "landscape" || currentOrientation == "landscape-primary"){
          return "landscape-primary";
      }
      else if (currentOrientation == "landscape-secondary"){
          return "landscape-secondary";
      }
      else {
          console.log("MOTION: unexpected screen orientation " + currentOrientation);
          return "any";
      }
  }

  lockOrientation(doLock: boolean) {
    console.log("MOTION: " + (doLock?"locking":"unlocking"));
    if (doLock) {
      /* normally returns a promise */
      let newOrientation = this.getScreenOrientation();
      ScreenOrientation.lockOrientation(newOrientation).then(function (obj) {
          console.log("MOTION: screen " + obj);
      }).catch(function(err) {
          console.log("MOTION: error changing orientation to " + newOrientation + " [" + JSON.stringify(err) + "]");
      });
    }
    else {
      ScreenOrientation.unlockOrientation();
    }
  }

  sendBluetoothCommands(commands: Array<DriveCommand>) {
      let p = Promise.resolve();
      for (let cmd of commands) {
          p.then((success) => {
            console.log("DRIVE: new command " + cmd.getCommand());
            return this.bluetoothService.writeAndWaitReply(cmd.getCommand());
          });
      };
      p.catch((err) => {
          console.log("DRIVE: failed to send all commands due to " + err);
      });
  }

  sendDriveCommands() {
      /* Based on last motion recorded, send remote
       * command to car over Bluetooth
       */
      let commands: Array<DriveCommand> = this.driveService.getCommandsFromMotion(this.motion);
      this.sendBluetoothCommands(commands);
  }

  sendResetCommands() {
      let commands: Array<DriveCommand> = this.driveService.getInitCommands();
      this.sendBluetoothCommands(commands);
  }

  scheduleNextSpeechToText(whenMs: number) {
    /* Listen for next instruction */
    setTimeout(() => {
            this.startSpeechToText()
        ;}, whenMs);
  }

  startSpeechToText() {
      if (!this.isActive) {
          return;
      }
      this.speechRecognition = new SpeechRecognition();
      this.speechRecognition.lang = 'en-US';
      this.speechRecognition.continuous = false;
      this.speechRecognition.onnomatch = ((event) =>  {
          console.log("SPEECH: no match found");
          this.scheduleNextSpeechToText(100);
      });
      this.speechRecognition.onerror = ((err) => {
        console.log("SPEECH: error " + JSON.stringify(err));
        this.scheduleNextSpeechToText(100);
      });
      this.speechRecognition.onstart = (() => {
          console.log("SPEECH: start listening");
      });

      this.speechRecognition.onresult = ((event) => {
          if (event.results.length > 0) {
              /* pick the one with highest confidence
               * output format :
               * [[{"transcript":"hey what's up buddy","final":true,"confidence":0.930260181427002}],
               *  [{"transcript":"hey whats up buddy","final":true,"confidence":0.9313551783561707}],
               *  [{"transcript":"hey what's up Barry","final":true,"confidence":0.9324639439582825}],
               *  [{"transcript":"hey what's up Birdy","final":true,"confidence":0.9324639439582825}],
               *  [{"transcript":"hey whatsup very","final":true,"confidence":0.9273926615715027}]]"
               */
              let bestTranscript: any = null;
              var bestNN: string = 'idle';
              let perfectMatch: boolean = false;
              for (let text of event.results) {
                  console.log("SPEECH: analyzing " + JSON.stringify(text[0]));
                  for (var nn in this.neuralNetworks) {
                      if (!perfectMatch && this.neuralNetworks.hasOwnProperty(nn)) {
                          if (text[0].transcript.includes(nn)) {
                              bestTranscript = text[0];
                              bestNN = nn;
                          }
                          if (text[0].transcript == nn) {
                              /* stop looking if perfect match */
                              perfectMatch = true;
                          }
                      }
                  }
              }
              if (bestTranscript) {
                  console.log("SPEECH: found neural network '" + bestNN + "' in transcript '" + bestTranscript.transcript + "'");
                  this.selectNN(bestNN);

                  /* forcing change detection as it does not work well in observable */
                  this.ref.detectChanges();

                  /* Tell user which neural network is selected */
                  this.tts.speak(this.getCurrentNN()).then(() => {
                        console.log('SPEECH: just said ' + this.getCurrentNN())
                      }).catch((reason: any) => {
                        console.log("SPEECH: failed to say " + this.getCurrentNN() + " error " + reason)
                      });
              }
          }
          /* schedule next attempt */
          this.scheduleNextSpeechToText(50);
      });
      this.speechRecognition.start();
  }

  startMotion() {
    console.log("MOTION: starting");
    this.lockOrientation(true);
    this.insomnia.keepAwake().then(
        () => console.log("DRIVE: keep awake"),
        () => console.log("DRIVE: failed to keep awake")
      );
    this.motionService.start(this.getScreenOrientation());
    this.sendResetCommands();
    this.cmdIntervalId = setInterval(() => {this.sendDriveCommands(); },
                                     this.DRIVE_CMD_INTERVAL);
    this.isActive = true;
    this.startSpeechToText();
  }

  stopMotion() {
    console.log("MOTION: stopping");
    if (this.cmdIntervalId) {
        clearInterval(this.cmdIntervalId);
        this.cmdIntervalId = null;
    }
    this.sendResetCommands();
    this.lockOrientation(false);
    this.insomnia.allowSleepAgain().then(
        () => console.log("DRIVE: allow to sleep again"),
        () => console.log("DRIVE: failed to allow to sleep again")
      );
    this.motionService.stop();
    this.isActive = false;
  }

  isDirection(direction: string) {
      let result: boolean;
      let isRight: boolean = (this.motion.right && !this.motion.left);
      let isLeft: boolean = (this.motion.left && !this.motion.right);
      let isForward: boolean = (this.motion.forward && !this.motion.backward);
      let isBackward: boolean = (this.motion.backward && !this.motion.forward);
      switch(direction) {
          case 'forward-left':
            result = isForward && isLeft;
            break;
          case 'forward':
            result = isForward && !isLeft && !isRight;
            break;
          case 'forward-right':
            result = isForward && isRight;
            break;
          case 'backward-left':
            result = isBackward && isLeft;
            break;
          case 'backward':
            result = isBackward && !isRight && !isLeft;
            break;
          case 'backward-right':
            result = isBackward && isRight;
            break;
          case 'left':
            result = isLeft && !isForward && !isBackward;
            break;
          case 'right':
            result = isRight && !isForward && !isBackward;
            break;
          case 'stop':
            result = !isRight && !isLeft && !isForward && !isBackward;
            break;
          default:
            result = false;
            break;
      }
      return result;
  }

  setNNColors() {
      for (var nn in this.neuralNetworks) {
          if (this.neuralNetworks.hasOwnProperty(nn)) {
              this.nnColors[nn] = this.neuralNetworks[nn]?'primary':'light';
          }
      }
  }

  getCurrentNN(): string {
      for (var nn in this.neuralNetworks) {
          if (this.neuralNetworks[nn] == true) {
              return nn;
          }
      }
      console.log("DRIVE: warning, no neural network selected");
  }

  selectNN(nnSelected: string) {
      let noneSelected = true;
      for (var nn in this.neuralNetworks) {
          if (nn == nnSelected) {
              this.neuralNetworks[nn] = !this.neuralNetworks[nn];
              if (this.neuralNetworks[nn]) {
                  noneSelected = false;
              }
          }
          else {
              this.neuralNetworks[nn] = false;
          }
      }
      if (noneSelected) {
          this.neuralNetworks['idle'] = true;
      }
      this.setNNColors();
      let currentNN = this.getCurrentNN();
      console.log("DRIVE: current neural network is " + currentNN);
      /* Send current neural network to car */
      let nnCommands: Array<DriveCommand> = this.driveService.getNeuralNetworkCommands(currentNN);
      this.sendBluetoothCommands(nnCommands);
  }

}
