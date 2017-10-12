import {BluetoothSerial} from 'ionic-native';
import {Observable} from 'rxjs/Observable';

import 'rxjs/add/operator/toPromise';
import { ReplaySubject } from 'rxjs/ReplaySubject';

export class BluetoothService {

    /*
     * Note about message protocol:
     * - messages between RFCOMM client and server must be separated by '\n'
     */

    private END_MSG = '\n';

    private device: any = {};

    private connected: boolean;
    private isConnectedSubject: ReplaySubject<boolean>;

    constructor() {
      console.log("BLUETOOTH: new instance");
      this.isConnectedSubject = new ReplaySubject<boolean>(1);
      this.setDisconnected();
    }

    enable(): Promise<any> {
        return BluetoothSerial.enable();
    }

    /* return a promise with isEnabled: boolean */
    isEnabled(): Promise<any> {
        return BluetoothSerial.isEnabled();
    }

    setConnected(device): void {
        this.connected = true;
        this.isConnectedSubject.next(this.connected);
        this.device = device;
    }

    setDisconnected(): void {
        this.connected = false;
        this.isConnectedSubject.next(this.connected);
        this.device = null;
    }

    isConnected(): Observable<boolean> {
        return this.isConnectedSubject.asObservable();
    }

    connect(device: any): Promise<any> {
        if (!device) return Promise.reject("missing device");
        if (!device.hasOwnProperty('address')) return Promise.reject("missing address " + JSON.stringify(device));
        /* BluetoothSerial.connect returns an Observable<any>
         * to notify connection success and eventual disconnection
         */
        return new Promise(function(resolve, reject) {
            BluetoothSerial.connect(device.address).subscribe((result) => {
                resolve(result);
              }, (err) => {
                  console.log("BLUETOOTH: detected disconnection " + JSON.stringify(err));
                  this.setDisconnected();
                  reject(err);
              });
        }.bind(this)).then(() => {
            console.log("BLUETOOTH: connected to " + JSON.stringify(device));
            this.setConnected(device);
        });
    }

    disconnect(): Promise<any> {
        return BluetoothSerial.disconnect().then(() => {
            console.log("BLUETOOTH: disconnected from " + JSON.stringify(this.device));
            this.setDisconnected();
        })
    }

    /* return a promise */
    list() {
        return BluetoothSerial.list();
    }

    rejectAfterDelay(reason) {
        let delay = 500; /* ms */
        return new Promise(function(resolve, reject) {
            console.log("BLUETOOTH: new attempt to read");
            setTimeout(reject.bind(null, reason), delay);
        });
    }

    testAvailable(bytes) {
        if (bytes > 0) {
            return bytes;
        }
        else {
            throw "nothing received";
        }
    }

    waitUntilAvailable(command) {
        let max = 10;
        let p: any = Promise.reject("fake");
        for (let i = 0; i < max; i++) {
            /* chaining promise to be sure to receive a reply */
            p = p.catch((error) => BluetoothSerial.available())
                  .then((bytes) => this.testAvailable(bytes))
                  .catch((error) =>
                      this.rejectAfterDelay("failed with error " + error));
        }
        return p;
    }

    writeAndWaitReply(command) {
        command = command + this.END_MSG;
        if (!this.connected) {
            return Promise.resolve();
        }
        let p: any = BluetoothSerial.write(command).then((result: any) => {
            if (result == "OK") {
                console.log("BLUETOOTH: wrote " + command);
                return this.waitUntilAvailable(command).then((bytes: any) => {
                    console.log("BLUETOOTH: available " + bytes + " bytes");
                    return BluetoothSerial.readUntil(this.END_MSG).then((buffer) => {
                        console.log("BLUETOOTH: read buffer " + JSON.stringify(buffer));
                        return Promise.resolve(buffer);
                    });
                });
            };
        });
        return p;
    }
}
