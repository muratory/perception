import { Component, ChangeDetectorRef } from '@angular/core';
import { NavController, Platform, AlertController, ToastController, LoadingController } from 'ionic-angular';

import { BluetoothService } from '../../services/BluetoothService';
import { ConfigService } from '../../services/ConfigService';

@Component({
  selector: 'page-home',
  templateUrl: 'home.html',
})
export class HomePage {

  private foundDevices: any[];
  private isDevReady: boolean = false;
  private bluetoothIsConnected: boolean = false;
  private bluetoothAllowScan: boolean = false;
  private bluetoothScanStarted: boolean = false;
  private loadingSpinner: any = null;

  constructor(public navCtrl: NavController,
              public plt: Platform,
              private ref: ChangeDetectorRef,
              public alertCtrl: AlertController,
              private bluetoothService: BluetoothService,
              private ConfigService: ConfigService,
              private toastCtrl: ToastController,
              private loadingCtrl: LoadingController) {
      this.plt.ready().then((readySource) => {
          console.log("APP: ready from source " + readySource);
          this.isDevReady = true;
          this.bluetoothService.isConnected().subscribe((isConnected) => {
              let state = 'connected';
              if (!isConnected) state = 'disconnected';
              console.log("BLUETOOTH: connection state change to " + state);
              this.bluetoothIsConnected = isConnected;
              /* forcing change detection as it does not work well in observable */
              this.ref.detectChanges();
          })
          this.checkBluetoothAvailability();
      });
  }

  checkBluetoothAvailability() {
      this.bluetoothService.isEnabled().then((isEnabled: any) => {
          console.log("BLUETOOTH:" + (isEnabled?" is ":" is not ") +
                      "enabled");
          this.onBluetoothEnabled();
      }).catch((error) => {
          console.log("BLUETOOTH: failed to check availability: " + error);
          this.onBluetoothDisabled();
          this.askForBluetoothActivation();
      });
  }

  showLoadingSpinner(msg) {
      this.loadingSpinner = this.loadingCtrl.create({
        spinner: 'circles',
        content: msg
      });
      this.loadingSpinner.present();
  }

  hideLoadingSpinner() {
      if (this.loadingSpinner) {
          this.loadingSpinner.dismiss();
          this.loadingSpinner = null;
      }
  }

  showToast(isFailure: boolean, msg: string, duration: number) {
      let cssClass = 'toast-success';
      if (isFailure == true) {
          cssClass = 'toast-failure';
      }
      let toast = this.toastCtrl.create({
          message: msg,
          duration: duration,
          position: 'top',
          cssClass: cssClass,
          dismissOnPageChange: true,
      });
      toast.present();
  }

  showToastSuccess(msg: string, duration: number) {
      this.showToast(false, msg, duration);
  }

  showToastFailure(msg: string, duration: number) {
      this.showToast(true, msg, duration);
  }

  askForBluetoothActivation() {
      this.bluetoothService.enable().then(() => {
          console.log("BLUETOOTH: enabled with success by user");
          this.showToastSuccess("Bluetooth successfully activated", 3000);
          this.onBluetoothEnabled();
      }, () => {
          console.log("BLUETOOTH: user failed to enable");
          this.showToastFailure("Please enable Bluetooth to connect to car", 3000);
          this.onBluetoothDisabled();
      });
  }

  onBluetoothEnabled() {
      this.bluetoothAllowScan = true;
  }

  onBluetoothDisabled() {
      this.bluetoothAllowScan = false;
  }

  bluetoothFailure(operation, error) {
      let errMsg = "Failed to " + operation +
        " (reason: " + JSON.stringify(error) + ""
      this.alertFailure('Bluetooth', errMsg);
      console.log('BLUETOOTH: ' + errMsg);
  }

  bluetoothScanDevices() {
      console.log("SCAN: start scanning for devices");
      this.foundDevices = []; // empty list before scanning
      this.bluetoothService.list().then((result: any) => {
          let self = this;
          this.bluetoothScanStarted = true;
          result.forEach(function(device, index, array) {
              /* Elements in device:
               * - class
               * - id
               * - address
               * - name
               */
              console.log("BLUETOOTH: adding device " + JSON.stringify(device));
              self.foundDevices.push(device);
          });
      }, (error) => {
          this.bluetoothFailure("BLUETOOTH", error);
      });
  }

  alertFailure(type, reason) {
      let alert = this.alertCtrl.create({
          title: type,
          subTitle: reason,
          buttons: ['OK']
      });
      alert.present();
  }

  alertGeneric(title, subtitle) {
      let alert = this.alertCtrl.create({
          title: title,
          subTitle: subtitle,
          buttons: ['OK']
      });
      alert.present();
  }

  bluetoothConnect(device: any) {
      this.showLoadingSpinner("Connection ongoing, please wait...");
      console.log("BLUETOOTH: connecting to " + JSON.stringify(device));
      this.bluetoothService.connect(device).then(() => {
          this.hideLoadingSpinner();
          this.showToastSuccess("Now connected to " + device.name, 3000);
      }).catch((err) => {
          this.hideLoadingSpinner();
          this.showToastFailure("Failed to connect to " + device.name, 3000);
          console.log("BLUETOOTH: failed to connect to " + JSON.stringify(device));
      });
  }

  bluetoothDisconnect() {
      console.log("BLUETOOTH: disconnecting from car");
      this.bluetoothService.disconnect();
  }

  bluetoothShowDetails() {
      this.bluetoothService.writeAndWaitReply("details").then((reply) => {
              this.alertGeneric('Car details', reply);
          }).catch((error) => {
              console.log("BLUETOOTH: " + error);
              this.showToastFailure("Failed to get car details", 3000);
          });
  }

  bluetoothDevSelected(item: any) {
      this.bluetoothConnect(item);
  }
}
