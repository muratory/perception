import { Component } from '@angular/core';
import { Platform } from 'ionic-angular';
import { StatusBar, Splashscreen } from 'ionic-native';

import { Storage } from '@ionic/storage';

import { TabsPage } from '../pages/tabs/tabs';
import { BluetoothService } from '../services/BluetoothService';
import { MotionService } from '../services/MotionService';
import { DriveControlService } from '../services/DriveControlService';
import { ConfigService } from '../services/ConfigService';

@Component({
  templateUrl: 'app.html',
  providers: [BluetoothService, MotionService, DriveControlService,
              Storage, ConfigService],
})
export class MyApp {
  rootPage = TabsPage;

  constructor(platform: Platform, bluetoothService: BluetoothService) {
    platform.ready().then(() => {
      // Okay, so the platform is ready and our plugins are available.
      // Here you can do any higher level native things you might need.
      StatusBar.styleDefault();
      Splashscreen.hide();
    });
  }
}
