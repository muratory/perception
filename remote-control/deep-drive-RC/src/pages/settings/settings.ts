import { Component } from '@angular/core';

// Platform required to know when access is ready to HW feature
import { NavController, Platform } from 'ionic-angular';

import { ConfigService } from '../../services/ConfigService';

@Component({
  selector: 'page-settings',
  templateUrl: 'settings.html'
})
export class SettingsPage {

  private speed: any;
  private rcOrientation: string;

  constructor(public navCtrl: NavController,
              private platform: Platform,
              private config: ConfigService) {
      this.platform.ready().then(() => {
          console.log("SETTINGS: new instance");
          this.initSettings();
      });
  }

  initSettings() {
      this.config.getSpeedRange().subscribe((range) => {
          console.log("SETTINGS: detected change in speed range " + range);
          this.speed = range;
      });
      this.config.getRCOrientation().subscribe((preferred) => {
          console.log("SETTINGS: detected change in rcOrientation " + preferred);
          this.rcOrientation = preferred;
      });
  }

  rcOrientationChange() {
      if (this.rcOrientation) {
          this.config.saveRCOrientation(this.rcOrientation);
      }
  }

  speedChange() {
      if (this.speed) {
          this.config.saveSpeedRange(this.speed);
      }
  }

}
