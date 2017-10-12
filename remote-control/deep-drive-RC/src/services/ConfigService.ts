import { Injectable } from '@angular/core';
import { Storage } from '@ionic/storage';
import { Observable } from 'rxjs/Observable';
import { ReplaySubject } from 'rxjs/ReplaySubject';

@Injectable()
export class ConfigService {

    private DEFAULT_SPEED_RANGE: any = { lower: 0, upper: 50 };
    private DEFAULT_RC_ORIENTATION: string = 'flatStop';

    public speedRangeSubject: ReplaySubject<any>;
    public rcOrientationSubject: ReplaySubject<string>;

    constructor(private storage: Storage) {
      console.log("CONFIG: new instance");
      this.speedRangeSubject = new ReplaySubject<any>(1);
      this.rcOrientationSubject = new ReplaySubject<string>(1);
      this.reloadSettings();
    }

    reloadSettings() {
      this.reloadSpeed();
      this.reloadRCOrientation();
    }

    save(key: string, data: any): Promise<any> {
      return this.storage.set(key, data);
    }

    loadNoDefault(key: string): Promise<any> {
      return this.storage.get(key);
    }

    load(key: string, defaultValue: any): Promise<any> {
        return this.loadNoDefault(key).then((value:any) => {
            let setting: any = value;
            if (value == null) {
                setting = defaultValue;
            }
            console.log("CONFIG: retrieving " + key + " as " + JSON.stringify(setting));
            return setting;
        }).catch((err) => {
            console.error("CONFIG: failed retrieving " + key +
                          ", setting default " + JSON.stringify(defaultValue) +
                          " " + JSON.stringify(err));
            return defaultValue;
        });
    }

    getRCOrientation(): Observable<string> {
        return this.rcOrientationSubject.asObservable();
    }

    reloadRCOrientation() {
      this.load('rcOrientation', this.DEFAULT_RC_ORIENTATION).then((value) => {
          this.rcOrientationSubject.next(value);
        });
    }

    saveRCOrientation(value: string): void {
        this.save('rcOrientation', value).then(() => {
            console.log("CONFIG: new remote control orientation stored as " + value);
            this.rcOrientationSubject.next(value);
        }, () => {
            console.error("CONFIG: failed to store remote control orientation as " + value);
        });
    }

    getSpeedRange(): Observable<any> {
        return this.speedRangeSubject.asObservable();
    }

    reloadSpeed() {
      this.load('speed', this.DEFAULT_SPEED_RANGE).then((range) => {
          this.speedRangeSubject.next(range);
        });
    }

    saveSpeedRange(range: any): void {
        this.save('speed', range).then(() => {
            console.log("CONFIG: speed range stored as " + JSON.stringify(range));
            this.speedRangeSubject.next(range);
        }).catch((err) => {
            console.error("CONFIG: failed to store speed range as " + JSON.stringify(range) + " (" + err + ")");
        });
    }
}
