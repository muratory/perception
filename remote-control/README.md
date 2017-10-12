# Ionic 2 Deep Drive Remote Control application

## Supported OS

### Android
The application has been successfully tested on :
* Android 5.1.1 / Moto X 2013

### iOS
Not (yet) supported.

## First time build setup

* Note : the Git project does not contain build artefacts or node modules.
* To build the project, the following steps should be followed

```
# from freshly pulled deep-drive-RC repository
npm Install
# apply patches on node_modules (required to build)
for f in patches/*; do
	patch -p1 < $f
done
# Copy images to www/ folder
cp -r resources/img www/
# Install cordova bluetooth plugin (not sure why it is not in config.xml)
ionic plugin add cordova-plugin-bluetooth-serial
# Install @ionic/storage
sudo npm install @ionic/storage --save
# Install motion plugin
ionic plugin add cordova-plugin-device-motion
# Install screen-orientation plugin (specific version because 2.0-dev fails )
ionic plugin add cordova-plugin-screen-orientation@1.4.3
# Install speech recognition plugin (different from ionic-native one)
ionic plugin add https://github.com/macdonst/SpeechRecognitionPlugin
# Install Text to Speech plugin
ionic plugin add cordova-plugin-tts
sudo npm install --save @ionic-native/text-to-speech
sudo npm install @ionic-native/core --save
# Insomnia plugin to prevent from sleeping while driving
ionic plugin add https://github.com/EddyVerbruggen/Insomnia-Phonegap-Plugin.git
sudo npm install --save @ionic-native/insomnia
# Build Android flavor
ionic platform add android@5.1.1
ionic build android
```

### Notes about permissions

I have not digged into it, but I met permission issues while dealing with npm.
The -dirty- workaround found so far is to call `npm` with sudoer's rights (`sudo npm`).

```
sudo npm install -g cordova
sudo npm install -g ionic
sudo npm install
sudo chmod -R 755 *
sudo chown -R <myuser> .
```

## Build & Test

* Build only with `ionic build android`
* Build and launch on device with `ionic run android --device && adb logcat -v time | grep chromium`

## Debug vs Production builds

* Default builds are debuggable. Main drawback is that the app startup phase is long.
* Production builds made with option `--prod` do not have this drawback.

# Bluetooth communication between Smartphone & car

## Car side / Raspbian

### Prerequisites :
`sudo apt-get install libbluetooth-dev python-bluez`

## Linux Development host (Ubuntu)

Check Bluetooth driver versions

```
dmesg | grep -i bluetooth
[   23.525316] Bluetooth: Core ver 2.17
[   23.525333] Bluetooth: HCI device and connection manager initialized
[   23.525339] Bluetooth: HCI socket layer initialized
[   23.525341] Bluetooth: L2CAP socket layer initialized
[   23.525344] Bluetooth: SCO socket layer initialized
[   23.600853] Bluetooth: BNEP (Ethernet Emulation) ver 1.3
[   23.600856] Bluetooth: BNEP filters: protocol multicast
[   23.600863] Bluetooth: BNEP socket layer initialized
[   23.766192] Bluetooth: RFCOMM TTY layer initialized
[   23.766202] Bluetooth: RFCOMM socket layer initialized
[   23.766207] Bluetooth: RFCOMM ver 1.11
```

Get local configuration with `hciconfig -a`
```
hci0:	Type: BR/EDR  Bus: USB
	BD Address: 5C:F3:70:78:49:55  ACL MTU: 1021:8  SCO MTU: 64:1
	UP RUNNING PSCAN ISCAN
	RX bytes:4718 acl:59 sco:0 events:131 errors:0
	TX bytes:3690 acl:62 sco:0 commands:61 errors:0
	Features: 0xbf 0xfe 0xcf 0xfe 0xdb 0xff 0x7b 0x87
	Packet type: DM1 DM3 DM5 DH1 DH3 DH5 HV1 HV2 HV3
	Link policy: RSWITCH SNIFF
	Link mode: SLAVE ACCEPT
	Name: 'tldlab199.tl.intel.com-0'
	Class: 0x6c0100
	Service Classes: Rendering, Capturing, Audio, Telephony
	Device Class: Computer, Uncategorized
	HCI Version: 4.0 (0x6)  Revision: 0x1000
	LMP Version: 4.0 (0x6)  Subversion: 0x220e
	Manufacturer: Broadcom Corporation (15)
```

Get local device MAC address with `hcitool dev`
`Devices:
	hci0	5C:F3:70:78:49:55`

Prerequisite : `sudo apt-get install bluez-hcidump`
Dump with `hcidump --raw` during `hciconfig scan`:
```
Scanning ...
	02:34:56:78:9A:BC	T100TA-0
	F8:E0:79:99:FD:16	XT1052
```

## Bluetooth Python server

Prerequisites :
`sudo apt-get install python-bluez`

Known issues
* `all_proxy=http://proxy-chain.intel.com:911 ./bin/pip install pybluez` fails because of error :

```
bluez/btmodule.h:5:33: fatal error: bluetooth/bluetooth.h: No such file or directory
   #include <bluetooth/bluetooth.h>
```

Solution : `sudo apt-get install libbluetooth-dev`

### Automatic start of Deep Drive Bluetooth server at car boot

A systemd service is available to automate server start at car boot.

* Copy `deepdrive.service` to `/lib/systemd/system/deepdrive.service`
* Enable `deepdrive.service` with `systemctl enable deepdrive.service`

`deepdrive.service` is configured:
* to point to `bluetooth_server.py` in `/home/pi/deep_drive/raspberry/server/`
* to restart automatically when it crashes or gracefully closes on user disconnection

Further usages:
* `systemctl is-enabled deepdrive.service`
* `systemctl status deepdrive.service`
