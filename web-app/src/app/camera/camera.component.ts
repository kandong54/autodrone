import { Component, OnInit, ViewChild, ElementRef, AfterViewInit } from '@angular/core';
import { Router } from '@angular/router';
import { ClientService } from '../grpc/client.service';
import { CameraReply } from '../../protos/drone_pb';
import { ClientReadableStream } from 'grpc-web';

@Component({
  selector: 'app-camera',
  templateUrl: './camera.component.html',
  styleUrls: ['./camera.component.sass']
})
export class CameraComponent implements OnInit, AfterViewInit {

  cameraWidth: number;
  cameraHeight: number;
  imageWidth: number;
  imageHeight: number;
  imageStream: ClientReadableStream<CameraReply> | null = null;
  interval: number;
  lastTime: number;

  @ViewChild('myCanvas')
  private myCanvas: ElementRef = {} as ElementRef;

  constructor(private clientService: ClientService,
    private router: Router) {
    // TODO: get width and height
    this.cameraWidth = 960;
    this.cameraHeight = 720;
    this.imageWidth = 640;
    this.imageHeight = 640;
    this.interval = 230;
    this.lastTime = Date.now();
  }

  ngOnInit(): void {
    this.clientService.connect()
      .then(result => {
        console.log('Connect result:', result);
      })
      .catch((error) => {
        alert('Connection error: ' + error);
        this.router.navigate(['/login']);
      });

  }

  ngAfterViewInit(): void {
    this.myCanvas.nativeElement.width = this.cameraWidth;
    this.myCanvas.nativeElement.height = this.cameraHeight;
    // this.myCanvas.nativeElement.style.aspectRatio = this.cameraWidth / this.cameraHeight;
    var ctx = this.myCanvas.nativeElement.getContext('2d');
    ctx.strokeStyle = 'red';
    ctx.font = '48px Roboto';
    this.clientService.connect()
      .then((result) => {
        if (result === true) {
          this.imageStream = this.clientService.getCamera();
          if (this.imageStream === null) {
            alert('Failed to get camera data!');
            return;
          }
          this.imageStream.on('error', (err) => alert('Connection error: ' + err));
          this.imageStream.on('end', () => alert('Connection ends.'));
          let image = new Image();
          this.imageStream.on('data', (response) => {
            // fps
            let thisTime = Date.now();
            let delta = thisTime - this.lastTime;
            this.lastTime = thisTime;
            let k = 0.9;
            this.interval = this.interval * k + delta * (1 - k);
            let imageRGB = response.getImage_asU8();
            let boxes = response.getBoxList();
            // TODO: offscreen render
            // jpg
            let blob = new Blob([imageRGB], { 'type': 'image/jpeg' });
            URL.revokeObjectURL(image.src);
            image.src = URL.createObjectURL(blob);
            image.onload = () => {
              ctx.drawImage(image, 0, 0, this.imageWidth, this.imageHeight, 0, 0, this.cameraWidth, this.cameraHeight);
              ctx.fillText('FPS: ' + (1000 / this.interval).toFixed(2), 10, 50);
              for (const box of boxes) {
                let x = box.getLeft();
                let y = box.getTop();
                let width = box.getWidth();
                let height = box.getHeight();
                ctx.strokeRect(x, y, width, height);
                let confidence = box.getConfidence();
                ctx.fillText(Math.round(confidence * 100) + '%', x, y + height);
              }
            }
          });
        }
      });
  }
}
